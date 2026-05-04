# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.types import Group
from forge.types import TrainBatch


def _make_causal_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build a causal attention mask that hides pad keys from real tokens."""
    batch_size, seq_len = attention_mask.shape
    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device)
    )
    key_mask = attention_mask[:, None, None, :]
    causal_padding_mask = causal[None, None, :, :] & key_mask

    # Pad-token query rows are ignored by the RL loss, but SDPA should still see
    # at least one valid key to avoid undefined all-masked rows.
    query_is_pad = ~attention_mask
    self_mask = torch.eye(seq_len, dtype=torch.bool, device=attention_mask.device)
    pad_query_self_mask = query_is_pad[:, None, :, None] & self_mask[None, None, :, :]
    return causal_padding_mask | pad_query_self_mask


def _make_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Assign RoPE positions as if pad tokens were stripped before scoring."""
    positions = torch.cumsum(attention_mask.to(torch.long), dim=1) - 1
    return positions.clamp_min_(0)


def make_padding_metadata(attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build TorchTitan model kwargs for scoring left-padded RL sequences."""
    return {
        "attention_masks": _make_causal_padding_mask(attention_mask),
        "positions": _make_positions(attention_mask),
    }


def collate(
    batches: list[Group], *, include_padding_metadata: bool = False
) -> list[TrainBatch]:
    """
    Collates a list of batches into TrainBatch objects.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    result = []
    for batch in batches:
        request = [e.request_tensor for e in batch]
        request = torch.stack(request)  # [b x s]

        response = [e.response_tensor for e in batch]
        response = torch.stack(response)  # [b x s]

        input_ids = torch.cat([request, response], dim=1)
        seq_len = input_ids.shape[1]
        model_inputs = {"tokens": input_ids}

        if include_padding_metadata:
            request_attention_mask = torch.stack(
                [e.request_attention_mask for e in batch]
            )
            response_attention_mask = torch.stack(
                [e.response_attention_mask for e in batch]
            )
            attention_mask = torch.cat(
                [request_attention_mask, response_attention_mask], dim=1
            )
            model_inputs.update(make_padding_metadata(attention_mask))

        # ref_logprobs is optional - only stack if all episodes have it
        ref_logprobs = None
        if all(e.ref_logprobs is not None for e in batch):
            ref_logprobs = torch.stack([e.ref_logprobs for e in batch])

        advantages = [e.advantage for e in batch]
        advantages = torch.tensor(advantages).unsqueeze(-1)  # [b x 1]
        advantages = advantages.expand(-1, seq_len)  # [b x s]

        generator_logprobs = torch.stack([e.generator_logprobs for e in batch])
        loss_mask = torch.stack([e.loss_mask for e in batch])

        loss_inputs = {
            "generator_logprobs": generator_logprobs,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
        if ref_logprobs is not None:
            loss_inputs["ref_logprobs"] = ref_logprobs

        result.append(
            TrainBatch(
                model_inputs=model_inputs,
                loss_inputs=loss_inputs,
                meta={
                    "group_ids": [e.group_id for e in batch],
                    "rollout_group_sizes": [e.rollout_group_size for e in batch],
                    "slate_ids": [e.slate_id for e in batch],
                    "slate_sizes": [e.slate_size for e in batch],
                    "slate_ranks": [e.slate_rank for e in batch],
                    "slate_reward_ranks": [e.slate_reward_rank for e in batch],
                    "target_actions": [e.target_action for e in batch],
                    "response_actions": [e.response_action for e in batch],
                    "advantage_signs": [e.advantage_sign for e in batch],
                    "proposal_origins": [e.proposal_origin for e in batch],
                },
            )
        )
    return result


def collate_with_padding_metadata(batches: list[Group]) -> list[TrainBatch]:
    return collate(batches, include_padding_metadata=True)
