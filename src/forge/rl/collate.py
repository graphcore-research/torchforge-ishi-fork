# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.types import Group
from forge.types import TrainBatch
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torchtitan.experiments.forge.train_spec import get_train_spec

PackedAttentionMasks = BlockMask | dict[str, BlockMask]


def _response_logprobs(episode) -> torch.Tensor:
    if episode.completion.logprobs is None:
        raise ValueError("Completion.logprobs is required for packed RL collation")
    return episode.completion.logprobs.detach().cpu().to(torch.float32)


def _pad_1d_tensors(
    tensors: list[torch.Tensor],
    *,
    pad_value: float | int = 0,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    max_len = max((t.shape[0] for t in tensors), default=0)
    if dtype is None:
        dtype = tensors[0].dtype
    result = torch.full((len(tensors), max_len), pad_value, dtype=dtype)
    for i, tensor in enumerate(tensors):
        result[i, : tensor.shape[0]] = tensor.detach().cpu().to(dtype)
    return result


def pack_episode_tokens(
    batch: Group,
) -> tuple[torch.Tensor, list[int], list[int], list[int]]:
    """Pack episodes into a single TorchTitan token stream."""
    packed_tokens: list[torch.Tensor] = []
    prompt_lens: list[int] = []
    response_lens: list[int] = []

    for episode in batch:
        prompt_ids = episode.completion.prompt_ids.to(torch.long)
        response_ids = episode.completion.token_ids.to(torch.long)
        packed_tokens.extend([prompt_ids, response_ids])
        prompt_lens.append(prompt_ids.shape[0])
        response_lens.append(response_ids.shape[0])

    if packed_tokens:
        tokens = torch.cat(packed_tokens).unsqueeze(0)
    else:
        tokens = torch.empty((1, 0), dtype=torch.long)
    seq_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
    return tokens, prompt_lens, response_lens, seq_lens


def _create_packed_attention_mask(
    seq_lens: list[int],
    device: torch.device,
    *,
    sliding_window_size: int | None = None,
) -> BlockMask:
    total_len = sum(seq_lens)
    document_ids = torch.empty((1, total_len), dtype=torch.int32, device=device)
    seq_start = 0
    for document_id, seq_len in enumerate(seq_lens):
        document_ids[0, seq_start : seq_start + seq_len] = document_id
        seq_start += seq_len

    def mask_mod(b, h, q_idx, kv_idx):
        same_episode = (q_idx >= kv_idx) & (
            document_ids[b, q_idx] == document_ids[b, kv_idx]
        )
        if sliding_window_size is None:
            return same_episode
        return same_episode & (q_idx - kv_idx < sliding_window_size)

    return create_block_mask(mask_mod, 1, None, total_len, total_len, device=device)


def create_packed_attention_masks(
    model_config,
    seq_lens: list[int],
    device: torch.device,
) -> PackedAttentionMasks:
    if model_config.name != "gpt_oss":
        return _create_packed_attention_mask(seq_lens, device)

    model_args = get_train_spec(model_config.name).model_args[model_config.flavor]
    return {
        "basic_mask": _create_packed_attention_mask(seq_lens, device),
        "sliding_window_mask": _create_packed_attention_mask(
            seq_lens,
            device,
            sliding_window_size=model_args.sliding_window_size,
        ),
    }


def create_positions_from_seq_lens(
    seq_lens: list[int], device: torch.device
) -> torch.Tensor:
    positions = [torch.arange(seq_len, device=device) for seq_len in seq_lens]
    if not positions:
        return torch.empty((1, 0), dtype=torch.long, device=device)
    return torch.cat(positions).unsqueeze(0)


def extract_response_slices(
    values: torch.Tensor,
    seq_lens: list[int],
    prompt_lens: list[int],
    response_lens: list[int],
) -> list[torch.Tensor]:
    """Extract response prediction positions from packed next-token values."""
    seq_start = 0
    result = []
    for seq_len, prompt_len, response_len in zip(
        seq_lens, prompt_lens, response_lens, strict=True
    ):
        start = seq_start + prompt_len - 1
        end = start + response_len
        result.append(values[0, start:end])
        seq_start += seq_len
    return result


def pad_response_slices(
    values: list[torch.Tensor],
    *,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((value.shape[0] for value in values), default=0)
    if not values:
        return torch.empty((0, 0)), torch.empty((0, 0), dtype=torch.float32)

    sample = values[0]
    shape = (len(values), max_len, *sample.shape[1:])
    padded = torch.full(
        shape,
        pad_value,
        dtype=sample.dtype,
        device=sample.device,
    )
    mask = torch.zeros((len(values), max_len), dtype=torch.float32, device=sample.device)
    for i, value in enumerate(values):
        padded[i, : value.shape[0], ...] = value
        mask[i, : value.shape[0]] = 1.0
    return padded, mask


def materialize_dtensor(value: torch.Tensor) -> torch.Tensor:
    if isinstance(value, DTensor):
        return value.full_tensor()
    return value


def _token_ids(tokens: torch.Tensor) -> list[int]:
    return [int(token) for token in tokens.detach().cpu().tolist()]


def _pack_response_values(
    batch: Group,
    response_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    target_ids = _pad_1d_tensors(
        [e.completion.token_ids.to(torch.long) for e in batch],
        pad_value=0,
        dtype=torch.long,
    )
    generator_logprobs = _pad_1d_tensors(
        [_response_logprobs(e) for e in batch],
        pad_value=0.0,
        dtype=torch.float32,
    )
    loss_mask = _pad_1d_tensors(
        [torch.ones(response_len) for response_len in response_lens],
        pad_value=0.0,
        dtype=torch.float32,
    )
    advantages = torch.tensor(
        [float(e.advantage) for e in batch],
        dtype=torch.float32,
    ).unsqueeze(-1)
    advantages = advantages.expand_as(loss_mask)
    ref_logprobs = None
    if batch and all(e.ref_logprobs is not None for e in batch):
        ref_logprobs = _pad_1d_tensors(
            [e.ref_logprobs for e in batch],
            pad_value=0.0,
            dtype=torch.float32,
        )
    return target_ids, generator_logprobs, loss_mask, advantages, ref_logprobs


def configure_packed_attention(model_config) -> None:
    """Force TorchTitan model args onto flex attention for packed RL scoring."""
    train_spec = get_train_spec(model_config.name)
    model_args = train_spec.model_args[model_config.flavor]
    model_args.attn_type = "flex"


def collate(batches: list[Group]) -> list[TrainBatch]:
    """
    Collates a list of batches into TrainBatch objects.
    Each batch is a list of episodes, and each episode is a dict of tensors.
    """
    result = []
    for batch in batches:
        input_ids, prompt_lens, response_lens, seq_lens = pack_episode_tokens(batch)
        target_ids, generator_logprobs, loss_mask, advantages, ref_logprobs = (
            _pack_response_values(batch, response_lens)
        )

        loss_inputs = {
            "target_ids": target_ids,
            "generator_logprobs": generator_logprobs,
            "loss_mask": loss_mask,
            "advantages": advantages,
        }
        if ref_logprobs is not None:
            loss_inputs["ref_logprobs"] = ref_logprobs

        result.append(
            TrainBatch(
                model_inputs={"tokens": input_ids},
                loss_inputs=loss_inputs,
                meta={
                    "prompt_lens": prompt_lens,
                    "response_lens": response_lens,
                    "seq_lens": seq_lens,
                    "episode_ids": [episode.episode_id for episode in batch],
                    "generator_tokens": [
                        _token_ids(
                            torch.cat(
                                [
                                    episode.completion.prompt_ids.to(torch.long),
                                    episode.completion.token_ids.to(torch.long),
                                ]
                            )
                        )
                        for episode in batch
                    ],
                },
            )
        )
    return result
