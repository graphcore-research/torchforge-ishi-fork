# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.rl.types import Group
from forge.types import TrainBatch
from torch.distributed.tensor import DTensor
from torchtitan.experiments.forge.train_spec import get_train_spec
from torchtitan.models.attention import VarlenMetadata


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
    """Pack episodes into a single TorchTitan varlen token stream."""
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


def create_varlen_metadata(
    seq_lens: list[int], device: torch.device
) -> VarlenMetadata:
    cu_seq = torch.zeros(len(seq_lens) + 1, dtype=torch.int32, device=device)
    if seq_lens:
        cu_seq[1:] = torch.tensor(seq_lens, dtype=torch.int32, device=device).cumsum(0)
    max_len = max(seq_lens, default=0)
    return VarlenMetadata(
        cu_seq_q=cu_seq,
        cu_seq_k=cu_seq,
        max_q=max_len,
        max_k=max_len,
    )


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


def configure_varlen_attention(model_config) -> None:
    """Force TorchTitan model args onto varlen attention for packed RL scoring."""
    train_spec = get_train_spec(model_config.name)
    model_args = train_spec.model_args[model_config.flavor]
    attn_type = getattr(model_args, "attn_type", None)
    if attn_type == "flex":
        raise ValueError(
            f"Packed RL requires VarlenMetadata attention, but "
            f"{model_config.name}/{model_config.flavor} uses flex attention."
        )
    if attn_type is not None:
        model_args.attn_type = "varlen"


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
                },
            )
        )
    return result
