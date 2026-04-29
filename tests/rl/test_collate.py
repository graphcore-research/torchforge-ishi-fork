# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import sys
import types
from pathlib import Path

import torch
from forge.data_models.completion import Completion
from forge.data_models.prompt import to_prompt


def _load_rl_module(module_name: str, relative_path: str) -> types.ModuleType:
    path = Path(__file__).parents[2] / "src" / "forge" / "rl" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Avoid importing forge.rl.__init__, which pulls in Monarch/Kubernetes. This
# test only needs the pure tensor collation code.
sys.modules.setdefault("forge.rl", types.ModuleType("forge.rl"))
types_module = _load_rl_module("forge.rl.types", "types.py")
collate_module = _load_rl_module("forge.rl.collate", "collate.py")
Episode = types_module.Episode
collate = collate_module.collate


def _aligned_response_tensors(
    response_logprobs: torch.Tensor,
    *,
    request_len: int,
    response_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror post_training.app.main's response-token alignment."""
    seq_len = request_len + response_len
    actual_response_len = response_logprobs.shape[0]
    generator_logprobs = torch.zeros(seq_len, dtype=response_logprobs.dtype)
    generator_logprobs[request_len : request_len + actual_response_len] = (
        response_logprobs
    )
    generator_logprobs = torch.roll(generator_logprobs, shifts=-1, dims=0)
    generator_logprobs[-1] = 0.0

    response_mask = torch.zeros(seq_len, dtype=torch.float32)
    response_mask[request_len : request_len + actual_response_len] = 1.0
    loss_mask = torch.roll(response_mask, shifts=-1, dims=0)
    loss_mask[-1] = 0.0
    return generator_logprobs, loss_mask


def _episode() -> Episode:
    prompt_ids = torch.tensor([11, 12, 13], dtype=torch.long)
    token_ids = torch.tensor([21, 22], dtype=torch.long)
    seq_len = 9
    return Episode(
        episode_id="episode-0",
        pad_id=0,
        request_len=5,
        response_len=4,
        completion=Completion(
            prompt=to_prompt("decode this"),
            text="0",
            prompt_ids=prompt_ids,
            token_ids=token_ids,
            logprobs=torch.zeros_like(token_ids, dtype=torch.float),
            generator_version=0,
        ),
        generator_logprobs=torch.zeros(seq_len),
        loss_mask=torch.zeros(seq_len),
        advantage=1.0,
    )


def test_collate_default_preserves_legacy_model_inputs() -> None:
    batch = collate([[_episode()]])[0]

    assert set(batch.model_inputs) == {"tokens"}
    assert batch.model_inputs["tokens"].tolist() == [[0, 0, 11, 12, 13, 21, 22, 0, 0]]


def test_collate_can_emit_padding_mask_and_rope_positions() -> None:
    batch = collate([[_episode()]], include_padding_metadata=True)[0]

    assert set(batch.model_inputs) == {"tokens", "attention_masks", "positions"}
    assert batch.model_inputs["positions"].tolist() == [[0, 0, 0, 1, 2, 3, 4, 4, 4]]

    attention_masks = batch.model_inputs["attention_masks"]
    assert attention_masks.shape == (1, 1, 9, 9)
    assert attention_masks.dtype == torch.bool

    # Real tokens see prior real tokens, but not left-pad keys or future tokens.
    assert not attention_masks[0, 0, 5, 0]
    assert attention_masks[0, 0, 5, 2]
    assert not attention_masks[0, 0, 5, 6]

    # Pad query rows are not trained, but keep a self edge so SDPA stays finite.
    assert attention_masks[0, 0, 0, 0]
    assert attention_masks[0, 0, 1, 1]


def test_response_logprobs_and_mask_align_with_next_token_targets() -> None:
    """Canary for the Ishikori full-sequence bridge vs TRL completion tensors."""
    request_len = 5
    response_len = 4
    prompt_ids = torch.tensor([11, 12, 13], dtype=torch.long)
    token_ids = torch.tensor([21, 22], dtype=torch.long)
    response_logprobs = torch.tensor([-0.25, -0.75], dtype=torch.float32)
    generator_logprobs, loss_mask = _aligned_response_tensors(
        response_logprobs,
        request_len=request_len,
        response_len=response_len,
    )
    episode = Episode(
        episode_id="alignment-canary",
        pad_id=0,
        request_len=request_len,
        response_len=response_len,
        completion=Completion(
            prompt=to_prompt("decode this"),
            text="0",
            prompt_ids=prompt_ids,
            token_ids=token_ids,
            logprobs=response_logprobs,
            generator_version=0,
        ),
        generator_logprobs=generator_logprobs,
        loss_mask=loss_mask,
        advantage=1.0,
    )

    batch = collate([[episode]], include_padding_metadata=True)[0]
    tokens = batch.model_inputs["tokens"]
    target_ids = torch.roll(tokens, shifts=-1, dims=-1)
    active_positions = batch.loss_inputs["loss_mask"].bool()

    assert target_ids[active_positions].tolist() == token_ids.tolist()
    assert batch.loss_inputs["generator_logprobs"][active_positions].tolist() == (
        response_logprobs.tolist()
    )
    assert active_positions.nonzero().tolist() == [
        [0, request_len - 1],
        [0, request_len],
    ]
    assert batch.model_inputs["positions"].tolist() == [[0, 0, 0, 1, 2, 3, 4, 4, 4]]
