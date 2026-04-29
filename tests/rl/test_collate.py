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
    assert batch.model_inputs["tokens"].tolist() == [
        [0, 0, 11, 12, 13, 21, 22, 0, 0]
    ]


def test_collate_can_emit_padding_mask_and_rope_positions() -> None:
    batch = collate([[_episode()]], include_padding_metadata=True)[0]

    assert set(batch.model_inputs) == {"tokens", "attention_masks", "positions"}
    assert batch.model_inputs["positions"].tolist() == [
        [0, 0, 0, 1, 2, 3, 4, 4, 4]
    ]

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
