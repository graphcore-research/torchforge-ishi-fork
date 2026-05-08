# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility entrypoint for reference-model packed logprob regression tests."""

import pytest
import torch
from forge.actors.reference_model import ReferenceModel
from forge.controller.provisioner import shutdown
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.rl.collate import pack_episode_tokens
from forge.rl.types import Episode
from forge.util.config import _resolve_hf_model_path
from torchtitan.config.job_config import Checkpoint, Compile, Model, Parallelism


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _episode(prompt_ids: list[int], token_ids: list[int]) -> Episode:
    return Episode(
        episode_id=f"episode-{len(prompt_ids)}-{len(token_ids)}",
        request="prompt",
        response="response",
        completion=Completion(
            prompt=Prompt.from_prompt("prompt"),
            text="response",
            prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
            token_ids=torch.tensor(token_ids, dtype=torch.long),
            logprobs=torch.zeros(len(token_ids), dtype=torch.float32),
            generator_version=0,
        ),
        advantage=0.0,
    )


def _qwen_reference_config() -> dict:
    model_path = _resolve_hf_model_path("hf://Qwen/Qwen3-0.6B")
    return {
        "model": Model(
            name="qwen3",
            flavor="0.6B",
            hf_assets_path=model_path,
        ),
        "parallelism": Parallelism(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
        ),
        "checkpoint": Checkpoint(
            enable=True,
            initial_load_path=model_path,
            initial_load_model_only=True,
            initial_load_in_hf=True,
        ),
        "compile": Compile(enable=False),
    }


@pytest.mark.asyncio
@requires_cuda
async def test_reference_model_packed_matches_unpacked_qwen() -> None:
    episodes = [
        _episode([101, 102, 103], [201, 202]),
        _episode([111, 112], [211, 212, 213]),
    ]
    model = await ReferenceModel.options(
        procs=1, num_replicas=1, with_gpus=True
    ).as_service(**_qwen_reference_config())
    try:
        packed_inputs = pack_episode_tokens(episodes)
        single_a_inputs = pack_episode_tokens([episodes[0]])
        single_b_inputs = pack_episode_tokens([episodes[1]])

        packed = await model.forward.route(*packed_inputs)
        single_a = await model.forward.route(*single_a_inputs)
        single_b = await model.forward.route(*single_b_inputs)
    finally:
        await model.shutdown()
        await shutdown()

    assert packed.shape == (2, 3)
    assert torch.allclose(
        packed[0, :2].cpu(), single_a[0, :2].cpu(), rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        packed[1, :3].cpu(), single_b[0, :3].cpu(), rtol=1e-4, atol=1e-4
    )
    assert packed[0, 2].item() == 0.0
