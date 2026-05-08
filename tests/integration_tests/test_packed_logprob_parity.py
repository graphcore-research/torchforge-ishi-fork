# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from forge.actors.reference_model import ReferenceModel
from forge.controller.provisioner import shutdown
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.rl.collate import pack_episode_tokens
from forge.rl.types import Episode
from forge.util.config import _resolve_hf_model_path, resolve_hf_hub_paths
from omegaconf import DictConfig, OmegaConf
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


def _config_from_yaml(config_path: str) -> dict:
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    return {
        "model": cfg.trainer.model,
        "parallelism": cfg.trainer.parallelism,
        "checkpoint": cfg.trainer.checkpoint,
        "compile": cfg.trainer.compile,
        "training": cfg.trainer.training,
    }


async def _run_reference_model(
    episodes: list[Episode],
    config: dict,
    *,
    procs: int = 1,
) -> torch.Tensor:
    model = await ReferenceModel.options(
        procs=procs, num_replicas=1, with_gpus=True
    ).as_service(**config)
    try:
        input_ids, prompt_lens, response_lens, seq_lens = pack_episode_tokens(episodes)
        return await model.forward.route(
            input_ids, prompt_lens, response_lens, seq_lens
        )
    finally:
        await model.shutdown()


async def _assert_packed_matches_unpacked(config: dict, *, procs: int = 1) -> None:
    episodes = [
        _episode([101, 102, 103], [201, 202]),
        _episode([111, 112], [211, 212, 213]),
    ]

    packed = await _run_reference_model(episodes, config, procs=procs)
    single_a = await _run_reference_model([episodes[0]], config, procs=procs)
    single_b = await _run_reference_model([episodes[1]], config, procs=procs)

    assert packed.shape == (2, 3)
    assert torch.allclose(
        packed[0, :2].cpu(), single_a[0, :2].cpu(), rtol=1e-4, atol=1e-4
    )
    assert torch.allclose(
        packed[1, :3].cpu(), single_b[0, :3].cpu(), rtol=1e-4, atol=1e-4
    )
    assert packed[0, 2].item() == 0.0


@pytest.mark.asyncio
@requires_cuda
async def test_reference_model_packed_matches_unpacked_qwen() -> None:
    try:
        await _assert_packed_matches_unpacked(_qwen_reference_config())
    finally:
        await shutdown()


@pytest.mark.asyncio
@requires_cuda
async def test_reference_model_no_cross_episode_leakage_qwen() -> None:
    episode_a = _episode([101, 102, 103], [201, 202])
    episode_b = _episode([111, 112], [211, 212, 213])
    changed_episode_b = _episode([121, 122], [221, 222, 223])

    try:
        original = await _run_reference_model(
            [episode_a, episode_b], _qwen_reference_config()
        )
        changed = await _run_reference_model(
            [episode_a, changed_episode_b], _qwen_reference_config()
        )
    finally:
        await shutdown()

    assert torch.allclose(
        original[0, :2].cpu(), changed[0, :2].cpu(), rtol=1e-4, atol=1e-4
    )


@pytest.mark.asyncio
@requires_cuda
async def test_reference_model_packed_matches_unpacked_gpt_oss(
    config_path: str | None,
) -> None:
    if not config_path:
        pytest.skip("GPT-OSS parity requires --config pointing to a GPT-OSS fixture")
    cfg = OmegaConf.load(config_path)
    if cfg.trainer.model.name != "gpt_oss":
        pytest.skip("GPT-OSS parity only runs with a GPT-OSS --config")

    try:
        await _assert_packed_matches_unpacked(
            _config_from_yaml(config_path),
            procs=cfg.actors.trainer.procs,
        )
    finally:
        await shutdown()
