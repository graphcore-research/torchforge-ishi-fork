# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest
import torch
from forge.actors.reference_model import ReferenceModel
from forge.controller.provisioner import shutdown
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.rl.collate import create_packed_attention_masks, pack_episode_tokens
from forge.rl.types import Episode
from forge.util.config import _resolve_hf_model_path, resolve_hf_hub_paths
from omegaconf import DictConfig, OmegaConf
from torchtitan.config.job_config import Checkpoint, Compile, Model, Parallelism
from torchtitan.models.attention import FlexAttentionWrapper


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


def _test_episodes() -> tuple[Episode, Episode, Episode]:
    return (
        _episode([101, 102, 103], [201, 202]),
        _episode([111, 112], [211, 212, 213]),
        _episode([121, 122], [221, 222, 223]),
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


def _gpt_oss_config(config_path: str | None) -> DictConfig:
    if not config_path:
        pytest.skip("GPT-OSS tests require --config pointing to a GPT-OSS fixture")
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    if cfg.trainer.model.name != "gpt_oss":
        pytest.skip("GPT-OSS tests only run with a GPT-OSS --config")
    return cfg


async def _run_reference_model(
    episodes: list[Episode],
    config: dict,
    *,
    procs: int = 1,
) -> torch.Tensor:
    return (await _run_reference_model_cases([episodes], config, procs=procs))[0]


async def _run_reference_model_cases(
    cases: list[list[Episode]],
    config: dict,
    *,
    procs: int = 1,
) -> list[torch.Tensor]:
    model = await ReferenceModel.options(
        procs=procs, num_replicas=1, with_gpus=True
    ).as_service(**config)
    try:
        results = []
        for episodes in cases:
            input_ids, prompt_lens, response_lens, seq_lens = pack_episode_tokens(
                episodes
            )
            results.append(
                await model.forward.route(
                    input_ids, prompt_lens, response_lens, seq_lens
                )
            )
        return results
    finally:
        await model.shutdown()


def _assert_tensor_pairs_close(
    comparisons: list[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> None:
    failures = []
    for name, actual, expected in comparisons:
        actual = actual.detach().cpu()
        expected = expected.detach().cpu()
        if torch.allclose(actual, expected, rtol=rtol, atol=atol):
            continue
        max_abs = (actual - expected).abs().max().item()
        failures.append(
            f"{name}: max_abs={max_abs:.6g}, actual={actual.tolist()}, "
            f"expected={expected.tolist()}"
        )
    assert not failures, "\n".join(failures)


async def _assert_debug_scenarios(config: dict, *, procs: int = 1) -> None:
    episode_a, episode_b, episode_c = _test_episodes()
    (
        single_a_first,
        single_a_second,
        packed_ab,
        packed_ba,
        packed_ac,
    ) = await _run_reference_model_cases(
        [
            [episode_a],
            [episode_a],
            [episode_a, episode_b],
            [episode_b, episode_a],
            [episode_a, episode_c],
        ],
        config,
        procs=procs,
    )

    _assert_tensor_pairs_close(
        [
            (
                "[A] vs [A] reproducibility",
                single_a_first[0, :2],
                single_a_second[0, :2],
            ),
            ("[A, B] vs [B, A] episode A", packed_ab[0, :2], packed_ba[1, :2]),
            ("[A, B] vs [B, A] episode B", packed_ab[1, :3], packed_ba[0, :3]),
            (
                "[A, B] vs [A, C] episode A isolation",
                packed_ab[0, :2],
                packed_ac[0, :2],
            ),
        ]
    )


def _episode_response_query_indices(
    seq_lens: list[int], prompt_lens: list[int], response_lens: list[int]
) -> list[list[int]]:
    indices = []
    seq_start = 0
    for seq_len, prompt_len, response_len in zip(
        seq_lens, prompt_lens, response_lens, strict=True
    ):
        start = seq_start + prompt_len - 1
        indices.append(list(range(start, start + response_len)))
        seq_start += seq_len
    return indices


def _synthetic_gpt_oss_attention_inputs(
    seq_lens: list[int],
    *,
    seed: int = 0,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_len = sum(seq_lens)
    n_heads = 16
    n_kv_heads = 2
    head_dim = 64
    q = torch.randn((1, n_heads, total_len, head_dim), generator=generator)
    k = torch.randn((1, n_kv_heads, total_len, head_dim), generator=generator)
    v = torch.randn((1, n_kv_heads, total_len, head_dim), generator=generator)
    return (
        q.to(device=device, dtype=torch.bfloat16),
        k.to(device=device, dtype=torch.bfloat16),
        v.to(device=device, dtype=torch.bfloat16),
    )


def _swap_packed_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    first_len: int,
    second_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def swap(tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                tensor[:, :, first_len : first_len + second_len],
                tensor[:, :, :first_len],
            ],
            dim=2,
        ).contiguous()

    return swap(q), swap(k), swap(v)


def _format_inner_attention_mismatch(
    name: str, actual: torch.Tensor, expected: torch.Tensor
) -> str:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    diff = (actual - expected).abs()
    flat_index = diff.argmax().item()
    return (
        f"{name}: max_abs={diff.max().item():.8g}, "
        f"actual={actual.flatten()[flat_index].item():.8g}, "
        f"expected={expected.flatten()[flat_index].item():.8g}, "
        f"flat_index={flat_index}"
    )


async def _assert_packed_matches_unpacked(config: dict, *, procs: int = 1) -> None:
    episode_a, episode_b, _ = _test_episodes()
    packed, single_a, single_b = await _run_reference_model_cases(
        [[episode_a, episode_b], [episode_a], [episode_b]], config, procs=procs
    )

    assert packed.shape == (2, 3)
    _assert_tensor_pairs_close(
        [
            ("episode A packed vs unpacked", packed[0, :2], single_a[0, :2]),
            ("episode B packed vs unpacked", packed[1, :3], single_b[0, :3]),
        ]
    )
    assert packed[0, 2].item() == 0.0


@requires_cuda
def test_reference_model_packed_matches_unpacked_qwen() -> None:
    """Check Qwen packed reference logprobs match per-episode scoring."""

    async def run() -> None:
        await _assert_packed_matches_unpacked(_qwen_reference_config())

    asyncio.run(run())


@requires_cuda
def test_reference_model_no_cross_episode_leakage_qwen() -> None:
    """Check changing one packed Qwen episode does not change another episode."""

    async def run() -> None:
        episode_a, episode_b, changed_episode_b = _test_episodes()
        original, changed = await _run_reference_model_cases(
            [[episode_a, episode_b], [episode_a, changed_episode_b]],
            _qwen_reference_config(),
        )

        _assert_tensor_pairs_close(
            [
                (
                    "episode A unchanged when episode B changes",
                    original[0, :2],
                    changed[0, :2],
                )
            ]
        )

    asyncio.run(run())


@requires_cuda
def test_reference_model_debug_scenarios_qwen() -> None:
    """Run Qwen packed reproducibility, order, and isolation diagnostics."""

    async def run() -> None:
        await _assert_debug_scenarios(_qwen_reference_config())

    asyncio.run(run())


@requires_cuda
def test_reference_model_packed_matches_unpacked_gpt_oss(
    config_path: str | None,
) -> None:
    """Check GPT-OSS packed reference logprobs match per-episode scoring."""
    cfg = _gpt_oss_config(config_path)

    async def run() -> None:
        try:
            await _assert_packed_matches_unpacked(
                _config_from_yaml(config_path),
                procs=cfg.actors.trainer.procs,
            )
        finally:
            await shutdown()

    asyncio.run(run())


@requires_cuda
def test_gpt_oss_inner_attention_order_invariance(
    config_path: str | None,
) -> None:
    """Check GPT-OSS flex attention is invariant to packed episode order."""
    cfg = _gpt_oss_config(config_path)

    device = torch.device("cuda")
    seq_lens = [5, 5]
    prompt_lens = [3, 2]
    response_lens = [2, 3]
    first_len, second_len = seq_lens

    q_ab, k_ab, v_ab = _synthetic_gpt_oss_attention_inputs(
        seq_lens, seed=52, device=device
    )
    q_ba, k_ba, v_ba = _swap_packed_attention_inputs(
        q_ab, k_ab, v_ab, first_len=first_len, second_len=second_len
    )

    attention = FlexAttentionWrapper().to(device)
    model_config = resolve_hf_hub_paths(cfg).trainer.model
    masks_ab = create_packed_attention_masks(model_config, seq_lens, device)
    masks_ba = create_packed_attention_masks(model_config, seq_lens, device)

    with torch.no_grad():
        output_ab, lse_ab = attention(
            q_ab,
            k_ab,
            v_ab,
            block_mask=masks_ab["sliding_window_mask"],
            scale=1.0 / (64**0.5),
            return_lse=True,
            enable_gqa=True,
        )
        output_ba, lse_ba = attention(
            q_ba,
            k_ba,
            v_ba,
            block_mask=masks_ba["sliding_window_mask"],
            scale=1.0 / (64**0.5),
            return_lse=True,
            enable_gqa=True,
        )

    response_queries = _episode_response_query_indices(
        seq_lens, prompt_lens, response_lens
    )
    ab_a = response_queries[0]
    ba_a = [idx + second_len for idx in response_queries[0]]
    ab_b = response_queries[1]
    ba_b = [idx - first_len for idx in response_queries[1]]

    failures = [
        _format_inner_attention_mismatch(name, actual, expected)
        for name, actual, expected in [
            ("episode A output", output_ab[:, :, ab_a], output_ba[:, :, ba_a]),
            ("episode A lse", lse_ab[:, :, ab_a], lse_ba[:, :, ba_a]),
            ("episode B output", output_ab[:, :, ab_b], output_ba[:, :, ba_b]),
            ("episode B lse", lse_ab[:, :, ab_b], lse_ba[:, :, ba_b]),
        ]
        if not torch.allclose(actual.float(), expected.float(), rtol=0.0, atol=0.0)
    ]
    assert not failures, "\n".join(failures)


@requires_cuda
def test_reference_model_debug_scenarios_gpt_oss(
    config_path: str | None,
) -> None:
    """Run GPT-OSS packed reproducibility, order, and isolation diagnostics."""
    cfg = _gpt_oss_config(config_path)

    async def run() -> None:
        try:
            await _assert_debug_scenarios(
                _config_from_yaml(config_path),
                procs=cfg.actors.trainer.procs,
            )
        finally:
            await shutdown()

    asyncio.run(run())
