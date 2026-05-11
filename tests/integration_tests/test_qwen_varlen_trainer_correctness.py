# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import monarch
import pytest
import pytest_asyncio
import torch
import torchstore as ts
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner
from forge.controller.service.service import uuid
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.data.utils import batch_to_device
from forge.rl.collate import collate, materialize_dtensor
from forge.rl.loss import compute_logprobs
from forge.rl.loss.types import LossOutput
from forge.rl.types import Episode
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import resolve_hf_hub_paths
from huggingface_hub import snapshot_download
from monarch.actor import endpoint
from omegaconf import DictConfig, OmegaConf

monarch.actor.unhandled_fault_hook = lambda failure: None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

FIXTURE_DIR = Path("tests/integration_tests/fixtures")
NO_TP_CONFIG = FIXTURE_DIR / "qwen3_1_7b_no_tp.yaml"
TP_CONFIG = FIXTURE_DIR / "qwen3_1_7b_tp.yaml"
RESULT_DIR = Path("/data/seanc/tmp/qwen_varlen_trainer_correctness")
CHECKPOINT_DIR = Path("/data/seanc/tmp/qwen_varlen_trainer_correctness_checkpoints")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _json_tensor(value: torch.Tensor | Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = materialize_dtensor(value)
        return value.detach().cpu().tolist()
    return value


def _tensor_shape(value: torch.Tensor | Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        value = materialize_dtensor(value)
        return list(value.shape)
    return None


def _write_result(name: str, payload: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(
        f"qwen-varlen-trainer result {name}: {json.dumps(payload, sort_keys=True)}",
        flush=True,
    )


def _unwrap_actor_result(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "item"):
        try:
            item = value.item()
            if isinstance(item, dict):
                return item
        except Exception:
            pass
    if hasattr(value, "values"):
        values = list(value.values())
        if values and isinstance(values[0], dict):
            return values[0]
    raise TypeError(f"Expected dict-like actor result, got {type(value)!r}: {value!r}")


def _load_config(config_path: Path) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    cfg = resolve_hf_hub_paths(cfg)
    if cfg.trainer.model.name != "qwen3":
        pytest.skip("Qwen varlen trainer tests only run with Qwen3 configs")
    return cfg


def _completion(prompt_ids: list[int], token_ids: list[int]) -> Completion:
    return Completion(
        prompt=Prompt.from_prompt("prompt"),
        text="response",
        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        logprobs=torch.zeros(len(token_ids), dtype=torch.float32),
        generator_version=0,
    )


def _episode(name: str, prompt_ids: list[int], token_ids: list[int]) -> Episode:
    return Episode(
        episode_id=name,
        request="prompt",
        response="response",
        completion=_completion(prompt_ids, token_ids),
        advantage=0.0,
    )


def _base_episodes() -> list[Episode]:
    return [
        _episode("A", [101, 102, 103], [201, 202]),
        _episode("B", [111, 112], [211, 212, 213]),
        _episode("C", [121, 122, 123, 124], [221]),
    ]


def _variable_length_episodes() -> list[Episode]:
    return [
        _episode("short-short", [101, 102], [201]),
        _episode("long-short", [111, 112, 113, 114, 115, 116, 117], [211, 212]),
        _episode("short-long", [121], [221, 222, 223, 224]),
    ]


def _long_prompt_episodes() -> list[Episode]:
    prompt_a = [1000 + (i % 100) for i in range(128)]
    prompt_b = [2000 + (i % 100) for i in range(256)]
    return [
        _episode("long-128", prompt_a, [3101, 3102, 3103]),
        _episode("long-256", prompt_b, [3201, 3202]),
    ]


def _collated(episodes: list[Episode]):
    return collate([episodes])


def _assert_shape_probe(
    result: dict[str, Any], *, num_episodes: int, max_response_len: int
) -> None:
    assert result["reached_loss"] is True
    assert result["logits_shape"][:2] == [num_episodes, max_response_len]
    assert result["target_ids_shape"] == [num_episodes, max_response_len]
    assert result["loss_mask_shape"] == [num_episodes, max_response_len]
    assert result["trainer_logprobs_shape"] == [num_episodes, max_response_len]
    assert result["finite_masked_logprobs"] is True
    assert result["finite_loss"] is True


def _valid_logprobs(
    result: dict[str, Any], episode_index: int, response_len: int
) -> torch.Tensor:
    return torch.tensor(result["trainer_logprobs"][episode_index][:response_len])


def _compare_tensors(
    name: str, actual: torch.Tensor, expected: torch.Tensor, *, rtol: float, atol: float
) -> dict[str, Any]:
    diff = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    violation = diff - allowed
    return {
        "name": name,
        "max_error": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_error": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_violation": float(violation.max().item()) if violation.numel() else 0.0,
        "actual": actual.tolist(),
        "expected": expected.tolist(),
        "passed": bool(torch.allclose(actual, expected, rtol=rtol, atol=atol)),
    }


def _assert_comparisons(test_name: str, comparisons: list[dict[str, Any]]) -> None:
    payload = {"comparisons": comparisons}
    _write_result(test_name, payload)
    failures = [item for item in comparisons if not item["passed"]]
    assert not failures, json.dumps(failures, indent=2, sort_keys=True)


class QwenVarlenProbeTrainer(TitanTrainer):
    @endpoint
    async def probe_forward_backward(
        self, batches, loss_mode: str = "zero"
    ) -> dict[str, Any]:
        batch = batches[self.engine.dp_rank]
        batch_to_device(batch.model_inputs, self.engine.device)
        batch_to_device(batch.loss_inputs, self.engine.device)

        original_loss = self.loss
        probe: dict[str, Any] = {"reached_loss": False, "loss_mode": loss_mode}

        def probe_loss(
            logits, target_ids, generator_logprobs, loss_mask, **_
        ) -> LossOutput:
            nonlocal probe
            logprobs, _ = compute_logprobs(logits, target_ids)
            mask = loss_mask.bool()
            masked_logprobs = logprobs[mask]
            finite_logits = torch.isfinite(logits).all()
            finite_masked_logprobs = torch.isfinite(masked_logprobs).all()

            if loss_mode == "nll":
                loss = -(logprobs * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            elif loss_mode == "zero":
                loss = logits.sum() * 0.0
            else:
                raise ValueError(f"unknown probe loss mode: {loss_mode}")

            probe = {
                "reached_loss": True,
                "loss_mode": loss_mode,
                "logits_shape": list(logits.shape),
                "target_ids_shape": list(target_ids.shape),
                "generator_logprobs_shape": list(generator_logprobs.shape),
                "loss_mask_shape": list(loss_mask.shape),
                "trainer_logprobs_shape": list(logprobs.shape),
                "finite_logits": bool(finite_logits.detach().cpu()),
                "finite_masked_logprobs": bool(finite_masked_logprobs.detach().cpu()),
                "target_ids": _json_tensor(target_ids),
                "generator_logprobs": _json_tensor(generator_logprobs),
                "trainer_logprobs": _json_tensor(logprobs),
                "loss_mask": _json_tensor(loss_mask),
                "model_input_shapes": {
                    key: _tensor_shape(value)
                    for key, value in batch.model_inputs.items()
                },
                "loss_input_shapes": {
                    key: _tensor_shape(value)
                    for key, value in batch.loss_inputs.items()
                },
                "meta": batch.meta,
            }
            return LossOutput(loss=loss, metrics=[])

        try:
            self.loss = probe_loss
            loss = self.forward_backward(batch)
            probe["loss"] = float(loss.detach().cpu())
            probe["finite_loss"] = bool(torch.isfinite(loss.detach()).cpu())
            probe["grad_summary"] = self._grad_summary()
            return probe
        finally:
            self.loss = original_loss
            self.engine.optimizers.zero_grad()
            self._accumulated_microbatches = 0

    def _grad_summary(self) -> dict[str, Any]:
        num_with_grad = 0
        num_finite = 0
        num_nonzero = 0
        max_abs = 0.0
        sampled = 0
        for model_part in self.engine.model_parts:
            for name, param in model_part.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                grad = materialize_dtensor(param.grad.detach())
                grad = grad.float()
                num_with_grad += 1
                finite = torch.isfinite(grad).all()
                nonzero = bool((grad != 0).any().detach().cpu())
                num_finite += int(bool(finite.detach().cpu()))
                num_nonzero += int(nonzero)
                if grad.numel():
                    max_abs = max(max_abs, float(grad.abs().max().detach().cpu()))
                sampled += 1
                if sampled >= 32:
                    break
            if sampled >= 32:
                break
        return {
            "num_with_grad_sampled": num_with_grad,
            "num_finite_grad_sampled": num_finite,
            "num_nonzero_grad_sampled": num_nonzero,
            "max_abs_grad_sampled": max_abs,
        }


async def _setup_probe_trainer(config_path: Path):
    cfg = _load_config(config_path)
    checkpoint_path = cfg.get("local_checkpoint", None)
    if checkpoint_path is None:
        logger.info("Downloading model checkpoint from HuggingFace Hub")
        checkpoint_path = snapshot_download(repo_id=cfg.model)
        logger.info("Finished downloading model checkpoint from HuggingFace Hub")

    cfg.trainer.checkpoint = {
        "enable": True,
        "folder": str(CHECKPOINT_DIR / config_path.stem),
        "initial_load_path": checkpoint_path,
        "initial_load_in_hf": True,
    }
    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    try:
        await ts.initialize(strategy=ts.ControllerStorageVolumes())
    except Exception as exc:
        if "already initialized" not in str(exc):
            raise
        logger.warning("Reusing existing TorchStore initialization for %s", config_path)
    trainer = await QwenVarlenProbeTrainer.options(**cfg.actors.trainer).as_actor(
        **cfg.trainer
    )
    return cfg, trainer


async def _teardown_probe_trainer(trainer) -> None:
    try:
        await trainer.cleanup.call()
    except Exception as exc:
        logger.warning("Ignoring trainer cleanup failure: %r", exc)
    try:
        await TitanTrainer.shutdown(trainer)
    except Exception as exc:
        logger.warning("Ignoring trainer shutdown failure: %r", exc)
    try:
        await ts.shutdown()
    except Exception as exc:
        logger.warning("Ignoring torchstore shutdown failure: %r", exc)


@pytest_asyncio.fixture(scope="module")
async def no_tp_trainer():
    cfg, trainer = await _setup_probe_trainer(NO_TP_CONFIG)
    try:
        yield cfg, trainer
    finally:
        await _teardown_probe_trainer(trainer)


@pytest_asyncio.fixture(scope="module")
async def tp_trainer():
    cfg, trainer = await _setup_probe_trainer(TP_CONFIG)
    try:
        yield cfg, trainer
    finally:
        await _teardown_probe_trainer(trainer)


async def _probe(
    trainer, episodes: list[Episode], *, loss_mode: str = "zero"
) -> dict[str, Any]:
    return _unwrap_actor_result(
        await trainer.probe_forward_backward.call(_collated(episodes), loss_mode)
    )


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_forward_backward_shapes_tp1(no_tp_trainer):
    """Check no-TP Qwen varlen trainer reaches loss with response-shaped logits."""
    _, trainer = no_tp_trainer
    episodes = _base_episodes()
    result = await _probe(trainer, episodes)
    _write_result("test_qwen_varlen_trainer_forward_backward_shapes_tp1", result)
    _assert_shape_probe(result, num_episodes=3, max_response_len=3)


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_forward_backward_shapes_tp(tp_trainer):
    """Check TP Qwen varlen trainer reaches loss with response-shaped logits."""
    _, trainer = tp_trainer
    episodes = _base_episodes()
    result = await _probe(trainer, episodes)
    _write_result("test_qwen_varlen_trainer_forward_backward_shapes_tp", result)
    _assert_shape_probe(result, num_episodes=3, max_response_len=3)


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_variable_length_batch(no_tp_trainer):
    """Check variable response lengths produce correct masks and padded shapes."""
    _, trainer = no_tp_trainer
    episodes = _variable_length_episodes()
    result = await _probe(trainer, episodes)
    _write_result("test_qwen_varlen_trainer_variable_length_batch", result)
    _assert_shape_probe(result, num_episodes=3, max_response_len=4)
    assert result["loss_mask"] == [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ]


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_response_slicing_matches_single_episode_runs(
    no_tp_trainer,
):
    """Check packed trainer response logprobs match separate per-episode runs."""
    _, trainer = no_tp_trainer
    episodes = _base_episodes()
    packed = await _probe(trainer, episodes)
    singles = [await _probe(trainer, [episode]) for episode in episodes]

    comparisons = []
    for index, episode in enumerate(episodes):
        response_len = len(episode.completion.token_ids)
        comparisons.append(
            _compare_tensors(
                f"episode_{episode.episode_id}",
                _valid_logprobs(packed, index, response_len),
                _valid_logprobs(singles[index], 0, response_len),
                rtol=1e-3,
                atol=1e-3,
            )
        )
    _assert_comparisons(
        "test_qwen_varlen_trainer_response_slicing_matches_single_episode_runs",
        comparisons,
    )


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_no_cross_episode_leakage(no_tp_trainer):
    """Check changing one packed episode does not change another episode's logprobs."""
    _, trainer = no_tp_trainer
    episode_a = _episode("A", [101, 102, 103], [201, 202])
    episode_b = _episode("B", [111, 112], [211, 212, 213])
    episode_c = _episode("C", [121, 122, 123], [221, 222, 223])

    ab = await _probe(trainer, [episode_a, episode_b])
    ac = await _probe(trainer, [episode_a, episode_c])
    comparison = _compare_tensors(
        "episode_A_in_AB_vs_AC",
        _valid_logprobs(ab, 0, 2),
        _valid_logprobs(ac, 0, 2),
        rtol=1e-3,
        atol=1e-3,
    )
    _assert_comparisons(
        "test_qwen_varlen_trainer_no_cross_episode_leakage", [comparison]
    )


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_packed_order_invariance(no_tp_trainer):
    """Check trainer logprobs are invariant to packed episode ordering."""
    _, trainer = no_tp_trainer
    episode_a = _episode("A", [101, 102, 103], [201, 202])
    episode_b = _episode("B", [111, 112], [211, 212, 213])

    ab = await _probe(trainer, [episode_a, episode_b])
    ba = await _probe(trainer, [episode_b, episode_a])
    comparisons = [
        _compare_tensors(
            "episode_A_AB_vs_BA",
            _valid_logprobs(ab, 0, 2),
            _valid_logprobs(ba, 1, 2),
            rtol=1e-3,
            atol=1e-3,
        ),
        _compare_tensors(
            "episode_B_AB_vs_BA",
            _valid_logprobs(ab, 1, 3),
            _valid_logprobs(ba, 0, 3),
            rtol=1e-3,
            atol=1e-3,
        ),
    ]
    _assert_comparisons("test_qwen_varlen_trainer_packed_order_invariance", comparisons)


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_tp1_vs_tp_parity(no_tp_trainer, tp_trainer):
    """Compare no-TP and TP trainer logprobs on the same packed batch."""
    _, no_tp = no_tp_trainer
    _, tp = tp_trainer
    episodes = _base_episodes()
    no_tp_result = await _probe(no_tp, episodes)
    tp_result = await _probe(tp, episodes)

    comparisons = []
    for index, episode in enumerate(episodes):
        response_len = len(episode.completion.token_ids)
        comparisons.append(
            _compare_tensors(
                f"episode_{episode.episode_id}_tp1_vs_tp",
                _valid_logprobs(tp_result, index, response_len),
                _valid_logprobs(no_tp_result, index, response_len),
                rtol=1e-2,
                atol=1e-2,
            )
        )
    _assert_comparisons("test_qwen_varlen_trainer_tp1_vs_tp_parity", comparisons)


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_backward_gradient_sanity_tp1_and_tp(
    no_tp_trainer, tp_trainer
):
    """Check masked NLL backward produces finite nonzero gradients for no-TP and TP."""
    results = {}
    for label, fixture in [("tp1", no_tp_trainer), ("tp", tp_trainer)]:
        _, trainer = fixture
        result = await _probe(trainer, _base_episodes(), loss_mode="nll")
        results[label] = result
        _assert_shape_probe(result, num_episodes=3, max_response_len=3)
        grad_summary = result["grad_summary"]
        assert grad_summary["num_with_grad_sampled"] > 0
        assert (
            grad_summary["num_finite_grad_sampled"]
            == grad_summary["num_with_grad_sampled"]
        )
        assert grad_summary["num_nonzero_grad_sampled"] > 0
    _write_result(
        "test_qwen_varlen_trainer_backward_gradient_sanity_tp1_and_tp", results
    )


@pytest.mark.asyncio
@requires_cuda
async def test_qwen_varlen_trainer_long_prompt_forward_backward(
    no_tp_trainer, tp_trainer
):
    """Check longer packed prompts reach trainer forward/backward with finite logprobs."""
    episodes = _long_prompt_episodes()
    results = {}
    for label, fixture in [("tp1", no_tp_trainer), ("tp", tp_trainer)]:
        _, trainer = fixture
        result = await _probe(trainer, episodes)
        results[label] = result
        _assert_shape_probe(result, num_episodes=2, max_response_len=3)
    _write_result("test_qwen_varlen_trainer_long_prompt_forward_backward", results)
