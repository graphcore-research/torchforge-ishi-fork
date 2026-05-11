# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import monarch
import pytest
import torch
import torchstore as ts
from forge.actors.trainer import TitanTrainer
from forge.controller.provisioner import init_provisioner
from forge.controller.service.service import uuid
from forge.data_models.completion import Completion
from forge.rl.collate import collate
from forge.rl.loss import compute_logprobs
from forge.rl.loss.types import LossOutput
from forge.rl.types import Episode
from forge.types import LauncherConfig, ProvisionerConfig
from forge.util.config import resolve_hf_hub_paths
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from monarch.actor import endpoint
from omegaconf import DictConfig, OmegaConf

monarch.actor.unhandled_fault_hook = lambda failure: None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PARITY_RESULT_FILENAME = "torchforge_trainer_generator_logprob_parity.json"
PARITY_GENERATOR_DEBUG_FILENAME = (
    "torchforge_trainer_generator_logprob_parity_generator_debug.json"
)
PARITY_TRAINER_DEBUG_FILENAME = (
    "torchforge_trainer_generator_logprob_parity_trainer_debug.json"
)


def _gc_tmp_dir() -> Path:
    gc_user = os.environ.get("GC_USER")
    if not gc_user:
        raise RuntimeError(
            "GC_USER must be set to run trainer/generator parity tests; "
            "expected artifacts under /data/{GC_USER}/tmp."
        )
    path = Path("/data") / gc_user / "tmp"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parity_result_path() -> Path:
    return _gc_tmp_dir() / PARITY_RESULT_FILENAME


def _parity_generator_debug_path() -> Path:
    return _gc_tmp_dir() / PARITY_GENERATOR_DEBUG_FILENAME


def _parity_trainer_debug_path() -> Path:
    return _gc_tmp_dir() / PARITY_TRAINER_DEBUG_FILENAME


def _parity_artifact_paths() -> tuple[Path, ...]:
    return (
        _parity_result_path(),
        _parity_generator_debug_path(),
        _parity_trainer_debug_path(),
    )


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _json_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload))


def _clear_parity_artifacts() -> None:
    for path in _parity_artifact_paths():
        path.unlink(missing_ok=True)


def _debug_enabled() -> bool:
    return os.environ.get("FORGE_LOGPROB_PARITY_DEBUG") == "1"


def _parity_result(max_error, mean_error, max_violation) -> dict[str, float]:
    return {
        "max_error": float(max_error.detach().cpu()),
        "mean_error": float(mean_error.detach().cpu()),
        "max_violation": float(max_violation.detach().cpu()),
    }


def _result_message(prefix: str, result: dict[str, float]) -> str:
    return (
        f"{prefix}: "
        f"max_error={result['max_error']:.6g}, "
        f"mean_error={result['mean_error']:.6g}, "
        f"max_violation={result['max_violation']:.6g}"
    )


def _tensor_debug_map(mapping) -> dict:
    return {
        key: {
            "shape": list(value.shape) if isinstance(value, torch.Tensor) else None,
            "value": _json_tensor(value),
        }
        for key, value in mapping.items()
    }


def _decode_token(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([token_id])
    except Exception:
        return "<decode-error>"


def _write_trainer_debug(
    result: dict[str, float],
    *,
    target_ids,
    generator_logprobs,
    trainer_logprobs,
    loss_mask,
    diff,
    allowed,
    masked_diff,
) -> None:
    if not _debug_enabled():
        return
    _write_json(
        _parity_trainer_debug_path(),
        {
            **result,
            "target_ids": _json_tensor(target_ids),
            "generator_logprobs": _json_tensor(generator_logprobs),
            "trainer_logprobs": _json_tensor(trainer_logprobs),
            "loss_mask": _json_tensor(loss_mask),
            "diff": _json_tensor(diff),
            "allowed": _json_tensor(allowed),
            "masked_diff": _json_tensor(masked_diff),
        },
    )


def _print_parity_debug(model_name: str) -> None:
    if not (
        _parity_generator_debug_path().exists()
        and _parity_trainer_debug_path().exists()
    ):
        return

    generator_debug = json.loads(_parity_generator_debug_path().read_text())
    trainer_debug = json.loads(_parity_trainer_debug_path().read_text())
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("trainer/generator parity debug summary:", flush=True)
    completion = generator_debug["completions"][0]
    print(f"  prompt_text={completion['prompt_text']!r}", flush=True)
    print(f"  response_text={completion['response_text']!r}", flush=True)
    print(f"  generator_prompt_ids={completion['prompt_ids']}", flush=True)
    print(f"  generator_token_ids={completion['token_ids']}", flush=True)
    print(
        f"  collated_model_inputs={generator_debug['batch']['model_inputs']}",
        flush=True,
    )
    for key in ("prompt_lens", "response_lens", "seq_lens"):
        if key in generator_debug["batch"]["model_inputs"]:
            print(
                f"  collated_{key}="
                f"{generator_debug['batch']['model_inputs'][key]['value']}",
                flush=True,
            )

    target_ids = trainer_debug["target_ids"][0]
    generator_logprobs = trainer_debug["generator_logprobs"][0]
    trainer_logprobs = trainer_debug["trainer_logprobs"][0]
    loss_mask = trainer_debug["loss_mask"][0]
    diffs = trainer_debug["diff"][0]

    print("  per-token comparison:", flush=True)
    print(
        "    idx | mask | target_id | token | generator_lp | trainer_lp | abs_delta",
        flush=True,
    )
    for idx, (target_id, gen_lp, trainer_lp, mask, diff) in enumerate(
        zip(
            target_ids,
            generator_logprobs,
            trainer_logprobs,
            loss_mask,
            diffs,
            strict=False,
        )
    ):
        token = _decode_token(tokenizer, int(target_id)).replace("\n", "\\n")
        print(
            f"    {idx:03d} | {int(mask)} | {int(target_id)} | {token!r} | "
            f"{float(gen_lp): .6f} | {float(trainer_lp): .6f} | {float(diff): .6f}",
            flush=True,
        )


def _configure_sampling_for_parity(cfg: DictConfig) -> None:
    cfg.generator.sampling_params.logprobs = 1
    if _debug_enabled():
        cfg.generator.sampling_params.n = 1
        cfg.generator.sampling_params.max_tokens = 8
        cfg.generator.sampling_params.temperature = 0.0
        cfg.generator.sampling_params.top_p = 1.0


def _write_generator_debug(episodes, completions, batch) -> None:
    if not _debug_enabled():
        return
    _write_json(
        _parity_generator_debug_path(),
        {
            "completions": [
                {
                    "prompt_text": episode.request,
                    "response_text": completion.text,
                    "prompt_ids": _json_tensor(completion.prompt_ids),
                    "token_ids": _json_tensor(completion.token_ids),
                    "logprobs": _json_tensor(completion.logprobs),
                    "metadata": completion.metadata,
                }
                for episode, completion in zip(episodes, completions, strict=False)
            ],
            "batch": {
                "model_inputs": _tensor_debug_map(batch.model_inputs),
                "loss_inputs": _tensor_debug_map(batch.loss_inputs),
            },
        },
    )


def _is_gpt_oss_config(cfg: DictConfig) -> bool:
    return cfg.trainer.model.name == "gpt_oss"


class LogprobParityTrainer(TitanTrainer):
    @endpoint
    async def assert_generator_logprob_parity(
        self,
        batches,
        rtol: float,
        atol: float,
    ) -> dict[str, float]:
        """Invoke production forward_backward with a test loss that checks logprobs."""
        from forge.data.utils import batch_to_device

        batch = batches[self.engine.dp_rank]
        batch_to_device(batch.model_inputs, self.engine.device)
        batch_to_device(batch.loss_inputs, self.engine.device)

        original_loss = self.loss
        max_error = torch.tensor(0.0, device=self.engine.device)
        mean_error = torch.tensor(0.0, device=self.engine.device)
        max_violation = torch.tensor(0.0, device=self.engine.device)

        def parity_loss(
            logits,
            target_ids,
            generator_logprobs,
            loss_mask,
            **_,
        ) -> LossOutput:
            nonlocal max_error, mean_error, max_violation
            logprobs, _ = compute_logprobs(logits, target_ids)
            mask = loss_mask.bool()
            diff = (logprobs - generator_logprobs).abs()
            allowed = atol + rtol * generator_logprobs.abs()
            masked_diff = torch.where(mask, diff, torch.zeros_like(diff))
            violation = torch.where(mask, diff - allowed, torch.zeros_like(diff))
            max_error = masked_diff.max()
            mean_error = masked_diff.sum() / loss_mask.sum().clamp(min=1.0)
            max_violation = violation.max()
            if self.engine.dp_rank == 0:
                result = _parity_result(max_error, mean_error, max_violation)
                _write_json(_parity_result_path(), result)
                _write_trainer_debug(
                    result,
                    target_ids=target_ids,
                    generator_logprobs=generator_logprobs,
                    trainer_logprobs=logprobs,
                    loss_mask=loss_mask,
                    diff=diff,
                    allowed=allowed,
                    masked_diff=masked_diff,
                )
            return LossOutput(loss=logits.sum() * 0.0, metrics=[])

        try:
            self.loss = parity_loss
            loss = self.forward_backward(batch)
            loss.detach()
            self.engine.optimizers.zero_grad()
            self._accumulated_microbatches = 0
            return {
                "max_error": float(max_error.detach().cpu()),
                "mean_error": float(mean_error.detach().cpu()),
                "max_violation": float(max_violation.detach().cpu()),
            }
        finally:
            self.loss = original_loss


def _load_config(config_path: str) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    assert isinstance(cfg, DictConfig)
    return resolve_hf_hub_paths(cfg)


def _episode_from_completion(completion: Completion) -> Episode:
    return Episode(
        episode_id=str(uuid.uuid4()),
        request="prompt",
        response=completion.text,
        completion=completion,
        advantage=0.0,
    )


async def _setup_generator_and_trainer(config_path: str):
    try:
        from forge.actors.generator import Generator
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Generator dependencies are unavailable: {exc}")

    cfg = _load_config(config_path)
    checkpoint_path = cfg.get("local_checkpoint", None)
    if checkpoint_path is None:
        logger.info("Downloading model checkpoint from HuggingFace Hub")
        checkpoint_path = snapshot_download(repo_id=cfg.model)
        logger.info("Finished downloading model checkpoint from HuggingFace Hub")

    _configure_sampling_for_parity(cfg)
    cfg.services.generator.num_replicas = 1
    cfg.trainer.checkpoint = {
        "enable": True,
        "folder": str(_gc_tmp_dir() / "logprob_parity_checkpoints"),
        "initial_load_path": checkpoint_path,
        "initial_load_in_hf": True,
    }

    if cfg.get("provisioner", None) is not None:
        await init_provisioner(
            ProvisionerConfig(launcher_config=LauncherConfig(**cfg.provisioner))
        )
    await ts.initialize(strategy=ts.ControllerStorageVolumes())

    generator, trainer = await asyncio.gather(
        Generator.options(**cfg.services.generator).as_service(**cfg.generator),
        LogprobParityTrainer.options(**cfg.actors.trainer).as_actor(**cfg.trainer),
    )
    return cfg, generator, trainer


async def _teardown(generator, trainer) -> None:
    # This test's signal is the parity assertion; vLLM/Monarch cleanup can
    # time out after the result is known, so teardown is best-effort.
    cleanup_steps = []
    if trainer is not None:
        cleanup_steps.extend(
            [
                ("trainer cleanup", trainer.cleanup.call),
                ("trainer shutdown", lambda: TitanTrainer.shutdown(trainer)),
            ]
        )
    if generator is not None:
        logger.warning(
            "Skipping generator shutdown in this parity test; Monarch/vLLM "
            "shutdown timeouts can hide the already-computed parity result."
        )
    cleanup_steps.append(("torchstore shutdown", ts.shutdown))

    for label, cleanup in cleanup_steps:
        try:
            await cleanup()
        except Exception as exc:
            logger.warning(
                "Ignoring %s failure during best-effort teardown: %r", label, exc
            )


async def _assert_trainer_generator_parity(
    generator,
    trainer,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    prompt = os.environ.get("FORGE_LOGPROB_PARITY_PROMPT", "What is 2 + 2?")
    completions = await generator.generate.route(prompt)
    episodes = [_episode_from_completion(completion) for completion in completions]
    batches = collate([episodes])
    _write_generator_debug(episodes, completions, batches[0])
    await trainer.assert_generator_logprob_parity.call(batches, rtol, atol)
    return json.loads(_parity_result_path().read_text())


async def _run_parity_case(
    config_path: str | None,
    *,
    expect_gpt_oss: bool,
    rtol: float,
    atol: float,
) -> None:
    if not config_path:
        pytest.skip("Logprob parity requires --config")

    cfg = _load_config(config_path)
    is_gpt_oss = _is_gpt_oss_config(cfg)
    if expect_gpt_oss and not is_gpt_oss:
        pytest.skip("GPT-OSS parity only runs with a GPT-OSS --config")
    if not expect_gpt_oss and is_gpt_oss:
        pytest.skip("Use the GPT-OSS-specific parity test for GPT-OSS configs")

    generator = None
    trainer = None
    _clear_parity_artifacts()
    try:
        _, generator, trainer = await _setup_generator_and_trainer(config_path)
        result = await _assert_trainer_generator_parity(
            generator,
            trainer,
            rtol=rtol,
            atol=atol,
        )
    except Exception:
        if _parity_result_path().exists():
            result = json.loads(_parity_result_path().read_text())
            print(
                _result_message(
                    "trainer/generator parity result before failure", result
                ),
                flush=True,
            )
        if _debug_enabled():
            _print_parity_debug(cfg.model)
        raise
    else:
        print(_result_message("trainer/generator parity result", result), flush=True)
        if _debug_enabled():
            _print_parity_debug(cfg.model)
        assert result["max_violation"] <= 0.0
    finally:
        await _teardown(generator, trainer)


@requires_cuda
def test_trainer_generator_logprob_parity_qwen_tp_or_config(
    config_path: str | None,
) -> None:
    """Compare Qwen generator rollout logprobs against trainer recomputed logprobs."""
    asyncio.run(
        _run_parity_case(
            config_path,
            expect_gpt_oss=False,
            rtol=1e-2,
            atol=1e-2,
        )
    )


@requires_cuda
def test_trainer_generator_logprob_parity_gpt_oss_config(
    config_path: str | None,
) -> None:
    """Compare GPT-OSS generator and trainer logprobs when an explicit config is supplied."""
    asyncio.run(
        _run_parity_case(
            config_path,
            expect_gpt_oss=True,
            rtol=2e-2,
            atol=2e-2,
        )
    )
