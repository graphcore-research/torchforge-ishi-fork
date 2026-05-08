# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging

import monarch
import pytest
import pytest_asyncio
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
from monarch.actor import endpoint
from omegaconf import DictConfig, OmegaConf

monarch.actor.unhandled_fault_hook = lambda failure: None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


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

        def parity_loss(
            logits,
            target_ids,
            generator_logprobs,
            loss_mask,
            **_,
        ) -> LossOutput:
            nonlocal max_error, mean_error
            logprobs, _ = compute_logprobs(logits, target_ids)
            diff = (logprobs - generator_logprobs).abs() * loss_mask
            max_error = diff.max()
            mean_error = diff.sum() / loss_mask.sum().clamp(min=1.0)
            torch.testing.assert_close(
                logprobs * loss_mask,
                generator_logprobs * loss_mask,
                rtol=rtol,
                atol=atol,
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


@pytest_asyncio.fixture
async def generator_and_trainer(config_path):
    if not config_path:
        pytest.skip("Logprob parity requires --config")

    try:
        from forge.actors.generator import Generator
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Generator dependencies are unavailable: {exc}")

    cfg = _load_config(config_path)
    model_card = cfg.model
    logger.info("Downloading model checkpoint from HuggingFace Hub")
    cached_dir = snapshot_download(repo_id=model_card)
    logger.info("Finished downloading model checkpoint from HuggingFace Hub")

    cfg.generator.sampling_params.logprobs = 1
    cfg.services.generator.num_replicas = 1
    cfg.trainer.checkpoint = {
        "enable": True,
        "folder": "/tmp/logprob_parity_checkpoints",
        "initial_load_path": cached_dir,
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

    try:
        yield cfg, generator, trainer
    finally:
        await trainer.cleanup.call()
        await generator.shutdown()
        await TitanTrainer.shutdown(trainer)
        await ts.shutdown()


async def _assert_trainer_generator_parity(
    generator,
    trainer,
    *,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    completions = await generator.generate.route("What is 2 + 2?")
    episodes = [_episode_from_completion(completion) for completion in completions]
    batches = collate([episodes])
    result = await trainer.assert_generator_logprob_parity.call(batches, rtol, atol)
    assert result["max_error"] <= atol
    return result


@pytest.mark.asyncio
@requires_cuda
async def test_trainer_generator_logprob_parity_qwen_tp_or_config(
    generator_and_trainer,
) -> None:
    cfg, generator, trainer = generator_and_trainer
    if cfg.trainer.model.name == "gpt_oss":
        pytest.skip("Use the GPT-OSS-specific parity test for GPT-OSS configs")

    await _assert_trainer_generator_parity(
        generator,
        trainer,
        rtol=1e-2,
        atol=1e-2,
    )


@pytest.mark.asyncio
@requires_cuda
async def test_trainer_generator_logprob_parity_gpt_oss_config(
    generator_and_trainer,
) -> None:
    cfg, generator, trainer = generator_and_trainer
    if cfg.trainer.model.name != "gpt_oss":
        pytest.skip("GPT-OSS parity only runs with a GPT-OSS --config")

    await _assert_trainer_generator_parity(
        generator,
        trainer,
        rtol=2e-2,
        atol=2e-2,
    )
