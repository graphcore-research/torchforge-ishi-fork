# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Callable

import torch
import torchstore as ts
from forge.actors._torchstore_utils import get_param_key
from forge.api.trainer import ParallelismConfig, TrainerConfig, TrainerStatus
from forge.controller import ForgeActor
from forge.data.utils import batch_to_device
from forge.observability.metrics import record_metric, Reduce
from forge.observability.perf_tracker import Tracer
from forge.rl.loss import compute_logprobs, create_shifted_targets, masked_mean
from forge.types import TrainBatch
from monarch.actor import endpoint
from torch import Tensor
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torchtitan.config.job_config import (
    ActivationCheckpoint,
    Checkpoint,
    Comm,
    Compile,
    Job,
    LRScheduler,
    MemoryEstimation,
    Model,
    Optimizer,
    Parallelism,
    Quantize,
    Training,
)
from torchtitan.experiments.forge.engine import ForgeEngine
from torchtitan.experiments.forge.job_config import ForgeJobConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class TitanTrainer(ForgeActor):
    """A generic trainer actor implementation built on top of TorchTitan.

    Built on top of TorchTitan's training engine, this actor provides a complete training
    loop for reinforcement learning. It performs forward and backward passes with gradient
    computation, optimization steps, and checkpoint management. Unlike the ReferenceModel
    actor which only runs forward passes, RLTrainer actively updates the policy model
    parameters through gradient descent.

    The trainer supports the same distributed training strategies that TorchTitan does,
    including but not limited to, tensor parallelism, data parallelism, and FSDP
    (Fully Sharded Data Parallel). It is typically used in conjunction with ReferenceModel
    for policy optimization algorithms like GRPO (Group Relative Policy Optimization),
    where it optimizes the policy against a loss that includes KL divergence penalties
    from the reference model.

    The trainer handles:
    - Forward and backward propagation with automatic mixed precision (AMP)
    - Optimizer steps with learning rate scheduling
    """

    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    optimizer: Optimizer = field(default_factory=Optimizer)
    lr_scheduler: LRScheduler = field(default_factory=LRScheduler)
    training: Training = field(default_factory=Training)
    parallelism: Parallelism = field(default_factory=Parallelism)
    checkpoint: Checkpoint = field(default_factory=Checkpoint)
    activation_checkpoint: ActivationCheckpoint = field(
        default_factory=ActivationCheckpoint
    )
    compile: Compile = field(default_factory=Compile)
    quantize: Quantize = field(default_factory=Quantize)
    comm: Comm = field(default_factory=Comm)
    memory_estimation: MemoryEstimation = field(default_factory=MemoryEstimation)
    # Non JobConfig-related fields
    loss: Callable = lambda logits, **targets: logits
    state_dict_key: str = "model_state_dict"
    recompute_generator_logprobs: bool = False

    def __post_init__(self):
        super().__init__()

        for f in fields(self):
            attr = getattr(self, f.name)
            if isinstance(attr, Mapping):
                setattr(self, f.name, f.type(**attr))
            elif not isinstance(attr, f.type):
                raise TypeError(
                    f"{f.name} should be a {f.type} type or a dict like object"
                )

        self.step = 1  # fragile contract.
        self.num_training_steps = self.training.steps
        self.gradient_accumulation_steps = 1
        self._accumulated_microbatches = 0
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Compiling loss")
        self.loss = torch.compile(self.loss)

    @endpoint
    async def setup(self):
        # TODO: update ForgeEngine to not use ForgeJobConfig
        engine_config = {f.name: getattr(self, f.name) for f in fields(self)}
        for key in {
            "loss",
            "recompute_generator_logprobs",
            "state_dict_key",
        }:
            engine_config.pop(key)  # Not part of job config
        self.engine = ForgeEngine(ForgeJobConfig(**engine_config))
        self.engine.checkpointer.load(step=self.step)
        self.engine.optimizers.zero_grad()

    def _replace_generator_logprobs_with_trainer_context(
        self,
        logits: torch.Tensor,
        loss_inputs: dict,
    ) -> None:
        """Use trainer-context old logprobs as the PPO/GRPO denominator.

        vLLM can report sampled-token logprobs in a subtly different probability
        space from the trainer model. TRL avoids using those sampled logprobs as
        the canonical old-policy denominator by recomputing old logprobs with the
        trainer model under the same tokenization, masks, and positions used for
        optimization. This helper mirrors that behavior while logging the gap to
        the sampled logprobs as a canary.
        """
        target_ids = loss_inputs["target_ids"]
        sampled_logprobs = loss_inputs.get("generator_logprobs")
        trainer_logprobs, _ = compute_logprobs(logits.detach(), target_ids)

        if sampled_logprobs is not None:
            loss_mask = loss_inputs.get("loss_mask")
            if loss_mask is None:
                loss_mask = torch.ones_like(trainer_logprobs)
            with torch.no_grad():
                delta = trainer_logprobs - sampled_logprobs
                record_metric(
                    "old_logprobs/trainer_minus_sampler/mean",
                    masked_mean(delta, loss_mask),
                    Reduce.MEAN,
                )
                record_metric(
                    "old_logprobs/trainer_minus_sampler/abs_mean",
                    masked_mean(delta.abs(), loss_mask),
                    Reduce.MEAN,
                )
                record_metric(
                    "old_logprobs/trainer_over_sampler_ratio/mean",
                    masked_mean(torch.exp(delta.clamp(min=-50.0, max=50.0)), loss_mask),
                    Reduce.MEAN,
                )

        loss_inputs["generator_logprobs"] = trainer_logprobs.detach()

    def _record_batch_signal_metrics(self, batch: TrainBatch) -> None:
        loss_mask = batch.loss_inputs.get("loss_mask")
        advantages = batch.loss_inputs.get("advantages")
        if advantages is None:
            return

        with torch.no_grad():
            if loss_mask is None:
                active_mask = torch.ones_like(advantages, dtype=torch.bool)
            else:
                active_mask = loss_mask.to(dtype=torch.bool)
            nonzero_advantage_mask = advantages.abs() > 1e-6
            active_token_count = int(active_mask.sum().item())
            nonzero_advantage_token_count = int(
                (active_mask & nonzero_advantage_mask).sum().item()
            )

        record_metric(
            "learning/trainer/active_tokens",
            active_token_count,
            Reduce.SUM,
        )
        record_metric(
            "learning/trainer/nonzero_advantage_tokens",
            nonzero_advantage_token_count,
            Reduce.SUM,
        )
        record_metric(
            "learning/trainer/nonzero_advantage_token_fraction",
            nonzero_advantage_token_count / max(active_token_count, 1),
            Reduce.MEAN,
        )
        record_metric(
            "learning/trainer/batch_has_nonzero_advantage",
            int(nonzero_advantage_token_count > 0),
            Reduce.SUM,
        )

    def _record_gradient_signal_metrics(self, max_parameters: int = 16) -> None:
        with torch.no_grad():
            squared_norm = torch.zeros((), device=self.engine.device)
            parameter_count = 0
            for model_part in self.engine.model_parts:
                for parameter in model_part.parameters():
                    if not parameter.requires_grad or parameter.grad is None:
                        continue
                    grad = parameter.grad.detach()
                    if getattr(grad, "is_sparse", False):
                        grad = grad.coalesce().values()
                    squared_norm = squared_norm + grad.float().pow(2).sum()
                    parameter_count += 1
                    if parameter_count >= max_parameters:
                        break
                if parameter_count >= max_parameters:
                    break
            grad_norm = float(torch.sqrt(squared_norm).item())

        record_metric(
            "learning/trainer/selected_grad_norm",
            grad_norm,
            Reduce.MEAN,
        )
        record_metric(
            "learning/trainer/nonzero_gradient_step",
            int(grad_norm > 0.0),
            Reduce.SUM,
        )
        record_metric(
            "learning/trainer/selected_grad_parameter_count",
            parameter_count,
            Reduce.MEAN,
        )

    def forward_backward(self, batch: TrainBatch) -> Tensor:
        model_parts = self.engine.model_parts
        parallel_dims = self.engine.parallel_dims
        optional_context_parallel_ctx = None

        # Create shifted target_ids for next-token prediction
        # target_ids[i] = input_ids[i+1], with loss_mask applied
        batch.loss_inputs["target_ids"] = create_shifted_targets(
            batch.model_inputs["tokens"], batch.loss_inputs.get("loss_mask")
        )

        if parallel_dims.pp_enabled:
            raise NotImplementedError("PP not implemented yet")
        else:
            with self.engine.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.engine.maybe_enable_amp:
                    logits = model_parts[0](**batch.model_inputs)
                    if self.recompute_generator_logprobs:
                        self._replace_generator_logprobs_with_trainer_context(
                            logits,
                            batch.loss_inputs,
                        )
                    loss_output = self.loss(logits, **batch.loss_inputs)
                    loss = loss_output.loss

                # Record metrics from loss output
                for metric in loss_output.metrics:
                    value = (
                        metric.value.item()
                        if isinstance(metric.value, torch.Tensor)
                        else metric.value
                    )
                    record_metric(metric.key, value, metric.reduction, metric.timestamp)

                # Free to before bwd to avoid peaking memory
                del logits, loss_output.metrics
                loss.backward()
                self._record_gradient_signal_metrics()
        self._accumulated_microbatches += 1
        return loss

    @endpoint
    async def train_step(self, batches: list[TrainBatch]) -> float:
        t = Tracer("rl_trainer_perf/step", timer="gpu", track_memory=True)
        t.start()

        self.engine.gc_handler.run(self.step)
        batch = batches[self.engine.dp_rank]
        batch_to_device(batch.model_inputs, self.engine.device)
        batch_to_device(batch.loss_inputs, self.engine.device)
        self._record_batch_signal_metrics(batch)

        loss = self.forward_backward(batch)
        torch.distributed.all_reduce(loss)

        t.step("forward_backward")

        current_lr = self.engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        record_metric("rl_trainer/learning_rate", current_lr, Reduce.MIN)

        self.engine.optimizers.step()
        self.engine.optimizers.zero_grad()
        self.engine.lr_schedulers.step()
        self._accumulated_microbatches = 0
        self.step += 1
        t.step("optimizer_step")

        # TODO: delete item() to avoid cpu-gpu sync
        loss = loss.detach().item()
        record_metric("rl_trainer/loss", loss, Reduce.MEAN)

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=self.step == self.num_training_steps,
        )
        t.step("save_checkpoint")
        t.stop()
        return loss

    @endpoint
    async def get_config(self) -> TrainerConfig:
        """Get static trainer and model configuration.

        Returns configuration information that doesn't change during training.
        For runtime state like current step, use get_status() instead.

        Returns:
            TrainerConfig containing model name, model_config, and parallelism settings

        """
        parallel_dims = self.engine.parallel_dims
        parallelism = ParallelismConfig(
            dp_degree=parallel_dims.dp_shard * parallel_dims.dp_replicate,
            tp_degree=parallel_dims.tp,
            pp_degree=parallel_dims.pp,
            cp_degree=parallel_dims.cp,
            ep_degree=parallel_dims.ep,
            world_size=parallel_dims.world_size,
            dp_rank=self.engine.dp_rank,
            tp_rank=parallel_dims.tp_coord,
            device=str(self.engine.device),
        )
        return TrainerConfig(
            model_name=self.model.name,
            model_config=self.model.model_dump(),
            parallelism=parallelism,
        )

    @endpoint
    async def get_status(self) -> TrainerStatus:
        """Get current runtime status of the trainer.

        Returns dynamic information about the trainer's current state that changes
        during training.

        Returns:
            TrainerStatus containing current step and accumulated batch count

        """
        return TrainerStatus(
            step=self.step,
            accumulated_microbatches=self._accumulated_microbatches,
        )

    @endpoint
    async def clear_gradients(self) -> None:
        """Clear accumulated gradients without applying them.

        Use this when you need to discard accumulated gradients without performing
        an optimizer step. Common scenarios:
        - Exception during gradient accumulation
        - Skipping a training step due to some condition
        - Recovering from OOM or other errors

        This is equivalent to calling optimizer.zero_grad() and resetting internal
        accumulation counters.
        """
        self.engine.optimizers.zero_grad()
        self._accumulated_microbatches = 0

    @endpoint
    async def save(
        self,
        name: str | None = None,
        path: str | None = None,
        weights_only: bool = False,
    ) -> str:
        """Save trainer state or weights to persistent storage.

        By default, saves complete training state (model weights, optimizer state,
        learning rate scheduler state, and step counter).

        Args:
            name: Not supported. TitanTrainer uses step-based checkpoint naming.
            path: Not supported. TitanTrainer uses checkpoint.folder from config.
            weights_only: Not supported. TitanTrainer always saves full training state.

        Returns:
            Full path where checkpoint was saved
        """
        if name is not None:
            raise NotImplementedError(
                "TitanTrainer uses step-based checkpoint naming; custom names are not supported"
            )
        if path is not None:
            raise NotImplementedError(
                "TitanTrainer uses the checkpoint.folder from config; custom paths are not supported"
            )
        if weights_only:
            raise NotImplementedError(
                "weights_only is not supported; TitanTrainer always saves full training state"
            )

        self.engine.checkpointer.save(
            curr_step=self.step,
            last_step=False,
        )
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def load(self, path: str | None = None) -> str:
        """Load a previously saved checkpoint.

        Restores training state from a checkpoint.

        Args:
            path: Not supported. TitanTrainer uses checkpoint.folder from config.

        Returns:
            Path that was loaded
        """
        if path is not None:
            raise NotImplementedError(
                "TitanTrainer uses the checkpoint.folder from config; custom paths are not supported"
            )

        self.engine.checkpointer.load(step=self.step)
        return f"{self.checkpoint.folder}/step-{self.step}"

    @endpoint
    async def push_weights(self, policy_version: int) -> None:
        """Push weights to torchstore in HF format."""
        logger.info(f"Pushing weights for policy version {policy_version}")

        start_time = time.perf_counter()
        if "model" not in self.engine.checkpointer.states:
            raise RuntimeError("Model state not found in checkpointer state")

        sd = self.engine.checkpointer.states["model"].state_dict()
        flattened_state_dict, _ = flatten_state_dict(sd)
        if self.engine.checkpointer.sd_adapter is None:
            raise RuntimeError(
                "Trying to save checkpoint in HF safetensors format, but sd_adapter is not provided."
            )
        hf_state_dict = self.engine.checkpointer.sd_adapter.to_hf(flattened_state_dict)
        for name, param in hf_state_dict.items():
            key = get_param_key(policy_version, name)
            await ts.put(key, param)
        end_time = time.perf_counter()
        logger.info("Completed weights push in %.2f seconds", end_time - start_time)

    @endpoint
    async def cleanup(self) -> None:
        if self.engine.checkpointer:
            self.engine.checkpointer.close()
