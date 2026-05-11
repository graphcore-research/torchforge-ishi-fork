# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare TorchTitan reference-model packed scoring against Hugging Face."""

import asyncio

import pytest
import torch
from forge.actors.reference_model import ReferenceModel
from forge.controller.provisioner import shutdown
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.rl.collate import pack_episode_tokens
from forge.rl.types import Episode
from forge.util.config import _resolve_hf_model_path
from torch.nn.utils.rnn import pad_sequence
from torchtitan.config.job_config import Checkpoint, Compile, Model, Parallelism
from transformers import AutoModelForCausalLM

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


def _response_logprobs_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
    response_len: int,
) -> torch.Tensor:
    target_ids = torch.full_like(input_ids, -100)
    target_ids[:, :-1] = input_ids[:, 1:]
    gathered_targets = target_ids.clamp(min=0)
    logprobs = logits.float().log_softmax(dim=-1)
    token_logprobs = logprobs.gather(
        dim=-1, index=gathered_targets.unsqueeze(-1)
    ).squeeze(-1)
    start = prompt_len - 1
    return token_logprobs[0, start : start + response_len]


@torch.inference_mode()
def _hf_response_logprobs(
    hf_model: AutoModelForCausalLM,
    episodes: list[Episode],
) -> torch.Tensor:
    slices = []
    for episode in episodes:
        input_ids = torch.cat(
            [episode.completion.prompt_ids, episode.completion.token_ids]
        ).unsqueeze(0)
        input_ids = input_ids.to(hf_model.device)
        logits = hf_model(input_ids=input_ids).logits
        slices.append(
            _response_logprobs_from_logits(
                logits,
                input_ids,
                prompt_len=len(episode.completion.prompt_ids),
                response_len=len(episode.completion.token_ids),
            ).cpu()
        )
    return pad_sequence(slices, batch_first=True, padding_value=0.0)


@requires_cuda
def test_reference_model_packed_logprobs_match_hf_qwen() -> None:
    """Check production packed Qwen reference logprobs against Hugging Face."""

    async def run() -> None:
        config = _qwen_reference_config()
        episodes = [
            _episode([101, 102, 103], [201, 202]),
            _episode([111, 112], [211, 212, 213]),
        ]

        titan_model = await ReferenceModel.options(
            procs=1, num_replicas=1, with_gpus=True
        ).as_service(**config)
        try:
            titan_logprobs = await titan_model.forward.route(
                *pack_episode_tokens(episodes)
            )
        finally:
            await titan_model.shutdown()
            await shutdown()

        hf_model = AutoModelForCausalLM.from_pretrained(
            config["model"].hf_assets_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda")
        hf_model.eval()
        try:
            hf_logprobs = _hf_response_logprobs(hf_model, episodes)
        finally:
            del hf_model
            torch.cuda.empty_cache()

        assert titan_logprobs.shape == hf_logprobs.shape
        torch.testing.assert_close(
            titan_logprobs.cpu(), hf_logprobs.cpu(), rtol=1e-3, atol=1e-3
        )

    asyncio.run(run())
