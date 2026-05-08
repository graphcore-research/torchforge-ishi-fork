# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from forge.data_models.completion import Completion
from forge.data_models.prompt import Prompt
from forge.rl.collate import collate, pack_episode_tokens
from forge.rl.types import Episode


def _completion(
    prompt_ids: list[int],
    token_ids: list[int],
    logprobs: list[float],
    *,
    generator_version: int = 0,
) -> Completion:
    return Completion(
        prompt=Prompt.from_prompt("prompt"),
        text="response",
        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        logprobs=torch.tensor(logprobs, dtype=torch.float32),
        generator_version=generator_version,
    )


def _episode(
    prompt_ids: list[int],
    token_ids: list[int],
    logprobs: list[float],
    *,
    advantage: float,
    ref_logprobs: list[float] | None = None,
) -> Episode:
    episode = Episode(
        episode_id=f"episode-{len(prompt_ids)}-{len(token_ids)}",
        request="prompt",
        response="response",
        completion=_completion(prompt_ids, token_ids, logprobs),
        advantage=advantage,
    )
    if ref_logprobs is not None:
        episode.ref_logprobs = torch.tensor(ref_logprobs, dtype=torch.float32)
    return episode


def test_packed_collate_variable_lengths() -> None:
    episodes = [
        _episode([11, 12, 13], [21, 22], [-0.1, -0.2], advantage=1.5),
        _episode([31], [41, 42, 43], [-0.3, -0.4, -0.5], advantage=-0.5),
    ]

    tokens, prompt_lens, response_lens, seq_lens = pack_episode_tokens(episodes)

    assert tokens.shape == (1, 9)
    assert tokens.tolist() == [[11, 12, 13, 21, 22, 31, 41, 42, 43]]
    assert prompt_lens == [3, 1]
    assert response_lens == [2, 3]
    assert seq_lens == [5, 4]


def test_collate_builds_response_only_loss_inputs() -> None:
    episodes = [
        _episode(
            [11, 12, 13],
            [21, 22],
            [-0.1, -0.2],
            advantage=1.5,
            ref_logprobs=[-1.1, -1.2],
        ),
        _episode(
            [31],
            [41, 42, 43],
            [-0.3, -0.4, -0.5],
            advantage=-0.5,
            ref_logprobs=[-1.3, -1.4, -1.5],
        ),
    ]

    [batch] = collate([episodes])

    assert batch.model_inputs["tokens"].tolist() == [
        [11, 12, 13, 21, 22, 31, 41, 42, 43]
    ]
    assert batch.meta == {
        "prompt_lens": [3, 1],
        "response_lens": [2, 3],
        "seq_lens": [5, 4],
    }

    assert batch.loss_inputs["target_ids"].tolist() == [[21, 22, 0], [41, 42, 43]]
    assert torch.allclose(
        batch.loss_inputs["generator_logprobs"],
        torch.tensor([[-0.1, -0.2, 0.0], [-0.3, -0.4, -0.5]]),
    )
    assert batch.loss_inputs["loss_mask"].tolist() == [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]
    assert batch.loss_inputs["advantages"].tolist() == [
        [1.5, 1.5, 1.5],
        [-0.5, -0.5, -0.5],
    ]
    assert torch.allclose(
        batch.loss_inputs["ref_logprobs"],
        torch.tensor([[-1.1, -1.2, 0.0], [-1.3, -1.4, -1.5]]),
    )
