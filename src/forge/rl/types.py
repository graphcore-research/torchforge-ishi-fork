# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
from forge.data_models.completion import Completion


@dataclass
class Episode:
    episode_id: str
    target: Any | None = None
    request: str | None = None
    response: str | None = None
    # Processed data
    completion: Completion | None = None
    ref_logprobs: torch.Tensor | None = None
    reward: float | None = None
    reward_breakdown: dict[str, float] | None = None
    advantage: float | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version

    def to_dict(self, exclude: list[str] | None = None) -> dict[str, Any]:
        """Convert episode to dict, optionally excluding specified fields."""
        result = {
            "episode_id": self.episode_id,
            "policy_version": self.policy_version,
            "prompt": self.request,
            "response": self.response,
            "target": str(self.target),
            "reward": self.reward,
            "advantage": self.advantage,
            "ref_logprobs": self.ref_logprobs,
            "completion": self.completion,
        }

        if self.reward_breakdown is not None and "reward_breakdown" not in exclude:
            result.update(self.reward_breakdown)

        if exclude:
            for key in exclude:
                result.pop(key, None)

        return result


# Represents the group (G) of episodes in GRPO
Group = list[Episode]
