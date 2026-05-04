# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from forge.data_models.completion import Completion


@dataclass
class Episode:
    episode_id: str
    pad_id: int
    request_len: int
    response_len: int
    target: Any | None = None
    request: str | None = None
    response: str | None = None
    # Processed data
    completion: Completion | None = None
    generator_logprobs: torch.Tensor | None = None  # [seq_len]
    ref_logprobs: torch.Tensor | None = None  # [seq_len]
    reward: float | None = None
    reward_breakdown: dict[str, float] | None = None
    advantage: float | None = None
    loss_priority: float | None = None
    group_id: str | None = None
    rollout_group_size: int | None = None
    slate_id: str | None = None
    slate_size: int | None = None
    slate_rank: int | None = None
    slate_reward_rank: int | None = None
    normalized_response: str | None = None
    target_action: str | None = None
    response_action: str | None = None
    validity_class: str | None = None
    advantage_sign: str | None = None
    proposal_origin: str | None = None
    loss_mask: torch.Tensor | None = None

    @property
    def policy_version(self) -> int | None:
        return self.completion.generator_version

    @property
    def request_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.prompt_ids.to(torch.long)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=self.pad_id)
        return tensor

    @property
    def request_attention_mask(self) -> torch.Tensor:
        tensor = torch.ones_like(self.completion.prompt_ids, dtype=torch.bool)
        if tensor.shape[0] < self.request_len:  # left pad
            diff = self.request_len - tensor.shape[0]
            tensor = F.pad(tensor, (diff, 0), value=False)
        return tensor

    @property
    def response_tensor(self) -> torch.Tensor:
        tensor: torch.Tensor = self.completion.token_ids.to(torch.long)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=self.pad_id)
        return tensor

    @property
    def response_attention_mask(self) -> torch.Tensor:
        tensor = torch.ones_like(self.completion.token_ids, dtype=torch.bool)
        if tensor.shape[0] < self.response_len:  # right pad
            diff = self.response_len - tensor.shape[0]
            tensor = F.pad(tensor, (0, diff), value=False)
        return tensor

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
            "loss_priority": self.loss_priority,
            "group_id": self.group_id,
            "rollout_group_size": self.rollout_group_size,
            "slate_id": self.slate_id,
            "slate_size": self.slate_size,
            "slate_rank": self.slate_rank,
            "slate_reward_rank": self.slate_reward_rank,
            "normalized_response": self.normalized_response,
            "target_action": self.target_action,
            "response_action": self.response_action,
            "validity_class": self.validity_class,
            "advantage_sign": self.advantage_sign,
            "proposal_origin": self.proposal_origin,
            "request_len": self.request_len,
            "response_len": self.response_len,
            "pad_id": self.pad_id,
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
