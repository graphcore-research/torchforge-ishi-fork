# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Callable

from forge.controller import ForgeActor
from forge.observability.metrics import record_metric, Reduce
from monarch.actor import endpoint

if TYPE_CHECKING:
    from forge.rl.types import Episode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_nonzero_advantage(episode: "Episode", eps: float = 1e-6) -> bool:
    advantage = _as_float(getattr(episode, "advantage", None))
    return advantage is not None and abs(advantage) > eps


@dataclass
class BufferEntry:
    data: "Episode"
    sample_count: int = 0


def age_evict(
    buffer: deque, policy_version: int, max_samples: int = None, max_age: int = None
) -> list[int]:
    """Buffer eviction policy, remove old or over-sampled entries"""
    indices = []
    for i, entry in enumerate(buffer):
        if max_age is not None and policy_version - entry.data.policy_version > max_age:
            continue
        if max_samples is not None and entry.sample_count >= max_samples:
            continue
        indices.append(i)
    return indices


def random_sample(buffer: deque, sample_size: int, policy_version: int) -> list[int]:
    """Buffer random sampling policy"""
    if sample_size > len(buffer):
        return None
    return random.sample(range(len(buffer)), k=sample_size)


@dataclass
class ReplayBuffer(ForgeActor):
    """Simple in-memory replay buffer implementation."""

    batch_size: int
    dp_size: int = 1
    max_policy_age: int | None = None
    max_buffer_size: int | None = None
    max_resample_count: int | None = 0
    seed: int | None = None
    collate: Callable = lambda batch: batch
    eviction_policy: Callable = age_evict
    sample_policy: Callable = random_sample

    @endpoint
    async def setup(self) -> None:
        self.buffer: deque = deque(maxlen=self.max_buffer_size)
        if self.seed is None:
            self.seed = random.randint(0, 2**32)
        random.seed(self.seed)

    @endpoint
    async def add(self, episode: "Episode") -> None:
        self.buffer.append(BufferEntry(episode))
        record_metric("buffer/add/count_episodes_added", 1, Reduce.SUM)

    @endpoint
    async def sample(
        self, curr_policy_version: int
    ) -> tuple[tuple[Any, ...], ...] | None:
        """Sample from the replay buffer.

        Args:
            curr_policy_version (int): The current policy version.

        Returns:
            A list of sampled episodes with shape (dp_size, bsz, ...) or None if there are not enough episodes in the buffer.
        """

        total_samples = self.dp_size * self.batch_size

        # Evict episodes
        self._evict(curr_policy_version)

        # Calculate metrics
        if len(self.buffer) > 0:
            record_metric(
                "buffer/sample/demand_to_size_ratio",
                total_samples / len(self.buffer),
                Reduce.MEAN,
            )
        if self.max_buffer_size:
            record_metric(
                "buffer/sample/avg_buffer_utilization",
                len(self.buffer) / self.max_buffer_size,
                Reduce.MEAN,
            )

        # TODO: prefetch samples in advance
        sampled_indices = self.sample_policy(
            self.buffer, total_samples, curr_policy_version
        )
        if sampled_indices is None:
            return None
        sampled_episodes = []
        for entry in self._collect(sampled_indices):
            entry.sample_count += 1
            sampled_episodes.append(entry.data)

        self._record_sample_advantage_metrics(sampled_episodes)

        # Calculate and record policy age metrics for sampled episodes
        sampled_policy_ages = [
            curr_policy_version - ep.policy_version for ep in sampled_episodes
        ]
        if sampled_policy_ages:
            record_metric(
                "buffer/sample/avg_sampled_policy_age",
                sum(sampled_policy_ages) / len(sampled_policy_ages),
                Reduce.MEAN,
            )
            record_metric(
                "buffer/sample/max_sampled_policy_age",
                max(sampled_policy_ages),
                Reduce.MAX,
            )
        # Reshape into (dp_size, bsz, ...)
        reshaped_episodes = [
            sampled_episodes[dp_idx * self.batch_size : (dp_idx + 1) * self.batch_size]
            for dp_idx in range(self.dp_size)
        ]

        # Call the underlying collate function to collate the episodes into a batch
        return self.collate(reshaped_episodes)

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict episodes from the replay buffer if they are too old based on the current policy version
        and the max policy age allowed.

        Args:
            curr_policy_version (int): The current policy version.
        """
        self._evict(curr_policy_version)

    def _evict(self, curr_policy_version):
        buffer_len_before_evict = len(self.buffer)
        max_samples = (
            None if self.max_resample_count is None else self.max_resample_count + 1
        )
        self._record_pending_eviction_reasons(curr_policy_version, max_samples)
        indices = self.eviction_policy(
            self.buffer,
            curr_policy_version,
            max_samples,
            self.max_policy_age,
        )
        kept_indices = set(indices)
        evicted_nonzero_advantage_count = sum(
            _has_nonzero_advantage(entry.data)
            for i, entry in enumerate(self.buffer)
            if i not in kept_indices
        )
        self.buffer = deque(self._collect(indices), maxlen=self.max_buffer_size)

        evicted_count = buffer_len_before_evict - len(self.buffer)
        record_metric("buffer/evict/sum_episodes_evicted", evicted_count, Reduce.SUM)
        record_metric(
            "learning/replay/evicted_nonzero_advantage_episodes",
            evicted_nonzero_advantage_count,
            Reduce.SUM,
        )

        logger.debug(
            f"maximum policy age: {self.max_policy_age}, current policy version: {curr_policy_version}, "
            f"{evicted_count} episodes expired, {len(self.buffer)} episodes left"
        )

    def _record_sample_advantage_metrics(self, sampled_episodes) -> None:
        advantages = [
            advantage
            for advantage in (
                _as_float(getattr(episode, "advantage", None))
                for episode in sampled_episodes
            )
            if advantage is not None
        ]
        if not advantages:
            return

        zero_count = sum(1 for advantage in advantages if abs(advantage) < 1e-12)
        nonzero_count = len(advantages) - zero_count
        record_metric("buffer/sample/count_zero_advantage", zero_count, Reduce.SUM)
        record_metric(
            "buffer/sample/count_nonzero_advantage", nonzero_count, Reduce.SUM
        )
        record_metric(
            "buffer/sample/frac_zero_advantage",
            zero_count / len(advantages),
            Reduce.MEAN,
        )
        record_metric(
            "buffer/sample/avg_abs_advantage",
            sum(abs(advantage) for advantage in advantages) / len(advantages),
            Reduce.MEAN,
        )
        record_metric(
            "learning/replay/sampled_episodes",
            len(sampled_episodes),
            Reduce.SUM,
        )
        record_metric(
            "learning/replay/sampled_nonzero_advantage_episodes",
            nonzero_count,
            Reduce.SUM,
        )
        record_metric(
            "learning/replay/sampled_nonzero_advantage_fraction",
            nonzero_count / max(len(advantages), 1),
            Reduce.MEAN,
        )

    def _record_pending_eviction_reasons(
        self,
        curr_policy_version: int,
        max_samples: int | None,
    ) -> None:
        age_count = 0
        resample_count = 0
        either_count = 0
        for entry in self.buffer:
            policy_version = getattr(entry.data, "policy_version", None)
            age_expired = (
                self.max_policy_age is not None
                and policy_version is not None
                and curr_policy_version - policy_version > self.max_policy_age
            )
            resample_expired = (
                max_samples is not None and entry.sample_count >= max_samples
            )
            if age_expired:
                age_count += 1
            if resample_expired:
                resample_count += 1
            if age_expired or resample_expired:
                either_count += 1

        record_metric("buffer/evict/pending_age_expired", age_count, Reduce.SUM)
        record_metric(
            "buffer/evict/pending_resample_expired", resample_count, Reduce.SUM
        )
        record_metric("buffer/evict/pending_total_expired", either_count, Reduce.SUM)

    def _collect(self, indices: list[int]):
        """Efficiently traverse deque and collect elements at each requested index"""
        n = len(self.buffer)
        if n == 0 or len(indices) == 0:
            return []

        # Normalize indices and store with their original order
        indexed = [(pos, idx % n) for pos, idx in enumerate(indices)]
        indexed.sort(key=itemgetter(1))

        result = [None] * len(indices)
        rotations = 0  # logical current index
        total_rotation = 0  # total net rotation applied

        for orig_pos, idx in indexed:
            move = idx - rotations
            self.buffer.rotate(-move)
            total_rotation += move
            rotations = idx
            result[orig_pos] = self.buffer[0]

        # Restore original deque orientation
        self.buffer.rotate(total_rotation)

        return result

    @endpoint
    async def _getitem(self, idx: int):
        return self.buffer[idx].data

    @endpoint
    async def _numel(self) -> int:
        """Number of elements (episodes) in the replay buffer."""
        return len(self.buffer)

    @endpoint
    async def clear(self) -> None:
        """Clear the replay buffer immediately - dropping all episodes."""
        self.buffer.clear()
        logger.debug("replay buffer cleared")

    @endpoint
    async def state_dict(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "rng_state": random.getstate(),
            "seed": self.seed,
        }

    @endpoint
    async def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.buffer = state_dict["buffer"]
        random.setstate(state_dict["rng_state"])
