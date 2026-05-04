# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import math
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


def _metric_fragment(value: Any) -> str:
    fragment = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return fragment.strip("_") or "empty"


def _metadata_value(episode: "Episode", attr: str, default: str = "unknown") -> str:
    value = getattr(episode, attr, None)
    if value is None:
        return default
    text = str(value).strip()
    return text if text else "empty"


def _metadata_parts(episode: "Episode", attr: str) -> tuple[str, ...]:
    value = _metadata_value(episode, attr, default="")
    if not value or value == "empty":
        return ()
    return tuple(part for part in value.split() if part)


def _hamming_distance(left: tuple[str, ...], right: tuple[str, ...]) -> int | None:
    if not left or not right:
        return None
    shared = min(len(left), len(right))
    distance = sum(1 for idx in range(shared) if left[idx] != right[idx])
    return distance + abs(len(left) - len(right))


def _position_cells(parts: tuple[str, ...]) -> set[tuple[int, str]]:
    return {(position, value) for position, value in enumerate(parts)}


def _format_position_cells(cells: set[tuple[int, str]]) -> list[str]:
    return [f"{position}:{value}" for position, value in sorted(cells)]


def _advantage_sign(episode: "Episode", eps: float = 1e-6) -> str:
    advantage = _as_float(getattr(episode, "advantage", None))
    if advantage is None or abs(advantage) <= eps:
        return "zero"
    return "positive" if advantage > 0 else "negative"


def _sample_ranked_group_subset(
    buffer: deque,
    indices: list[int],
    limit: int,
) -> list[int]:
    """Pick a compact contrastive subset from one oversized prompt slate."""
    if limit <= 0:
        return []
    if len(indices) <= limit:
        return list(indices)

    selected: list[int] = []

    def add(index: int | None) -> None:
        if index is not None and index not in selected and len(selected) < limit:
            selected.append(index)

    scored: list[tuple[int, float, float | None]] = []
    for index in indices:
        episode = buffer[index].data
        reward = _as_float(getattr(episode, "reward", None))
        loss_priority = _as_float(getattr(episode, "loss_priority", None))
        scored.append((index, reward if reward is not None else float("-inf"), loss_priority))

    if scored:
        max_reward = max(reward for _, reward, _ in scored)
        add(max(scored, key=lambda item: item[1])[0])
        add(min(scored, key=lambda item: item[1])[0])

        high_policy_wrong = [
            item
            for item in scored
            if item[1] < max_reward and item[2] is not None
        ]
        if high_policy_wrong:
            add(min(high_policy_wrong, key=lambda item: item[2])[0])

    remaining = [index for index in indices if index not in selected]
    random.shuffle(remaining)
    selected.extend(remaining[: limit - len(selected)])
    return selected


def _sample_contrast_coverage_group_subset(
    buffer: deque,
    indices: list[int],
    limit: int,
    *,
    metric_prefix: str,
) -> list[int]:
    """Pick best, hard-wrong, nearest-wrong, and coverage examples from a slate."""
    if limit <= 0:
        return []
    if len(indices) <= limit:
        return list(indices)

    selected: list[int] = []
    selected_slots: dict[int, str] = {}
    slot_filled = {
        "best_reward": 0,
        "high_policy_wrong": 0,
        "nearest_wrong": 0,
        "coverage": 0,
    }

    def add(index: int | None, slot: str | None = None) -> None:
        if index is not None and index not in selected and len(selected) < limit:
            selected.append(index)
            if slot is not None:
                selected_slots[index] = slot
            if slot is not None:
                slot_filled[slot] = 1

    items: list[dict[str, Any]] = []
    items_by_index: dict[int, dict[str, Any]] = {}
    for index in indices:
        episode = buffer[index].data
        reward = _as_float(getattr(episode, "reward", None))
        loss_priority = _as_float(getattr(episode, "loss_priority", None))
        response_parts = _metadata_parts(episode, "response_action")
        if not response_parts:
            response_parts = _metadata_parts(episode, "normalized_response")
        items.append(
            {
                "index": index,
                "reward": reward if reward is not None else float("-inf"),
                "loss_priority": loss_priority,
                "response_parts": response_parts,
                "position_cells": _position_cells(response_parts),
                "proposal_origin": _metadata_value(episode, "proposal_origin"),
                "slate_id": _metadata_value(episode, "slate_id"),
                "group_id": _metadata_value(episode, "group_id"),
                "target_action": _metadata_value(episode, "target_action"),
                "response_action": _metadata_value(episode, "response_action"),
                "normalized_response": _metadata_value(
                    episode,
                    "normalized_response",
                    default=_metadata_value(episode, "response"),
                ),
                "slate_rank": _as_float(getattr(episode, "slate_rank", None)),
                "reward_rank": _as_float(
                    getattr(episode, "slate_reward_rank", None)
                ),
            }
        )
        items_by_index[index] = items[-1]

    if not items:
        return []

    best = max(
        items,
        key=lambda item: (
            item["reward"],
            -float(item["loss_priority"])
            if item["loss_priority"] is not None
            else float("-inf"),
        ),
    )
    best_reward = float(best["reward"])
    best_parts = tuple(best["response_parts"])
    add(int(best["index"]), "best_reward")

    wrong_items = [item for item in items if float(item["reward"]) < best_reward]
    high_policy_wrong = [
        item for item in wrong_items if item["loss_priority"] is not None
    ]
    if high_policy_wrong:
        add(
            int(min(high_policy_wrong, key=lambda item: item["loss_priority"])["index"]),
            "high_policy_wrong",
        )

    nearest_candidates: list[tuple[int, float, float, dict[str, Any]]] = []
    for item in wrong_items:
        distance = _hamming_distance(best_parts, tuple(item["response_parts"]))
        if distance is None:
            continue
        loss_priority = (
            float(item["loss_priority"])
            if item["loss_priority"] is not None
            else float("inf")
        )
        nearest_candidates.append(
            (distance, -float(item["reward"]), loss_priority, item)
        )
    if nearest_candidates:
        add(int(min(nearest_candidates, key=itemgetter(0, 1, 2))[3]["index"]), "nearest_wrong")

    selected_cells: set[tuple[int, str]] = set()
    for index in selected:
        episode = buffer[index].data
        selected_cells.update(_position_cells(_metadata_parts(episode, "response_action")))

    remaining_items = [item for item in items if int(item["index"]) not in selected]
    if remaining_items:
        coverage_item = max(
            remaining_items,
            key=lambda item: (
                len(set(item["position_cells"]) - selected_cells),
                len(set(item["position_cells"])),
                item["reward"],
                -float(item["loss_priority"])
                if item["loss_priority"] is not None
                else float("-inf"),
            ),
        )
        add(int(coverage_item["index"]), "coverage")

    remaining = [index for index in indices if index not in selected]
    random.shuffle(remaining)
    fallback_count = max(0, min(len(remaining), limit - len(selected)))
    for index in remaining[: limit - len(selected)]:
        selected.append(index)
        selected_slots[index] = "fallback"

    selected_items = [items_by_index[index] for index in selected]
    selected_rewards = [float(item["reward"]) for item in selected_items]
    selected_cells = set()
    possible_cells = set()
    proposal_origin_counts: dict[str, int] = {}
    hamming_distances: list[int] = []
    slate_ranks: list[float] = []
    reward_ranks: list[float] = []
    for item in items:
        possible_cells.update(set(item["position_cells"]))
    for item in selected_items:
        selected_cells.update(set(item["position_cells"]))
        origin = str(item["proposal_origin"])
        proposal_origin_counts[origin] = proposal_origin_counts.get(origin, 0) + 1
        distance = _hamming_distance(best_parts, tuple(item["response_parts"]))
        if distance is not None and int(item["index"]) != int(best["index"]):
            hamming_distances.append(distance)
        if item["slate_rank"] is not None:
            slate_ranks.append(float(item["slate_rank"]))
        if item["reward_rank"] is not None:
            reward_ranks.append(float(item["reward_rank"]))
        selector_slot = selected_slots.get(int(item["index"]), "unknown")
        reward_value = float(item["reward"])
        reward_delta = (
            best_reward - reward_value
            if math.isfinite(best_reward) and math.isfinite(reward_value)
            else None
        )
        record_metric(
            f"{metric_prefix}/selector_trace_table",
            {
                "selector_slot": selector_slot,
                "fallback_reason": "capacity_fill"
                if selector_slot == "fallback"
                else "",
                "reward": item["reward"],
                "reward_delta_from_best": reward_delta,
                "loss_priority": item["loss_priority"],
                "slate_id": item["slate_id"],
                "group_id": item["group_id"],
                "target_action": item["target_action"],
                "response_action": item["response_action"],
                "normalized_response": item["normalized_response"],
                "proposal_origin": item["proposal_origin"],
                "slate_rank": item["slate_rank"],
                "reward_rank": item["reward_rank"],
                "hamming_to_best": distance,
                "position_cells": _format_position_cells(set(item["position_cells"])),
                "position_cell_count": len(set(item["position_cells"])),
            },
            Reduce.SAMPLE,
        )

    record_metric(f"{metric_prefix}/subset_strategy/contrast_coverage", 1, Reduce.MEAN)
    for slot, filled in slot_filled.items():
        record_metric(
            f"{metric_prefix}/selector_slot/{slot}_filled",
            filled,
            Reduce.MEAN,
        )
    record_metric(
        f"{metric_prefix}/selector_slot/fallback_count",
        fallback_count,
        Reduce.MEAN,
    )
    if selected_rewards:
        record_metric(
            f"{metric_prefix}/selector/selected_reward_range",
            max(selected_rewards) - min(selected_rewards),
            Reduce.MEAN,
        )
    if hamming_distances:
        record_metric(
            f"{metric_prefix}/selector/mean_hamming_distance_to_best",
            sum(hamming_distances) / len(hamming_distances),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/selector/min_hamming_distance_to_best",
            min(hamming_distances),
            Reduce.MEAN,
        )
    record_metric(
        f"{metric_prefix}/selector/position_cell_count",
        len(selected_cells),
        Reduce.MEAN,
    )
    record_metric(
        f"{metric_prefix}/selector/position_cell_coverage_fraction",
        len(selected_cells) / max(len(possible_cells), 1),
        Reduce.MEAN,
    )
    record_metric(
        f"{metric_prefix}/selector/proposal_origin_count",
        len(proposal_origin_counts),
        Reduce.MEAN,
    )
    for origin, count in proposal_origin_counts.items():
        fragment = _metric_fragment(origin)
        record_metric(
            f"{metric_prefix}/selector/proposal_origin/{fragment}/fraction",
            count / max(len(selected_items), 1),
            Reduce.MEAN,
        )
    if slate_ranks:
        record_metric(
            f"{metric_prefix}/selector/slate_rank_min",
            min(slate_ranks),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/selector/slate_rank_max",
            max(slate_ranks),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/selector/slate_rank_mean",
            sum(slate_ranks) / len(slate_ranks),
            Reduce.MEAN,
        )
    if reward_ranks:
        record_metric(
            f"{metric_prefix}/selector/reward_rank_min",
            min(reward_ranks),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/selector/reward_rank_max",
            max(reward_ranks),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/selector/reward_rank_mean",
            sum(reward_ranks) / len(reward_ranks),
            Reduce.MEAN,
        )
    return selected


def _sample_group_subset(
    buffer: deque,
    indices: list[int],
    limit: int,
    *,
    strategy: str,
    metric_prefix: str,
) -> list[int]:
    strategy = strategy.lower()
    if strategy in {"ranked", "default", "best_worst_high_policy_random"}:
        record_metric(f"{metric_prefix}/subset_strategy/ranked", 1, Reduce.MEAN)
        return _sample_ranked_group_subset(buffer, indices, limit)
    if strategy == "contrast_coverage":
        return _sample_contrast_coverage_group_subset(
            buffer,
            indices,
            limit,
            metric_prefix=metric_prefix,
        )
    raise ValueError(
        "slate/group subset strategy must be one of 'ranked' or "
        f"'contrast_coverage', got {strategy!r}."
    )


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
    sample_nonzero_advantage_only: bool = False
    advantage_epsilon: float = 1e-6
    group_preserving_sampling: bool = False
    group_preserving_group_attr: str = "group_id"
    group_preserving_group_size_attr: str = "rollout_group_size"
    group_preserving_require_complete_groups: bool = True
    group_preserving_allow_partial_group: bool = False
    group_preserving_allow_ungrouped: bool = False
    group_preserving_fallback_to_episode_sampling: bool = False
    group_preserving_priority: str = "random"
    group_preserving_priority_epsilon: float = 1e-3
    group_preserving_subset_strategy: str = "ranked"
    slate_preserving_sampling: bool = False
    slate_preserving_group_attr: str = "slate_id"
    slate_preserving_group_size_attr: str = "slate_size"
    slate_preserving_require_complete_groups: bool = False
    slate_preserving_allow_partial_group: bool = True
    slate_preserving_allow_ungrouped: bool = False
    slate_preserving_fallback_to_episode_sampling: bool = False
    slate_subset_strategy: str = "ranked"
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
        record_metric(
            "learning/replay/added_nonzero_advantage_episodes",
            int(_has_nonzero_advantage(episode, self.advantage_epsilon)),
            Reduce.SUM,
        )
        origin = _metadata_value(episode, "proposal_origin")
        record_metric(
            f"learning/replay/add/proposal_origin/{_metric_fragment(origin)}",
            1,
            Reduce.SUM,
        )

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
        self._record_contrast_metrics(
            [entry.data for entry in self.buffer],
            prefix="learning/replay/contrast/buffer",
            record_cells=False,
        )

        # TODO: prefetch samples in advance
        sampled_indices = None
        if self.slate_preserving_sampling:
            sampled_indices = self._sample_group_preserving_indices(
                total_samples,
                group_attr=self.slate_preserving_group_attr,
                group_size_attr=self.slate_preserving_group_size_attr,
                require_complete_groups=self.slate_preserving_require_complete_groups,
                allow_partial_group=self.slate_preserving_allow_partial_group,
                allow_ungrouped=self.slate_preserving_allow_ungrouped,
                metric_prefix="learning/replay/slate_preserving",
                subset_strategy=self.slate_subset_strategy,
            )
            if sampled_indices is None:
                record_metric(
                    "learning/replay/slate_preserving/sample_unavailable",
                    1,
                    Reduce.SUM,
                )
                if not self.slate_preserving_fallback_to_episode_sampling:
                    return None
        elif self.group_preserving_sampling:
            sampled_indices = self._sample_group_preserving_indices(
                total_samples,
                subset_strategy=self.group_preserving_subset_strategy,
            )
            if sampled_indices is None:
                record_metric(
                    "learning/replay/group_preserving/sample_unavailable",
                    1,
                    Reduce.SUM,
                )
                if not self.group_preserving_fallback_to_episode_sampling:
                    return None

        if sampled_indices is None:
            sampled_indices = self._sample_episode_indices(
                total_samples,
                curr_policy_version,
            )
        if sampled_indices is None:
            return None
        sampled_episodes = []
        for entry in self._collect(sampled_indices):
            entry.sample_count += 1
            sampled_episodes.append(entry.data)

        self._record_sample_advantage_metrics(sampled_episodes)
        self._record_contrast_metrics(
            sampled_episodes,
            prefix="learning/replay/contrast/selected",
            record_cells=True,
        )

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

    def _sample_episode_indices(
        self,
        total_samples: int,
        curr_policy_version: int,
    ) -> list[int] | None:
        if self.sample_nonzero_advantage_only:
            eligible_indices = [
                index
                for index, entry in enumerate(self.buffer)
                if _has_nonzero_advantage(entry.data, self.advantage_epsilon)
            ]
            record_metric(
                "learning/replay/nonzero_advantage_eligible_episodes",
                len(eligible_indices),
                Reduce.MEAN,
            )
            record_metric(
                "learning/replay/nonzero_advantage_eligible_fraction",
                len(eligible_indices) / max(len(self.buffer), 1),
                Reduce.MEAN,
            )
            if total_samples > len(eligible_indices):
                return None
            eligible_buffer = deque(self.buffer[index] for index in eligible_indices)
            eligible_sampled_indices = self.sample_policy(
                eligible_buffer, total_samples, curr_policy_version
            )
            if eligible_sampled_indices is None:
                return None
            return [eligible_indices[index] for index in eligible_sampled_indices]

        return self.sample_policy(self.buffer, total_samples, curr_policy_version)

    def _sample_group_preserving_indices(
        self,
        total_samples: int,
        *,
        group_attr: str | None = None,
        group_size_attr: str | None = None,
        require_complete_groups: bool | None = None,
        allow_partial_group: bool | None = None,
        allow_ungrouped: bool | None = None,
        metric_prefix: str = "learning/replay/group_preserving",
        subset_strategy: str | None = None,
    ) -> list[int] | None:
        """Sample intact rollout groups instead of detached individual episodes."""
        group_attr = group_attr or self.group_preserving_group_attr
        group_size_attr = group_size_attr or self.group_preserving_group_size_attr
        require_complete_groups = (
            self.group_preserving_require_complete_groups
            if require_complete_groups is None
            else require_complete_groups
        )
        allow_partial_group = (
            self.group_preserving_allow_partial_group
            if allow_partial_group is None
            else allow_partial_group
        )
        allow_ungrouped = (
            self.group_preserving_allow_ungrouped
            if allow_ungrouped is None
            else allow_ungrouped
        )
        subset_strategy = subset_strategy or self.group_preserving_subset_strategy
        groups: dict[str, list[int]] = {}
        ungrouped_count = 0
        for index, entry in enumerate(self.buffer):
            group_value = getattr(entry.data, group_attr, None)
            if group_value is None or str(group_value).strip() == "":
                if not allow_ungrouped:
                    ungrouped_count += 1
                    continue
                group_value = f"__ungrouped_{index}"
            groups.setdefault(str(group_value), []).append(index)

        eligible_groups: list[tuple[str, list[int]]] = []
        incomplete_group_count = 0
        oversized_group_count = 0
        nonzero_filtered_group_count = 0
        for group_id, indices in groups.items():
            entries = [self.buffer[index] for index in indices]
            expected_sizes = {
                int(size)
                for size in (
                    getattr(entry.data, group_size_attr, None)
                    for entry in entries
                )
                if size is not None
            }
            expected_size = max(expected_sizes) if expected_sizes else None
            if (
                require_complete_groups
                and expected_size is not None
                and len(indices) < expected_size
            ):
                incomplete_group_count += 1
                continue
            if not allow_partial_group and len(indices) > total_samples:
                oversized_group_count += 1
                continue
            if self.sample_nonzero_advantage_only and not any(
                _has_nonzero_advantage(entry.data, self.advantage_epsilon)
                for entry in entries
            ):
                nonzero_filtered_group_count += 1
                continue
            eligible_groups.append((group_id, list(indices)))

        eligible_episode_count = sum(len(indices) for _, indices in eligible_groups)
        record_metric(f"{metric_prefix}/enabled", 1, Reduce.MEAN)
        record_metric(
            f"{metric_prefix}/buffer_group_count",
            len(groups),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/eligible_group_count",
            len(eligible_groups),
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/eligible_episode_count",
            eligible_episode_count,
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/requested_episode_count",
            total_samples,
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/incomplete_group_count",
            incomplete_group_count,
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/oversized_group_count",
            oversized_group_count,
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/nonzero_filtered_group_count",
            nonzero_filtered_group_count,
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/ungrouped_ignored_count",
            ungrouped_count,
            Reduce.MEAN,
        )

        if eligible_episode_count < total_samples:
            return None

        priority = str(self.group_preserving_priority).lower()
        if priority not in {"random", "loss_priority_max", "loss_priority_mean"}:
            raise ValueError(
                "group_preserving_priority must be one of "
                "'random', 'loss_priority_max', or 'loss_priority_mean'."
            )

        shuffled_groups = list(eligible_groups)
        if priority == "random":
            random.shuffle(shuffled_groups)
        else:
            weighted_groups: list[tuple[float, tuple[str, list[int]]]] = []
            for group in shuffled_groups:
                _, indices = group
                priorities = [
                    max(
                        _as_float(getattr(self.buffer[index].data, "loss_priority", None))
                        or 0.0,
                        0.0,
                    )
                    for index in indices
                ]
                if priority == "loss_priority_mean":
                    score = sum(priorities) / max(len(priorities), 1)
                else:
                    score = max(priorities, default=0.0)
                weighted_groups.append(
                    (score + float(self.group_preserving_priority_epsilon), group)
                )
            shuffled_groups = []
            while weighted_groups:
                total_weight = sum(weight for weight, _ in weighted_groups)
                if total_weight <= 0:
                    random.shuffle(weighted_groups)
                    shuffled_groups.extend(group for _, group in weighted_groups)
                    break
                pick = random.random() * total_weight
                cumulative = 0.0
                selected_idx = len(weighted_groups) - 1
                for idx, (weight, _) in enumerate(weighted_groups):
                    cumulative += weight
                    if cumulative >= pick:
                        selected_idx = idx
                        break
                _, group = weighted_groups.pop(selected_idx)
                shuffled_groups.append(group)
            record_metric(f"{metric_prefix}/priority_enabled", 1, Reduce.MEAN)
        record_metric(
            f"{metric_prefix}/priority_mode/{_metric_fragment(priority)}",
            1,
            Reduce.MEAN,
        )
        selected_indices: list[int] = []
        selected_group_count = 0
        for _, indices in shuffled_groups:
            remaining_capacity = total_samples - len(selected_indices)
            if remaining_capacity <= 0:
                break
            if len(indices) > remaining_capacity:
                if not allow_partial_group:
                    continue
                subset = _sample_group_subset(
                    self.buffer,
                    indices,
                    remaining_capacity,
                    strategy=subset_strategy,
                    metric_prefix=metric_prefix,
                )
                selected_indices.extend(subset)
            else:
                selected_indices.extend(indices)
            selected_group_count += 1
            if len(selected_indices) == total_samples:
                break

        if len(selected_indices) < total_samples and allow_partial_group:
            already_selected = set(selected_indices)
            for _, indices in shuffled_groups:
                if len(selected_indices) >= total_samples:
                    break
                remaining = [index for index in indices if index not in already_selected]
                subset = _sample_group_subset(
                    self.buffer,
                    remaining,
                    total_samples - len(selected_indices),
                    strategy=subset_strategy,
                    metric_prefix=metric_prefix,
                )
                selected_indices.extend(subset)
                already_selected.update(subset)

        exact_fill = len(selected_indices) == total_samples
        record_metric(
            f"{metric_prefix}/sample_exact_fill",
            int(exact_fill),
            Reduce.SUM,
        )
        record_metric(
            f"{metric_prefix}/sampled_group_count",
            selected_group_count,
            Reduce.MEAN,
        )
        record_metric(
            f"{metric_prefix}/sampled_episode_count",
            len(selected_indices),
            Reduce.MEAN,
        )
        if not exact_fill:
            return None
        return selected_indices

    @endpoint
    async def evict(self, curr_policy_version: int) -> None:
        """Evict episodes from the replay buffer if they are too old based on the current policy version
        and the max policy age allowed.

        Args:
            curr_policy_version (int): The current policy version.
        """
        self._evict(curr_policy_version)

    @endpoint
    async def purge_by_proposal_origin(self, origins: list[str] | tuple[str, ...]) -> int:
        """Drop all buffered episodes whose proposal origin is in ``origins``."""
        origin_set = {str(origin) for origin in origins}
        before = len(self.buffer)
        self.buffer = deque(
            (
                entry
                for entry in self.buffer
                if _metadata_value(entry.data, "proposal_origin") not in origin_set
            ),
            maxlen=self.max_buffer_size,
        )
        purged = before - len(self.buffer)
        record_metric("learning/replay/purge_by_origin/episodes", purged, Reduce.SUM)
        for origin in origin_set:
            record_metric(
                f"learning/replay/purge_by_origin/{_metric_fragment(origin)}",
                purged,
                Reduce.SUM,
            )
        logger.warning(
            "Purged %d replay episodes by proposal_origin in %s; remaining=%d",
            purged,
            sorted(origin_set),
            len(self.buffer),
        )
        return purged

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
            _has_nonzero_advantage(entry.data, self.advantage_epsilon)
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

    def _record_contrast_metrics(
        self,
        episodes: list["Episode"],
        *,
        prefix: str,
        record_cells: bool,
    ) -> None:
        if not episodes:
            return

        targets: set[str] = set()
        responses: set[str] = set()
        target_actions: set[str] = set()
        response_actions: set[str] = set()
        validity_classes: set[str] = set()
        advantage_signs: set[str] = set()
        proposal_origins: set[str] = set()
        proposal_origin_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        slate_counts: dict[str, int] = {}
        responses_by_target: dict[str, set[str]] = {}
        response_actions_by_target_action: dict[str, set[str]] = {}
        cell_counts: dict[tuple[str, str], int] = {}
        cell_advantage_sums: dict[tuple[str, str], float] = {}
        action_cell_counts: dict[tuple[str, str], int] = {}
        action_cell_advantage_sums: dict[tuple[str, str], float] = {}
        positive_count = 0
        negative_count = 0
        zero_count = 0

        for episode in episodes:
            target = _metadata_value(episode, "target")
            target_action = _metadata_value(
                episode,
                "target_action",
                default=target,
            )
            response = _metadata_value(
                episode,
                "normalized_response",
                default=_metadata_value(episode, "response"),
            )
            response_action = _metadata_value(
                episode,
                "response_action",
                default=response,
            )
            validity_class = _metadata_value(episode, "validity_class")
            proposal_origin = _metadata_value(episode, "proposal_origin")
            sign = _metadata_value(
                episode,
                "advantage_sign",
                default=_advantage_sign(episode, self.advantage_epsilon),
            )
            group_id = _metadata_value(episode, "group_id")
            slate_id = _metadata_value(episode, "slate_id")
            advantage = _as_float(getattr(episode, "advantage", None)) or 0.0

            targets.add(target)
            responses.add(response)
            target_actions.add(target_action)
            response_actions.add(response_action)
            validity_classes.add(validity_class)
            advantage_signs.add(sign)
            proposal_origins.add(proposal_origin)
            proposal_origin_counts[proposal_origin] = (
                proposal_origin_counts.get(proposal_origin, 0) + 1
            )
            group_counts[group_id] = group_counts.get(group_id, 0) + 1
            slate_counts[slate_id] = slate_counts.get(slate_id, 0) + 1
            responses_by_target.setdefault(target, set()).add(response)
            response_actions_by_target_action.setdefault(target_action, set()).add(
                response_action
            )
            cell = (target, response)
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
            cell_advantage_sums[cell] = cell_advantage_sums.get(cell, 0.0) + advantage
            action_cell = (target_action, response_action)
            action_cell_counts[action_cell] = action_cell_counts.get(action_cell, 0) + 1
            action_cell_advantage_sums[action_cell] = (
                action_cell_advantage_sums.get(action_cell, 0.0) + advantage
            )
            if sign == "positive":
                positive_count += 1
            elif sign == "negative":
                negative_count += 1
            else:
                zero_count += 1

        record_metric(f"{prefix}/episode_count", len(episodes), Reduce.MEAN)
        record_metric(f"{prefix}/target_count", len(targets), Reduce.MEAN)
        record_metric(f"{prefix}/response_count", len(responses), Reduce.MEAN)
        record_metric(f"{prefix}/cell_count", len(cell_counts), Reduce.MEAN)
        record_metric(
            f"{prefix}/target_action_count",
            len(target_actions),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/response_action_count",
            len(response_actions),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/action_cell_count",
            len(action_cell_counts),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/validity_class_count",
            len(validity_classes),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/advantage_sign_count",
            len(advantage_signs),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/proposal_origin_count",
            len(proposal_origins),
            Reduce.MEAN,
        )
        for origin, count in proposal_origin_counts.items():
            record_metric(
                f"{prefix}/proposal_origin/{_metric_fragment(origin)}/fraction",
                count / max(len(episodes), 1),
                Reduce.MEAN,
            )
        record_metric(
            f"{prefix}/positive_fraction",
            positive_count / max(len(episodes), 1),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/negative_fraction",
            negative_count / max(len(episodes), 1),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/zero_fraction",
            zero_count / max(len(episodes), 1),
            Reduce.MEAN,
        )
        record_metric(f"{prefix}/group_count", len(group_counts), Reduce.MEAN)
        record_metric(f"{prefix}/slate_count", len(slate_counts), Reduce.MEAN)
        record_metric(
            f"{prefix}/max_episodes_per_group",
            max(group_counts.values()) if group_counts else 0,
            Reduce.MAX,
        )
        record_metric(
            f"{prefix}/max_episodes_per_slate",
            max(slate_counts.values()) if slate_counts else 0,
            Reduce.MAX,
        )
        record_metric(
            f"{prefix}/mean_responses_per_target",
            sum(len(values) for values in responses_by_target.values())
            / max(len(responses_by_target), 1),
            Reduce.MEAN,
        )
        record_metric(
            f"{prefix}/mean_response_actions_per_target_action",
            sum(len(values) for values in response_actions_by_target_action.values())
            / max(len(response_actions_by_target_action), 1),
            Reduce.MEAN,
        )

        if record_cells:
            b2d1_direction = 0.0
            b2d1_action_direction = 0.0
            for (target, response), count in cell_counts.items():
                cell_prefix = (
                    f"{prefix}/cell/"
                    f"{_metric_fragment(target)}/{_metric_fragment(response)}"
                )
                record_metric(f"{cell_prefix}/count", count, Reduce.MEAN)
                record_metric(
                    f"{cell_prefix}/advantage_sum",
                    cell_advantage_sums[(target, response)],
                    Reduce.MEAN,
                )
                if target == "0" and response == "0":
                    b2d1_direction += cell_advantage_sums[(target, response)]
                elif target == "0" and response == "1":
                    b2d1_direction -= cell_advantage_sums[(target, response)]
                elif target == "1" and response == "1":
                    b2d1_direction += cell_advantage_sums[(target, response)]
                elif target == "1" and response == "0":
                    b2d1_direction -= cell_advantage_sums[(target, response)]
            record_metric(
                f"{prefix}/b2d1_directional_advantage",
                b2d1_direction,
                Reduce.MEAN,
            )
            for (target_action, response_action), count in action_cell_counts.items():
                cell_prefix = (
                    f"{prefix}/action_cell/"
                    f"{_metric_fragment(target_action)}/{_metric_fragment(response_action)}"
                )
                record_metric(f"{cell_prefix}/count", count, Reduce.MEAN)
                record_metric(
                    f"{cell_prefix}/advantage_sum",
                    action_cell_advantage_sums[(target_action, response_action)],
                    Reduce.MEAN,
                )
                if target_action == "0" and response_action == "0":
                    b2d1_action_direction += action_cell_advantage_sums[
                        (target_action, response_action)
                    ]
                elif target_action == "0" and response_action == "1":
                    b2d1_action_direction -= action_cell_advantage_sums[
                        (target_action, response_action)
                    ]
                elif target_action == "1" and response_action == "1":
                    b2d1_action_direction += action_cell_advantage_sums[
                        (target_action, response_action)
                    ]
                elif target_action == "1" and response_action == "0":
                    b2d1_action_direction -= action_cell_advantage_sums[
                        (target_action, response_action)
                    ]
            record_metric(
                f"{prefix}/b2d1_action_directional_advantage",
                b2d1_action_direction,
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
