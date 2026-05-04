# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test for data/replay_buffer.py"""

from dataclasses import dataclass

import pytest
import pytest_asyncio
from forge.actors.replay_buffer import BufferEntry, ReplayBuffer


@dataclass
class TestEpisode:
    """
    Dummy Episode containing just a policy version

    ReplayBuffer expects any construct (typically an Episode) that contains a
    `policy_version`.

    TODO: Replaced with a unified interface in the future.
    """

    policy_version: int
    advantage: float | None = None
    reward: float | None = None
    loss_priority: float | None = None
    group_id: str | None = None
    rollout_group_size: int | None = None
    slate_id: str | None = None
    slate_size: int | None = None
    slate_rank: int | None = None
    slate_reward_rank: int | None = None
    normalized_response: str | None = None
    response_action: str | None = None
    proposal_origin: str | None = None


class TestReplayBuffer:
    @pytest_asyncio.fixture
    async def replay_buffer(self) -> ReplayBuffer:
        replay_buffer = await ReplayBuffer.options(procs=1, with_gpus=False).as_actor(
            batch_size=2, max_policy_age=1
        )
        await replay_buffer.setup.call()
        return replay_buffer

    @pytest.mark.asyncio
    async def test_add(self, replay_buffer: ReplayBuffer) -> None:
        episode = TestEpisode(policy_version=0)
        await replay_buffer.add.call_one(episode)
        assert replay_buffer._numel.call_one().get() == 1
        assert replay_buffer._getitem.call_one(0).get() == episode
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_add_multiple(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        assert replay_buffer._numel.call_one().get() == 2
        assert replay_buffer._getitem.call_one(0).get() == episode_0
        assert replay_buffer._getitem.call_one(1).get() == episode_1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_state_dict_save_load(self, replay_buffer) -> None:
        episode = TestEpisode(policy_version=0)
        await replay_buffer.add.call_one(episode)
        state_dict = replay_buffer.state_dict.call_one().get()
        replay_buffer.clear.call_one().get()
        assert replay_buffer._numel.call_one().get() == 0
        await replay_buffer.load_state_dict.call_one(state_dict)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_evict(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        assert replay_buffer._numel.call_one().get() == 2
        await replay_buffer.evict.call_one(curr_policy_version=2)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        assert replay_buffer._numel.call_one().get() == 2

        # Test a simple sampling
        samples = await replay_buffer.sample.call_one(curr_policy_version=1)
        assert samples is not None
        assert len(samples[0]) == 2
        assert replay_buffer._numel.call_one().get() == 2

        # Test sampling (not enough samples in buffer, returns None)
        await replay_buffer.add.call_one(episode_0)
        samples = await replay_buffer.sample.call_one(curr_policy_version=1)
        assert samples is None
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_with_evictions(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        episode_2 = TestEpisode(policy_version=2)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        await replay_buffer.add.call_one(episode_2)
        assert replay_buffer._numel.call_one().get() == 3
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=2,
        )
        assert samples is not None
        assert len(samples[0]) == 2
        assert samples[0][0].policy_version > 0
        assert samples[0][1].policy_version > 0
        assert replay_buffer._numel.call_one().get() == 2
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_dp_size(self) -> None:
        """Test that len(samples) == dp_size when sampling."""
        # Create replay buffer with dp_size=3
        replay_buffer = await ReplayBuffer.options(procs=1, with_gpus=False).as_actor(
            batch_size=2, max_policy_age=1, dp_size=3
        )
        await replay_buffer.setup.call()

        # Add enough trajectories to sample
        for i in range(10):
            episode = TestEpisode(policy_version=0)
            await replay_buffer.add.call_one(episode)

        # Sample and verify len(samples) == dp_size
        samples = await replay_buffer.sample.call_one(curr_policy_version=0)
        assert samples is not None
        assert len(samples) == 3  # dp_size
        # Each sub-list should have batch_size samples
        for dp_samples in samples:
            assert len(dp_samples) == 2  # batch_size

        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_collect(self) -> None:
        """Test _collect method"""
        local_rb = ReplayBuffer(batch_size=1)
        await local_rb.setup._method(local_rb)
        for i in range(1, 6):
            local_rb.buffer.append(i)
        values = local_rb._collect([2, 0, -1])
        assert values == [3, 1, 5]
        values = local_rb._collect([1, 3])
        assert values == [2, 4]
        assert local_rb.buffer[0] == 1

    @pytest.mark.asyncio
    async def test_evict_preserves_max_buffer_size(self) -> None:
        """Bounded replay should stay bounded after explicit eviction."""
        local_rb = ReplayBuffer(batch_size=1, max_buffer_size=2)
        await local_rb.setup._method(local_rb)

        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0)))
        await local_rb.evict._method(local_rb, curr_policy_version=0)

        assert local_rb.buffer.maxlen == 2

        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0)))
        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0)))
        assert local_rb.buffer.maxlen == 2
        assert len(local_rb.buffer) == 2

    @pytest.mark.asyncio
    async def test_sample_nonzero_advantage_only(self) -> None:
        """Optional GRPO guard samples only episodes with useful advantage."""
        local_rb = ReplayBuffer(
            batch_size=2,
            max_policy_age=1,
            sample_nonzero_advantage_only=True,
            seed=1,
        )
        await local_rb.setup._method(local_rb)

        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0, advantage=0.0)))
        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0, advantage=1.0)))
        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0, advantage=-1.0)))

        samples = await local_rb.sample._method(local_rb, curr_policy_version=0)

        assert samples is not None
        assert len(samples[0]) == 2
        assert all(abs(sample.advantage) > 0 for sample in samples[0])

    @pytest.mark.asyncio
    async def test_sample_nonzero_advantage_only_waits_when_insufficient(
        self,
    ) -> None:
        local_rb = ReplayBuffer(
            batch_size=2,
            max_policy_age=1,
            sample_nonzero_advantage_only=True,
            seed=1,
        )
        await local_rb.setup._method(local_rb)

        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0, advantage=0.0)))
        local_rb.buffer.append(BufferEntry(TestEpisode(policy_version=0, advantage=1.0)))

        samples = await local_rb.sample._method(local_rb, curr_policy_version=0)

        assert samples is None

    @pytest.mark.asyncio
    async def test_group_preserving_sampling_returns_intact_groups(self) -> None:
        local_rb = ReplayBuffer(
            batch_size=4,
            max_policy_age=1,
            group_preserving_sampling=True,
            sample_nonzero_advantage_only=True,
            seed=1,
        )
        await local_rb.setup._method(local_rb)

        for group_id in ("a", "b"):
            for advantage in (1.0, -1.0):
                local_rb.buffer.append(
                    BufferEntry(
                        TestEpisode(
                            policy_version=0,
                            advantage=advantage,
                            group_id=group_id,
                            rollout_group_size=2,
                        )
                    )
                )
        local_rb.buffer.append(
            BufferEntry(
                TestEpisode(
                    policy_version=0,
                    advantage=1.0,
                    group_id="partial",
                    rollout_group_size=2,
                )
            )
        )

        samples = await local_rb.sample._method(local_rb, curr_policy_version=0)

        assert samples is not None
        assert len(samples[0]) == 4
        sampled_group_ids = [sample.group_id for sample in samples[0]]
        assert sampled_group_ids.count("a") == 2
        assert sampled_group_ids.count("b") == 2
        assert "partial" not in sampled_group_ids

    @pytest.mark.asyncio
    async def test_slate_contrast_coverage_selector_keeps_useful_cells(self) -> None:
        local_rb = ReplayBuffer(
            batch_size=4,
            max_policy_age=1,
            slate_preserving_sampling=True,
            slate_subset_strategy="contrast_coverage",
            seed=1,
        )
        await local_rb.setup._method(local_rb)

        episodes = [
            TestEpisode(
                policy_version=0,
                reward=3.0,
                loss_priority=2.0,
                slate_id="slate-a",
                slate_size=6,
                slate_rank=0,
                slate_reward_rank=0,
                response_action="0 0 0",
                proposal_origin="grammar_candidate",
            ),
            TestEpisode(
                policy_version=0,
                reward=1.0,
                loss_priority=0.1,
                slate_id="slate-a",
                slate_size=6,
                slate_rank=1,
                slate_reward_rank=3,
                response_action="3 3 3",
                proposal_origin="grammar_candidate",
            ),
            TestEpisode(
                policy_version=0,
                reward=2.0,
                loss_priority=5.0,
                slate_id="slate-a",
                slate_size=6,
                slate_rank=2,
                slate_reward_rank=1,
                response_action="0 0 1",
                proposal_origin="grammar_candidate",
            ),
            TestEpisode(
                policy_version=0,
                reward=1.5,
                loss_priority=2.0,
                slate_id="slate-a",
                slate_size=6,
                slate_rank=3,
                slate_reward_rank=2,
                response_action="1 2 3",
                proposal_origin="grammar_candidate",
            ),
            TestEpisode(
                policy_version=0,
                reward=0.0,
                loss_priority=4.0,
                slate_id="slate-a",
                slate_size=6,
                slate_rank=4,
                slate_reward_rank=5,
                response_action="2 2 2",
                proposal_origin="grammar_candidate",
            ),
            TestEpisode(
                policy_version=0,
                reward=0.5,
                loss_priority=3.0,
                slate_id="slate-a",
                slate_size=6,
                slate_rank=5,
                slate_reward_rank=4,
                response_action="0 3 0",
                proposal_origin="grammar_candidate",
            ),
        ]
        for episode in episodes:
            local_rb.buffer.append(BufferEntry(episode))

        samples = await local_rb.sample._method(local_rb, curr_policy_version=0)

        assert samples is not None
        selected_actions = {sample.response_action for sample in samples[0]}
        assert selected_actions == {"0 0 0", "3 3 3", "0 0 1", "1 2 3"}
