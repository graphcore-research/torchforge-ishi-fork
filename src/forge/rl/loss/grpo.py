# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Annotated, Any, Literal

import torch
from forge.observability.metrics import Metric, Reduce
from forge.rl.loss.ops import (
    aggregate,
    compute_entropy,
    compute_kl,
    compute_logprobs,
    compute_ratio,
    pg_ppo_clip,
)
from forge.rl.loss.types import AggType, BaseLossConfig, LossOutput
from pydantic import Field


class GRPOLoss(BaseLossConfig):
    """DR-GRPO: "Done Right" GRPO with unbiased aggregation.

    Reference: Liu et al., "Understanding R1-Zero-Like Training" (2025).
    https://arxiv.org/abs/2503.20783

    Per-token: L_t = max(-r*A, -clip(r, 1-ε, 1+ε)*A) + β*KL
    Aggregated: L = sum(L_t * mask) / (B * MAX_LEN)

    where:
        r = π_θ(y_t|q,y_<t) / π_old(y_t|q,y_<t)  — importance ratio
        A = R - mean(R)                          - No std norm, to avoid difficulty bias
        KL = r_ref - log(r_ref) - 1              — k3 estimator, r_ref = π_ref/π_θ
        B * MAX_LEN = fixed denominator batch_size * max sequence length

    GRPO replaces PPO's learned value function with group-relative advantages.
    Sample multiple responses per prompt, compute advantages by comparing rewards
    within each group. This eliminates the need for a separate critic model at
    the cost of sampling more responses.

    DR-GRPO fixes two biases in vanilla GRPO:
    1. Length bias: GRPO divides by |o_i|, i.e. agg_type='sequence_mean',
       rewarding the model for producing shorter correct and longer incorrect sequences,
       resulting in unnecessarily increased lengths during training.
       DR-GRPO uses agg_type='fixed_horizon' to remove this bias, dividing by a constant
       denominator (sequence dimension size) instead.
    2. Difficulty bias: GRPO normalizes advantages by std, over-weighting easy
       problems with low variance. DR-GRPO uses mean-only advantages. NOTE:
       This should be changed at the **advantage** computation level.

    NOTE: Default sets clip_high>clip_low, as this reportedly better, although not
    explored in the original paper.

    Args:
        clip_low (float): Lower clip bound (default 0.2).
        clip_high (float): Upper clip bound (default 0.28).
        beta (float): KL penalty coefficient (default 0.1).
        agg_type (AggType): Aggregation method (default "fixed_horizon").
    """

    clip_low: Annotated[float, Field(ge=0, le=1)] = 0.2
    clip_high: Annotated[float, Field(ge=0, le=1)] = 0.28
    beta: Annotated[float, Field(ge=0)] = 0.1
    valid_action_support_floor_coef: Annotated[float, Field(ge=0)] = 0.0
    valid_action_support_floor: Annotated[float, Field(ge=0, le=1)] = 0.2
    valid_action_support_token_ids: tuple[int, ...] = ()
    valid_action_support_symbols: tuple[str, ...] = ()
    action_schema: dict[str, Any] = Field(default_factory=dict)
    valid_action_support_floor_scope: Literal["first_active", "digit_positions"] = (
        "first_active"
    )
    valid_action_support_floor_max_positions: Annotated[int, Field(ge=1)] = 1
    valid_action_support_eps: Annotated[float, Field(gt=0)] = 1e-8
    agg_type: AggType = "fixed_horizon"

    def __call__(
        self,
        logits: torch.Tensor,  # (B, S, V)
        target_ids: torch.Tensor,  # (B, S)
        advantages: torch.Tensor,  # (B, S)
        generator_logprobs: torch.Tensor,  # (B, S)
        loss_mask: torch.Tensor,  # (B, S)
        ref_logprobs: torch.Tensor | None = None,  # (B, S) or None
        loss_scale: torch.Tensor | None = None,
    ) -> LossOutput:
        logprobs, lp_m = compute_logprobs(logits, target_ids)
        entropy, ent_m = compute_entropy(logits, loss_mask)  # logging only
        ratio, log_ratio, ratio_m = compute_ratio(
            logprobs, generator_logprobs, loss_mask, ratio_type="token"
        )
        pg_loss, clip_m = pg_ppo_clip(
            ratio, advantages, loss_mask, self.clip_low, self.clip_high
        )

        kl_m: list[Metric] = []
        if self.beta > 0:
            if ref_logprobs is None:
                raise ValueError("ref_logprobs required when beta > 0")
            kl, kl_m = compute_kl(logprobs, ref_logprobs, loss_mask)
            pg_loss = pg_loss + self.beta * kl

        support_floor_loss = None
        support_floor_m: list[Metric] = []
        if self.valid_action_support_floor_coef > 0:
            if not self.valid_action_support_token_ids:
                raise ValueError(
                    "valid_action_support_floor_coef > 0 requires "
                    "valid_action_support_token_ids to be configured."
                )
            token_ids = torch.tensor(
                self.valid_action_support_token_ids,
                device=logits.device,
                dtype=torch.long,
            )
            scoped_logits = logits.index_select(dim=-1, index=token_ids)
            scoped_probs = torch.softmax(scoped_logits, dim=-1)
            active = loss_mask.to(dtype=torch.bool)
            support_masks: list[torch.Tensor] = []
            if self.valid_action_support_floor_scope == "first_active":
                first_active = (active.to(dtype=torch.int64).cumsum(dim=1) == 1) & active
                support_masks.append(first_active)
            else:
                token_matches = (target_ids.unsqueeze(-1) == token_ids).any(dim=-1)
                digit_positions = active & token_matches
                slot_indices = digit_positions.to(dtype=torch.int64).cumsum(dim=1)
                for position in range(self.valid_action_support_floor_max_positions):
                    support_masks.append(digit_positions & (slot_indices == position + 1))

            q_values: list[torch.Tensor] = []
            active_position_count = 0
            for support_mask in support_masks:
                active_scoped_probs = scoped_probs[support_mask]
                if active_scoped_probs.numel() == 0:
                    continue
                active_position_count += 1
                q_values.append(active_scoped_probs.mean(dim=0))

            if not q_values:
                q_values.append(
                    torch.full(
                        (len(self.valid_action_support_token_ids),),
                        1.0 / max(len(self.valid_action_support_token_ids), 1),
                        device=logits.device,
                        dtype=scoped_probs.dtype,
                    )
                )
            q_by_position = torch.stack(q_values, dim=0)

            floor = torch.as_tensor(
                self.valid_action_support_floor,
                device=logits.device,
                dtype=q_by_position.dtype,
            )
            eps = torch.as_tensor(
                self.valid_action_support_eps,
                device=logits.device,
                dtype=q_by_position.dtype,
            )
            deficits = torch.relu(torch.log(floor / (q_by_position + eps)))
            support_floor_loss = self.valid_action_support_floor_coef * (
                deficits * deficits
            ).sum()
            q = q_by_position.mean(dim=0)
            support_floor_m = [
                Metric(
                    key="loss/valid_action_support_floor/coef",
                    value=self.valid_action_support_floor_coef,
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/floor",
                    value=self.valid_action_support_floor,
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/loss",
                    value=support_floor_loss.detach(),
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/min_q",
                    value=q.min().detach(),
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/max_q",
                    value=q.max().detach(),
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/q_variance",
                    value=q.var(unbiased=False).detach(),
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/token_id_count",
                    value=len(self.valid_action_support_token_ids),
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/configured_position_count",
                    value=len(support_masks),
                    reduction=Reduce.MEAN,
                ),
                Metric(
                    key="loss/valid_action_support_floor/active_position_count",
                    value=active_position_count,
                    reduction=Reduce.MEAN,
                ),
            ]
            for position, position_q in enumerate(q_values):
                support_floor_m.extend(
                    [
                        Metric(
                            key=f"loss/valid_action_support_floor/position_{position}/min_q",
                            value=position_q.min().detach(),
                            reduction=Reduce.MEAN,
                        ),
                        Metric(
                            key=f"loss/valid_action_support_floor/position_{position}/max_q",
                            value=position_q.max().detach(),
                            reduction=Reduce.MEAN,
                        ),
                    ]
                )
            for idx, token_id in enumerate(self.valid_action_support_token_ids):
                support_floor_m.append(
                    Metric(
                        key=f"loss/valid_action_support_floor/q/token_{token_id}",
                        value=q[idx].detach(),
                        reduction=Reduce.MEAN,
                    )
                )

        loss, agg_m = aggregate(pg_loss, loss_mask, self.agg_type, loss_scale)
        if support_floor_loss is not None:
            loss = loss + support_floor_loss

        return LossOutput(
            loss,
            lp_m
            + ent_m
            + ratio_m
            + clip_m
            + kl_m
            + support_floor_m
            + agg_m,
        )
