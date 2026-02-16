# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re


class MathReward:
    """Reward class for evaluating math correctness."""

    def __init__(self, tolerance: float = 1e-6, partial_credit: float = 0.1):
        self.tolerance = tolerance
        self.partial_credit = partial_credit

    def __call__(self, prompt: str, response: str, target: str) -> float:
        """Compute math correctness reward."""
        target_number = self._to_float(target)
        if target_number is None:
            return 0.0

        # Look for answer in <answer></answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

        if answer_match:
            model_answer = self._to_float(answer_match.group(1).strip())
            if (
                model_answer is not None
                and abs(target_number - model_answer) < self.tolerance
            ):
                return 1.0  # Correct answer

        # Check for partial credit: target number appears elsewhere in response
        response_without_answer_tags = re.sub(
            r"<answer>.*?</answer>", "", response, flags=re.DOTALL
        )
        # Convert to int if it's a whole number to avoid "117.0" vs "117" mismatch
        target_str = (
            str(int(target_number))
            if target_number.is_integer()
            else str(target_number)
        )
        if target_str in response_without_answer_tags:
            return self.partial_credit

        return 0.0  # No match

    def _to_float(self, text: str) -> float | None:
        """Convert text to float, return None if invalid."""
        try:
            # Remove common non-numeric characters like $, commas, etc.
            cleaned_text = re.sub(r"[$,\s]", "", text.strip())
            return float(cleaned_text)
        except (ValueError, AttributeError):
            return None


class ThinkingReward:
    """Reward class for evaluating use of thinking tags in reasoning.

    Args:
        partial_reward: Reward for partial tag usage (incomplete/malformed)
        full_reward: Reward for well-formed thinking blocks with content
        tag: Tag name to use (default "think", can use "思考" for Japanese, etc.)
    """

    def __init__(
        self, partial_reward: float = 0.2, full_reward: float = 1.0, tag: str = "think"
    ):
        self.partial_reward = partial_reward
        self.full_reward = full_reward
        self.tag = tag
        # Build regex patterns for the specified tag
        self._THINK_BLOCK_RE = re.compile(
            rf"<\s*{re.escape(tag)}\s*>(.*?)<\s*/\s*{re.escape(tag)}\s*>",
            re.IGNORECASE | re.DOTALL,
        )
        self._THINK_TAG_ATTEMPT_RE = re.compile(
            rf"<\s*/?\s*{re.escape(tag)}\s*>", re.IGNORECASE
        )

    def __call__(self, prompt: str, response: str, target: str | None = None) -> float:
        """Compute thinking reward."""
        if not response:
            return 0.0

        matches = self._THINK_BLOCK_RE.findall(response)
        has_well_formed = any(len(re.sub(r"\s+", "", m)) >= 1 for m in matches)
        has_attempt = bool(self._THINK_TAG_ATTEMPT_RE.search(response)) or bool(matches)
        if has_well_formed:
            return self.full_reward
        elif has_attempt:
            return self.partial_reward
        return 0.0


class AlienDigitsReward:
    """Dense analytical reward for alien-token digit decoding tasks.

    Matches the reward design in reports/20260215-rl-smoke-test/README.md:
    - per-digit aligned accuracy,
    - strong penalty for non-digit output,
    - mild length mismatch penalty.
    """

    def __init__(
        self,
        non_digit_penalty: float = 1.0,
        length_penalty_weight: float = 0.2,
        mapping: dict[str, str] | None = None,
    ):
        self.non_digit_penalty = non_digit_penalty
        self.length_penalty_weight = length_penalty_weight
        self.mapping = mapping or {
            "dax": "0",
            "wug": "1",
            "zib": "2",
            "kef": "3",
            "mon": "4",
            "pav": "5",
            "lur": "6",
            "sot": "7",
            "bim": "8",
            "teg": "9",
        }

    def __call__(self, prompt: str, response: str, target: str | None = None) -> float:
        """Compute alien-token decoding reward."""
        target_text = self._target_from_prompt(prompt)
        if target_text is None:
            target_text = self._normalize_digits(target)
        if not target_text:
            return 0.0

        completion_line = (response or "").splitlines()[0].strip() if response else ""
        prediction = completion_line.replace(" ", "")
        non_digit = 1 if re.search(r"\D", prediction) else 0

        target_len = len(target_text)
        prefix = prediction[:target_len]
        correct = sum(a == b for a, b in zip(prefix, target_text))
        accuracy = correct / target_len
        length_penalty = abs(len(prediction) - target_len) / target_len

        reward = accuracy
        reward -= self.non_digit_penalty * non_digit
        reward -= self.length_penalty_weight * length_penalty
        return float(reward)

    def _target_from_prompt(self, prompt: str) -> str | None:
        """Decode alien tokens from the final non-empty line of prompt."""
        if not prompt:
            return None

        lines = [line.strip() for line in prompt.splitlines() if line.strip()]
        if not lines:
            return None

        candidate_tokens = lines[-1].split()
        if not candidate_tokens:
            return None
        if any(token not in self.mapping for token in candidate_tokens):
            return None

        return "".join(self.mapping[token] for token in candidate_tokens)

    def _normalize_digits(self, text: str | None) -> str | None:
        """Fallback parser that keeps only digits."""
        if text is None:
            return None

        digits = "".join(ch for ch in str(text) if ch.isdigit())
        return digits if digits else None
