#!/usr/bin/env python3
"""Verify strict epistemic gating suppresses unsafe code tokens."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


SAFE_KEYWORDS = {"let", "in", "inherit", "import", "with", "rec", "fn", "pub", "return"}


@dataclass(frozen=True)
class EpistemicCubeGate:
    penalty: float = 5.0
    certainty_threshold: float = 0.8

    def apply_strict_gate(
        self, logits: list[float], tokens: list[str], certainty_score: float
    ) -> list[float]:
        if len(logits) != len(tokens):
            raise ValueError("logits and tokens must have the same length")
        if not 0.0 <= certainty_score <= 1.0:
            raise ValueError("certainty_score must be in [0, 1]")
        if certainty_score >= self.certainty_threshold:
            return list(logits)

        penalty_factor = (1.0 - certainty_score) * self.penalty
        return [
            logit if token in SAFE_KEYWORDS else logit - penalty_factor
            for logit, token in zip(logits, tokens, strict=True)
        ]


def run(certainty: float) -> int:
    vocab = ["fn", "pub", "main", "hallucination_X", "unknown_Y", "return"]
    logits = [2.0] * len(vocab)
    gate = EpistemicCubeGate()
    gated = gate.apply_strict_gate(logits, vocab, certainty)

    hallucinated_idx = vocab.index("hallucination_X")
    safe_idx = vocab.index("fn")
    suppressed = gated[hallucinated_idx] < logits[hallucinated_idx]
    safe_unchanged = gated[safe_idx] == logits[safe_idx]

    print("Distillation gating verification")
    print(f"  certainty: {certainty}")
    print(f"  logits:    {logits}")
    print(f"  gated:     {gated}")

    if not suppressed:
        print("FAIL: hallucinated token was not suppressed")
        return 1
    if not safe_unchanged:
        print("FAIL: safe keyword was modified")
        return 1

    print("PASS: unsafe token suppressed while safe keyword stayed stable")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--certainty", type=float, default=0.2)
    args = parser.parse_args()
    return run(args.certainty)


if __name__ == "__main__":
    raise SystemExit(main())
