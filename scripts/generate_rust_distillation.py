#!/usr/bin/env python3
"""Generate deterministic Broca distillation seed pairs for logic substrates."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


CHANNEL_COUNT = 43


@dataclass(frozen=True)
class Golden:
    substrate: str
    prompt: str
    code: str


GOLDENS: tuple[Golden, ...] = (
    Golden(
        "rust",
        "calculate fibonacci sequence iteratively",
        """
pub fn fibonacci(n: u64) -> u64 {
    let mut a = 0;
    let mut b = 1;
    for _ in 0..n {
        let next = a + b;
        a = b;
        b = next;
    }
    a
}
""",
    ),
    Golden(
        "rust",
        "binary search on a sorted slice",
        """
pub fn binary_search(xs: &[i32], target: i32) -> Option<usize> {
    let mut lo = 0;
    let mut hi = xs.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match xs[mid].cmp(&target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}
""",
    ),
    Golden(
        "rust",
        "filter and collect even numbers",
        """
pub fn evens(input: &[i32]) -> Vec<i32> {
    input.iter().copied().filter(|x| x % 2 == 0).collect()
}
""",
    ),
    Golden(
        "rust",
        "0-1 knapsack dynamic program",
        """
pub fn knapsack(capacity: usize, weights: &[usize], values: &[u32]) -> u32 {
    let mut dp = vec![0; capacity + 1];
    for (i, &weight) in weights.iter().enumerate() {
        for slot in (weight..=capacity).rev() {
            dp[slot] = dp[slot].max(dp[slot - weight] + values[i]);
        }
    }
    dp[capacity]
}
""",
    ),
    Golden(
        "python",
        "calculate fibonacci sequence iteratively",
        """
def fibonacci(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""",
    ),
    Golden(
        "python",
        "binary search on a sorted list",
        """
def binary_search(xs, target):
    lo, hi = 0, len(xs)
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if xs[mid] == target:
            return mid
        if xs[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return None
""",
    ),
)


def stable_channels(substrate: str, prompt: str, seed: str) -> list[float]:
    digest = hashlib.sha256(f"{seed}:{substrate}:{prompt}".encode()).digest()
    channels: list[float] = []
    counter = 0
    while len(channels) < CHANNEL_COUNT:
        block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        for i in range(0, len(block), 4):
            raw = int.from_bytes(block[i : i + 4], "big")
            channels.append(round(raw / 0xFFFFFFFF, 6))
            if len(channels) == CHANNEL_COUNT:
                break
        counter += 1
    return channels


def is_holdout(substrate: str, prompt: str, holdout_mod: int) -> bool:
    if holdout_mod <= 0:
        return False
    h = hashlib.blake2s(f"{substrate}:{prompt}".encode()).digest()
    return int.from_bytes(h[:4], "big") % holdout_mod == 0


def write_jsonl(output: Path, seed: str, holdout_mod: int) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for golden in GOLDENS:
            record = {
                "substrate": golden.substrate,
                "prompt": golden.prompt,
                "channels": stable_channels(golden.substrate, golden.prompt, seed),
                "code": golden.code.strip(),
                "iterations": 1,
                "repair_steps": 0,
                "holdout": is_holdout(golden.substrate, golden.prompt, holdout_mod),
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(GOLDENS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rust-logic-pairs-standalone.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--seed",
        default="symthaea-distillation-v1",
        help="Deterministic channel seed.",
    )
    parser.add_argument(
        "--holdout-mod",
        type=int,
        default=0,
        help="Mark roughly 1/N rows as holdout; 0 disables holdouts.",
    )
    args = parser.parse_args()

    count = write_jsonl(args.out, args.seed, args.holdout_mod)
    print(f"generated {count} distillation pairs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
