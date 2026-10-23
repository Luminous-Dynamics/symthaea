#!/usr/bin/env python3
"""Validate MATH-BUILD-002 receipt-chain continuity.

Single-receipt validity proves that one receipt is internally coherent. It does
not prove that the predecessor named by digest is the predecessor whose content
is being presented to a reviewer. This validator closes that substitution gap.

It is intentionally stdlib-only and dynamically loads the sibling receipt
validator so the chain theorem cannot diverge from single-receipt semantics.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
VALIDATOR_PATH = HERE / "validate-math-build-receipt.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("symthaea_math_build_receipt", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load receipt validator from {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V = load_validator()


class ChainError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ChainError(message)


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ChainError(f"{path}: cannot load receipt: {error}") from error
    require(isinstance(value, dict), f"{path}: receipt must be a JSON object")
    return value


def validate_generation_replay_chain(
    generation: dict[str, Any], replay: dict[str, Any]
) -> None:
    V.validate_receipt(generation)
    V.validate_receipt(replay)

    require(
        generation["receipt_role"] == V.ROLE_GENERATION,
        "first receipt must have role artifact_generation",
    )
    require(
        replay["receipt_role"] == V.ROLE_REPLAY,
        "second receipt must have role artifact_replay",
    )
    require(
        generation["outcome"] == "pass",
        "promotion chain requires artifact_generation outcome=pass",
    )
    require(
        replay["outcome"] == "pass",
        "promotion chain requires artifact_replay outcome=pass",
    )

    generation_payload = generation["payload"]
    replay_payload = replay["payload"]

    require(
        replay_payload["generation_receipt_sha256"]
        == generation["receipt_sha256"],
        "replay predecessor digest does not identify the supplied generation receipt",
    )
    require(
        replay_payload["old_subject"] == generation_payload["source_subject"],
        "replay old_subject does not equal generation source_subject",
    )
    require(
        replay_payload["generated_artifact_sha256"]
        == generation_payload["lock_candidate_sha256"],
        "replay generated artifact does not equal generation lock candidate",
    )
    require(
        replay_payload["old_subject"] != replay_payload["repaired_subject"],
        "replay repaired_subject must differ from old_subject",
    )


def validate_full_chain(
    generation: dict[str, Any],
    replay: dict[str, Any],
    exact_head: dict[str, Any],
) -> None:
    validate_generation_replay_chain(generation, replay)
    V.validate_receipt(exact_head)

    require(
        exact_head["receipt_role"] == V.ROLE_EXACT_HEAD,
        "third receipt must have role exact_head_build_qualification",
    )
    require(
        exact_head["outcome"] == "pass",
        "promotion chain requires exact_head_build_qualification outcome=pass",
    )

    replay_payload = replay["payload"]
    exact_payload = exact_head["payload"]

    require(
        exact_payload["replay_receipt_sha256"] == replay["receipt_sha256"],
        "exact-head predecessor digest does not identify the supplied replay receipt",
    )
    require(
        exact_payload["subject"] == replay_payload["repaired_subject"],
        "exact-head subject does not equal replay repaired_subject",
    )
    require(
        exact_payload["contained_lock_sha256"]
        == replay_payload["replayed_artifact_sha256"],
        "exact-head contained lock does not equal replayed artifact",
    )


def resign(receipt: dict[str, Any]) -> dict[str, Any]:
    receipt["receipt_sha256"] = V.compute_receipt_sha256(receipt)
    return receipt


def expect_chain_rejected(
    generation: dict[str, Any],
    replay: dict[str, Any],
    exact_head: dict[str, Any] | None,
    label: str,
) -> None:
    try:
        if exact_head is None:
            validate_generation_replay_chain(generation, replay)
        else:
            validate_full_chain(generation, replay, exact_head)
    except (ChainError, V.ReceiptError):
        return
    raise ChainError(f"self-test expected chain rejection: {label}")


def self_test() -> None:
    generation = V.generation_fixture()
    replay = V.replay_fixture(generation["receipt_sha256"])
    exact_head = V.exact_fixture(replay["receipt_sha256"])

    validate_generation_replay_chain(generation, replay)
    validate_full_chain(generation, replay, exact_head)

    bad_replay = copy.deepcopy(replay)
    bad_replay["payload"]["generation_receipt_sha256"] = "d" * 64
    resign(bad_replay)
    V.validate_receipt(bad_replay)
    expect_chain_rejected(
        generation,
        bad_replay,
        None,
        "replay points to a different generation digest",
    )

    bad_replay = copy.deepcopy(replay)
    bad_replay["payload"]["old_subject"]["commit_sha"] = "1" * 40
    resign(bad_replay)
    V.validate_receipt(bad_replay)
    expect_chain_rejected(
        generation,
        bad_replay,
        None,
        "replay substitutes a different old subject",
    )

    bad_replay = copy.deepcopy(replay)
    bad_replay["payload"]["generated_artifact_sha256"] = "d" * 64
    bad_replay["payload"]["replayed_artifact_sha256"] = "d" * 64
    resign(bad_replay)
    V.validate_receipt(bad_replay)
    expect_chain_rejected(
        generation,
        bad_replay,
        None,
        "replay substitutes different but internally equal artifact bytes",
    )

    bad_exact = copy.deepcopy(exact_head)
    bad_exact["payload"]["replay_receipt_sha256"] = "e" * 64
    resign(bad_exact)
    V.validate_receipt(bad_exact)
    expect_chain_rejected(
        generation,
        replay,
        bad_exact,
        "exact-head points to a different replay digest",
    )

    bad_exact = copy.deepcopy(exact_head)
    bad_exact["payload"]["subject"]["commit_sha"] = "2" * 40
    resign(bad_exact)
    V.validate_receipt(bad_exact)
    expect_chain_rejected(
        generation,
        replay,
        bad_exact,
        "exact-head substitutes a different repaired subject",
    )

    bad_exact = copy.deepcopy(exact_head)
    bad_exact["payload"]["contained_lock_sha256"] = "f" * 64
    resign(bad_exact)
    V.validate_receipt(bad_exact)
    expect_chain_rejected(
        generation,
        replay,
        bad_exact,
        "exact-head substitutes a different contained lock",
    )

    failed_generation = copy.deepcopy(generation)
    failed_generation["outcome"] = "fail"
    failed_generation["payload"]["gates"][0]["result"] = "fail"
    resign(failed_generation)
    V.validate_receipt(failed_generation)
    expect_chain_rejected(
        failed_generation,
        replay,
        None,
        "failed generation cannot participate in promotion chain",
    )

    print("math_build_receipt_chain_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("generation", nargs="?", type=Path)
    parser.add_argument("replay", nargs="?", type=Path)
    parser.add_argument("exact_head", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        try:
            V.self_test()
            self_test()
        except (ChainError, V.ReceiptError, RuntimeError) as error:
            print(f"math-build-receipt-chain: FAIL: {error}", file=sys.stderr)
            return 1
        return 0

    if args.generation is None or args.replay is None:
        parser.error("generation and replay receipt paths are required")

    try:
        generation = load_receipt(args.generation)
        replay = load_receipt(args.replay)
        if args.exact_head is None:
            validate_generation_replay_chain(generation, replay)
            print("math_build_receipt_chain=PASS depth=2")
        else:
            exact_head = load_receipt(args.exact_head)
            validate_full_chain(generation, replay, exact_head)
            print("math_build_receipt_chain=PASS depth=3")
    except (ChainError, V.ReceiptError, RuntimeError) as error:
        print(f"math-build-receipt-chain: FAIL: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
