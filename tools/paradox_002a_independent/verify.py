#!/usr/bin/env python3
"""PARADOX-002A-I independent cross-language verifier.

Qualification flow:
1. Reproduce the already-qualified Rust census bytes externally.
2. Gate them by exact byte count and SHA-256.
3. Parse and independently derive the manipulation theorem in Python.
4. Compare every derived semantic field against paired Rust report bytes.
5. Run twelve fail-closed mutation controls.
6. Emit a deterministic normalized receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

from format import (
    Census, F64, FormatError, Pair, encode_census, parse_census, record_fixture_start,
)
from theorem import TheoremError, verify_pair

QUALIFIED_002A_SHA = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
QUALIFIED_CENSUS_SHA256 = "49ff56a49ac730d960d7625b65ddb0139171fa3862d145a6deb3d68245c7e711"
QUALIFIED_CENSUS_BYTES = 1_830_977
RECEIPT_SCHEMA = "SYMT-PARADOX-002A-I-RECEIPT-V1"
MUTATION_COUNT = 12


class VerificationError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_corpus(data: bytes, *, enforce_commitment: bool, run_mutations: bool) -> tuple[Census, str]:
    if enforce_commitment:
        if len(data) != QUALIFIED_CENSUS_BYTES:
            raise VerificationError(
                f"QualifiedCorpusReproductionMismatch: bytes={len(data)} expected={QUALIFIED_CENSUS_BYTES}"
            )
        digest = _sha256(data)
        if digest != QUALIFIED_CENSUS_SHA256:
            raise VerificationError(
                f"QualifiedCorpusReproductionMismatch: sha256={digest} expected={QUALIFIED_CENSUS_SHA256}"
            )
    try:
        census = parse_census(data)
        summary = hashlib.sha256()
        for pair in census.pairs:
            _, line = verify_pair(pair)
            summary.update(line)
    except (FormatError, TheoremError) as exc:
        raise VerificationError(str(exc)) from exc

    if run_mutations:
        _run_mutations(census, data)
    return census, summary.hexdigest()


def _pair_index(condition: int, *, seed: int = 11, trial: int = 0) -> int:
    seeds = (11, 29, 47, 71, 101, 149, 197, 257, 331, 419, 521, 631, 751, 887, 1021, 1171)
    seed_index = seeds.index(seed)
    return ((seed_index * 7 + condition) * 64) + trial


def _mutated_census(census: Census, index: int, mutate) -> bytes:
    pairs = list(census.pairs)
    pair = copy.deepcopy(pairs[index])
    mutate(pair)
    pairs[index] = pair
    return encode_census(Census(pairs))


def _expect_rejected(name: str, data: bytes) -> None:
    try:
        _verify_corpus(data, enforce_commitment=False, run_mutations=False)
    except VerificationError:
        return
    raise VerificationError(f"mutation control unexpectedly accepted: {name}")


def _run_mutations(census: Census, canonical_bytes: bytes) -> None:
    controls: list[tuple[str, bytes]] = []

    # 1. Duplicate fault-domain provenance across opposing C4 evidence.
    def m1(pair: Pair) -> None:
        pair.fixture.agent_view.events[1].fault_domain_id = pair.fixture.agent_view.events[0].fault_domain_id
    controls.append(("duplicate_fault_domain", _mutated_census(census, _pair_index(4), m1)))

    # 2. Inject contradiction into C1.
    def m2(pair: Pair) -> None:
        first = pair.fixture.agent_view.events[0].polarity
        pair.fixture.agent_view.events[1].polarity = 2 if first == 1 else 1
    controls.append(("c1_accidental_contradiction", _mutated_census(census, _pair_index(1), m2)))

    # 3. Remove C2 supersession.
    def m3(pair: Pair) -> None:
        pair.fixture.agent_view.events[2].supersedes_slot = None
    controls.append(("c2_missing_supersession", _mutated_census(census, _pair_index(2), m3)))

    # 4. Remove C3's usable claim->context relation while preserving context token shape.
    def m4(pair: Pair) -> None:
        wrong = pair.fixture.agent_view.events[1].visible_context
        pair.fixture.agent_view.events[0].visible_context = wrong
        pair.fixture.agent_view.events[2].visible_context = wrong
    controls.append(("c3_context_relation_removed", _mutated_census(census, _pair_index(3), m4)))

    # 5. Give C4 a hidden unique repair, converting it into an ontology-repair case.
    def m5(pair: Pair) -> None:
        a = 0x1111
        b = 0x2222
        target = pair.fixture.agent_view.events[0].polarity
        pair.fixture.truth.hidden_claim_contexts = [a, b, a, None]
        pair.fixture.truth.latent_context_is_causal = True
        pair.fixture.truth.context_dimension_available = False
        pair.fixture.truth.target_context = a
    controls.append(("c4_hidden_unique_repair", _mutated_census(census, _pair_index(4), m5)))

    # 6. Remove C5 causal self-reference markers.
    def m6(pair: Pair) -> None:
        for event in pair.fixture.agent_view.events:
            event.caused_by_self_prediction = False
    controls.append(("c5_self_reference_removed", _mutated_census(census, _pair_index(5), m6)))

    # 7. Leak explicit context into C6.
    def m7(pair: Pair) -> None:
        pair.fixture.agent_view.events[3].visible_context = pair.fixture.truth.target_context
    controls.append(("c6_explicit_context_disclosure", _mutated_census(census, _pair_index(6), m7)))

    # 8. Mark the hidden C6 context dimension as available.
    def m8(pair: Pair) -> None:
        pair.fixture.truth.context_dimension_available = True
    controls.append(("c6_context_dimension_available", _mutated_census(census, _pair_index(6), m8)))

    # 9. Raw length-preserving binary mutation: corrupt first event polarity option tag.
    raw = bytearray(canonical_bytes)
    fixture_start = record_fixture_start(canonical_bytes, 0)
    # Fixture prefix: condition(1)+seed(8)+trial(8)+agent_len(8)=25.
    # Agent prefix: prior(1)+query(2)+budget(8)=11.
    # Event prefix before polarity tag: slot(1)+source(2)+fault-domain(2)=5.
    option_tag_offset = fixture_start + 25 + 11 + 5
    if raw[option_tag_offset] != 1:
        raise VerificationError("internal mutation locator failed: expected Some polarity tag")
    raw[option_tag_offset] = 2
    controls.append(("corrupt_optional_presence_tag", bytes(raw)))

    # 10. Alter fixture identity.
    def m10(pair: Pair) -> None:
        pair.fixture.seed += 1
    controls.append(("altered_fixture_identity", _mutated_census(census, 0, m10)))

    # 11. Reorder two records without rewriting their identities.
    reordered = list(census.pairs)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    controls.append(("record_reordered", encode_census(Census(reordered))))

    # 12. Change one Rust report field while fixture bytes stay fixed.
    def m12(pair: Pair) -> None:
        pair.report.expected_response = 2 if pair.report.expected_response != 2 else 1
    controls.append(("rust_report_field_changed", _mutated_census(census, 0, m12)))

    if len(controls) != MUTATION_COUNT:
        raise VerificationError(f"internal mutation count mismatch: {len(controls)}")
    for name, mutated in controls:
        _expect_rejected(name, mutated)


def _validate_sha(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise VerificationError("verifier SHA must be a lowercase 40-hex Git commit SHA")
    return value


def _receipt(verifier_sha: str, semantic_summary_sha256: str) -> bytes:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "qualified_002a_sha": QUALIFIED_002A_SHA,
        "census_sha256": QUALIFIED_CENSUS_SHA256,
        "census_bytes": QUALIFIED_CENSUS_BYTES,
        "verifier_sha": verifier_sha,
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "format_protocol": "confirmatory-census-v1",
        "fixtures_checked": 7168,
        "reports_checked": 7168,
        "mutation_controls": MUTATION_COUNT,
        "semantic_summary_sha256": semantic_summary_sha256,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("census", type=Path)
    parser.add_argument("--verifier-sha", required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    try:
        verifier_sha = _validate_sha(args.verifier_sha)
        data = args.census.read_bytes()
        _, summary_sha = _verify_corpus(data, enforce_commitment=True, run_mutations=True)
        receipt = _receipt(verifier_sha, summary_sha)
        # Internal determinism guard; qualification should also invoke this program twice.
        _, summary_sha_2 = _verify_corpus(data, enforce_commitment=True, run_mutations=True)
        receipt_2 = _receipt(verifier_sha, summary_sha_2)
        if receipt != receipt_2:
            raise VerificationError("non-deterministic normalized receipt")
        if args.receipt:
            args.receipt.write_bytes(receipt)
        sys.stdout.buffer.write(receipt)
    except (OSError, VerificationError) as exc:
        print(f"PARADOX-002A-I FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
