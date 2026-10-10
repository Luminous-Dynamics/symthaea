#!/usr/bin/env python3
"""Conformance verifier for the SYM-FIN-SYS-003 synthetic oracle v1.

This checks corpus structure and the public/oracle firewall only. It does not
classify financing regimes and is not evidence about any real financial system.
"""

from __future__ import annotations

import hashlib
import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
PUBLIC_PATH = CRATE / "fixtures" / "fin_sys_v1" / "public.json"
ORACLE_PATH = CRATE / "fixtures" / "fin_sys_v1" / "oracle.json"

EXPECTED_PUBLIC_SCHEMA = "symthaea.fin-sys.synthetic-public.v1"
EXPECTED_ORACLE_SCHEMA = "symthaea.fin-sys.synthetic-oracle.v1"
EXPECTED_PROFILE = "minsky-cashflow-v1"

REQUIRED_FIXTURES = {
    "corp_cashflow_cover_001",
    "corp_refinance_gap_001",
    "corp_interest_shortfall_001",
    "household_missing_income_001",
    "sovereign_long_maturity_001",
    "scheme_payout_dependency_001",
    "economy_sector_heterogeneity_001",
    "bank_liquidity_buffer_001",
}

FORBIDDEN_ID_TOKENS = {
    "hedge",
    "speculative",
    "ponzi",
    "fraud",
    "control",
    "expected",
    "pass",
    "fail",
}

FORBIDDEN_PUBLIC_KEYS = {
    "expected_financing_regime",
    "literal_fraud_predicate",
    "construction_facts",
    "expected_fragility",
    "allowed_alternates",
    "rationale",
}

FORBIDDEN_PUBLIC_LABEL_STRINGS = {
    "hedgefinancecandidate",
    "speculativefinancecandidate",
    "ponzifinancecandidate",
    "mixedregime",
    "syntheticliteralponzifraud",
    "positive control",
    "negative control",
}

ALLOWED_REGIMES = {
    "HedgeFinanceCandidate",
    "SpeculativeFinanceCandidate",
    "PonziFinanceCandidate",
    "MixedRegime",
    "Indeterminate",
    "UnsupportedProfile",
}

ALLOWED_FRAUD_PREDICATES = {
    "FalseByConstruction",
    "NotAssessed",
    "SyntheticLiteralPonziFraud",
}


class ConformanceError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ConformanceError(message)


def load_json(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"{path}: invalid JSON: {exc}")
    if not isinstance(value, dict):
        fail(f"{path}: top-level value must be an object")
    return raw, value


def fixture_map(doc: dict, path: Path) -> dict[str, dict]:
    fixtures = doc.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        fail(f"{path}: fixtures must be a non-empty array")

    result: dict[str, dict] = {}
    for index, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            fail(f"{path}: fixture {index} must be an object")
        fixture_id = fixture.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id.strip():
            fail(f"{path}: fixture {index} has invalid fixture_id")
        if fixture_id in result:
            fail(f"{path}: duplicate fixture_id {fixture_id!r}")
        result[fixture_id] = fixture
    return result


def walk_public(value: object, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_PUBLIC_KEYS:
                fail(f"public corpus leaks evaluator-only key {key!r} at {path}")
            walk_public(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_public(child, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.casefold()
        for label in FORBIDDEN_PUBLIC_LABEL_STRINGS:
            if label in lowered:
                fail(f"public corpus leaks evaluator/control label {label!r} at {path}")


def verify_decimal_string(value: object, path: str) -> None:
    if not isinstance(value, str):
        fail(f"{path}: financial observation must be encoded as a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        fail(f"{path}: {value!r} is not a valid decimal string")
    if not parsed.is_finite():
        fail(f"{path}: non-finite numeric values are forbidden")


def verify_public(public: dict, public_fixtures: dict[str, dict]) -> None:
    if public.get("schema") != EXPECTED_PUBLIC_SCHEMA:
        fail("public schema mismatch")
    if public.get("profile_id") != EXPECTED_PROFILE:
        fail("public profile mismatch")

    walk_public(public)

    if set(public_fixtures) != REQUIRED_FIXTURES:
        missing = sorted(REQUIRED_FIXTURES - set(public_fixtures))
        extra = sorted(set(public_fixtures) - REQUIRED_FIXTURES)
        fail(f"public fixture set mismatch; missing={missing}, extra={extra}")

    for fixture_id, fixture in public_fixtures.items():
        lowered_id = fixture_id.casefold()
        leaked = sorted(token for token in FORBIDDEN_ID_TOKENS if token in lowered_id)
        if leaked:
            fail(f"{fixture_id}: fixture id leaks evaluator semantics: {leaked}")

        observations = fixture.get("observations")
        if not isinstance(observations, dict) or not observations:
            fail(f"{fixture_id}: observations must be a non-empty object")
        for key, value in observations.items():
            verify_decimal_string(value, f"{fixture_id}.observations.{key}")

        sources = fixture.get("source_families")
        if not isinstance(sources, list) or not sources:
            fail(f"{fixture_id}: source_families must be a non-empty array")
        if any(not isinstance(value, str) or not value.strip() for value in sources):
            fail(f"{fixture_id}: source_families contains an invalid identifier")
        if len(sources) != len(set(sources)):
            fail(f"{fixture_id}: duplicate source family")

        missing = fixture.get("missing_coordinates")
        if not isinstance(missing, list):
            fail(f"{fixture_id}: missing_coordinates must be an array")
        if any(not isinstance(value, str) or not value.strip() for value in missing):
            fail(f"{fixture_id}: missing_coordinates contains an invalid identifier")
        if len(missing) != len(set(missing)):
            fail(f"{fixture_id}: duplicate missing coordinate")
        overlap = set(missing) & set(observations)
        if overlap:
            fail(f"{fixture_id}: coordinate cannot be both observed and missing: {sorted(overlap)}")


def verify_oracle(oracle: dict, oracle_fixtures: dict[str, dict]) -> None:
    if oracle.get("schema") != EXPECTED_ORACLE_SCHEMA:
        fail("oracle schema mismatch")
    if oracle.get("profile_id") != EXPECTED_PROFILE:
        fail("oracle profile mismatch")
    if oracle.get("visibility") != "evaluator_only":
        fail("oracle visibility must be evaluator_only")

    for fixture_id, fixture in oracle_fixtures.items():
        regime = fixture.get("expected_financing_regime")
        if regime not in ALLOWED_REGIMES:
            fail(f"{fixture_id}: unsupported oracle regime {regime!r}")
        fraud = fixture.get("literal_fraud_predicate")
        if fraud not in ALLOWED_FRAUD_PREDICATES:
            fail(f"{fixture_id}: unsupported fraud predicate {fraud!r}")


def main() -> int:
    try:
        public_raw, public = load_json(PUBLIC_PATH)
        oracle_raw, oracle = load_json(ORACLE_PATH)
        public_fixtures = fixture_map(public, PUBLIC_PATH)
        oracle_fixtures = fixture_map(oracle, ORACLE_PATH)

        if set(public_fixtures) != set(oracle_fixtures):
            missing_in_oracle = sorted(set(public_fixtures) - set(oracle_fixtures))
            missing_in_public = sorted(set(oracle_fixtures) - set(public_fixtures))
            fail(
                "public/oracle fixture-id mismatch; "
                f"missing_in_oracle={missing_in_oracle}, missing_in_public={missing_in_public}"
            )

        verify_public(public, public_fixtures)
        verify_oracle(oracle, oracle_fixtures)

        print("FIN-SYS synthetic oracle v1: CONFORMANT")
        print(f"fixtures={len(public_fixtures)}")
        print(f"public_sha256={hashlib.sha256(public_raw).hexdigest()}")
        print(f"oracle_sha256={hashlib.sha256(oracle_raw).hexdigest()}")
        return 0
    except (OSError, ConformanceError) as exc:
        print(f"FIN-SYS synthetic oracle v1: NOT CONFORMANT: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
