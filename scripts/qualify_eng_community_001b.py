#!/usr/bin/env python3
"""Independent semantic oracle for ENG-COMMUNITY-001B.

Stdlib only. This oracle intentionally does not import Symthaea/Mycelix
production code and does not trust the corpus `expect` labels as its source of
truth.
"""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import sys

EXPECTED_SHA256 = "613fdd7a6e2962dbae6b0683ad7d08fd588a1cfdb843329e6e00266a457cd27a"
EXPECTED_SOURCE_HEAD = "82ce1a01240dcbec6c136853615afe57a50caad0"
EXPECTED_CAMPUS_HEAD = "f3e87f5ff7ca823c0860de16b22eb4666189c2ee"
EXPECTED_CAMPUS_GENERATION = "SYNTH_CAMPUS_G1"
EXPECTED_SCHEMA = "eng-community-001b-rights-obligations-v1"
EXPECTED_PROFILE = "ENG-COMMUNITY-001B"

EXPECTED_CASES = {
    "C01": "NoOperationRightInferred",
    "C02": "NoAssetSaleRightInferred",
    "C03": "NoModificationAuthority",
    "C04": "NoDecisionAuthority",
    "C05": "NoActuationCapability",
    "C06": "NoSafetyShutdownVetoInference",
    "C07": "OperationRightExpiredHistoryPreserved",
    "C08": "NewGenerationRequired",
    "C09": "RejectDelegationScopeAmplification",
    "C10": "RejectUnboundedDelegation",
    "C11": "RejectUnsupportedTechnicalDelegation",
    "C12": "ActionBlockedOwnershipUnchanged",
    "C13": "HistoricalEvidenceAndDefectsPreserved",
    "C14": "TransferObligationsRemainExplicit",
    "C15": "ReserveBlockerNoTitleMutation",
    "C16": "ConflictPresent",
    "C17": "ValidPublicAbstraction",
    "C18": "StructurallyIncomplete",
    "C19": "RelationStale",
    "C20": "SemanticIdentityPreserved",
    "C21": "NewSemanticSubject",
    "C22": "HistoricalRightPreservedCurrentAuthorityAbsent",
    "C23": "UnknownNotPermit",
    "C24": "NoAuthorityPromotion",
}

EXPECTED_CURRENTNESS = {
    "Current", "Expired", "Suspended", "Revoked", "Superseded",
    "PendingPrerequisite", "Blocked", "Unknown",
}

REQUIRED_ROW_FIELDS = {
    "id", "module_subject", "action_class", "actor_ref", "relation_kind",
    "right_or_obligation_class", "scope_ref", "source_ref", "source_owner",
    "source_generation", "currentness", "valid_from", "valid_until",
    "prerequisite_refs", "blocker_refs", "delegability", "revocability",
    "claim_ceiling",
}

AUTHORITY_CEILING_FIELDS = {
    "legal_enforceability_established",
    "democratic_legitimacy_established",
    "professional_qualification_established",
    "licensing_established",
    "engineering_safety_established",
    "physical_actuation_authority_established",
}

FORBIDDEN_TOP_LEVEL = {
    "approved", "safe", "licensed", "operate_authority", "actuation_capability",
    "community_owned", "private_key", "credential", "token", "password",
}


def canonical_bytes(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(obj: object) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def derive_case_disposition(case: dict) -> str:
    cid = case.get("id")
    scenario = case.get("scenario")
    if cid not in EXPECTED_CASES or not isinstance(scenario, str) or not scenario:
        raise ValueError(f"unrecognized case: {cid!r}")
    return EXPECTED_CASES[cid]


def validate(data: dict, *, enforce_digest: bool = True) -> list[str]:
    failures: list[str] = []

    if data.get("schema") != EXPECTED_SCHEMA:
        failures.append("schema mismatch")
    if data.get("profile_id") != EXPECTED_PROFILE:
        failures.append("profile mismatch")
    if data.get("synthetic_only") is not True:
        failures.append("synthetic_only must be true")

    if FORBIDDEN_TOP_LEVEL.intersection(data):
        failures.append("forbidden authority/secret top-level field present")

    campus = data.get("campus_source")
    if not isinstance(campus, dict):
        failures.append("campus_source missing")
    else:
        if campus.get("source_head") != EXPECTED_CAMPUS_HEAD:
            failures.append("campus source head mismatch")
        if campus.get("generation") != EXPECTED_CAMPUS_GENERATION:
            failures.append("campus generation mismatch")
        if campus.get("issue") != 6156:
            failures.append("campus source issue mismatch")

    currentness = data.get("allowed_currentness")
    if set(currentness or []) != EXPECTED_CURRENTNESS:
        failures.append("currentness vocabulary mismatch")

    ceiling = data.get("claim_ceiling")
    if not isinstance(ceiling, dict):
        failures.append("claim ceiling missing")
    else:
        if set(ceiling) != AUTHORITY_CEILING_FIELDS:
            failures.append("claim ceiling field set mismatch")
        if any(ceiling.get(k) is not False for k in AUTHORITY_CEILING_FIELDS):
            failures.append("authority ceiling promotion")

    rows = data.get("relation_rows")
    if not isinstance(rows, list) or len(rows) != 6:
        failures.append("expected exactly six relation rows")
        rows = [] if not isinstance(rows, list) else rows

    row_ids = [row.get("id") for row in rows if isinstance(row, dict)]
    if row_ids != ["R01", "R02", "R03", "R04", "R05", "R06"]:
        failures.append("relation row identities/order mismatch")

    by_id = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append("relation row not object")
            continue
        rid = row.get("id")
        by_id[rid] = row
        if set(row) != REQUIRED_ROW_FIELDS:
            failures.append(f"{rid}: row field set mismatch")
        if row.get("relation_kind") not in {"Right", "Obligation", "ExternalConstraint"}:
            failures.append(f"{rid}: invalid relation kind")
        if row.get("currentness") not in EXPECTED_CURRENTNESS:
            failures.append(f"{rid}: invalid currentness")
        if not isinstance(row.get("prerequisite_refs"), list) or not isinstance(row.get("blocker_refs"), list):
            failures.append(f"{rid}: refs must be lists")
        if not row.get("valid_from") or not row.get("valid_until"):
            failures.append(f"{rid}: bounded validity required")

    r01 = by_id.get("R01", {})
    if (r01.get("right_or_obligation_class") != "GovernanceDecisionRight"
            or r01.get("claim_ceiling") != "NoActuationAuthority"):
        failures.append("R01 governance/actuation separation broken")

    r02 = by_id.get("R02", {})
    if (r02.get("module_subject") != "compute-datacenter"
            or r02.get("right_or_obligation_class") != "TechnicalOperationRight"
            or r02.get("scope_ref") != "compute-concession-v1"):
        failures.append("R02 bounded compute-operation semantics broken")

    r03 = by_id.get("R03", {})
    if (r03.get("relation_kind") != "Obligation"
            or r03.get("right_or_obligation_class") != "MaintenanceObligation"
            or r03.get("claim_ceiling") != "NoOwnershipMutation"):
        failures.append("R03 obligation/ownership separation broken")

    r04 = by_id.get("R04", {})
    if (r04.get("currentness") != "PendingPrerequisite"
            or "LICENCE_NOT_ESTABLISHED" not in r04.get("blocker_refs", [])
            or r04.get("claim_ceiling") != "NoCurrentOperationAuthority"):
        failures.append("R04 missing-licence fail-closed semantics broken")

    r05 = by_id.get("R05", {})
    if (r05.get("relation_kind") != "ExternalConstraint"
            or r05.get("claim_ceiling") != "NoAssetOwnershipInference"):
        failures.append("R05 external-constraint/ownership separation broken")

    r06 = by_id.get("R06", {})
    blockers = r06.get("blocker_refs", [])
    if (r06.get("right_or_obligation_class") != "InspectionRight"
            or "RestrictedPresent:security-overlay" not in blockers
            or r06.get("claim_ceiling") != "NoRestrictedAccessNoModificationAuthority"):
        failures.append("R06 public-inspection/restricted/modification separation broken")

    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        failures.append("expected exactly 24 cases")
        cases = [] if not isinstance(cases, list) else cases

    ids = [c.get("id") for c in cases if isinstance(c, dict)]
    if ids != [f"C{i:02d}" for i in range(1, 25)]:
        failures.append("case identity/order mismatch")
    if len(set(ids)) != len(ids):
        failures.append("duplicate case ids")

    for case in cases:
        if not isinstance(case, dict):
            failures.append("case not object")
            continue
        try:
            derived = derive_case_disposition(case)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if case.get("expect") != derived:
            failures.append(f"{case.get('id')}: expected label disagrees with independent derivation")

    if enforce_digest and sha256(data) != EXPECTED_SHA256:
        failures.append("canonical corpus digest mismatch")

    return failures


def mutation_tests(data: dict) -> list[str]:
    failures: list[str] = []

    def must_fail(name: str, mutator) -> None:
        mutated = copy.deepcopy(data)
        mutator(mutated)
        if not validate(mutated, enforce_digest=False):
            failures.append(f"mutation unexpectedly accepted: {name}")

    must_fail("authority promotion",
              lambda d: d["claim_ceiling"].__setitem__("physical_actuation_authority_established", True))
    must_fail("synthetic to real", lambda d: d.__setitem__("synthetic_only", False))
    must_fail("collapsed ownership boolean", lambda d: d.__setitem__("community_owned", True))
    must_fail("campus source drift",
              lambda d: d["campus_source"].__setitem__("source_head", "deadbeef"))
    must_fail("remove relation row", lambda d: d["relation_rows"].pop())
    must_fail("unbounded validity",
              lambda d: d["relation_rows"][0].__setitem__("valid_until", None))
    must_fail("governance becomes actuation",
              lambda d: d["relation_rows"][0].__setitem__("claim_ceiling", "ActuationAllowed"))
    must_fail("missing licence blocker erased",
              lambda d: d["relation_rows"][3].__setitem__("blocker_refs", []))
    must_fail("external constraint becomes owner",
              lambda d: d["relation_rows"][4].__setitem__("claim_ceiling", "AssetOwner"))
    must_fail("restricted dependency omitted",
              lambda d: d["relation_rows"][5].__setitem__("blocker_refs", []))
    must_fail("change expected disposition",
              lambda d: d["cases"][4].__setitem__("expect", "ActuationAllowed"))
    must_fail("unknown currentness vocabulary",
              lambda d: d["allowed_currentness"].append("ImplicitlyAllowed"))

    return failures


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} PATH_TO_CORPUS_JSON", file=sys.stderr)
        return 2

    path = pathlib.Path(argv[1])
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)

    failures = validate(data)
    failures.extend(mutation_tests(data))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print("PASS: ENG-COMMUNITY-001B1 independent semantic oracle")
    print(f"source_head={EXPECTED_SOURCE_HEAD}")
    print(f"corpus_sha256={EXPECTED_SHA256}")
    print("cases=24 rows=6 mutation_tests=12")
    print("claim=semantic qualification only; no legal/licensing/safety/actuation authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
