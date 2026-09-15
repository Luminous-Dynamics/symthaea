#!/usr/bin/env python3
"""Independent standard-library campaign.v1 conformance oracle.

This oracle deliberately treats ASSURE-000/001/002A digests as upstream
qualified inputs. It independently owns only ASSURE-002 campaign framing.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
VECTORS = HERE.parent / "vectors" / "campaign_v1.json"

TIER_RANK = {
    "structural": 0,
    "observed": 1,
    "causally-supported": 2,
    "functionally-supported": 3,
}


def field(label: str, value: str) -> str:
    return f"{label} {len(value.encode('utf-8'))}:{value}\n"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_digest(value: str) -> None:
    assert len(value) == 64
    assert all(c in "0123456789abcdef" for c in value)


def semantic(data: dict, key: str) -> dict:
    return data["upstream"]["semantics"][key]


def campaign_plan(data: dict, vector: dict, *, overrides: dict | None = None) -> str:
    overrides = overrides or {}
    upstream = data["upstream"]
    plan = upstream["plan"]
    subject = upstream["subject"]
    claim = upstream["claim"]

    claim_digest = overrides.get("claim_digest", claim["expected_digest"])
    manifest_id = overrides.get("manifest_id", subject["expected_manifest_id"])
    core_subject_id = overrides.get("core_subject_id", subject["expected_core_subject_id"])
    nonce = overrides.get("campaign_nonce", vector["campaign_nonce"])

    criteria = []
    for entry in plan["support_criteria"]:
        sem = semantic(data, entry["semantic"])
        criteria.append((entry["tier"], sem["semantic_id"], sem["expected_digest"]))
    criteria.sort(key=lambda item: (TIER_RANK[item[0]], item[1].encode("utf-8")))

    control_keys = [plan["base_control"], vector["extra_control_semantic"]]
    controls = [
        (semantic(data, key)["semantic_id"], semantic(data, key)["expected_digest"])
        for key in control_keys
    ]
    controls.sort(key=lambda item: item[0].encode("utf-8"))

    replacement = overrides.get("extra_control_replacement")
    if replacement is not None:
        target_id = semantic(data, vector["extra_control_semantic"])["semantic_id"]
        controls = [
            (target_id, replacement) if item[0] == target_id else item
            for item in controls
        ]

    out = "symthaea-assurance-campaign-plan-v1\n"
    for label, value in [
        ("schema", "symthaea.assurance.campaign.v1"),
        ("plan-key", plan["plan_key"]),
        ("campaign-nonce", nonce),
        ("claim", claim_digest),
        ("assure-001-subject", manifest_id),
        ("core-subject", core_subject_id),
        ("core-plan", plan["expected_core_plan_digest"]),
        ("maximum-support", plan["maximum_support"]),
        ("reproduction-requirement", plan["reproduction_requirement"]),
    ]:
        out += field(label, value)

    requirements = sorted(plan["evidence_requirements"], key=lambda x: x.encode("utf-8"))
    out += field("evidence-kind-count", str(len(requirements)))
    for kind in requirements:
        out += field("evidence-kind", kind)
        out += field("evidence-kind-semantic-commitment", "")

    out += field("support-criterion-count", str(len(criteria)))
    for tier, semantic_id, semantic_digest in criteria:
        out += field("support-criterion-tier", tier)
        out += field("support-criterion-id", semantic_id)
        out += field("support-criterion-commitment", semantic_digest)

    for label, values in [
        ("control", controls),
        ("failure-condition", []),
        ("contradiction-condition", []),
        ("inconclusive-condition", []),
        ("invalidation-condition", []),
    ]:
        out += field(f"{label}-count", str(len(values)))
        for semantic_id, semantic_digest in values:
            out += field(f"{label}-id", semantic_id)
            out += field(f"{label}-commitment", semantic_digest)
    return digest(out)


def ordering(
    data: dict,
    statement_digest: str,
    epoch: int,
    sequence: int,
    external_receipt_digest: str,
    *,
    source: str | None = None,
    profile_digest: str | None = None,
) -> str:
    ordering_input = data["upstream"]["ordering"]
    profile = semantic(data, ordering_input["validation_semantic"])
    out = "symthaea-assurance-ordering-receipt-v1\n"
    for label, value in [
        ("source", source or ordering_input["source"]),
        ("validation-profile-id", profile["semantic_id"]),
        ("validation-profile-commitment", profile_digest or profile["expected_digest"]),
        ("epoch", str(epoch)),
        ("sequence", str(sequence)),
        ("statement", statement_digest),
        ("external-receipt", external_receipt_digest),
    ]:
        out += field(label, value)
    return digest(out)


def empty_root(vector: dict, plan_digest: str, predecessor: str = "") -> str:
    out = "symthaea-assurance-empty-evidence-root-v1\n"
    out += field("campaign-nonce", vector["campaign_nonce"])
    out += field("plan", plan_digest)
    out += field("predecessor-registration", predecessor)
    return digest(out)


def registration_statement(data: dict, vector: dict, plan_digest: str, root: str, predecessor: str = "") -> str:
    upstream = data["upstream"]
    out = "symthaea-assurance-registration-statement-v1\n"
    for label, value in [
        ("plan", plan_digest),
        ("campaign-nonce", vector["campaign_nonce"]),
        ("claim", upstream["claim"]["expected_digest"]),
        ("assure-001-subject", upstream["subject"]["expected_manifest_id"]),
        ("issuer", "registrar-a"),
        ("predecessor-registration", predecessor),
        ("pre-evidence-count", "0"),
        ("pre-evidence-root", root),
    ]:
        out += field(label, value)
    return digest(out)


def preregistration_receipt(statement_digest: str, ordering_digest: str) -> str:
    out = "symthaea-assurance-preregistration-receipt-v1\n"
    out += field("statement", statement_digest)
    out += field("ordering", ordering_digest)
    return digest(out)


def withdrawal_statement(vector: dict, registration_digest: str) -> str:
    out = "symthaea-assurance-registration-withdrawal-statement-v1\n"
    out += field("campaign-nonce", vector["campaign_nonce"])
    out += field("target-registration", registration_digest)
    out += field("reason", "operator-withdrawal")
    return digest(out)


def withdrawal(statement_digest: str, ordering_digest: str) -> str:
    out = "symthaea-assurance-registration-withdrawal-v1\n"
    out += field("statement", statement_digest)
    out += field("ordering", ordering_digest)
    return digest(out)


def commitment_statement(vector: dict, registration_digest: str, plan_digest: str, evidence_digest: str) -> str:
    out = "symthaea-assurance-evidence-commitment-statement-v1\n"
    out += field("registration", registration_digest)
    out += field("campaign-nonce", vector["campaign_nonce"])
    out += field("plan", plan_digest)
    out += field("evidence", evidence_digest)
    return digest(out)


def admission_statement(
    vector: dict,
    registration_digest: str,
    plan_digest: str,
    ordinal: int,
    evidence_digest: str,
    prior_root: str,
    previous_admission_ordering: str,
    commitment_ordering: str,
) -> str:
    out = "symthaea-assurance-evidence-admission-statement-v1\n"
    for label, value in [
        ("registration", registration_digest),
        ("campaign-nonce", vector["campaign_nonce"]),
        ("plan", plan_digest),
        ("ordinal", str(ordinal)),
        ("evidence", evidence_digest),
        ("prior-root", prior_root),
        ("previous-admission-ordering", previous_admission_ordering),
        ("commitment-ordering", commitment_ordering),
    ]:
        out += field(label, value)
    return digest(out)


def evidence_root(
    previous: str,
    ordinal: int,
    evidence_digest: str,
    commitment_ordering: str,
    admission_ordering: str,
) -> str:
    out = "symthaea-assurance-evidence-root-step-v1\n"
    for label, value in [
        ("previous", previous),
        ("ordinal", str(ordinal)),
        ("evidence", evidence_digest),
        ("commitment-ordering", commitment_ordering),
        ("admission-ordering", admission_ordering),
    ]:
        out += field(label, value)
    return digest(out)


def evidence_admission(
    vector: dict,
    registration_digest: str,
    plan_digest: str,
    ordinal: int,
    evidence_digest: str,
    commitment_ordering: str,
    admission_ordering: str,
    previous_root: str,
    root: str,
) -> str:
    out = "symthaea-assurance-campaign-evidence-admission-v1\n"
    for label, value in [
        ("registration", registration_digest),
        ("plan", plan_digest),
        ("campaign-nonce", vector["campaign_nonce"]),
        ("ordinal", str(ordinal)),
        ("evidence", evidence_digest),
        ("commitment-ordering", commitment_ordering),
        ("admission-ordering", admission_ordering),
        ("previous-evidence-root", previous_root),
        ("evidence-root", root),
    ]:
        out += field(label, value)
    return digest(out)


def verify_vector(data: dict, vector: dict) -> dict:
    expected = vector["expected"]
    plan_digest = campaign_plan(data, vector)
    assert plan_digest == expected["campaign_plan_digest"], vector["name"]

    root0 = empty_root(vector, plan_digest)
    assert root0 == expected["empty_evidence_root"], vector["name"]

    reg_statement = registration_statement(data, vector, plan_digest, root0)
    assert reg_statement == expected["registration_statement_digest"], vector["name"]
    reg_ordering = ordering(data, reg_statement, vector["epoch"], vector["registration_sequence"], "e" * 64)
    assert reg_ordering == expected["registration_ordering_digest"], vector["name"]
    registration = preregistration_receipt(reg_statement, reg_ordering)
    assert registration == expected["preregistration_receipt_digest"], vector["name"]

    withdrawal_stmt = withdrawal_statement(vector, registration)
    assert withdrawal_stmt == expected["withdrawal_statement_digest"], vector["name"]
    withdrawal_ordering = ordering(
        data, withdrawal_stmt, vector["epoch"], vector["registration_sequence"] + 5, "f" * 64
    )
    assert withdrawal_ordering == expected["withdrawal_ordering_digest"], vector["name"]
    withdrawal_digest = withdrawal(withdrawal_stmt, withdrawal_ordering)
    assert withdrawal_digest == expected["withdrawal_digest"], vector["name"]

    evidence1 = vector["evidence"][0]["digest"]
    commitment1 = commitment_statement(vector, registration, plan_digest, evidence1)
    assert commitment1 == expected["commitment_statement_1_digest"], vector["name"]
    commitment_ordering1 = ordering(
        data, commitment1, vector["epoch"], vector["registration_sequence"] + 1, "1" * 64
    )
    assert commitment_ordering1 == expected["commitment_ordering_1_digest"], vector["name"]
    admission_stmt1 = admission_statement(
        vector, registration, plan_digest, 1, evidence1, root0, reg_ordering, commitment_ordering1
    )
    assert admission_stmt1 == expected["admission_statement_1_digest"], vector["name"]
    admission_ordering1 = ordering(
        data, admission_stmt1, vector["epoch"], vector["registration_sequence"] + 2, "2" * 64
    )
    assert admission_ordering1 == expected["admission_ordering_1_digest"], vector["name"]
    root1 = evidence_root(root0, 1, evidence1, commitment_ordering1, admission_ordering1)
    assert root1 == expected["evidence_root_1"], vector["name"]
    admission1 = evidence_admission(
        vector, registration, plan_digest, 1, evidence1, commitment_ordering1, admission_ordering1, root0, root1
    )
    assert admission1 == expected["evidence_admission_1_digest"], vector["name"]

    evidence2 = vector["evidence"][1]["digest"]
    commitment2 = commitment_statement(vector, registration, plan_digest, evidence2)
    assert commitment2 == expected["commitment_statement_2_digest"], vector["name"]
    commitment_ordering2 = ordering(
        data, commitment2, vector["epoch"], vector["registration_sequence"] + 3, "3" * 64
    )
    assert commitment_ordering2 == expected["commitment_ordering_2_digest"], vector["name"]
    admission_stmt2 = admission_statement(
        vector, registration, plan_digest, 2, evidence2, root1, admission_ordering1, commitment_ordering2
    )
    assert admission_stmt2 == expected["admission_statement_2_digest"], vector["name"]
    admission_ordering2 = ordering(
        data, admission_stmt2, vector["epoch"], vector["registration_sequence"] + 4, "4" * 64
    )
    assert admission_ordering2 == expected["admission_ordering_2_digest"], vector["name"]
    root2 = evidence_root(root1, 2, evidence2, commitment_ordering2, admission_ordering2)
    assert root2 == expected["evidence_root_2"], vector["name"]
    admission2 = evidence_admission(
        vector, registration, plan_digest, 2, evidence2, commitment_ordering2, admission_ordering2, root1, root2
    )
    assert admission2 == expected["evidence_admission_2_digest"], vector["name"]

    return {
        "plan": plan_digest,
        "registration": registration,
        "root0": root0,
        "root1": root1,
        "root2": root2,
        "reg_ordering": reg_ordering,
        "commitment1": commitment1,
        "commitment_ordering1": commitment_ordering1,
        "admission_statement1": admission_stmt1,
        "admission_ordering1": admission_ordering1,
    }


def verify_mutations(data: dict, ascii_vector: dict, baseline: dict) -> None:
    assert campaign_plan(data, ascii_vector, overrides={"campaign_nonce": "campaign-ascii-mutated"}) != baseline["plan"]
    assert campaign_plan(data, ascii_vector, overrides={"claim_digest": "0" * 64}) != baseline["plan"]
    assert campaign_plan(data, ascii_vector, overrides={"manifest_id": "0" * 64}) != baseline["plan"]
    assert campaign_plan(data, ascii_vector, overrides={"core_subject_id": "0" * 64}) != baseline["plan"]

    # The qualified semantic layer proves schema-id/specification/definition
    # drift changes these nested commitment digests; campaign.v1 proves each
    # changed nested digest changes the richer campaign identity.
    for mutation in data["upstream"]["semantic_mutations"].values():
        require_digest(mutation["expected_digest"])
        assert campaign_plan(
            data,
            ascii_vector,
            overrides={"extra_control_replacement": mutation["expected_digest"]},
        ) != baseline["plan"]

    statement = ascii_vector["expected"]["registration_statement_digest"]
    epoch = ascii_vector["epoch"]
    sequence = ascii_vector["registration_sequence"]
    assert ordering(data, statement, epoch, sequence, "e" * 64, source="transparency-log-b") != baseline["reg_ordering"]
    assert ordering(data, statement, epoch, sequence, "e" * 64, profile_digest="0" * 64) != baseline["reg_ordering"]
    assert ordering(data, statement, epoch + 1, sequence, "e" * 64) != baseline["reg_ordering"]
    assert ordering(data, statement, epoch, sequence + 1, "e" * 64) != baseline["reg_ordering"]

    evidence1 = ascii_vector["evidence"][0]["digest"]
    assert commitment_statement(ascii_vector, baseline["registration"], baseline["plan"], "0" * 64) != baseline["commitment1"]
    assert empty_root(ascii_vector, baseline["plan"], "0" * 64) != baseline["root0"]
    assert admission_statement(
        ascii_vector, baseline["registration"], baseline["plan"], 2, evidence1,
        baseline["root0"], baseline["reg_ordering"], baseline["commitment_ordering1"]
    ) != baseline["admission_statement1"]
    assert admission_statement(
        ascii_vector, baseline["registration"], baseline["plan"], 1, evidence1,
        "0" * 64, baseline["reg_ordering"], baseline["commitment_ordering1"]
    ) != baseline["admission_statement1"]
    assert admission_statement(
        ascii_vector, baseline["registration"], baseline["plan"], 1, evidence1,
        baseline["root0"], "0" * 64, baseline["commitment_ordering1"]
    ) != baseline["admission_statement1"]


def main() -> int:
    data = json.loads(VECTORS.read_text(encoding="utf-8"))
    assert data["schema"] == "symthaea.assurance.campaign-vectors.v1"
    assert data["qualified_input_head"] == "9e236b9da1d15ce486d7d6c308ac5f748e8abe30"

    for section in ("subject", "claim", "plan"):
        for key, value in data["upstream"][section].items():
            if key.startswith("expected_") and key.endswith(("digest", "id")):
                require_digest(value)
    for sem in data["upstream"]["semantics"].values():
        require_digest(sem["expected_digest"])

    results = {vector["name"]: verify_vector(data, vector) for vector in data["vectors"]}

    nfc = next(v for v in data["vectors"] if v["name"] == "unicode-nfc")
    nfd = next(v for v in data["vectors"] if v["name"] == "unicode-nfd")
    assert nfc["campaign_nonce"] != nfd["campaign_nonce"]
    assert nfc["campaign_nonce"].encode("utf-8") != nfd["campaign_nonce"].encode("utf-8")
    assert results["unicode-nfc"]["plan"] != results["unicode-nfd"]["plan"]

    integer = next(v for v in data["vectors"] if v["name"] == "integer-boundary")
    assert integer["epoch"] == 18446744073709551615
    assert integer["registration_sequence"] + 5 == 18446744073709551615

    ascii_vector = next(v for v in data["vectors"] if v["name"] == "ascii-baseline")
    verify_mutations(data, ascii_vector, results["ascii-baseline"])

    for name, result in results.items():
        print(f"{name}: plan={result['plan']} root2={result['root2']}")
    print("ASSURE-002G independent campaign.v1 vectors: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
