"""Independent test-only oracle for SCI-REFERENCE-ASSURANCE-001.

This intentionally uses only the Python standard library and reconstructs the
v1 commitment serialization independently of the Rust implementation. It is
not a production validator or evidence-authenticity verifier.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import struct
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "reference-assurance-package.example.json"
SCHEMA = ROOT / "schema" / "reference-assurance-package.schema.json"
PREFIX = b"symthaea.science_bench_refresh.reference_assurance.v1\0"
DOMAIN = b"package.v1\0"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TOP_FIELDS = (
    "schema", "purpose", "package_id", "validator_source_revision",
    "rubric_sha256", "scope_policy_sha256", "claims", "sources", "spans",
    "adjudicators", "initial_adjudications", "resolutions",
)
CLAIM_FIELDS = ("claim_id", "text_sha256", "material", "scope")
SCOPE_FIELDS = (
    "domain", "jurisdiction", "target_date", "valid_from", "valid_until",
    "unit_context", "time_independent",
)
SOURCE_FIELDS = (
    "source_id", "snapshot_sha256", "source_identity_sha256",
    "ancestry_group_sha256", "canonicalization_profile", "publication_date",
    "retrieved_date", "lifecycle",
)
SPAN_FIELDS = (
    "span_id", "source_id", "start_offset", "end_offset", "offset_unit",
    "span_sha256",
)
ADJUDICATOR_FIELDS = (
    "adjudicator_id_sha256", "independence_group_sha256", "role",
    "blind_to_system_identity", "blind_to_candidate_identity",
    "blind_to_judge_output", "conflict_of_interest_declared",
    "conflict_disclosure_sha256",
)
JUDGMENT_FIELDS = (
    "record_id", "claim_id", "adjudicator_id_sha256", "rubric_sha256",
    "decision", "confidence_millionths", "abstention_reason",
    "evidence_span_ids", "rationale_sha256",
)
RESOLUTION_FIELDS = (
    "resolution_id", "claim_id", "decision", "prior_record_ids",
    "evidence_span_ids", "rationale_sha256",
)


def ordered(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {key: row[key] for key in fields}


def commitment_payload(package: dict[str, Any]) -> bytes:
    fields = ordered(package, TOP_FIELDS)
    fields["claims"] = [
        {**ordered(claim, CLAIM_FIELDS), "scope": ordered(claim["scope"], SCOPE_FIELDS)}
        for claim in package["claims"]
    ]
    fields["sources"] = [ordered(row, SOURCE_FIELDS) for row in package["sources"]]
    fields["spans"] = [ordered(row, SPAN_FIELDS) for row in package["spans"]]
    fields["adjudicators"] = [
        ordered(row, ADJUDICATOR_FIELDS) for row in package["adjudicators"]
    ]
    fields["initial_adjudications"] = [
        ordered(row, JUDGMENT_FIELDS) for row in package["initial_adjudications"]
    ]
    fields["resolutions"] = [
        ordered(row, RESOLUTION_FIELDS) for row in package["resolutions"]
    ]
    return json.dumps(
        fields, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def framed_commitment(payload: bytes) -> str:
    framed = (
        PREFIX + struct.pack(">Q", len(DOMAIN)) + DOMAIN
        + struct.pack(">Q", len(payload)) + payload
    )
    return hashlib.sha256(framed).hexdigest()


def verify_commitment(package: dict[str, Any]) -> bool:
    return framed_commitment(commitment_payload(package)) == package.get(
        "package_commitment_sha256"
    )


def valid_digest(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def valid_date(value: Any) -> bool:
    if not isinstance(value, str) or DATE.fullmatch(value) is None:
        return False
    year, month, day = (int(part) for part in value.split("-"))
    if not 1 <= month <= 12:
        return False
    month_days = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                  31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return 1 <= day <= month_days[month - 1]


def validate_package(package: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    example = package.get("purpose") == "example_only"

    claims = package.get("claims", [])
    sources = package.get("sources", [])
    spans = package.get("spans", [])
    adjudicators = package.get("adjudicators", [])
    judgments = package.get("initial_adjudications", [])
    resolutions = package.get("resolutions", [])

    claim_ids = [row.get("claim_id") for row in claims]
    source_ids = [row.get("source_id") for row in sources]
    span_ids = [row.get("span_id") for row in spans]
    adjudicator_ids = [row.get("adjudicator_id_sha256") for row in adjudicators]
    judgment_ids = [row.get("record_id") for row in judgments]

    for name, values in (
        ("claim", claim_ids), ("source", source_ids), ("span", span_ids),
        ("adjudicator", adjudicator_ids), ("adjudication", judgment_ids),
    ):
        if len(values) != len(set(values)):
            issues.append(f"duplicate {name} identifier")

    known_claims = set(claim_ids)
    known_sources = set(source_ids)
    known_spans = set(span_ids)
    known_adjudicators = {row["adjudicator_id_sha256"]: row for row in adjudicators}
    known_judgments = {row["record_id"]: row for row in judgments}

    for claim in claims:
        scope = claim["scope"]
        if not valid_digest(claim["text_sha256"]):
            issues.append(f"claim {claim['claim_id']} has invalid digest")
        if not scope["domain"].strip():
            issues.append(f"claim {claim['claim_id']} has no domain")
        if scope["time_independent"]:
            if any(scope[key] is not None for key in ("target_date", "valid_from", "valid_until")):
                issues.append(f"claim {claim['claim_id']} has temporal bounds but is time-independent")
        elif not valid_date(scope["target_date"]):
            issues.append(f"claim {claim['claim_id']} has invalid target date")
        for key in ("target_date", "valid_from", "valid_until"):
            if scope[key] is not None and not valid_date(scope[key]):
                issues.append(f"claim {claim['claim_id']} has invalid {key}")
        target = scope["target_date"]
        if target and scope["valid_from"] and target < scope["valid_from"]:
            issues.append(f"claim {claim['claim_id']} target precedes validity")
        if target and scope["valid_until"] and target > scope["valid_until"]:
            issues.append(f"claim {claim['claim_id']} target exceeds validity")

    for source in sources:
        for key in ("snapshot_sha256", "source_identity_sha256"):
            if not valid_digest(source[key]):
                issues.append(f"source {source['source_id']} invalid {key}")
        if source["ancestry_group_sha256"] is not None and not valid_digest(source["ancestry_group_sha256"]):
            issues.append(f"source {source['source_id']} invalid ancestry digest")
        if not valid_date(source["retrieved_date"]):
            issues.append(f"source {source['source_id']} invalid retrieval date")
        if source["publication_date"] is not None and not valid_date(source["publication_date"]):
            issues.append(f"source {source['source_id']} invalid publication date")

    source_by_id = {row["source_id"]: row for row in sources}
    span_by_id = {row["span_id"]: row for row in spans}
    for span in spans:
        if span["source_id"] not in known_sources:
            issues.append(f"span {span['span_id']} has orphan source")
        if span["offset_unit"] != "byte" or span["start_offset"] >= span["end_offset"]:
            issues.append(f"span {span['span_id']} has invalid byte range")
        if not valid_digest(span["span_sha256"]):
            issues.append(f"span {span['span_id']} has invalid digest")

    group_by_adjudicator: dict[str, str] = {}
    for adjudicator in adjudicators:
        aid = adjudicator["adjudicator_id_sha256"]
        group = adjudicator["independence_group_sha256"]
        group_by_adjudicator[aid] = group
        if not valid_digest(aid) or not valid_digest(group):
            issues.append("adjudicator identity/group digest invalid")
        if not valid_digest(adjudicator["conflict_disclosure_sha256"]):
            issues.append(f"adjudicator {aid} has invalid COI disclosure")
        if not adjudicator["role"].strip():
            issues.append(f"adjudicator {aid} has empty role")
        if not example and not all(adjudicator[key] for key in (
            "blind_to_system_identity", "blind_to_candidate_identity",
            "blind_to_judge_output", "conflict_of_interest_declared"
        )):
            issues.append(f"adjudicator {aid} failed blinding/COI declaration")

    per_claim_adjudicators: dict[str, set[str]] = {}
    per_claim_groups: dict[str, set[str]] = {}
    seen_claim_voters: set[tuple[str, str]] = set()
    decisions: dict[str, set[str]] = {}
    for record in judgments:
        claim_id = record["claim_id"]
        aid = record["adjudicator_id_sha256"]
        if claim_id not in known_claims:
            issues.append(f"adjudication {record['record_id']} has orphan claim")
            continue
        if aid not in known_adjudicators:
            issues.append(f"adjudication {record['record_id']} has orphan adjudicator")
            continue
        if (claim_id, aid) in seen_claim_voters:
            issues.append(f"claim {claim_id} has duplicate voter")
        seen_claim_voters.add((claim_id, aid))
        per_claim_adjudicators.setdefault(claim_id, set()).add(aid)
        per_claim_groups.setdefault(claim_id, set()).add(group_by_adjudicator[aid])
        if record["rubric_sha256"] != package["rubric_sha256"] or not valid_digest(record["rationale_sha256"]):
            issues.append(f"adjudication {record['record_id']} rubric/rationale mismatch")
        confidence = record["confidence_millionths"]
        if confidence is not None and not 0 <= confidence <= 1_000_000:
            issues.append(f"adjudication {record['record_id']} invalid confidence")
        if record["decision"] == "abstain":
            if not record["abstention_reason"]:
                issues.append(f"adjudication {record['record_id']} abstention lacks reason")
        elif record["abstention_reason"] is not None:
            issues.append(f"adjudication {record['record_id']} has stray abstention reason")
        for span_id in record["evidence_span_ids"]:
            if span_id not in known_spans:
                issues.append(f"adjudication {record['record_id']} has orphan span")
        if record["decision"] not in {
            "abstain", "insufficient_evidence", "scope_ambiguous",
            "not_applicable", "not_evaluable"
        } and not record["evidence_span_ids"]:
            issues.append(f"adjudication {record['record_id']} requires evidence")
        if record["decision"] in {"supported_within_scope", "contradicted_within_scope"}:
            target = next(row["scope"]["target_date"] for row in claims if row["claim_id"] == claim_id)
            for span_id in record["evidence_span_ids"]:
                span = span_by_id.get(span_id)
                source = source_by_id.get(span["source_id"]) if span else None
                if source and source["lifecycle"] in {"deleted", "retracted"}:
                    issues.append(f"adjudication {record['record_id']} uses withdrawn source")
                if source and target and source["publication_date"] and source["publication_date"] > target:
                    issues.append(f"adjudication {record['record_id']} uses future source")
        if record["decision"] != "abstain":
            decisions.setdefault(claim_id, set()).add(record["decision"])

    if not example:
        for claim in claims:
            if claim["material"]:
                if len(per_claim_adjudicators.get(claim["claim_id"], set())) < 3:
                    issues.append(f"material claim {claim['claim_id']} lacks three adjudicators")
                if len(per_claim_groups.get(claim["claim_id"], set())) < 3:
                    issues.append(f"material claim {claim['claim_id']} lacks three declared independence groups")

    for event in resolutions:
        if event["claim_id"] not in known_claims:
            issues.append(f"resolution {event['resolution_id']} has orphan claim")
        if len(event["prior_record_ids"]) < 2:
            issues.append(f"resolution {event['resolution_id']} lacks prior judgments")
        for record_id in event["prior_record_ids"]:
            previous = known_judgments.get(record_id)
            if previous is None or previous["claim_id"] != event["claim_id"]:
                issues.append(f"resolution {event['resolution_id']} has invalid predecessor")
        for span_id in event["evidence_span_ids"]:
            if span_id not in known_spans:
                issues.append(f"resolution {event['resolution_id']} has orphan span")

    if not verify_commitment(package):
        issues.append("package commitment mismatch")
    return issues


class ReferenceAssuranceOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_schema_and_example_parse(self) -> None:
        self.assertEqual(self.schema["additionalProperties"], False)
        self.assertIn("initial_adjudications", self.schema["required"])
        self.assertEqual(validate_package(copy.deepcopy(self.fixture)), [])

    def test_independent_commitment_matches_fixture(self) -> None:
        self.assertEqual(
            framed_commitment(commitment_payload(self.fixture)),
            self.fixture["package_commitment_sha256"],
        )

    def test_original_disagreement_is_preserved(self) -> None:
        values = {row["decision"] for row in self.fixture["initial_adjudications"]}
        self.assertEqual(len(values), 2)
        self.assertEqual(len(self.fixture["initial_adjudications"]), 3)

    def test_duplicate_adjudicator_vote_is_detected(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["initial_adjudications"][1]["adjudicator_id_sha256"] = value["initial_adjudications"][0]["adjudicator_id_sha256"]
        self.assertTrue(any("duplicate voter" in issue for issue in validate_package(value)))

    def test_retracted_source_cannot_support_current_claim(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["sources"][0]["lifecycle"] = "retracted"
        self.assertTrue(any("withdrawn source" in issue for issue in validate_package(value)))

    def test_future_source_cannot_support_past_claim(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["sources"][0]["publication_date"] = "2030-01-01"
        self.assertTrue(any("future source" in issue for issue in validate_package(value)))

    def test_scope_outside_validity_is_detected(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["claims"][0]["scope"]["valid_until"] = "2020-01-01"
        self.assertTrue(any("target exceeds validity" in issue for issue in validate_package(value)))

    def test_mirror_pages_share_one_ancestry_group(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["sources"][1]["ancestry_group_sha256"] = value["sources"][0]["ancestry_group_sha256"]
        value["package_commitment_sha256"] = framed_commitment(commitment_payload(value))
        ancestry_groups = {
            source["ancestry_group_sha256"]
            for source in value["sources"]
            if source["ancestry_group_sha256"] is not None
        }
        self.assertEqual(len(ancestry_groups), 1)
        self.assertEqual(validate_package(value), [])

    def test_commitment_mutation_is_detected(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["claims"][0]["text_sha256"] = "0" * 64
        self.assertTrue(any("commitment mismatch" in issue for issue in validate_package(value)))

    def test_abstention_requires_explanation(self) -> None:
        value = copy.deepcopy(self.fixture)
        value["initial_adjudications"][0]["decision"] = "abstain"
        value["initial_adjudications"][0]["abstention_reason"] = None
        self.assertTrue(any("abstention lacks reason" in issue for issue in validate_package(value)))


if __name__ == "__main__":
    unittest.main()
