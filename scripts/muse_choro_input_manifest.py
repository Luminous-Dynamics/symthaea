#!/usr/bin/env python3
"""Fail-closed structural gate for a source-bound Choro input manifest.

This script validates recorded rights decisions and manifest integrity; it does
not make legal determinations. An empty template is valid but never authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE = ROOT / "crates/domains/symthaea-muse/CHORO_SOURCE_INTAKE_V1.json"
DEFAULT_MANIFEST = ROOT / "crates/domains/symthaea-muse/CHORO_APPROVED_INPUT_MANIFEST_V1.json"
DEFAULT_APPROVAL = ROOT / "crates/domains/symthaea-muse/CHORO_APPROVED_INPUT_APPROVAL_V1.json"
MANIFEST_SCHEMA = "muse-choro-approved-input-manifest-v1"
APPROVAL_SCHEMA = "muse-choro-input-manifest-approval-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
USES = {
    "empirical_analysis",
    "transcription",
    "model_training",
    "publish_aggregate_results",
    "redistribute_source_material",
}
REQUIRED_DIMENSIONS = {
    "underlying_composition",
    "score_edition_or_arrangement",
    "scan_or_digital_file",
    "third_party_materials",
}


class ManifestError(ValueError):
    """The manifest is malformed and cannot be safely interpreted."""


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read valid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: JSON root must be an object")
    return value, raw


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a non-empty string")
    return value.strip()


def require_https(value: Any, label: str) -> str:
    url = require_text(value, label)
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ManifestError(f"{label} must be an absolute HTTPS URL")
    return url


def require_date(value: Any, label: str) -> str:
    text = require_text(value, label)
    if not re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}", text):
        raise ManifestError(f"{label} must be an ISO date (YYYY-MM-DD)")
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ManifestError(f"{label} must be an ISO date (YYYY-MM-DD)") from exc
    return text


def require_string_list(value: Any, label: str, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a list")
    items = [require_text(item, f"{label}[]") for item in value]
    if len(items) != len(set(items)):
        raise ManifestError(f"{label} must not contain duplicates")
    if allowed is not None:
        unknown = sorted(set(items) - allowed)
        if unknown:
            raise ManifestError(f"{label} contains unsupported values: {unknown}")
    return items


def evaluate_manifest(
    manifest_path: Path,
    approval_path: Path,
    intake_path: Path = DEFAULT_INTAKE,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    manifest, manifest_bytes = read_json(manifest_path)
    approval, _approval_bytes = read_json(approval_path)
    intake, _intake_bytes = read_json(intake_path)

    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(f"manifest schema must be {MANIFEST_SCHEMA!r}")
    if approval.get("schema") != APPROVAL_SCHEMA:
        raise ManifestError(f"approval schema must be {APPROVAL_SCHEMA!r}")

    intake_sources = intake.get("sources")
    if not isinstance(intake_sources, list):
        raise ManifestError("source intake must contain a sources list")
    intake_source_urls = {
        require_text(source.get("id"), "source intake id"): require_https(
            source.get("source_url"), "source intake source_url"
        )
        for source in intake_sources
        if isinstance(source, dict)
    }
    known_intake_ids = set(intake_source_urls)
    if len(known_intake_ids) != len(intake_sources):
        raise ManifestError("source intake contains malformed or duplicate source ids")

    blockers: list[str] = []
    scope = require_object(manifest.get("scope"), "scope")
    purpose = require_text(scope.get("purpose"), "scope.purpose")
    requested_uses = require_string_list(
        scope.get("requested_uses"), "scope.requested_uses", USES
    )
    if not requested_uses:
        blockers.append("No intended use is declared.")
    commercial_use = scope.get("commercial_use")
    if type(commercial_use) is not bool:
        blockers.append("Commercial/non-commercial purpose is unresolved; set scope.commercial_use explicitly.")
    jurisdictions = require_string_list(scope.get("jurisdictions"), "scope.jurisdictions")
    if not jurisdictions:
        blockers.append("No intended jurisdiction is declared.")
    planned_outputs = require_string_list(scope.get("planned_outputs"), "scope.planned_outputs")
    retention_policy = require_text(scope.get("retention_policy"), "scope.retention_policy")

    declared_dimensions = require_string_list(
        manifest.get("required_rights_dimensions"), "required_rights_dimensions"
    )
    if set(declared_dimensions) != REQUIRED_DIMENSIONS:
        raise ManifestError(
            "required_rights_dimensions must match the four required rights layers exactly"
        )

    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise ManifestError("sources must be a list")
    if not sources:
        blockers.append("No exact source artifacts are in the approved-input manifest.")

    intake_ids_seen: set[str] = set()
    artifact_hashes_seen: set[str] = set()
    reviewed_by: set[str] = set()
    analyzed_ids: list[str] = []
    verified_artifacts: list[dict[str, Any]] = []

    for index, raw_entry in enumerate(sources):
        label = f"sources[{index}]"
        entry = require_object(raw_entry, label)
        source_id = require_text(entry.get("intake_source_id"), f"{label}.intake_source_id")
        if source_id not in known_intake_ids:
            raise ManifestError(f"{label}: intake_source_id is absent from CHORO_SOURCE_INTAKE_V1.json: {source_id}")
        if source_id in intake_ids_seen:
            raise ManifestError(f"duplicate intake_source_id: {source_id}")
        intake_ids_seen.add(source_id)

        source_record_url = require_https(
            entry.get("source_record_url"), f"{label}.source_record_url"
        )
        if source_record_url != intake_source_urls[source_id]:
            raise ManifestError(
                f"{label}.source_record_url does not match the exact intake record for {source_id}"
            )
        require_https(entry.get("source_file_url"), f"{label}.source_file_url")
        relative_artifact = Path(require_text(entry.get("artifact_path"), f"{label}.artifact_path"))
        if relative_artifact.is_absolute() or ".." in relative_artifact.parts:
            raise ManifestError(f"{label}.artifact_path must be a relative path without traversal")
        artifact_sha = require_text(entry.get("artifact_sha256"), f"{label}.artifact_sha256")
        if not SHA256_RE.fullmatch(artifact_sha):
            raise ManifestError(f"{label}.artifact_sha256 must be a lowercase 64-character SHA-256")
        if artifact_sha in artifact_hashes_seen:
            raise ManifestError(f"duplicate artifact SHA-256: {artifact_sha}")
        artifact_hashes_seen.add(artifact_sha)

        artifact_bytes = entry.get("artifact_bytes")
        if type(artifact_bytes) is not int or artifact_bytes <= 0:
            raise ManifestError(f"{label}.artifact_bytes must be a positive integer")
        if artifact_root is None:
            blockers.append(f"{source_id}: provide --artifact-root to verify the exact artifact bytes.")
        else:
            root_resolved = artifact_root.resolve()
            artifact_file = (root_resolved / relative_artifact).resolve()
            try:
                artifact_file.relative_to(root_resolved)
            except ValueError as exc:
                raise ManifestError(f"{label}.artifact_path escapes --artifact-root") from exc
            if not artifact_file.is_file():
                blockers.append(f"{source_id}: exact artifact is missing under --artifact-root.")
            else:
                artifact_content = artifact_file.read_bytes()
                actual_hash = hashlib.sha256(artifact_content).hexdigest()
                if actual_hash != artifact_sha:
                    blockers.append(f"{source_id}: on-disk artifact SHA-256 does not match manifest.")
                if len(artifact_content) != artifact_bytes:
                    blockers.append(f"{source_id}: on-disk artifact byte count does not match manifest.")
        retrieved_on = require_date(entry.get("retrieved_on"), f"{label}.retrieved_on")
        verified_artifacts.append({
            "source_id": source_id,
            "artifact_path": relative_artifact.as_posix(),
            "artifact_sha256": artifact_sha,
            "artifact_bytes": artifact_bytes,
        })

        review = require_object(entry.get("rights_review"), f"{label}.rights_review")
        decision = require_text(review.get("decision"), f"{label}.rights_review.decision")
        if decision != "approved":
            blockers.append(f"{source_id}: rights review decision is {decision!r}, not 'approved'.")

        reviewer = require_text(review.get("reviewer_id"), f"{label}.rights_review.reviewer_id")
        reviewed_by.add(reviewer)
        reviewed_on = require_date(review.get("reviewed_on"), f"{label}.rights_review.reviewed_on")
        if reviewed_on < retrieved_on:
            blockers.append(
                f"{source_id}: rights review date predates retrieval of the exact artifact."
            )
        require_https(review.get("review_evidence_url"), f"{label}.rights_review.review_evidence_url")
        require_https(review.get("license_or_basis_url"), f"{label}.rights_review.license_or_basis_url")
        require_text(review.get("attribution_text"), f"{label}.rights_review.attribution_text")

        reviewed_hash = require_text(
            review.get("reviewed_artifact_sha256"),
            f"{label}.rights_review.reviewed_artifact_sha256",
        )
        if not SHA256_RE.fullmatch(reviewed_hash):
            raise ManifestError(
                f"{label}.rights_review.reviewed_artifact_sha256 must be a lowercase 64-character SHA-256"
            )
        if reviewed_hash != artifact_sha:
            blockers.append(
                f"{source_id}: rights review does not bind the exact artifact SHA-256."
            )

        covered_uses = set(require_string_list(
            review.get("permitted_uses"), f"{label}.rights_review.permitted_uses", USES
        ))
        missing_uses = sorted(set(requested_uses) - covered_uses)
        if missing_uses:
            blockers.append(f"{source_id}: rights review does not permit requested uses {missing_uses}.")
        covered_jurisdictions = set(require_string_list(
            review.get("jurisdictions"), f"{label}.rights_review.jurisdictions"
        ))
        if not covered_jurisdictions:
            blockers.append(f"{source_id}: rights review has no jurisdiction scope.")
        elif not set(jurisdictions).issubset(covered_jurisdictions):
            blockers.append(
                f"{source_id}: rights review does not cover all requested jurisdictions."
            )

        commercial_allowed = review.get("commercial_use_allowed")
        if type(commercial_allowed) is not bool:
            raise ManifestError(f"{label}.rights_review.commercial_use_allowed must be boolean")
        if commercial_use is True and commercial_allowed is not True:
            blockers.append(f"{source_id}: commercial use is requested but not approved.")
        if commercial_use is None:
            blockers.append(f"{source_id}: commercial use cannot be evaluated until scope is explicit.")

        dimensions = require_object(
            review.get("rights_dimensions"), f"{label}.rights_review.rights_dimensions"
        )
        if set(dimensions) != REQUIRED_DIMENSIONS:
            raise ManifestError(
                f"{label}.rights_review.rights_dimensions must contain exactly the four required layers"
            )
        for dimension in sorted(REQUIRED_DIMENSIONS):
            layer = require_object(dimensions[dimension], f"{label}.rights_review.rights_dimensions.{dimension}")
            layer_decision = require_text(layer.get("decision"), f"{label}.{dimension}.decision")
            if layer_decision not in {"approved", "not_applicable", "blocked", "unclear", "pending"}:
                raise ManifestError(f"{label}.{dimension}.decision has an unsupported value")
            require_text(layer.get("rationale"), f"{label}.{dimension}.rationale")
            require_https(layer.get("evidence_url"), f"{label}.{dimension}.evidence_url")
            if layer_decision not in {"approved", "not_applicable"}:
                blockers.append(
                    f"{source_id}: rights layer {dimension} is {layer_decision!r}."
                )

        analyzed_ids.append(source_id)

    # Independent approval is a separate receipt over the exact bytes of the manifest.
    # Updating even one byte invalidates this approval.
    if approval.get("status") != "approved":
        blockers.append("Independent manifest approval receipt is not approved.")
    approval_hash = approval.get("manifest_sha256")
    expected_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if not isinstance(approval_hash, str) or approval_hash != expected_hash:
        blockers.append("Approval receipt SHA-256 does not match the exact manifest bytes.")
    if approval.get("status") == "approved":
        approver = require_text(approval.get("approver_id"), "approval.approver_id")
        approved_on = require_date(approval.get("approved_on"), "approval.approved_on")
        require_text(approval.get("scope_statement"), "approval.scope_statement")
        latest_review_date = max(
            (require_date(entry["rights_review"].get("reviewed_on"), "rights_review.reviewed_on")
             for entry in sources),
            default=approved_on,
        )
        if approved_on < latest_review_date:
            blockers.append("Independent manifest approval predates one or more source rights reviews.")
        if approver in reviewed_by:
            blockers.append("The manifest approver must be independent of every source rights reviewer.")

    blockers = list(dict.fromkeys(blockers))
    authorized = bool(sources) and not blockers
    return {
        "schema": "muse-choro-input-manifest-gate-report-v1",
        "manifest_sha256": expected_hash,
        "manifest_id": require_text(manifest.get("manifest_id"), "manifest_id"),
        "authorization_status": "authorized_for_declared_scope" if authorized else "blocked",
        "authorized": authorized,
        "source_count": len(sources),
        "source_ids": analyzed_ids,
        "artifacts": verified_artifacts,
        "scope": {
            "purpose": purpose,
            "requested_uses": requested_uses,
            "commercial_use": commercial_use,
            "jurisdictions": jurisdictions,
            "planned_outputs": planned_outputs,
            "retention_policy": retention_policy,
        },
        "blockers": blockers,
        "interpretation": (
            "A successful authorization status means the recorded scope, artifact hashes, "
            "rights-layer decisions and independent receipt are internally consistent. "
            "It is not legal advice and does not override the rights holders' terms."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--intake", type=Path, default=DEFAULT_INTAKE)
    parser.add_argument(
        "--artifact-root", type=Path,
        help="directory containing the exact reviewed files (the script hashes bytes only; it does not parse score content)",
    )
    parser.add_argument(
        "--require-authorized",
        action="store_true",
        help="exit non-zero unless the manifest is authorized for its exact declared scope",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the manifest contract and print its authorization state (blocked is a valid template state)",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate_manifest(args.manifest, args.approval, args.intake, args.artifact_root)
    except ManifestError as exc:
        print(f"INVALID MANIFEST: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_authorized and not report["authorized"]:
        print("BLOCKED: no source-bound analysis authorization", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
