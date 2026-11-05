#!/usr/bin/env python3
"""Fail-closed gate for source artifacts, rights scope, and registered Choro analysis plans.

This script validates recorded evidence and exact byte digests; it does not make
legal determinations or independently verify the truth of an external registration.
An empty template is valid but never authorized.
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
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
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


def validate_analysis_plan(
    scope: dict[str, Any],
    analysis_plan_root: Path = ROOT,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a separately frozen, externally registered analysis-plan artifact."""
    plan = require_object(scope.get("analysis_plan"), "scope.analysis_plan")
    plan_id = require_text(plan.get("plan_id"), "scope.analysis_plan.plan_id")
    raw_path = require_text(plan.get("path"), "scope.analysis_plan.path")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ManifestError("scope.analysis_plan.path must be a relative path without traversal")

    root_resolved = analysis_plan_root.resolve()
    plan_file = (root_resolved / relative_path).resolve()
    try:
        plan_file.relative_to(root_resolved)
    except ValueError as exc:
        raise ManifestError("scope.analysis_plan.path escapes --analysis-plan-root") from exc
    try:
        plan_bytes = plan_file.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read analysis plan at {plan_file}: {exc}") from exc

    expected_hash = require_text(plan.get("sha256"), "scope.analysis_plan.sha256")
    if not SHA256_RE.fullmatch(expected_hash):
        raise ManifestError("scope.analysis_plan.sha256 must be a lowercase 64-character SHA-256")
    actual_hash = hashlib.sha256(plan_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise ManifestError("analysis plan SHA-256 does not match the exact on-disk plan bytes")

    try:
        plan_document = json.loads(plan_bytes)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"analysis plan is not valid JSON: {exc}") from exc
    if not isinstance(plan_document, dict):
        raise ManifestError("analysis plan JSON root must be an object")
    if require_text(plan_document.get("plan_id"), "analysis-plan file plan_id") != plan_id:
        raise ManifestError("scope.analysis_plan.plan_id does not match the plan file")

    if require_text(plan_document.get("status"), "analysis-plan file status") != "registered":
        raise ManifestError("analysis plan top-level status must be 'registered'")

    preregistration = require_object(
        plan_document.get("pre_registration"),
        "analysis-plan file pre_registration",
    )
    if require_text(preregistration.get("status"), "pre_registration.status") != "registered":
        raise ManifestError("analysis plan is not externally registered")
    for frozen_field in ("hypotheses_frozen", "null_models_frozen", "thresholds_frozen"):
        if preregistration.get(frozen_field) is not True:
            raise ManifestError(f"analysis plan requires pre_registration.{frozen_field} == true")

    if plan_document.get("schema") != "muse-choro-preregistered-analysis-plan-v1":
        raise ManifestError("analysis plan schema must be 'muse-choro-preregistered-analysis-plan-v1'")

    design = require_object(plan_document.get("design"), "analysis-plan file design")
    require_text(design.get("source_version_pin"), "analysis-plan design.source_version_pin")
    require_text(design.get("study_question"), "analysis-plan design.study_question")

    for list_field in (
        "inclusion_criteria",
        "exclusion_criteria",
        "primary_outcomes",
        "sensitivity_analyses",
    ):
        items = require_string_list(
            design.get(list_field),
            f"analysis-plan design.{list_field}",
        )
        if not items:
            raise ManifestError(f"analysis-plan design.{list_field} must not be empty")

    hypotheses = design.get("hypotheses")
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ManifestError("analysis-plan design.hypotheses must contain at least one concrete hypothesis")
    hypothesis_ids: set[str] = set()
    for index, raw_hypothesis in enumerate(hypotheses):
        label = f"analysis-plan design.hypotheses[{index}]"
        hypothesis = require_object(raw_hypothesis, label)
        hypothesis_id = require_text(hypothesis.get("id"), f"{label}.id")
        if hypothesis_id in hypothesis_ids:
            raise ManifestError("analysis-plan hypothesis IDs must be unique")
        hypothesis_ids.add(hypothesis_id)
        for field in ("statement", "primary_test", "unit_of_analysis", "falsification_rule"):
            require_text(hypothesis.get(field), f"{label}.{field}")

    null_models = design.get("null_models")
    if not isinstance(null_models, list) or not null_models:
        raise ManifestError("analysis-plan design.null_models must contain at least one concrete null model")
    null_model_ids: set[str] = set()
    for index, raw_null_model in enumerate(null_models):
        label = f"analysis-plan design.null_models[{index}]"
        null_model = require_object(raw_null_model, label)
        null_model_id = require_text(null_model.get("id"), f"{label}.id")
        if null_model_id in null_model_ids:
            raise ManifestError("analysis-plan null-model IDs must be unique")
        null_model_ids.add(null_model_id)
        for field in (
            "description",
            "preserved_properties",
            "randomization_or_permutation",
            "unit_of_resampling",
        ):
            require_text(null_model.get(field), f"{label}.{field}")

    declared_bindings = design.get("authorized_source_bindings")
    if not isinstance(declared_bindings, list):
        raise ManifestError("analysis-plan design.authorized_source_bindings must be a list")
    normalized_plan_bindings: list[dict[str, Any]] = []
    seen_binding_ids: set[str] = set()
    for index, raw_binding in enumerate(declared_bindings):
        label = f"analysis-plan design.authorized_source_bindings[{index}]"
        binding = require_object(raw_binding, label)
        binding_id = require_text(binding.get("intake_source_id"), f"{label}.intake_source_id")
        if binding_id in seen_binding_ids:
            raise ManifestError("analysis-plan source-binding IDs must be unique")
        seen_binding_ids.add(binding_id)
        file_url = require_https(binding.get("source_file_url"), f"{label}.source_file_url")
        record_url = require_https(binding.get("source_record_url"), f"{label}.source_record_url")
        artifact_path = Path(require_text(binding.get("artifact_path"), f"{label}.artifact_path"))
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ManifestError(f"{label}.artifact_path must be a relative path without traversal")
        artifact_sha = require_text(binding.get("artifact_sha256"), f"{label}.artifact_sha256")
        if not SHA256_RE.fullmatch(artifact_sha):
            raise ManifestError(f"{label}.artifact_sha256 must be a lowercase 64-character SHA-256")
        artifact_bytes = binding.get("artifact_bytes")
        if type(artifact_bytes) is not int or artifact_bytes <= 0:
            raise ManifestError(f"{label}.artifact_bytes must be a positive integer")
        selected_ids = require_string_list(binding.get("selected_work_ids"), f"{label}.selected_work_ids")
        if not selected_ids:
            raise ManifestError(f"{label}.selected_work_ids must not be empty")
        normalized_plan_bindings.append({
            "intake_source_id": binding_id,
            "source_record_url": record_url,
            "source_file_url": file_url,
            "artifact_path": artifact_path.as_posix(),
            "artifact_sha256": artifact_sha,
            "artifact_bytes": artifact_bytes,
            "selected_work_ids": selected_ids,
        })
    normalized_plan_bindings.sort(key=lambda row: row["intake_source_id"])

    if manifest is not None:
        manifest_sources = manifest.get("sources")
        if not isinstance(manifest_sources, list):
            raise ManifestError("input manifest sources must be a list")
        if manifest_sources:
            expected_bindings: list[dict[str, Any]] = []
            for index, raw_entry in enumerate(manifest_sources):
                entry = require_object(raw_entry, f"manifest sources[{index}]")
                source_id = require_text(entry.get("intake_source_id"), "manifest intake_source_id")
                artifact_path = Path(require_text(entry.get("artifact_path"), "manifest artifact_path"))
                artifact_bytes = entry.get("artifact_bytes")
                if type(artifact_bytes) is not int or artifact_bytes <= 0:
                    raise ManifestError("manifest artifact_bytes must be a positive integer")
                expected_bindings.append({
                    "intake_source_id": source_id,
                    "source_record_url": require_https(entry.get("source_record_url"), "manifest source_record_url"),
                    "source_file_url": require_https(entry.get("source_file_url"), "manifest source_file_url"),
                    "artifact_path": artifact_path.as_posix(),
                    "artifact_sha256": require_text(entry.get("artifact_sha256"), "manifest artifact_sha256"),
                    "artifact_bytes": artifact_bytes,
                    "selected_work_ids": require_string_list(entry.get("selected_work_ids"), "manifest selected_work_ids"),
                })
            expected_bindings.sort(key=lambda row: row["intake_source_id"])
            if normalized_plan_bindings != expected_bindings:
                raise ManifestError(
                    "analysis plan authorized_source_bindings do not exactly match source artifacts and selected works in the input manifest"
                )
        elif normalized_plan_bindings:
            raise ManifestError("empty input manifest cannot bind planned source artifacts")

    thresholds = require_object(
        design.get("effect_and_uncertainty_thresholds"),
        "analysis-plan design.effect_and_uncertainty_thresholds",
    )
    for field in ("effect_size_rule", "uncertainty_rule", "decision_rule"):
        require_text(thresholds.get(field), f"analysis-plan thresholds.{field}")
    for field in (
        "multiplicity_correction",
        "missing_data_policy",
        "split_strategy",
        "deviation_policy",
    ):
        require_text(design.get(field), f"analysis-plan design.{field}")

    registered_on = require_date(
        preregistration.get("registered_on"),
        "analysis-plan file pre_registration.registered_on",
    )
    registration_url = require_https(
        preregistration.get("registration_record_url"),
        "analysis-plan file pre_registration.registration_record_url",
    )
    if require_date(plan.get("registered_on"), "scope.analysis_plan.registered_on") != registered_on:
        raise ManifestError("manifest analysis-plan registration date does not match the plan file")
    if require_https(
        plan.get("registration_record_url"),
        "scope.analysis_plan.registration_record_url",
    ) != registration_url:
        raise ManifestError("manifest analysis-plan registration URL does not match the plan file")

    return {
        "plan_id": plan_id,
        "path": relative_path.as_posix(),
        "sha256": actual_hash,
        "registered_on": registered_on,
        "registration_record_url": registration_url,
        "hypothesis_count": len(hypotheses),
        "null_model_count": len(null_models),
        "source_binding_count": len(normalized_plan_bindings),
        "primary_outcome_count": len(design["primary_outcomes"]),
        "sensitivity_analysis_count": len(design["sensitivity_analyses"]),
    }


def review_scope_sha256(
    manifest: dict[str, Any],
    entry: dict[str, Any],
    analysis_plan_root: Path = ROOT,
) -> str:
    """Hash exact selected works plus artifact and requested study scope.

    The rights reviewer records this digest only after reviewing the selected
    works, exact artifact, exact frozen analysis plan, intended uses, commercial
    status, jurisdictions and planned outputs together. Canonical JSON makes
    the digest reproducible.
    """
    scope = require_object(manifest.get("scope"), "scope")
    analysis_plan = validate_analysis_plan(scope, analysis_plan_root, manifest)
    selected_work_ids = require_string_list(
        entry.get("selected_work_ids"), "selected_work_ids"
    )
    if not selected_work_ids:
        raise ManifestError("selected_work_ids must name at least one work or movement")
    payload = {
        "manifest_id": require_text(manifest.get("manifest_id"), "manifest_id"),
        "intake_source_id": require_text(entry.get("intake_source_id"), "intake_source_id"),
        "source_record_url": require_https(entry.get("source_record_url"), "source_record_url"),
        "source_file_url": require_https(entry.get("source_file_url"), "source_file_url"),
        "artifact_path": Path(require_text(entry.get("artifact_path"), "artifact_path")).as_posix(),
        "artifact_sha256": require_text(entry.get("artifact_sha256"), "artifact_sha256"),
        "artifact_bytes": entry.get("artifact_bytes"),
        "retrieved_on": require_date(entry.get("retrieved_on"), "retrieved_on"),
        "selected_work_ids": selected_work_ids,
        "selection_basis": require_text(entry.get("selection_basis"), "selection_basis"),
        "required_rights_dimensions": sorted(
            require_string_list(
                manifest.get("required_rights_dimensions"),
                "required_rights_dimensions",
            )
        ),
        "analysis_plan": analysis_plan,
        "scope": {
            "purpose": require_text(scope.get("purpose"), "scope.purpose"),
            "requested_uses": sorted(
                require_string_list(scope.get("requested_uses"), "scope.requested_uses", USES)
            ),
            "commercial_use": scope.get("commercial_use"),
            "jurisdictions": sorted(
                require_string_list(scope.get("jurisdictions"), "scope.jurisdictions")
            ),
            "planned_outputs": sorted(
                require_string_list(scope.get("planned_outputs"), "scope.planned_outputs")
            ),
            "retention_policy": require_text(
                scope.get("retention_policy"), "scope.retention_policy"
            ),
        },
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def evaluate_manifest(
    manifest_path: Path,
    approval_path: Path,
    intake_path: Path = DEFAULT_INTAKE,
    artifact_root: Path | None = None,
    analysis_plan_root: Path = ROOT,
) -> dict[str, Any]:
    manifest, manifest_bytes = read_json(manifest_path)
    approval, approval_bytes = read_json(approval_path)
    intake, intake_bytes = read_json(intake_path)

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
    analysis_plan_summary: dict[str, Any] | None = None
    if not sources:
        blockers.append("No exact source artifacts are in the approved-input manifest.")
        raw_plan = scope.get("analysis_plan")
        plan_fields = ("plan_id", "path", "sha256", "registered_on", "registration_record_url")
        if isinstance(raw_plan, dict) and all(
            isinstance(raw_plan.get(field), str) and raw_plan[field].strip()
            for field in plan_fields
        ):
            try:
                analysis_plan_summary = validate_analysis_plan(scope, analysis_plan_root, manifest)
            except ManifestError as exc:
                blockers.append(f"Analysis plan is blocked: {exc}")
        else:
            blockers.append("No hash-bound pre-registered analysis plan is declared.")
    else:
        analysis_plan_summary = validate_analysis_plan(scope, analysis_plan_root, manifest)

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
        selected_work_ids = require_string_list(
            entry.get("selected_work_ids"), f"{label}.selected_work_ids"
        )
        if not selected_work_ids:
            raise ManifestError(f"{label}.selected_work_ids must name at least one work or movement")
        selection_basis = require_text(entry.get("selection_basis"), f"{label}.selection_basis")
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
        expected_review_scope_sha256 = review_scope_sha256(manifest, entry, analysis_plan_root)
        verified_artifacts.append({
            "source_id": source_id,
            "artifact_path": relative_artifact.as_posix(),
            "artifact_sha256": artifact_sha,
            "artifact_bytes": artifact_bytes,
            "selected_work_ids": selected_work_ids,
            "selection_basis": selection_basis,
            "review_scope_sha256": expected_review_scope_sha256,
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
        if analysis_plan_summary and reviewed_on < analysis_plan_summary["registered_on"]:
            blockers.append(
                f"{source_id}: rights review predates registration of the frozen analysis plan."
            )
        require_https(review.get("review_evidence_url"), f"{label}.rights_review.review_evidence_url")
        require_https(review.get("license_or_basis_url"), f"{label}.rights_review.license_or_basis_url")
        require_text(review.get("attribution_text"), f"{label}.rights_review.attribution_text")

        reviewed_scope_hash = require_text(
            review.get("reviewed_scope_sha256"),
            f"{label}.rights_review.reviewed_scope_sha256",
        )
        if not SHA256_RE.fullmatch(reviewed_scope_hash):
            raise ManifestError(
                f"{label}.rights_review.reviewed_scope_sha256 must be a lowercase 64-character SHA-256"
            )
        if reviewed_scope_hash != expected_review_scope_sha256:
            blockers.append(
                f"{source_id}: rights review does not bind selected works and the exact requested scope."
            )

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
        if analysis_plan_summary and approved_on < analysis_plan_summary["registered_on"]:
            blockers.append("Independent manifest approval predates registration of the frozen analysis plan.")
        if approver in reviewed_by:
            blockers.append("The manifest approver must be independent of every source rights reviewer.")

    blockers = list(dict.fromkeys(blockers))
    authorized = bool(sources) and not blockers
    return {
        "schema": "muse-choro-input-manifest-gate-report-v1",
        "manifest_sha256": expected_hash,
        "approval_receipt_sha256": hashlib.sha256(approval_bytes).hexdigest(),
        "intake_sha256": hashlib.sha256(intake_bytes).hexdigest(),
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
            "analysis_plan": analysis_plan_summary,
        },
        "blockers": blockers,
        "interpretation": (
            "A successful authorization status means the recorded scope, source/plan artifact hashes, "
            "registered-plan fields, rights-layer decisions and independent receipt are internally consistent. "
            "It is not legal advice and does not prove an external registration record is truthful or current."
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
        "--analysis-plan-root",
        type=Path,
        default=ROOT,
        help="repository-relative root containing the exact frozen analysis-plan JSON",
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
        report = evaluate_manifest(
            args.manifest, args.approval, args.intake, args.artifact_root, args.analysis_plan_root
        )
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
