#!/usr/bin/env python3
"""Audit structural bindings in evaluator-produced VLEB shift-event evidence.

This tool detects malformed or inconsistent evidence records; it does not prove
that a runner actually applied the shift. Confirmatory use requires external
attestation verification over the exact evidence bundle and trusted signer policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

HEX = re.compile(r"^[a-f0-9]{64}$")
REVISION = re.compile(r"^[a-f0-9]{40}$")
IMAGE = re.compile(r"^sha256:[a-f0-9]{64}$")
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
TOP_KEYS = {"schema_version", "cohort_id", "protocol_sha256", "scenarios", "seeds", "receipts"}
SCENARIO_KEYS = {
    "scenario_id", "task_family", "change_family", "intervention_position",
    "task_contract_before_sha256", "task_contract_after_sha256", "shift_payload_sha256",
    "initial_workspace_sha256", "initial_checkpoint_sha256", "initial_revision",
    "initial_image_id", "required_check_ids", "stale_artifact_sha256s",
}
RECEIPT_KEYS = {
    "receipt_payload_sha256", "scenario_id", "seed", "condition", "run_id", "attempt_number",
    "scenario_sha256", "protocol_sha256", "task_contract_sha256", "initial_workspace_sha256",
    "initial_checkpoint_sha256", "initial_revision", "initial_image_id", "model_id",
    "model_version", "scaffold_sha256", "initial_execution_profile_sha256",
    "initial_network_mode", "post_shift_execution_profile_sha256", "post_shift_network_mode",
    "final_workspace_sha256", "outcome", "runner_event", "stale_evidence_disposition",
    "validation_results", "audit_report_sha256",
}
EVENT_KEYS = {
    "event_id", "event_kind", "change_family", "sequence_number", "intervention_position", "applied", "payload_sha256",
    "pre_workspace_sha256", "post_workspace_sha256", "transition_diff_sha256",
    "runner_event_log_sha256", "observed_monotonic_ns", "event_source",
}
STALE_KEYS = {"artifact_sha256", "disposition", "reason_event_id"}
CHECK_KEYS = {"check_id", "input_workspace_sha256", "status", "evidence_sha256"}
CONDITIONS = {"clean", "shifted"}
OUTCOMES = {"pass", "fail", "unresolved"}
CHECK_STATUSES = {"pass", "fail", "unresolved"}
NETWORK_MODES = {"none", "allowlisted", "enabled", "not_applicable"}
EVENT_KINDS = {
    "requirement_update", "source_update", "dependency_update", "tool_session_change",
    "runtime_state_change", "evidence_invalidation", "other",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return digest(canonical_json_bytes(value))


def strict_json_loads(raw: bytes) -> Any:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r} rejected")

    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label} keys invalid: missing={missing}, unexpected={extra}")


def require_digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def require_label(value: Any, label: str) -> None:
    if not isinstance(value, str) or not LABEL.fullmatch(value):
        raise ValueError(f"{label} must match {LABEL.pattern!r}")


def require_identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} is invalid")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_manifest(manifest: Any, expected_protocol_sha256: str) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    exact_keys(manifest, TOP_KEYS, "manifest")
    if manifest["schema_version"] != "vleb-shift-evidence-v1":
        raise ValueError("unsupported shift-evidence manifest schema_version")
    if not isinstance(manifest["cohort_id"], str) or not manifest["cohort_id"].strip():
        raise ValueError("cohort_id must be a non-empty string")
    require_digest(manifest["protocol_sha256"], "manifest protocol_sha256")
    require_digest(expected_protocol_sha256, "externally expected protocol_sha256")
    if manifest["protocol_sha256"] != expected_protocol_sha256:
        raise ValueError("manifest protocol differs from evaluator-supplied protocol pin")

    scenarios = manifest["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) < 1:
        raise ValueError("scenarios must be a non-empty array")
    scenario_by_id: dict[str, dict[str, Any]] = {}
    scenario_hashes: dict[str, str] = {}
    for index, scenario in enumerate(scenarios):
        label = f"scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise ValueError(f"{label} must be an object")
        exact_keys(scenario, SCENARIO_KEYS, label)
        require_identifier(scenario["scenario_id"], label + ".scenario_id")
        if scenario["scenario_id"] in scenario_by_id:
            raise ValueError("scenario IDs must be unique")
        require_label(scenario["task_family"], label + ".task_family")
        require_label(scenario["change_family"], label + ".change_family")
        if type(scenario["intervention_position"]) is not int or not 1 <= scenario["intervention_position"] <= 9:
            raise ValueError(f"{label}.intervention_position must be an integer in [1, 9]")
        for key in (
            "task_contract_before_sha256", "task_contract_after_sha256", "shift_payload_sha256",
            "initial_workspace_sha256", "initial_checkpoint_sha256",
        ):
            require_digest(scenario[key], label + "." + key)
        if not isinstance(scenario["initial_revision"], str) or not REVISION.fullmatch(scenario["initial_revision"]):
            raise ValueError(f"{label}.initial_revision must be a lowercase 40-character commit SHA")
        if not isinstance(scenario["initial_image_id"], str) or not IMAGE.fullmatch(scenario["initial_image_id"]):
            raise ValueError(f"{label}.initial_image_id must be an immutable sha256 image reference")
        required = scenario["required_check_ids"]
        if (
            not isinstance(required, list) or not required
            or any(not isinstance(item, str) or not IDENTIFIER.fullmatch(item) for item in required)
            or len(required) != len(set(required))
        ):
            raise ValueError(f"{label}.required_check_ids must be a non-empty unique identifier list")
        stale = scenario["stale_artifact_sha256s"]
        if (
            not isinstance(stale, list)
            or any(not isinstance(item, str) or not HEX.fullmatch(item) for item in stale)
            or len(stale) != len(set(stale))
        ):
            raise ValueError(f"{label}.stale_artifact_sha256s must be a unique list of SHA-256 digests")
        scenario_by_id[scenario["scenario_id"]] = scenario
        scenario_hashes[scenario["scenario_id"]] = canonical_digest(scenario)

    seeds = manifest["seeds"]
    if (
        not isinstance(seeds, list) or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise ValueError("seeds must be a non-empty list of unique non-negative integers")

    receipts = manifest["receipts"]
    expected_count = len(scenarios) * len(seeds) * 2
    if not isinstance(receipts, list) or len(receipts) != expected_count:
        observed = len(receipts) if isinstance(receipts, list) else "non-list"
        raise ValueError(
            f"receipts must cover the complete locked scenario × seed × condition cohort: "
            f"expected {expected_count}, observed {observed}"
        )

    expected_cells = {
        (scenario_id, seed, condition)
        for scenario_id in scenario_by_id for seed in seeds for condition in CONDITIONS
    }
    observed_cells: set[tuple[str, int, str]] = set()
    run_ids: set[str] = set()
    receipt_digests: set[str] = set()
    audit_digests: set[str] = set()
    configuration_by_scenario: dict[str, tuple[Any, ...]] = {}
    post_shift_config_by_scenario: dict[str, tuple[Any, ...]] = {}
    normalized_receipts: list[dict[str, Any]] = []
    for index, receipt in enumerate(receipts):
        label = f"receipts[{index}]"
        if not isinstance(receipt, dict):
            raise ValueError(f"{label} must be an object")
        exact_keys(receipt, RECEIPT_KEYS, label)
        scenario_id, seed, condition = receipt["scenario_id"], receipt["seed"], receipt["condition"]
        if (
            not isinstance(scenario_id, str) or scenario_id not in scenario_by_id
            or type(seed) is not int or seed not in set(seeds)
            or not isinstance(condition, str) or condition not in CONDITIONS
        ):
            raise ValueError(f"{label} is outside the locked scenario/seed/condition inventory")
        cell = (scenario_id, seed, condition)
        if cell in observed_cells:
            raise ValueError(f"duplicate receipt cell {cell!r}")
        observed_cells.add(cell)
        scenario = scenario_by_id[scenario_id]
        if receipt["scenario_sha256"] != scenario_hashes[scenario_id]:
            raise ValueError(f"{label}.scenario_sha256 differs from canonical locked scenario")
        if receipt["protocol_sha256"] != expected_protocol_sha256:
            raise ValueError(f"{label}.protocol_sha256 differs from the external protocol pin")
        if receipt["task_contract_sha256"] != (
            scenario["task_contract_after_sha256"] if condition == "shifted"
            else scenario["task_contract_before_sha256"]
        ):
            raise ValueError(f"{label} task-contract digest is inconsistent with its condition")
        for key, expected in (
            ("initial_workspace_sha256", scenario["initial_workspace_sha256"]),
            ("initial_checkpoint_sha256", scenario["initial_checkpoint_sha256"]),
            ("initial_revision", scenario["initial_revision"]),
            ("initial_image_id", scenario["initial_image_id"]),
        ):
            if receipt[key] != expected:
                raise ValueError(f"{label}.{key} differs from the locked starting identity")
        require_identifier(receipt["run_id"], label + ".run_id")
        if receipt["run_id"] in run_ids:
            raise ValueError("run_id must be unique for every scheduled condition")
        run_ids.add(receipt["run_id"])
        if type(receipt["attempt_number"]) is not int or receipt["attempt_number"] != 1:
            raise ValueError(f"{label}.attempt_number must be 1; retry substitution is rejected")
        for key in (
            "receipt_payload_sha256", "scaffold_sha256", "initial_execution_profile_sha256",
            "final_workspace_sha256", "audit_report_sha256",
        ):
            require_digest(receipt[key], label + "." + key)
        payload_for_digest = dict(receipt)
        del payload_for_digest["receipt_payload_sha256"]
        if canonical_digest(payload_for_digest) != receipt["receipt_payload_sha256"]:
            raise ValueError(f"{label} receipt payload digest does not match canonical receipt bytes")
        if receipt["receipt_payload_sha256"] in receipt_digests:
            raise ValueError("receipt payload digest replay detected")
        receipt_digests.add(receipt["receipt_payload_sha256"])
        if receipt["audit_report_sha256"] in audit_digests:
            raise ValueError("audit-report digest replay detected")
        audit_digests.add(receipt["audit_report_sha256"])
        if not isinstance(receipt["model_id"], str) or not receipt["model_id"].strip():
            raise ValueError(f"{label}.model_id must be non-empty")
        if not isinstance(receipt["model_version"], str) or not receipt["model_version"].strip():
            raise ValueError(f"{label}.model_version must be non-empty")
        if not isinstance(receipt["initial_network_mode"], str) or receipt["initial_network_mode"] not in NETWORK_MODES:
            raise ValueError(f"{label}.initial_network_mode is invalid")
        if not isinstance(receipt["outcome"], str) or receipt["outcome"] not in OUTCOMES:
            raise ValueError(f"{label}.outcome must be pass, fail, or unresolved")

        configuration = (
            receipt["model_id"], receipt["model_version"], receipt["scaffold_sha256"],
            receipt["initial_execution_profile_sha256"], receipt["initial_network_mode"],
        )
        previous_configuration = configuration_by_scenario.setdefault(scenario_id, configuration)
        if previous_configuration != configuration:
            raise ValueError(f"scenario {scenario_id!r} drifts in model/scaffold/initial profile across seeds")
        event = receipt["runner_event"]
        stale_disposition = receipt["stale_evidence_disposition"]
        checks = receipt["validation_results"]
        if not isinstance(stale_disposition, list) or not isinstance(checks, list):
            raise ValueError(f"{label} stale_evidence_disposition and validation_results must be arrays")
        if condition == "clean":
            if event is not None:
                raise ValueError(f"{label} clean condition must not claim a shift event")
            if receipt["post_shift_execution_profile_sha256"] is not None or receipt["post_shift_network_mode"] is not None:
                raise ValueError(f"{label} clean condition must not claim post-shift settings")
            if stale_disposition:
                raise ValueError(f"{label} clean condition must not declare shift-triggered evidence dispositions")
        else:
            if not isinstance(event, dict):
                raise ValueError(f"{label} shifted condition requires a runner_event object")
            exact_keys(event, EVENT_KEYS, label + ".runner_event")
            require_identifier(event["event_id"], label + ".runner_event.event_id")
            if not isinstance(event["event_kind"], str) or event["event_kind"] not in EVENT_KINDS:
                raise ValueError(f"{label}.runner_event.event_kind is invalid")
            if event["change_family"] != scenario["change_family"]:
                raise ValueError(f"{label}.runner_event change family differs from locked scenario")
            if type(event["sequence_number"]) is not int or event["sequence_number"] < 1:
                raise ValueError(f"{label}.runner_event.sequence_number must be a positive integer")
            if event["intervention_position"] != scenario["intervention_position"]:
                raise ValueError(f"{label}.runner_event intervention position differs from locked scenario")
            if event["applied"] is not True or event["event_source"] != "trusted_runner":
                raise ValueError(f"{label} event must be marked applied and sourced from trusted_runner")
            for key in (
                "payload_sha256", "pre_workspace_sha256", "post_workspace_sha256",
                "transition_diff_sha256", "runner_event_log_sha256",
            ):
                require_digest(event[key], label + ".runner_event." + key)
            if event["payload_sha256"] != scenario["shift_payload_sha256"]:
                raise ValueError(f"{label} observed shift payload differs from locked scenario")
            if event["pre_workspace_sha256"] != scenario["initial_workspace_sha256"]:
                raise ValueError(f"{label} shift event did not start from the locked workspace state")
            if type(event["observed_monotonic_ns"]) is not int or event["observed_monotonic_ns"] < 0:
                raise ValueError(f"{label}.runner_event.observed_monotonic_ns is invalid")
            if receipt["post_shift_execution_profile_sha256"] is None:
                raise ValueError(f"{label} shifted condition requires a post-shift execution-profile digest")
            require_digest(receipt["post_shift_execution_profile_sha256"], label + ".post_shift_execution_profile_sha256")
            if not isinstance(receipt["post_shift_network_mode"], str) or receipt["post_shift_network_mode"] not in NETWORK_MODES:
                raise ValueError(f"{label}.post_shift_network_mode is invalid")
            post_shift_configuration = (
                receipt["post_shift_execution_profile_sha256"], receipt["post_shift_network_mode"]
            )
            previous_post_shift = post_shift_config_by_scenario.setdefault(scenario_id, post_shift_configuration)
            if previous_post_shift != post_shift_configuration:
                raise ValueError(f"scenario {scenario_id!r} drifts in post-shift profile/network mode across seeds")

            if len(stale_disposition) != len(scenario["stale_artifact_sha256s"]):
                raise ValueError(f"{label} stale-evidence dispositions do not cover the locked stale-artifact inventory")
            seen_stale: set[str] = set()
            for stale_index, item in enumerate(stale_disposition):
                stale_label = f"{label}.stale_evidence_disposition[{stale_index}]"
                if not isinstance(item, dict):
                    raise ValueError(f"{stale_label} must be an object")
                exact_keys(item, STALE_KEYS, stale_label)
                require_digest(item["artifact_sha256"], stale_label + ".artifact_sha256")
                require_identifier(item["reason_event_id"], stale_label + ".reason_event_id")
                if item["artifact_sha256"] in seen_stale:
                    raise ValueError(f"{label} duplicate stale-evidence disposition")
                seen_stale.add(item["artifact_sha256"])
                if not isinstance(item["disposition"], str) or item["disposition"] not in {"invalidated", "regenerated"}:
                    raise ValueError(f"{stale_label}.disposition must be invalidated or regenerated")
                if item["reason_event_id"] != event["event_id"]:
                    raise ValueError(f"{stale_label} is not bound to the observed shift event")
            if seen_stale != set(scenario["stale_artifact_sha256s"]):
                raise ValueError(f"{label} stale-evidence disposition differs from locked inventory")

        if not isinstance(checks, list) or len(checks) != len(scenario["required_check_ids"]):
            raise ValueError(f"{label} validation results must cover every required check exactly once")
        seen_checks: set[str] = set()
        for check_index, check in enumerate(checks):
            check_label = f"{label}.validation_results[{check_index}]"
            if not isinstance(check, dict):
                raise ValueError(f"{check_label} must be an object")
            exact_keys(check, CHECK_KEYS, check_label)
            require_identifier(check["check_id"], check_label + ".check_id")
            if check["check_id"] in seen_checks:
                raise ValueError(f"{label} duplicate validation check ID")
            seen_checks.add(check["check_id"])
            if not isinstance(check["status"], str) or check["status"] not in CHECK_STATUSES:
                raise ValueError(f"{check_label}.status is invalid")
            require_digest(check["input_workspace_sha256"], check_label + ".input_workspace_sha256")
            if check["input_workspace_sha256"] != receipt["final_workspace_sha256"]:
                raise ValueError(f"{check_label} ran against a workspace other than the pinned final state")
            if check["status"] == "unresolved":
                if check["evidence_sha256"] is not None:
                    require_digest(check["evidence_sha256"], check_label + ".evidence_sha256")
            else:
                require_digest(check["evidence_sha256"], check_label + ".evidence_sha256")
        if seen_checks != set(scenario["required_check_ids"]):
            raise ValueError(f"{label} required validation-check inventory mismatch")
        if receipt["outcome"] == "pass" and any(check["status"] != "pass" for check in checks):
            raise ValueError(f"{label} claims pass despite a required validation check not passing")
        if condition == "shifted" and receipt["outcome"] == "pass":
            if not event["applied"] or any(item["disposition"] not in {"invalidated", "regenerated"} for item in stale_disposition):
                raise ValueError(f"{label} shifted pass lacks the required intervention/evidence disposition")

        comparable = dict(receipt)
        del comparable["receipt_payload_sha256"]
        normalized_receipts.append({
            "scenario_id": scenario_id, "seed": seed, "condition": condition,
            "run_id": receipt["run_id"], "outcome": receipt["outcome"],
            "receipt_payload_sha256": receipt["receipt_payload_sha256"],
            "audit_report_sha256": receipt["audit_report_sha256"],
            "runner_event_log_sha256": (
                receipt["runner_event"]["runner_event_log_sha256"]
                if isinstance(receipt["runner_event"], dict) else None
            ),
            "stale_artifact_disposition_count": len(receipt["stale_evidence_disposition"]),
            "required_check_count": len(receipt["validation_results"]),
            "canonical_payload_sha256_recomputed": canonical_digest(comparable),
        })

    if observed_cells != expected_cells:
        raise ValueError("one or more locked scenario × seed × condition cells are missing")
    # Pair identities that must remain equal between clean and shifted conditions.
    receipts_by_cell = {
        (row["scenario_id"], row["seed"], row["condition"]): row for row in receipts
    }
    for scenario_id in scenario_by_id:
        for seed in seeds:
            clean = receipts_by_cell[(scenario_id, seed, "clean")]
            shifted = receipts_by_cell[(scenario_id, seed, "shifted")]
            shared_keys = (
                "scenario_sha256", "protocol_sha256", "initial_workspace_sha256",
                "initial_checkpoint_sha256", "initial_revision", "initial_image_id",
                "model_id", "model_version", "scaffold_sha256",
                "initial_execution_profile_sha256", "initial_network_mode",
            )
            if any(clean[key] != shifted[key] for key in shared_keys):
                raise ValueError(f"clean/shifted pair does not share starting identity for {(scenario_id, seed)!r}")
    return {
        "scenario_count": len(scenario_by_id),
        "seed_count": len(seeds),
        "expected_receipt_count": expected_count,
        "observed_receipt_count": len(normalized_receipts),
        "normalized_receipts": normalized_receipts,
        "scenarios": scenario_by_id,
        "protocol_sha256": manifest["protocol_sha256"],
    }


def analyze_manifest(
    manifest: Any,
    *,
    input_sha256: str,
    expected_protocol_sha256: str,
    expected_verifier_sha256: str,
) -> dict[str, Any]:
    require_digest(input_sha256, "input_sha256")
    require_digest(expected_verifier_sha256, "externally expected verifier SHA-256")
    observed_verifier_sha256 = digest(Path(__file__).read_bytes())
    if observed_verifier_sha256 != expected_verifier_sha256:
        raise ValueError("verifier source differs from evaluator-supplied SHA-256")
    validated = validate_manifest(manifest, expected_protocol_sha256)
    return {
        "schema_version": "vleb-shift-evidence-audit-report-v1",
        "valid": True,
        "cohort_id": manifest["cohort_id"],
        "input_manifest_sha256": input_sha256,
        "protocol_sha256": validated["protocol_sha256"],
        "verifier_sha256": observed_verifier_sha256,
        "scenario_count": validated["scenario_count"],
        "seed_count": validated["seed_count"],
        "expected_receipt_count": validated["expected_receipt_count"],
        "observed_receipt_count": validated["observed_receipt_count"],
        "normalized_receipts": validated["normalized_receipts"],
        "evidence_integrity_consistent": True,
        "runner_attestation_required": True,
        "score_eligible": False,
        "limitations": [
            "This report checks declared bytes, canonical receipt digests and internal bindings; it does not authenticate receipt or audit-report contents.",
            "The event_source field and applied flag are untrusted declarations unless the evidence bundle is independently attested by a trusted runner and signer policy.",
            "A confirmatory caller must verify the signed provenance for the exact bundle and run identity before using these records.",
            "The analyzer cannot prove that required checks were actually executed; it verifies only the consistency and identity of their declared records.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        raw = args.input.read_bytes()
        observed_input_sha256 = digest(raw)
        require_digest(args.expected_input_sha256, "expected input SHA-256")
        if observed_input_sha256 != args.expected_input_sha256:
            raise ValueError("input manifest differs from externally supplied SHA-256")
        result = analyze_manifest(
            strict_json_loads(raw),
            input_sha256=observed_input_sha256,
            expected_protocol_sha256=args.expected_protocol_sha256,
            expected_verifier_sha256=args.expected_verifier_sha256,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        result = {
            "schema_version": "vleb-shift-evidence-audit-report-v1",
            "valid": False, "evidence_integrity_consistent": False,
            "runner_attestation_required": True, "score_eligible": False,
            "issues": [{"code": "shift_evidence_rejected", "message": str(exc)}],
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
