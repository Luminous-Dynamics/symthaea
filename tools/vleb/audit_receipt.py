#!/usr/bin/env python3
"""Fail-closed independent auditor for VLEB evidence receipts.

Hashes establish artifact integrity, not execution provenance. Live benchmark use
also requires a trusted run manifest and a separately trusted runner/attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit("jsonschema==4.26.0 is required; refusing to skip schema validation") from exc

HEX = re.compile(r"^[a-f0-9]{64}$")
IMAGE = re.compile(r"^sha256:[a-f0-9]{64}$")
TEST_LINE = re.compile(r"^\s*test\s+(.+?)\s+\.\.\.\s+(ok|FAILED)\s*$")


@dataclass(frozen=True)
class AuditInputs:
    receipt: Path
    task_contract: Path
    task_schema: Path
    receipt_schema: Path
    protocol: Path
    candidate_patch: Path
    mutation_catalog: Path
    evidence_root: Path
    expected_revision: str
    expected_image_id: str
    expected_task_sha256: str
    expected_protocol_sha256: str
    expected_candidate_patch_sha256: str
    expected_auditor_sha256: str
    expected_verifier_binary_sha256: str


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    return digest(path.read_bytes())


def json_object(path: Path, label: str, issues: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        issues.append({"code": "invalid_json", "message": f"{label}: {exc}"})
        return None
    if not isinstance(value, dict):
        issues.append({"code": "invalid_json_root", "message": f"{label} must be a JSON object"})
        return None
    return value


def schema_check(value: dict[str, Any], path: Path, label: str, issues: list[dict[str, str]]) -> None:
    schema = json_object(path, label + " schema", issues)
    if schema is None:
        return
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
                        key=lambda error: (tuple(map(str, error.absolute_path)), error.message))
    except Exception as exc:
        issues.append({"code": "schema_validator_error", "message": f"{label}: {exc}"})
        return
    for error in errors:
        place = ".".join(map(str, error.absolute_path)) or "$"
        issues.append({"code": "schema_invalid", "message": f"{label} at {place}: {error.message}"})


def requirement_traceability_issues(task: dict[str, Any]) -> list[dict[str, str]]:
    """Require scored contracts to trace every declared requirement/effect to executable tests."""
    classification = task.get("classification", {})
    if not isinstance(classification, dict) or classification.get("exposure") not in ("private_holdout", "private_confirmatory"):
        return []

    behavior = task.get("behavior_contract", {})
    oracle = task.get("oracle_contract", {})
    if not isinstance(behavior, dict) or not isinstance(oracle, dict):
        return [{"code": "requirement_traceability_missing", "message": "scored task requires behavior and oracle contracts"}]

    raw_trace = behavior.get("requirement_traceability")
    if not isinstance(raw_trace, list) or not raw_trace:
        return [{"code": "requirement_traceability_missing", "message": "scored task must trace each task statement, invariant, allowed behavior change, and observable effect to executable tests"}]

    expected_sources: set[tuple[str, str]] = set()
    statement = behavior.get("task_statement")
    if isinstance(statement, str) and statement:
        expected_sources.add(("task_statement", statement))
    for source_kind, field in (
        ("invariant", "invariants"),
        ("allowed_behavior_change", "allowed_behavior_changes"),
        ("observable_effect", "observable_effects"),
    ):
        values = behavior.get(field, [])
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and value:
                    expected_sources.add((source_kind, value))

    declared_tests_value = oracle.get("required_test_ids", [])
    declared_tests = {value for value in declared_tests_value if isinstance(value, str) and value}
    issues: list[dict[str, str]] = []
    seen_requirement_ids: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    traced_tests: set[str] = set()
    declared_effects_value = behavior.get("observable_effects", [])
    declared_effects = {value for value in declared_effects_value if isinstance(value, str)}

    for index, row in enumerate(raw_trace):
        label = f"behavior_contract.requirement_traceability[{index}]"
        if not isinstance(row, dict):
            issues.append({"code": "requirement_traceability_invalid", "message": f"{label} must be an object"})
            continue
        requirement_id = row.get("requirement_id")
        if not isinstance(requirement_id, str) or not requirement_id:
            issues.append({"code": "requirement_traceability_invalid", "message": f"{label} has no requirement_id"})
        elif requirement_id in seen_requirement_ids:
            issues.append({"code": "requirement_id_duplicate", "message": f"duplicate requirement ID {requirement_id!r}"})
        else:
            seen_requirement_ids.add(requirement_id)

        source_kind = row.get("source_kind")
        source_text = row.get("source_text")
        source_key = (source_kind, source_text) if isinstance(source_kind, str) and isinstance(source_text, str) else None
        if source_key is None or source_key not in expected_sources:
            issues.append({"code": "requirement_source_unmatched", "message": f"{label} must match a declared task statement, invariant, allowed behavior change, or observable effect"})
        else:
            if source_key in seen_sources:
                issues.append({"code": "requirement_source_duplicate", "message": f"declared source {source_kind!r} is mapped more than once"})
            seen_sources.add(source_key)

        test_ids = row.get("required_test_ids")
        if not isinstance(test_ids, list) or not test_ids or any(not isinstance(value, str) or not value for value in test_ids):
            issues.append({"code": "requirement_test_map_invalid", "message": f"{label} requires one or more concrete logical test IDs"})
        else:
            if len(test_ids) != len(set(test_ids)):
                issues.append({"code": "requirement_test_map_duplicate_test_id", "message": f"{label} repeats a logical test ID"})
            for test_id in test_ids:
                if test_id not in declared_tests:
                    issues.append({"code": "requirement_unknown_test_id", "message": f"{label} references undeclared test ID {test_id!r}"})
                traced_tests.add(test_id)

        effects = row.get("observable_effects")
        if not isinstance(effects, list) or not effects or any(not isinstance(value, str) or not value for value in effects):
            issues.append({"code": "requirement_observable_mismatch", "message": f"{label} must declare at least one observable effect"})
        else:
            undeclared = sorted(set(effects) - declared_effects)
            if undeclared:
                issues.append({"code": "requirement_observable_mismatch", "message": f"{label} references undeclared observable effects: {undeclared}"})
            if source_kind == "observable_effect" and source_text not in effects:
                issues.append({"code": "requirement_observable_mismatch", "message": f"{label} must measure its source observable effect"})

    missing_sources = expected_sources - seen_sources
    if missing_sources:
        examples = sorted(f"{kind}:{value}" for kind, value in missing_sources)[:5]
        issues.append({"code": "requirement_source_coverage_incomplete", "message": f"{len(missing_sources)} declared contract item(s) lack a requirement-to-test trace: {examples}"})
    missing_tests = declared_tests - traced_tests
    extra_tests = traced_tests - declared_tests
    if missing_tests or extra_tests:
        issues.append({"code": "requirement_test_inventory_incomplete", "message": f"traceability must cover every required test exactly within the declared inventory; missing={sorted(missing_tests)}, extra={sorted(extra_tests)}"})
    return issues


def evidence_path(root: Path, relative: Any, label: str, issues: list[dict[str, str]]) -> Path | None:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        issues.append({"code": "unsafe_evidence_path", "message": f"{label}: expected a non-empty POSIX relative path"})
        return None
    part_path = PurePosixPath(relative)
    if part_path.is_absolute() or ".." in part_path.parts:
        issues.append({"code": "unsafe_evidence_path", "message": f"{label}: absolute or traversal path rejected"})
        return None
    try:
        resolved_root = root.resolve(strict=True)
        cursor = resolved_root
        for part in part_path.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                issues.append({"code": "unsafe_evidence_path", "message": f"{label}: symlink evidence path rejected"})
                return None
        resolved = cursor.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
            issues.append({"code": "unsafe_evidence_path", "message": f"{label}: evidence must be a regular file under evidence root"})
            return None
        return resolved
    except OSError as exc:
        issues.append({"code": "missing_evidence", "message": f"{label}: {exc}"})
        return None


def checked_artifact(root: Path, relative: Any, expected: Any, label: str,
                     issues: list[dict[str, str]]) -> bytes | None:
    path = evidence_path(root, relative, label, issues)
    if path is None:
        return None
    if not isinstance(expected, str) or not HEX.fullmatch(expected):
        issues.append({"code": "invalid_digest", "message": f"{label}: missing or malformed SHA-256"})
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        issues.append({"code": "missing_evidence", "message": f"{label}: cannot read artifact: {exc}"})
        return None
    observed = digest(raw)
    if observed != expected:
        issues.append({"code": "digest_mismatch", "message": f"{label}: artifact digest does not match receipt"})
    return raw


def parse_test_log(raw: bytes) -> dict[str, str]:
    """Return unique test function leaf names; ambiguous leaf names become CONFLICT."""
    seen: dict[str, str] = {}
    origins: dict[str, str] = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        match = TEST_LINE.match(line)
        if not match:
            continue
        qualified, status = match.group(1).strip(), match.group(2)
        leaf = qualified.split("::")[-1]
        if leaf in origins and origins[leaf] != qualified:
            seen[leaf] = "CONFLICT"
        elif leaf in seen and seen[leaf] != status:
            seen[leaf] = "CONFLICT"
        else:
            origins[leaf] = qualified
            seen[leaf] = status
    return seen


def audit_bundle(inputs: AuditInputs) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    receipt = json_object(inputs.receipt, "receipt", issues)
    task = json_object(inputs.task_contract, "task contract", issues)
    catalog = json_object(inputs.mutation_catalog, "mutation catalog", issues)
    if receipt is None or task is None or catalog is None:
        return {"schema_version": "vleb-independent-audit-v1", "eligible": False, "verdict": "unresolved", "issues": issues}

    active = catalog.get("active_campaign")
    catalog_rows = catalog.get("mutations")
    if (not isinstance(active, dict) or not isinstance(active.get("required_mutation_ids"), list)
            or not isinstance(catalog_rows, list)
            or any(not isinstance(row, dict) or not isinstance(row.get("mutation_id"), str) or not row.get("mutation_id") for row in catalog_rows)):
        return {"schema_version": "vleb-independent-audit-v1", "eligible": False, "verdict": "unresolved",
                "issues": [{"code": "invalid_mutation_catalog", "message": "catalog must contain active mutation IDs and well-formed mutation objects"}]}

    schema_check(task, inputs.task_schema, "task contract", issues)
    schema_check(receipt, inputs.receipt_schema, "receipt", issues)
    if any(issue["code"].startswith("schema_") for issue in issues):
        return {"schema_version": "vleb-independent-audit-v1", "eligible": False, "verdict": "fail", "issues": issues}
    issues.extend(requirement_traceability_issues(task))

    # The receipt cannot choose its own trust roots: the evaluator supplies these pins independently.
    for label, path, claimed, pin in (
        ("task contract", inputs.task_contract, receipt.get("task_contract_sha256"), inputs.expected_task_sha256),
        ("protocol", inputs.protocol, receipt.get("protocol_sha256"), inputs.expected_protocol_sha256),
        ("candidate patch", inputs.candidate_patch, receipt.get("candidate_patch_sha256"), inputs.expected_candidate_patch_sha256),
    ):
        if not isinstance(pin, str) or not HEX.fullmatch(pin):
            issues.append({"code": "invalid_trusted_pin", "message": f"{label}: external trusted SHA-256 pin missing or malformed"})
            continue
        try:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            issues.append({"code": "missing_evidence", "message": f"{label}: {exc}"})
            continue
        if actual != pin or claimed != pin:
            issues.append({"code": "trusted_pin_mismatch", "message": f"{label}: file and receipt must match evaluator-supplied pin"})
    try:
        actual_auditor = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not HEX.fullmatch(inputs.expected_auditor_sha256) or actual_auditor != inputs.expected_auditor_sha256:
            issues.append({"code": "auditor_identity_mismatch", "message": "auditor source differs from independently pinned implementation"})
        if not HEX.fullmatch(inputs.expected_verifier_binary_sha256):
            issues.append({"code": "invalid_verifier_binary_pin", "message": "independent verifier binary SHA-256 pin missing or malformed"})
        elif receipt.get("verifier", {}).get("verifier_binary_sha256") != inputs.expected_verifier_binary_sha256:
            issues.append({"code": "receipt_verifier_identity_mismatch", "message": "receipt verifier binary digest does not match independent evaluator pin"})
    except OSError as exc:
        issues.append({"code": "auditor_identity_unavailable", "message": str(exc)})

    if receipt.get("task_id") != task.get("task_id"):
        issues.append({"code": "task_identity_mismatch", "message": "receipt and task task_id differ"})
    if catalog.get("task_id") != task.get("task_id"):
        issues.append({"code": "mutation_catalog_task_mismatch", "message": "mutation catalog and task task_id differ"})
    catalog_ids = [row["mutation_id"] for row in catalog_rows]
    if len(catalog_ids) != len(set(catalog_ids)):
        issues.append({"code": "duplicate_catalog_mutation_id", "message": "catalog mutation IDs must be unique"})
    active_ids = active["required_mutation_ids"]
    if any(not isinstance(mid, str) or not mid for mid in active_ids) or len(active_ids) != len(set(active_ids)):
        issues.append({"code": "invalid_mutation_catalog", "message": "active mutation IDs must be unique non-empty strings"})

    # Headline scores require a private, independently reviewed task and real two-sided controls.
    classification = task.get("classification", {})
    if classification.get("exposure") not in ("private_holdout", "private_confirmatory"):
        issues.append({"code": "task_not_score_eligible", "message": "public-development tasks cannot enter headline scores"})
    if classification.get("reviewer_count", 0) < 1:
        issues.append({"code": "oracle_not_independently_reviewed", "message": "task lacks an independent reviewer"})
    oracle = task.get("oracle_contract", {})
    review = oracle.get("oracle_review", {})
    if review.get("independent_reviewer_count", 0) < 1 or review.get("mutation_equivalence_reviewed") is not True:
        issues.append({"code": "oracle_not_independently_reviewed", "message": "oracle and mutant equivalence need independent review"})

    execution_rows = oracle.get("test_execution_map", [])
    required_ids = oracle.get("required_test_ids", [])
    mapping: dict[str, list[str]] = {}
    owner: dict[str, str] = {}
    for row in execution_rows:
        tid = row.get("test_id")
        execution_ids = row.get("execution_ids", [])
        if not isinstance(tid, str) or tid in mapping:
            issues.append({"code": "invalid_execution_map", "message": "logical test IDs must be unique strings"})
            continue
        if not isinstance(execution_ids, list) or not execution_ids or any(not isinstance(name, str) or not name for name in execution_ids):
            issues.append({"code": "invalid_execution_map", "message": f"{tid!r} needs concrete test names"})
            continue
        mapping[tid] = execution_ids
        for name in execution_ids:
            if name in owner and owner[name] != tid:
                issues.append({"code": "execution_mapping_reused", "message": f"concrete test {name!r} is mapped to more than one logical ID"})
            owner[name] = tid
    if len(required_ids) != len(set(required_ids)) or set(mapping) != set(required_ids) or len(execution_rows) != len(required_ids):
        issues.append({"code": "execution_map_incomplete", "message": "execution map must cover the required logical test inventory exactly once"})

    ref_hashes = oracle.get("reference_evidence_sha256", {})
    ref_paths = oracle.get("reference_evidence_artifacts", {})
    for name in ("baseline_log", "reference_log", "reference_patch", "test_bundle"):
        if name not in ref_hashes or name not in ref_paths:
            issues.append({"code": "oracle_control_evidence_missing", "message": f"task lacks reference evidence path/hash for {name}"})
            continue
        raw = checked_artifact(inputs.evidence_root, ref_paths[name], ref_hashes[name], "task admission " + name, issues)
        if raw is None:
            continue
        if name == "baseline_log":
            tests = parse_test_log(raw)
            mapped = {test for rows in mapping.values() for test in rows}
            if not any(tests.get(test) == "FAILED" for test in mapped) or b"test result: FAILED" not in raw:
                issues.append({"code": "baseline_control_not_demonstrated", "message": "baseline must fail a preregistered behavior test"})
        if name == "reference_log":
            tests = parse_test_log(raw)
            missing = sorted(test for rows in mapping.values() for test in rows if tests.get(test) != "ok")
            if missing or b"test result: ok." not in raw or b"test result: FAILED" in raw:
                issues.append({"code": "reference_control_not_demonstrated", "message": "reference must pass every mapped test; missing=" + repr(missing)})

    # Runtime/checkout identity must match evaluator-provided values, not merely the receipt's own claims.
    env = receipt.get("environment", {})
    runtime = env.get("runtime_identity", {})
    checkout = env.get("checkout_identity", {})
    if checkout.get("expected_revision") != inputs.expected_revision or checkout.get("observed_revision") != inputs.expected_revision:
        issues.append({"code": "checkout_identity_mismatch", "message": "receipt revision differs from externally expected revision"})
    if checkout.get("clean_worktree") is not True:
        issues.append({"code": "checkout_not_clean", "message": "tested checkout must be clean after the candidate was materialized"})
    if not IMAGE.fullmatch(inputs.expected_image_id) or runtime.get("immutable_image_id") != inputs.expected_image_id:
        issues.append({"code": "runtime_identity_mismatch", "message": "immutable runtime image differs from external image pin"})
    if runtime.get("network_mode_observed") is not True:
        issues.append({"code": "runtime_network_unobserved", "message": "runtime network policy was not independently observed"})
    if env.get("trust_boundary") != "independently_inspected":
        issues.append({"code": "trust_boundary_unverified", "message": "external attestations are unsupported by this auditor; inspected identity required"})
    task_env = task.get("environment", {})
    policy = receipt.get("agent", {}).get("tool_policy", {})
    if task_env.get("platform") != env.get("platform"):
        issues.append({"code": "platform_mismatch", "message": "runtime platform differs from task"})
    if task_env.get("network_mode") != runtime.get("network_mode"):
        issues.append({"code": "network_policy_mismatch", "message": "runtime network mode differs from task"})
    if task_env.get("network_mode") == "none" and (
        policy.get("container_network") != "none" or policy.get("browser_allowed") is not False or policy.get("external_retrieval_allowed") is not False
    ):
        issues.append({"code": "agent_network_policy_mismatch", "message": "offline task must deny network, browser and external retrieval"})
    task_tools = {(str(row.get("name")), str(row.get("version"))): row for row in task_env.get("toolchains", [])}
    receipt_tools = {(str(row.get("name")), str(row.get("version"))): row for row in env.get("toolchains", [])}
    if not set(task_tools).issubset(receipt_tools):
        issues.append({"code": "toolchain_mismatch", "message": "receipt is missing a required task toolchain"})
    for key, row in task_tools.items():
        pinned = row.get("identity_sha256")
        if pinned is not None and receipt_tools.get(key, {}).get("identity_sha256") != pinned:
            issues.append({"code": "toolchain_identity_mismatch", "message": f"toolchain identity mismatch for {key!r}"})
    sanitation = env.get("sanitization", {})
    for key, expected in {"git_metadata_exit_code": 0, "hooks_disabled": True, "unexpected_refs_checked": True, "unreachable_objects_checked": True}.items():
        if sanitation.get(key) != expected:
            issues.append({"code": "sanitization_incomplete", "message": f"sanitization {key!r} must equal {expected!r}"})
    if sanitation.get("ignored_cache_policy") not in ("removed", "inspected_and_preserved", "not_applicable"):
        issues.append({"code": "sanitization_incomplete", "message": "ignored cache policy not inspected or removed"})

    # Every recorded command artifact must hash correctly and match its declared exit contract.
    commands = receipt.get("commands", [])
    command_ids = [row.get("command_id") for row in commands]
    if len(command_ids) != len(set(command_ids)):
        issues.append({"code": "duplicate_command_id", "message": "command IDs must be unique"})
    for row in commands:
        label = "command " + str(row.get("command_id"))
        checked_artifact(inputs.evidence_root, row.get("stdout_artifact"), row.get("stdout_sha256"), label + " stdout", issues)
        checked_artifact(inputs.evidence_root, row.get("stderr_artifact"), row.get("stderr_sha256"), label + " stderr", issues)
        if row.get("timed_out") is not False or not isinstance(row.get("expected_exit_code"), int) or row.get("exit_code") != row.get("expected_exit_code"):
            issues.append({"code": "command_outcome_mismatch", "message": label + " has timeout/exit mismatch"})

    # Receipt test inventory must exactly equal the task inventory; every raw Cargo log must be all-green.
    results = receipt.get("test_results", [])
    by_test: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(results):
        tid = row.get("test_id")
        if tid in by_test:
            issues.append({"code": "duplicate_test_result", "message": f"duplicate logical test result {tid!r}"})
            continue
        by_test[tid] = row
        raw = checked_artifact(inputs.evidence_root, row.get("log_artifact"), row.get("log_sha256"), f"test result {tid!r}", issues)
        if raw is not None:
            observed = parse_test_log(raw)
            if b"test result: ok." not in raw or b"test result: FAILED" in raw or any(status != "ok" for status in observed.values()):
                issues.append({"code": "candidate_test_suite_not_clean", "message": f"test log for {tid!r} has failures or lacks a passing Cargo summary"})
            for execution in mapping.get(tid, []):
                if observed.get(execution) != "ok":
                    issues.append({"code": "execution_not_observed_passing", "message": f"log for {tid!r} does not show {execution!r} passing"})
        if row.get("status") != "passed" or row.get("exit_code") != 0:
            issues.append({"code": "required_test_not_passed", "message": f"logical test {tid!r} not evidenced as passed"})
    if set(by_test) != set(required_ids):
        issues.append({"code": "test_inventory_mismatch", "message": "receipt test result IDs differ from task inventory"})

    # Mutant must compile and its preregistered target test—not an unrelated test—must fail.
    expected_active = active_ids
    mutation_rows = receipt.get("mutation_results", [])
    by_result: dict[str, dict[str, Any]] = {}
    for row in mutation_rows:
        mid = row.get("mutation_id")
        if mid in by_result:
            issues.append({"code": "duplicate_mutation_result", "message": f"duplicate mutation result {mid!r}"})
        by_result[mid] = row
    if set(by_result) != set(expected_active):
        issues.append({"code": "mutation_inventory_mismatch", "message": "receipt does not exactly match active mutation inventory"})
    catalog_map = {row["mutation_id"]: row for row in catalog_rows}
    for mid in expected_active:
        planned, row = catalog_map.get(mid), by_result.get(mid)
        if planned is None or row is None:
            issues.append({"code": "mutation_evidence_missing", "message": f"missing mutation evidence for {mid!r}"})
            continue
        target = planned.get("expected_killer_test")
        if (row.get("mutation_class") != planned.get("mutation_class") or row.get("expected_killer_test") != target
                or row.get("equivalence_status") != planned.get("expected_equivalence", "non_equivalent")):
            issues.append({"code": "mutation_contract_mismatch", "message": f"mutation {mid!r} differs from preregistered contract"})
        if row.get("compilable") is not True or row.get("compile_exit_code") != 0:
            issues.append({"code": "mutation_not_compilable", "message": f"mutation {mid!r} did not pass compilation"})
        compile_raw = checked_artifact(inputs.evidence_root, row.get("compile_log_artifact"), row.get("compile_log_sha256"), mid + " compile log", issues)
        test_raw = checked_artifact(inputs.evidence_root, row.get("test_log_artifact"), row.get("test_log_sha256"), mid + " test log", issues)
        if compile_raw is not None and test_raw is not None:
            if b"Finished" not in compile_raw or b"Executable" not in compile_raw:
                issues.append({"code": "mutation_compile_evidence_unconvincing", "message": f"{mid!r} compile log lacks Cargo build evidence"})
            if row.get("evidence_sha256") != digest(compile_raw + b"\0" + test_raw):
                issues.append({"code": "mutation_evidence_digest_mismatch", "message": f"{mid!r} combined compile/test digest mismatch"})
            parsed = parse_test_log(test_raw)
            failed = sorted(name for name, status in parsed.items() if status == "FAILED")
            if sorted(row.get("failed_test_names", [])) != failed:
                issues.append({"code": "mutation_failed_test_inventory_mismatch", "message": f"{mid!r} failed-test list differs from raw log"})
            if target not in parsed or parsed.get(target) != "FAILED" or target not in row.get("failed_test_names", []):
                issues.append({"code": "mutation_not_killed_by_target", "message": f"{mid!r} did not fail target test {target!r}"})
        if row.get("test_exit_code") in (None, 0) or row.get("observed_verdict") != "rejected":
            issues.append({"code": "mutation_survived_or_unresolved", "message": f"{mid!r} is not a verified semantic kill"})

    # Differential evidence must cover every declared observable effect.
    checks = receipt.get("differential_checks", [])
    ids = [row.get("check_id") for row in checks]
    if len(ids) != len(set(ids)):
        issues.append({"code": "duplicate_differential_check_id", "message": "differential check IDs must be unique"})
    equal_effects = {row.get("observable") for row in checks if row.get("status") == "equal"}
    missing_effects = sorted(set(task.get("behavior_contract", {}).get("observable_effects", [])) - equal_effects)
    if missing_effects:
        issues.append({"code": "differential_effect_coverage_incomplete", "message": "missing equal comparison for " + repr(missing_effects)})
    if not checks:
        issues.append({"code": "differential_evidence_missing", "message": "no differential comparison evidence"})
    for row in checks:
        label = "differential " + str(row.get("check_id"))
        checked_artifact(inputs.evidence_root, row.get("details_artifact"), row.get("details_sha256"), label + " details", issues)
        if row.get("status") == "equal":
            if row.get("expected_sha256") != row.get("observed_sha256") or row.get("mismatch_count") != 0:
                issues.append({"code": "differential_mismatch", "message": label + " reports equal with mismatching digests/count"})
        elif row.get("status") != "not_applicable":
            issues.append({"code": "differential_unresolved", "message": label + " is not resolved equal"})

    timing = receipt.get("timing", {})
    try:
        start = datetime.fromisoformat(str(timing.get("started_at", "")).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(timing.get("ended_at", "")).replace("Z", "+00:00"))
        elapsed = int((end - start).total_seconds() * 1000)
        wall = timing.get("wall_clock_ms")
        if start.tzinfo is None or end.tzinfo is None or elapsed < 0 or not isinstance(wall, int) or wall < 0 or abs(elapsed - wall) > 2000:
            issues.append({"code": "timing_evidence_mismatch", "message": "wall time inconsistent with timezone-aware timestamps"})
    except (TypeError, ValueError, OverflowError):
        issues.append({"code": "timing_evidence_invalid", "message": "invalid ISO-8601 timestamps"})
    cost = receipt.get("cost", {})
    if cost.get("cost_accounting_complete") is not True or any(
        not isinstance(cost.get(key), (int, float)) or cost.get(key) < 0
        for key in ("model_cost_usd", "tool_cost_usd", "compute_cost_usd")
    ):
        issues.append({"code": "cost_accounting_incomplete", "message": "complete non-negative model/tool/compute cost required"})
    verifier = receipt.get("verifier", {})
    if receipt.get("outcome") != "pass":
        issues.append({"code": "receipt_not_claiming_pass", "message": "receipt outcome is not pass"})
    if verifier.get("receipt_recomputed") is not True or verifier.get("required_test_coverage") != 1 or verifier.get("environment_match") != "match" or verifier.get("verdict") != "pass":
        issues.append({"code": "self_reported_verifier_fields_inconsistent", "message": "receipt's verifier summary contradicts pass eligibility"})

    eligible = not issues
    unresolved_prefixes = ("missing_", "invalid_", "schema_", "unsafe_", "oracle_", "task_not_score", "execution_not_observed")
    return {
        "schema_version": "vleb-independent-audit-v1",
        "eligible": eligible,
        "verdict": "pass" if eligible else ("unresolved" if any(row["code"].startswith(unresolved_prefixes) for row in issues) else "fail"),
        "receipt_sha256": digest(inputs.receipt.read_bytes()),
        "auditor_sha256": digest(Path(__file__).read_bytes()),
        "task_id": task.get("task_id"),
        "expected_revision": inputs.expected_revision,
        "expected_image_id": inputs.expected_image_id,
        "required_test_count": len(required_ids),
        "observed_test_result_count": len(results),
        "required_mutation_count": len(expected_active),
        "observed_mutation_result_count": len(mutation_rows),
        "issues": issues,
    }


def args_for_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("receipt", "task", "task-schema", "receipt-schema", "protocol", "patch", "catalog", "evidence-root"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--expected-task-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-patch-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-verifier-binary-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = args_for_cli(argv)
    result = audit_bundle(AuditInputs(
        receipt=args.receipt, task_contract=args.task, task_schema=args.task_schema,
        receipt_schema=args.receipt_schema, protocol=args.protocol, candidate_patch=args.patch,
        mutation_catalog=args.catalog, evidence_root=args.evidence_root,
        expected_revision=args.expected_revision, expected_image_id=args.expected_image,
        expected_task_sha256=args.expected_task_sha256, expected_protocol_sha256=args.expected_protocol_sha256,
        expected_candidate_patch_sha256=args.expected_patch_sha256, expected_auditor_sha256=args.expected_auditor_sha256,
        expected_verifier_binary_sha256=args.expected_verifier_binary_sha256,
    ))
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
