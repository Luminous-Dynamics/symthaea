#!/usr/bin/env python3
"""CI-only validation for the shared CMAQ pre-run freeze contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SCHEMA_PATH = ROOT / "frozen-run-manifest.schema.json"
EXAMPLE_PATH = ROOT / "examples" / "cmaq-018-development-manifest.example.json"


def artifact_bytes(repo_root: Path, relative_path: str) -> tuple[bytes | None, str | None]:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or relative.as_posix() != relative_path:
        return None, "artifact path must be canonical and repository-relative"
    if any(part in {"", ".", ".."} for part in relative.parts):
        return None, "artifact path traversal is prohibited"

    root = repo_root.resolve()
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None, "symlink artifact paths are prohibited"
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None, "referenced artifact is missing or unreadable"
    if not resolved.is_relative_to(root):
        return None, "artifact resolves outside the repository"
    if not resolved.is_file():
        return None, "artifact path does not identify a regular file"
    try:
        return resolved.read_bytes(), None
    except OSError:
        return None, "artifact bytes could not be read"


def semantic_errors(instance: dict, repo_root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    baseline_ids = [baseline["id"] for baseline in instance["baseline_registry"]]
    if len(baseline_ids) != len(set(baseline_ids)):
        errors.append("duplicate baseline IDs")

    artifact_specs = [
        (instance["candidate"]["config_path"], instance["candidate"]["config_sha256"], "candidate config"),
        (instance["generator"]["config_path"], instance["generator"]["config_sha256"], "generator config"),
        (instance["evaluator"]["metric_bundle_path"], instance["evaluator"]["metric_bundle_sha256"], "metric bundle"),
        *[
            (source["manifest_path"], source["manifest_sha256"], f"data manifest {source['id']}")
            for source in instance["data_sources"]
        ],
        (instance["task_contract"]["observation_schema_path"], instance["task_contract"]["observation_schema_sha256"], "observation schema"),
        (instance["task_contract"]["action_schema_path"], instance["task_contract"]["action_schema_sha256"], "action schema"),
        (instance["task_contract"]["reward_definition_path"], instance["task_contract"]["reward_definition_sha256"], "reward definition"),
        (instance["task_contract"]["wrapper_stack_path"], instance["task_contract"]["wrapper_stack_sha256"], "wrapper stack"),
    ]
    for relative_path, expected_digest, label in artifact_specs:
        raw, path_error = artifact_bytes(repo_root, relative_path)
        if path_error:
            errors.append(f"{label}: {path_error}")
        elif hashlib.sha256(raw).hexdigest() != expected_digest:
            errors.append(f"{label}: SHA-256 does not match frozen manifest")

    baseline_set = set(baseline_ids)
    baseline_by_id = {baseline["id"]: baseline for baseline in instance["baseline_registry"]}
    for comparison in instance["comparisons"]:
        if comparison["treatment_id"] not in baseline_set:
            errors.append("comparison treatment ID is not registered")
        if comparison["control_id"] not in baseline_set:
            errors.append("comparison control ID is not registered")
        if comparison["status"] == "qualified":
            required_match = {"episodes", "seeds", "observations", "actions", "horizon"}
            comparison_baselines = (comparison["treatment_id"], comparison["control_id"])
            if any(baseline_id not in baseline_set for baseline_id in comparison_baselines):
                continue
            for baseline_id in comparison_baselines:
                baseline = baseline_by_id[baseline_id]
                if baseline["eligibility"] != "eligible_for_primary_contrast":
                    errors.append("qualified contrast uses an ineligible baseline")
                if baseline["kind"] == "oracle":
                    errors.append("qualified contrast cannot use evaluator oracle as a treatment/control")
            if not required_match.issubset(set(comparison["matched_dimensions"])):
                errors.append("qualified contrast is missing a core matched dimension")
            if "compute" in comparison["matched_dimensions"]:
                method = instance["budgets"]["matching_method"]
                if method not in {"measured_flops", "measured_wallclock", "matched_multi_resource"}:
                    errors.append("qualified compute contrast lacks a measured matching method")
                for baseline_id in (comparison["treatment_id"], comparison["control_id"]):
                    baseline = baseline_by_id.get(baseline_id)
                    if baseline is None or baseline["cost_accounting"] != "measured":
                        errors.append("qualified compute contrast includes an unmeasured baseline")

    build_status = instance["candidate"]["build_status"]
    artifact_digest = instance["candidate"]["artifact_sha256"]
    if build_status == "source_only_not_compiled" and artifact_digest is not None:
        errors.append("source-only candidate cannot claim a built artifact digest")
    if build_status in {"built_and_hashed", "external_frozen_artifact"} and artifact_digest is None:
        errors.append("built or external candidate must carry an artifact digest")

    if instance["qualification"]["stage"] == "qualification":
        qualification = instance["qualification"]
        if not (
            qualification["candidate_frozen"]
            and qualification["evaluator_frozen"]
            and qualification["thresholds_frozen"]
            and qualification["qualification_data_sealed"]
        ):
            errors.append("qualification stage requires frozen candidate/evaluator/thresholds and sealed data")

    split = instance["sample"]["mechanism_family_split"]
    if split["family_disjoint"]:
        development = set(split["development"])
        calibration = set(split["calibration"])
        qualification = set(split["qualification"])
        if development & calibration or development & qualification or calibration & qualification:
            errors.append("mechanism-family splits overlap despite family_disjoint=true")

    for source in instance["data_sources"]:
        if source["visibility"] == "evaluator_only" and source["candidate_access"]:
            errors.append("evaluator-only data source cannot expose candidate-access fields")
        if source["visibility"] == "candidate_visible" and not source["candidate_access"]:
            errors.append("candidate-visible data source must declare its visible fields")

    if not instance["repository"]["commit"]:
        errors.append("manifest repository commit must be frozen")
    if instance["metrics"]["bundle_sha256"] != instance["evaluator"]["metric_bundle_sha256"]:
        errors.append("metric bundle digest differs between metrics and evaluator")
    if (
        instance["budgets"]["matching_method"] == "candidate_budget_only"
        and instance["budgets"]["measurement_status"] == "measured_hardware_costs"
    ):
        errors.append("candidate-count-only matching cannot claim measured hardware matching")
    if instance["qualification"]["candidate_truth_access"] is not False:
        errors.append("candidate truth access must remain disabled")
    if instance["metrics"]["cross_track_pooling"] is not False:
        errors.append("cross-track pooling must remain disabled")

    for baseline in instance["baseline_registry"]:
        if (
            baseline["cost_accounting"] == "candidate_count_only"
            and baseline["eligibility"] == "eligible_for_primary_contrast"
        ):
            errors.append("candidate-count-only baseline cannot be eligible as compute matched")
        if baseline["kind"] == "oracle" and baseline["eligibility"] != "oracle_only":
            errors.append("oracle baseline must remain oracle-only")
        if (
            baseline["eligibility"] == "eligible_for_primary_contrast"
            and baseline.get("ineligibility_reason") is not None
        ):
            errors.append("eligible baseline cannot carry an ineligibility reason")

    return errors


def fail_closed(validator: Draft202012Validator, instance: dict) -> bool:
    return not list(validator.iter_errors(instance)) and not semantic_errors(instance)


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    errors = sorted(validator.iter_errors(example), key=lambda error: list(error.path))
    if errors:
        raise SystemExit(
            "example manifest failed schema validation:\n"
            + "\n".join(f"  {error.json_path}: {error.message}" for error in errors)
        )

    semantic_failures = semantic_errors(example)
    if semantic_failures:
        raise SystemExit("example failed semantic checks: " + "; ".join(semantic_failures))

    hostile_cases = []

    mutated = copy.deepcopy(example)
    mutated["schema_version"] = "cmaq-frozen-run-manifest-v999"
    hostile_cases.append(("unknown schema version", mutated))

    mutated = copy.deepcopy(example)
    mutated["repository"]["dirty_tree"] = True
    hostile_cases.append(("dirty repository", mutated))

    mutated = copy.deepcopy(example)
    mutated["candidate"]["config_sha256"] = "not-a-sha256"
    hostile_cases.append(("malformed config digest", mutated))

    mutated = copy.deepcopy(example)
    mutated["qualification"]["candidate_truth_access"] = True
    hostile_cases.append(("candidate can access hidden truth", mutated))

    mutated = copy.deepcopy(example)
    mutated["metrics"]["cross_track_pooling"] = True
    hostile_cases.append(("cross-track scalar pooling", mutated))

    mutated = copy.deepcopy(example)
    mutated["task_contract"]["simulator_steps_per_action"] = 0
    hostile_cases.append(("invalid action cadence", mutated))

    mutated = copy.deepcopy(example)
    mutated["budgets"]["matching_method"] = "strictly_compute_matched"
    hostile_cases.append(("unknown compute matching class", mutated))

    mutated = copy.deepcopy(example)
    mutated["baseline_registry"][0]["eligibility"] = "eligible_for_primary_contrast"
    mutated["baseline_registry"][0]["cost_accounting"] = "candidate_count_only"
    mutated["baseline_registry"][0]["ineligibility_reason"] = None
    hostile_cases.append(("candidate count mislabeled as eligible matched baseline", mutated))

    mutated = copy.deepcopy(example)
    mutated["baseline_registry"][-1]["eligibility"] = "eligible_for_primary_contrast"
    mutated["baseline_registry"][-1]["ineligibility_reason"] = None
    hostile_cases.append(("oracle admitted as primary deployable baseline", mutated))

    mutated = copy.deepcopy(example)
    mutated["comparisons"][0]["status"] = "qualified"
    hostile_cases.append(("qualified contrast missing core matched dimensions", mutated))

    mutated = copy.deepcopy(example)
    mutated["comparisons"][0]["status"] = "qualified"
    mutated["comparisons"][0]["matched_dimensions"].append("horizon")
    mutated["comparisons"][0]["matched_dimensions"].append("compute")
    mutated["budgets"]["matching_method"] = "candidate_budget_only"
    hostile_cases.append(("qualified compute claim uses candidate-count-only budget", mutated))

    mutated = copy.deepcopy(example)
    mutated["qualification"]["stage"] = "qualification"
    mutated["qualification"]["qualification_data_sealed"] = False
    hostile_cases.append(("qualification stage without sealed data", mutated))

    mutated = copy.deepcopy(example)
    mutated["candidate"]["build_status"] = "built_and_hashed"
    mutated["candidate"]["artifact_sha256"] = None
    hostile_cases.append(("built candidate without artifact digest", mutated))

    mutated = copy.deepcopy(example)
    mutated["data_sources"][0]["visibility"] = "evaluator_only"
    mutated["data_sources"][0]["candidate_access"] = ["hidden_truth"]
    hostile_cases.append(("evaluator-only dataset exposes fields to candidate", mutated))

    mutated = copy.deepcopy(example)
    mutated["comparisons"][0]["status"] = "qualified"
    mutated["comparisons"][0]["treatment_id"] = "unregistered_model"
    hostile_cases.append(("qualified comparison references unknown baseline", mutated))

    mutated = copy.deepcopy(example)
    mutated["candidate"]["config_path"] = "../README.md"
    hostile_cases.append(("artifact path traversal", mutated))

    mutated = copy.deepcopy(example)
    mutated["candidate"]["config_path"] = "research/cmaq-common/examples/assets/cmaq018/missing-artifact.json"
    hostile_cases.append(("missing artifact path", mutated))

    with tempfile.TemporaryDirectory(prefix=".cmaq-manifest-hostile-", dir=ROOT) as temporary:
        temporary_root = Path(temporary)
        tampered = temporary_root / "tampered-config.json"
        tampered.write_text('{"tampered":true}\n', encoding="utf-8")
        mutated = copy.deepcopy(example)
        mutated["candidate"]["config_path"] = tampered.relative_to(REPO_ROOT).as_posix()
        hostile_cases.append(("artifact bytes do not match declared SHA-256", mutated))

        symlink = temporary_root / "candidate-config-link.json"
        symlink.symlink_to(REPO_ROOT / example["candidate"]["config_path"])
        mutated = copy.deepcopy(example)
        mutated["candidate"]["config_path"] = symlink.relative_to(REPO_ROOT).as_posix()
        hostile_cases.append(("symlink artifact path", mutated))

        for label, hostile in hostile_cases:
            if fail_closed(validator, hostile):
                raise SystemExit(f"hostile manifest was accepted: {label}")

    print(
        json.dumps(
            {
                "schema_version": schema["properties"]["schema_version"]["const"],
                "example_valid": True,
                "semantic_invariants_checked": "passed",
                "content_addressed_artifacts_verified": len(artifact_specs),
                "hostile_cases_rejected": len(hostile_cases),
                "ci_only": True,
                "performance_result_present": example["claim_boundary"]["performance_result_present"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
