#!/usr/bin/env python3
"""Verify a VLEB run manifest against external pins and actual evidence bytes.

For confirmatory runs this CLI also verifies the evidence-bundle provenance with
GitHub CLI. A schema-valid manifest or matching hashes alone do not prove that
an execution actually occurred; a trusted signer/workflow policy is mandatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit("jsonschema==4.26.0 is required; refusing to skip schema validation") from exc

HEX = re.compile(r"^[a-f0-9]{64}$")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("artifact path must be a non-empty POSIX relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("absolute/traversal artifact path rejected")
    root_resolved = root.resolve(strict=True)
    current = root_resolved
    for component in rel.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("symlink evidence path rejected")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ValueError("artifact must be a regular file inside the evidence root")
    return resolved


def validate_manifest(
    manifest_path: Path,
    schema_path: Path,
    artifact_root: Path,
    expected_manifest_sha256: str,
    *,
    verify_attestation: bool,
    gh_executable: str = "gh",
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    if not HEX.fullmatch(expected_manifest_sha256):
        return {"eligible": False, "verdict": "fail", "issues": [{"code": "invalid_manifest_pin", "message": "expected manifest digest must be lowercase SHA-256"}]}
    try:
        raw_manifest = manifest_path.read_bytes()
    except OSError as exc:
        return {"eligible": False, "verdict": "unresolved", "issues": [{"code": "manifest_unavailable", "message": str(exc)}]}
    observed_manifest_sha = digest(raw_manifest)
    if observed_manifest_sha != expected_manifest_sha256:
        return {
            "eligible": False,
            "verdict": "fail",
            "observed_manifest_sha256": observed_manifest_sha,
            "issues": [{"code": "manifest_pin_mismatch", "message": "manifest does not match caller-supplied trusted digest"}],
        }
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (UnicodeError, OSError, json.JSONDecodeError) as exc:
        return {"eligible": False, "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha, "issues": [{"code": "invalid_json", "message": str(exc)}]}
    try:
        Draft202012Validator.check_schema(schema)
        errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(manifest))
    except Exception as exc:
        return {"eligible": False, "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha, "issues": [{"code": "schema_validator_error", "message": str(exc)}]}
    for error in errors:
        path = ".".join(map(str, error.absolute_path)) or "$"
        issues.append({"code": "schema_invalid", "message": f"{path}: {error.message}"})
    if issues:
        return {"eligible": False, "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha, "issues": issues}
    if not isinstance(manifest, dict):
        return {"eligible": False, "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha, "issues": [{"code": "manifest_root_invalid", "message": "manifest root must be an object"}]}

    repo = manifest["repository"]
    task = manifest["task"]
    campaign = manifest["mutation_campaign"]
    bundle = manifest["evidence_bundle"]
    provenance = manifest["provenance"]
    if manifest["run_id"] != repo["run_id"]:
        issues.append({"code": "run_id_mismatch", "message": "manifest and repository run IDs differ"})
    if provenance["subject_path"] != bundle["path"]:
        issues.append({"code": "attested_subject_mismatch", "message": "attestation subject must be the evidence bundle"})
    active_ids = campaign["active_mutation_ids"]
    if len(active_ids) != len(set(active_ids)):
        issues.append({"code": "duplicate_active_mutation_id", "message": "active mutation IDs must be unique"})
    counts = [campaign["killed"], campaign["survived"], campaign["unresolved"]]
    if sum(counts) != len(active_ids):
        issues.append({"code": "mutation_accounting_inconsistent", "message": "killed + survived + unresolved must equal the active mutation count"})
    if campaign["killed"] + campaign["survived"] > campaign["compilable"]:
        issues.append({"code": "mutation_compilable_count_inconsistent", "message": "killed and survived mutants cannot exceed compilable mutants"})
    if manifest["run_kind"] == "confirmatory":
        if task["exposure"] not in ("private_holdout", "private_confirmatory"):
            issues.append({"code": "confirmatory_task_not_private", "message": "confirmatory task must be private holdout or confirmatory"})
        if task["independent_reviewer_count"] < 1:
            issues.append({"code": "confirmatory_task_unreviewed", "message": "confirmatory task requires independent review"})
        if not task["baseline_empty_patch_failed"] or not task["reference_solution_passed"]:
            issues.append({"code": "confirmatory_oracle_controls_missing", "message": "confirmatory task requires baseline failure and reference success"})
        if not provenance["attestation_required"] or provenance["provider"] != "github_actions_sigstore":
            issues.append({"code": "confirmatory_attestation_missing", "message": "confirmatory manifest requires GitHub Actions/Sigstore provenance"})
        if provenance["signer_repository"] != repo["full_name"] or provenance["signer_workflow"] != repo["workflow_path"]:
            issues.append({"code": "confirmatory_signer_mismatch", "message": "expected signer repository/workflow must match the recorded run"})
    if issues:
        return {
            "eligible": False, "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha,
            "run_id": manifest["run_id"], "run_kind": manifest["run_kind"], "issues": issues,
        }

    hashed_fields = [
        ("task contract", task["contract_path"], task["contract_sha256"]),
        ("protocol", manifest["protocol"]["path"], manifest["protocol"]["sha256"]),
        ("candidate patch", manifest["candidate"]["patch_path"], manifest["candidate"]["patch_sha256"]),
        ("auditor source", manifest["auditor"]["source_path"], manifest["auditor"]["source_sha256"]),
        ("mutation catalog", campaign["catalog_path"], campaign["catalog_sha256"]),
        ("mutation results", campaign["results_path"], campaign["results_sha256"]),
        ("evidence bundle", bundle["path"], bundle["sha256"]),
        ("evidence receipt", bundle["receipt_path"], bundle["receipt_sha256"]),
    ]
    resolved: dict[str, Path] = {}
    for label, relative, expected in hashed_fields:
        try:
            path = safe_file(artifact_root, relative)
            observed = digest(path.read_bytes())
            resolved[label] = path
            if observed != expected:
                issues.append({"code": "artifact_digest_mismatch", "message": f"{label}: expected {expected}, observed {observed}"})
        except (OSError, ValueError) as exc:
            issues.append({"code": "artifact_path_invalid", "message": f"{label}: {exc}"})

    attestation_result: dict[str, Any] = {"required": provenance["attestation_required"], "verified": False, "status": "not_required"}
    if manifest["run_kind"] == "confirmatory" or provenance["attestation_required"]:
        attestation_result["status"] = "not_verified"
        subject = resolved.get("evidence bundle")
        if subject is None:
            issues.append({"code": "attestation_subject_missing", "message": "cannot verify an unavailable evidence bundle"})
        elif not verify_attestation:
            issues.append({"code": "attestation_verification_disabled", "message": "confirmatory mode cannot skip attestation verification"})
        else:
            argv = [
                gh_executable, "attestation", "verify", str(subject),
                "--repo", provenance["signer_repository"],
                "--signer-workflow", provenance["signer_workflow"],
                "--predicate-type", provenance["predicate_type"],
            ]
            try:
                completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=120)
                attestation_result["command_sha256"] = digest("\0".join(argv).encode())
                attestation_result["output_sha256"] = digest(completed.stdout)
                if completed.returncode == 0:
                    attestation_result["verified"] = True
                    attestation_result["status"] = "verified"
                else:
                    issues.append({"code": "attestation_verification_failed", "message": completed.stdout.decode("utf-8", errors="replace")[-4000:]})
            except (OSError, subprocess.TimeoutExpired) as exc:
                issues.append({"code": "attestation_verification_unavailable", "message": str(exc)})

    eligible = not issues and (
        manifest["run_kind"] != "confirmatory" or attestation_result["verified"]
    )
    return {
        "schema_version": "vleb-run-manifest-verification-v1",
        "eligible": eligible,
        "verdict": "pass" if eligible else "fail",
        "manifest_sha256": observed_manifest_sha,
        "run_id": manifest["run_id"],
        "run_kind": manifest["run_kind"],
        "repository": repo["full_name"],
        "workflow_path": repo["workflow_path"],
        "workflow_sha": repo["workflow_sha"],
        "expected_head_sha": repo["expected_head_sha"],
        "task_id": task["task_id"],
        "evidence_bundle_sha256": bundle["sha256"],
        "attestation": attestation_result,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("run-manifest.schema.json"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--gh-executable", default="gh")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        verify_attestation = manifest.get("run_kind") == "confirmatory" or manifest.get("provenance", {}).get("attestation_required") is True
    except (OSError, json.JSONDecodeError, UnicodeError):
        verify_attestation = True

    result = validate_manifest(
        args.manifest,
        args.schema,
        args.artifact_root,
        args.expected_manifest_sha256,
        verify_attestation=verify_attestation,
        gh_executable=args.gh_executable,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("eligible") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
