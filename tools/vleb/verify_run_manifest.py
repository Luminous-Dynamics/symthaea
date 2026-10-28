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
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
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


def extract_evidence_bundle(bundle_path: Path, destination: Path) -> None:
    """Extract only regular, canonical ZIP members with strict size/path limits."""
    max_members = 100_000
    max_member_bytes = 512 * 1024 * 1024
    max_total_bytes = 2 * 1024 * 1024 * 1024
    destination_root = destination.resolve(strict=True)

    try:
        archive = zipfile.ZipFile(bundle_path, "r")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"evidence bundle is not a valid ZIP archive: {exc}") from exc

    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > max_members:
            raise ValueError(f"evidence bundle member count outside allowed range: {len(infos)}")
        seen: set[str] = set()
        total_size = 0

        for info in infos:
            name = info.filename
            if not name or "\\" in name or "\x00" in name:
                raise ValueError("evidence bundle contains a non-canonical member path")
            rel = PurePosixPath(name)
            normalized = str(rel)
            allowed_name = name[:-1] if info.is_dir() and name.endswith("/") else name
            if (
                rel.is_absolute()
                or any(part in ("", ".", "..") for part in rel.parts)
                or normalized != allowed_name
            ):
                raise ValueError(f"evidence bundle path traversal or ambiguity rejected: {name!r}")
            if normalized in seen:
                raise ValueError(f"duplicate evidence bundle member rejected: {normalized!r}")
            seen.add(normalized)

            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ValueError(f"symlink evidence member rejected: {name!r}")
            if mode and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError(f"special file evidence member rejected: {name!r}")
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted evidence member rejected: {name!r}")
            if info.file_size < 0 or info.file_size > max_member_bytes:
                raise ValueError(f"evidence member exceeds size limit: {name!r}")
            total_size += info.file_size
            if total_size > max_total_bytes:
                raise ValueError("evidence bundle exceeds total uncompressed-size limit")
            if info.file_size > 1024 * 1024 and info.compress_size > 0:
                if info.file_size / info.compress_size > 1000:
                    raise ValueError(f"suspicious compression ratio rejected: {name!r}")

        for info in infos:
            rel = PurePosixPath(info.filename.rstrip("/"))
            target = destination_root.joinpath(*rel.parts)
            if not target.resolve(strict=False).is_relative_to(destination_root):
                raise ValueError(f"evidence member escapes extraction directory: {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with archive.open(info, "r") as source, target.open("xb") as output:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > info.file_size or written > max_member_bytes:
                        raise ValueError(f"evidence member actual size exceeds declared size: {info.filename!r}")
                    output.write(chunk)
            if written != info.file_size:
                raise ValueError(f"evidence member size mismatch: {info.filename!r}")


def attestation_output_issues(raw: bytes, manifest: dict[str, Any]) -> list[dict[str, str]]:
    """Check verified attestation metadata against trusted manifest values.

    GitHub CLI verifies cryptographic validity and signer policy. These extra
    checks bind the verified certificate and subject to this specific run,
    repository ID, source revision/ref, workflow identity and evidence bundle.
    """
    try:
        rows = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        return [{"code": "attestation_json_invalid", "message": str(exc)}]
    if not isinstance(rows, list) or not rows:
        return [{"code": "attestation_json_empty", "message": "verified attestation output must be a non-empty JSON array"}]

    repo = manifest["repository"]
    candidate = manifest["candidate"]
    bundle = manifest["evidence_bundle"]
    provenance = manifest["provenance"]
    expected_uri = f"https://github.com/{repo['full_name']}/{repo['workflow_path']}@{repo['workflow_ref']}"
    expected_run_uri = (
        f"https://github.com/{repo['full_name']}/actions/runs/"
        f"{repo['run_id']}/attempts/{repo['run_attempt']}"
    )
    errors: list[dict[str, str]] = []
    verified_subject_seen = False

    for index, row in enumerate(rows):
        location = f"attestation[{index}]"
        if not isinstance(row, dict):
            errors.append({"code": "attestation_entry_invalid", "message": f"{location} is not an object"})
            continue
        result = row.get("verificationResult")
        if not isinstance(result, dict):
            errors.append({"code": "attestation_result_missing", "message": f"{location} has no verificationResult"})
            continue
        signature = result.get("signature")
        certificate = signature.get("certificate") if isinstance(signature, dict) else None
        if not isinstance(certificate, dict):
            errors.append({"code": "attestation_certificate_missing", "message": f"{location} has no verified certificate"})
            continue

        san = certificate.get("subjectAlternativeName")
        san_value = san.get("value") if isinstance(san, dict) else san
        immutable_checks = (
            ("sourceRepositoryIdentifier", str(repo["repository_id"]), "attestation_repository_id_mismatch"),
            ("sourceRepositoryDigest", repo["workflow_sha"], "attestation_source_digest_mismatch"),
            ("sourceRepositoryRef", repo["workflow_ref"], "attestation_source_ref_mismatch"),
            ("githubWorkflowRepository", repo["full_name"], "attestation_workflow_repository_mismatch"),
            ("githubWorkflowRef", repo["workflow_ref"], "attestation_workflow_ref_mismatch"),
            ("githubWorkflowSHA", repo["workflow_sha"], "attestation_workflow_sha_mismatch"),
            ("buildSignerURI", expected_uri, "attestation_signer_uri_mismatch"),
            ("runInvocationURI", expected_run_uri, "attestation_run_id_mismatch"),
        )
        for key, expected, code in immutable_checks:
            observed = certificate.get(key)
            if observed != expected:
                errors.append({
                    "code": code,
                    "message": f"{location}.{key} expected {expected!r}, observed {observed!r}",
                })
        if san_value != expected_uri:
            errors.append({
                "code": "attestation_certificate_identity_mismatch",
                "message": f"{location} certificate SAN expected {expected_uri!r}, observed {san_value!r}",
            })
        if certificate.get("issuer") != "https://token.actions.githubusercontent.com":
            errors.append({
                "code": "attestation_issuer_mismatch",
                "message": f"{location} certificate issuer is not GitHub Actions OIDC",
            })

        statement = result.get("statement")
        if not isinstance(statement, dict):
            errors.append({"code": "attestation_statement_missing", "message": f"{location} has no statement"})
            continue
        if statement.get("predicateType") != provenance["predicate_type"]:
            errors.append({
                "code": "attestation_predicate_type_mismatch",
                "message": f"{location} predicate type differs from manifest",
            })
        subjects = statement.get("subject")
        matched_subject = False
        if isinstance(subjects, list):
            for subject in subjects:
                digest_obj = subject.get("digest") if isinstance(subject, dict) else None
                if isinstance(digest_obj, dict) and digest_obj.get("sha256") == bundle["sha256"]:
                    matched_subject = True
                    verified_subject_seen = True
                    break
        if not matched_subject:
            errors.append({
                "code": "attestation_subject_digest_mismatch",
                "message": f"{location} statement does not bind the declared evidence-bundle SHA-256",
            })

    if not verified_subject_seen and not any(row["code"] == "attestation_subject_digest_mismatch" for row in errors):
        errors.append({
            "code": "attestation_subject_digest_mismatch",
            "message": "no verified attestation binds the declared evidence bundle",
        })
    return errors


def validate_manifest(
    manifest_path: Path,
    schema_path: Path,
    artifact_root: Path,
    expected_manifest_sha256: str,
    *,
    expected_verifier_sha256: str,
    expected_schema_sha256: str,
    expected_auditor_sha256: str,
    expected_verifier_binary_sha256: str,
    verify_attestation: bool,
    gh_executable: str = "gh",
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    for pin_name, pin in (
        ("manifest", expected_manifest_sha256),
        ("verifier", expected_verifier_sha256),
        ("schema", expected_schema_sha256),
        ("auditor", expected_auditor_sha256),
        ("receipt verifier binary", expected_verifier_binary_sha256),
    ):
        if not HEX.fullmatch(pin):
            return {
                "eligible": False, "manifest_valid": False, "score_eligible": False,
                "verdict": "fail",
                "issues": [{"code": "invalid_trusted_pin", "message": f"expected {pin_name} digest must be lowercase SHA-256"}],
            }
    try:
        observed_verifier_sha = digest(Path(__file__).read_bytes())
        observed_schema_sha = digest(schema_path.read_bytes())
    except OSError as exc:
        return {
            "eligible": False, "manifest_valid": False, "score_eligible": False,
            "verdict": "unresolved",
            "issues": [{"code": "trusted_verifier_input_unavailable", "message": str(exc)}],
        }
    if observed_verifier_sha != expected_verifier_sha256:
        return {
            "eligible": False, "manifest_valid": False, "score_eligible": False,
            "verdict": "fail", "observed_verifier_sha256": observed_verifier_sha,
            "issues": [{"code": "verifier_pin_mismatch", "message": "running verifier does not match caller-supplied trusted digest"}],
        }
    if observed_schema_sha != expected_schema_sha256:
        return {
            "eligible": False, "manifest_valid": False, "score_eligible": False,
            "verdict": "fail", "observed_schema_sha256": observed_schema_sha,
            "issues": [{"code": "schema_pin_mismatch", "message": "manifest schema does not match caller-supplied trusted digest"}],
        }
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

    if manifest.get("auditor", {}).get("source_sha256") != expected_auditor_sha256:
        return {
            "eligible": False, "manifest_valid": False, "score_eligible": False,
            "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha,
            "issues": [{"code": "auditor_pin_mismatch", "message": "manifest-selected auditor differs from evaluator-pinned auditor source"}],
        }
    if manifest.get("auditor", {}).get("verifier_binary_sha256") != expected_verifier_binary_sha256:
        return {
            "eligible": False, "manifest_valid": False, "score_eligible": False,
            "verdict": "fail", "observed_manifest_sha256": observed_manifest_sha,
            "issues": [{"code": "receipt_verifier_binary_pin_mismatch", "message": "receipt-verifier binary differs from evaluator-pinned binary digest"}],
        }

    repo = manifest["repository"]
    task = manifest["task"]
    campaign = manifest["mutation_campaign"]
    bundle = manifest["evidence_bundle"]
    provenance = manifest["provenance"]
    if manifest["run_id"] != repo["run_id"]:
        issues.append({"code": "run_id_mismatch", "message": "manifest and repository run IDs differ"})
    if manifest["candidate"]["checkout_revision"] != repo["expected_head_sha"]:
        issues.append({"code": "candidate_head_mismatch", "message": "candidate checkout revision differs from evaluator-pinned expected head"})
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

    # The attested ZIP is the trust boundary. Verify it first, extract safely,
    # and hash/audit only members actually contained inside that sealed bundle.
    resolved: dict[str, Path] = {}
    bundle_path: Path | None = None
    try:
        bundle_path = safe_file(artifact_root, bundle["path"])
        observed_bundle_sha = digest(bundle_path.read_bytes())
        resolved["evidence bundle"] = bundle_path
        if observed_bundle_sha != bundle["sha256"]:
            issues.append({
                "code": "artifact_digest_mismatch",
                "message": f"evidence bundle: expected {bundle['sha256']}, observed {observed_bundle_sha}",
            })
    except (OSError, ValueError) as exc:
        issues.append({"code": "artifact_path_invalid", "message": f"evidence bundle: {exc}"})

    temporary_evidence = tempfile.TemporaryDirectory(prefix="vleb-sealed-evidence-")
    evidence_root = Path(temporary_evidence.name)
    if bundle_path is not None:
        try:
            extract_evidence_bundle(bundle_path, evidence_root)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            issues.append({"code": "evidence_bundle_invalid", "message": str(exc)})

    hashed_fields = [
        ("task contract", task["contract_path"], task["contract_sha256"]),
        ("protocol", manifest["protocol"]["path"], manifest["protocol"]["sha256"]),
        ("candidate patch", manifest["candidate"]["patch_path"], manifest["candidate"]["patch_sha256"]),
        ("auditor source", manifest["auditor"]["source_path"], manifest["auditor"]["source_sha256"]),
        ("task schema", manifest["auditor"]["task_schema_path"], manifest["auditor"]["task_schema_sha256"]),
        ("receipt schema", manifest["auditor"]["receipt_schema_path"], manifest["auditor"]["receipt_schema_sha256"]),
        ("mutation catalog", campaign["catalog_path"], campaign["catalog_sha256"]),
        ("mutation results", campaign["results_path"], campaign["results_sha256"]),
        ("evidence receipt", bundle["receipt_path"], bundle["receipt_sha256"]),
    ]
    for label, relative, expected in hashed_fields:
        try:
            path = safe_file(evidence_root, relative)
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
            expected_signer_workflow = (
                f"{repo['full_name']}/{provenance['signer_workflow']}"
            )
            argv = [
                gh_executable, "attestation", "verify", str(subject),
                "--repo", repo["full_name"],
                "--signer-repo", provenance["signer_repository"],
                "--signer-workflow", expected_signer_workflow,
                "--source-digest", repo["workflow_sha"],
                "--source-ref", repo["workflow_ref"],
                "--predicate-type", provenance["predicate_type"],
                "--format", "json",
            ]
            try:
                completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=120)
                attestation_result["command_sha256"] = digest("\0".join(argv).encode())
                attestation_result["output_sha256"] = digest(completed.stdout)
                if completed.returncode == 0:
                    claim_issues = attestation_output_issues(completed.stdout, manifest)
                    if claim_issues:
                        attestation_result["status"] = "claims_mismatch"
                        issues.extend(claim_issues)
                    else:
                        attestation_result["verified"] = True
                        attestation_result["status"] = "verified"
                else:
                    issues.append({"code": "attestation_verification_failed", "message": completed.stdout.decode("utf-8", errors="replace")[-4000:]})
            except (OSError, subprocess.TimeoutExpired) as exc:
                issues.append({"code": "attestation_verification_unavailable", "message": str(exc)})

    receipt_audit: dict[str, Any] = {"required": manifest["run_kind"] == "confirmatory", "status": "not_required"}
    if manifest["run_kind"] == "confirmatory" and not issues and attestation_result["verified"]:
        auditor = resolved.get("auditor source")
        task_schema = resolved.get("task schema")
        receipt_schema = resolved.get("receipt schema")
        required_paths = [
            resolved.get("evidence receipt"), resolved.get("task contract"),
            resolved.get("protocol"), resolved.get("candidate patch"),
            resolved.get("mutation catalog"), auditor, task_schema, receipt_schema,
        ]
        if any(path is None for path in required_paths):
            issues.append({"code": "independent_receipt_audit_inputs_missing", "message": "a required auditor input failed path/digest validation"})
            receipt_audit["status"] = "unresolved"
        else:
            argv = [
                sys.executable, str(auditor),
                "--receipt", str(resolved["evidence receipt"]),
                "--task", str(resolved["task contract"]),
                "--task-schema", str(task_schema),
                "--receipt-schema", str(receipt_schema),
                "--protocol", str(resolved["protocol"]),
                "--patch", str(resolved["candidate patch"]),
                "--catalog", str(resolved["mutation catalog"]),
                "--evidence-root", str(evidence_root.resolve()),
                "--expected-revision", manifest["candidate"]["checkout_revision"],
                "--expected-image", manifest["runtime"]["immutable_image_id"],
                "--expected-task-sha256", task["contract_sha256"],
                "--expected-protocol-sha256", manifest["protocol"]["sha256"],
                "--expected-patch-sha256", manifest["candidate"]["patch_sha256"],
                "--expected-auditor-sha256", expected_auditor_sha256,
                "--expected-verifier-binary-sha256", expected_verifier_binary_sha256,
            ]
            try:
                completed = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=180)
                receipt_audit["command_sha256"] = digest("\\0".join(argv).encode())
                receipt_audit["output_sha256"] = digest(completed.stdout)
                parsed = None
                if completed.stdout:
                    try:
                        parsed = json.loads(completed.stdout.decode("utf-8", errors="strict"))
                    except (UnicodeError, json.JSONDecodeError):
                        parsed = None
                receipt_audit["verdict"] = parsed.get("verdict") if isinstance(parsed, dict) else None
                receipt_audit["issue_count"] = len(parsed.get("issues", [])) if isinstance(parsed, dict) else None
                if completed.returncode == 0 and isinstance(parsed, dict) and parsed.get("eligible") is True:
                    receipt_audit["status"] = "passed"
                else:
                    receipt_audit["status"] = "failed"
                    issues.append({"code": "independent_receipt_audit_failed", "message": completed.stdout.decode("utf-8", errors="replace")[-4000:]})
            except (OSError, subprocess.TimeoutExpired) as exc:
                receipt_audit["status"] = "unresolved"
                issues.append({"code": "independent_receipt_audit_unavailable", "message": str(exc)})

    manifest_valid = not issues and (
        manifest["run_kind"] != "confirmatory" or (
            attestation_result["verified"] and receipt_audit["status"] == "passed"
        )
    )
    score_eligible = manifest_valid and manifest["run_kind"] == "confirmatory"
    temporary_evidence.cleanup()
    return {
        "schema_version": "vleb-run-manifest-verification-v1",
        "eligible": manifest_valid,
        "manifest_valid": manifest_valid,
        "score_eligible": score_eligible,
        "verdict": "pass" if manifest_valid else "fail",
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
        "independent_receipt_audit": receipt_audit,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("run-manifest.schema.json"))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-verifier-binary-sha256", required=True)
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
        expected_verifier_sha256=args.expected_verifier_sha256,
        expected_schema_sha256=args.expected_schema_sha256,
        expected_auditor_sha256=args.expected_auditor_sha256,
        expected_verifier_binary_sha256=args.expected_verifier_binary_sha256,
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
