#!/usr/bin/env python3
"""Fail-closed verifier-artifact provenance policy for GitHub artifact attestations.

Cryptographic signature/identity verification is delegated to an operator-pinned
GitHub CLI via gh attestation verify. This script additionally checks the verified
SLSA provenance statement against an independently provisioned policy. It does not
prove the verifier process was uncompromised or grant operational authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable

from scripts.package_cyber_crucible_verifier import BundleError, verify_bundle

POLICY_VERSION = "cyber-crucible-verifier-provenance-policy.v1"
IN_TOTO_STATEMENT_V1 = "https://in-toto.io/Statement/v1"
SLSA_PROVENANCE_V1 = "https://slsa.dev/provenance/v1"
GITHUB_OIDC_ISSUER = "https://token.actions.githubusercontent.com"
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
ALLOWED_EXTERNAL_PARAMETER_KEYS = {"workflow", "inputs", "release", "vars"}

POLICY_FIELDS = {
    "schema_version", "policy_id", "repository", "source_ref", "source_commit",
    "workflow_path", "signer_repository", "signer_workflow_path",
    "signer_workflow_commit", "oidc_issuer", "builder_id",
    "build_type", "artifact_name", "artifact_sha256", "gh_cli_path",
    "gh_cli_sha256", "expected_external_parameters", "deny_self_hosted_runners",
}


class ProvenanceVerificationError(ValueError):
    """The artifact or its provenance does not satisfy the trusted policy."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceVerificationError(f"duplicate JSON member: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProvenanceVerificationError(f"non-JSON numeric constant: {value}")


def load_json(path: str | Path) -> Any:
    """Load UTF-8 JSON while refusing duplicate member names and non-JSON constants."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(
                handle, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProvenanceVerificationError(f"cannot load strict JSON input: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON for policy fingerprinting and type-sensitive comparisons."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProvenanceVerificationError(f"value is not canonicalizable JSON: {exc}") from exc


def json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ProvenanceVerificationError(f"{label} must be a non-empty trimmed string")
    if any(ord(character) < 0x20 for character in value):
        raise ProvenanceVerificationError(f"{label} must not contain control characters")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX_256.fullmatch(value):
        raise ProvenanceVerificationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _git_sha1(value: Any, label: str) -> str:
    if not isinstance(value, str) or not GIT_SHA1.fullmatch(value):
        raise ProvenanceVerificationError(f"{label} must be a lowercase 40-character Git commit")
    return value


def _safe_regular_file(path_value: Any, label: str, *, executable: bool = False) -> Path:
    raw = _text(path_value, label)
    path = Path(raw)
    if not path.is_absolute():
        raise ProvenanceVerificationError(f"{label} must be an absolute path")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ProvenanceVerificationError(f"{label} cannot be inspected") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProvenanceVerificationError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise ProvenanceVerificationError(f"{label} must be executable")
    return path


def _hash_regular_file(path: Path, label: str) -> str:
    """Hash a regular file without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ProvenanceVerificationError(f"{label} could not be opened safely") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ProvenanceVerificationError(f"{label} must be a regular file")
        digest = hashlib.sha256()
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)



MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


def _copy_artifact_to_private_snapshot(source: Path, destination: Path) -> str:
    """Hash one opened artifact while copying its exact bytes into a private temp dir.

    The attestation verifier and bundle parser must inspect the same byte snapshot
    that was hashed against operator policy. Passing the original path to a second
    process allows a writable-path race between digest calculation and file open.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise ProvenanceVerificationError("verifier artifact could not be opened safely") from exc

    destination_fd: int | None = None
    try:
        source_info = os.fstat(source_fd)
        if not stat.S_ISREG(source_info.st_mode):
            raise ProvenanceVerificationError("verifier artifact must be a regular file")
        if source_info.st_size < 0 or source_info.st_size > MAX_ARTIFACT_BYTES:
            raise ProvenanceVerificationError("verifier artifact exceeds the 128 MiB size limit")
        create_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            destination_fd = os.open(destination, create_flags, 0o600)
        except OSError as exc:
            raise ProvenanceVerificationError("private verifier snapshot could not be created safely") from exc

        digest = hashlib.sha256()
        total = 0
        with os.fdopen(source_fd, "rb", closefd=True) as source_file:
            source_fd = -1
            with os.fdopen(destination_fd, "wb", closefd=True) as destination_file:
                destination_fd = None
                while True:
                    chunk = source_file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ARTIFACT_BYTES:
                        raise ProvenanceVerificationError("verifier artifact exceeds the 128 MiB size limit")
                    digest.update(chunk)
                    destination_file.write(chunk)
                destination_file.flush()
                os.fsync(destination_file.fileno())
        if total != source_info.st_size:
            raise ProvenanceVerificationError("verifier artifact size changed while taking a snapshot")
        os.chmod(destination, 0o400)
        return digest.hexdigest()
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, dict) or set(policy) != POLICY_FIELDS:
        raise ProvenanceVerificationError("provenance policy has missing or unknown fields")
    if policy["schema_version"] != POLICY_VERSION:
        raise ProvenanceVerificationError("unsupported verifier-provenance policy version")
    _text(policy["policy_id"], "policy.policy_id")
    repository = _text(policy["repository"], "policy.repository")
    if not REPOSITORY.fullmatch(repository):
        raise ProvenanceVerificationError("policy.repository must be owner/repository")
    source_ref = _text(policy["source_ref"], "policy.source_ref")
    if not re.fullmatch(r"refs/(?:heads|tags)/[^/].*", source_ref):
        raise ProvenanceVerificationError("policy.source_ref must be an exact non-empty branch or tag ref")
    _git_sha1(policy["source_commit"], "policy.source_commit")
    signer_repository = _text(policy["signer_repository"], "policy.signer_repository")
    if not REPOSITORY.fullmatch(signer_repository):
        raise ProvenanceVerificationError("policy.signer_repository must be owner/repository")
    signer_workflow_path = _text(policy["signer_workflow_path"], "policy.signer_workflow_path")
    if not signer_workflow_path.startswith(".github/workflows/") or not signer_workflow_path.endswith((".yml", ".yaml")):
        raise ProvenanceVerificationError("policy.signer_workflow_path must identify a workflow YAML file")
    if ".." in Path(signer_workflow_path).parts or signer_workflow_path.startswith("/"):
        raise ProvenanceVerificationError("policy.signer_workflow_path must be repository-relative")
    _git_sha1(policy["signer_workflow_commit"], "policy.signer_workflow_commit")
    workflow_path = _text(policy["workflow_path"], "policy.workflow_path")
    if not workflow_path.startswith(".github/workflows/") or not workflow_path.endswith((".yml", ".yaml")):
        raise ProvenanceVerificationError("policy.workflow_path must identify a workflow YAML file")
    if ".." in Path(workflow_path).parts or workflow_path.startswith("/"):
        raise ProvenanceVerificationError("policy.workflow_path must be a repository-relative path")
    _text(policy["builder_id"], "policy.builder_id")
    _text(policy["build_type"], "policy.build_type")
    _text(policy["artifact_name"], "policy.artifact_name")
    _sha256(policy["artifact_sha256"], "policy.artifact_sha256")
    if policy["oidc_issuer"] != GITHUB_OIDC_ISSUER:
        raise ProvenanceVerificationError("only the explicitly trusted GitHub Actions OIDC issuer is allowed")
    if policy["deny_self_hosted_runners"] is not True:
        raise ProvenanceVerificationError("high-assurance verifier policy must deny self-hosted runners")

    expected_external = policy["expected_external_parameters"]
    if not isinstance(expected_external, dict) or "workflow" not in expected_external:
        raise ProvenanceVerificationError("policy.expected_external_parameters must contain workflow")
    if set(expected_external) - ALLOWED_EXTERNAL_PARAMETER_KEYS:
        raise ProvenanceVerificationError("policy contains unsupported external parameter names")
    if "inputs" in expected_external and "release" in expected_external:
        raise ProvenanceVerificationError("SLSA workflow externalParameters cannot contain both inputs and release")
    expected_workflow = {
        "repository": f"https://github.com/{repository}",
        "ref": source_ref,
        "path": workflow_path,
    }
    workflow = expected_external["workflow"]
    if not isinstance(workflow, dict) or set(workflow) != {"repository", "ref", "path"}:
        raise ProvenanceVerificationError("expected workflow parameters must contain exactly repository, ref, and path")
    if canonical_json(workflow) != canonical_json(expected_workflow):
        raise ProvenanceVerificationError("expected workflow parameters conflict with policy repository/ref/path")
    for key in ("inputs", "release", "vars"):
        if key in expected_external and (not isinstance(expected_external[key], dict) or not expected_external[key]):
            raise ProvenanceVerificationError(f"policy external parameter {key} must be a non-empty object when present")

    cli_path = _safe_regular_file(policy["gh_cli_path"], "policy.gh_cli_path", executable=True)
    _sha256(policy["gh_cli_sha256"], "policy.gh_cli_sha256")
    if _hash_regular_file(cli_path, "GitHub CLI") != policy["gh_cli_sha256"]:
        raise ProvenanceVerificationError("GitHub CLI binary digest does not match operator policy")
    return policy


def _run_attestation_verification(
    artifact: Path,
    policy: dict[str, Any],
    *,
    runner: Callable[..., Any] | None = None,
    timeout_seconds: int = 90,
) -> Any:
    """Run gh attestation verify without a shell and with exact identity constraints."""
    if runner is None:
        runner = subprocess.run
    cli = Path(policy["gh_cli_path"])
    expected_cli_digest = policy["gh_cli_sha256"]
    command = [
        str(cli), "attestation", "verify", str(artifact),
        "--repo", policy["repository"],
        "--hostname", "github.com",
        "--signer-repo", policy["signer_repository"],
        "--signer-workflow", f'{policy["signer_repository"]}/{policy["signer_workflow_path"]}',
        "--signer-digest", policy["signer_workflow_commit"],
        "--source-ref", policy["source_ref"],
        "--source-digest", policy["source_commit"],
        "--cert-oidc-issuer", policy["oidc_issuer"],
        "--predicate-type", SLSA_PROVENANCE_V1,
        "--format", "json", "--limit", "5",
        "--deny-self-hosted-runners",
    ]
    try:
        with tempfile.TemporaryDirectory(prefix="symthaea-gh-verify-") as isolated_home:
            # Do not inherit GH_HOST, GH_CONFIG_DIR, aliases, pagers, or arbitrary
            # environment hooks from subject-controlled execution. Tokens, if any,
            # must be supplied by the verifier operator.
            env = {
                "PATH": os.defpath,
                "HOME": isolated_home,
                "XDG_CONFIG_HOME": str(Path(isolated_home) / ".config"),
                "GH_PROMPT_DISABLED": "1",
                "GH_NO_UPDATE_NOTIFIER": "1",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
            }
            for token_name in ("GH_TOKEN", "GITHUB_TOKEN"):
                if token_name in os.environ:
                    env[token_name] = os.environ[token_name]
            completed = runner(
                command, check=False, shell=False, capture_output=True, text=True,
                encoding="utf-8", timeout=timeout_seconds, env=env,
            )
        if completed.returncode != 0:
            raise ProvenanceVerificationError(
                f"GitHub CLI cryptographic/identity attestation verification failed (exit {completed.returncode})"
            )
        if not isinstance(completed.stdout, str) or len(completed.stdout.encode("utf-8")) > 8 * 1024 * 1024:
            raise ProvenanceVerificationError("GitHub CLI output is absent or exceeds the 8 MiB limit")
        try:
            return json.loads(
                completed.stdout, object_pairs_hook=_unique_object, parse_constant=_reject_constant
            )
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ProvenanceVerificationError("GitHub CLI returned invalid verification JSON") from exc
    except subprocess.TimeoutExpired as exc:
        raise ProvenanceVerificationError("GitHub CLI attestation verification timed out") from exc
    except OSError as exc:
        raise ProvenanceVerificationError("GitHub CLI attestation verifier could not be started") from exc
    finally:
        # Hash checks reduce replacement races but cannot secure an attacker-writable
        # parent directory. Provision this executable in an operator-controlled path.
        if _hash_regular_file(cli, "GitHub CLI") != expected_cli_digest:
            raise ProvenanceVerificationError("GitHub CLI binary changed during verification")


def _single_verified_statement(output: Any) -> dict[str, Any]:
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
        raise ProvenanceVerificationError("exactly one verified attestation result is required")
    verification = output[0].get("verificationResult")
    if not isinstance(verification, dict):
        raise ProvenanceVerificationError("verified attestation output lacks verificationResult")
    signature = verification.get("signature")
    certificate = signature.get("certificate") if isinstance(signature, dict) else None
    if not isinstance(certificate, dict) or not certificate:
        raise ProvenanceVerificationError("verified attestation output lacks verified certificate identity")
    timestamps = verification.get("verifiedTimestamps")
    if not isinstance(timestamps, list) or not timestamps or not all(
        isinstance(timestamp, dict) and timestamp for timestamp in timestamps
    ):
        raise ProvenanceVerificationError("a non-empty verified transparency-log/timestamp proof is required")
    statement = verification.get("statement")
    if not isinstance(statement, dict):
        raise ProvenanceVerificationError("verified attestation output lacks a parsed statement")
    return statement


def _validate_statement(statement: Any, policy: dict[str, Any], artifact_digest: str) -> dict[str, Any]:
    if not isinstance(statement, dict):
        raise ProvenanceVerificationError("provenance statement must be an object")
    if statement.get("_type") != IN_TOTO_STATEMENT_V1:
        raise ProvenanceVerificationError("provenance must use in-toto Statement v1")
    if statement.get("predicateType") != SLSA_PROVENANCE_V1:
        raise ProvenanceVerificationError("provenance predicate type must be SLSA v1")

    subjects = statement.get("subject")
    if not isinstance(subjects, list) or len(subjects) != 1 or not isinstance(subjects[0], dict):
        raise ProvenanceVerificationError("project policy requires exactly one provenance subject")
    subject = subjects[0]
    if subject.get("name") != policy["artifact_name"]:
        raise ProvenanceVerificationError("provenance subject name does not match operator policy")
    digest = subject.get("digest")
    if not isinstance(digest, dict) or digest.get("sha256") != artifact_digest:
        raise ProvenanceVerificationError("provenance subject SHA-256 does not match the artifact")
    if set(digest) != {"sha256"}:
        raise ProvenanceVerificationError("provenance subject must contain only the pinned SHA-256 digest")

    predicate = statement.get("predicate")
    if not isinstance(predicate, dict):
        raise ProvenanceVerificationError("SLSA predicate must be an object")
    build_definition = predicate.get("buildDefinition")
    run_details = predicate.get("runDetails")
    if not isinstance(build_definition, dict) or not isinstance(run_details, dict):
        raise ProvenanceVerificationError("SLSA provenance lacks buildDefinition or runDetails")
    if build_definition.get("buildType") != policy["build_type"]:
        raise ProvenanceVerificationError("SLSA buildType does not match operator policy")
    external = build_definition.get("externalParameters")
    if not isinstance(external, dict) or canonical_json(external) != canonical_json(policy["expected_external_parameters"]):
        raise ProvenanceVerificationError("SLSA externalParameters do not exactly match the frozen policy")

    dependency_uri = f'git+https://github.com/{policy["repository"]}@{policy["source_ref"]}'
    dependencies = build_definition.get("resolvedDependencies")
    if not isinstance(dependencies, list):
        raise ProvenanceVerificationError("SLSA resolvedDependencies must be an array")
    matching = [
        dep for dep in dependencies
        if isinstance(dep, dict) and dep.get("uri") == dependency_uri
    ]
    if len(matching) != 1:
        raise ProvenanceVerificationError("exactly one resolved source dependency must match repository and ref")
    source_digest = matching[0].get("digest")
    if not isinstance(source_digest, dict) or source_digest.get("gitCommit") != policy["source_commit"]:
        raise ProvenanceVerificationError("resolved source dependency commit does not match operator policy")

    builder = run_details.get("builder")
    if not isinstance(builder, dict) or builder.get("id") != policy["builder_id"]:
        raise ProvenanceVerificationError("SLSA builder.id does not match operator policy")
    metadata = run_details.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(metadata.get("invocationId"), str) or not metadata["invocationId"]:
        raise ProvenanceVerificationError("SLSA runDetails.metadata.invocationId is required")
    return {
        "artifact_name": subject["name"],
        "artifact_sha256": artifact_digest,
        "repository": policy["repository"],
        "source_ref": policy["source_ref"],
        "source_commit": policy["source_commit"],
        "workflow_path": policy["workflow_path"],
        "signer_repository": policy["signer_repository"],
        "signer_workflow_path": policy["signer_workflow_path"],
        "signer_workflow_commit": policy["signer_workflow_commit"],
        "builder_id": policy["builder_id"],
        "build_type": policy["build_type"],
        "invocation_id": metadata["invocationId"],
    }


def verify_verifier_provenance(
    artifact_path: str | Path,
    policy: Any,
    *,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Verify attestation and deterministic bundle inventory on one byte snapshot.

    GitHub CLI verifies the cryptographic attestation, issuer, signer workflow,
    source ref/digest, predicate type, and runner policy. This function snapshots
    the artifact once, pins its digest, verifies the attestation against that
    snapshot, then validates the exact same snapshot's manifest. Output is
    non-authorizing and does not prove the live process executed the bundle.
    """
    validated_policy = _validate_policy(policy)
    artifact = _safe_regular_file(str(Path(artifact_path).absolute()), "artifact")
    cli_path = Path(validated_policy["gh_cli_path"])
    cli_digest_before = _hash_regular_file(cli_path, "GitHub CLI")
    if cli_digest_before != validated_policy["gh_cli_sha256"]:
        raise ProvenanceVerificationError("GitHub CLI binary digest does not match operator policy")

    with tempfile.TemporaryDirectory(prefix="symthaea-verifier-snapshot-") as snapshot_directory:
        snapshot = Path(snapshot_directory) / "cyber-crucible-verifier-v1.tar.gz"
        artifact_digest = _copy_artifact_to_private_snapshot(artifact, snapshot)
        if artifact_digest != validated_policy["artifact_sha256"]:
            raise ProvenanceVerificationError("verifier artifact digest does not match operator policy")

        output = _run_attestation_verification(snapshot, validated_policy, runner=runner)
        if _hash_regular_file(snapshot, "verifier artifact snapshot") != artifact_digest:
            raise ProvenanceVerificationError("private verifier snapshot changed during attestation verification")
        if _hash_regular_file(cli_path, "GitHub CLI") != cli_digest_before:
            raise ProvenanceVerificationError("GitHub CLI binary changed during verification")

        statement = _single_verified_statement(output)
        verified_claims = _validate_statement(statement, validated_policy, artifact_digest)
        try:
            bundle_result = verify_bundle(
                snapshot,
                expected_source_commit=validated_policy["source_commit"],
            )
        except BundleError as exc:
            raise ProvenanceVerificationError(f"attested artifact is not a valid deterministic verifier bundle: {exc}") from exc
        if bundle_result["bundle_sha256"] != artifact_digest:
            raise ProvenanceVerificationError("verified bundle digest differs from the attested artifact digest")

        return {
            "schema_version": "cyber-crucible-verifier-provenance-result.v1",
            "provenance_verified": True,
            "attestation_signature_verified": True,
            "verified_transparency_timestamp_present": True,
            "bundle_verified": True,
            "bundle_schema_version": bundle_result["schema_version"],
            "bundle_file_count": bundle_result["file_count"],
            "verification_tool": "gh attestation verify",
            "policy_id": validated_policy["policy_id"],
            "policy_digest": json_digest(validated_policy),
            **verified_claims,
            "non_authorizing": True,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="exact verifier artifact to qualify")
    parser.add_argument("--policy", required=True, help="operator-controlled provenance trust policy")
    args = parser.parse_args(argv)
    try:
        result = verify_verifier_provenance(args.artifact, load_json(args.policy))
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (ProvenanceVerificationError, OSError, subprocess.SubprocessError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
