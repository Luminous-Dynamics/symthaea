#!/usr/bin/env python3
"""Run one pinned VLEB witness command and record an evaluator-bindable receipt.

This recorder is intended for a trusted, isolated evaluation workflow. The
caller must independently enforce the declared network/runtime policy and must
not pass secrets in command arguments. The recorded receipt is not a signature:
it becomes trustworthy only when its bytes are in the attested evidence bundle
and the evaluator checks every run, verifier, source, and artifact binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

WITNESS_KINDS = (
    "source_index",
    "static_analysis",
    "compiler",
    "independent_execution",
    "independent_expert_review",
)
RESULTS = ("supports", "contradicts", "inconclusive")
VERDICT_KEYS = {
    "schema_version",
    "task_id",
    "claim_id",
    "source_sha256",
    "verifier_sha256",
    "result",
}
ENV_ALLOWLIST = (
    "PATH", "HOME", "TMPDIR", "TEMP", "LANG", "LC_ALL", "TZ",
    "CARGO_HOME", "RUSTUP_HOME", "SYSTEMROOT", "WINDIR",
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def invocation_digest(argv: list[str], working_directory: str, timeout_ms: int) -> str:
    canonical = json.dumps(
        {
            "argv": argv,
            "working_directory": working_directory,
            "timeout_ms": timeout_ms,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return digest(canonical)


def safe_path(root: Path, value: str, *, must_exist: bool) -> Path:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError("path must be a non-empty canonical POSIX relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != value:
        raise ValueError("absolute, traversal, or ambiguous paths are rejected")
    base = root.resolve(strict=True)
    current = base
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink paths are rejected")
    resolved = current.resolve(strict=must_exist)
    if not resolved.is_relative_to(base):
        raise ValueError("path escapes artifact root")
    return resolved


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"required trusted runner variable {name} is missing")
    return value


def env_context() -> dict[str, Any]:
    # These values must be injected by the protected workflow, not from the
    # candidate patch or the agent-controlled command.
    context = {
        "repository_full_name": required_env("GITHUB_REPOSITORY"),
        "repository_id": int(required_env("VLEB_REPOSITORY_ID")),
        "run_id": required_env("GITHUB_RUN_ID"),
        "run_attempt": int(required_env("GITHUB_RUN_ATTEMPT")),
        "source_repository_digest": required_env("GITHUB_SHA"),
        "source_repository_ref": required_env("GITHUB_REF"),
        "candidate_checkout_revision": required_env("VLEB_CANDIDATE_CHECKOUT_REVISION"),
        "workflow_sha": required_env("VLEB_WORKFLOW_SHA"),
        "workflow_ref": required_env("GITHUB_WORKFLOW_REF"),
    }
    if context["repository_id"] < 1 or not context["run_id"].isdigit() or context["run_attempt"] < 1:
        raise ValueError("trusted repository/run identity is invalid")
    for field in ("source_repository_digest", "candidate_checkout_revision", "workflow_sha"):
        value = context[field]
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"trusted runner variable for {field} must be a lowercase Git SHA")
    return context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--claim-id", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--verifier-kind", choices=WITNESS_KINDS, required=True)
    parser.add_argument("--verifier-id", required=True)
    parser.add_argument("--verifier-sha256", required=True)
    parser.add_argument("--verifier-path", required=True)
    parser.add_argument("--verifier-argv-index", type=int, required=True)
    parser.add_argument("--verdict-path", required=True)
    parser.add_argument("--runtime-image-id", required=True)
    parser.add_argument("--network-mode", choices=("none", "restricted", "enabled"), required=True)
    parser.add_argument("--network-mode-observed", choices=("true", "false"), required=True)
    parser.add_argument("--working-directory", default=".")
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--expected-invocation-sha256", required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--stdout-path", required=True)
    parser.add_argument("--stderr-path", required=True)
    parser.add_argument("--input", action="append", default=[], dest="inputs")
    parser.add_argument("--output-artifact", action="append", default=[], dest="output_artifacts")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.timeout_ms < 1:
            raise ValueError("timeout must be positive")
        if not args.command or not args.command[0]:
            raise ValueError("a verifier command must follow --")
        if len(args.verifier_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in args.verifier_sha256):
            raise ValueError("verifier SHA-256 must be 64 lowercase hexadecimal characters")
        if args.verifier_argv_index < 0 or args.verifier_argv_index >= len(args.command):
            raise ValueError("verifier argv index is outside the exact invocation")
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{1,127}", args.claim_id):
            raise ValueError("claim ID is invalid")
        if len(args.expected_invocation_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in args.expected_invocation_sha256
        ):
            raise ValueError("expected invocation SHA-256 must be 64 lowercase hexadecimal characters")
        observed_invocation_sha256 = invocation_digest(
            args.command, args.working_directory, args.timeout_ms
        )
        if observed_invocation_sha256 != args.expected_invocation_sha256:
            raise ValueError("command argv, working directory or timeout differs from evaluator-pinned invocation")

        root = args.artifact_root.resolve(strict=True)
        source = safe_path(root, args.source_path, must_exist=True)
        if not source.is_file():
            raise ValueError("source path must name a regular file")
        verifier_path = safe_path(root, args.verifier_path, must_exist=True)
        if not verifier_path.is_file():
            raise ValueError("verifier path must name a regular file")
        actual_verifier_sha256 = digest(verifier_path.read_bytes())
        if actual_verifier_sha256 != args.verifier_sha256:
            raise ValueError("verifier file bytes differ from evaluator-pinned verifier digest")
        cwd = safe_path(root, args.working_directory, must_exist=True)
        if not cwd.is_dir():
            raise ValueError("working directory must be a directory within artifact root")
        invocation_verifier_path = safe_path(
            cwd, args.command[args.verifier_argv_index], must_exist=True
        )
        if invocation_verifier_path != verifier_path:
            raise ValueError("verifier argv entry does not resolve to the hashed verifier file from the working directory")

        receipt_path = safe_path(root, args.receipt_path, must_exist=False)
        stdout_path = safe_path(root, args.stdout_path, must_exist=False)
        stderr_path = safe_path(root, args.stderr_path, must_exist=False)
        verdict_path = safe_path(root, args.verdict_path, must_exist=False)
        if len({receipt_path, stdout_path, stderr_path, verdict_path}) != 4:
            raise ValueError("receipt, stdout, stderr and verdict paths must be distinct")
        if any(path == cwd for path in (receipt_path, stdout_path, stderr_path, verdict_path)):
            raise ValueError("receipt/output paths may not name the working directory")
        if any(path.exists() for path in (receipt_path, stdout_path, stderr_path, verdict_path)):
            raise ValueError("receipt, stream and verdict outputs must not exist before the invocation")
        if any(path == source or path == verifier_path for path in (receipt_path, stdout_path, stderr_path, verdict_path)):
            raise ValueError("receipt/output paths may not overwrite source or verifier bytes")

        if not args.runtime_image_id.startswith("sha256:") or len(args.runtime_image_id) != 71 or any(
            char not in "0123456789abcdef" for char in args.runtime_image_id[7:]
        ):
            raise ValueError("runtime image ID must use sha256:<64 lowercase hexadecimal characters>")
        run_context = env_context()

        input_rows: list[dict[str, str]] = []
        for value in [args.source_path, args.verifier_path, *args.inputs]:
            path = safe_path(root, value, must_exist=True)
            if not path.is_file():
                raise ValueError(f"input is not a regular file: {value}")
            row = {"path": value, "sha256": digest(path.read_bytes())}
            if row not in input_rows:
                input_rows.append(row)

        verifier_input_row = next(
            row for row in input_rows if row["path"] == args.verifier_path
        )
        if verifier_input_row["sha256"] != actual_verifier_sha256:
            raise ValueError("verifier bytes changed while preparing the pinned invocation")
        source_sha256_before = next(
            row["sha256"] for row in input_rows if row["path"] == args.source_path
        )

        output_values = [
            ("stdout", args.stdout_path),
            ("stderr", args.stderr_path),
            ("verdict", args.verdict_path),
        ]
        output_values.extend(("artifact", value) for value in args.output_artifacts)
        input_paths = {row["path"] for row in input_rows}
        if any(
            value == args.receipt_path or value in input_paths
            for _, value in output_values
        ):
            raise ValueError("output artifacts may not overwrite the receipt, source, verifier or any declared input")
        if len({value for _, value in output_values}) != len(output_values):
            raise ValueError("output artifact paths must be unique")
        for _, value in output_values:
            path = safe_path(root, value, must_exist=False)
            if path.exists():
                raise ValueError(f"output artifact must not exist before invocation: {value}")

        clean_env = {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ}
        env_sha = digest(json.dumps(clean_env, sort_keys=True, separators=(",", ":")).encode())
        timed_out = False
        signal_name: str | None = None
        exit_code: int | None = None
        try:
            completed = subprocess.run(
                args.command,
                cwd=cwd,
                env=clean_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=args.timeout_ms / 1000,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            exit_code = completed.returncode
            if exit_code < 0:
                try:
                    signal_name = signal.Signals(-exit_code).name
                except ValueError:
                    signal_name = f"SIG{-exit_code}"
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = exc.stdout or b""
            stderr = exc.stderr or b""

        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)

        input_drift_paths: list[str] = []
        for input_row in input_rows:
            try:
                current_input = safe_path(root, input_row["path"], must_exist=True)
                if digest(current_input.read_bytes()) != input_row["sha256"]:
                    input_drift_paths.append(input_row["path"])
            except (OSError, ValueError):
                input_drift_paths.append(input_row["path"])
        input_drift_paths = sorted(set(input_drift_paths))

        output_rows: list[dict[str, str]] = []
        for role, value in output_values:
            path = safe_path(root, value, must_exist=True)
            if not path.is_file():
                raise ValueError(f"output is not a regular file: {value}")
            output_rows.append({"path": value, "sha256": digest(path.read_bytes()), "role": role})

        source_sha256 = source_sha256_before
        observed_result = "inconclusive"
        verdict_errors: list[str] = []
        if input_drift_paths:
            verdict_errors.append(
                "declared input bytes changed during verifier execution: "
                + ", ".join(input_drift_paths)
            )
        try:
            verdict_raw = verdict_path.read_bytes()
            verdict = json.loads(
                verdict_raw.decode("utf-8", errors="strict"),
                object_pairs_hook=_unique_json_object,
            )
            if not isinstance(verdict, dict) or set(verdict) != VERDICT_KEYS:
                verdict_errors.append("verdict object has missing or unexpected fields")
            else:
                if verdict.get("schema_version") != "vleb-witness-verdict-v1":
                    verdict_errors.append("verdict schema_version is unsupported")
                if verdict.get("task_id") != args.task_id:
                    verdict_errors.append("verdict task ID differs from evaluator task")
                if verdict.get("claim_id") != args.claim_id:
                    verdict_errors.append("verdict claim ID differs from evaluator claim")
                if verdict.get("source_sha256") != source_sha256:
                    verdict_errors.append("verdict source digest differs from source bytes")
                if verdict.get("verifier_sha256") != args.verifier_sha256:
                    verdict_errors.append("verdict verifier digest differs from pinned verifier bytes")
                if verdict.get("result") not in RESULTS:
                    verdict_errors.append("verdict result is not a supported classification")
                if (
                    not verdict_errors
                    and not input_drift_paths
                    and not timed_out
                    and signal_name is None
                    and exit_code == 0
                ):
                    observed_result = verdict["result"]
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            verdict_errors.append(f"verdict output is not valid, unique-key JSON: {exc}")

        receipt = {
            "schema_version": "vleb-witness-execution-receipt-v1",
            "task_id": args.task_id,
            "claim_id": args.claim_id,
            "source_sha256": source_sha256,
            "run_binding": run_context,
            "runtime": {
                "immutable_image_id": args.runtime_image_id,
                "network_mode": args.network_mode,
                "network_mode_observed": args.network_mode_observed == "true",
            },
            "producer": {
                "id": "vleb-record-witness-execution",
                "source_sha256": digest(Path(__file__).read_bytes()),
            },
            "verifier": {
                "kind": args.verifier_kind,
                "id": args.verifier_id,
                "sha256": actual_verifier_sha256,
                "path": args.verifier_path,
                "argv_index": args.verifier_argv_index,
            },
            "invocation": {
                "argv": args.command,
                "argv_sha256": observed_invocation_sha256,
                "working_directory": args.working_directory,
                "timeout_ms": args.timeout_ms,
                "environment_sha256": env_sha,
            },
            "inputs": input_rows,
            "input_drift_paths": input_drift_paths,
            "outputs": output_rows,
            "process": {
                "exit_code": exit_code,
                "signal": signal_name,
                "timed_out": timed_out,
            },
            "observed_result": observed_result,
        }
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({
            "receipt_path": args.receipt_path,
            "receipt_sha256": digest(receipt_path.read_bytes()),
            "process_exit_code": exit_code,
            "timed_out": timed_out,
            "observed_result": receipt["observed_result"],
            "verdict_errors": verdict_errors,
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc), "receipt_created": False}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
