#!/usr/bin/env python3
"""CIV-BOOT-002D differential qualifier for productive-capability closure.

The harness compares the exact vendored PIE #1619 Python oracle with the Rust
candidate over one frozen synthetic/adversarial corpus. It grants no physical,
manufacturing, procurement, or resource-allocation authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

EXPECTED_SCHEMA = "civ-boot-capability-closure-v1"
EXPECTED_ORACLE_PR = 1619
EXPECTED_ORACLE_HEAD = "58499378af3fd3777b916daac1aa5bd573f2c680"
EXPECTED_ORACLE_BLOB = "0f7bf960ed4a0575e8b4e2d6d50fe6676e295b80"
EXPECTED_CORPUS_BLOB = "b15dec817d22620ebeddcf0def6e0ef52a5e7268"


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    prefix = f"blob {len(data)}\0".encode()
    return hashlib.sha1(prefix + data).hexdigest()


def require_blob(path: Path, expected: str, label: str) -> None:
    actual = git_blob_sha(path)
    if actual != expected:
        raise RuntimeError(f"{label} blob mismatch: expected {expected}, got {actual}")


def load_reference(path: Path) -> ModuleType:
    require_blob(path, EXPECTED_ORACLE_BLOB, "reference oracle")

    spec = importlib.util.spec_from_file_location("pie_dependency_closure_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture_payload(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "local_primitives": fixture.get("local_primitives", []),
        "imports": fixture.get("imports", []),
        "targets": fixture.get("targets", []),
        "recipes": fixture.get("recipes", []),
    }


def reference_result(module: ModuleType, payload: dict[str, Any]) -> dict[str, Any]:
    recipes = [
        module.Recipe(item["output"], tuple(item["requires"]), item["recipe_id"])
        for item in payload["recipes"]
    ]
    try:
        report = module.evaluate(
            payload["local_primitives"],
            payload["imports"],
            payload["targets"],
            recipes,
        )
    except ValueError:
        return {"disposition": "InvalidInput"}

    # JSON round-trip converts dataclass tuples to the same array representation
    # emitted by the Rust compatibility adapter.
    normalized = json.loads(json.dumps(asdict(report), sort_keys=True))
    return {"disposition": "Valid", "result": normalized}


def cargo_target_directory(repo_root: Path) -> Path:
    completed = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(completed.stdout)
    return Path(metadata["target_directory"])


def resolve_rust_binary(repo_root: Path, supplied: str | None) -> Path:
    if supplied:
        binary = Path(supplied)
        if not binary.is_absolute():
            binary = repo_root / binary
        return binary

    subprocess.run(
        [
            "cargo",
            "build",
            "--locked",
            "--quiet",
            "-p",
            "symthaea-capability-closure",
            "--example",
            "reference_compat",
        ],
        cwd=repo_root,
        check=True,
    )

    target_dir = cargo_target_directory(repo_root)
    suffix = ".exe" if sys.platform == "win32" else ""
    return target_dir / "debug" / "examples" / f"reference_compat{suffix}"


def rust_result(binary: Path, payload: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    completed = subprocess.run(
        [str(binary), "--json", encoded],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Rust compatibility adapter failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Rust compatibility adapter returned invalid JSON: {completed.stdout!r}"
        ) from error


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    default_base = repo_root / "crates" / "domains" / "symthaea-capability-closure"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default=str(default_base / "fixtures" / "differential-v1.json"),
    )
    parser.add_argument(
        "--oracle",
        default=str(default_base / "tests" / "reference" / "pie-dependency-closure-oracle.py"),
    )
    parser.add_argument("--rust-binary")
    args = parser.parse_args()

    corpus_path = Path(args.corpus).resolve()
    oracle_path = Path(args.oracle).resolve()
    require_blob(corpus_path, EXPECTED_CORPUS_BLOB, "fixture corpus")
    corpus = json.loads(corpus_path.read_text())

    if corpus.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError(f"unexpected corpus schema: {corpus.get('schema_version')!r}")
    reference_meta = corpus.get("reference_oracle", {})
    if reference_meta.get("pr") != EXPECTED_ORACLE_PR:
        raise RuntimeError("corpus reference-oracle PR does not match qualifier")
    if reference_meta.get("head_sha") != EXPECTED_ORACLE_HEAD:
        raise RuntimeError("corpus reference-oracle head does not match qualifier")
    if reference_meta.get("git_blob_sha") != EXPECTED_ORACLE_BLOB:
        raise RuntimeError("corpus reference-oracle blob does not match qualifier")

    reference = load_reference(oracle_path)
    rust_binary = resolve_rust_binary(repo_root, args.rust_binary)
    if not rust_binary.is_file():
        raise RuntimeError(f"Rust compatibility adapter not found: {rust_binary}")

    group_results: dict[str, dict[str, Any]] = {}
    fixture_count = 0

    for fixture in corpus.get("fixtures", []):
        fixture_count += 1
        fixture_id = fixture["id"]
        expected = fixture["expected_disposition"]
        payload = fixture_payload(fixture)

        reference_output = reference_result(reference, payload)
        candidate_output = rust_result(rust_binary, payload, repo_root)

        if reference_output.get("disposition") != expected:
            raise AssertionError(
                f"{fixture_id}: reference disposition {reference_output.get('disposition')} != {expected}"
            )
        if candidate_output.get("disposition") != expected:
            raise AssertionError(
                f"{fixture_id}: Rust disposition {candidate_output.get('disposition')} != {expected}"
            )
        if reference_output != candidate_output:
            raise AssertionError(
                f"{fixture_id}: differential mismatch\n"
                f"reference={json.dumps(reference_output, sort_keys=True)}\n"
                f"rust={json.dumps(candidate_output, sort_keys=True)}"
            )

        group = fixture.get("equivalence_group")
        if group and expected == "Valid":
            previous = group_results.get(group)
            if previous is not None and previous != candidate_output:
                raise AssertionError(
                    f"{fixture_id}: canonical output differs within equivalence group {group}"
                )
            group_results[group] = candidate_output

    print(
        "ok: "
        f"{fixture_count} fixtures; "
        f"{len(group_results)} equivalence group(s); "
        f"oracle_blob={EXPECTED_ORACLE_BLOB}; corpus_blob={EXPECTED_CORPUS_BLOB}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
