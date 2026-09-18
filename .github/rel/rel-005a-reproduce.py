#!/usr/bin/env python3
"""REL-005A independent reproduction harness v3.

Authority: IndependentReproductionOnly.
Runs the threshold-blind execution from a temporary detached worktree at the
literal frozen ExecutionOnly subject, constructs fresh evidence/seal bytes, and
applies the frozen generic ComparisonOnly evaluator from the harness checkout.
It never reads the original observation bytes and has no authority to state or
change the original qualification result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import shlex
import subprocess
import sys
import tempfile
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=no_duplicates)
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def write(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: pathlib.Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def git_at(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def run_capture(
    command: list[str],
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)


def command_from(recipe: dict[str, Any], name: str) -> list[str]:
    return shlex.split(recipe["commands"][name])


def inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def clean_status(root: pathlib.Path) -> str:
    return git_at(root, "status", "--porcelain=v1", "--untracked-files=all")


def save_process(output: pathlib.Path, stem: str, result: subprocess.CompletedProcess[str]) -> None:
    (output / f"{stem}.stdout.log").write_text(result.stdout)
    (output / f"{stem}.stderr.log").write_text(result.stderr)


def verify_harness_checkout(root: pathlib.Path, recipe: dict[str, Any]) -> str:
    head = git_at(root, "rev-parse", "HEAD")
    base = recipe["harness_base_head"]
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, head], cwd=root
    )
    require(ancestor.returncode == 0, "harness base is not an ancestor of current HEAD")
    changed = sorted(
        line for line in git_at(root, "diff", "--name-only", f"{base}..{head}").splitlines() if line
    )
    require(changed == sorted(recipe["harness_allowed_diff"]), f"unexpected harness diff: {changed}")
    require(clean_status(root) == "", "reproduction requires a clean harness worktree")
    for path, expected_blob in recipe["expected_git_blobs"].items():
        actual = git_at(root, "rev-parse", f"HEAD:{path}")
        require(actual == expected_blob, f"Git blob mismatch for {path}: {actual}")
    return head


def verify_execution_worktree(exec_root: pathlib.Path, recipe: dict[str, Any]) -> None:
    require(
        git_at(exec_root, "rev-parse", "HEAD") == recipe["execution_subject_head"],
        "detached execution worktree head mismatch",
    )
    for path in (
        "crates/core/symthaea-fep/tests/rel_graft_frame_execution_v3.rs",
        "crates/core/symthaea-fep/tests/rel_graft_frame.rs",
        ".github/rel/rel-005a-full-predicate-contract.json",
    ):
        actual = git_at(exec_root, "rev-parse", f"HEAD:{path}")
        require(actual == recipe["expected_git_blobs"][path], f"execution worktree blob mismatch for {path}")
    require(clean_status(exec_root) == "", "detached execution worktree is not clean")


def postflight(root: pathlib.Path, recipe: dict[str, Any], harness_head: str, worktrees_before: str) -> None:
    require(git_at(root, "rev-parse", "HEAD") == harness_head, "harness HEAD changed")
    require(clean_status(root) == "", "harness worktree changed during reproduction")
    require(git_at(root, "worktree", "list", "--porcelain") == worktrees_before, "temporary execution worktree leaked")
    base = recipe["harness_base_head"]
    changed = sorted(
        line for line in git_at(root, "diff", "--name-only", f"{base}..{harness_head}").splitlines() if line
    )
    require(changed == sorted(recipe["harness_allowed_diff"]), "harness scope changed during reproduction")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    recipe = load(args.recipe)
    require(recipe["schema"] == "symthaea.rel.reproduction-recipe.v3", "recipe schema mismatch")
    require(recipe["authority"] == "ReproductionRecipeOnly", "recipe authority mismatch")
    require(recipe["reproduction_semantics"]["may_assert_original_qualification_status"] is False, "recipe exceeds reproduction authority")
    require(recipe["reproduction_semantics"]["fresh_seal_execution_subject_is_literal_detached_git_head"] is True, "literal execution-subject rule missing")

    root = pathlib.Path(git_at(pathlib.Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    harness_head = verify_harness_checkout(root, recipe)
    worktrees_before = git_at(root, "worktree", "list", "--porcelain")

    output = args.output_dir.expanduser().resolve()
    require(not inside(output, root), "output directory must be outside the harness worktree")
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "output directory must start empty")

    try:
        install = run_capture(command_from(recipe, "install_toolchain"), root)
        save_process(output, "toolchain-install", install)
        require(install.returncode == 0, "failed to install frozen Rust toolchain")

        versions: dict[str, str] = {}
        for key, filename in [
            ("rustc_version", "rustc-version.txt"),
            ("cargo_version", "cargo-version.txt"),
            ("rustfmt_version", "rustfmt-version.txt"),
            ("clippy_version", "clippy-version.txt"),
        ]:
            result = run_capture(command_from(recipe, key), root)
            require(result.returncode == 0, f"{key} unavailable")
            (output / filename).write_text(result.stdout)
            versions[key] = result.stdout.strip()
        require("1.96.0" in versions["rustc_version"], f"wrong Rust toolchain: {versions['rustc_version']}")

        environment = {
            "schema": "symthaea.rel.reproduction-environment.v2",
            "authority": "IndependentReproductionInfrastructureOnly",
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "python_version": platform.python_version(),
            "rustc": versions["rustc_version"],
            "cargo": versions["cargo_version"],
            "rustfmt": versions["rustfmt_version"],
            "clippy": versions["clippy_version"],
            "harness_subject_head": harness_head,
            "harness_base_head": recipe["harness_base_head"],
            "execution_subject_head": recipe["execution_subject_head"],
        }
        write(output / "reproduction-environment.json", environment)

        observation = output / "reproduction-observation.json"

        with tempfile.TemporaryDirectory(prefix="rel005a-exec-parent-") as exec_parent, tempfile.TemporaryDirectory(prefix="rel005a-cargo-target-") as cargo_tmp:
            exec_root = pathlib.Path(exec_parent).resolve() / "subject"
            cargo_target = pathlib.Path(cargo_tmp).resolve()
            require(not inside(pathlib.Path(exec_parent).resolve(), root), "execution worktree parent must be outside harness worktree")
            require(not inside(cargo_target, root), "Cargo target directory must be outside harness worktree")

            add = run_capture(
                ["git", "worktree", "add", "--detach", str(exec_root), recipe["execution_subject_head"]],
                root,
            )
            save_process(output, "worktree-add", add)
            require(add.returncode == 0, "failed to create detached exact-subject execution worktree")
            try:
                verify_execution_worktree(exec_root, recipe)
                require(not inside(output, exec_root), "output directory must be outside execution worktree")
                require(not inside(cargo_target, exec_root), "Cargo target directory must be outside execution worktree")
                (output / "execution-worktree-head.txt").write_text(git_at(exec_root, "rev-parse", "HEAD") + "\n")

                fmt = run_capture(command_from(recipe, "format_check"), exec_root)
                save_process(output, "rustfmt", fmt)
                require(fmt.returncode == 0, "ExecutionOnly reproduction source is not rustfmt-clean")

                env = dict(os.environ)
                env["REL005A_EXECUTION_OBSERVATION_PATH"] = str(observation)
                env["RUST_BACKTRACE"] = "1"
                env["RUSTC_WRAPPER"] = ""
                env["SCCACHE_DISABLE"] = "1"
                env["CARGO_TARGET_DIR"] = str(cargo_target)

                execution = run_capture(command_from(recipe, "execute"), exec_root, env)
                save_process(output, "execution", execution)
                (output / "execution-exit-code.txt").write_text(f"{execution.returncode}\n")

                if execution.returncode != 0 or not observation.is_file() or observation.stat().st_size == 0:
                    receipt = {
                        "schema": "symthaea.rel.independent-reproduction-receipt.v3",
                        "authority": "IndependentReproductionOnly",
                        "harness_subject_head": harness_head,
                        "execution_subject_head": recipe["execution_subject_head"],
                        "execution_worktree_head": git_at(exec_root, "rev-parse", "HEAD"),
                        "reproduction_process_result": "EXECUTION_RED",
                        "execution_exit_code": execution.returncode,
                        "observation_present": observation.is_file() and observation.stat().st_size > 0,
                        "original_rel_005a_qualification_changed": False,
                        "claims": {
                            "independent_reproduction_completed": False,
                            "original_rel_005a_qualification_changed": False,
                            "scientific_pass": False,
                            "scientific_fail": False,
                        },
                    }
                    write(output / "independent-reproduction-receipt.json", receipt)
                    print(json.dumps(receipt, indent=2, sort_keys=True))
                    return 1

                observed = load(observation)
                require(observed.get("schema") == recipe["observation"]["schema"], "observation schema mismatch")
                require(observed.get("authority") == recipe["observation"]["authority"], "observation authority mismatch")
                require(observed.get("predicate_contract_head") == recipe["predicate_contract_head"], "observation predicate head mismatch")
                require(observed.get("frozen_scientific_subject") == recipe["frozen_scientific_subject"], "observation frozen subject mismatch")
                require(observed.get("frozen_test_blob") == recipe["expected_git_blobs"]["crates/core/symthaea-fep/tests/rel_graft_frame.rs"], "observation frozen test blob mismatch")
                require(observed.get("claims") == {
                    "observation_sealed": False,
                    "comparison_only_adjudicated": False,
                    "rel_005a_qualified": False,
                    "scientific_pass": False,
                    "scientific_fail": False,
                }, "reproduction observation exceeds ExecutionOnly authority")

                clippy = run_capture(command_from(recipe, "clippy"), exec_root, env)
                save_process(output, "clippy", clippy)
                require(clippy.returncode == 0, "warnings-denied Clippy failed on reproduction source")
                require(clean_status(exec_root) == "", "exact execution worktree changed during reproduction")
            finally:
                remove = run_capture(["git", "worktree", "remove", "--force", str(exec_root)], root)
                save_process(output, "worktree-remove", remove)
                require(remove.returncode == 0, "failed to remove temporary execution worktree")
                require(not exec_root.exists(), "temporary execution worktree directory still exists")

        observation_sha = sha(observation)
        execution_receipt = {
            "schema": "symthaea.rel.reproduction-execution-receipt.v3",
            "authority": "IndependentReproductionExecutionOnly",
            "harness_subject_head": harness_head,
            "execution_subject_head": recipe["execution_subject_head"],
            "execution_worktree_head": recipe["execution_subject_head"],
            "predicate_contract_head": recipe["predicate_contract_head"],
            "frozen_scientific_subject": recipe["frozen_scientific_subject"],
            "measurement_exit_code": "0",
            "observation_present": True,
            "observation_sha256": observation_sha,
            "clippy_status": "success",
            "ambient_toolchain_override_used": False,
            "cargo_target_inside_any_git_worktree": False,
            "claims": {
                "original_observation_reused": False,
                "original_observation_seal_reused": False,
                "original_rel_005a_qualification_changed": False,
            },
        }
        execution_receipt_path = output / "reproduction-execution-receipt.json"
        write(execution_receipt_path, execution_receipt)

        manifest_names = [
            "toolchain-install.stdout.log",
            "toolchain-install.stderr.log",
            "rustc-version.txt",
            "cargo-version.txt",
            "rustfmt-version.txt",
            "clippy-version.txt",
            "reproduction-environment.json",
            "worktree-add.stdout.log",
            "worktree-add.stderr.log",
            "execution-worktree-head.txt",
            "rustfmt.stdout.log",
            "rustfmt.stderr.log",
            "execution.stdout.log",
            "execution.stderr.log",
            "execution-exit-code.txt",
            "reproduction-observation.json",
            "clippy.stdout.log",
            "clippy.stderr.log",
            "worktree-remove.stdout.log",
            "worktree-remove.stderr.log",
            "reproduction-execution-receipt.json",
        ]
        manifest = {
            "schema": "symthaea.rel.reproduction-artifact-manifest.v3",
            "authority": "IndependentReproductionOnly",
            "files": [
                {
                    "basename": name,
                    "byte_length": (output / name).stat().st_size,
                    "sha256": sha(output / name),
                }
                for name in sorted(manifest_names, key=lambda s: s.encode())
            ],
        }
        manifest_path = output / "reproduction-artifact-manifest.json"
        write(manifest_path, manifest)

        chain_payload = {
            "artifact_manifest_sha256": sha(manifest_path),
            "execution_receipt_sha256": sha(execution_receipt_path),
            "execution_subject_head": recipe["execution_subject_head"],
            "observation_sha256": observation_sha,
        }
        chain_sha = sha_bytes(canonical(chain_payload))
        seal = {
            "schema": "symthaea.rel.observation-seal.v3",
            "authority": "ObservationSeal",
            "observation_status": "sealed",
            "adjudication": "not_run",
            "scientific_result": "not_run",
            "execution_subject_head": recipe["execution_subject_head"],
            "predicate_contract_head": recipe["predicate_contract_head"],
            "frozen_scientific_subject": recipe["frozen_scientific_subject"],
            "frozen_test_blob": recipe["expected_git_blobs"]["crates/core/symthaea-fep/tests/rel_graft_frame.rs"],
            "execution_receipt_sha256": sha(execution_receipt_path),
            "observation_sha256": observation_sha,
            "artifact_manifest_sha256": sha(manifest_path),
            "chain_commitment_sha256": chain_sha,
            "claims": {
                "observation_sealed": True,
                "comparison_only_adjudicated": False,
                "rel_005a_qualified": False,
                "scientific_pass": False,
                "scientific_fail": False,
            },
        }
        seal_path = output / "reproduction-observation-seal.json"
        write(seal_path, seal)

        comparison_path = output / "reproduction-comparison-only.json"
        compare = run_capture(
            [
                sys.executable,
                recipe["adjudication"]["evaluator"],
                "--predicate-contract", recipe["adjudication"]["predicate_contract"],
                "--comparison-contract", recipe["adjudication"]["comparison_contract"],
                "--observation", str(observation),
                "--seal", str(seal_path),
                "--output", str(comparison_path),
            ],
            root,
        )
        save_process(output, "comparison", compare)
        require(compare.returncode == 0 and comparison_path.is_file(), "frozen ComparisonOnly evaluator failed")

        comparison = load(comparison_path)
        result = comparison.get("comparison_result")
        require(result in set(recipe["adjudication"]["valid_results"]), "invalid reproduction comparison result")
        require(comparison.get("predicate_count") == recipe["adjudication"]["predicate_count"], "reproduction predicate count mismatch")
        require(comparison.get("passed_count", 0) + comparison.get("failed_count", 0) == 41, "reproduction counts inconsistent")
        require(comparison.get("claims") == {
            "comparison_only_adjudicated": True,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        }, "reproduction comparison exceeded ComparisonOnly authority")

        original_sha = recipe["observation"]["original_qualified_lineage_observation_sha256"]
        receipt = {
            "schema": "symthaea.rel.independent-reproduction-receipt.v3",
            "authority": "IndependentReproductionOnly",
            "harness_subject_head": harness_head,
            "harness_base_head": recipe["harness_base_head"],
            "execution_subject_head": recipe["execution_subject_head"],
            "execution_worktree_head": recipe["execution_subject_head"],
            "reproduction_process_result": "REPRODUCTION_ADJUDICATED",
            "reproduction_predicate_result": result,
            "observation_sha256": observation_sha,
            "original_observation_sha256": original_sha,
            "observation_byte_identical_to_original": observation_sha == original_sha,
            "observation_byte_identity_required": False,
            "predicate_count": comparison["predicate_count"],
            "passed_count": comparison["passed_count"],
            "failed_count": comparison["failed_count"],
            "failed_predicate_ids": comparison["failed_predicate_ids"],
            "fresh_execution_receipt_sha256": sha(execution_receipt_path),
            "fresh_manifest_sha256": sha(manifest_path),
            "fresh_seal_sha256": sha(seal_path),
            "fresh_seal_chain_commitment_sha256": chain_sha,
            "original_observation_reused": False,
            "original_observation_seal_reused": False,
            "original_rel_005a_qualification_changed": False,
            "claims": {
                "independent_reproduction_completed": True,
                "original_rel_005a_qualification_changed": False,
                "scientific_pass": False,
                "scientific_fail": False,
            },
        }
        write(output / "independent-reproduction-receipt.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    finally:
        postflight(root, recipe, harness_head, worktrees_before)


if __name__ == "__main__":
    raise SystemExit(main())
