#!/usr/bin/env python3
"""Guard REL-005A independent reproduction against ambient hierarchical Cargo config.

Authority: IndependentReproductionEnvironmentGuardOnly.
Creates a dedicated temporary root, verifies that no `.cargo/config` or
`.cargo/config.toml` exists in its ancestor chain, forces TMPDIR/TMP/TEMP to that
root, and invokes the frozen v5 locked/offline reproduction wrapper unchanged.
It does not parse scientific evidence or interpret a reproduction result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

AUTHORITY = "IndependentReproductionEnvironmentGuardOnly"
SCHEMA = "symthaea.rel.reproduction-ancestor-cargo-config-guard-receipt.v1"
V5_AUTHORITY = "IndependentReproductionEnvironmentOnly"


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


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_at(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def cargo_configs_in_ancestors(path: pathlib.Path) -> list[str]:
    found: list[str] = []
    current = path.resolve()
    for parent in [current, *current.parents]:
        for rel in (pathlib.Path(".cargo/config"), pathlib.Path(".cargo/config.toml")):
            candidate = parent / rel
            if candidate.is_file():
                found.append(str(candidate))
    return sorted(set(found))


def validate_contract(root: pathlib.Path, recipe: dict[str, Any]) -> dict[str, Any]:
    contract = recipe.get("ancestor_cargo_config_guard")
    require(isinstance(contract, dict), "ancestor Cargo-config guard contract missing")
    require(contract.get("authority") == AUTHORITY, "guard authority mismatch")
    require(contract.get("dedicated_temp_root_required") is True, "dedicated temp root not required")
    require(contract.get("pre_scan_required") is True and contract.get("post_scan_required") is True, "ancestor scan requirements missing")
    require(contract.get("may_parse_scientific_payload") is False, "guard may parse scientific payload")
    require(contract.get("may_interpret_scientific_result") is False, "guard may interpret scientific result")
    require(contract.get("may_change_original_qualification") is False, "guard may change original qualification")
    require(not any(contract.get("claims", {}).values()), "guard contract makes runtime/scientific claims")
    wrapper = root / contract["wrapper_path"]
    v5 = root / contract["v5_wrapper_path"]
    require(wrapper.is_file() and v5.is_file(), "guard/v5 wrapper missing")
    require(git_at(root, "rev-parse", f"HEAD:{contract['wrapper_path']}") == contract["wrapper_blob"], "guard wrapper blob mismatch")
    require(git_at(root, "rev-parse", f"HEAD:{contract['v5_wrapper_path']}") == contract["v5_wrapper_blob"], "v5 wrapper blob mismatch")
    return contract


def self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rel005a-guard-selftest-") as tmp:
        root = pathlib.Path(tmp)
        clean = root / "clean" / "child"
        clean.mkdir(parents=True)
        require(cargo_configs_in_ancestors(clean) == [], "clean ancestor chain rejected")
        poisoned = root / "poisoned"
        (poisoned / ".cargo").mkdir(parents=True)
        (poisoned / ".cargo/config.toml").write_text("[net]\noffline = false\n")
        child = poisoned / "child"
        child.mkdir()
        found = cargo_configs_in_ancestors(child)
        require(any(path.endswith(".cargo/config.toml") for path in found), "ambient Cargo config not detected")
    return {
        "schema": "symthaea.rel.reproduction-ancestor-cargo-config-guard-self-test.v1",
        "authority": AUTHORITY,
        "clean_chain_accepted": True,
        "poisoned_chain_rejected": True,
        "scientific_payload_parsed": False,
        "scientific_result_interpreted": False,
        "claims": {"scientific_pass": False, "scientific_fail": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=pathlib.Path)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return 0
    require(args.recipe is not None and args.output_dir is not None, "--recipe and --output-dir required")

    root = pathlib.Path(git_at(pathlib.Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    recipe_path = args.recipe.expanduser().resolve()
    recipe = load(recipe_path)
    contract = validate_contract(root, recipe)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        require(not any(output.iterdir()), "output directory must start empty")

    with tempfile.TemporaryDirectory(prefix="rel005a-v6-root-") as temp_raw:
        temp_root = pathlib.Path(temp_raw).resolve()
        pre = cargo_configs_in_ancestors(temp_root)
        require(not pre, f"ambient ancestor Cargo config present before reproduction: {pre}")
        env = dict(os.environ)
        for key in ("TMPDIR", "TMP", "TEMP"):
            env[key] = str(temp_root)
        v5 = root / contract["v5_wrapper_path"]
        result = subprocess.run(
            [sys.executable, str(v5), "--recipe", str(recipe_path), "--output-dir", str(output)],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
        )
        post = cargo_configs_in_ancestors(temp_root)
        require(not post, f"ambient ancestor Cargo config appeared during reproduction: {post}")
        temp_root_path = str(temp_root)

    require(not pathlib.Path(temp_root_path).exists(), "dedicated v6 temp root leaked")
    output.mkdir(parents=True, exist_ok=True)
    (output / "ancestor-guard-v5.stdout.log").write_text(result.stdout)
    (output / "ancestor-guard-v5.stderr.log").write_text(result.stderr)

    env_receipt = output / "reproduction-environment-envelope-receipt.json"
    env_manifest = output / "reproduction-environment-envelope-manifest.json"
    if env_receipt.is_file():
        env_doc = load(env_receipt)
        require(env_doc.get("authority") == V5_AUTHORITY, "v5 environment receipt authority mismatch")
        require(env_doc.get("scientific_payload_parsed") is False and env_doc.get("scientific_result_interpreted") is False, "v5 environment receipt exceeded authority")
        claims = env_doc.get("claims", {})
        require(claims.get("scientific_pass") is False and claims.get("scientific_fail") is False, "v5 environment receipt made scientific claim")

    receipt = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "relation": "REL-005A",
        "guard_subject_head": git_at(root, "rev-parse", "HEAD"),
        "guard_wrapper_blob": contract["wrapper_blob"],
        "v5_wrapper_blob": contract["v5_wrapper_blob"],
        "v5_exit_code": result.returncode,
        "dedicated_temp_root_used": True,
        "dedicated_temp_root_removed": True,
        "pre_ancestor_cargo_config_paths": [],
        "post_ancestor_cargo_config_paths": [],
        "ancestor_cargo_config_guard_passed": True,
        "v5_environment_receipt_sha256": sha(env_receipt) if env_receipt.is_file() else None,
        "v5_environment_manifest_sha256": sha(env_manifest) if env_manifest.is_file() else None,
        "scientific_payload_parsed": False,
        "scientific_result_interpreted": False,
        "original_rel_005a_qualification_changed": False,
        "claims": {
            "environment_guard_applied": True,
            "independent_reproduction_completed": False,
            "original_rel_005a_qualification_changed": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    write(output / "reproduction-ancestor-cargo-config-guard-receipt.json", receipt)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
