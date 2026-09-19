#!/usr/bin/env python3
"""Run REL-005A independent reproduction inside a sanitized Cargo/Rust environment.

Authority: IndependentReproductionEnvironmentOnly.
This wrapper constrains ambient process state around the frozen v3 reproduction
harness. It does not parse scientific observations, apply predicates, or change
the original REL-005A qualification result.
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

AUTHORITY = "IndependentReproductionEnvironmentOnly"
SCHEMA = "symthaea.rel.reproduction-environment-envelope-receipt.v1"
MANIFEST_SCHEMA = "symthaea.rel.reproduction-environment-envelope-manifest.v1"

DEFAULT_ALLOWLIST = {
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "TERM",
    "RUSTUP_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "NIX_LD",
    "NIX_LD_LIBRARY_PATH",
}

SENSITIVE_ALLOWED = {
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
}

FORBIDDEN_EXACT = {
    "RUSTFLAGS",
    "RUSTDOCFLAGS",
    "RUSTC",
    "RUSTDOC",
    "RUSTUP_TOOLCHAIN",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_HOME",
    "CARGO_TARGET_DIR",
    "CARGO_INCREMENTAL",
    "CC",
    "CXX",
    "AR",
    "RANLIB",
    "LD",
    "PKG_CONFIG",
    "PKG_CONFIG_PATH",
    "PKG_CONFIG_LIBDIR",
    "PKG_CONFIG_SYSROOT_DIR",
}

FORBIDDEN_PREFIXES = (
    "CARGO_BUILD_",
    "CARGO_PROFILE_",
    "CARGO_TARGET_",
    "CARGO_REGISTRIES_",
    "CARGO_SOURCE_",
    "CARGO_NET_",
    "CARGO_HTTP_",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
)


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


def git_at(cwd: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def inside(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_forbidden(name: str) -> bool:
    return name in FORBIDDEN_EXACT or any(name.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)


def validate_contract(root: pathlib.Path, recipe_path: pathlib.Path, recipe: dict[str, Any]) -> dict[str, Any]:
    require(recipe.get("schema") == "symthaea.rel.reproduction-recipe.v3", "recipe schema mismatch")
    require(recipe.get("authority") == "ReproductionRecipeOnly", "recipe authority mismatch")
    contract = recipe.get("environment_envelope")
    require(isinstance(contract, dict), "environment envelope contract missing")
    require(contract.get("authority") == AUTHORITY, "environment envelope authority mismatch")
    require(contract.get("fresh_cargo_home_required") is True, "fresh CARGO_HOME not required")
    require(contract.get("cargo_home_outside_harness_worktree") is True, "CARGO_HOME harness isolation not required")
    require(contract.get("ambient_environment_allowlist") == sorted(DEFAULT_ALLOWLIST), "environment allowlist mismatch")
    require(contract.get("forbidden_exact") == sorted(FORBIDDEN_EXACT), "forbidden exact-variable set mismatch")
    require(contract.get("forbidden_prefixes") == list(FORBIDDEN_PREFIXES), "forbidden prefix set mismatch")
    harness_path = root / contract["harness_path"]
    wrapper_path = root / contract["wrapper_path"]
    require(harness_path.is_file(), "frozen reproduction harness missing")
    require(wrapper_path.is_file(), "environment wrapper missing")
    require(
        git_at(root, "rev-parse", f"HEAD:{contract['harness_path']}") == contract["harness_blob"],
        "frozen reproduction harness blob mismatch",
    )
    require(
        git_at(root, "rev-parse", f"HEAD:{contract['wrapper_path']}") == contract["wrapper_blob"],
        "environment wrapper blob mismatch",
    )
    require(recipe_path.resolve() == (root / ".github/rel/rel-005a-reproduction-recipe.json").resolve(), "unexpected recipe path")
    return contract


def build_sanitized_env(cargo_home: pathlib.Path) -> tuple[dict[str, str], int, list[str], dict[str, str]]:
    inherited: dict[str, str] = {}
    for key in sorted(DEFAULT_ALLOWLIST):
        if key in os.environ:
            inherited[key] = os.environ[key]
    env = dict(inherited)
    env["CARGO_HOME"] = str(cargo_home)
    env["RUSTC_WRAPPER"] = ""
    env["RUSTC_WORKSPACE_WRAPPER"] = ""
    env["SCCACHE_DISABLE"] = "1"
    dropped_non_allowlisted = [key for key in os.environ if key not in inherited and key not in {"CARGO_HOME"}]
    dropped_build_control = sorted(key for key in os.environ if is_forbidden(key))
    require(not any(is_forbidden(key) for key in env if key not in {"CARGO_HOME", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"}), "forbidden build-control variable survived sanitization")
    inherited_hashes = {
        key: sha_bytes(value.encode())
        for key, value in inherited.items()
        if key not in SENSITIVE_ALLOWED
    }
    return env, len(dropped_non_allowlisted), dropped_build_control, inherited_hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    root = pathlib.Path(git_at(pathlib.Path.cwd(), "rev-parse", "--show-toplevel")).resolve()
    recipe_path = args.recipe.expanduser().resolve()
    recipe = load(recipe_path)
    contract = validate_contract(root, recipe_path, recipe)

    output = args.output_dir.expanduser().resolve()
    require(not inside(output, root), "output directory must be outside harness worktree")
    if output.exists():
        require(not any(output.iterdir()), "output directory must start empty")

    ambient_forbidden = sorted(key for key in os.environ if is_forbidden(key))
    harness_path = root / contract["harness_path"]

    with tempfile.TemporaryDirectory(prefix="rel005a-cargo-home-") as cargo_home_tmp:
        cargo_home = pathlib.Path(cargo_home_tmp).resolve()
        require(not inside(cargo_home, root), "fresh CARGO_HOME must be outside harness worktree")
        sanitized, dropped_count, dropped_build_control, inherited_hashes = build_sanitized_env(cargo_home)
        result = subprocess.run(
            [
                sys.executable,
                str(harness_path),
                "--recipe",
                str(recipe_path),
                "--output-dir",
                str(output),
            ],
            cwd=root,
            env=sanitized,
        )
        cargo_home_path = str(cargo_home)

    require(not pathlib.Path(cargo_home_path).exists(), "temporary CARGO_HOME leaked after reproduction")
    output.mkdir(parents=True, exist_ok=True)

    inner_receipt_path = output / "independent-reproduction-receipt.json"
    inner_receipt_present = inner_receipt_path.is_file()
    inner_receipt_sha256 = sha(inner_receipt_path) if inner_receipt_present else None
    inner_reproduction_completed = False
    if inner_receipt_present:
        inner_receipt = load(inner_receipt_path)
        require(inner_receipt.get("authority") == "IndependentReproductionOnly", "inner reproduction authority mismatch")
        require(inner_receipt.get("original_rel_005a_qualification_changed") is False, "inner reproduction changed original qualification")
        inner_claims = inner_receipt.get("claims")
        require(isinstance(inner_claims, dict), "inner reproduction claims missing")
        require(inner_claims.get("original_rel_005a_qualification_changed") is False, "inner claims changed original qualification")
        require(inner_claims.get("scientific_pass") is False and inner_claims.get("scientific_fail") is False, "inner reproduction exceeded authority")
        inner_reproduction_completed = inner_claims.get("independent_reproduction_completed") is True

    envelope = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "relation": "REL-005A",
        "harness_subject_head": git_at(root, "rev-parse", "HEAD"),
        "harness_blob": contract["harness_blob"],
        "wrapper_blob": contract["wrapper_blob"],
        "recipe_sha256": sha(recipe_path),
        "harness_exit_code": result.returncode,
        "fresh_cargo_home_used": True,
        "fresh_cargo_home_removed": True,
        "cargo_home_outside_harness_worktree": True,
        "ambient_forbidden_variables_present_before_sanitization": ambient_forbidden,
        "ambient_forbidden_variable_count": len(ambient_forbidden),
        "dropped_non_allowlisted_environment_key_count": dropped_count,
        "dropped_build_control_keys": dropped_build_control,
        "dropped_build_control_key_count": len(dropped_build_control),
        "inherited_environment_keys": sorted(inherited_hashes),
        "inherited_nonsecret_environment_value_sha256": inherited_hashes,
        "sensitive_inherited_values_recorded": False,
        "path_sha256": inherited_hashes.get("PATH"),
        "rustup_home_inherited": "RUSTUP_HOME" in inherited_hashes,
        "inner_reproduction_receipt_present": inner_receipt_present,
        "inner_reproduction_receipt_sha256": inner_receipt_sha256,
        "inner_reproduction_completed": inner_reproduction_completed,
        "scientific_payload_parsed": False,
        "scientific_result_interpreted": False,
        "original_rel_005a_qualification_changed": False,
        "claims": {
            "environment_envelope_applied": True,
            "independent_reproduction_completed": False,
            "original_rel_005a_qualification_changed": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    envelope_path = output / "reproduction-environment-envelope-receipt.json"
    write(envelope_path, envelope)

    manifest_targets = [
        "reproduction-environment.json",
        "reproduction-execution-receipt.json",
        "reproduction-observation-seal.json",
        "reproduction-comparison-only.json",
        "independent-reproduction-receipt.json",
        "reproduction-environment-envelope-receipt.json",
    ]
    entries = []
    for name in manifest_targets:
        path = output / name
        if path.is_file():
            entries.append({"basename": name, "byte_length": path.stat().st_size, "sha256": sha(path)})
    outer_manifest = {
        "schema": MANIFEST_SCHEMA,
        "authority": AUTHORITY,
        "relation": "REL-005A",
        "files": sorted(entries, key=lambda item: item["basename"].encode()),
        "harness_exit_code": result.returncode,
        "original_rel_005a_qualification_changed": False,
        "claims": {
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    write(output / "reproduction-environment-envelope-manifest.json", outer_manifest)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
