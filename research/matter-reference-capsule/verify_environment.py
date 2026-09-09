#!/usr/bin/env python3
"""Independently qualify the prepared Matter Reference Capsule environment.

This verifier is post-bootstrap and read-only. It may inspect AiiDA provenance
state and registered executables, but it cannot submit processes, install
pseudopotentials, invoke Quantum ESPRESSO, or establish a scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

EXPECTED_PARENT_HEAD = "ad9cd5c2408fd9e976f777155b0a7631ce096283"
EXPECTED_NATIVE_RUNTIME_AUTHORITY = "LockedPythonNativeRuntimeQualifiedV1"
EXPECTED_LOCK_SHA256 = "1ecb6c8a0752476c9731d0d44a4113c4ed1fabfee6e339724ee4534926bda646"
EXPECTED_LOCK_SOURCE_HEAD = "e78974e7d1f77b3059a9da8e578920c875987e9e"
EXPECTED_NIXPKGS_REV = "9ae611a455b90cf061d8f332b977e387bda8e1ca"
EXPECTED_NIX_VERSION = "nix (Nix) 2.35.2"
EXPECTED_INSTALL_NIX_ACTION = "13d8dd58da0234aa297dedd986986ccb8e7f3e24"
EXPECTED_LOCK_GENERATOR = {"python": "3.13.15", "uv": "uv 0.10.0"}
EXPECTED_UV_SEMVER = "0.11.19"
EXPECTED_PYTHON_VERSION = "3.13.13"
EXPECTED_DISTRIBUTIONS = {
    "aiida-core": "2.9.2",
    "aiida-pseudo": "1.9.0",
    "aiida-quantumespresso": "5.0.0",
    "pyzmq": "27.2.0",
}
EXPECTED_PROFILE = "symthaea-matter-si-001"
EXPECTED_COMPUTER = "matter-local"
EXPECTED_CODES = {
    "qe-pw": ("quantumespresso.pw", "pw.x"),
    "qe-ph": ("quantumespresso.ph", "ph.x"),
    "qe-q2r": ("quantumespresso.q2r", "q2r.x"),
    "qe-matdyn": ("quantumespresso.matdyn", "matdyn.x"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def nix_store_output(path: Path) -> Path:
    resolved = path.resolve()
    store = Path("/nix/store")
    require(inside(resolved, store), f"path is not Nix-store backed: {resolved}")
    relative = resolved.relative_to(store)
    require(relative.parts, f"invalid Nix store path: {resolved}")
    output = store / relative.parts[0]
    require(output.exists(), f"Nix store output is missing: {output}")
    return output


def nix_evidence(output: Path) -> dict[str, str]:
    nar_hash = run("nix-store", "--query", "--hash", str(output))
    deriver = run("nix-store", "--query", "--deriver", str(output))
    require(deriver.startswith("/nix/store/") and deriver.endswith(".drv"), f"invalid Nix deriver: {deriver}")
    return {"store_output": str(output), "nar_hash": nar_hash, "deriver": deriver}


def uv_semver(version_output: str) -> str:
    fields = version_output.split()
    require(len(fields) >= 2 and fields[0] == "uv", f"malformed uv --version output: {version_output!r}")
    version = fields[1].split("+", 1)[0]
    require(version == EXPECTED_UV_SEMVER, f"unexpected uv semantic version: {version_output}")
    return version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", default=EXPECTED_PROFILE)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--subject-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(len(args.subject_head) == 40, "subject head must be an exact 40-character Git SHA")
    require(args.profile == EXPECTED_PROFILE, "unexpected AiiDA profile identity")

    repo = args.repo_root.resolve()
    capsule = repo / "research/matter-reference-capsule"
    state_root = args.state_root.resolve()
    manifest_path = state_root / "environment-manifest.json"
    require(os.environ.get("AIIDA_PATH") == str(state_root / "aiida"), "AIIDA_PATH is not the isolated capsule path")
    require(manifest_path.is_file(), "bootstrap environment manifest is missing")

    lock_path = capsule / "uv.lock"
    lock_provenance = load_json(capsule / "uv.lock.provenance.json")
    lock_source = load_json(capsule / "uv.lock.source-manifest.json")
    flake_lock_path = repo / "flake.lock"
    flake_lock = load_json(flake_lock_path)
    manifest = load_json(manifest_path)

    lock_digest = sha256(lock_path)
    require(lock_digest == EXPECTED_LOCK_SHA256, "committed uv.lock digest changed")
    require(lock_provenance["uv_lock_sha256"] == lock_digest, "lock provenance digest mismatch")
    require(lock_source["candidate_uv_lock_sha256"] == lock_digest, "lock source digest mismatch")
    require(lock_source["head_sha"] == EXPECTED_LOCK_SOURCE_HEAD, "lock source head changed")
    require(lock_provenance["source_head_sha"] == EXPECTED_LOCK_SOURCE_HEAD, "lock provenance source head changed")
    tooling = lock_source.get("tooling", {})
    lock_generator = {"python": tooling.get("python"), "uv": tooling.get("uv")}
    require(lock_generator == EXPECTED_LOCK_GENERATOR, f"lock generator identity drift: {lock_generator}")

    root = flake_lock["nodes"][flake_lock["root"]]
    nixpkgs_node = root["inputs"]["nixpkgs"]
    nixpkgs_locked = flake_lock["nodes"][nixpkgs_node]["locked"]
    require(nixpkgs_locked["rev"] == EXPECTED_NIXPKGS_REV, "root nixpkgs revision changed")
    nix_version = run("nix", "--version")
    require(nix_version == EXPECTED_NIX_VERSION, f"unexpected Nix version: {nix_version}")
    require(os.environ.get("INSTALL_NIX_ACTION_SHA") == EXPECTED_INSTALL_NIX_ACTION, "install-nix action identity changed")

    uv_text = shutil.which("uv")
    require(uv_text is not None, "uv is unavailable after bootstrap")
    uv_path = Path(uv_text).resolve()
    uv_version_output = run(str(uv_path), "--version")
    uv_evidence = nix_evidence(nix_store_output(uv_path))
    uv_evidence.update({
        "executable": str(uv_path),
        "executable_sha256": sha256(uv_path),
        "semantic_version": uv_semver(uv_version_output),
        "version_output": uv_version_output,
    })
    run(str(uv_path), "lock", "--check", cwd=capsule)
    require(sha256(lock_path) == lock_digest, "uv lock --check changed the committed lock")

    python_executable = Path(os.environ["SYMTHAEA_PYTHON"]).resolve()
    require(python_executable.is_file(), "pinned Python executable is missing")
    require(run(str(python_executable), "--version") == f"Python {EXPECTED_PYTHON_VERSION}", "unexpected pinned Python version")
    require(Path(sys.executable).resolve() == python_executable, "venv Python does not resolve to the pinned Nix interpreter")
    require(sys.version.split()[0] == EXPECTED_PYTHON_VERSION, "verifier is running under an unexpected Python version")
    python_evidence = nix_evidence(nix_store_output(python_executable))
    python_evidence.update({
        "executable": str(python_executable),
        "executable_sha256": sha256(python_executable),
        "version": EXPECTED_PYTHON_VERSION,
        "venv_launcher": sys.executable,
    })

    distributions = {name: importlib.metadata.version(name) for name in EXPECTED_DISTRIBUTIONS}
    require(distributions == EXPECTED_DISTRIBUTIONS, f"frozen Python distribution identity drift: {distributions}")

    cpp_store = Path(os.environ.get("SYMTHAEA_CPP_RUNTIME_STORE", "")).resolve()
    require(cpp_store.parent == Path("/nix/store") and cpp_store.is_dir(), "inherited C++ runtime is not an exact Nix store output")
    require(os.environ.get("LD_LIBRARY_PATH") == str(cpp_store / "lib"), "loader path drifted from the qualified native-runtime parent")
    libstdcpp_link = cpp_store / "lib/libstdc++.so.6"
    require(libstdcpp_link.exists(), "qualified parent C++ runtime lacks libstdc++.so.6")
    libstdcpp = libstdcpp_link.resolve()
    require(libstdcpp.is_file() and inside(libstdcpp, cpp_store), "libstdc++.so.6 escaped inherited C++ runtime")
    cpp_evidence = nix_evidence(cpp_store)
    cpp_evidence.update({
        "libstdcxx": str(libstdcpp),
        "libstdcxx_sha256": sha256(libstdcpp),
        "ld_library_path": os.environ["LD_LIBRARY_PATH"],
    })

    qe_store = Path(os.environ["SYMTHAEA_QE_STORE"]).resolve()
    require(qe_store.parent == Path("/nix/store") and qe_store.is_dir(), "QE is not an exact Nix store output")
    qe_evidence = nix_evidence(qe_store)
    qe_evidence.update({"store_name": qe_store.name})

    require(manifest["authority"] == "ExecutionEnvironmentPreparedOnly", "bootstrap authority drift")
    require(manifest["profile"] == args.profile, "bootstrap profile mismatch")
    require(Path(manifest["state_root"]).resolve() == state_root, "bootstrap state-root mismatch")
    require(manifest["quantum_espresso_store"] == str(qe_store), "bootstrap QE store mismatch")
    require(manifest["python"]["version"] == EXPECTED_PYTHON_VERSION, "bootstrap Python version mismatch")
    require(Path(manifest["python"]["executable"]).resolve() == python_executable, "bootstrap Python executable mismatch")
    require(manifest["caching"] == "disabled", "bootstrap caching claim changed")
    require(manifest["broker"] == "none", "bootstrap broker claim changed")

    executable_evidence: dict[str, dict[str, str]] = {}
    for name in ("pw.x", "ph.x", "q2r.x", "matdyn.x"):
        evidence = manifest["executables"][name]
        path = Path(evidence["path"]).resolve()
        require(path.is_file() and inside(path, qe_store), f"{name} escaped the exact QE store output")
        digest = sha256(path)
        require(digest == evidence["sha256"], f"{name} byte digest mismatch")
        require(os.access(path, os.X_OK), f"{name} is not executable")
        executable_evidence[name] = {"path": str(path), "sha256": digest}

    shell_driver_text = os.environ.get("SYMTHAEA_SHELL_DRIVER_PATH", "")
    shell_driver_version = os.environ.get("SYMTHAEA_SHELL_DRIVER_VERSION", "")
    require(shell_driver_text and shell_driver_version, "shell-driver provenance was not supplied")
    shell_driver_path = Path(shell_driver_text).resolve()
    require(shell_driver_path.is_file(), "shell-driver path is not a file")
    shell_driver = {
        "path": str(shell_driver_path),
        "sha256": sha256(shell_driver_path),
        "version": shell_driver_version,
        "nix_store_backed": str(shell_driver_path).startswith("/nix/store/"),
        "authority_role": "orchestration_only_not_scientific_solver",
    }

    # The inherited native-runtime parent has already qualified the loader
    # boundary separately. Importing AiiDA here exercises that boundary while
    # this verifier establishes the distinct prepared-environment theorem.
    from aiida import load_profile
    from aiida.orm import CalcJobNode, InstalledCode, ProcessNode, QueryBuilder, load_code, load_computer

    profile = load_profile(args.profile)
    require(profile.name == args.profile, "loaded the wrong AiiDA profile")
    require(profile.storage_backend == "core.sqlite_dos", f"unexpected AiiDA storage backend: {profile.storage_backend}")
    require(profile.process_control_backend is None, "reference capsule unexpectedly has a process-control broker")

    process_count = QueryBuilder().append(ProcessNode, project="id").count()
    calcjob_count = QueryBuilder().append(CalcJobNode, project="id").count()
    require(process_count == 0, f"prepared environment already contains {process_count} ProcessNode(s)")
    require(calcjob_count == 0, f"prepared environment already contains {calcjob_count} CalcJobNode(s)")

    computer = load_computer(EXPECTED_COMPUTER)
    require(computer.hostname == "localhost", "reference computer hostname drift")
    require(computer.transport_type == "core.local", "reference computer transport drift")
    require(computer.scheduler_type == "core.direct", "reference computer scheduler drift")

    codes: dict[str, dict[str, str]] = {}
    for label, (plugin, executable_name) in EXPECTED_CODES.items():
        code = load_code(f"{label}@{EXPECTED_COMPUTER}")
        require(isinstance(code, InstalledCode), f"{label} is not an InstalledCode")
        require(code.default_calc_job_plugin == plugin, f"{label} plugin mismatch")
        require(code.computer.uuid == computer.uuid, f"{label} is bound to a different computer")
        code_path = Path(str(code.filepath_executable)).resolve()
        expected_path = Path(executable_evidence[executable_name]["path"])
        require(code_path == expected_path, f"{label} executable path mismatch")
        code.validate_filepath_executable()
        codes[label] = {
            "uuid": str(code.uuid),
            "plugin": code.default_calc_job_plugin,
            "computer_uuid": str(code.computer.uuid),
            "filepath_executable": str(code_path),
            "sha256": sha256(code_path),
        }

    verdi = capsule / ".venv/bin/verdi"
    require(verdi.is_file(), "frozen AiiDA environment does not contain verdi")
    cache_value = run(str(verdi), "-p", args.profile, "config", "get", "caching.default_enabled")
    require(cache_value.lower().endswith("false"), f"AiiDA caching is not disabled: {cache_value}")

    receipt = {
        "schema_version": "symthaea.matter.reference-capsule-environment-qualification/v1",
        "authority": "ExecutionEnvironmentPreparedQualifiedV1",
        "subject_head": args.subject_head,
        "parent_head": EXPECTED_PARENT_HEAD,
        "prerequisite": {
            "native_runtime_parent_head": EXPECTED_PARENT_HEAD,
            "separate_authority": EXPECTED_NATIVE_RUNTIME_AUTHORITY,
            "authority_rederived_here": False,
            "runtime_identity_rebound_here": True,
        },
        "source": {
            "flake_lock_sha256": sha256(flake_lock_path),
            "uv_lock_sha256": lock_digest,
            "uv_lock_source_head": EXPECTED_LOCK_SOURCE_HEAD,
            "nixpkgs_node": nixpkgs_node,
            "nixpkgs_rev": nixpkgs_locked["rev"],
            "nixpkgs_nar_hash": nixpkgs_locked["narHash"],
            "lock_generator": lock_generator,
        },
        "runtime": {
            "nix_version": nix_version,
            "install_nix_action_sha": EXPECTED_INSTALL_NIX_ACTION,
            "lock_consumer": {
                "uv": uv_evidence,
                "python": python_evidence,
                "python_distributions": distributions,
                "uv_lock_check_read_only": True,
                "uv_lock_sha256_after_check": sha256(lock_path),
            },
            "cpp_runtime": cpp_evidence,
            "shell_driver": shell_driver,
            "quantum_espresso": qe_evidence,
            "executables": executable_evidence,
        },
        "aiida": {
            "profile": profile.name,
            "storage_backend": profile.storage_backend,
            "process_control_backend": profile.process_control_backend,
            "computer": {
                "uuid": str(computer.uuid),
                "label": computer.label,
                "hostname": computer.hostname,
                "transport_type": computer.transport_type,
                "scheduler_type": computer.scheduler_type,
            },
            "codes": codes,
            "caching_default_enabled": False,
            "process_node_count": process_count,
            "calcjob_node_count": calcjob_count,
        },
        "scientific_execution": {
            "process_node_created": False,
            "pseudopotential_installed": False,
            "pseudopotential_selected": False,
            "qe_calculation_executed": False,
            "scientific_result_observed": False,
            "sssp_download_executed": False,
        },
        "next_required_transition": "environment-bound-prospective-si-pseudopotential-freeze",
        "runner": {
            "image_os": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "architecture": os.uname().machine,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    reparsed = load_json(args.output)
    require(reparsed["authority"] == "ExecutionEnvironmentPreparedQualifiedV1", "receipt authority self-check failed")
    require(reparsed["parent_head"] == EXPECTED_PARENT_HEAD, "receipt parent self-check failed")
    require(reparsed["aiida"]["process_node_count"] == 0, "receipt process-count self-check failed")
    require(reparsed["aiida"]["calcjob_node_count"] == 0, "receipt CalcJob-count self-check failed")
    require(reparsed["scientific_execution"]["qe_calculation_executed"] is False, "receipt science-boundary self-check failed")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
