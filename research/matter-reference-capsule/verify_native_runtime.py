#!/usr/bin/env python3
"""Qualify the Nix-pinned native runtime used by locked Python wheels.

This verifier establishes only loader/runtime compatibility. It does not create
an AiiDA profile, install pseudopotentials, invoke Quantum ESPRESSO, or establish
any scientific result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPECTED_PARENT_HEAD = "53e32afe7aba5ae3426d2c66a75a2be466dc1bc6"
EXPECTED_UV_LOCK_SHA256 = "1ecb6c8a0752476c9731d0d44a4113c4ed1fabfee6e339724ee4534926bda646"
EXPECTED_PYZMQ_VERSION = "27.2.0"
AUTHORITY = "LockedPythonNativeRuntimeQualifiedV1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--subject-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_ldd_libstdcxx(module: Path) -> Path:
    output = run("ldd", str(module))
    matches: list[Path] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped.startswith("libstdc++.so.6 "):
            continue
        fields = stripped.split()
        require(len(fields) >= 3 and fields[1] == "=>", f"malformed libstdc++ ldd line: {stripped}")
        matches.append(Path(fields[2]).resolve())
    require(len(matches) == 1, f"expected one libstdc++.so.6 resolution, found {len(matches)}")
    return matches[0]


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    capsule = repo / "research/matter-reference-capsule"
    lock = capsule / "uv.lock"

    require(len(args.subject_head) == 40, "subject head must be an exact 40-character Git SHA")
    require(os.environ.get("SYMTHAEA_MATTER_CAPSULE_SHELL") == "1", "not running in Matter capsule shell")

    cpp_store_text = os.environ.get("SYMTHAEA_CPP_RUNTIME_STORE", "")
    require(cpp_store_text.startswith("/nix/store/"), "declared C++ runtime is not a Nix store output")
    cpp_store = Path(cpp_store_text).resolve()
    require(cpp_store.is_dir(), "declared C++ runtime store output is missing")

    expected_library_path = str(cpp_store / "lib")
    require(os.environ.get("LD_LIBRARY_PATH") == expected_library_path, "loader path is not exactly the declared C++ runtime")

    libstdcpp_link = cpp_store / "lib/libstdc++.so.6"
    require(libstdcpp_link.exists(), "declared C++ runtime does not contain libstdc++.so.6")
    libstdcpp = libstdcpp_link.resolve()
    require(libstdcpp.is_file(), "resolved libstdc++.so.6 is not a file")
    try:
        libstdcpp.relative_to(cpp_store)
    except ValueError as error:
        raise SystemExit("libstdc++.so.6 escaped the declared C++ runtime store") from error

    require(sha256(lock) == EXPECTED_UV_LOCK_SHA256, "committed uv.lock digest changed")

    import ctypes
    import importlib.metadata
    import zmq
    import zmq.backend.cython._zmq as native

    require(importlib.metadata.version("pyzmq") == EXPECTED_PYZMQ_VERSION, "locked pyzmq version changed")
    require(zmq.__version__ == EXPECTED_PYZMQ_VERSION, "imported pyzmq version changed")
    ctypes.CDLL("libstdc++.so.6")
    require(bool(zmq.zmq_version()), "libzmq version is empty")

    native_module = Path(native.__file__).resolve()
    require(native_module.is_file(), "pyzmq native module is missing")
    ldd_libstdcpp = parse_ldd_libstdcxx(native_module)
    require(ldd_libstdcpp == libstdcpp, "pyzmq did not resolve the declared libstdc++.so.6")

    runtime_nar_hash = run("nix-store", "--query", "--hash", str(cpp_store))
    runtime_deriver = run("nix-store", "--query", "--deriver", str(cpp_store))
    require(runtime_deriver.startswith("/nix/store/") and runtime_deriver.endswith(".drv"), "invalid C++ runtime deriver")

    receipt = {
        "schema_version": "symthaea.matter.locked-python-native-runtime-qualification/v1",
        "authority": AUTHORITY,
        "subject_head": args.subject_head,
        "parent_head": EXPECTED_PARENT_HEAD,
        "uv_lock_sha256": EXPECTED_UV_LOCK_SHA256,
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": sys.version.split()[0],
        },
        "cpp_runtime": {
            "store_output": str(cpp_store),
            "nar_hash": runtime_nar_hash,
            "deriver": runtime_deriver,
            "ld_library_path": os.environ["LD_LIBRARY_PATH"],
            "libstdcxx_path": str(libstdcpp),
            "libstdcxx_sha256": sha256(libstdcpp),
        },
        "pyzmq": {
            "version": zmq.__version__,
            "libzmq_version": zmq.zmq_version(),
            "native_module_path": str(native_module),
            "native_module_sha256": sha256(native_module),
            "libstdcxx_resolved_path": str(ldd_libstdcpp),
            "libstdcxx_resolved_inside_declared_runtime": True,
        },
        "scientific_execution": False,
        "aiida_profile_created": False,
        "pseudopotential_installed": False,
        "qe_calculation_executed": False,
        "limitations": [
            "this proves native-wheel loader compatibility only",
            "it does not qualify an AiiDA profile or scientific calculation",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    reparsed = json.loads(args.output.read_text(encoding="utf-8"))
    require(reparsed["authority"] == AUTHORITY, "receipt authority self-check failed")
    require(reparsed["scientific_execution"] is False, "receipt scientific-authority self-check failed")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
