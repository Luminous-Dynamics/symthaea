#!/usr/bin/env python3
"""Qualify the Nix-pinned native runtime used by locked Python wheels.

This verifier establishes only loader/runtime compatibility. It does not create
an AiiDA profile, install pseudopotentials, invoke Quantum ESPRESSO, or establish
any scientific result.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Any

EXPECTED_PARENT_HEAD = "53e32afe7aba5ae3426d2c66a75a2be466dc1bc6"
EXPECTED_NIXPKGS_REV = "9ae611a455b90cf061d8f332b977e387bda8e1ca"
EXPECTED_NIX_VERSION = "nix (Nix) 2.35.2"
EXPECTED_UV_LOCK_SHA256 = "1ecb6c8a0752476c9731d0d44a4113c4ed1fabfee6e339724ee4534926bda646"
EXPECTED_UV_CONSUMER_SEMVER = "0.11.19"
EXPECTED_PYZMQ_VERSION = "27.2.0"
EXPECTED_PILLOW_VERSION = "12.3.0"
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


def inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def nix_store_output(path: Path) -> Path:
    resolved = path.resolve()
    require(str(resolved).startswith("/nix/store/"), f"path is not Nix-store backed: {resolved}")
    relative = resolved.relative_to("/nix/store")
    require(len(relative.parts) >= 1, f"invalid Nix store path: {resolved}")
    output = Path("/nix/store") / relative.parts[0]
    require(output.is_dir(), f"Nix store output is missing: {output}")
    return output


def nix_output_evidence(output: Path) -> dict[str, str]:
    deriver = run("nix-store", "--query", "--deriver", str(output))
    require(deriver.startswith("/nix/store/") and deriver.endswith(".drv"), f"invalid Nix deriver: {deriver}")
    return {
        "store_output": str(output),
        "nar_hash": run("nix-store", "--query", "--hash", str(output)),
        "deriver": deriver,
    }


def declared_store_env(name: str) -> Path:
    value = os.environ.get(name, "")
    require(value.startswith("/nix/store/"), f"{name} is not a declared Nix store output")
    store = Path(value).resolve()
    require(store.parent == Path("/nix/store") and store.is_dir(), f"invalid declared Nix store output: {store}")
    return store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--subject-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def uv_semantic_version(version_output: str) -> str:
    fields = version_output.split()
    require(len(fields) >= 2 and fields[0] == "uv", f"malformed uv --version output: {version_output!r}")
    semantic = fields[1].split("+", 1)[0]
    require(semantic == EXPECTED_UV_CONSUMER_SEMVER, f"unexpected uv semantic version: {version_output}")
    return semantic


def parse_ldd_libstdcxx(ldd: Path, module: Path) -> Path:
    output = run(str(ldd), str(module))
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


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def normalized_ldd_lines(ldd: Path, path: Path) -> list[str]:
    completed = subprocess.run(
        [str(ldd), str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = completed.stdout.strip()
    require(completed.returncode == 0, f"ldd failed for {path}: {output}")
    require("not found" not in output, f"unresolved ELF dependency for {path}: {output}")
    lines: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Loader addresses are run-specific and carry no identity value.
        if " (0x" in line:
            line = line.split(" (0x", 1)[0]
        lines.append(line)
    return sorted(lines)


def resolved_dependency_path(line: str) -> Path | None:
    fields = line.split()
    if "=>" in fields:
        index = fields.index("=>")
        require(index + 1 < len(fields), f"malformed ldd dependency line: {line}")
        target = fields[index + 1]
        require(target != "not", f"unresolved dependency survived normalization: {line}")
        return Path(target).resolve() if target.startswith("/") else None
    if line.startswith("/"):
        return Path(fields[0]).resolve()
    # linux-vdso and other kernel pseudo-objects do not have filesystem paths.
    return None


def audit_native_elf_closure(
    ldd: Path,
    site_packages: Path,
    declared_runtime_stores: list[Path],
    cpp_store: Path,
    libstdcpp: Path,
    libgcc_store: Path,
    libgcc: Path,
    zlib_store: Path,
    libz: Path,
) -> dict[str, Any]:
    require(site_packages.is_dir(), f"site-packages directory is missing: {site_packages}")
    seen: set[Path] = set()
    manifest: list[dict[str, Any]] = []
    libstdcxx_edges = 0
    libgcc_edges = 0
    zlib_edges = 0
    resolved_edges = 0
    venv_edges = 0
    declared_store_edges = 0

    for candidate in sorted(site_packages.rglob("*.so*")):
        if not candidate.is_file():
            continue
        resolved = candidate.resolve()
        if resolved in seen or not is_elf(resolved):
            continue
        seen.add(resolved)
        lines = normalized_ldd_lines(ldd, resolved)
        resolved_dependencies: list[str] = []

        for line in lines:
            dependency = resolved_dependency_path(line)
            if dependency is None:
                continue
            resolved_edges += 1
            resolved_dependencies.append(str(dependency))

            if inside(dependency, site_packages):
                venv_edges += 1
            else:
                owning_stores = [store for store in declared_runtime_stores if inside(dependency, store)]
                require(
                    len(owning_stores) >= 1,
                    f"{candidate} resolved undeclared host/native dependency: {dependency}",
                )
                declared_store_edges += 1

            if dependency.name.startswith("libstdc++.so.6"):
                require(dependency == libstdcpp, f"{candidate} escaped declared libstdc++ runtime: {dependency}")
                require(inside(dependency, cpp_store), f"{candidate} resolved libstdc++ outside declared C++ runtime")
                libstdcxx_edges += 1

            if dependency.name.startswith("libgcc_s.so.1"):
                require(dependency == libgcc, f"{candidate} escaped declared libgcc runtime: {dependency}")
                require(inside(dependency, libgcc_store), f"{candidate} resolved libgcc outside declared libgcc runtime")
                libgcc_edges += 1

            if dependency.name.startswith("libz.so.1"):
                require(dependency == libz, f"{candidate} escaped declared zlib runtime: {dependency}")
                require(inside(dependency, zlib_store), f"{candidate} resolved zlib outside declared zlib runtime")
                zlib_edges += 1

        manifest.append(
            {
                "path": str(candidate.relative_to(site_packages)),
                "sha256": sha256(resolved),
                "ldd": lines,
                "resolved_dependencies": resolved_dependencies,
            }
        )

    require(manifest, "no ELF shared objects found in frozen site-packages")
    require(libstdcxx_edges >= 1, "native environment did not exercise any libstdc++ dependency")
    require(libgcc_edges >= 1, "native environment did not exercise any libgcc dependency")
    require(zlib_edges >= 1, "native environment did not exercise any zlib dependency")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "audited_elf_object_count": len(manifest),
        "resolved_dependency_edge_count": resolved_edges,
        "venv_vendored_dependency_edge_count": venv_edges,
        "declared_nix_store_dependency_edge_count": declared_store_edges,
        "unresolved_dependency_count": 0,
        "undeclared_host_dependency_count": 0,
        "libstdcxx_edge_count": libstdcxx_edges,
        "libgcc_edge_count": libgcc_edges,
        "zlib_edge_count": zlib_edges,
        "all_resolved_dependencies_inside_declared_roots": True,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    capsule = repo / "research/matter-reference-capsule"
    lock = capsule / "uv.lock"
    flake_lock_path = repo / "flake.lock"

    require(len(args.subject_head) == 40, "subject head must be an exact 40-character Git SHA")
    require(os.environ.get("SYMTHAEA_MATTER_CAPSULE_SHELL") == "1", "not running in Matter capsule shell")

    flake_lock = json.loads(flake_lock_path.read_text(encoding="utf-8"))
    root = flake_lock["nodes"][flake_lock["root"]]
    nixpkgs_node = root["inputs"]["nixpkgs"]
    nixpkgs_locked = flake_lock["nodes"][nixpkgs_node]["locked"]
    require(nixpkgs_locked["rev"] == EXPECTED_NIXPKGS_REV, "root nixpkgs revision changed")
    nix_version = run("nix", "--version")
    require(nix_version == EXPECTED_NIX_VERSION, f"unexpected Nix version: {nix_version}")

    cpp_store = declared_store_env("SYMTHAEA_CPP_RUNTIME_STORE")
    libgcc_store = declared_store_env("SYMTHAEA_LIBGCC_RUNTIME_STORE")
    zlib_store = declared_store_env("SYMTHAEA_ZLIB_RUNTIME_STORE")
    glibc_store = declared_store_env("SYMTHAEA_GLIBC_RUNTIME_STORE")

    runtime_store_texts = os.environ.get("SYMTHAEA_NATIVE_RUNTIME_STORES", "").split(":")
    require(all(runtime_store_texts), "declared native runtime store list is missing")
    declared_runtime_stores = [Path(value).resolve() for value in runtime_store_texts]
    expected_runtime_stores = [cpp_store, libgcc_store, zlib_store, glibc_store]
    require(
        declared_runtime_stores == expected_runtime_stores,
        f"native runtime store ordering/identity drift: {declared_runtime_stores}",
    )
    require(len(declared_runtime_stores) == len(set(declared_runtime_stores)) == 4, "native runtime stores are not four unique outputs")

    expected_library_path = f"{cpp_store / 'lib'}:{libgcc_store / 'lib'}:{zlib_store / 'lib'}"
    require(os.environ.get("LD_LIBRARY_PATH") == expected_library_path, "loader path is not exactly the declared support path")

    libstdcpp_link = cpp_store / "lib/libstdc++.so.6"
    libgcc_link = libgcc_store / "lib/libgcc_s.so.1"
    libz_link = zlib_store / "lib/libz.so.1"
    require(libstdcpp_link.exists(), "declared C++ runtime does not contain libstdc++.so.6")
    require(libgcc_link.exists(), "declared libgcc runtime does not contain libgcc_s.so.1")
    require(libz_link.exists(), "declared zlib runtime does not contain libz.so.1")
    libstdcpp = libstdcpp_link.resolve()
    libgcc = libgcc_link.resolve()
    libz = libz_link.resolve()
    require(libstdcpp.is_file() and inside(libstdcpp, cpp_store), "libstdc++.so.6 escaped declared C++ runtime")
    require(libgcc.is_file() and inside(libgcc, libgcc_store), "libgcc_s.so.1 escaped declared libgcc runtime")
    require(libz.is_file() and inside(libz, zlib_store), "libz.so.1 escaped declared zlib runtime")

    ldd_path_text = shutil.which("ldd")
    require(ldd_path_text is not None, "ldd is unavailable in the pinned shell")
    ldd_path = Path(ldd_path_text).resolve()
    ldd_store = nix_store_output(ldd_path)

    lock_digest = sha256(lock)
    require(lock_digest == EXPECTED_UV_LOCK_SHA256, "committed uv.lock digest changed")

    uv_path_text = shutil.which("uv")
    require(uv_path_text is not None, "uv is unavailable in the pinned shell")
    uv_path = Path(uv_path_text).resolve()
    uv_version_output = run(str(uv_path), "--version")
    uv_semver = uv_semantic_version(uv_version_output)
    uv_store = nix_store_output(uv_path)

    import PIL._imaging as pillow_native
    import zmq
    import zmq.backend.cython._zmq as zmq_native

    require(importlib.metadata.version("pyzmq") == EXPECTED_PYZMQ_VERSION, "locked pyzmq version changed")
    require(zmq.__version__ == EXPECTED_PYZMQ_VERSION, "imported pyzmq version changed")
    require(importlib.metadata.version("pillow") == EXPECTED_PILLOW_VERSION, "locked Pillow version changed")
    ctypes.CDLL("libstdc++.so.6")
    ctypes.CDLL("libgcc_s.so.1")
    ctypes.CDLL("libz.so.1")
    require(bool(zmq.zmq_version()), "libzmq version is empty")

    zmq_module = Path(zmq_native.__file__).resolve()
    pillow_module = Path(pillow_native.__file__).resolve()
    require(zmq_module.is_file(), "pyzmq native module is missing")
    require(pillow_module.is_file(), "Pillow native module is missing")
    ldd_libstdcpp = parse_ldd_libstdcxx(ldd_path, zmq_module)
    require(ldd_libstdcpp == libstdcpp, "pyzmq did not resolve the declared libstdc++.so.6")

    site_packages = Path(sysconfig.get_paths()["platlib"]).resolve()
    require(inside(zmq_module, site_packages), "pyzmq native module escaped frozen site-packages")
    require(inside(pillow_module, site_packages), "Pillow native module escaped frozen site-packages")
    native_elf_closure = audit_native_elf_closure(
        ldd_path,
        site_packages,
        declared_runtime_stores,
        cpp_store,
        libstdcpp,
        libgcc_store,
        libgcc,
        zlib_store,
        libz,
    )

    support_evidence = [nix_output_evidence(store) for store in declared_runtime_stores]

    receipt = {
        "schema_version": "symthaea.matter.locked-python-native-runtime-qualification/v1",
        "authority": AUTHORITY,
        "subject_head": args.subject_head,
        "parent_head": EXPECTED_PARENT_HEAD,
        "source": {
            "flake_lock_sha256": sha256(flake_lock_path),
            "nixpkgs_node": nixpkgs_node,
            "nixpkgs_rev": nixpkgs_locked["rev"],
            "nixpkgs_nar_hash": nixpkgs_locked["narHash"],
            "uv_lock_sha256": lock_digest,
        },
        "runtime": {
            "nix_version": nix_version,
            "uv": {
                "executable": str(uv_path),
                "executable_sha256": sha256(uv_path),
                "semantic_version": uv_semver,
                "version_output": uv_version_output,
                **nix_output_evidence(uv_store),
            },
            "python": {
                "executable": str(Path(sys.executable).resolve()),
                "version": sys.version.split()[0],
                "platform": sysconfig.get_platform(),
            },
            "ldd": {
                "executable": str(ldd_path),
                "executable_sha256": sha256(ldd_path),
                **nix_output_evidence(ldd_store),
            },
        },
        "native_support": {
            "ld_library_path": os.environ["LD_LIBRARY_PATH"],
            "declared_store_outputs": [str(store) for store in declared_runtime_stores],
            "store_evidence": support_evidence,
            "cpp_runtime_store": str(cpp_store),
            "libstdcxx_path": str(libstdcpp),
            "libstdcxx_sha256": sha256(libstdcpp),
            "libgcc_runtime_store": str(libgcc_store),
            "libgcc_path": str(libgcc),
            "libgcc_sha256": sha256(libgcc),
            "zlib_runtime_store": str(zlib_store),
            "libz_path": str(libz),
            "libz_sha256": sha256(libz),
            "glibc_runtime_store": str(glibc_store),
        },
        "pyzmq": {
            "version": zmq.__version__,
            "libzmq_version": zmq.zmq_version(),
            "native_module_venv_relative_path": str(zmq_module.relative_to(capsule / ".venv")),
            "native_module_sha256": sha256(zmq_module),
            "libstdcxx_resolved_path": str(ldd_libstdcpp),
            "libstdcxx_resolved_inside_declared_runtime": True,
        },
        "pillow": {
            "version": importlib.metadata.version("pillow"),
            "native_module_venv_relative_path": str(pillow_module.relative_to(capsule / ".venv")),
            "native_module_sha256": sha256(pillow_module),
            "libgcc_load_succeeded": True,
            "libz_load_succeeded": True,
        },
        "native_elf_closure": native_elf_closure,
        "scientific_execution": False,
        "aiida_profile_created": False,
        "pseudopotential_installed": False,
        "qe_calculation_executed": False,
        "limitations": [
            "this proves frozen native-wheel loader compatibility only",
            "it does not qualify an AiiDA profile or scientific calculation",
            "ELF closure auditing proves resolution under this qualified Linux environment, not cross-platform portability",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    reparsed = json.loads(args.output.read_text(encoding="utf-8"))
    require(reparsed["authority"] == AUTHORITY, "receipt authority self-check failed")
    require(reparsed["scientific_execution"] is False, "receipt scientific-authority self-check failed")
    require(reparsed["source"]["nixpkgs_rev"] == EXPECTED_NIXPKGS_REV, "receipt nixpkgs self-check failed")
    require(reparsed["native_elf_closure"]["unresolved_dependency_count"] == 0, "receipt ELF closure self-check failed")
    require(reparsed["native_elf_closure"]["undeclared_host_dependency_count"] == 0, "receipt host closure self-check failed")
    require(
        reparsed["native_elf_closure"]["all_resolved_dependencies_inside_declared_roots"] is True,
        "receipt declared-root closure self-check failed",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
