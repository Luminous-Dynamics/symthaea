#!/usr/bin/env python3
"""Build a deterministic REL-005A qualification capsule for provenance attestation.

Authority: AttestationInputOnly. This module packages already-qualified evidence bytes.
It does not interpret scientific metrics, predicates, thresholds, or qualification results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tarfile
import tempfile
from typing import Iterable

REQUIRED = {
    "predicate-contract-receipt.json",
    "execution-v3-receipt.json",
    "observation-seal-v3.json",
    "comparison-only-qualification-receipt.json",
    "qualification-input-manifest.json",
    "qualification-assembly-receipt.json",
    "qualification-only.json",
}
GENERATED = "qualification-capsule-manifest.json"
OUTPUT = "rel-005a-qualified-capsule.tar"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recursive_files(root: pathlib.Path) -> list[pathlib.Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix().encode())


def validate_input(root: pathlib.Path) -> dict[str, pathlib.Path]:
    files = recursive_files(root)
    by_name: dict[str, pathlib.Path] = {}
    for path in files:
        rel = path.relative_to(root)
        if len(rel.parts) != 1:
            raise ValueError(f"nested capsule input is forbidden: {rel}")
        if path.name in by_name:
            raise ValueError(f"duplicate capsule basename: {path.name}")
        by_name[path.name] = path
    if set(by_name) != REQUIRED:
        raise ValueError(
            f"capsule input census mismatch: got={sorted(by_name)} expected={sorted(REQUIRED)}"
        )
    return by_name


def write_manifest(root: pathlib.Path, files: dict[str, pathlib.Path]) -> pathlib.Path:
    entries = [
        {
            "basename": name,
            "byte_length": files[name].stat().st_size,
            "sha256": sha256(files[name]),
        }
        for name in sorted(files, key=lambda s: s.encode())
    ]
    manifest = {
        "schema": "symthaea.rel.qualification-capsule-manifest.v1",
        "authority": "AttestationInputOnly",
        "relation": "REL-005A",
        "canonicalization": {
            "compression": "none",
            "sort": "bytewise basename ascending",
            "mtime": 0,
            "uid": 0,
            "gid": 0,
            "uname": "",
            "gname": "",
            "file_mode": "0644",
            "pax_headers": False,
        },
        "files": entries,
        "claims": {
            "attestation_created": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    path = root / GENERATED
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def add_file(tf: tarfile.TarFile, path: pathlib.Path, arcname: str) -> None:
    data = path.read_bytes()
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    import io

    tf.addfile(info, io.BytesIO(data))


def build(input_dir: pathlib.Path, output_path: pathlib.Path) -> dict[str, object]:
    files = validate_input(input_dir)
    with tempfile.TemporaryDirectory() as tmp:
        stage = pathlib.Path(tmp)
        staged: dict[str, pathlib.Path] = {}
        for name in sorted(files, key=lambda s: s.encode()):
            target = stage / name
            target.write_bytes(files[name].read_bytes())
            staged[name] = target
        manifest_path = write_manifest(stage, staged)
        names = sorted([*staged.keys(), GENERATED], key=lambda s: s.encode())
        with tarfile.open(output_path, mode="w", format=tarfile.USTAR_FORMAT) as tf:
            for name in names:
                add_file(tf, stage / name, name)

    return {
        "schema": "symthaea.rel.attestation-input-receipt.v1",
        "authority": "AttestationInputOnly",
        "capsule_path": output_path.name,
        "capsule_sha256": sha256(output_path),
        "capsule_byte_length": output_path.stat().st_size,
        "member_count": len(REQUIRED) + 1,
        "members": names,
        "claims": {
            "attestation_created": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def self_test() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        a = root / "a"
        b = root / "b"
        a.mkdir(); b.mkdir()
        # Create the exact same logical bytes in opposite creation orders.
        ordered = sorted(REQUIRED, key=lambda s: s.encode())
        for i, name in enumerate(ordered):
            payload = f"{name}\nsynthetic-{i}\n".encode()
            (a / name).write_bytes(payload)
        for i, name in reversed(list(enumerate(ordered))):
            payload = f"{name}\nsynthetic-{i}\n".encode()
            (b / name).write_bytes(payload)
        out_a = root / "a.tar"
        out_b = root / "b.tar"
        ra = build(a, out_a)
        rb = build(b, out_b)
        if ra["capsule_sha256"] != rb["capsule_sha256"]:
            raise ValueError("deterministic capsule self-test failed")

        nested = root / "nested"
        nested.mkdir()
        for name in ordered:
            (nested / name).write_text("x\n")
        (nested / "extra").mkdir()
        (nested / "extra" / "raw-observation.json").write_text("{}\n")
        try:
            build(nested, root / "nested.tar")
        except ValueError as exc:
            if "nested capsule input" not in str(exc):
                raise
        else:
            raise ValueError("nested extra file was accepted")

        missing = root / "missing"
        missing.mkdir()
        for name in ordered[:-1]:
            (missing / name).write_text("x\n")
        try:
            build(missing, root / "missing.tar")
        except ValueError as exc:
            if "census mismatch" not in str(exc):
                raise
        else:
            raise ValueError("missing required file was accepted")

    return {
        "schema": "symthaea.rel.attestation-input-self-test.v1",
        "authority": "AttestationContractOnly",
        "deterministic_archive": True,
        "creation_order_independent": True,
        "nested_extra_rejected": True,
        "missing_member_rejected": True,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=pathlib.Path)
    p.add_argument("--output", type=pathlib.Path)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2, sort_keys=True))
        return
    if args.input_dir is None or args.output is None:
        raise SystemExit("--input-dir and --output are required")
    print(json.dumps(build(args.input_dir, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
