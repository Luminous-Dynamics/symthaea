#!/usr/bin/env python3
"""Build content-addressed LL-009D study/frame/epoch contracts.

The input file contains three semantic objects: study, frame, and epoch. Each is
canonicalized independently and receives an ID derived from its SHA-256. The
result prevents downstream evidence from reusing an opaque contract ID after
its actual definition changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable

INPUT_SCHEMA = "ll009d.contract-input.v1"
MANIFEST_SCHEMA = "ll009d.contract-manifest.v1"


class ContractError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def require_string(obj: dict[str, Any], field: str, label: str) -> None:
    if not nonempty(obj.get(field)):
        raise ContractError(f"{label} missing {field}")


def require_string_list(obj: dict[str, Any], field: str, label: str, allow_empty: bool = False) -> None:
    value = obj.get(field)
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ContractError(f"{label}.{field} must be {'a' if allow_empty else 'a non-empty'} list")
    if any(not nonempty(item) for item in value):
        raise ContractError(f"{label}.{field} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise ContractError(f"{label}.{field} contains duplicates")


def validate_study(study: dict[str, Any]) -> None:
    for field in ("name", "purpose", "hypothesis_status", "source_node_ref", "destination_node_ref", "cargo_class", "demand_scenario_ref"):
        require_string(study, field, "study")
    require_string_list(study, "candidate_architectures", "study")
    require_string_list(study, "required_evidence_roles", "study")
    approximations = study.get("approximations")
    if not isinstance(approximations, list):
        raise ContractError("study.approximations must be a list")
    non_claims = study.get("non_claims")
    if not isinstance(non_claims, list) or any(not nonempty(item) for item in non_claims):
        raise ContractError("study.non_claims must be a list of strings")
    envelope = study.get("cargo_envelope")
    if not isinstance(envelope, dict):
        raise ContractError("study.cargo_envelope must be an object")
    mass = envelope.get("mass_kg")
    if not isinstance(mass, dict):
        raise ContractError("study.cargo_envelope.mass_kg must be a range object")
    lo, hi = mass.get("min"), mass.get("max")
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (lo, hi)) or lo < 0 or hi < lo:
        raise ContractError("study cargo mass range invalid")


def validate_frame(frame: dict[str, Any]) -> None:
    for field in ("native_terrain_frame", "study_body_fixed_frame", "inertial_frame", "terrain_projection", "bridge_policy", "gravity_constants_profile_ref", "length_units", "velocity_units"):
        require_string(frame, field, "frame")
    if frame["native_terrain_frame"] != frame["study_body_fixed_frame"] and frame["bridge_policy"] == "none":
        raise ContractError("frame mismatch requires an explicit bridge policy")
    aliases = frame.get("forbidden_silent_aliases", [])
    if not isinstance(aliases, list) or any(not nonempty(item) for item in aliases):
        raise ContractError("frame.forbidden_silent_aliases must be a string list")


def validate_epoch(epoch: dict[str, Any]) -> None:
    for field in ("timescale", "mode"):
        require_string(epoch, field, "epoch")
    mode = epoch["mode"]
    if mode not in {"single_epoch", "bounded_window", "explicit_epochs"}:
        raise ContractError("epoch.mode invalid")
    if mode == "single_epoch":
        require_string(epoch, "epoch", "epoch")
    elif mode == "bounded_window":
        require_string(epoch, "start", "epoch")
        require_string(epoch, "end", "epoch")
    else:
        require_string_list(epoch, "epochs", "epoch")
    gap = epoch.get("max_target_interpolation_gap_s")
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap <= 0:
        raise ContractError("epoch.max_target_interpolation_gap_s must be positive")
    require_string(epoch, "artifact_semantics", "epoch")


def load_input(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != INPUT_SCHEMA:
        raise ContractError(f"input schema_version must be {INPUT_SCHEMA}")
    for name in ("study", "frame", "epoch"):
        if not isinstance(value.get(name), dict):
            raise ContractError(f"input missing {name} object")
    validate_study(value["study"])
    validate_frame(value["frame"])
    validate_epoch(value["epoch"])
    return value


def contract_record(kind: str, value: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    semantic_payload = canonical_bytes(value)
    digest = sha256_bytes(semantic_payload)
    record = {
        "schema_version": f"ll009d.{kind}-contract.v1",
        "contract_id": f"{kind}-{digest}",
        "semantic_sha256": digest,
        "contract": value,
    }
    return record, canonical_bytes(record)


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(x for x in root.rglob("*") if x.is_file())
    ]


def build(input_path: Path, output_dir: Path) -> dict[str, Any]:
    source = load_input(input_path)
    records: dict[str, tuple[dict[str, Any], bytes]] = {
        kind: contract_record(kind, source[kind]) for kind in ("study", "frame", "epoch")
    }
    with tempfile.TemporaryDirectory(prefix="ll009d-") as tmp:
        stage = Path(tmp) / "contracts"
        stage.mkdir(parents=True)
        for kind, (_record, payload) in records.items():
            (stage / f"{kind}_contract.json").write_bytes(payload)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "input_filename": input_path.name,
            "input_sha256": sha256_file(input_path),
            "study_id": records["study"][0]["contract_id"],
            "frame_contract_id": records["frame"][0]["contract_id"],
            "epoch_contract_id": records["epoch"][0]["contract_id"],
            "study_semantic_sha256": records["study"][0]["semantic_sha256"],
            "frame_semantic_sha256": records["frame"][0]["semantic_sha256"],
            "epoch_semantic_sha256": records["epoch"][0]["semantic_sha256"],
            "non_claim": "Contract identity is provenance/evidence semantics, not site selection or operational qualification.",
        }
        (stage / "contract_manifest.json").write_bytes(canonical_bytes(manifest))

        if output_dir.exists():
            if tree_manifest(output_dir) == tree_manifest(stage):
                return manifest
            raise ContractError(f"refusing to overwrite differing contract lineage: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output_dir)
        if tree_manifest(output_dir) != tree_manifest(stage):
            shutil.rmtree(output_dir, ignore_errors=True)
            raise ContractError("contract output copy verification failed")
    return manifest


def self_test() -> None:
    sample = {
        "schema_version": INPUT_SCHEMA,
        "study": {
            "name": "synthetic-corridor",
            "purpose": "test",
            "hypothesis_status": "study_hypothesis",
            "source_node_ref": "source:test",
            "destination_node_ref": "destination:test",
            "cargo_class": "bulk",
            "cargo_envelope": {"mass_kg": {"min": 10.0, "max": 100.0}},
            "demand_scenario_ref": "D-test",
            "candidate_architectures": ["rover", "ballistic"],
            "required_evidence_roles": ["terrain_pack_receipt"],
            "approximations": ["spherical moon"],
            "non_claims": ["synthetic only"],
        },
        "frame": {
            "native_terrain_frame": "FRAME_A",
            "study_body_fixed_frame": "FRAME_B",
            "inertial_frame": "J2000",
            "terrain_projection": "synthetic",
            "bridge_policy": "explicit_transform_required",
            "gravity_constants_profile_ref": "constants:test",
            "length_units": "km",
            "velocity_units": "km/s",
            "forbidden_silent_aliases": ["FRAME_ALIAS"],
        },
        "epoch": {
            "timescale": "TDB",
            "mode": "explicit_epochs",
            "epochs": ["2460000.5"],
            "max_target_interpolation_gap_s": 10.0,
            "artifact_semantics": "single shared epoch set",
        },
    }
    with tempfile.TemporaryDirectory(prefix="ll009d-selftest-") as tmp:
        root = Path(tmp)
        inp = root / "input.json"
        inp.write_bytes(canonical_bytes(sample))
        out = root / "out"
        first = build(inp, out)
        second = build(inp, out)
        assert first == second
        changed = json.loads(json.dumps(sample))
        changed["study"]["candidate_architectures"].append("elevator")
        changed_path = root / "changed.json"
        changed_path.write_bytes(canonical_bytes(changed))
        changed_out = root / "changed-out"
        changed_manifest = build(changed_path, changed_out)
        assert changed_manifest["study_id"] != first["study_id"]
        assert changed_manifest["frame_contract_id"] == first["frame_contract_id"]
        assert changed_manifest["epoch_contract_id"] == first["epoch_contract_id"]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009D contract self-test: PASS")
        return 0
    if args.input is None or args.output_dir is None:
        raise ContractError("--input and --output-dir are required unless --self-test is used")
    manifest = build(args.input, args.output_dir)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
