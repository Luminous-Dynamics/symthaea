#!/usr/bin/env python3
"""Independent stdlib-only verifier for simulation canonical digest v1."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct

DOMAIN_TAG = {
    "civil": 0,
    "mechanical": 1,
    "electrical": 2,
    "aerospace": 3,
    "chemical_process": 4,
    "robotics": 5,
    "nuclear": 6,
    "materials": 7,
    "environmental": 8,
    "systems": 9,
}

SOLVER_TAG = {
    "finite_element": 0,
    "computational_fluid_dynamics": 1,
    "multibody_dynamics": 2,
    "circuit": 3,
    "process": 4,
    "cad_geometry": 5,
    "multi_physics": 6,
    "custom": 7,
}

REQUEST_DOMAIN = b"symthaea.simulation.request.v1\0"
OUTPUT_DOMAIN = b"symthaea.simulation.output.v1\0"


def u64(value: int) -> bytes:
    if value < 0 or value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("length outside canonical u64 range")
    return struct.pack("<Q", value)


def text(value: str) -> bytes:
    raw = value.encode("utf-8")
    return u64(len(raw)) + raw


def f64(value: float) -> bytes:
    if not math.isfinite(value):
        raise ValueError("canonical v1 rejects non-finite floats")
    return struct.pack("<d", value)


def optional(value: object | None, encoder) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + encoder(value)


def interval(value: dict[str, float]) -> bytes:
    return f64(value["lower"]) + f64(value["upper"])


def uncertainty(value: dict[str, object]) -> bytes:
    return (
        f64(value["epistemic"])
        + f64(value["aleatoric"])
        + optional(value["interval"], interval)
    )


def parameter(value: dict[str, object]) -> bytes:
    return (
        text(value["name"])
        + f64(value["value"])
        + text(value["unit"])
        + text(value["provenance"])
        + optional(value["uncertainty"], uncertainty)
    )


def metric(value: dict[str, object]) -> bytes:
    return (
        text(value["name"])
        + f64(value["value"])
        + text(value["unit"])
        + optional(value["uncertainty"], uncertainty)
    )


def list_of(values: list, encoder) -> bytes:
    return u64(len(values)) + b"".join(encoder(value) for value in values)


def encode_request(value: dict[str, object]) -> bytes:
    return (
        REQUEST_DOMAIN
        + text(value["id"])
        + bytes([DOMAIN_TAG[value["domain"]]])
        + bytes([SOLVER_TAG[value["solver"]]])
        + text(value["objective"])
        + list_of(value["parameters"], parameter)
        + list_of(value["requested_metrics"], text)
    )


def encode_output(value: dict[str, object]) -> bytes:
    converged = b"\x01" if value["converged"] else b"\x00"
    return (
        OUTPUT_DOMAIN
        + text(value["request_id"])
        + converged
        + f64(value["confidence"])
        + uncertainty(value["uncertainty"])
        + list_of(value["metrics"], metric)
        + list_of(value["warnings"], text)
    )


def verify_vector(name: str, vector: dict[str, object], encoder) -> None:
    canonical = encoder(vector["value"])
    expected = bytes.fromhex(vector["canonical_hex"])
    if canonical != expected:
        raise SystemExit(f"{name}: independent canonical bytes do not match frozen vector")
    digest = hashlib.sha256(canonical).hexdigest()
    if digest != vector["sha256"]:
        raise SystemExit(f"{name}: SHA-256 does not match frozen vector")


def main() -> None:
    path = Path(__file__).with_name("v1.json")
    vectors = json.loads(path.read_text(encoding="utf-8"))
    if vectors["profile"] != "simulation-provider-v1-canonical-v1":
        raise SystemExit("unexpected digest profile")
    verify_vector("request", vectors["request"], encode_request)
    verify_vector("output", vectors["output"], encode_output)
    print("simulation canonical digest v1 vectors verified independently")


if __name__ == "__main__":
    main()
