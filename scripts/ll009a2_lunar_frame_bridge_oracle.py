#!/usr/bin/env python3
"""Independent LL-009A2 lunar body-frame reconciliation oracle.

This tool does not read SPICE or GIS data. It consumes provenance-bound source
and destination body-fixed->common-inertial rotation samples, verifies their
frame/epoch closure, and transforms Cartesian lunar surface/query vectors
between the body-fixed frames. It is intentionally independent of Symthaea.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


class BridgeError(RuntimeError):
    pass


def transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*m)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def matvec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def determinant3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def as_rotation(value: Any) -> list[list[float]]:
    try:
        matrix = [[float(value[i][j]) for j in range(3)] for i in range(3)]
    except (TypeError, ValueError, IndexError) as exc:
        raise BridgeError("rotation must be a finite 3x3 matrix") from exc
    if any(not math.isfinite(x) for row in matrix for x in row):
        raise BridgeError("rotation contains non-finite value")
    product = matmul(matrix, transpose(matrix))
    error = max(abs(product[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3))
    det = determinant3(matrix)
    if error > 1.0e-10 or abs(det - 1.0) > 1.0e-10:
        raise BridgeError(f"rotation is not proper orthonormal: error={error}, det={det}")
    return matrix


def validate_snapshot(snapshot: dict[str, Any]) -> list[list[float]]:
    for field in ("body_fixed_frame", "inertial_frame", "timescale", "source_ref"):
        if not isinstance(snapshot.get(field), str) or not snapshot[field].strip():
            raise BridgeError(f"orientation snapshot missing {field}")
    epoch = snapshot.get("epoch_jd")
    if not isinstance(epoch, (int, float)) or not math.isfinite(epoch) or epoch <= 0.0:
        raise BridgeError("orientation snapshot has invalid epoch_jd")
    return as_rotation(snapshot.get("body_fixed_to_inertial_rotation"))


def bridge_rotation(source: dict[str, Any], destination: dict[str, Any], epoch_tolerance_days: float = 1.0e-12) -> list[list[float]]:
    r_src = validate_snapshot(source)
    r_dst = validate_snapshot(destination)
    if source["inertial_frame"] != destination["inertial_frame"]:
        raise BridgeError("source/destination orientation snapshots use different inertial frames")
    if source["timescale"] != destination["timescale"]:
        raise BridgeError("source/destination orientation snapshots use different timescales")
    if abs(float(source["epoch_jd"]) - float(destination["epoch_jd"])) > epoch_tolerance_days:
        raise BridgeError("source/destination orientation snapshots are not at the same epoch")
    # r_inertial = R_src * r_src ; r_dst = R_dst^T * r_inertial
    return matmul(transpose(r_dst), r_src)


def transform_vector(vector: list[float], source: dict[str, Any], destination: dict[str, Any]) -> dict[str, Any]:
    if len(vector) != 3 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in vector):
        raise BridgeError("vector must contain three finite scalars")
    source_vector = [float(x) for x in vector]
    bridge = bridge_rotation(source, destination)
    result = matvec(bridge, source_vector)
    source_norm = norm(source_vector)
    destination_norm = norm(result)
    norm_error = abs(destination_norm - source_norm)
    if norm_error > max(1.0e-12, source_norm * 1.0e-12):
        raise BridgeError(f"rigid transform failed norm check: {norm_error}")
    return {
        "source_frame": source["body_fixed_frame"],
        "destination_frame": destination["body_fixed_frame"],
        "common_inertial_frame": source["inertial_frame"],
        "epoch_jd": float(source["epoch_jd"]),
        "timescale": source["timescale"],
        "source_vector": source_vector,
        "destination_vector": result,
        "source_radius": source_norm,
        "destination_radius": destination_norm,
        "norm_error": norm_error,
        "bridge_rotation": bridge,
        "source_orientation_ref": source["source_ref"],
        "destination_orientation_ref": destination["source_ref"],
    }


def rotation_z(theta: float) -> list[list[float]]:
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def snapshot(name: str, rotation: list[list[float]], epoch: float = 2460000.5) -> dict[str, Any]:
    return {
        "body_fixed_frame": name,
        "inertial_frame": "SYNTHETIC_INERTIAL",
        "timescale": "TDB",
        "epoch_jd": epoch,
        "body_fixed_to_inertial_rotation": rotation,
        "source_ref": f"synthetic:{name}",
    }


def self_test() -> None:
    identity = snapshot("SRC", rotation_z(0.0))
    same = snapshot("DST", rotation_z(0.0))
    value = transform_vector([1.0, 2.0, 3.0], identity, same)
    assert max(abs(a - b) for a, b in zip(value["destination_vector"], [1.0, 2.0, 3.0])) < 1.0e-12

    # Destination frame is +90 degrees relative to the common inertial frame,
    # so an inertial/source +X vector has destination coordinates -Y.
    dst90 = snapshot("DST90", rotation_z(math.pi / 2.0))
    value = transform_vector([1.0, 0.0, 0.0], identity, dst90)
    assert abs(value["destination_vector"][0]) < 1.0e-12
    assert abs(value["destination_vector"][1] + 1.0) < 1.0e-12

    arbitrary_src = snapshot("A", rotation_z(0.123))
    arbitrary_dst = snapshot("B", rotation_z(-0.456))
    original = [1737.4, -12.0, 3.0]
    forward = transform_vector(original, arbitrary_src, arbitrary_dst)
    reverse = transform_vector(forward["destination_vector"], arbitrary_dst, arbitrary_src)
    assert norm(sub(reverse["destination_vector"], original)) < 1.0e-9
    assert abs(forward["destination_radius"] - norm(original)) < 1.0e-9

    bad = snapshot("BAD", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.0, 1.0]])
    try:
        transform_vector([1.0, 0.0, 0.0], bad, same)
    except BridgeError:
        pass
    else:
        raise AssertionError("non-orthonormal rotation must fail")

    shifted = snapshot("SHIFTED", rotation_z(0.0), epoch=2460000.6)
    try:
        transform_vector([1.0, 0.0, 0.0], identity, shifted)
    except BridgeError:
        pass
    else:
        raise AssertionError("epoch mismatch must fail")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON containing source_orientation, destination_orientation, vector")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009A2 self-test: PASS")
        return 0
    if args.input is None:
        raise BridgeError("--input is required unless --self-test is used")
    request = json.loads(args.input.read_text(encoding="utf-8"))
    result = transform_vector(request["vector"], request["source_orientation"], request["destination_orientation"])
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BridgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
