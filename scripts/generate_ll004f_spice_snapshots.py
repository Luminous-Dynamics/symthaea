#!/usr/bin/env python3
"""Generate provenance-bound LL-004F lunar/cislunar SPICE snapshots.

The runtime physics path does not use this tool. It is an offline evidence
producer: callers provide an explicit kernel directory plus a checked-in JSON
configuration, and the tool emits a normalized JSON fixture containing the
actual kernel hashes, orientation samples, angular velocity estimates, and
requested moving-body states.

Real generation requires ``spiceypy``. ``--self-test`` is dependency-free.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

SCHEMA_VERSION = "ll004f.spice-snapshot.v1"
SECONDS_PER_DAY = 86_400.0


class SnapshotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode(
        "utf-8"
    )


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def determinant3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def orthonormality_error(rotation: list[list[float]]) -> float:
    product = matmul(rotation, transpose(rotation))
    return max(
        abs(product[i][j] - (1.0 if i == j else 0.0))
        for i in range(3)
        for j in range(3)
    )


def angular_velocity_from_rotations(
    rotation_minus: list[list[float]],
    rotation_now: list[list[float]],
    rotation_plus: list[list[float]],
    dt_s: float,
) -> list[float]:
    """Return omega of body-fixed relative to inertial, expressed inertially.

    ``rotation_*`` map body-fixed coordinates into inertial coordinates.
    For R(t), the skew matrix ``Rdot R^T`` is [omega]_x.
    """
    if not math.isfinite(dt_s) or dt_s <= 0.0:
        raise SnapshotError("finite-difference interval must be positive")
    rdot = [
        [
            (rotation_plus[i][j] - rotation_minus[i][j]) / (2.0 * dt_s)
            for j in range(3)
        ]
        for i in range(3)
    ]
    omega_matrix = matmul(rdot, transpose(rotation_now))
    skew = [
        [0.5 * (omega_matrix[i][j] - omega_matrix[j][i]) for j in range(3)]
        for i in range(3)
    ]
    return [skew[2][1], skew[0][2], skew[1][0]]


def as_matrix3(value: Any) -> list[list[float]]:
    matrix = [[float(value[i][j]) for j in range(3)] for i in range(3)]
    if any(not math.isfinite(item) for row in matrix for item in row):
        raise SnapshotError("non-finite rotation matrix")
    return matrix


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "lineage_id",
        "kernels",
        "body_fixed_frame",
        "inertial_frame",
        "epochs",
        "targets",
        "orientation_finite_difference_s",
        "output_filename",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise SnapshotError(f"config missing fields: {', '.join(missing)}")
    if not isinstance(config["kernels"], list) or not config["kernels"]:
        raise SnapshotError("kernels must be a non-empty list")
    if not isinstance(config["epochs"], list) or not config["epochs"]:
        raise SnapshotError("epochs must be a non-empty list")
    if not isinstance(config["targets"], list):
        raise SnapshotError("targets must be a list")
    return config


def kernel_manifest(config: dict[str, Any], kernel_dir: Path) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in config["kernels"]:
        filename = str(entry.get("filename", "")).strip()
        role = str(entry.get("role", "")).strip()
        source_url = str(entry.get("source_url", "")).strip()
        if not filename or not role or not source_url:
            raise SnapshotError("each kernel needs filename, role, and source_url")
        if filename in seen:
            raise SnapshotError(f"duplicate kernel filename: {filename}")
        seen.add(filename)
        path = kernel_dir / filename
        if not path.is_file():
            raise SnapshotError(f"missing kernel: {path}")
        manifest.append(
            {
                "filename": filename,
                "role": role,
                "source_url": source_url,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return manifest


def load_spice() -> Any:
    try:
        return importlib.import_module("spiceypy")
    except ModuleNotFoundError as exc:
        raise SnapshotError(
            "real generation requires spiceypy/CSPICE; --self-test does not"
        ) from exc


def target_state(spice: Any, target: dict[str, Any], et: float, frame: str) -> dict[str, Any]:
    name = str(target.get("name", "")).strip()
    observer = str(target.get("observer", "")).strip()
    aberration = str(target.get("aberration", "NONE")).strip()
    if not name or not observer:
        raise SnapshotError("target entries require name and observer")
    state, light_time = spice.spkezr(name, et, frame, aberration, observer)
    state = [float(x) for x in state]
    return {
        "target": name,
        "observer": observer,
        "aberration": aberration,
        "position_km": state[:3],
        "velocity_km_s": state[3:],
        "one_way_light_time_s": float(light_time),
    }


def generate_snapshot(config_path: Path, kernel_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    kernels = kernel_manifest(config, kernel_dir)
    spice = load_spice()
    dt = float(config["orientation_finite_difference_s"])
    if not math.isfinite(dt) or dt <= 0.0:
        raise SnapshotError("orientation_finite_difference_s must be positive")

    spice.kclear()
    try:
        for kernel in config["kernels"]:
            spice.furnsh(str(kernel_dir / kernel["filename"]))

        samples: list[dict[str, Any]] = []
        for epoch_text in config["epochs"]:
            epoch_text = str(epoch_text).strip()
            if not epoch_text:
                raise SnapshotError("epochs cannot contain empty values")
            et = float(spice.str2et(epoch_text))
            rotation_now = as_matrix3(
                spice.pxform(config["body_fixed_frame"], config["inertial_frame"], et)
            )
            rotation_minus = as_matrix3(
                spice.pxform(config["body_fixed_frame"], config["inertial_frame"], et - dt)
            )
            rotation_plus = as_matrix3(
                spice.pxform(config["body_fixed_frame"], config["inertial_frame"], et + dt)
            )
            ortho_error = orthonormality_error(rotation_now)
            determinant = determinant3(rotation_now)
            if ortho_error > 1.0e-10 or abs(determinant - 1.0) > 1.0e-10:
                raise SnapshotError(
                    f"rotation validation failed at {epoch_text}: "
                    f"orthonormality={ortho_error}, det={determinant}"
                )
            omega = angular_velocity_from_rotations(
                rotation_minus, rotation_now, rotation_plus, dt
            )
            samples.append(
                {
                    "epoch_input": epoch_text,
                    "epoch_et_s": et,
                    "epoch_tdb_jd": float(spice.unitim(et, "ET", "JDTDB")),
                    "body_fixed_to_inertial_rotation": rotation_now,
                    "rotation_determinant": determinant,
                    "rotation_orthonormality_max_abs_error": ortho_error,
                    "angular_velocity_body_fixed_relative_to_inertial_rad_s": omega,
                    "angular_velocity_expression_frame": config["inertial_frame"],
                    "targets": [
                        target_state(spice, target, et, config["inertial_frame"])
                        for target in config["targets"]
                    ],
                }
            )
    finally:
        spice.kclear()

    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "lineage_id": config["lineage_id"],
        "generator": {
            "script": script_path.name,
            "script_sha256": sha256_file(script_path),
            "config_filename": config_path.name,
            "config_sha256": sha256_file(config_path),
            "python_version": platform.python_version(),
            "spiceypy_version": getattr(spice, "__version__", "unknown"),
        },
        "frames": {
            "body_fixed": config["body_fixed_frame"],
            "inertial": config["inertial_frame"],
            "orientation_convention": "r_inertial = R_body_fixed_to_inertial * r_body_fixed",
            "angular_velocity_convention": (
                "body-fixed frame relative to inertial, expressed in inertial coordinates; "
                "derived from centered finite difference of Rdot*R^T"
            ),
        },
        "kernel_manifest": kernels,
        "orientation_finite_difference_s": dt,
        "samples": samples,
        "evidence_refs": list(config.get("evidence_refs", [])),
        "non_claims": list(config.get("non_claims", [])),
    }


def write_snapshot(snapshot: dict[str, Any], output: Path, force: bool) -> None:
    payload = canonical_json_bytes(snapshot)
    if output.exists() and not force:
        existing = output.read_bytes()
        if existing == payload:
            return
        raise SnapshotError(
            f"refusing to replace differing evidence fixture {output}; "
            "choose a new lineage/output name or pass --force for an intentional rewrite"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)


def rotation_z(theta: float) -> list[list[float]]:
    c, s = math.cos(theta), math.sin(theta)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def self_test() -> None:
    identity = rotation_z(0.0)
    assert orthonormality_error(identity) < 1.0e-15
    assert abs(determinant3(identity) - 1.0) < 1.0e-15

    rate = 2.5e-6
    dt = 0.25
    r_minus = rotation_z(-rate * dt)
    r_plus = rotation_z(rate * dt)
    omega = angular_velocity_from_rotations(r_minus, identity, r_plus, dt)
    assert abs(omega[0]) < 1.0e-14
    assert abs(omega[1]) < 1.0e-14
    assert abs(omega[2] - rate) < 1.0e-14

    quarter = rotation_z(math.pi / 2.0)
    assert orthonormality_error(quarter) < 1.0e-15
    assert abs(determinant3(quarter) - 1.0) < 1.0e-15

    payload_a = canonical_json_bytes({"b": 2, "a": [1, 3]})
    payload_b = canonical_json_bytes({"a": [1, 3], "b": 2})
    assert payload_a == payload_b


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--kernel-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research/evidence/ll004f"))
    parser.add_argument("--verify-kernels", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-004F self-test: PASS")
        return 0
    if args.config is None or args.kernel_dir is None:
        raise SnapshotError("--config and --kernel-dir are required for kernel verification/generation")

    config = load_config(args.config)
    if args.verify_kernels:
        manifest = kernel_manifest(config, args.kernel_dir)
        sys.stdout.buffer.write(canonical_json_bytes({"kernel_manifest": manifest}))
        return 0

    snapshot = generate_snapshot(args.config, args.kernel_dir)
    output = args.output_dir / str(config["output_filename"])
    write_snapshot(snapshot, output, args.force)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SnapshotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
