#!/usr/bin/env python3
"""Execute the pinned D0FUS ITER reference deck and emit a replay receipt.

This is an integration harness for the external D0FUS program, not production
Luminous Dynamics code. It supports two modes:
  - wheel: import the checksum-verified PyPI wheel;
  - source: import the exact upstream Git commit.

Both modes run the same immutable ITER deck with figures disabled and emit a
canonical digest of the returned output tuple.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any


REQUIRED_DISTRIBUTIONS = (
    "numpy",
    "scipy",
    "matplotlib",
    "tqdm",
    "deap",
    "joblib",
)

# Core results tuple indices are defined by the exact pinned D0FUS RUN
# implementation. The last 15 entries are the plant power-balance extension.
CORE_OUTPUTS = {
    "B0_T": (0, True),
    "tau_E_s": (3, True),
    "Q": (5, True),
    "I_p_MA": (8, True),
    "P_elec_MW": (25, False),
    "P_wallplug_MW": (26, True),
    "P_th_MW": (-15, True),
    "P_e_gross_MWe": (-12, True),
    "P_recirc_var_MWe": (-5, True),
    "P_recirc_fix_MWe": (-4, True),
    "P_recirculated_MWe": (-3, True),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize(value: Any) -> Any:
    """Convert D0FUS tuple values to stable JSON-compatible primitives."""
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("NumPy is required by the pinned D0FUS runtime") from exc

    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": canonicalize(value.tolist()),
        }
    if isinstance(value, np.generic):
        return canonicalize(value.item())
    if isinstance(value, dict):
        return {
            str(key): canonicalize(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "nan"}
        if math.isinf(value):
            return {"__float__": "positive-infinity" if value > 0 else "negative-infinity"}
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, bytes):
        return {"__type__": "bytes", "hex": value.hex()}
    raise TypeError(f"Unsupported D0FUS output value type: {type(value)!r}")


def ensure_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"{label} is outside expected root: {path}") from exc


def verify_upstream_identity(source_root: Path, lock: dict[str, Any]) -> None:
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if actual_commit != lock["git_commit"]:
        raise RuntimeError(
            f"Wrong D0FUS commit: expected {lock['git_commit']}, got {actual_commit}"
        )

    deck = source_root / lock["deck_path"]
    pyproject = source_root / lock["pyproject_path"]
    if not deck.is_file():
        raise FileNotFoundError(f"Pinned ITER deck not found: {deck}")
    if not pyproject.is_file():
        raise FileNotFoundError(f"Pinned pyproject.toml not found: {pyproject}")

    deck_blob = subprocess.check_output(
        ["git", "-C", str(source_root), "hash-object", str(deck)],
        text=True,
    ).strip()
    if deck_blob != lock["deck_git_blob_sha"]:
        raise RuntimeError(
            f"ITER deck Git blob mismatch: expected {lock['deck_git_blob_sha']}, got {deck_blob}"
        )

    pyproject_blob = subprocess.check_output(
        ["git", "-C", str(source_root), "hash-object", str(pyproject)],
        text=True,
    ).strip()
    if pyproject_blob != lock["pyproject_git_blob_sha"]:
        raise RuntimeError(
            "pyproject.toml Git blob mismatch: "
            f"expected {lock['pyproject_git_blob_sha']}, got {pyproject_blob}"
        )

    with pyproject.open("rb") as stream:
        project = tomllib.load(stream)
    actual_version = project.get("project", {}).get("version")
    if actual_version != lock["version"]:
        raise RuntimeError(
            f"Source project version mismatch: expected {lock['version']}, got {actual_version}"
        )


def verify_runtime(lock: dict[str, Any]) -> dict[str, str]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if version != lock["python_minor"]:
        raise RuntimeError(
            f"Expected Python {lock['python_minor']}, got {version}"
        )
    installed = {"python": platform.python_version()}
    for distribution in REQUIRED_DISTRIBUTIONS:
        installed[distribution] = importlib.metadata.version(distribution)

    if importlib.metadata.version("d0fus") != lock["version"]:
        raise RuntimeError(
            "Installed d0fus distribution version mismatch: "
            f"expected {lock['version']}, got {importlib.metadata.version('d0fus')}"
        )
    return installed


def verify_wheel_module_parity(source_root: Path, wheel_root: Path) -> dict[str, Any]:
    """Check every packaged Python module against the exact source checkout."""
    module_paths = [Path("D0FUS.py")]
    for package in ("D0FUS_BIB", "D0FUS_EXE"):
        module_paths.extend(
            sorted(
                path.relative_to(source_root)
                for path in (source_root / package).rglob("*.py")
                if path.is_file()
            )
        )

    missing: list[str] = []
    differing: list[dict[str, str]] = []
    identical_count = 0
    for relative in module_paths:
        source_path = source_root / relative
        wheel_path = wheel_root / relative
        if not wheel_path.is_file():
            missing.append(relative.as_posix())
            continue
        source_sha = sha256_file(source_path)
        wheel_sha = sha256_file(wheel_path)
        if source_sha == wheel_sha:
            identical_count += 1
        else:
            differing.append(
                {
                    "path": relative.as_posix(),
                    "source_sha256": source_sha,
                    "wheel_sha256": wheel_sha,
                }
            )

    # Module divergence is evidence, not an automatic skip. Both artifacts
    # are executed separately; the parity step evaluates actual results.
    return {
        "module_count": len(module_paths),
        "byte_identical_count": identical_count,
        "missing": missing,
        "differing": differing,
        "result": "byte-identical" if not missing and not differing else "source-package-divergence",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("wheel", "source"), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    source_root = args.source_root.resolve()
    deck = args.deck.resolve()
    ensure_inside(deck, source_root, "ITER deck")
    verify_upstream_identity(source_root, lock)

    deck_sha256 = sha256_file(deck)
    runtime_versions = verify_runtime(lock)

    # Source mode deliberately adds the pinned checkout before importing
    # D0FUS. Wheel mode deliberately does not, so the installed wheel is used.
    if args.mode == "source":
        sys.path.insert(0, str(source_root))

    from D0FUS_BIB.D0FUS_parameterization import DEFAULT_CONFIG
    from D0FUS_EXE import D0FUS_run

    module_path = Path(D0FUS_run.__file__).resolve()
    wheel_root = module_path.parents[1]
    if args.mode == "source":
        ensure_inside(module_path, source_root, "D0FUS source module")
        module_parity: dict[str, Any] | None = None
    else:
        purelib = Path(__import__("sysconfig").get_paths()["purelib"]).resolve()
        ensure_inside(module_path, purelib, "installed D0FUS wheel module")
        module_parity = verify_wheel_module_parity(source_root, wheel_root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = D0FUS_run.main(
        input_file=str(deck),
        save_figures=False,
        verbose=1,
        output_dir=str(args.output_dir),
        view_3d=False,
    )

    if not isinstance(results, tuple):
        raise RuntimeError(f"D0FUS RUN did not return a tuple: {type(results)!r}")
    expected_length = int(lock["observed_run_output_tuple_length"])
    if len(results) != expected_length:
        raise RuntimeError(
            f"D0FUS v3.2.0 output tuple schema drift: expected {expected_length}, got {len(results)}"
        )
    if len(results) <= max(index for index, _ in CORE_OUTPUTS.values()):
        raise RuntimeError(
            f"D0FUS output tuple is too short for the declared core observables: {len(results)}"
        )

    core_observables: dict[str, float | None] = {}
    for name, (index, must_be_positive) in CORE_OUTPUTS.items():
        value = float(results[index])
        if not math.isfinite(value):
            if must_be_positive:
                raise RuntimeError(
                    f"Core D0FUS output {name} is non-finite at result index {index}"
                )
            core_observables[name] = None
            continue
        if must_be_positive and value <= 0.0:
            raise RuntimeError(
                f"Core D0FUS output {name} is not positive: {value}"
            )
        core_observables[name] = value

    canonical = canonicalize(results)
    canonical_bytes = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    reports = sorted(args.output_dir.rglob("output_detailed.txt"))
    if len(reports) != 1:
        raise RuntimeError(
            f"Expected exactly one output_detailed.txt report, found {len(reports)}"
        )
    report = reports[0]

    config_field_count = len(dataclasses.fields(type(DEFAULT_CONFIG)))
    expected_config_fields = int(lock["observed_global_config_dataclass_field_count"])
    if config_field_count != expected_config_fields:
        raise RuntimeError(
            "D0FUS GlobalConfig schema drift: "
            f"expected {expected_config_fields} fields, got {config_field_count}"
        )
    net_electric = core_observables["P_elec_MW"]
    gross_electric = core_observables["P_e_gross_MWe"]
    recirculated_electric = core_observables["P_recirculated_MWe"]
    variable_recirculated = core_observables["P_recirc_var_MWe"]
    fixed_recirculated = core_observables["P_recirc_fix_MWe"]
    engineering_gain = (
        gross_electric / recirculated_electric
        if recirculated_electric is not None and recirculated_electric > 0.0
        else None
    )
    power_balance = {
        "gross_electric_mwe": gross_electric,
        "variable_recirculated_mwe": variable_recirculated,
        "fixed_recirculated_mwe": fixed_recirculated,
        "total_recirculated_mwe": recirculated_electric,
        "reported_net_electric_mwe": net_electric,
        "calculated_total_recirculated_mwe": (
            variable_recirculated + fixed_recirculated
            if variable_recirculated is not None and fixed_recirculated is not None
            else None
        ),
        "calculated_net_electric_mwe": (
            gross_electric - variable_recirculated - fixed_recirculated
            if gross_electric is not None
            and variable_recirculated is not None
            and fixed_recirculated is not None
            else None
        ),
    }
    power_balance["total_recirculated_residual_mw"] = (
        recirculated_electric - power_balance["calculated_total_recirculated_mwe"]
        if recirculated_electric is not None
        and power_balance["calculated_total_recirculated_mwe"] is not None
        else None
    )
    power_balance["net_electric_residual_mw"] = (
        net_electric - power_balance["calculated_net_electric_mwe"]
        if net_electric is not None
        and power_balance["calculated_net_electric_mwe"] is not None
        else None
    )
    balance_tolerance_mw = 1e-6
    for residual_name in (
        "total_recirculated_residual_mw",
        "net_electric_residual_mw",
    ):
        residual = power_balance[residual_name]
        if residual is None or abs(residual) > balance_tolerance_mw:
            raise RuntimeError(
                f"D0FUS ITER power-balance identity {residual_name} "
                f"exceeds {balance_tolerance_mw} MW: {residual}"
            )
    power_balance["absolute_balance_tolerance_mw"] = balance_tolerance_mw
    receipt = {
        "receipt_version": "d0fus-reference-run/0.2",
        "mode": args.mode,
        "solver": lock["solver"],
        "solver_version": lock["version"],
        "source_git_ref": lock["git_ref"],
        "source_git_commit": lock["git_commit"],
        "deck_path": lock["deck_path"],
        "deck_git_blob_sha": lock["deck_git_blob_sha"],
        "deck_sha256": deck_sha256,
        "pyproject_git_blob_sha": lock["pyproject_git_blob_sha"],
        "pypi_distribution": lock["pypi_distribution"],
        "wheel_sha256": lock["wheel_sha256"],
        "sdist_sha256": lock["sdist_sha256"],
        "python_and_dependency_versions": runtime_versions,
        "module_origin": str(module_path),
        "wheel_source_module_parity": module_parity,
        "global_config_dataclass_field_count": config_field_count,
        "run_output_tuple_length": len(results),
        "canonical_output_tuple_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "core_observables": core_observables,
        "power_balance": power_balance,
        "power_balance_interpretation": {
            "gross_electric_over_recirculated_power": engineering_gain,
            "net_electricity_state": (
                "positive_net_export"
                if net_electric is not None and net_electric > 0.0
                else "zero_or_negative_net_export"
            ),
            "annual_cost_of_electricity_defined": (
                net_electric is not None and net_electric > 0.0
            ),
            "note": (
                "A negative P_elec is not a failed RUN; it means this reference case "
                "does not export net electricity."
            ),
        },
        "output_report_relative_path": str(report.relative_to(args.output_dir)),
        "output_report_sha256": sha256_file(report),
        "output_report_bytes": report.stat().st_size,
        "claim_ceiling": (
            "pinned D0FUS ITER reference-deck execution with source/wheel module "
            "divergence inventory; not independent physical validation"
        ),
    }

    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("D0FUS_REFERENCE_RECEIPT=" + str(args.receipt))
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
