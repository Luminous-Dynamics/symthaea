#!/usr/bin/env python3
"""One-time NASLib NAS-Bench-201 learning-curve -> multi-fidelity canonical import.

Security: pickle deserialization can execute code. This tool checks the required raw SHA-256
before calling pickle.load, but an expected digest only protects you when it is obtained from
a trusted, independently verified source. Never use this tool on an untrusted pickle.

Observed NASLib loader contract:
  get_nasbench201_api(dataset) -> {"nb201_data": pickle.load(...)}
  nb201_data[arch_string][dataset_key]["eval_acc1es"] -> 200-point validation-accuracy curve
  CIFAR-10 maps to "cifar10-valid"; CIFAR-100 maps to "cifar100".

The script merges those trajectories with the public proxy feature rows from a prior SuiteZero
canonical import. It writes a canonical multi-fidelity import for the Rust evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

CANONICAL_SCHEMA = "symthaea.tabular-nas.multi-fidelity-canonical-import"
CANONICAL_VERSION = "0.1.0"
SUITEZERO_SCHEMA = "symthaea.tabular-nas.canonical-import"
BENCHMARK_SCHEMA = "symthaea.tabular-nas.benchmark"
NASLIB_REPO = "automl/NASLib"
NASLIB_BRANCH = "zerocost"
NASLIB_SOURCE_COMMIT = "8cb5d2ba1e29784de43039d9824c68e88fb1a1da"

NB201_CURVE_FILES = {
    "cifar10": "nb201_cifar10_full_training.pickle",
    "cifar100": "nb201_cifar100_full_training.pickle",
    "ImageNet16-120": "nb201_ImageNet16_full_training.pickle",
}
NB201_DATASET_KEYS = {
    "cifar10": "cifar10-valid",
    "cifar100": "cifar100",
    "ImageNet16-120": "ImageNet16-120",
}
NB201_GOOGLE_DRIVE_IDS = {
    "cifar10": "1sh8pEhdrgZ97-VFBVL94rI36gedExVgJ",
    "cifar100": "1hV6-mCUKInIK1iqZ0jfBkcKaFmftlBtp",
    "ImageNet16-120": "1FVCn54aQwD6X6NazaIZ_yjhj47mOGdIH",
}


class ImportErrorDetail(ValueError):
    pass


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_epochs(value: str) -> tuple[int, ...]:
    try:
        epochs = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ImportErrorDetail("epochs must be comma-separated positive integers") from error
    if not epochs:
        raise ImportErrorDetail("at least one fidelity epoch is required")
    if any(epoch < 1 for epoch in epochs):
        raise ImportErrorDetail("fidelity epochs are one-based and must be positive")
    if tuple(sorted(set(epochs))) != epochs:
        raise ImportErrorDetail("fidelity epochs must be strictly increasing and unique")
    return epochs


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImportErrorDetail(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ImportErrorDetail(f"{field} must be finite")
    return number


def _strict_sha256(value: str, field: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value or ""):
        raise ImportErrorDetail(f"{field} must be exactly 64 hexadecimal characters")
    return value.lower()


def _read_suitezero(path: Path) -> tuple[dict[str, Any], bytes, str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ImportErrorDetail(f"cannot read SuiteZero canonical import: {error}") from error
    digest = sha256_hex(raw)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ImportErrorDetail(f"invalid SuiteZero canonical JSON: {error}") from error
    if not isinstance(value, dict):
        raise ImportErrorDetail("SuiteZero canonical import root must be an object")
    if value.get("schema") != SUITEZERO_SCHEMA or value.get("version") != CANONICAL_VERSION:
        raise ImportErrorDetail("unsupported SuiteZero canonical import schema/version")
    if not isinstance(value.get("manifest"), dict):
        raise ImportErrorDetail("SuiteZero canonical import is missing manifest")
    if not isinstance(value.get("architectures"), list) or not isinstance(value.get("target_labels"), list):
        raise ImportErrorDetail("SuiteZero canonical import requires architectures and target_labels arrays")
    return value, raw, digest


def _load_trusted_pickle(path: Path, expected_sha256: str) -> tuple[Any, bytes, str]:
    expected = _strict_sha256(expected_sha256, "expected_curve_artifact_sha256")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ImportErrorDetail(f"cannot read learning-curve artifact: {error}") from error
    actual = sha256_hex(raw)
    if actual != expected:
        raise ImportErrorDetail(
            f"learning-curve artifact SHA-256 mismatch: expected {expected}, got {actual}; refusing to unpickle"
        )
    # WARNING: unpickling is unsafe for untrusted content. The expected digest must be trusted
    # independently; a caller-supplied digest of malicious bytes is not a security boundary.
    try:
        data = pickle.loads(raw)
    except Exception as error:
        raise ImportErrorDetail(f"could not deserialize trusted NAS-Bench-201 pickle: {error}") from error
    return data, raw, actual


def build_multifidelity_import(
    *,
    suitezero_canonical: dict[str, Any],
    suitezero_source_sha256: str,
    curve_data: Any,
    curve_source_sha256: str,
    dataset: str,
    epochs: tuple[int, ...],
    max_final_queries: int,
    max_total_cost_units: int,
    final_accuracy_tolerance: float = 0.1,
    fidelity_cost_model: str = "resumable_prefix",
) -> tuple[dict[str, Any], dict[str, Any]]:
    if dataset not in NB201_DATASET_KEYS:
        raise ImportErrorDetail(f"unsupported NAS-Bench-201 dataset {dataset!r}")
    if not epochs or tuple(sorted(set(epochs))) != epochs or any(epoch < 1 for epoch in epochs):
        raise ImportErrorDetail("epochs must be strictly increasing positive integers")
    if not isinstance(max_final_queries, int) or isinstance(max_final_queries, bool) or max_final_queries < 1:
        raise ImportErrorDetail("max_final_queries must be a positive integer")
    if not isinstance(max_total_cost_units, int) or isinstance(max_total_cost_units, bool) or max_total_cost_units < 1:
        raise ImportErrorDetail("max_total_cost_units must be a positive integer")
    if not math.isfinite(final_accuracy_tolerance) or final_accuracy_tolerance < 0.0:
        raise ImportErrorDetail("final_accuracy_tolerance must be finite and non-negative")
    if fidelity_cost_model not in {"resumable_prefix", "independent_runs"}:
        raise ImportErrorDetail("fidelity_cost_model must be resumable_prefix or independent_runs")

    suitezero_manifest = suitezero_canonical["manifest"]
    if suitezero_manifest.get("benchmark_id") is None:
        raise ImportErrorDetail("SuiteZero manifest requires benchmark_id")
    if dataset not in suitezero_manifest.get("benchmark_id", ""):
        # Dataset identity is explicit but need not be encoded in every local benchmark ID.
        # The target manifest source_description is also checked below.
        if f"dataset={dataset}" not in suitezero_manifest.get("source_description", ""):
            raise ImportErrorDetail("SuiteZero import task identity does not match requested dataset")
    if suitezero_manifest.get("metric_direction") != "maximize":
        raise ImportErrorDetail("SuiteZero validation accuracy source must declare maximize direction")
    if suitezero_manifest.get("metric_id") not in {"val_accuracy", "validation_accuracy"}:
        raise ImportErrorDetail("SuiteZero source metric must be validation accuracy")

    feature_rows = suitezero_canonical["architectures"]
    suitezero_labels = suitezero_canonical["target_labels"]
    if not feature_rows:
        raise ImportErrorDetail("SuiteZero feature table is empty")

    features_by_id: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(feature_rows):
        architecture_id = feature.get("architecture_id")
        if not isinstance(architecture_id, str) or not architecture_id:
            raise ImportErrorDetail(f"SuiteZero architecture row {index} has invalid architecture_id")
        if architecture_id in features_by_id:
            raise ImportErrorDetail(f"duplicate SuiteZero architecture ID {architecture_id!r}")
        features_by_id[architecture_id] = feature

    suitezero_target_by_id: dict[str, float] = {}
    for index, label in enumerate(suitezero_labels):
        architecture_id = label.get("architecture_id")
        if architecture_id not in features_by_id:
            raise ImportErrorDetail(f"SuiteZero target label {index} references unknown architecture")
        if architecture_id in suitezero_target_by_id:
            raise ImportErrorDetail(f"duplicate SuiteZero target label for {architecture_id!r}")
        suitezero_target_by_id[architecture_id] = _finite_float(label.get("value"), f"SuiteZero target[{index}]")
    if set(suitezero_target_by_id) != set(features_by_id):
        raise ImportErrorDetail("SuiteZero target labels must cover every architecture exactly once")

    if not isinstance(curve_data, dict) or not isinstance(curve_data.get("nb201_data"), dict):
        raise ImportErrorDetail("NASLib pickle must contain a top-level nb201_data dictionary")
    nb201_data = curve_data["nb201_data"]
    dataset_key = NB201_DATASET_KEYS[dataset]
    curve_architecture_keys = set(nb201_data)
    expected_architecture_keys = set(features_by_id)
    if curve_architecture_keys != expected_architecture_keys:
        missing = sorted(expected_architecture_keys - curve_architecture_keys)[:5]
        unexpected = sorted(curve_architecture_keys - expected_architecture_keys)[:5]
        raise ImportErrorDetail(
            f"SuiteZero and learning-curve architecture sets differ; missing curves={missing}, unexpected curves={unexpected}"
        )

    expected_final_epoch = epochs[-1]
    fidelity_levels = [
        {"id": f"epoch_{epoch}", "cumulative_cost_units": epoch}
        for epoch in epochs
    ]
    target_labels: list[dict[str, Any]] = []
    maximum_final_accuracy_delta = 0.0
    minimum_curve_length = None

    for architecture_id in sorted(features_by_id):
        architecture_record = nb201_data[architecture_id]
        if not isinstance(architecture_record, dict) or not isinstance(architecture_record.get(dataset_key), dict):
            raise ImportErrorDetail(f"learning-curve record for {architecture_id!r} is missing dataset {dataset_key!r}")
        task_record = architecture_record[dataset_key]
        curve = task_record.get("eval_acc1es")
        if not isinstance(curve, (list, tuple)):
            raise ImportErrorDetail(f"{architecture_id}.{dataset_key}.eval_acc1es must be a list")
        minimum_curve_length = len(curve) if minimum_curve_length is None else min(minimum_curve_length, len(curve))
        if len(curve) < expected_final_epoch:
            raise ImportErrorDetail(
                f"{architecture_id} has {len(curve)} validation points; requested fidelity requires epoch {expected_final_epoch}"
            )
        final_curve_accuracy = _finite_float(
            curve[expected_final_epoch - 1],
            f"{architecture_id}.{dataset_key}.eval_acc1es[{expected_final_epoch - 1}]",
        )
        source_accuracy = suitezero_target_by_id[architecture_id]
        final_delta = abs(final_curve_accuracy - source_accuracy)
        maximum_final_accuracy_delta = max(maximum_final_accuracy_delta, final_delta)
        if final_delta > final_accuracy_tolerance:
            raise ImportErrorDetail(
                f"final curve accuracy for {architecture_id} differs from SuiteZero val_accuracy by {final_delta:.6g}, "
                f"exceeding tolerance {final_accuracy_tolerance}"
            )
        for fidelity_index, epoch in enumerate(epochs):
            value = _finite_float(
                curve[epoch - 1],
                f"{architecture_id}.{dataset_key}.eval_acc1es[{epoch - 1}]",
            )
            target_labels.append(
                {
                    "architecture_id": architecture_id,
                    "fidelity_index": fidelity_index,
                    "value": value,
                }
            )

    if max_final_queries > len(features_by_id):
        raise ImportErrorDetail("max_final_queries cannot exceed architecture count")
    top_k = min(100, len(features_by_id))
    combined_source = hashlib.sha256(
        b"symthaea.multi-fidelity-source.v0.1"
        + suitezero_source_sha256.encode("ascii")
        + curve_source_sha256.encode("ascii")
    ).hexdigest()

    manifest = {
        "base": {
            **suitezero_manifest,
            "metric_id": "val_accuracy",
            "metric_direction": "maximize",
            "artifact_sha256": combined_source,
            "source_description": (
                f"SuiteZero feature table SHA-256={suitezero_source_sha256}; "
                f"NAS-Bench-201 learning-curve pickle={NB201_CURVE_FILES[dataset]} "
                f"Drive ID={NB201_GOOGLE_DRIVE_IDS[dataset]} SHA-256={curve_source_sha256}; "
                f"dataset={dataset}; curve=eval_acc1es; epoch indexing is 1-based in this import"
            ),
            "max_unique_queries": max_final_queries,
            "top_k": top_k,
        },
        "fidelity_levels": fidelity_levels,
        "fidelity_cost_model": fidelity_cost_model,
        "resource_unit": "epoch_equivalent",
        "max_total_cost_units": max_total_cost_units,
    }

    canonical = {
        "schema": CANONICAL_SCHEMA,
        "version": CANONICAL_VERSION,
        "manifest": manifest,
        "architectures": [features_by_id[key] for key in sorted(features_by_id)],
        "target_labels": target_labels,
    }
    report = {
        "dataset": dataset,
        "dataset_key": dataset_key,
        "architecture_count": len(features_by_id),
        "fidelity_epochs": list(epochs),
        "fidelity_cost_model": fidelity_cost_model,
        "resource_unit": "epoch_equivalent",
        "max_total_cost_units": max_total_cost_units,
        "max_final_queries": max_final_queries,
        "suitezero_raw_sha256": suitezero_source_sha256,
        "learning_curve_raw_sha256": curve_source_sha256,
        "maximum_final_accuracy_delta": maximum_final_accuracy_delta,
        "minimum_curve_length": minimum_curve_length,
        "proxy_feature_bank_total_seconds": sum(
            sum(float(value) for value in row["proxy_times_seconds"].values())
            for row in features_by_id.values()
        ),
    }
    return canonical, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suitezero-import", required=True, type=Path)
    parser.add_argument("--suitezero-sha256", required=True, help="expected digest of the canonical SuiteZero feature import")
    parser.add_argument("--curve-pickle", required=True, type=Path)
    parser.add_argument("--curve-sha256", required=True, help="independently verified expected SHA-256; checked before unpickling")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(NB201_DATASET_KEYS))
    parser.add_argument("--epochs", type=parse_epochs, default=(5, 15, 200))
    parser.add_argument("--max-final-queries", required=True, type=int)
    parser.add_argument("--max-total-cost-units", required=True, type=int)
    parser.add_argument("--final-accuracy-tolerance", type=float, default=0.1)
    parser.add_argument("--fidelity-cost-model", choices=("resumable_prefix", "independent_runs"), default="resumable_prefix")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output_path = args.output.resolve()
    suitezero_path = args.suitezero_import.resolve()
    curve_path = args.curve_pickle.resolve()
    if output_path in {suitezero_path, curve_path}:
        parser.error("output path must differ from input paths")
    if output_path.exists() and not args.force:
        parser.error("output exists; pass --force to replace")

    try:
        suitezero, suitezero_raw, suitezero_actual_sha = _read_suitezero(suitezero_path)
        if sha256_hex(suitezero_raw) != _strict_sha256(args.suitezero_sha256, "suitezero_sha256"):
            raise ImportErrorDetail("SuiteZero canonical import SHA-256 mismatch")
        curve_data, curve_raw, curve_actual_sha = _load_trusted_pickle(curve_path, args.curve_sha256)
        canonical, report = build_multifidelity_import(
            suitezero_canonical=suitezero,
            suitezero_source_sha256=suitezero_actual_sha,
            curve_data=curve_data,
            curve_source_sha256=curve_actual_sha,
            dataset=args.dataset,
            epochs=tuple(args.epochs),
            max_final_queries=args.max_final_queries,
            max_total_cost_units=args.max_total_cost_units,
            final_accuracy_tolerance=args.final_accuracy_tolerance,
            fidelity_cost_model=args.fidelity_cost_model,
        )
        encoded = (
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        report["canonical_import_bytes"] = len(encoded)
        report["canonical_import_sha256"] = sha256_hex(encoded)
        report["suitezero_input_bytes"] = len(suitezero_raw)
        report["curve_pickle_bytes"] = len(curve_raw)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, output_path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise
        print(json.dumps(report, sort_keys=True, indent=2), file=sys.stderr)
        return 0
    except (OSError, ImportErrorDetail) as error:
        print(f"curve import failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
