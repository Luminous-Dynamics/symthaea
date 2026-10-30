#!/usr/bin/env python3
"""One-time NASLib NAS-Bench-Suite-Zero -> Symthaea canonical NAS import.

This is an import-time tool only. The searcher/evaluator runtime is Rust and consumes the
canonical JSON file; this script is not a runtime dependency.

Observed upstream layout (NASLib branch zerocost):
    root[dataset][str(tuple_of_six_NASBench201_op_indices)] = {
        "val_accuracy": ...,
        "jacov": {"score": ..., "time": ...},
        ...
    }

NASLib's ZC evaluator maps +/-inf proxy scores to +/-1e9. We preserve that behaviour and
record the number of substitutions in the import report written to stderr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

NASLIB_REPO = "automl/NASLib"
NASLIB_BRANCH = "zerocost"
NASLIB_OBSERVED_COMMIT = "8cb5d2ba1e29784de43039d9824c68e88fb1a1da"
SUITEZERO_FILENAME = "zc_nasbench201.json"
NASBENCH201_ZC_DRIVE_FILE_ID = "1R7n7GpFHAjUZpPISzbhxH0QjubnvZM5H"

CANONICAL_SCHEMA = "symthaea.tabular-nas.canonical-import"
BENCHMARK_SCHEMA = "symthaea.tabular-nas.benchmark"
CANONICAL_VERSION = "0.1.0"

NASBENCH201_EDGE_COUNT = 6
NASBENCH201_OPERATION_COUNT = 5

ALLOWED_PROXIES = (
    "params",
    "flops",
    "l2_norm",
    "synflow",
    "nwot",
    "jacov",
    "epe_nas",
    "zen",
    "plain",
    "fisher",
    "snip",
    "grad_norm",
    "grasp",
)


class ImportErrorDetail(ValueError):
    """Invalid upstream or manifest input."""


def _reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ImportErrorDetail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json_constant(value: str) -> float:
    # Python's json module accepts these non-standard constants by default. Preserve +/-inf
    # because NASLib explicitly normalizes them; NaN is rejected downstream.
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    if value == "NaN":
        return math.nan
    raise ImportErrorDetail(f"unsupported JSON numeric constant: {value}")


def parse_json_strict(raw: bytes) -> Any:
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ImportErrorDetail(f"artifact is not valid UTF-8: {error}") from error
    try:
        return json.loads(
            source,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_parse_json_constant,
        )
    except (json.JSONDecodeError, ImportErrorDetail) as error:
        raise ImportErrorDetail(f"invalid SuiteZero JSON: {error}") from error


def parse_nasbench201_architecture_key(key: str) -> tuple[int, ...]:
    """Validate the upstream str(tuple) key and return its six operation indices."""
    import ast

    try:
        value = ast.literal_eval(key)
    except (SyntaxError, ValueError) as error:
        raise ImportErrorDetail(f"invalid NAS-Bench-201 architecture key {key!r}") from error
    if not isinstance(value, (tuple, list)) or len(value) != NASBENCH201_EDGE_COUNT:
        raise ImportErrorDetail(
            f"architecture key {key!r} must contain exactly {NASBENCH201_EDGE_COUNT} operation indices"
        )
    operations: list[int] = []
    for index, operation in enumerate(value):
        if isinstance(operation, bool) or not isinstance(operation, int):
            raise ImportErrorDetail(
                f"architecture key {key!r} operation {index} must be an integer"
            )
        if not 0 <= operation < NASBENCH201_OPERATION_COUNT:
            raise ImportErrorDetail(
                f"architecture key {key!r} operation {index} is outside 0..{NASBENCH201_OPERATION_COUNT - 1}"
            )
        operations.append(operation)
    if key != str(tuple(operations)):
        raise ImportErrorDetail(
            f"architecture key {key!r} is not in canonical NASLib tuple-string form {str(tuple(operations))!r}"
        )
    return tuple(operations)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImportErrorDetail(f"{field} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ImportErrorDetail(f"{field} must be finite")
    return result


def _sha256_digest(value: str, field: str) -> str:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value or ""):
        raise ImportErrorDetail(f"{field} must be exactly 64 hexadecimal characters")
    return value.lower()


def build_canonical_import(
    raw_bytes: bytes,
    *,
    dataset: str,
    benchmark_id: str,
    evaluation_split_id: str,
    proxy_calibration_split_id: str,
    proxy_calibration_sha256: str,
    max_unique_queries: int,
    top_k: int,
    proxy_names: tuple[str, ...] = ALLOWED_PROXIES,
    source_commit: str = NASLIB_OBSERVED_COMMIT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not raw_bytes:
        raise ImportErrorDetail("source artifact is empty")
    if not dataset.strip():
        raise ImportErrorDetail("dataset must be non-empty")
    if not benchmark_id.strip():
        raise ImportErrorDetail("benchmark_id must be non-empty")
    if not evaluation_split_id.strip() or not proxy_calibration_split_id.strip():
        raise ImportErrorDetail("evaluation and proxy-calibration split IDs must be non-empty")
    if evaluation_split_id == proxy_calibration_split_id:
        raise ImportErrorDetail("strict no-oracle import requires distinct calibration and evaluation split IDs")
    calibration_digest = _sha256_digest(proxy_calibration_sha256, "proxy_calibration_sha256")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source_commit or ""):
        raise ImportErrorDetail("source_commit must be a 40-character hexadecimal Git commit SHA")
    if not isinstance(max_unique_queries, int) or isinstance(max_unique_queries, bool) or max_unique_queries < 1:
        raise ImportErrorDetail("max_unique_queries must be a positive integer")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ImportErrorDetail("top_k must be a positive integer")

    chosen = tuple(proxy_names)
    if not chosen:
        raise ImportErrorDetail("at least one zero-cost proxy must be selected")
    if len(set(chosen)) != len(chosen):
        raise ImportErrorDetail("proxy selection contains duplicates")
    unknown_proxies = sorted(set(chosen) - set(ALLOWED_PROXIES))
    if unknown_proxies:
        raise ImportErrorDetail(f"unsupported SuiteZero proxy names: {unknown_proxies}")

    artifact_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    root = parse_json_strict(raw_bytes)
    if not isinstance(root, dict):
        raise ImportErrorDetail("SuiteZero JSON root must be an object keyed by dataset")
    if dataset not in root:
        raise ImportErrorDetail(f"dataset {dataset!r} is missing from the SuiteZero artifact")
    task = root[dataset]
    if not isinstance(task, dict) or not task:
        raise ImportErrorDetail(f"SuiteZero dataset entry {dataset!r} must be a non-empty architecture object")
    architecture_count = len(task)
    if max_unique_queries > architecture_count:
        raise ImportErrorDetail("max_unique_queries cannot exceed architecture_count")
    if top_k < min(100, architecture_count) or top_k > architecture_count:
        raise ImportErrorDetail(f"top_k must be in {min(100, architecture_count)}..={architecture_count}")

    architectures: list[dict[str, Any]] = []
    target_labels: list[dict[str, Any]] = []
    normalized_infinite_proxy_scores = 0

    for architecture_id in sorted(task):
        if not isinstance(architecture_id, str):
            raise ImportErrorDetail("architecture IDs must be strings")
        operations = parse_nasbench201_architecture_key(architecture_id)
        record = task[architecture_id]
        if not isinstance(record, dict):
            raise ImportErrorDetail(f"record for {architecture_id} must be an object")
        if "val_accuracy" not in record:
            raise ImportErrorDetail(f"record for {architecture_id} is missing val_accuracy")
        target_value = _finite_number(record["val_accuracy"], f"{architecture_id}.val_accuracy")

        proxy_scores: dict[str, float] = {}
        proxy_times_seconds: dict[str, float] = {}
        for proxy_name in sorted(chosen):
            proxy_record = record.get(proxy_name)
            if not isinstance(proxy_record, dict):
                raise ImportErrorDetail(f"record for {architecture_id} is missing proxy object {proxy_name!r}")
            if "score" not in proxy_record or "time" not in proxy_record:
                raise ImportErrorDetail(
                    f"proxy {proxy_name!r} for {architecture_id} must contain score and time"
                )
            raw_score = proxy_record["score"]
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ImportErrorDetail(f"{architecture_id}.{proxy_name}.score must be a number")
            score = float(raw_score)
            if math.isnan(score):
                raise ImportErrorDetail(f"{architecture_id}.{proxy_name}.score must not be NaN")
            if score == math.inf:
                score = 1e9
                normalized_infinite_proxy_scores += 1
            elif score == -math.inf:
                score = -1e9
                normalized_infinite_proxy_scores += 1
            if not math.isfinite(score):
                raise ImportErrorDetail(f"{architecture_id}.{proxy_name}.score failed normalization")
            runtime = _finite_number(proxy_record["time"], f"{architecture_id}.{proxy_name}.time")
            if runtime < 0:
                raise ImportErrorDetail(f"{architecture_id}.{proxy_name}.time must be non-negative")
            proxy_scores[proxy_name] = score
            proxy_times_seconds[proxy_name] = runtime

        architectures.append(
            {
                "architecture_id": architecture_id,
                "proxy_scores": proxy_scores,
                "proxy_times_seconds": proxy_times_seconds,
                "topology_encoding": list(operations),
            }
        )
        target_labels.append({"architecture_id": architecture_id, "value": target_value})

    source_description = (
        f"NAS-Bench-Suite-Zero {SUITEZERO_FILENAME}; dataset={dataset}; "
        f"{NASLIB_REPO}@{source_commit}; NASLib {NASLIB_BRANCH} loader; "
        f"Google Drive artifact ID={NASBENCH201_ZC_DRIVE_FILE_ID}; "
        "NAS-Bench-201 topology encoding=6 operation indices in range 0..4; "
        f"proxy names={','.join(sorted(chosen))}; proxy runtime unit=seconds"
    )
    canonical = {
        "schema": CANONICAL_SCHEMA,
        "version": CANONICAL_VERSION,
        "manifest": {
            "schema": BENCHMARK_SCHEMA,
            "version": CANONICAL_VERSION,
            "benchmark_id": benchmark_id,
            "metric_id": "val_accuracy",
            "metric_direction": "maximize",
            "split_id": evaluation_split_id,
            "proxy_calibration_split_id": proxy_calibration_split_id,
            "proxy_calibration_sha256": calibration_digest,
            "source_description": source_description,
            "artifact_sha256": artifact_sha256,
            "max_unique_queries": max_unique_queries,
            "top_k": top_k,
        },
        "architectures": architectures,
        "target_labels": target_labels,
    }
    report = {
        "dataset": dataset,
        "architecture_count": architecture_count,
        "proxy_names": list(sorted(chosen)),
        "raw_artifact_bytes": len(raw_bytes),
        "raw_artifact_sha256": artifact_sha256,
        "upstream_google_drive_file_id": NASBENCH201_ZC_DRIVE_FILE_ID,
        "naslib_source_commit": source_commit,
        "normalized_infinite_proxy_scores": normalized_infinite_proxy_scores,
        "target_metric": "val_accuracy",
        "target_metric_direction": "maximize",
        "proxy_runtime_unit": "seconds",
        "evaluation_split_id": evaluation_split_id,
        "proxy_calibration_split_id": proxy_calibration_split_id,
        "max_unique_queries": max_unique_queries,
        "top_k": top_k,
    }
    return canonical, report


def _parse_proxy_names(value: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("supply one or more comma-separated proxy names")
    unknown = sorted(set(names) - set(ALLOWED_PROXIES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unsupported proxy names: {unknown}")
    if len(names) != len(set(names)):
        raise argparse.ArgumentTypeError("proxy names must be unique")
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="raw NASLib zc_nasbench201.json")
    parser.add_argument("--output", required=True, type=Path, help="canonical Symthaea JSON output")
    parser.add_argument("--dataset", required=True, help="SuiteZero task key, e.g. cifar10 or cifar100")
    parser.add_argument("--benchmark-id", required=True, help="immutable benchmark/task identifier")
    parser.add_argument("--evaluation-split-id", required=True, help="challenge evaluation split ID")
    parser.add_argument("--proxy-calibration-split-id", required=True, help="disjoint development/calibration split ID")
    parser.add_argument("--proxy-calibration-sha256", required=True, help="digest of frozen calibration data + proxy-selection decision")
    parser.add_argument("--query-budget", required=True, type=int, help="maximum distinct target-label queries")
    parser.add_argument("--top-k", type=int, default=100, help="oracle top-K stored for evaluator; default 100")
    parser.add_argument("--proxy-names", type=_parse_proxy_names, default=ALLOWED_PROXIES, help="comma-separated subset of the 13 SuiteZero proxies")
    parser.add_argument("--source-commit", default=NASLIB_OBSERVED_COMMIT, help="NASLib source commit used to define the import contract")
    parser.add_argument("--force", action="store_true", help="replace an existing output file")
    args = parser.parse_args(argv)

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        parser.error("input and output paths must differ")
    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")
    if output_path.exists() and not args.force:
        parser.error(f"output already exists (pass --force to replace): {output_path}")

    try:
        raw = input_path.read_bytes()
        canonical, report = build_canonical_import(
            raw,
            dataset=args.dataset,
            benchmark_id=args.benchmark_id,
            evaluation_split_id=args.evaluation_split_id,
            proxy_calibration_split_id=args.proxy_calibration_split_id,
            proxy_calibration_sha256=args.proxy_calibration_sha256,
            max_unique_queries=args.query_budget,
            top_k=args.top_k,
            proxy_names=tuple(args.proxy_names),
            source_commit=args.source_commit,
        )
        encoded = (
            json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        output_sha256 = hashlib.sha256(encoded).hexdigest()
        report["canonical_import_bytes"] = len(encoded)
        report["canonical_import_sha256"] = output_sha256

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # os.replace is atomic on the same filesystem and safely replaces an existing
            # destination when --force was explicitly requested.
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
        print(f"import failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
