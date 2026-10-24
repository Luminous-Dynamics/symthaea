#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Provenance-checked WER/CER scoring for a local Common Voice ASR run.

This scores hypotheses; it does not run inference. It hashes the supplied model
snapshot, pins the prediction file to an intake report and exact test.tsv, rejects
missing/duplicate/unknown hypotheses, and writes aggregate-only reports. It never
writes transcripts, sample paths, clip identifiers, or speaker tokens to reports.

A smoke run is exploratory. Even a valid held-out evaluation is not itself a
production-qualification or language-support claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
NORMALIZATION_ID = "nfkc-casefold-punctuation-to-space-v1"
PREDICTION_KEYS = {
    "schema_version", "locale", "dataset_id", "dataset_version",
    "archive_sha256", "test_metadata_sha256", "model_id", "model_revision",
    "model_license", "source_model_snapshot_sha256", "outcomes_sha256", "runtime", "predictions",
}
PILOT_OUTCOME_KEYS = {
    "id", "language", "detected_language", "language_confidence", "transcript",
    "word_error_rate", "character_error_rate", "entity_preservation",
    "translation_en", "translation_chrf", "response_language_correct",
    "latency_ms", "preflight_latency_ms", "transcription_latency_ms",
    "translation_latency_ms", "error",
}
WHISPER_CPP_RUNTIME_KEYS = {
    "kind", "runtime_commit", "converter_commit", "runtime_binary_sha256",
    "artifact_path", "artifact_sha256", "quantization",
}
TRANSFORMERS_RUNTIME_KEYS = {
    "kind", "python_executable_sha256", "worker_script_sha256",
    "model_snapshot_manifest_sha256", "startup_metadata_log_sha256",
    "python_version", "torch_version", "transformers_version", "numpy_version",
    "device", "dtype", "offline",
}
RUNTIME_METADATA_PREFIX = "SYMTHEA_TRANSFORMERS_RUNTIME "
RUNTIME_METADATA_KEYS = {
    "python_version", "torch_version", "transformers_version", "numpy_version",
    "device", "dtype", "offline", "model_snapshot_manifest_sha256",
    "model_snapshot_file_count", "model_snapshot_bytes",
    "python_executable_sha256", "worker_script_sha256",
}
# Backwards-compatible name for code that imports the v1 whisper.cpp schema.
RUNTIME_KEYS = WHISPER_CPP_RUNTIME_KEYS


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


def read_transformers_runtime_metadata(path: Path) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(),
          "Transformers runtime metadata log must be a regular non-symlink file")
    before = sha256_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [line[len(RUNTIME_METADATA_PREFIX):] for line in lines
               if line.startswith(RUNTIME_METADATA_PREFIX)]
    _need(len(matches) == 1,
          "Transformers runtime metadata log must contain exactly one worker startup record")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value}")

    try:
        metadata = json.loads(
            matches[0], object_pairs_hook=_unique_object, parse_constant=reject_constant
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Transformers startup metadata is malformed: {error}") from error
    _need(isinstance(metadata, dict) and set(metadata) == RUNTIME_METADATA_KEYS,
          "Transformers startup metadata fields mismatch")
    for field in ("python_version", "torch_version", "transformers_version", "numpy_version"):
        _need(isinstance(metadata[field], str) and metadata[field].strip()
              and "\n" not in metadata[field] and "\r" not in metadata[field],
              f"Transformers startup metadata {field} must be explicit")
    _need(metadata["device"] in {"cpu", "cuda"}, "Transformers startup device must be cpu or cuda")
    _need(metadata["dtype"] in {"float32", "float16", "bfloat16"},
          "Transformers startup dtype is unsupported")
    _need(metadata["offline"] is True, "Transformers startup metadata must declare offline=true")
    for field in ("model_snapshot_manifest_sha256", "python_executable_sha256", "worker_script_sha256"):
        _need(isinstance(metadata[field], str) and SHA256_RE.fullmatch(metadata[field]) is not None,
              f"Transformers startup metadata {field} must be a 64-hex digest")
    for field in ("model_snapshot_file_count", "model_snapshot_bytes"):
        _need(isinstance(metadata[field], int) and not isinstance(metadata[field], bool)
              and metadata[field] >= 0, f"Transformers startup metadata {field} must be non-negative")
    _need(metadata["model_snapshot_file_count"] > 0, "Transformers startup snapshot has no files")
    _need(before == sha256_file(path), "Transformers runtime metadata log changed while being read")
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: Any, where: str) -> Path:
    _need(isinstance(value, str) and value.strip(), f"{where} must be a non-empty relative path")
    _need("\x00" not in value and "\\" not in value, f"{where} contains invalid path characters")
    parsed = PurePosixPath(value)
    _need(not parsed.is_absolute() and not any(part in {"", ".", ".."} for part in parsed.parts),
          f"{where} must not be absolute or contain traversal components")
    _need(bool(parsed.parts) and not parsed.parts[0].endswith(":"), f"{where} must be repository-relative")
    return Path(*parsed.parts)


def _read_pilot_outcomes(
    path: Path,
    *,
    expected_paths: set[str],
    locale: str,
) -> dict[str, dict[str, Any]]:
    """Read exact outcomes, bind transcripts to predictions, and validate phase timings."""
    _need(path.is_file() and not path.is_symlink(), "outcomes must be a regular non-symlink file")
    rows: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        object_pairs_hook=_unique_object,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-standard JSON numeric constant {value}")
                        ),
                    )
                except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"malformed pilot outcome JSON on line {line_number}: {exc}") from exc
                _need(isinstance(row, dict) and set(row) == PILOT_OUTCOME_KEYS,
                      f"pilot outcome line {line_number} has unexpected fields")
                relative_id = safe_relative_path(row["id"], f"pilot outcome line {line_number} id").as_posix()
                _need(relative_id in expected_paths,
                      f"pilot outcome line {line_number} references an unknown clip")
                _need(relative_id not in rows, f"duplicate pilot outcome for clip {relative_id}")
                _need(row["language"] == locale,
                      f"pilot outcome line {line_number} language differs from locale")
                _need(row["error"] is None,
                      f"pilot outcome for {relative_id} records an inference error")
                _need(isinstance(row["transcript"], str),
                      f"pilot outcome for {relative_id} transcript must be text")
                for field in ("latency_ms", "preflight_latency_ms", "transcription_latency_ms", "translation_latency_ms"):
                    metric = row[field]
                    if metric is None and field != "latency_ms":
                        continue
                    _need(isinstance(metric, (int, float)) and not isinstance(metric, bool)
                          and math.isfinite(metric) and metric >= 0,
                          f"pilot outcome line {line_number} {field} must be finite and non-negative"
                          if field == "latency_ms" else
                          f"pilot outcome line {line_number} {field} must be null or finite and non-negative")
                rows[relative_id] = row
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read pilot outcomes: {exc}") from exc
    missing = expected_paths - set(rows)
    _need(not missing, f"missing {len(missing)} pilot outcomes; phase timings are incomplete")
    return rows


def _latency_summary(outcomes: dict[str, dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted(
        float(row[field]) for row in outcomes.values() if row[field] is not None
    )
    if not values:
        return {
            "sample_count": 0, "mean_ms": None, "p50_ms": None,
            "p95_ms": None, "max_ms": None,
        }

    def nearest_rank(percentile: float) -> float:
        index = max(0, min(len(values) - 1, math.ceil(percentile * len(values)) - 1))
        return values[index]

    return {
        "sample_count": len(values),
        "mean_ms": sum(values) / len(values),
        "p50_ms": nearest_rank(0.50),
        "p95_ms": nearest_rank(0.95),
        "max_ms": values[-1],
    }


def normalize_text(value: str) -> str:
    """Case-fold Unicode, map punctuation to spaces, preserve diacritics, collapse whitespace."""
    value = unicodedata.normalize("NFKC", value).casefold()
    mapped = "".join(" " if unicodedata.category(char).startswith("P") else char for char in value)
    return " ".join(mapped.split())


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance in Unicode code points using O(min(n,m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def _hash_tree(root: Path) -> tuple[str, int, int]:
    """Hash an exported model snapshot; symlinks/special files fail closed."""
    _need(root.is_dir() and not root.is_symlink(), "model_root must be a real exported directory, not a symlink")
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for dirname in list(directories):
            child = current_path / dirname
            _need(not child.is_symlink(), "model snapshot contains a symlinked directory")
        for filename in sorted(files):
            file_path = current_path / filename
            _need(not file_path.is_symlink(), "model snapshot contains a symlinked file")
            _need(file_path.is_file(), "model snapshot contains a non-regular file")
            relative = file_path.relative_to(root).as_posix()
            _need(not relative.startswith("../"), "model snapshot path escaped model_root")
            size = file_path.stat().st_size
            entries.append({"path": relative, "size_bytes": size, "sha256": sha256_file(file_path)})
            total_bytes += size
        directories.sort()
    _need(entries, "model snapshot contains no files")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), len(entries), total_bytes


def _load_test_rows(test_tsv: Path) -> tuple[list[dict[str, str]], str]:
    _need(test_tsv.is_file() and not test_tsv.is_symlink(), "test.tsv must be a regular non-symlink file")
    before = sha256_file(test_tsv)
    rows: list[dict[str, str]] = []
    try:
        with test_tsv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = reader.fieldnames or []
            _need(len(fields) == len(set(fields)), "test.tsv has duplicate column names")
            _need("path" in fields and "sentence" in fields and "client_id" in fields,
                  "test.tsv requires path, sentence, and client_id columns")
            seen: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                _need(None not in row, f"test.tsv line {line_number} contains extra fields")
                relative = safe_relative_path(row.get("path"), f"test.tsv line {line_number} path").as_posix()
                _need(relative not in seen, f"test.tsv has duplicate clip path {relative}")
                seen.add(relative)
                reference = row.get("sentence")
                _need(isinstance(reference, str) and normalize_text(reference),
                      f"test.tsv line {line_number} has an empty/normalization-empty reference")
                client_id = row.get("client_id")
                _need(isinstance(client_id, str) and client_id.strip(),
                      f"test.tsv line {line_number} requires a pseudonymous client_id")
                rows.append({"path": relative, "reference": reference, "client_id": client_id})
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot parse test.tsv: {exc}") from exc
    after = sha256_file(test_tsv)
    _need(before == after, "test.tsv changed while it was being scored")
    _need(rows, "test.tsv contains no rows")
    return rows, after


def _load_intake(
    intake_report: Any,
    *,
    dataset_root: Path,
    archive: Path,
    locale: str,
    expected_mode: str,
) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    _need(isinstance(intake_report, dict) and isinstance(intake_report.get("schema_version"), int)
          and not isinstance(intake_report.get("schema_version"), bool)
          and intake_report.get("schema_version") == 1,
          "intake report must have integer schema_version=1")
    _need(intake_report.get("audit_kind") == "common_voice_local_dataset_intake",
          "intake report was not produced by the Common Voice intake auditor")
    _need(intake_report.get("language_tag") == locale, "locale mismatch between CLI and intake report")
    wanted_intake_mode = "ingestion-smoke" if expected_mode == "smoke" else "heldout-evaluation"
    _need(intake_report.get("mode") == wanted_intake_mode,
          f"{expected_mode} scoring requires intake mode {wanted_intake_mode}")
    _need(dataset_root.is_dir() and not dataset_root.is_symlink(), "dataset_root must be a real directory")
    _need(archive.is_file() and not archive.is_symlink(), "archive must be a regular non-symlink file")
    dataset = intake_report.get("dataset")
    _need(isinstance(dataset, dict), "intake report lacks dataset provenance")
    current_archive_sha = sha256_file(archive)
    _need(current_archive_sha == dataset.get("archive_sha256"), "archive SHA-256 differs from intake report")
    splits = intake_report.get("split_audits")
    _need(isinstance(splits, dict) and isinstance(splits.get("test"), dict),
          "intake report lacks test split audit")
    test_audit = splits["test"]
    _need(isinstance(test_audit.get("audio_files_checked"), int) and test_audit["audio_files_checked"] > 0,
          "ASR scoring requires an intake report that checked test audio files")
    test_tsv = dataset_root / "test.tsv"
    rows, test_sha = _load_test_rows(test_tsv)
    _need(test_sha == test_audit.get("metadata_sha256"), "test.tsv SHA-256 differs from intake report")
    _need(len(rows) == test_audit.get("rows"), "test.tsv row count differs from intake report")
    if expected_mode == "heldout-evaluation":
        integrity = intake_report.get("integrity")
        _need(intake_report.get("mode") == "heldout-evaluation"
              and isinstance(integrity, dict)
              and integrity.get("speaker_disjoint") is True
              and integrity.get("speaker_tokens_missing_in_any_split") is False,
              "heldout scoring requires a speaker-disjoint intake audit")
        unique_speakers = len({row["client_id"] for row in rows})
        _need(len(rows) >= 100, "heldout scoring requires at least 100 test clips")
        _need(unique_speakers >= 10, "heldout scoring requires at least 10 test speakers")
    clips_root = dataset_root / "clips"
    _need(clips_root.is_dir() and not clips_root.is_symlink(), "clips directory must be present and not a symlink")
    for row in rows:
        audio = clips_root / safe_relative_path(row["path"], "audio path")
        _need(audio.is_file() and not audio.is_symlink(), f"missing or symlinked test audio file: {row['path']}")
        _need(audio.stat().st_size > 0, f"empty test audio file: {row['path']}")
    return dataset, rows, test_sha


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def _rate(errors: int, references: int) -> float:
    _need(references > 0, "normalized reference denominator must be positive")
    return errors / references


def _bootstrap_ci(
    speaker_clusters: list[list[dict[str, int]]],
    metric: str,
    iterations: int = 1000,
    seed: int = 20261013,
) -> dict[str, Any]:
    """Percentile bootstrap at speaker-cluster level, never treating same-speaker clips as independent."""
    _need(iterations > 0, "bootstrap iterations must be positive")
    cluster_count = len(speaker_clusters)
    if cluster_count < 2:
        return {
            "low_95": None,
            "high_95": None,
            "iterations": iterations,
            "seed": seed,
            "cluster_count": cluster_count,
            "method": "speaker-cluster percentile bootstrap",
            "interval_status": "not_estimable_fewer_than_2_speakers",
        }
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        selected_clusters = [speaker_clusters[rng.randrange(cluster_count)] for _ in range(cluster_count)]
        selected = [item for cluster in selected_clusters for item in cluster]
        denominator = sum(item[f"ref_{metric}"] for item in selected)
        numerator = sum(item[f"err_{metric}"] for item in selected)
        values.append(numerator / denominator if denominator else 0.0)
    values.sort()
    return {
        "low_95": _percentile(values, 0.025),
        "high_95": _percentile(values, 0.975),
        "iterations": iterations,
        "seed": seed,
        "cluster_count": cluster_count,
        "method": "deterministic percentile bootstrap resampling pseudonymous speaker clusters with replacement",
        "interval_status": "estimated",
    }


def score_predictions(
    *,
    prediction_doc: Any,
    intake_report: Any,
    dataset_root: Path,
    archive: Path,
    model_root: Path,
    source_model_root: Path,
    predictions_path: Path,
    outcomes_path: Path,
    model_id: str,
    model_revision: str,
    model_license: str,
    locale: str,
    intake_report_path: Path,
    runtime_binary_path: Path,
    runtime_worker_path: Path | None = None,
    runtime_metadata_log_path: Path | None = None,
    mode: str = "smoke",
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    _need(mode in {"smoke", "heldout-evaluation"}, "mode must be smoke or heldout-evaluation")
    _need(isinstance(locale, str) and LANGUAGE_TAG_RE.fullmatch(locale) is not None, "locale must be a BCP 47-shaped tag")
    _need(isinstance(model_id, str) and model_id.strip(), "model_id must be explicit")
    _need(isinstance(model_revision, str) and SHA1_RE.fullmatch(model_revision) is not None,
          "model_revision must be the exact 40-hex repository commit")
    _need(isinstance(model_license, str) and model_license.strip(), "model_license must be explicitly declared")
    _need(isinstance(predictions_path, Path) and predictions_path.is_file() and not predictions_path.is_symlink(),
          "predictions_path must be a regular non-symlink file")
    _need(isinstance(outcomes_path, Path) and outcomes_path.is_file() and not outcomes_path.is_symlink(),
          "outcomes_path must be a regular non-symlink file")
    _need(isinstance(intake_report_path, Path) and intake_report_path.is_file() and not intake_report_path.is_symlink(),
          "intake_report_path must be a regular non-symlink file")
    _need(isinstance(runtime_binary_path, Path) and runtime_binary_path.is_file() and not runtime_binary_path.is_symlink(),
          "runtime_binary_path must be a regular non-symlink file")
    _need(isinstance(prediction_doc, dict), "prediction document must be an object")
    _need(set(prediction_doc) == PREDICTION_KEYS,
          f"prediction document keys mismatch; missing={sorted(PREDICTION_KEYS-set(prediction_doc))}, extra={sorted(set(prediction_doc)-PREDICTION_KEYS)}")
    _need(isinstance(prediction_doc["schema_version"], int) and not isinstance(prediction_doc["schema_version"], bool)
          and prediction_doc["schema_version"] == 1, "prediction schema_version must equal integer 1")
    _need(prediction_doc["locale"] == locale, "prediction locale mismatch")
    _need(prediction_doc["model_id"] == model_id, "prediction model_id mismatch")
    _need(prediction_doc["model_revision"] == model_revision, "prediction model_revision mismatch")
    _need(prediction_doc["model_license"] == model_license, "prediction model_license mismatch")
    source_digest = prediction_doc["source_model_snapshot_sha256"]
    _need(isinstance(source_digest, str) and SHA256_RE.fullmatch(source_digest) is not None,
          "source_model_snapshot_sha256 must be a 64-hex SHA-256 digest")
    _need(source_model_root.is_dir() and not source_model_root.is_symlink(),
          "source_model_root must be a real directory, not a symlink")
    converted_root_real = model_root.resolve(strict=True)
    source_root_real = source_model_root.resolve(strict=True)
    _need(converted_root_real == source_root_real or (
          converted_root_real not in source_root_real.parents
          and source_root_real not in converted_root_real.parents),
          "source_model_root and runtime model_root must be the same snapshot or separate non-nested directories")
    source_tree_sha, source_file_count, source_bytes = _hash_tree(source_model_root)
    _need(source_tree_sha == source_digest, "source model snapshot SHA-256 differs from prediction provenance")
    outcomes_sha = prediction_doc["outcomes_sha256"]
    _need(isinstance(outcomes_sha, str) and SHA256_RE.fullmatch(outcomes_sha) is not None,
          "outcomes_sha256 must be a 64-hex SHA-256 digest")
    actual_outcomes_sha = sha256_file(outcomes_path)
    _need(actual_outcomes_sha == outcomes_sha,
          "raw pilot outcomes SHA-256 differs from prediction provenance")
    runtime = prediction_doc["runtime"]
    _need(isinstance(runtime, dict) and isinstance(runtime.get("kind"), str),
          "runtime must be an object with a kind")
    actual_runtime_sha = sha256_file(runtime_binary_path)
    artifact_relative: PurePosixPath | None = None
    if runtime["kind"] == "whisper.cpp":
        _need(set(runtime) == WHISPER_CPP_RUNTIME_KEYS,
              f"whisper.cpp runtime keys mismatch; missing={sorted(WHISPER_CPP_RUNTIME_KEYS-set(runtime))}, extra={sorted(set(runtime)-WHISPER_CPP_RUNTIME_KEYS)}")
        for field in ("runtime_commit", "converter_commit"):
            _need(isinstance(runtime[field], str) and SHA1_RE.fullmatch(runtime[field]) is not None,
                  f"runtime.{field} must be a 40-hex commit SHA")
        _need(isinstance(runtime["quantization"], str) and re.fullmatch(r"[A-Za-z0-9._+-]{1,32}", runtime["quantization"]) is not None,
              "runtime.quantization must be an explicit compact identifier")
        _need(isinstance(runtime["runtime_binary_sha256"], str) and SHA256_RE.fullmatch(runtime["runtime_binary_sha256"]) is not None,
              "runtime.runtime_binary_sha256 must be a 64-hex digest")
        _need(actual_runtime_sha == runtime["runtime_binary_sha256"],
              "runtime binary SHA-256 differs from prediction provenance")
        artifact_relative = safe_relative_path(runtime["artifact_path"], "runtime artifact_path")
        _need(isinstance(runtime["artifact_sha256"], str) and SHA256_RE.fullmatch(runtime["artifact_sha256"]) is not None,
              "runtime.artifact_sha256 must be a 64-hex digest")
        _need(model_root.is_dir() and not model_root.is_symlink(), "model_root must be a real directory, not a symlink")
        artifact_path = model_root / artifact_relative
        current = model_root
        for part in artifact_relative.parts:
            current = current / part
            _need(not current.is_symlink(), "runtime model artifact path contains a symlink")
        _need(artifact_path.is_file(), "runtime model artifact_path does not name a regular file")
        _need(artifact_path.resolve(strict=True).is_relative_to(model_root.resolve(strict=True)),
              "runtime model artifact escapes model_root")
        _need(sha256_file(artifact_path) == runtime["artifact_sha256"],
              "runtime model artifact SHA-256 differs from prediction provenance")
        _need(converted_root_real != source_root_real,
              "whisper.cpp converted export and source snapshot must be separate directories")
    elif runtime["kind"] == "transformers":
        _need(set(runtime) == TRANSFORMERS_RUNTIME_KEYS,
              f"Transformers runtime keys mismatch; missing={sorted(TRANSFORMERS_RUNTIME_KEYS-set(runtime))}, extra={sorted(set(runtime)-TRANSFORMERS_RUNTIME_KEYS)}")
        _need(converted_root_real == source_root_real,
              "Transformers runtime model-root must be the exact source-model-root snapshot")
        for field in ("python_executable_sha256", "worker_script_sha256"):
            _need(isinstance(runtime[field], str) and SHA256_RE.fullmatch(runtime[field]) is not None,
                  f"runtime.{field} must be a 64-hex digest")
        _need(actual_runtime_sha == runtime["python_executable_sha256"],
              "Python executable SHA-256 differs from prediction provenance")
        _need(runtime["model_snapshot_manifest_sha256"] == source_tree_sha,
              "model snapshot manifest SHA-256 differs from value observed by the native worker")
        _need(isinstance(runtime_worker_path, Path) and runtime_worker_path.is_file()
              and not runtime_worker_path.is_symlink(),
              "Transformers runtime requires a regular --runtime-worker script")
        _need(sha256_file(runtime_worker_path) == runtime["worker_script_sha256"],
              "Transformers worker script SHA-256 differs from prediction provenance")
        for field in ("python_version", "torch_version", "transformers_version", "numpy_version"):
            _need(isinstance(runtime[field], str) and runtime[field].strip()
                  and "\n" not in runtime[field] and "\r" not in runtime[field],
                  f"runtime.{field} must be explicit")
        _need(runtime["device"] in {"cpu", "cuda"}, "runtime.device must be cpu or cuda")
        _need(runtime["dtype"] in {"float32", "float16", "bfloat16"}, "runtime.dtype is unsupported")
        _need(runtime["offline"] is True, "Transformers runtime must declare offline=true")
        _need(isinstance(runtime["startup_metadata_log_sha256"], str)
              and SHA256_RE.fullmatch(runtime["startup_metadata_log_sha256"]) is not None,
              "runtime.startup_metadata_log_sha256 must be a 64-hex digest")
        _need(isinstance(runtime_metadata_log_path, Path)
              and runtime_metadata_log_path.is_file()
              and not runtime_metadata_log_path.is_symlink(),
              "Transformers runtime requires a regular --runtime-metadata-log file")
        _need(sha256_file(runtime_metadata_log_path) == runtime["startup_metadata_log_sha256"],
              "Transformers startup metadata log SHA-256 differs from prediction provenance")
        observed = read_transformers_runtime_metadata(runtime_metadata_log_path)
        for field in ("python_version", "torch_version", "transformers_version", "numpy_version",
                      "device", "dtype", "offline", "python_executable_sha256", "worker_script_sha256",
                      "model_snapshot_manifest_sha256"):
            runtime_field = field
            _need(runtime[runtime_field] == observed[field],
                  f"runtime.{runtime_field} differs from the startup metadata log")
        _need(observed["python_executable_sha256"] == actual_runtime_sha,
              "startup metadata Python executable digest differs from the actual runtime binary")
        _need(observed["worker_script_sha256"] == sha256_file(runtime_worker_path),
              "startup metadata worker script digest differs from the actual worker script")
        _need(observed["model_snapshot_file_count"] == source_file_count
              and observed["model_snapshot_bytes"] == source_bytes,
              "startup metadata model snapshot inventory differs from the supplied snapshot")
    else:
        raise ValueError(f"unsupported runtime.kind: {runtime['kind']}")

    dataset, test_rows, test_sha = _load_intake(
        intake_report, dataset_root=dataset_root, archive=archive, locale=locale, expected_mode=mode
    )
    for field in ("id", "version", "archive_sha256"):
        prediction_key = {"id": "dataset_id", "version": "dataset_version", "archive_sha256": "archive_sha256"}[field]
        _need(prediction_doc[prediction_key] == dataset.get(field),
              f"prediction document dataset {field} mismatch")
    _need(prediction_doc["test_metadata_sha256"] == test_sha,
          "prediction document test metadata hash mismatch")
    hypotheses = prediction_doc["predictions"]
    _need(isinstance(hypotheses, list), "predictions must be an array")
    expected = {row["path"]: row for row in test_rows}
    seen: dict[str, str] = {}
    raw_seen: dict[str, str] = {}
    for index, row in enumerate(hypotheses):
        _need(isinstance(row, dict) and set(row) == {"path", "text"},
              f"prediction {index} must contain only path and text")
        path = safe_relative_path(row["path"], f"prediction {index} path").as_posix()
        _need(path in expected, f"prediction {index} references an unknown clip")
        _need(path not in seen, f"duplicate prediction for clip path {path}")
        _need(isinstance(row["text"], str), f"prediction {index} text must be a string")
        raw_seen[path] = row["text"]
        seen[path] = normalize_text(row["text"])
    missing = set(expected) - set(seen)
    _need(not missing, f"missing {len(missing)} ASR predictions; scoring is incomplete")
    outcome_rows = _read_pilot_outcomes(
        outcomes_path, expected_paths=set(expected), locale=locale
    )
    for clip_path, transcript in raw_seen.items():
        _need(outcome_rows[clip_path]["transcript"] == transcript,
              f"prediction text for {clip_path} differs from raw pilot outcome transcript")
    latency_metrics = {
        "end_to_end_ms": _latency_summary(outcome_rows, "latency_ms"),
        "preflight_request_ms": _latency_summary(outcome_rows, "preflight_latency_ms"),
        "transcription_ms": _latency_summary(outcome_rows, "transcription_latency_ms"),
        "translation_ms": _latency_summary(outcome_rows, "translation_latency_ms"),
        "first_sample_preflight_ms": next(iter(outcome_rows.values()))["preflight_latency_ms"],
        "interpretation": (
            "end_to_end_ms includes all requested operations; preflight_request_ms for a monolingual "
            "worker may include first-use model startup, while transcription_ms measures the ASR operation; "
            "runtime-specific loading semantics differ, and whisper.cpp may load its model per clip"
        ),
    }

    model_tree_sha, model_files, model_bytes = _hash_tree(model_root)
    if runtime["kind"] == "transformers":
        _need(model_tree_sha == runtime["model_snapshot_manifest_sha256"],
              "native model snapshot changed during scoring")
    per_clip: list[dict[str, int]] = []
    word_speaker_clusters: dict[str, list[dict[str, int]]] = {}
    char_speaker_clusters: dict[str, list[dict[str, int]]] = {}
    bucket_rows: dict[str, list[dict[str, int]]] = {
        "short_1_to_5_words": [], "medium_6_to_10_words": [], "long_11_plus_words": []
    }
    for path, row in expected.items():
        ref = normalize_text(row["reference"])
        hyp = seen[path]
        ref_words = ref.split()
        hyp_words = hyp.split()
        ref_chars = len(ref)
        _need(ref_words and ref_chars > 0, "test reference normalized to empty")
        # Word-level Levenshtein operates on tokens, not joined characters.
        word_errors = _token_edit_distance(ref_words, hyp_words)
        char_errors = edit_distance(ref, hyp)
        item = {
            "ref_words": len(ref_words), "err_words": word_errors,
            "ref_chars": ref_chars, "err_chars": char_errors,
        }
        per_clip.append(item)
        # Pseudonymous client_id is used only to keep same-speaker clips in
        # one bootstrap cluster. No token is copied into the score report.
        speaker_token = row["client_id"]
        word_speaker_clusters.setdefault(speaker_token, []).append({
            "ref_words": len(ref_words), "err_words": word_errors,
        })
        char_speaker_clusters.setdefault(speaker_token, []).append({
            "ref_chars": ref_chars, "err_chars": char_errors,
        })
        if len(ref_words) <= 5:
            bucket_rows["short_1_to_5_words"].append(item)
        elif len(ref_words) <= 10:
            bucket_rows["medium_6_to_10_words"].append(item)
        else:
            bucket_rows["long_11_plus_words"].append(item)

    ref_words_total = sum(row["ref_words"] for row in per_clip)
    err_words_total = sum(row["err_words"] for row in per_clip)
    ref_chars_total = sum(row["ref_chars"] for row in per_clip)
    err_chars_total = sum(row["err_chars"] for row in per_clip)
    def aggregate(items: list[dict[str, int]]) -> dict[str, Any]:
        ref_words = sum(item.get("ref_words", item.get("ref_word", 0)) for item in items)
        word_errors = sum(item.get("err_words", item.get("err_word", 0)) for item in items)
        ref_chars = sum(item.get("ref_chars", item.get("ref_char", 0)) for item in items)
        char_errors = sum(item.get("err_chars", item.get("err_char", 0)) for item in items)
        return {
            "sample_count": len(items),
            "reference_word_count": ref_words,
            "word_edit_count": word_errors,
            "wer": _rate(word_errors, ref_words),
            "wer_percent": 100.0 * word_errors / ref_words,
            "reference_character_count": ref_chars,
            "character_edit_count": char_errors,
            "cer": _rate(char_errors, ref_chars),
            "cer_percent": 100.0 * char_errors / ref_chars,
        }

    # Cluster bootstrap treats same-speaker clips as a unit rather than assuming
    # that every utterance is statistically independent.
    word_ci = _bootstrap_ci(
        [[{"ref_word": item["ref_words"], "err_word": item["err_words"]} for item in cluster]
         for cluster in word_speaker_clusters.values()],
        "word",
        bootstrap_iterations,
    )
    char_ci = _bootstrap_ci(
        [[{"ref_char": item["ref_chars"], "err_char": item["err_chars"]} for item in cluster]
         for cluster in char_speaker_clusters.values()],
        "char",
        bootstrap_iterations,
    )

    # Macro averages give every speaker equal weight, complementing micro WER/CER
    # where speakers with more clips naturally contribute more reference units.
    speaker_word_rates = []
    speaker_char_rates = []
    for speaker_token in word_speaker_clusters:
        word_items = word_speaker_clusters[speaker_token]
        char_items = char_speaker_clusters[speaker_token]
        word_denominator = sum(item["ref_words"] for item in word_items)
        char_denominator = sum(item["ref_chars"] for item in char_items)
        if word_denominator:
            speaker_word_rates.append(sum(item["err_words"] for item in word_items) / word_denominator)
        if char_denominator:
            speaker_char_rates.append(sum(item["err_chars"] for item in char_items) / char_denominator)

    archive_sha = dataset["archive_sha256"]
    test_meta_sha = test_sha
    prediction_sha = sha256_file(predictions_path)
    model_record = {
        "id": model_id, "revision_claimed": model_revision, "license_declared": model_license,
        "source_snapshot_manifest_sha256": source_tree_sha,
        "source_snapshot_file_count": source_file_count,
        "source_snapshot_bytes": source_bytes,
    }
    if runtime["kind"] == "whisper.cpp":
        model_record.update({
            "converted_export_manifest_sha256": model_tree_sha,
            "converted_export_file_count": model_files,
            "converted_export_bytes": model_bytes,
        })
        runtime_record = {
            "kind": runtime["kind"],
            "runtime_commit": runtime["runtime_commit"],
            "converter_commit": runtime["converter_commit"],
            "runtime_binary_sha256": actual_runtime_sha,
            "artifact_path": artifact_relative.as_posix(),
            "artifact_sha256": runtime["artifact_sha256"],
            "quantization": runtime["quantization"],
        }
    else:
        model_record.update({
            "runtime_model_snapshot_manifest_sha256": model_tree_sha,
            "runtime_model_snapshot_file_count": model_files,
            "runtime_model_snapshot_bytes": model_bytes,
        })
        runtime_record = {
            "kind": runtime["kind"],
            "python_executable_sha256": actual_runtime_sha,
            "worker_script_sha256": runtime["worker_script_sha256"],
            "model_snapshot_manifest_sha256": runtime["model_snapshot_manifest_sha256"],
            "startup_metadata_log_sha256": runtime["startup_metadata_log_sha256"],
            "python_version": runtime["python_version"],
            "torch_version": runtime["torch_version"],
            "transformers_version": runtime["transformers_version"],
            "numpy_version": runtime["numpy_version"],
            "device": runtime["device"],
            "dtype": runtime["dtype"],
            "offline": runtime["offline"],
        }
    return {
        "schema_version": 1,
        "report_kind": "voice_asr_score",
        "mode": mode,
        "interpretation": (
            "exploratory smoke metric only; not evidence of production readiness or language support"
            if mode == "smoke"
            else "held-out ASR metrics for this exact supplied dataset split; model-training corpus overlap is not assessed and this is not by itself a production-support claim"
        ),
        "training_data_overlap_status": "not_assessed",
        "locale": locale,
        "normalization": NORMALIZATION_ID,
        "scorer": {
            "script_sha256": sha256_file(Path(__file__).resolve(strict=True)),
            "python_version": sys.version.split()[0],
            "implementation": "stdlib deterministic edit-distance and speaker-cluster bootstrap",
        },
        "model": model_record,
        "runtime": runtime_record,
        "dataset": {
            "id": dataset["id"], "version": dataset["version"], "license_id_operator_supplied": dataset.get("license_id_operator_supplied"),
            "archive_sha256": archive_sha, "test_metadata_sha256": test_meta_sha,
        },
        "inputs": {
            "prediction_file_sha256": prediction_sha,
            "outcomes_sha256": actual_outcomes_sha,
            "intake_report_sha256": sha256_file(intake_report_path),
        },
        "metrics": aggregate(per_clip),
        "latency": latency_metrics,
        "speaker_macro_metrics": {
            "speaker_count": len(speaker_word_clusters),
            "mean_speaker_wer": sum(speaker_word_rates) / len(speaker_word_rates) if speaker_word_rates else None,
            "mean_speaker_cer": sum(speaker_char_rates) / len(speaker_char_rates) if speaker_char_rates else None,
            "interpretation": "each pseudonymous speaker contributes one equally weighted score; no speaker tokens are emitted",
        },
        "confidence_intervals_95": {
            "wer": word_ci,
            "cer": char_ci,
            "bootstrap_seed": 20261013,
            "resampling_unit": "pseudonymous speaker clusters",
            "speaker_cluster_count": len(word_speaker_clusters),
        },
        "reference_length_buckets": {
            name: aggregate(items) if items else {"sample_count": 0, "wer": None, "cer": None}
            for name, items in bucket_rows.items()
        },
        "privacy": {
            "raw_transcripts_included": False,
            "clip_paths_included": False,
            "speaker_tokens_included": False,
        },
    }


def _token_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(hypothesis, start=1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return previous[-1]


def write_report_exclusive(path: Path, report: dict[str, Any], forbidden_roots: list[Path]) -> None:
    _need(not path.is_symlink(), "report output must not be a symlink")
    _need(not path.exists(), "report output must be new; refusing to overwrite evidence")
    _need(path.parent.is_dir(), "report output parent must already exist")
    resolved_parent = path.parent.resolve(strict=True)
    target = resolved_parent / path.name
    for root in forbidden_roots:
        real_root = root.resolve(strict=True)
        _need(target != real_root and real_root not in target.parents,
              "aggregate report must be outside dataset and model directories")
    data = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--intake-report", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--outcomes", required=True, type=Path, help="exact private outcomes JSONL hashed by the prediction pack")
    parser.add_argument("--model-root", required=True, type=Path, help="runtime model directory: converted whisper.cpp export or native Transformers snapshot")
    parser.add_argument("--source-model-root", required=True, type=Path, help="exact local upstream model snapshot directory; same as --model-root for Transformers")
    parser.add_argument("--locale", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="40-hex repository commit SHA used to export model-root")
    parser.add_argument("--model-license", required=True, help="license identifier from the model card")
    parser.add_argument("--runtime-binary", required=True, type=Path, help="exact local whisper-cli executable or Python interpreter used for inference")
    parser.add_argument("--runtime-worker", type=Path, help="exact Transformers worker script; required when runtime.kind is transformers")
    parser.add_argument("--runtime-metadata-log", type=Path, help="private stderr log with the exact Transformers startup record")
    parser.add_argument("--mode", choices=("smoke", "heldout-evaluation"), default="smoke")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100 or args.bootstrap_iterations > 10000:
        parser.error("--bootstrap-iterations must be between 100 and 10000")
    try:
        intake = _read_json(args.intake_report)
        prediction_doc = _read_json(args.predictions)
        report = score_predictions(
            prediction_doc=prediction_doc,
            intake_report=intake,
            dataset_root=args.dataset_root,
            archive=args.archive,
            model_root=args.model_root,
            source_model_root=args.source_model_root,
            predictions_path=args.predictions,
            outcomes_path=args.outcomes,
            model_id=args.model_id,
            model_revision=args.model_revision,
            model_license=args.model_license,
            locale=args.locale,
            intake_report_path=args.intake_report,
            runtime_binary_path=args.runtime_binary,
            runtime_worker_path=args.runtime_worker,
            runtime_metadata_log_path=args.runtime_metadata_log,
            mode=args.mode,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        forbidden_roots = [args.dataset_root, args.model_root, args.source_model_root]
        if args.runtime_worker is not None:
            forbidden_roots.append(args.runtime_worker.parent)
        write_report_exclusive(args.report, report, forbidden_roots)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "report": str(args.report),
        "locale": report["locale"],
        "mode": report["mode"],
        "sample_count": report["metrics"]["sample_count"],
        "wer_percent": report["metrics"]["wer_percent"],
        "cer_percent": report["metrics"]["cer_percent"],
        "interpretation": report["interpretation"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
