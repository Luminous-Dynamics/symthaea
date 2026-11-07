#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Convert private Symthaea Pilot outcomes into a provenance-pinned ASR prediction pack.

This script reads the private JSONL emitted by symthaea-comm-eval pilot. It
binds every result to the exact Common Voice test.tsv and archive through the
prior intake report, and requires exact model/runtime/converter commits plus
hashes of the whisper.cpp executable and converted model file. Transcripts and
clip paths stay in the prediction file; keep it private and never commit it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from voice_asr_score import (
    SHA1_RE,
    SHA256_RE,
    _hash_tree,
    _load_intake,
    _need,
    _read_json,
    safe_relative_path,
    sha256_file,
)

def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


RUNTIME_METADATA_PREFIX = "SYMTHEA_TRANSFORMERS_RUNTIME "
RUNTIME_METADATA_KEYS = {
    "python_version", "torch_version", "transformers_version", "numpy_version",
    "device", "dtype", "offline", "model_snapshot_manifest_sha256",
    "model_snapshot_file_count", "model_snapshot_bytes",
    "python_executable_sha256", "worker_script_sha256",
}


def read_transformers_runtime_metadata(path: Path) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(),
          "runtime metadata log must be a regular non-symlink file")
    before = sha256_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [line[len(RUNTIME_METADATA_PREFIX):] for line in lines
               if line.startswith(RUNTIME_METADATA_PREFIX)]
    _need(len(matches) == 1, "runtime metadata log must contain exactly one worker startup record")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value}")

    try:
        metadata = json.loads(
            matches[0], object_pairs_hook=_unique_object, parse_constant=reject_constant
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"runtime startup metadata is malformed: {error}") from error
    _need(isinstance(metadata, dict) and set(metadata) == RUNTIME_METADATA_KEYS,
          "runtime startup metadata fields mismatch")
    for field in ("python_version", "torch_version", "transformers_version", "numpy_version"):
        _need(isinstance(metadata[field], str) and metadata[field].strip()
              and "\n" not in metadata[field] and "\r" not in metadata[field],
              f"runtime metadata {field} must be explicit")
    _need(metadata["device"] in {"cpu", "cuda"}, "runtime metadata device must be cpu or cuda")
    _need(metadata["dtype"] in {"float32", "float16", "bfloat16"}, "runtime metadata dtype is unsupported")
    _need(metadata["offline"] is True, "runtime metadata must declare offline=true")
    for field in ("model_snapshot_manifest_sha256", "python_executable_sha256", "worker_script_sha256"):
        _need(isinstance(metadata[field], str) and SHA256_RE.fullmatch(metadata[field]) is not None,
              f"runtime metadata {field} must be a 64-hex digest")
    for field in ("model_snapshot_file_count", "model_snapshot_bytes"):
        _need(isinstance(metadata[field], int) and not isinstance(metadata[field], bool)
              and metadata[field] >= 0, f"runtime metadata {field} must be a non-negative integer")
    _need(metadata["model_snapshot_file_count"] > 0, "runtime metadata model snapshot cannot be empty")
    _need(before == sha256_file(path), "runtime metadata log changed while it was being read")
    return metadata


OUTCOME_KEYS = {
    "id", "language", "detected_language", "language_confidence", "transcript",
    "word_error_rate", "character_error_rate", "entity_preservation",
    "translation_en", "translation_chrf", "response_language_correct",
    "latency_ms", "preflight_latency_ms", "transcription_latency_ms", "translation_latency_ms", "error",
}


def read_outcomes(path: Path) -> tuple[list[dict[str, Any]], str]:
    _need(path.is_file() and not path.is_symlink(), "outcomes must be a regular non-symlink file")
    before = sha256_file(path)
    outcomes: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(
                        line,
                        parse_constant=lambda value: (_ for _ in ()).throw(
                            ValueError(f"non-standard JSON numeric constant {value}")
                        ),
                        object_pairs_hook=_unique_object,
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"malformed outcome JSON on line {line_number}: {exc}") from exc
                _need(isinstance(row, dict) and set(row) == OUTCOME_KEYS,
                      f"outcome line {line_number} has unexpected fields")
                for field in ("latency_ms", "preflight_latency_ms", "transcription_latency_ms", "translation_latency_ms"):
                    metric = row[field]
                    if metric is None and field != "latency_ms":
                        continue
                    _need(isinstance(metric, (int, float)) and not isinstance(metric, bool)
                          and math.isfinite(metric) and metric >= 0,
                          f"outcome line {line_number} {field} must be a finite non-negative number"
                          if field == "latency_ms" else
                          f"outcome line {line_number} {field} must be null or a finite non-negative number")
                outcomes.append(row)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read pilot outcomes: {exc}") from exc
    after = sha256_file(path)
    _need(before == after, "outcomes file changed while it was being read")
    _need(outcomes, "outcomes file contains no rows")
    return outcomes, after


def build_predictions(
    *,
    outcomes_path: Path,
    intake_report_path: Path,
    intake_report: Any,
    dataset_root: Path,
    archive: Path,
    model_root: Path,
    source_model_root: Path,
    runtime_binary: Path,
    locale: str,
    model_id: str,
    model_revision: str,
    model_license: str,
    runtime_commit: str | None = None,
    converter_commit: str | None = None,
    artifact_path: str | None = None,
    quantization: str | None = None,
    runtime_kind: str = "whisper.cpp",
    runtime_worker_path: Path | None = None,
    runtime_metadata_log_path: Path | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    transformers_version: str | None = None,
    numpy_version: str | None = None,
    device: str | None = None,
    dtype: str | None = None,
    runtime_model_snapshot_sha256: str | None = None,
    runtime_python_executable_sha256: str | None = None,
    runtime_worker_sha256: str | None = None,
    mode: str = "smoke",
) -> dict[str, Any]:
    _need(mode in {"smoke", "heldout-evaluation"}, "mode must be smoke or heldout-evaluation")
    _need(isinstance(model_id, str) and model_id.strip(), "model_id must be explicit")
    _need(isinstance(model_revision, str) and SHA1_RE.fullmatch(model_revision),
          "model_revision must be a 40-hex repository commit SHA")
    _need(isinstance(model_license, str) and model_license.strip(), "model license must be explicit")
    _need(runtime_kind in {"whisper.cpp", "transformers"}, "runtime_kind must be whisper.cpp or transformers")
    if runtime_kind == "whisper.cpp":
        for name, value in (("runtime_commit", runtime_commit), ("converter_commit", converter_commit)):
            _need(isinstance(value, str) and SHA1_RE.fullmatch(value),
                  f"{name} must be a 40-hex repository commit SHA")
        _need(isinstance(quantization, str) and re.fullmatch(r"[A-Za-z0-9._+-]{1,32}", quantization) is not None,
              "quantization must be an explicit compact identifier")
        _need(isinstance(artifact_path, str) and artifact_path.strip(),
              "artifact_path is required for whisper.cpp")
    else:
        _need(runtime_worker_path is not None and runtime_worker_path.is_file()
              and not runtime_worker_path.is_symlink(),
              "Transformers runtime requires a regular runtime_worker_path")
        _need(runtime_metadata_log_path is not None,
              "Transformers runtime requires --runtime-metadata-log from worker stderr")
        observed = read_transformers_runtime_metadata(runtime_metadata_log_path)
        runtime_startup_log_sha256 = sha256_file(runtime_metadata_log_path)
        observed_fields = {
            "python_version": observed["python_version"],
            "torch_version": observed["torch_version"],
            "transformers_version": observed["transformers_version"],
            "numpy_version": observed["numpy_version"],
            "device": observed["device"],
            "dtype": observed["dtype"],
            "runtime_model_snapshot_sha256": observed["model_snapshot_manifest_sha256"],
            "runtime_python_executable_sha256": observed["python_executable_sha256"],
            "runtime_worker_sha256": observed["worker_script_sha256"],
        }
        provided_fields = {
            "python_version": python_version, "torch_version": torch_version,
            "transformers_version": transformers_version, "numpy_version": numpy_version,
            "device": device, "dtype": dtype,
            "runtime_model_snapshot_sha256": runtime_model_snapshot_sha256,
            "runtime_python_executable_sha256": runtime_python_executable_sha256,
            "runtime_worker_sha256": runtime_worker_sha256,
        }
        for name, provided in provided_fields.items():
            if provided is not None:
                _need(provided == observed_fields[name],
                      f"{name} differs from the worker startup metadata")
        python_version = observed["python_version"]
        torch_version = observed["torch_version"]
        transformers_version = observed["transformers_version"]
        numpy_version = observed["numpy_version"]
        device = observed["device"]
        dtype = observed["dtype"]
        runtime_model_snapshot_sha256 = observed["model_snapshot_manifest_sha256"]
        runtime_python_executable_sha256 = observed["python_executable_sha256"]
        runtime_worker_sha256 = observed["worker_script_sha256"]

    dataset, test_rows, test_sha = _load_intake(
        intake_report, dataset_root=dataset_root, archive=archive, locale=locale, expected_mode=mode
    )
    outcomes, outcomes_hash = read_outcomes(outcomes_path)
    expected = {row["path"]: row for row in test_rows}
    seen: dict[str, str] = {}
    for index, row in enumerate(outcomes):
        raw_id = row["id"]
        relative_id = safe_relative_path(raw_id, f"outcome {index} id").as_posix()
        _need(relative_id in expected, f"outcome {index} references an unknown test clip")
        _need(relative_id not in seen, f"duplicate outcome for test clip {relative_id}")
        _need(row["language"] == locale, f"outcome {index} language differs from locale")
        _need(row["error"] is None, f"pilot outcome for {relative_id} records a worker/inference error")
        transcript = row["transcript"]
        _need(isinstance(transcript, str), f"pilot outcome for {relative_id} lacks a transcript")
        seen[relative_id] = transcript
    missing = set(expected) - set(seen)
    _need(not missing, f"missing {len(missing)} pilot outcomes; prediction pack must be complete")

    _need(model_root.is_dir() and not model_root.is_symlink(),
          "model_root must be a real directory, not a symlink")
    _need(source_model_root.is_dir() and not source_model_root.is_symlink(),
          "source_model_root must be a real directory, not a symlink")
    converted_root_real = model_root.resolve(strict=True)
    source_root_real = source_model_root.resolve(strict=True)
    _need(converted_root_real == source_root_real or (
          converted_root_real not in source_root_real.parents
          and source_root_real not in converted_root_real.parents),
          "source_model_root and runtime model_root must be the same snapshot or separate non-nested directories")
    source_snapshot_sha, _source_file_count, _source_bytes = _hash_tree(source_model_root)
    _need(runtime_binary.is_file() and not runtime_binary.is_symlink(),
          "runtime_binary must be a regular non-symlink file")
    runtime_sha = sha256_file(runtime_binary)
    runtime_record: dict[str, Any]
    runtime_startup_log_sha256: str | None = None
    if runtime_kind == "whisper.cpp":
        _need(converted_root_real != source_root_real,
              "whisper.cpp converted export and source snapshot must be separate directories")
        relative_artifact = safe_relative_path(artifact_path or "", "artifact_path")
        artifact = model_root / relative_artifact
        current = model_root
        for part in relative_artifact.parts:
            current = current / part
            _need(not current.is_symlink(), "model artifact path contains a symlink")
        _need(artifact.is_file(), "artifact_path must identify a regular model file")
        _need(artifact.resolve(strict=True).is_relative_to(converted_root_real),
              "artifact_path escapes model_root")
        artifact_sha = sha256_file(artifact)
        runtime_record = {
            "kind": "whisper.cpp",
            "runtime_commit": runtime_commit,
            "converter_commit": converter_commit,
            "runtime_binary_sha256": runtime_sha,
            "artifact_path": relative_artifact.as_posix(),
            "artifact_sha256": artifact_sha,
            "quantization": quantization,
        }
    else:
        _need(converted_root_real == source_root_real,
              "Transformers model-root must be the exact source-model-root snapshot")
        actual_worker_sha = sha256_file(runtime_worker_path)
        _need(source_snapshot_sha == runtime_model_snapshot_sha256,
              "local model snapshot digest differs from the value observed at worker startup")
        observed = read_transformers_runtime_metadata(runtime_metadata_log_path)
        _need(sha256_file(runtime_metadata_log_path) == runtime_startup_log_sha256,
              "runtime metadata log changed during prediction packaging")
        _need(_source_file_count == observed["model_snapshot_file_count"]
              and _source_bytes == observed["model_snapshot_bytes"],
              "model snapshot file count/size differs from the worker startup metadata")
        _need(runtime_sha == runtime_python_executable_sha256,
              "Python executable digest differs from the value observed at worker startup")
        _need(actual_worker_sha == runtime_worker_sha,
              "worker script digest differs from the value observed at worker startup")
        runtime_record = {
            "kind": "transformers",
            "python_executable_sha256": runtime_sha,
            "worker_script_sha256": actual_worker_sha,
            "model_snapshot_manifest_sha256": source_snapshot_sha,
            "startup_metadata_log_sha256": runtime_startup_log_sha256,
            "python_version": python_version,
            "torch_version": torch_version,
            "transformers_version": transformers_version,
            "numpy_version": numpy_version,
            "device": device,
            "dtype": dtype,
            "offline": True,
        }

    return {
        "schema_version": 1,
        "locale": locale,
        "dataset_id": dataset["id"],
        "dataset_version": dataset["version"],
        "archive_sha256": dataset["archive_sha256"],
        "test_metadata_sha256": test_sha,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_license": model_license,
        "source_model_snapshot_sha256": source_snapshot_sha,
        "outcomes_sha256": outcomes_hash,
        "runtime": runtime_record,
        "predictions": [
            {"path": path, "text": seen[path]}
            for path in sorted(expected)
        ],
    }


def write_private_exclusive(path: Path, doc: dict[str, Any], forbidden_roots: list[Path]) -> None:
    _need(not path.exists() and not path.is_symlink(),
          "prediction output must be new; refusing to overwrite private transcripts")
    _need(path.parent.is_dir() and not path.parent.is_symlink(),
          "prediction output parent must be an existing real directory")
    resolved_parent = path.parent.resolve(strict=True)
    target = resolved_parent / path.name
    for root in forbidden_roots:
        resolved = root.resolve(strict=True)
        _need(target != resolved and resolved not in target.parents,
              "prediction file must be outside dataset, model, and source-outcome trees")
    encoded = (json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(target, flags, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True, type=Path, help="private JSONL created by symthaea-comm-eval pilot")
    parser.add_argument("--intake-report", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path, help="runtime model directory: converted whisper.cpp export or native Transformers snapshot")
    parser.add_argument("--source-model-root", required=True, type=Path, help="exact upstream snapshot; same directory as --model-root for Transformers")
    parser.add_argument("--runtime-binary", required=True, type=Path, help="whisper-cli executable or exact Python interpreter used for inference")
    parser.add_argument("--runtime-kind", choices=("whisper.cpp", "transformers"), default="whisper.cpp")
    parser.add_argument("--runtime-worker", type=Path, help="exact local Transformers worker script; required for runtime-kind transformers")
    parser.add_argument("--runtime-metadata-log", type=Path, help="private stderr log containing the exact SYMTHEA_TRANSFORMERS_RUNTIME startup record")
    parser.add_argument("--locale", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="exact 40-hex upstream model repository commit")
    parser.add_argument("--model-license", required=True)
    parser.add_argument("--runtime-commit", help="40-hex whisper.cpp source commit used to build runtime")
    parser.add_argument("--converter-commit", help="40-hex whisper.cpp source commit used to convert model")
    parser.add_argument("--artifact-path", help="model path relative to model-root for whisper.cpp")
    parser.add_argument("--quantization", help="e.g. f16 or q5_0; never inferred from filename")
    parser.add_argument("--python-version", help="exact Python version reported by the worker")
    parser.add_argument("--torch-version", help="exact PyTorch version reported by the worker")
    parser.add_argument("--transformers-version", help="exact Transformers version reported by the worker")
    parser.add_argument("--numpy-version", help="exact NumPy version reported by the worker")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="actual device reported by the worker")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), help="actual dtype reported by the worker")
    parser.add_argument("--runtime-model-snapshot-sha256", help="model manifest digest printed by the worker at startup")
    parser.add_argument("--runtime-python-executable-sha256", help="Python executable digest printed by the worker at startup")
    parser.add_argument("--runtime-worker-sha256", help="worker script digest printed by the worker at startup")
    parser.add_argument("--mode", choices=("smoke", "heldout-evaluation"), default="smoke")
    parser.add_argument("--output", required=True, type=Path, help="new private predictions JSON; contains all reference and hypothesis text")
    args = parser.parse_args()
    try:
        intake = _read_json(args.intake_report)
        doc = build_predictions(
            outcomes_path=args.outcomes, intake_report_path=args.intake_report, intake_report=intake,
            dataset_root=args.dataset_root, archive=args.archive, model_root=args.model_root,
            source_model_root=args.source_model_root, runtime_binary=args.runtime_binary, locale=args.locale, model_id=args.model_id,
            model_revision=args.model_revision, model_license=args.model_license,
            runtime_commit=args.runtime_commit, converter_commit=args.converter_commit,
            artifact_path=args.artifact_path, quantization=args.quantization,
            runtime_kind=args.runtime_kind, runtime_worker_path=args.runtime_worker,
            runtime_metadata_log_path=args.runtime_metadata_log,
            python_version=args.python_version, torch_version=args.torch_version,
            transformers_version=args.transformers_version, numpy_version=args.numpy_version,
            device=args.device, dtype=args.dtype,
            runtime_model_snapshot_sha256=args.runtime_model_snapshot_sha256,
            runtime_python_executable_sha256=args.runtime_python_executable_sha256,
            runtime_worker_sha256=args.runtime_worker_sha256, mode=args.mode,
        )
        forbidden_roots = [args.dataset_root, args.model_root, args.source_model_root]
        if args.runtime_worker is not None:
            forbidden_roots.append(args.runtime_worker.parent)
        write_private_exclusive(args.output, doc, forbidden_roots)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output": str(args.output),
        "locale": args.locale,
        "sample_count": len(doc["predictions"]),
        "outcomes_sha256": doc["outcomes_sha256"],
        "training_data_overlap_status": "not_assessed",
        "interpretation": "private prediction packaging only; no ASR quality or production-support claim",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
