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
import os
import re
from pathlib import Path
from typing import Any

from voice_asr_score import (
    SHA1_RE,
    _hash_tree,
    _load_intake,
    _need,
    _read_json,
    safe_relative_path,
    sha256_file,
)

OUTCOME_KEYS = {
    "id", "language", "detected_language", "language_confidence", "transcript",
    "word_error_rate", "character_error_rate", "entity_preservation",
    "translation_en", "translation_chrf", "response_language_correct",
    "latency_ms", "error",
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
                    )
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"malformed outcome JSON on line {line_number}: {exc}") from exc
                _need(isinstance(row, dict) and set(row) == OUTCOME_KEYS,
                      f"outcome line {line_number} has unexpected fields")
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
    runtime_commit: str,
    converter_commit: str,
    artifact_path: str,
    quantization: str,
    mode: str = "smoke",
) -> dict[str, Any]:
    _need(mode in {"smoke", "heldout-evaluation"}, "mode must be smoke or heldout-evaluation")
    _need(isinstance(model_id, str) and model_id.strip(), "model_id must be explicit")
    _need(isinstance(model_revision, str) and SHA1_RE.fullmatch(model_revision),
          "model_revision must be a 40-hex repository commit SHA")
    _need(isinstance(model_license, str) and model_license.strip(), "model license must be explicit")
    for name, value in (("runtime_commit", runtime_commit), ("converter_commit", converter_commit)):
        _need(isinstance(value, str) and SHA1_RE.fullmatch(value),
              f"{name} must be a 40-hex repository commit SHA")
    _need(isinstance(quantization, str) and re.fullmatch(r"[A-Za-z0-9._+-]{1,32}", quantization) is not None,
          "quantization must be an explicit compact identifier")

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
    source_snapshot_sha, _source_file_count, _source_bytes = _hash_tree(source_model_root)
    _need(runtime_binary.is_file() and not runtime_binary.is_symlink(),
          "runtime_binary must be a regular non-symlink file")
    model_root_real = model_root.resolve(strict=True)
    relative_artifact = safe_relative_path(artifact_path, "artifact_path")
    artifact = model_root / relative_artifact
    current = model_root
    for part in relative_artifact.parts:
        current = current / part
        _need(not current.is_symlink(), "model artifact path contains a symlink")
    _need(artifact.is_file(), "artifact_path must identify a regular model file")
    _need(artifact.resolve(strict=True).is_relative_to(model_root_real),
          "artifact_path escapes model_root")
    artifact_sha = sha256_file(artifact)
    runtime_sha = sha256_file(runtime_binary)

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
        "runtime": {
            "kind": "whisper.cpp",
            "runtime_commit": runtime_commit,
            "converter_commit": converter_commit,
            "runtime_binary_sha256": runtime_sha,
            "artifact_path": relative_artifact.as_posix(),
            "artifact_sha256": artifact_sha,
            "quantization": quantization,
        },
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
    parser.add_argument("--model-root", required=True, type=Path, help="converted local whisper.cpp model package")
    parser.add_argument("--source-model-root", required=True, type=Path, help="exact upstream model snapshot as acquired locally")
    parser.add_argument("--runtime-binary", required=True, type=Path)
    parser.add_argument("--locale", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="exact 40-hex upstream model repository commit")
    parser.add_argument("--model-license", required=True)
    parser.add_argument("--runtime-commit", required=True, help="40-hex whisper.cpp source commit used to build runtime")
    parser.add_argument("--converter-commit", required=True, help="40-hex whisper.cpp source commit used to convert model")
    parser.add_argument("--artifact-path", required=True, help="model path relative to model-root")
    parser.add_argument("--quantization", required=True, help="e.g. f16 or q5_0; never inferred from filename")
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
            artifact_path=args.artifact_path, quantization=args.quantization, mode=args.mode,
        )
        write_private_exclusive(
            args.output, doc,
            [args.dataset_root, args.model_root, args.source_model_root],
        )
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
