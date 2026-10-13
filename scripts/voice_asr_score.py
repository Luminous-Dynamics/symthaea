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
import statistics
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
NORMALIZATION_ID = "nfkc-casefold-punctuation-to-space-v1"
PREDICTION_KEYS = {
    "schema_version", "locale", "dataset_id", "dataset_version",
    "archive_sha256", "test_metadata_sha256", "model_id", "model_revision",
    "model_license", "predictions",
}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {value}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON from {path}: {exc}") from exc


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
    _need(isinstance(intake_report, dict) and intake_report.get("schema_version") == 1,
          "intake report must have schema_version=1")
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


def _bootstrap_ci(samples: list[dict[str, int]], metric: str, iterations: int = 1000, seed: int = 20261013) -> dict[str, float]:
    _need(iterations > 0, "bootstrap iterations must be positive")
    rng = random.Random(seed)
    values: list[float] = []
    count = len(samples)
    for _ in range(iterations):
        selected = [samples[rng.randrange(count)] for _ in range(count)]
        denominator = sum(item[f"ref_{metric}"] for item in selected)
        numerator = sum(item[f"err_{metric}"] for item in selected)
        values.append(numerator / denominator if denominator else 0.0)
    values.sort()
    return {
        "low_95": _percentile(values, 0.025),
        "high_95": _percentile(values, 0.975),
        "iterations": iterations,
        "seed": seed,
        "method": "deterministic percentile bootstrap resampling clips with replacement",
    }


def score_predictions(
    *,
    prediction_doc: Any,
    intake_report: Any,
    dataset_root: Path,
    archive: Path,
    model_root: Path,
    predictions_path: Path,
    model_id: str,
    model_revision: str,
    model_license: str,
    locale: str,
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
    _need(isinstance(prediction_doc, dict), "prediction document must be an object")
    _need(set(prediction_doc) == PREDICTION_KEYS,
          f"prediction document keys mismatch; missing={sorted(PREDICTION_KEYS-set(prediction_doc))}, extra={sorted(set(prediction_doc)-PREDICTION_KEYS)}")
    _need(prediction_doc["schema_version"] == 1, "prediction schema_version must equal 1")
    _need(prediction_doc["locale"] == locale, "prediction locale mismatch")
    _need(prediction_doc["model_id"] == model_id, "prediction model_id mismatch")
    _need(prediction_doc["model_revision"] == model_revision, "prediction model_revision mismatch")
    _need(prediction_doc["model_license"] == model_license, "prediction model_license mismatch")

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
    for index, row in enumerate(hypotheses):
        _need(isinstance(row, dict) and set(row) == {"path", "text"},
              f"prediction {index} must contain only path and text")
        path = safe_relative_path(row["path"], f"prediction {index} path").as_posix()
        _need(path in expected, f"prediction {index} references an unknown clip")
        _need(path not in seen, f"duplicate prediction for clip path {path}")
        _need(isinstance(row["text"], str), f"prediction {index} text must be a string")
        seen[path] = normalize_text(row["text"])
    missing = set(expected) - set(seen)
    _need(not missing, f"missing {len(missing)} ASR predictions; scoring is incomplete")

    model_root = model_root.resolve(strict=True)
    model_tree_sha, model_files, model_bytes = _hash_tree(model_root)
    per_clip: list[dict[str, int]] = []
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
        word_errors = edit_distance(" ".join(ref_words), " ".join(hyp_words))
        # Word-level Levenshtein must operate on tokens, not the joined characters.
        word_errors = _token_edit_distance(ref_words, hyp_words)
        char_errors = edit_distance(ref, hyp)
        item = {
            "ref_words": len(ref_words), "err_words": word_errors,
            "ref_chars": ref_chars, "err_chars": char_errors,
        }
        per_clip.append(item)
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
    aggregate_samples = [
        {
            "ref_word": row["ref_words"], "err_word": row["err_words"],
            "ref_char": row["ref_chars"], "err_char": row["err_chars"],
        }
        for row in per_clip
    ]
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

    word_bootstrap_rows = [{"ref_word": x["ref_words"], "err_word": x["err_words"]} for x in per_clip]
    char_bootstrap_rows = [{"ref_char": x["ref_chars"], "err_char": x["err_chars"]} for x in per_clip]
    def ci_rows(items: list[dict[str, int]], ref_key: str, err_key: str) -> list[dict[str, int]]:
        return [{"ref_metric": item[ref_key], "err_metric": item[err_key]} for item in items]
    # Avoid a Python/package dependency on jiwer or evaluate; the normalization and edit distance are versioned above.
    word_ci = _bootstrap_ci(ci_rows(word_bootstrap_rows, "ref_word", "err_word"), "metric", bootstrap_iterations)
    char_ci = _bootstrap_ci(ci_rows(char_bootstrap_rows, "ref_char", "err_char"), "metric", bootstrap_iterations)

    archive_sha = dataset["archive_sha256"]
    test_meta_sha = test_sha
    prediction_sha = sha256_file(predictions_path)
    return {
        "schema_version": 1,
        "report_kind": "voice_asr_score",
        "mode": mode,
        "interpretation": (
            "exploratory smoke metric only; not evidence of production readiness or language support"
            if mode == "smoke"
            else "held-out ASR metrics for this exact model/data split; not by themselves a production-support claim"
        ),
        "locale": locale,
        "normalization": NORMALIZATION_ID,
        "model": {
            "id": model_id, "revision_claimed": model_revision, "license_declared": model_license,
            "snapshot_manifest_sha256": model_tree_sha, "snapshot_file_count": model_files,
            "snapshot_bytes": model_bytes,
        },
        "dataset": {
            "id": dataset["id"], "version": dataset["version"], "license_id_operator_supplied": dataset.get("license_id_operator_supplied"),
            "archive_sha256": archive_sha, "test_metadata_sha256": test_meta_sha,
        },
        "inputs": {
            "prediction_file_sha256": prediction_sha,
            "intake_report_sha256": sha256_file(intake_path) if (intake_path := _INTAKE_PATH.get()) else None,
        },
        "metrics": aggregate(per_clip),
        "confidence_intervals_95": {
            "wer": word_ci,
            "cer": char_ci,
            "bootstrap_seed": 20261013,
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


# Context set by CLI only so that the score report can bind the exact intake receipt.
class _IntakePath:
    def __init__(self) -> None:
        self.path: Path | None = None
    def set(self, path: Path) -> None:
        self.path = path
    def get(self) -> Path | None:
        return self.path

_INTAKE_PATH = _IntakePath()


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
    parser.add_argument("--model-root", required=True, type=Path, help="regular-file export of the exact offline model snapshot")
    parser.add_argument("--locale", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True, help="40-hex repository commit SHA used to export model-root")
    parser.add_argument("--model-license", required=True, help="license identifier from the model card")
    parser.add_argument("--mode", choices=("smoke", "heldout-evaluation"), default="smoke")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100 or args.bootstrap_iterations > 10000:
        parser.error("--bootstrap-iterations must be between 100 and 10000")
    try:
        intake = _read_json(args.intake_report)
        prediction_doc = _read_json(args.predictions)
        _INTAKE_PATH.set(args.intake_report)
        report = score_predictions(
            prediction_doc=prediction_doc,
            intake_report=intake,
            dataset_root=args.dataset_root,
            archive=args.archive,
            model_root=args.model_root,
            predictions_path=args.predictions,
            model_id=args.model_id,
            model_revision=args.model_revision,
            model_license=args.model_license,
            locale=args.locale,
            mode=args.mode,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        write_report_exclusive(args.report, report, [args.dataset_root, args.model_root])
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
