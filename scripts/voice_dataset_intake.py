#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Audit a local Common Voice corpus before language-pipeline experiments.

This tool verifies the locally supplied archive and TSV metadata, safe audio
paths, required split structure, and (in heldout-evaluation mode) speaker
partition separation and minimum sample sizes. It does not fetch data, identify
speakers, run an ASR model, or claim speech quality. Only aggregate counts and
cryptographic digests are written to the report; speaker tokens and transcripts
are never emitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

SPLITS = ("train", "dev", "test")
LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")
DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
REQUIRED_RESTRICTIONS = {"no_speaker_identification", "no_rehosting_or_redistribution"}


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, field: str) -> Path:
    _need(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty relative path")
    _need("\x00" not in value and "\\" not in value, f"{field} contains invalid path characters")
    candidate = PurePosixPath(value)
    _need(not candidate.is_absolute() and not any(part in {"", ".", ".."} for part in candidate.parts),
          f"{field} must not be absolute or contain traversal components")
    _need(candidate.parts and not candidate.parts[0].endswith(":"), f"{field} must be repository-relative")
    return Path(*candidate.parts)


def _resolve_inside(root: Path, relative: Path, *, field: str, must_exist: bool = True) -> Path:
    _need(not root.is_symlink(), f"{field} root must not be a symlink")
    root_real = root.resolve(strict=True)
    candidate = root / relative
    _need(not candidate.is_symlink(), f"{field} must not be a symlink")
    if must_exist:
        _need(candidate.exists(), f"{field} does not exist: {relative.as_posix()}")
        _need(candidate.is_file(), f"{field} must be a regular file: {relative.as_posix()}")
    resolved = candidate.resolve(strict=must_exist)
    _need(resolved != root_real and root_real in resolved.parents, f"{field} escapes the supplied dataset root")
    return candidate


def _read_split(path: Path, *, split: str, root: Path, check_audio: bool) -> dict[str, Any]:
    digest_before = _sha256(path)
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fields = reader.fieldnames or []
            _need(len(fields) == len(set(fields)), f"{split}.tsv has duplicate column names")
            required = {"client_id", "path"}
            text_column = "sentence" if "sentence" in fields else "text" if "text" in fields else None
            _need(required.issubset(fields) and text_column is not None,
                  f"{split}.tsv requires client_id, path, and sentence (or text) columns")
            rows = 0
            speakers: set[str] = set()
            clip_paths: set[str] = set()
            missing_speaker_rows = 0
            clips_root = root / "clips"
            for line_number, row in enumerate(reader, start=2):
                _need(None not in row, f"{split}.tsv line {line_number} has more fields than its header")
                relative = _safe_relative(row.get("path"), field=f"{split}.tsv line {line_number} path")
                clip_key = relative.as_posix()
                _need(clip_key not in clip_paths, f"{split}.tsv contains duplicate clip path {clip_key}")
                clip_paths.add(clip_key)
                transcript = row.get(text_column)
                _need(isinstance(transcript, str) and transcript.strip(),
                      f"{split}.tsv line {line_number} has an empty transcript")
                speaker = row.get("client_id")
                if isinstance(speaker, str) and speaker.strip():
                    speakers.add(speaker)
                else:
                    missing_speaker_rows += 1
                if check_audio:
                    audio = _resolve_inside(clips_root, relative, field=f"{split} audio path")
                    _need(audio.stat().st_size > 0, f"{split} audio clip is empty: {clip_key}")
                rows += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read {split} metadata {path}: {exc}") from exc
    _need(rows > 0, f"{split}.tsv has no data rows")
    digest_after = _sha256(path)
    _need(digest_before == digest_after, f"{split}.tsv changed while it was being audited")
    return {
        "metadata_file": path.name,
        "metadata_sha256": digest_after,
        "rows": rows,
        "unique_pseudonymous_speaker_tokens": len(speakers),
        "rows_missing_speaker_token": missing_speaker_rows,
        "unique_clip_paths": len(clip_paths),
        "audio_files_checked": rows if check_audio else 0,
        "_speaker_tokens": speakers,
        "_clip_paths": clip_paths,
    }


def audit_dataset(
    *,
    dataset_root: Path,
    archive: Path,
    locale: str,
    dataset_version: str,
    dataset_id: str,
    source_url: str,
    license_id: str,
    usage_restrictions_acknowledged: bool,
    mode: str = "ingestion-smoke",
    minimum_test_clips: int = 100,
    minimum_test_speakers: int = 10,
    check_audio: bool = True,
) -> dict[str, Any]:
    """Return a privacy-minimized audit report; fail closed on invalid inputs."""
    _need(isinstance(locale, str) and LANGUAGE_TAG.fullmatch(locale) is not None,
          "locale must be a BCP 47-shaped language tag")
    _need(isinstance(dataset_version, str) and dataset_version.strip(), "dataset_version must be explicit")
    _need(isinstance(dataset_id, str) and DATASET_ID.fullmatch(dataset_id) is not None,
          "dataset_id must be explicit and syntactically valid")
    _need(isinstance(source_url, str), "source_url must be a string")
    parsed_url = urlparse(source_url)
    _need(parsed_url.scheme == "https" and parsed_url.hostname is not None
          and parsed_url.hostname.casefold() in {"mozilladatacollective.com", "www.mozilladatacollective.com"},
          "source_url must be an HTTPS Mozilla Data Collective URL")
    _need(isinstance(license_id, str) and license_id.strip(),
          "license_id must be explicit; this field records operator-supplied metadata and is not legal advice")
    _need(usage_restrictions_acknowledged is True,
          "acknowledge Common Voice restrictions: no speaker identification and no re-hosting/redistribution")
    _need(mode in {"ingestion-smoke", "heldout-evaluation"}, "mode must be ingestion-smoke or heldout-evaluation")
    _need(isinstance(minimum_test_clips, int) and not isinstance(minimum_test_clips, bool) and minimum_test_clips > 0,
          "minimum_test_clips must be a positive integer")
    _need(isinstance(minimum_test_speakers, int) and not isinstance(minimum_test_speakers, bool) and minimum_test_speakers > 0,
          "minimum_test_speakers must be a positive integer")
    _need(dataset_root.is_dir() and not dataset_root.is_symlink(), "dataset_root must be a real directory, not a symlink")
    _need(archive.is_file() and not archive.is_symlink(), "archive must be an existing regular file, not a symlink")

    root = dataset_root.resolve(strict=True)
    archive_real = archive.resolve(strict=True)
    _need(archive_real != root and root not in archive_real.parents,
          "archive must be outside the extracted dataset root")
    _need(archive_real.stat().st_size > 0, "archive must not be empty")
    metadata: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        metadata_path = _resolve_inside(root, Path(f"{split}.tsv"), field=f"{split}.tsv")
        metadata[split] = _read_split(metadata_path, split=split, root=root, check_audio=check_audio)

    all_clip_paths: dict[str, str] = {}
    for split in SPLITS:
        for clip_path in metadata[split]["_clip_paths"]:
            _need(clip_path not in all_clip_paths,
                  f"audio path appears in both {all_clip_paths.get(clip_path)} and {split} splits")
            all_clip_paths[clip_path] = split

    overlaps: dict[str, int] = {}
    pairs = (("train", "dev"), ("train", "test"), ("dev", "test"))
    for left, right in pairs:
        overlaps[f"{left}_and_{right}"] = len(
            metadata[left]["_speaker_tokens"] & metadata[right]["_speaker_tokens"]
        )
    speaker_tokens_present = all(metadata[name]["rows_missing_speaker_token"] == 0 for name in SPLITS)
    speaker_disjoint = speaker_tokens_present and all(count == 0 for count in overlaps.values())
    test = metadata["test"]
    if mode == "heldout-evaluation":
        _need(speaker_tokens_present,
              "heldout-evaluation requires a non-empty pseudonymous client_id on every row")
        _need(speaker_disjoint,
              f"heldout-evaluation requires speaker-disjoint splits; overlap counts={overlaps}")
        _need(test["rows"] >= minimum_test_clips,
              f"test split has {test['rows']} clips; minimum is {minimum_test_clips}")
        _need(test["unique_pseudonymous_speaker_tokens"] >= minimum_test_speakers,
              f"test split has {test['unique_pseudonymous_speaker_tokens']} speakers; minimum is {minimum_test_speakers}")

    clean_splits: dict[str, dict[str, Any]] = {}
    for split, values in metadata.items():
        clean_splits[split] = {key: value for key, value in values.items()
                               if key not in {"_speaker_tokens", "_clip_paths"}}
    return {
        "schema_version": 1,
        "audit_kind": "common_voice_local_dataset_intake",
        "mode": mode,
        "language_tag": locale,
        "dataset": {
            "id": dataset_id,
            "version": dataset_version,
            "source_url": source_url,
            "license_id_operator_supplied": license_id,
            "usage_restrictions_acknowledged": sorted(REQUIRED_RESTRICTIONS),
            "archive_file_name": archive.name,
            "archive_size_bytes": archive_real.stat().st_size,
            "archive_sha256": _sha256(archive_real),
        },
        "split_audits": clean_splits,
        "integrity": {
            "audio_paths_unique_across_splits": True,
            "speaker_disjoint": speaker_disjoint,
            "speaker_overlap_counts": overlaps,
            "speaker_tokens_missing_in_any_split": not speaker_tokens_present,
        },
        "interpretation": (
            "ingestion smoke only; this report does not demonstrate ASR accuracy or production readiness"
            if mode == "ingestion-smoke"
            else "dataset integrity checks passed for a held-out evaluation setup; no model quality is implied"
        ),
    }


def write_report_exclusive(path: Path, report: dict[str, Any]) -> None:
    _need(not path.is_symlink(), "report output must not be a symlink")
    _need(not path.exists(), "report output must be new; refusing to overwrite evidence")
    _need(path.parent.is_dir(), "report parent directory must already exist")
    data = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, help="unpacked corpus root containing train.tsv/dev.tsv/test.tsv and clips/")
    parser.add_argument("--archive", required=True, type=Path, help="original archive used to create dataset-root")
    parser.add_argument("--locale", required=True, help="BCP 47 language tag, e.g. af")
    parser.add_argument("--dataset-version", required=True, help="exact dataset release/version")
    parser.add_argument("--dataset-id", required=True, help="Mozilla Data Collective dataset ID")
    parser.add_argument("--source-url", required=True, help="HTTPS dataset page on Mozilla Data Collective")
    parser.add_argument("--license-id", required=True, help="license identifier copied from the dataset's terms")
    parser.add_argument("--acknowledge-usage-restrictions", action="store_true", help="acknowledge no speaker identification and no re-hosting/redistribution")
    parser.add_argument("--mode", choices=("ingestion-smoke", "heldout-evaluation"), default="ingestion-smoke")
    parser.add_argument("--minimum-test-clips", type=int, default=100)
    parser.add_argument("--minimum-test-speakers", type=int, default=10)
    parser.add_argument("--skip-audio-files", action="store_true", help="check metadata only; never treats missing audio as a successful ingestion smoke test")
    parser.add_argument("--report", required=True, type=Path, help="new private report path; existing paths are refused")
    args = parser.parse_args()
    try:
        report = audit_dataset(
            dataset_root=args.dataset_root,
            archive=args.archive,
            locale=args.locale,
            dataset_version=args.dataset_version,
            dataset_id=args.dataset_id,
            source_url=args.source_url,
            license_id=args.license_id,
            usage_restrictions_acknowledged=args.acknowledge_usage_restrictions,
            mode=args.mode,
            minimum_test_clips=args.minimum_test_clips,
            minimum_test_speakers=args.minimum_test_speakers,
            check_audio=not args.skip_audio_files,
        )
        write_report_exclusive(args.report, report)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    summary = {
        "mode": report["mode"],
        "language_tag": report["language_tag"],
        "archive_sha256": report["dataset"]["archive_sha256"],
        "split_rows": {name: item["rows"] for name, item in report["split_audits"].items()},
        "speaker_disjoint": report["integrity"]["speaker_disjoint"],
        "report": str(args.report),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
