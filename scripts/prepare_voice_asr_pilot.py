#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Prepare private 16 kHz mono WAVs for the local Symthaea ASR pilot.

The Common Voice archive is treated as user-provided local data. Nothing is
downloaded or uploaded. This script creates a new private output directory and
does not rewrite the source corpus. transcripts, clip paths and absolute audio
paths remain in the private samples JSONL; only aggregate/hashes should be
shared. Every output WAV is checked after ffmpeg finishes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path, PurePosixPath
from typing import Any

LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: Any, where: str) -> Path:
    _need(isinstance(value, str) and value.strip(), f"{where} must be a non-empty relative path")
    _need("\x00" not in value and "\\" not in value, f"{where} contains invalid path characters")
    parsed = PurePosixPath(value)
    _need(not parsed.is_absolute() and not any(part in {"", ".", ".."} for part in parsed.parts),
          f"{where} must not be absolute or contain traversal components")
    _need(bool(parsed.parts) and not parsed.parts[0].endswith(":"), f"{where} must be relative")
    return Path(*parsed.parts)


def _resolve_audio(clips_root: Path, relative: Path) -> Path:
    _need(not clips_root.is_symlink() and clips_root.is_dir(), "dataset clips root must be a real directory")
    root_real = clips_root.resolve(strict=True)
    candidate = clips_root
    for part in relative.parts:
        candidate = candidate / part
        _need(not candidate.is_symlink(), f"audio path contains a symlink: {relative.as_posix()}")
    _need(candidate.is_file(), f"audio file is missing or not regular: {relative.as_posix()}")
    resolved = candidate.resolve(strict=True)
    _need(root_real in resolved.parents, f"audio path escapes clips root: {relative.as_posix()}")
    _need(candidate.stat().st_size > 0, f"audio file is empty: {relative.as_posix()}")
    return candidate


def _load_dataset(root: Path, archive: Path, intake_path: Path, locale: str) -> tuple[dict[str, Any], list[dict[str, str]], str]:
    _need(root.is_dir() and not root.is_symlink(), "dataset-root must be a real directory, not a symlink")
    _need(archive.is_file() and not archive.is_symlink(), "archive must be an existing regular non-symlink file")
    _need(intake_path.is_file() and not intake_path.is_symlink(), "intake report must be an existing regular non-symlink file")
    intake = _read_json(intake_path)
    _need(isinstance(intake, dict) and intake.get("schema_version") == 1,
          "intake report must have schema_version=1")
    _need(intake.get("audit_kind") == "common_voice_local_dataset_intake",
          "intake report does not match the Common Voice intake schema")
    _need(intake.get("language_tag") == locale, "requested locale differs from intake report")
    _need(intake.get("mode") in {"ingestion-smoke", "heldout-evaluation"},
          "intake report mode is invalid")
    dataset = intake.get("dataset")
    _need(isinstance(dataset, dict), "intake report lacks dataset identity")
    _need(sha256_file(archive) == dataset.get("archive_sha256"),
          "archive SHA-256 differs from intake report")
    splits = intake.get("split_audits")
    _need(isinstance(splits, dict) and isinstance(splits.get("test"), dict),
          "intake report lacks test split metadata")
    test_audit = splits["test"]
    _need(test_audit.get("audio_files_checked", 0) > 0,
          "intake report must record test audio file checks")
    test_tsv = root / "test.tsv"
    _need(test_tsv.is_file() and not test_tsv.is_symlink(), "test.tsv must be a regular file")
    before = sha256_file(test_tsv)
    _need(before == test_audit.get("metadata_sha256"), "test.tsv SHA-256 differs from intake report")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with test_tsv.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields = reader.fieldnames or []
        _need("path" in fields and "sentence" in fields and "client_id" in fields,
              "test.tsv requires path, sentence, and client_id")
        _need(len(fields) == len(set(fields)), "test.tsv has duplicate column names")
        for line_number, row in enumerate(reader, start=2):
            _need(None not in row, f"test.tsv line {line_number} contains extra fields")
            relative = safe_relative(row.get("path"), f"test.tsv line {line_number} path").as_posix()
            _need(relative not in seen, f"duplicate test clip path: {relative}")
            seen.add(relative)
            transcript = row.get("sentence")
            _need(isinstance(transcript, str) and transcript.strip(),
                  f"test.tsv line {line_number} has an empty transcript")
            audio = _resolve_audio(root / "clips", safe_relative(relative, "audio path"))
            rows.append({"id": relative, "reference_transcript": transcript, "source_audio": str(audio)})
    _need(rows, "test.tsv has no rows")
    after = sha256_file(test_tsv)
    _need(before == after, "test.tsv changed during sample preparation")
    _need(len(rows) == test_audit.get("rows"), "test row count differs from intake report")
    _need(len(rows) == test_audit.get("unique_clip_paths"), "test clip count differs from intake report")
    return intake, rows, before


def _private_new_directory(path: Path, *, forbidden_roots: list[Path]) -> Path:
    _need(not path.exists() and not path.is_symlink(),
          "output directory must be new; refusing to overwrite or follow existing paths")
    _need(path.parent.is_dir() and not path.parent.is_symlink(),
          "output parent must be an existing real directory")
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    for root in forbidden_roots:
        resolved_root = root.resolve(strict=True)
        _need(target != resolved_root and target not in resolved_root.parents and resolved_root not in target.parents,
              "output directory must not overlap the dataset or archive location")
    target.mkdir(mode=0o700)
    if os.name == "posix":
        target.chmod(0o700)
    return target


def _write_exclusive(path: Path, data: bytes, mode: int = 0o600) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _decode_wav(path: Path, max_duration_s: float) -> dict[str, Any]:
    _need(path.is_file() and not path.is_symlink(), "ffmpeg output must be a regular non-symlink WAV")
    try:
        with wave.open(str(path), "rb") as reader:
            _need(reader.getnchannels() == 1, "converted WAV must be mono")
            _need(reader.getsampwidth() == 2, "converted WAV must use 16-bit PCM")
            _need(reader.getframerate() == 16000, "converted WAV must be 16 kHz")
            frames = reader.getnframes()
            _need(frames > 0, "converted WAV has no audio frames")
            duration = frames / reader.getframerate()
            _need(duration <= max_duration_s, "converted WAV exceeds configured maximum duration")
            return {
                "sample_rate_hz": reader.getframerate(),
                "channels": reader.getnchannels(),
                "sample_width_bytes": reader.getsampwidth(),
                "frames": frames,
                "duration_s": round(duration, 6),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    except (OSError, wave.Error) as exc:
        raise ValueError(f"cannot validate converted WAV: {exc}") from exc


def prepare(
    *,
    dataset_root: Path,
    archive: Path,
    intake_report_path: Path,
    output_dir: Path,
    locale: str,
    ffmpeg: str = "ffmpeg",
    max_duration_s: int = 60,
    timeout_s: int = 180,
) -> dict[str, Any]:
    _need(LANGUAGE_TAG.fullmatch(locale) is not None, "locale must be a BCP 47-shaped language tag")
    _need(1 <= max_duration_s <= 600, "max_duration_s must be between 1 and 600")
    _need(1 <= timeout_s <= 3600, "timeout_s must be between 1 and 3600")
    binary = shutil.which(ffmpeg)
    _need(binary is not None, f"ffmpeg executable not found: {ffmpeg}")
    ffmpeg_path = Path(binary).resolve(strict=True)
    _need(ffmpeg_path.is_file(), "ffmpeg must resolve to a regular executable file")
    intake, rows, test_sha = _load_dataset(dataset_root, archive, intake_report_path, locale)
    root_real = dataset_root.resolve(strict=True)
    target = _private_new_directory(output_dir, forbidden_roots=[root_real, archive.resolve(strict=True)])
    audio_root = target / "audio"
    audio_root.mkdir(mode=0o700)
    test_samples: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    target_audio_root = output_dir.resolve(strict=False) / "audio"

    mapped_outputs: set[str] = set()
    try:
        for row in rows:
            source_relative = PurePosixPath(row["id"])
            relative_wav = Path(*source_relative.with_suffix(".wav").parts)
            output_key = relative_wav.as_posix()
            _need(output_key not in mapped_outputs, f"two source clips map to the same WAV output: {output_key}")
            mapped_outputs.add(output_key)
            actual_source = Path(row["source_audio"])
            wav_path = audio_root / relative_wav
            wav_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            _need(not wav_path.exists() and not wav_path.is_symlink(), "refusing to overwrite a converted WAV")
            command = [
                str(ffmpeg_path), "-nostdin", "-hide_banner", "-loglevel", "error",
                "-n", "-i", str(actual_source), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(wav_path),
            ]
            try:
                result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, check=False, timeout=timeout_s)
            except subprocess.TimeoutExpired as exc:
                raise ValueError(f"ffmpeg conversion timed out for one test clip after {timeout_s}s") from exc
            except OSError as exc:
                raise ValueError(f"cannot execute ffmpeg for one test clip: {exc}") from exc
            _need(result.returncode == 0, f"ffmpeg conversion failed with exit code {result.returncode}")
            if os.name == "posix":
                # ffmpeg's own file mode depends on the caller's umask; enforce
                # private clip permissions even though the parent is 0700.
                wav_path.chmod(0o600)
            audio_evidence = _decode_wav(wav_path, float(max_duration_s))
            test_samples.append({
                "id": row["id"],
                "language": locale,
                "audio_path": str(target_audio_root / relative_wav),
                "reference_transcript": row["reference_transcript"],
                "translation_reference_en": None,
                "required_entities": [],
            })
            audit_rows.append({
                "clip_id": row["id"],
                "converted_relative_path": output_key,
                **audio_evidence,
            })
        samples_jsonl = "".join(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n" for sample in test_samples).encode("utf-8")
        _write_exclusive(target / "samples.jsonl", samples_jsonl)
        report = {
            "schema_version": 1,
            "report_kind": "voice_asr_sample_preparation",
            "locale": locale,
            "dataset": {
                "id": intake["dataset"]["id"],
                "version": intake["dataset"]["version"],
                "license_id_operator_supplied": intake["dataset"]["license_id_operator_supplied"],
                "archive_sha256": intake["dataset"]["archive_sha256"],
                "test_metadata_sha256": test_sha,
                "intake_report_sha256": sha256_file(intake_report_path),
            },
            "converter": {
                "executable_sha256": sha256_file(ffmpeg_path),
                "version": "not_recorded; binary SHA-256 is authoritative",
                "sample_rate_hz": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
                "max_duration_s": max_duration_s,
                "per_clip_timeout_s": timeout_s,
            },
            "sample_count": len(test_samples),
            "audio_outputs": audit_rows,
            "privacy": {
                "transcripts_in_samples_jsonl": True,
                "clip_paths_in_private_manifest": True,
                "report_is_private_only": True,
                "dataset_audio_copied_outside_private_directory": False,
            },
            "interpretation": "preprocessing evidence only; no ASR inference or quality claim",
        }
        _write_exclusive(target / "prepared_manifest.json",
                         (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        return {
            "output_dir": str(target),
            "locale": locale,
            "sample_count": len(test_samples),
            "archive_sha256": intake["dataset"]["archive_sha256"],
            "test_metadata_sha256": test_sha,
            "interpretation": report["interpretation"],
        }
    except Exception:
        # Preserve failed work for forensic inspection but make it unmistakably
        # incomplete: remove only the final success manifest if one exists.
        try:
            (target / "prepared_manifest.json").unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--intake-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="new private directory; must not exist")
    parser.add_argument("--locale", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--max-duration-s", type=int, default=60)
    parser.add_argument("--timeout-s", type=int, default=180)
    args = parser.parse_args()
    try:
        summary = prepare(
            dataset_root=args.dataset_root, archive=args.archive, intake_report_path=args.intake_report,
            output_dir=args.output_dir, locale=args.locale, ffmpeg=args.ffmpeg,
            max_duration_s=args.max_duration_s, timeout_s=args.timeout_s,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
