#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline-only BuzzASR Afrikaans Transformers worker for Symthaea's JSONL protocol.

The model must already exist as a local snapshot. This worker sets Hugging Face
offline mode and uses local_files_only=True; it never downloads or executes
remote model code. It implements ASR only, not language identification or
translation. Stdout is reserved for one JSON response per request; diagnostics
and runtime metadata go to stderr.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 96 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 1024 * 1024
LOCALE = "af"


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value}")


def parse_request(line: str) -> dict[str, Any]:
    value = json.loads(
        line,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    request_id = value.get("request_id")
    operation = value.get("operation")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must be a non-empty string")
    request_fields = {
        "capabilities": {"operation", "request_id"},
        "identify_language": {"operation", "request_id", "observation"},
        "transcribe": {"operation", "request_id", "observation"},
        "translate": {"operation", "request_id", "input", "target_language"},
        "translate_speech": {"operation", "request_id", "observation", "target_language"},
    }
    if not isinstance(operation, str) or operation not in request_fields:
        raise ValueError("unknown human-provider operation")
    if set(value) != request_fields[operation]:
        raise ValueError(
            f"{operation} request fields mismatch; expected={sorted(request_fields[operation])}, "
            f"actual={sorted(value)}"
        )
    if operation in {"identify_language", "transcribe", "translate_speech"}:
        if not isinstance(value["observation"], dict):
            raise ValueError(f"{operation} requires an observation object")
    if operation == "translate":
        if not isinstance(value["input"], dict):
            raise ValueError("translate requires a preserved-text input object")
    if operation in {"translate", "translate_speech"}:
        target_language = value["target_language"]
        if not isinstance(target_language, str) or not target_language.strip():
            raise ValueError(f"{operation} target_language must be a non-empty string")
    return value


def validate_observation(
    observation: Any,
    *,
    max_audio_seconds: int,
) -> list[float]:
    if not isinstance(observation, dict):
        raise ValueError("observation must be an object")
    observation_id = observation.get("id")
    if not isinstance(observation_id, str) or not observation_id:
        raise ValueError("observation must have a non-empty id; the Rust caller validates its content address")
    modality = observation.get("modality")
    # Serde's externally tagged enum serializes this Rust variant as {"Audio": {...}}.
    if not isinstance(modality, dict) or set(modality) != {"Audio"} or not isinstance(modality["Audio"], dict):
        raise ValueError("Transformers worker only accepts Audio observations")
    audio = modality["Audio"]
    if set(audio) != {"sample_rate_hz", "channels"}:
        raise ValueError("Audio modality fields do not match the protocol")
    sample_rate_hz = audio.get("sample_rate_hz")
    channels = audio.get("channels")
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, int) or sample_rate_hz != 16000:
        raise ValueError("pilot audio must be mono 16 kHz")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels != 1:
        raise ValueError("pilot audio must be mono 16 kHz")
    samples = observation.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("audio sample array is empty")
    if len(samples) > 16000 * max_audio_seconds:
        raise ValueError(f"audio duration must be in (0, {max_audio_seconds}] seconds")
    checked: list[float] = []
    for index, sample in enumerate(samples):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)):
            raise ValueError(f"audio sample {index} is not numeric")
        number = float(sample)
        if not math.isfinite(number) or abs(number) > 1.0:
            raise ValueError(f"audio sample {index} must be finite and normalized to [-1, 1]")
        checked.append(number)
    return checked


def preserved_text(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise ValueError("decoded transcript must be text")
    if len(text.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
        raise ValueError("decoded transcript exceeds the 1 MiB safety limit")
    # An explicit empty hypothesis is a valid ASR outcome and must count as deletions.
    return {
        "original": text,
        "normalized": "" if text == "" else None,
        "primary_language": LOCALE,
        "script": "Latn",
        "spans": [],
    }


def transcribe_observation(
    observation: dict[str, Any],
    *,
    processor: Any,
    model: Any,
    torch_module: Any,
    numpy_module: Any,
    device: str,
    dtype: Any,
    max_audio_seconds: int,
) -> dict[str, Any]:
    samples = validate_observation(observation, max_audio_seconds=max_audio_seconds)
    waveform = numpy_module.asarray(samples, dtype=numpy_module.float32)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    features = inputs.get("input_features")
    if features is None:
        raise ValueError("Whisper processor did not return input_features")
    features = features.to(device=device, dtype=dtype)
    # BuzzASR's language/task prompt is supplied by the pinned model generation
    # config. Do not inject a generic or contradictory Whisper language prompt.
    with torch_module.inference_mode():
        generated_ids = model.generate(
            features,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
        raise ValueError("processor returned an unexpected decoded batch")
    return preserved_text(decoded[0].strip())


def response_envelope(provider_id: str, request_id: str, result: dict[str, Any] | None = None,
                      error: str | None = None) -> dict[str, Any]:
    if not provider_id or not request_id or ((result is None) == (error is None)):
        raise ValueError("response requires ids and exactly one result or error")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "provider_id": provider_id,
        "request_id": request_id,
        "result": result,
        "error": error,
    }


def handle_request(
    request: dict[str, Any],
    *,
    provider_id: str,
    processor: Any,
    model: Any,
    torch_module: Any,
    numpy_module: Any,
    device: str,
    dtype: Any,
    max_audio_seconds: int,
) -> dict[str, Any]:
    request_id = request["request_id"]
    operation = request["operation"]
    try:
        if operation == "identify_language":
            # A monolingual model's configured locale is not language identification.
            return response_envelope(provider_id, request_id, {"kind": "languages", "value": []})
        if operation == "transcribe":
            text = transcribe_observation(
                request.get("observation"),
                processor=processor,
                model=model,
                torch_module=torch_module,
                numpy_module=numpy_module,
                device=device,
                dtype=dtype,
                max_audio_seconds=max_audio_seconds,
            )
            return response_envelope(provider_id, request_id, {"kind": "text", "value": text})
        if operation == "capabilities":
            raise ValueError("capability enumeration is not implemented by this ASR-only worker")
        if operation in {"translate", "translate_speech"}:
            raise ValueError("translation is not implemented by this ASR-only worker")
        raise ValueError("unsupported operation")
    except Exception as error:
        return response_envelope(provider_id, request_id, error=f"{type(error).__name__}: {error}")


def validate_local_model_snapshot(model_dir: str) -> str:
    if os.path.islink(model_dir) or not os.path.isdir(model_dir):
        raise ValueError("model-dir must be an existing real local model directory, not a symlink")
    root = os.path.realpath(model_dir)
    file_count = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        for name in list(directories):
            candidate = os.path.join(current, name)
            if os.path.islink(candidate):
                raise ValueError("model snapshot contains a symlinked directory")
        directories[:] = [name for name in directories if not os.path.islink(os.path.join(current, name))]
        for name in files:
            candidate = os.path.join(current, name)
            if os.path.islink(candidate) or not os.path.isfile(candidate):
                raise ValueError("model snapshot contains a symlink or non-regular file")
            file_count += 1
    if file_count == 0:
        raise ValueError("model snapshot contains no regular files")
    return root


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_local_model_snapshot(model_dir: str) -> tuple[str, int, int]:
    """Match the scorer's SHA-256 manifest format for the exact startup snapshot."""
    root = Path(model_dir)
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories.sort()
        for dirname in directories:
            candidate = current_path / dirname
            if candidate.is_symlink():
                raise ValueError("model snapshot changed to contain a symlinked directory")
        for filename in sorted(files):
            candidate = current_path / filename
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("model snapshot changed to contain a symlink or non-regular file")
            size = candidate.stat().st_size
            entries.append({
                "path": candidate.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(candidate),
            })
            total_bytes += size
    if not entries:
        raise ValueError("model snapshot contains no regular files")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), len(entries), total_bytes


def load_runtime(model_dir: str, device_choice: str, dtype_choice: str) -> tuple[Any, Any, Any, Any, str, Any, dict[str, Any]]:
    # Set offline flags before importing libraries which may consult the Hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    model_dir = validate_local_model_snapshot(model_dir)
    model_snapshot_sha256, model_file_count, model_bytes = hash_local_model_snapshot(model_dir)
    worker_script = Path(__file__).resolve(strict=True)
    python_executable = Path(sys.executable).resolve(strict=True)
    worker_script_sha256 = sha256_file(worker_script)
    python_executable_sha256 = sha256_file(python_executable)

    import numpy as np
    import torch
    import transformers
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if dtype_choice == "auto":
        dtype_name = "float16" if device == "cuda" else "float32"
    else:
        dtype_name = dtype_choice
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtypes[dtype_name]
    processor = WhisperProcessor.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    # Detect snapshot mutation while the processor/model were being loaded.
    loaded_snapshot_sha256, loaded_file_count, loaded_bytes = hash_local_model_snapshot(model_dir)
    if (loaded_snapshot_sha256, loaded_file_count, loaded_bytes) != (
        model_snapshot_sha256, model_file_count, model_bytes
    ):
        raise ValueError("model snapshot changed while loading; refusing to report provenance")
    model.to(device)
    model.eval()
    metadata = {
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
        "numpy_version": str(np.__version__),
        "device": device,
        "dtype": dtype_name,
        "offline": True,
        "model_snapshot_manifest_sha256": model_snapshot_sha256,
        "model_snapshot_file_count": model_file_count,
        "model_snapshot_bytes": model_bytes,
        "python_executable_sha256": python_executable_sha256,
        "worker_script_sha256": worker_script_sha256,
    }
    return processor, model, torch, np, device, dtype, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="existing local BuzzASR Afrikaans snapshot; no downloads")
    parser.add_argument("--provider-id", default="buzzasr-afrikaans-transformers")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--max-audio-seconds", type=int, default=60)
    args = parser.parse_args()
    if not args.provider_id.strip() or args.max_audio_seconds < 1 or args.max_audio_seconds > 600:
        parser.error("provider-id and max-audio-seconds in [1, 600] are required")
    try:
        processor, model, torch_module, numpy_module, device, dtype, metadata = load_runtime(
            args.model_dir, args.device, args.dtype
        )
    except Exception as error:
        parser.exit(2, f"cannot load offline Transformers runtime: {type(error).__name__}: {error}\n")
    print("SYMTHEA_TRANSFORMERS_RUNTIME " + json.dumps(metadata, sort_keys=True), file=sys.stderr, flush=True)
    stream = sys.stdin.buffer
    out = sys.stdout
    while True:
        line = stream.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            break
        if len(line) > MAX_REQUEST_BYTES:
            parser.exit(2, "request exceeds the 96 MiB input safety limit\n")
        if not line.strip():
            continue
        try:
            request = parse_request(line.decode("utf-8"))
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            parser.exit(2, f"invalid provider request: {error}\n")
        envelope = handle_request(
            request,
            provider_id=args.provider_id,
            processor=processor,
            model=model,
            torch_module=torch_module,
            numpy_module=numpy_module,
            device=device,
            dtype=dtype,
            max_audio_seconds=args.max_audio_seconds,
        )
        encoded = json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_TRANSCRIPT_BYTES * 2:
            parser.exit(2, "provider response exceeds the worker response safety limit\n")
        out.write(encoded + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
