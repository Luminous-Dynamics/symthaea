#!/usr/bin/env python3
"""Select frozen semantic review frames from exact Spore preview matrices.

This is evidence tooling, not a renderer and not a beauty metric. It consumes the
PPM frame stream already emitted by ``spore_boot_preview_matrix`` and copies the
nearest frame that is actually inside each requested semantic stage. Every copied
frame is SHA-256 bound to its source path and the output manifest records timing
quantization/error explicitly instead of pretending an unavailable exact sample
was rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "docs/design/SPORE_VISUAL_REVIEW_PROTOCOL_V1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def safe_component(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)
    cleaned = cleaned.strip(".-")
    return cleaned or "stage"


def frame_elapsed_ms(index: int, fps: int) -> int:
    return (index * 1000) // fps


def ceil_frame_index(elapsed_ms: int, fps: int) -> int:
    return (elapsed_ms * fps + 999) // 1000


def floor_frame_index(elapsed_ms: int, fps: int) -> int:
    return (elapsed_ms * fps) // 1000


def nearest_stage_frame(
    *,
    cursor_ms: int,
    duration_ms: int,
    stage_index: int,
    requested_progress: float,
    fps: int,
    frame_count: int,
) -> tuple[int, int, float] | None:
    if duration_ms <= 0 or frame_count <= 0:
        return None

    # EcologyRenderer::frame_state assigns the exact boundary cursor of every
    # non-first stage to the preceding stage (`elapsed <= previous_end`). The
    # earliest representable millisecond inside a later stage is therefore +1.
    lower_ms = cursor_ms if stage_index == 0 else cursor_ms + 1
    upper_ms = cursor_ms + duration_ms

    lower_index = ceil_frame_index(lower_ms, fps)
    upper_index = min(floor_frame_index(upper_ms, fps), frame_count - 1)
    if lower_index > upper_index:
        return None

    requested_ms = cursor_ms + round(duration_ms * requested_progress)
    requested_index = round(requested_ms * fps / 1000)
    frame_index = min(max(requested_index, lower_index), upper_index)
    elapsed_ms = frame_elapsed_ms(frame_index, fps)
    resolved_progress = (elapsed_ms - cursor_ms) / duration_ms
    resolved_progress = min(max(resolved_progress, 0.0), 1.0)
    return frame_index, elapsed_ms, resolved_progress


def protocol_progress(protocol: dict[str, Any], mode: str) -> list[float]:
    field = "contact_sheet_progress" if mode == "contact" else "matrix_progress"
    values = protocol.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"protocol field {field!r} must be a non-empty array")
    progress = [float(value) for value in values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in progress):
        raise ValueError(f"protocol field {field!r} contains invalid progress")
    if any(a >= b for a, b in zip(progress, progress[1:])):
        raise ValueError(f"protocol field {field!r} must be strictly increasing")
    return progress


def sample_case(
    *,
    matrix_dir: Path,
    case: dict[str, Any],
    out_dir: Path,
    progress_points: list[float],
) -> dict[str, Any]:
    name = str(case["name"])
    frames_rel = Path(str(case["frames"]))
    frames_dir = matrix_dir / frames_rel
    preview_manifest_path = frames_dir / "preview-manifest.json"
    genome_path = frames_dir / "boot-genome.json"
    preview = read_json(preview_manifest_path)
    genome = read_json(genome_path)

    fps = int(preview["fps"])
    frame_count = int(preview["frame_count"])
    stages = genome.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{genome_path}: stages must be an array")

    case_out = out_dir / safe_component(name)
    case_out.mkdir(parents=True, exist_ok=True)
    cursor_ms = 0
    sampled_stages: list[dict[str, Any]] = []

    for stage_index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, dict):
            raise ValueError(f"{genome_path}: stage {stage_index} must be an object")
        kind = str(raw_stage.get("kind", f"stage-{stage_index}"))
        duration_ms = int(raw_stage.get("duration_ms", 0))
        stage_out = case_out / f"stage-{stage_index:02d}-{safe_component(kind)}"
        stage_out.mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []

        for sample_index, requested_progress in enumerate(progress_points):
            selected = nearest_stage_frame(
                cursor_ms=cursor_ms,
                duration_ms=duration_ms,
                stage_index=stage_index,
                requested_progress=requested_progress,
                fps=fps,
                frame_count=frame_count,
            )
            if selected is None:
                samples.append(
                    {
                        "sample_index": sample_index,
                        "requested_progress": requested_progress,
                        "status": "no-frame-inside-stage",
                    }
                )
                continue

            frame_index, elapsed_ms, resolved_progress = selected
            source = frames_dir / f"frame-{frame_index:05}.ppm"
            if not source.is_file():
                raise FileNotFoundError(f"missing exact preview frame: {source}")
            label = f"p-{round(requested_progress * 1000):04d}-frame-{frame_index:05}.ppm"
            destination = stage_out / label
            shutil.copyfile(source, destination)
            samples.append(
                {
                    "sample_index": sample_index,
                    "requested_progress": requested_progress,
                    "resolved_progress": round(resolved_progress, 9),
                    "progress_error": round(resolved_progress - requested_progress, 9),
                    "elapsed_ms": elapsed_ms,
                    "source_frame_index": frame_index,
                    "source": str(source.relative_to(matrix_dir)),
                    "selected": str(destination.relative_to(out_dir)),
                    "sha256": sha256(source),
                    "status": "selected",
                }
            )

        sampled_stages.append(
            {
                "stage_index": stage_index,
                "kind": kind,
                "duration_ms": duration_ms,
                "cursor_ms": cursor_ms,
                "samples": samples,
            }
        )
        cursor_ms += max(duration_ms, 0)

    return {
        "name": name,
        "source_preview_manifest": str(preview_manifest_path.relative_to(matrix_dir)),
        "source_preview_manifest_sha256": sha256(preview_manifest_path),
        "source_genome": str(genome_path.relative_to(matrix_dir)),
        "source_genome_sha256": sha256(genome_path),
        "fps": fps,
        "frame_count": frame_count,
        "stages": sampled_stages,
    }


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("contact", "matrix"), default="contact")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    matrix_dir = args.matrix_dir.resolve()
    out_dir = args.out.resolve()
    protocol_path = args.protocol.resolve()
    matrix_manifest_path = matrix_dir / "matrix-manifest.json"
    matrix = read_json(matrix_manifest_path)
    protocol = read_json(protocol_path)
    if protocol.get("schema") != "spore.visual.review-protocol.v1":
        raise ValueError(f"unsupported review protocol schema: {protocol.get('schema')!r}")
    progress_points = protocol_progress(protocol, args.mode)
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{matrix_manifest_path}: cases must be an array")

    out_dir.mkdir(parents=True, exist_ok=True)
    sampled_cases = [
        sample_case(
            matrix_dir=matrix_dir,
            case=case,
            out_dir=out_dir,
            progress_points=progress_points,
        )
        for case in cases
    ]

    manifest = {
        "schema": "spore.visual.temporal-samples.v1",
        "mode": args.mode,
        "semantic_basis": protocol.get("semantic_basis"),
        "progress_points": progress_points,
        "source_matrix_manifest": str(matrix_manifest_path),
        "source_matrix_manifest_sha256": sha256(matrix_manifest_path),
        "review_protocol": str(protocol_path),
        "review_protocol_sha256": sha256(protocol_path),
        "selection_rule": "nearest existing exact frame constrained to requested BootStageKind interval",
        "quantization_policy": "record resolved progress and error; never synthesize a missing exact frame",
        "cases": sampled_cases,
    }
    manifest_path = out_dir / "temporal-samples-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
