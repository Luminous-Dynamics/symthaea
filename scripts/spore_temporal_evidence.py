#!/usr/bin/env python3
"""Build semantic-time evidence from exact Spore Boot Ecology preview frames.

This tool never renders, modifies, or scores the boot experience. It reads the
existing exact PPM captures plus their serialized BootGenome and emits:

- deterministic samples at sequence start, every stage midpoint, and sequence end;
- exact source-frame hashes and timing error for every requested semantic sample;
- descriptive luminance/occupancy/centroid metrics (never an aesthetic score);
- per-case PPM + PNG contact sheets built from the exact captured pixels.

Only Python's standard library is used so the evidence layer stays disposable.
"""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import zlib

SCHEMA = "spore-temporal-evidence-v1"
BRIGHT_THRESHOLD = 96
VERY_BRIGHT_THRESHOLD = 160
NEAR_BLACK_THRESHOLD = 18
CENTROID_THRESHOLD = 64
GUTTER = 2
CONTACT_COLUMNS = 4


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    stride = width * 3
    scanlines = b"".join(
        b"\x00" + pixels[row * stride : (row + 1) * stride]
        for row in range(height)
    )
    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
    png += png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def read_ppm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise ValueError(f"{path}: expected binary P6 PPM")

    index = 2
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while index < len(data) and data[index] in b" \t\r\n":
            index += 1
        if index >= len(data):
            raise ValueError(f"{path}: truncated PPM header")
        if data[index] == ord("#"):
            while index < len(data) and data[index] not in b"\r\n":
                index += 1
            continue
        start = index
        while index < len(data) and data[index] not in b" \t\r\n":
            index += 1
        tokens.append(data[start:index])

    width, height, max_value = (int(token) for token in tokens)
    if width <= 0 or height <= 0 or max_value != 255:
        raise ValueError(f"{path}: unsupported PPM dimensions/max value")

    # P6 is binary after the max-value separator. Consume the separator itself,
    # not an arbitrary run of whitespace: a valid first pixel byte may be 0x09,
    # 0x0a, 0x0d, or 0x20 and must never be mistaken for more header padding.
    if index >= len(data) or data[index] not in b" \t\r\n":
        raise ValueError(f"{path}: missing PPM header/payload separator")
    if data[index : index + 2] == b"\r\n":
        index += 2
    else:
        index += 1

    pixels = data[index:]
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"{path}: expected {expected} RGB bytes, found {len(pixels)}")
    return width, height, pixels


def write_ppm(path: Path, width: int, height: int, pixels: bytes) -> None:
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"contact sheet expected {expected} bytes, got {len(pixels)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def frame_elapsed_ms(frame_index: int, fps: int, duration_ms: int) -> int:
    return min((frame_index * 1_000) // fps, duration_ms)


def semantic_targets(genome: dict, duration_ms: int) -> list[dict]:
    stages = genome.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("boot-genome.json must contain a non-empty stages array")

    targets: list[dict] = [
        {
            "sample_key": "00-sequence-start",
            "role": "sequence-start",
            "stage_index": None,
            "stage_kind": None,
            "stage_intensity": None,
            "target_elapsed_ms": 0,
        }
    ]

    cursor = 0
    for stage_index, stage in enumerate(stages):
        kind = stage.get("kind")
        stage_duration = stage.get("duration_ms")
        intensity = stage.get("intensity")
        if not isinstance(kind, str):
            raise ValueError(f"stage {stage_index}: kind must be a string")
        if not isinstance(stage_duration, int) or stage_duration < 0:
            raise ValueError(f"stage {stage_index}: invalid duration_ms")
        if not isinstance(intensity, (int, float)) or not math.isfinite(float(intensity)):
            raise ValueError(f"stage {stage_index}: invalid intensity")

        midpoint = cursor + stage_duration // 2
        targets.append(
            {
                "sample_key": f"{stage_index + 1:02d}-{kind.lower()}-mid",
                "role": "stage-midpoint",
                "stage_index": stage_index,
                "stage_kind": kind,
                "stage_intensity": float(intensity),
                "target_elapsed_ms": min(midpoint, duration_ms),
            }
        )
        cursor += stage_duration

    if cursor != duration_ms:
        raise ValueError(
            f"stage duration sum {cursor} does not equal preview duration {duration_ms}"
        )

    targets.append(
        {
            "sample_key": f"{len(stages) + 1:02d}-sequence-final",
            "role": "sequence-final",
            "stage_index": None,
            "stage_kind": None,
            "stage_intensity": None,
            "target_elapsed_ms": duration_ms,
        }
    )
    return targets


def choose_frame(target_ms: int, frame_count: int, fps: int, duration_ms: int) -> tuple[int, int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    best_index = min(
        range(frame_count),
        key=lambda index: (
            abs(frame_elapsed_ms(index, fps, duration_ms) - target_ms),
            index,
        ),
    )
    return best_index, frame_elapsed_ms(best_index, fps, duration_ms)


def luma_u8(red: int, green: int, blue: int) -> int:
    return (2_126 * red + 7_152 * green + 722 * blue + 5_000) // 10_000


def frame_metrics(width: int, height: int, pixels: bytes) -> dict:
    histogram = [0] * 256
    total_luma = 0
    bright = 0
    very_bright = 0
    non_near_black = 0
    luminous_weight = 0
    weighted_x = 0
    weighted_y = 0

    pixel_count = width * height
    for pixel_index in range(pixel_count):
        offset = pixel_index * 3
        luma = luma_u8(pixels[offset], pixels[offset + 1], pixels[offset + 2])
        histogram[luma] += 1
        total_luma += luma
        if luma >= BRIGHT_THRESHOLD:
            bright += 1
        if luma >= VERY_BRIGHT_THRESHOLD:
            very_bright += 1
        if luma > NEAR_BLACK_THRESHOLD:
            non_near_black += 1
        if luma >= CENTROID_THRESHOLD:
            x = pixel_index % width
            y = pixel_index // width
            luminous_weight += luma
            weighted_x += x * luma
            weighted_y += y * luma

    p95_rank = max(1, math.ceil(pixel_count * 0.95))
    cumulative = 0
    p95 = 0
    for value, count in enumerate(histogram):
        cumulative += count
        if cumulative >= p95_rank:
            p95 = value
            break

    if luminous_weight:
        centroid_x = weighted_x / luminous_weight / max(1, width - 1)
        centroid_y = weighted_y / luminous_weight / max(1, height - 1)
    else:
        centroid_x = None
        centroid_y = None

    return {
        "mean_luma": round(total_luma / pixel_count, 4),
        "p95_luma": p95,
        "bright_fraction": round(bright / pixel_count, 8),
        "very_bright_fraction": round(very_bright / pixel_count, 8),
        "non_near_black_fraction": round(non_near_black / pixel_count, 8),
        "luminous_centroid_x": None if centroid_x is None else round(centroid_x, 6),
        "luminous_centroid_y": None if centroid_y is None else round(centroid_y, 6),
        "thresholds": {
            "near_black_luma": NEAR_BLACK_THRESHOLD,
            "centroid_luma": CENTROID_THRESHOLD,
            "bright_luma": BRIGHT_THRESHOLD,
            "very_bright_luma": VERY_BRIGHT_THRESHOLD,
        },
    }


def contact_sheet(frames: list[tuple[int, int, bytes]], columns: int = CONTACT_COLUMNS) -> tuple[int, int, bytes]:
    if not frames:
        raise ValueError("contact sheet requires at least one frame")
    width = frames[0][0]
    height = frames[0][1]
    if any(frame_width != width or frame_height != height for frame_width, frame_height, _ in frames):
        raise ValueError("contact sheet frames must have identical dimensions")

    columns = max(1, min(columns, len(frames)))
    rows = math.ceil(len(frames) / columns)
    sheet_width = columns * width + (columns - 1) * GUTTER
    sheet_height = rows * height + (rows - 1) * GUTTER
    sheet = bytearray(sheet_width * sheet_height * 3)

    for index, (_, _, pixels) in enumerate(frames):
        column = index % columns
        row = index // columns
        origin_x = column * (width + GUTTER)
        origin_y = row * (height + GUTTER)
        for y in range(height):
            source_start = y * width * 3
            source_end = source_start + width * 3
            destination_start = ((origin_y + y) * sheet_width + origin_x) * 3
            sheet[destination_start : destination_start + width * 3] = pixels[
                source_start:source_end
            ]

    return sheet_width, sheet_height, bytes(sheet)


def build_case(root: Path, case: dict) -> dict:
    name = case["name"]
    frames_dir = root / case["frames"]
    preview_manifest = json.loads((frames_dir / "preview-manifest.json").read_text())
    genome = json.loads((frames_dir / "boot-genome.json").read_text())

    if preview_manifest.get("schema") != "spore-boot-preview-v1":
        raise ValueError(f"{name}: unsupported preview manifest schema")
    fps = int(preview_manifest["fps"])
    duration_ms = int(preview_manifest["duration_ms"])
    manifest_frame_count = int(preview_manifest["frame_count"])
    ppm_frames = sorted(frames_dir.glob("frame-*.ppm"))
    if len(ppm_frames) != manifest_frame_count:
        raise ValueError(
            f"{name}: manifest frame_count={manifest_frame_count}, files={len(ppm_frames)}"
        )

    targets = semantic_targets(genome, duration_ms)
    samples = []
    sheet_frames = []
    max_timing_error = 0

    for target in targets:
        frame_index, actual_ms = choose_frame(
            target["target_elapsed_ms"], manifest_frame_count, fps, duration_ms
        )
        frame_path = ppm_frames[frame_index]
        width, height, pixels = read_ppm(frame_path)
        timing_error = abs(actual_ms - target["target_elapsed_ms"])
        max_timing_error = max(max_timing_error, timing_error)
        samples.append(
            {
                **target,
                "frame_index": frame_index,
                "actual_elapsed_ms": actual_ms,
                "timing_error_ms": timing_error,
                "exact_semantic_time": timing_error == 0,
                "frame": frame_path.relative_to(root).as_posix(),
                "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
                "metrics": frame_metrics(width, height, pixels),
            }
        )
        sheet_frames.append((width, height, pixels))

    evidence_dir = root / name / "temporal"
    sheet_width, sheet_height, sheet_pixels = contact_sheet(sheet_frames)
    ppm_path = evidence_dir / "semantic-contact-sheet.ppm"
    png_path = evidence_dir / "semantic-contact-sheet.png"
    write_ppm(ppm_path, sheet_width, sheet_height, sheet_pixels)
    write_png(png_path, sheet_width, sheet_height, sheet_pixels)

    final_sample = samples[-1]
    return {
        "name": name,
        "family": case.get("family"),
        "cue": case.get("cue"),
        "duration_ms": duration_ms,
        "fps": fps,
        "source_frame_count": manifest_frame_count,
        "sample_count": len(samples),
        "max_timing_error_ms": max_timing_error,
        "terminal_frame_exact": bool(final_sample["exact_semantic_time"]),
        "terminal_timing_error_ms": final_sample["timing_error_ms"],
        "contact_sheet_ppm": ppm_path.relative_to(root).as_posix(),
        "contact_sheet_png": png_path.relative_to(root).as_posix(),
        "samples": samples,
    }


def build(root: Path) -> dict:
    matrix_path = root / "matrix-manifest.json"
    matrix = json.loads(matrix_path.read_text())
    if matrix.get("schema") != "spore-boot-preview-matrix-v1":
        raise ValueError("unsupported matrix manifest schema")

    cases = [build_case(root, case) for case in matrix["cases"]]
    report = {
        "schema": SCHEMA,
        "renderer": matrix.get("renderer"),
        "width": matrix.get("width"),
        "height": matrix.get("height"),
        "source_capture_fps": matrix.get("fps"),
        "policy": {
            "purpose": "descriptive-temporal-review-not-aesthetic-scoring",
            "sample_rule": "sequence-start + every BootStage midpoint + sequence-final",
            "selection": "nearest existing exact renderer frame; ties choose earlier frame",
            "metrics_are_scores": False,
            "contact_sheet_pixels": "exact captured PPM pixels; black gutters only",
        },
        "coverage": {
            "case_count": len(cases),
            "terminal_exact_case_count": sum(case["terminal_frame_exact"] for case in cases),
            "max_timing_error_ms": max((case["max_timing_error_ms"] for case in cases), default=0),
        },
        "cases": cases,
    }
    output = root / "temporal-evidence.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return report


def synthetic_ppm(path: Path, width: int, height: int, value: int) -> None:
    pixels = bytes([value, value // 2, 0] * (width * height))
    write_ppm(path, width, height, pixels)


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="spore-temporal-evidence-") as directory:
        root = Path(directory)
        frames = root / "case-a" / "frames"
        frames.mkdir(parents=True)
        genome = {
            "stages": [
                {"kind": "Grow", "duration_ms": 1_000, "intensity": 0.6},
                {"kind": "Handoff", "duration_ms": 1_000, "intensity": 0.4},
            ]
        }
        (frames / "boot-genome.json").write_text(json.dumps(genome))
        (frames / "preview-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "spore-boot-preview-v1",
                    "fps": 2,
                    "duration_ms": 2_000,
                    "frame_count": 5,
                }
            )
        )
        # The first frame begins with red=0x0a. A binary-safe P6 parser must
        # preserve this as pixel data rather than treating it as header whitespace.
        for index, value in enumerate([10, 40, 80, 120, 180]):
            synthetic_ppm(frames / f"frame-{index:05}.ppm", 4, 2, value)
        width, height, first_pixels = read_ppm(frames / "frame-00000.ppm")
        assert (width, height) == (4, 2)
        assert first_pixels[0] == 10

        (root / "matrix-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "spore-boot-preview-matrix-v1",
                    "renderer": "synthetic",
                    "width": 4,
                    "height": 2,
                    "fps": 2,
                    "cases": [
                        {
                            "name": "case-a",
                            "family": "Synthetic",
                            "cue": "Starting",
                            "frames": "case-a/frames",
                        }
                    ],
                }
            )
        )

        report = build(root)
        assert report["schema"] == SCHEMA
        assert report["coverage"]["case_count"] == 1
        case = report["cases"][0]
        assert case["terminal_frame_exact"] is True
        assert [sample["target_elapsed_ms"] for sample in case["samples"]] == [
            0,
            500,
            1_500,
            2_000,
        ]
        assert [sample["frame_index"] for sample in case["samples"]] == [0, 1, 3, 4]
        assert all(sample["timing_error_ms"] == 0 for sample in case["samples"])
        assert (root / case["contact_sheet_ppm"]).is_file()
        assert (root / case["contact_sheet_png"]).is_file()

        # Reproduce the current endpoint-omission shape: four frames at 2 fps
        # over a 2 s sequence cover 0, 0.5, 1.0, 1.5 s but not 2.0 s.
        (frames / "frame-00004.ppm").unlink()
        preview = json.loads((frames / "preview-manifest.json").read_text())
        preview["frame_count"] = 4
        (frames / "preview-manifest.json").write_text(json.dumps(preview))
        report = build(root)
        case = report["cases"][0]
        assert case["terminal_frame_exact"] is False
        assert case["terminal_timing_error_ms"] == 500

    print("spore_temporal_evidence self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, help="Spore boot preview matrix root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.root is None:
        parser.error("root is required unless --self-test is used")
    build(args.root)


if __name__ == "__main__":
    main()
