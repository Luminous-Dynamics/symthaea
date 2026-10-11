#!/usr/bin/env python3
"""Sanity-check exact Spore renderer galleries without encoding taste as policy.

This linter is intentionally conservative. It does not score whether a frame is
"beautiful". It catches evidence failures that file-count checks cannot:

- a lifecycle/install scenario that renders effectively blank;
- a gallery that collapses to one repeated frame;
- install-route signatures that accidentally become pixel-identical.

The resulting `visual-lint.json` is written into each evidence root and is later
covered by the SHA-256 visual-evidence seal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

ACTIVE_CHANNEL = 8
MIN_ACTIVE_RATIO = 0.0015


@dataclass(frozen=True)
class FrameStats:
    path: str
    sha256: str
    width: int
    height: int
    active_ratio: float
    mean_luma: float
    max_luma: float


def read_ppm(path: Path) -> tuple[int, int, bytes]:
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        raise ValueError(f"{path}: expected P6 PPM")

    cursor = 2
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while cursor < len(data) and chr(data[cursor]).isspace():
            cursor += 1
        if cursor < len(data) and data[cursor] == ord("#"):
            while cursor < len(data) and data[cursor] not in b"\r\n":
                cursor += 1
            continue
        start = cursor
        while cursor < len(data) and not chr(data[cursor]).isspace():
            cursor += 1
        if start == cursor:
            raise ValueError(f"{path}: truncated PPM header")
        tokens.append(data[start:cursor])

    width, height, max_value = (int(token) for token in tokens)
    if width <= 0 or height <= 0 or max_value != 255:
        raise ValueError(f"{path}: unsupported PPM geometry/max value")
    while cursor < len(data) and chr(data[cursor]).isspace():
        cursor += 1
    pixels = data[cursor:]
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError(f"{path}: expected {expected} RGB bytes, got {len(pixels)}")
    return width, height, pixels


def frame_stats(root: Path, path: Path) -> FrameStats:
    width, height, pixels = read_ppm(path)
    pixel_count = width * height
    active = 0
    luma_sum = 0.0
    max_luma = 0.0
    for index in range(0, len(pixels), 3):
        r, g, b = pixels[index], pixels[index + 1], pixels[index + 2]
        if max(r, g, b) > ACTIVE_CHANNEL:
            active += 1
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        luma_sum += luma
        max_luma = max(max_luma, luma)
    return FrameStats(
        path=path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        width=width,
        height=height,
        active_ratio=active / pixel_count,
        mean_luma=luma_sum / pixel_count,
        max_luma=max_luma,
    )


def require_visible(label: str, frames: list[FrameStats]) -> None:
    if not frames:
        raise AssertionError(f"{label}: no exact PPM frames")
    strongest = max(frames, key=lambda frame: frame.active_ratio)
    if strongest.active_ratio < MIN_ACTIVE_RATIO:
        raise AssertionError(
            f"{label}: effectively blank; strongest active ratio "
            f"{strongest.active_ratio:.6f} < {MIN_ACTIVE_RATIO:.6f}"
        )
    if strongest.max_luma < 18.0:
        raise AssertionError(
            f"{label}: insufficient dynamic range; max luma={strongest.max_luma:.2f}"
        )


def lint_boot(root: Path) -> dict:
    manifest = json.loads((root / "matrix-manifest.json").read_text())
    summaries = []
    representative_hashes: set[str] = set()
    for case in manifest["cases"]:
        label = case["name"]
        frames = [frame_stats(root, path) for path in sorted((root / case["frames"]).glob("frame-*.ppm"))]
        require_visible(label, frames)
        representative = max(frames, key=lambda frame: frame.active_ratio)
        representative_hashes.add(representative.sha256)
        summaries.append(
            {
                "case": label,
                "frame_count": len(frames),
                "max_active_ratio": representative.active_ratio,
                "representative": representative.path,
                "representative_sha256": representative.sha256,
            }
        )

    # The matrix exists specifically to prove state-dependent variation. Avoid a
    # brittle one-hash-per-case snapshot while still detecting wholesale visual
    # collapse to one or two outputs.
    minimum_unique = max(4, len(summaries) // 3)
    if len(representative_hashes) < minimum_unique:
        raise AssertionError(
            f"boot matrix visual diversity collapsed: {len(representative_hashes)} "
            f"unique representative frames < {minimum_unique}"
        )
    return {
        "kind": "boot-lifecycle",
        "scenario_count": len(summaries),
        "unique_representative_frames": len(representative_hashes),
        "scenarios": summaries,
    }


def lint_inoculation(root: Path) -> dict:
    manifest = json.loads((root / "inoculation-manifest.json").read_text())
    entries = manifest["phases"]
    summaries = []
    all_hashes: set[str] = set()
    by_lifecycle: dict[str, dict[str, str]] = {}

    for entry in entries:
        label = entry["phase"]
        paths = [root / relative for relative in entry["frames"]]
        frames = [frame_stats(root, path) for path in paths]
        require_visible(label, frames)
        all_hashes.update(frame.sha256 for frame in frames)
        representative = max(frames, key=lambda frame: frame.active_ratio)
        summaries.append(
            {
                "scenario": label,
                "frame_count": len(frames),
                "max_active_ratio": representative.active_ratio,
                "representative": representative.path,
                "representative_sha256": representative.sha256,
            }
        )

        path_label = entry.get("path")
        lifecycle = entry.get("lifecycle_phase")
        if path_label and lifecycle:
            by_lifecycle.setdefault(lifecycle, {})[path_label] = representative.sha256

    if len(all_hashes) < max(4, len(entries) // 2):
        raise AssertionError(
            f"inoculation gallery visual diversity collapsed: {len(all_hashes)} unique frames"
        )

    route_checks = []
    if by_lifecycle:
        expected_paths = int(manifest.get("paths", 0))
        for lifecycle, route_hashes in sorted(by_lifecycle.items()):
            if expected_paths and len(route_hashes) != expected_paths:
                raise AssertionError(
                    f"{lifecycle}: expected {expected_paths} route signatures, got {len(route_hashes)}"
                )
            unique = len(set(route_hashes.values()))
            if unique != len(route_hashes):
                raise AssertionError(
                    f"{lifecycle}: install-route signatures became pixel-identical "
                    f"({unique}/{len(route_hashes)} unique)"
                )
            route_checks.append(
                {
                    "lifecycle_phase": lifecycle,
                    "routes": len(route_hashes),
                    "unique_route_frames": unique,
                }
            )

    return {
        "kind": "inoculation-paths" if by_lifecycle else "inoculation-phases",
        "scenario_count": len(summaries),
        "unique_frames": len(all_hashes),
        "route_checks": route_checks,
        "scenarios": summaries,
    }


def lint_root(root: Path) -> dict:
    if (root / "matrix-manifest.json").is_file():
        report = lint_boot(root)
    elif (root / "inoculation-manifest.json").is_file():
        report = lint_inoculation(root)
    else:
        raise AssertionError(f"{root}: unrecognised Spore preview root")
    return {
        "schema": "spore-visual-lint-v1",
        "policy": {
            "active_channel_threshold": ACTIVE_CHANNEL,
            "minimum_active_ratio": MIN_ACTIVE_RATIO,
            "purpose": "sanity-and-semantic-diversity-not-aesthetic-scoring",
        },
        **report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    for root in args.roots:
        report = lint_root(root)
        output = root / "visual-lint.json"
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(
            f"{root}: {report['kind']} scenarios={report['scenario_count']} "
            f"lint=ok"
        )


if __name__ == "__main__":
    main()
