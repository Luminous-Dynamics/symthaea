#!/usr/bin/env python3
"""Generate MEL-BENCH-002B reference outputs from an exact MusPy source file.

The runner deliberately does not copy or reimplement MusPy metric functions. It
loads the audited ``muspy/metrics/metrics.py`` directly from a checkout whose
Git revision and metrics-file blob identity are verified first. Only the
``muspy.music.Music`` type import is stubbed; the selected metric functions are
duck-typed and execute their original external source bodies unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BUNDLE_VERSION = "melothaea-muspy-pitch-parity-v1"
MUSPY_REVISION = "2e1dc660dde6974c1693147f528d7437019d9580"
METRICS_BLOB = "b428fe331f7a8b64380a7d9a12f028e4f12c3033"
EXECUTION_MODE = "ExactMetricsFileWithTypeStub"


@dataclass(frozen=True)
class Note:
    pitch: int
    time: int = 0
    duration: int = 1

    @property
    def end(self) -> int:
        return self.time + self.duration


@dataclass
class Track:
    notes: list[Note]
    is_drum: bool = False

    def get_end_time(self) -> int:
        return max((note.end for note in self.notes), default=0)


@dataclass
class Music:
    tracks: list[Track]
    resolution: int = 24


FIXTURES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("c-major-octave", (60, 64, 67, 72)),
    ("chromatic-12", tuple(range(60, 72))),
    ("extreme-midi-range", (0, 127)),
    ("octave-duplicate", (60, 72)),
    ("singleton-c4", (60,)),
    ("weighted-c-major-outlier", (60, 60, 61, 64)),
)


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True
    ).strip()


def verify_checkout(root: Path) -> Path:
    if git_output(root, "rev-parse", "HEAD") != MUSPY_REVISION:
        raise RuntimeError("MusPy checkout is not the frozen MEL-BENCH-002B revision")
    metrics_path = root / "muspy" / "metrics" / "metrics.py"
    if not metrics_path.is_file():
        raise RuntimeError(f"missing audited metrics source: {metrics_path}")
    blob = git_output(root, "hash-object", "muspy/metrics/metrics.py")
    if blob != METRICS_BLOB:
        raise RuntimeError(
            f"unexpected metrics.py blob: expected {METRICS_BLOB}, found {blob}"
        )
    return metrics_path


def load_exact_metrics(root: Path, metrics_path: Path):
    # Supply only the type imported by metrics.py. The benchmarked function
    # bodies themselves come directly from the exact external source file.
    package = types.ModuleType("muspy")
    package.__path__ = [str(root / "muspy")]
    metrics_package = types.ModuleType("muspy.metrics")
    metrics_package.__path__ = [str(root / "muspy" / "metrics")]
    music_module = types.ModuleType("muspy.music")
    music_module.Music = Music
    sys.modules["muspy"] = package
    sys.modules["muspy.metrics"] = metrics_package
    sys.modules["muspy.music"] = music_module

    spec = importlib.util.spec_from_file_location(
        "muspy.metrics.metrics", metrics_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create import spec for exact MusPy metrics file")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def music_from_midis(midis: tuple[int, ...]) -> Music:
    if not midis:
        raise ValueError("MEL-BENCH-002B reference fixtures must be non-empty")
    if any(pitch < 0 or pitch > 127 for pitch in midis):
        raise ValueError("MEL-BENCH-002B fixture pitch outside MIDI range")
    return Music(tracks=[Track(notes=[Note(pitch=pitch) for pitch in midis])])


def generate_bundle(metrics) -> dict:
    fixtures = []
    for fixture_id, midis in FIXTURES:
        music = music_from_midis(midis)
        external = {
            "n_pitches_used": int(metrics.n_pitches_used(music)),
            "n_pitch_classes_used": int(metrics.n_pitch_classes_used(music)),
            "pitch_range": int(metrics.pitch_range(music)),
            "pitch_entropy": float(metrics.pitch_entropy(music)),
            "pitch_class_entropy": float(metrics.pitch_class_entropy(music)),
            "scale_consistency": float(metrics.scale_consistency(music)),
            "c_major_pitch_in_scale_rate": float(
                metrics.pitch_in_scale_rate(music, 0, "major")
            ),
        }
        if not all(np.isfinite(value) for value in external.values()):
            raise RuntimeError(f"non-finite external output for fixture {fixture_id}")
        fixtures.append(
            {
                "fixture_id": fixture_id,
                "midis": list(midis),
                "external": external,
            }
        )

    return {
        "bundle_version": BUNDLE_VERSION,
        "source_revision": MUSPY_REVISION,
        "metrics_source_blob": METRICS_BLOB,
        "execution_mode": EXECUTION_MODE,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "fixtures": fixtures,
    }


def self_test(bundle: dict) -> None:
    by_id = {fixture["fixture_id"]: fixture for fixture in bundle["fixtures"]}
    c_major = by_id["c-major-octave"]["external"]
    assert c_major["n_pitches_used"] == 4
    assert c_major["n_pitch_classes_used"] == 3
    assert c_major["pitch_range"] == 12
    assert abs(c_major["pitch_entropy"] - 2.0) < 1e-15
    assert abs(c_major["pitch_class_entropy"] - 1.5) < 1e-15
    assert abs(c_major["scale_consistency"] - 1.0) < 1e-15
    assert abs(c_major["c_major_pitch_in_scale_rate"] - 1.0) < 1e-15

    chromatic = by_id["chromatic-12"]["external"]
    assert abs(chromatic["scale_consistency"] - (7.0 / 12.0)) < 1e-15
    assert abs(chromatic["pitch_class_entropy"] - np.log2(12.0)) < 1e-15


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muspy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    root = args.muspy_root.resolve()
    metrics_path = verify_checkout(root)
    metrics = load_exact_metrics(root, metrics_path)
    bundle = generate_bundle(metrics)
    if args.self_test:
        self_test(bundle)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(bundle, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
