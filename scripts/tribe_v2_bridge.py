#!/usr/bin/env python3
"""
TRIBE v2 bridge for Symthaea neural-validation research.

This bridge is intentionally provenance-strict:

* Real TRIBE v2 inference emits the native fsaverage5 cortical surface.
  It does NOT truncate or reinterpret the first 360 vertices as Glasser parcels.
* Mock mode is explicit and emits a synthetic-only schema that the current
  empirical Rust loader cannot mistake for external fMRI predictions.
* If TRIBE v2 is unavailable, real mode fails closed. It never falls back to
  synthetic data.

The fsaverage5 -> atlas -> Symthaea12 mapping belongs in the next qualified
bridge layer. Until that mapping exists, real output remains in fsaverage5.

Usage:
    python scripts/tribe_v2_bridge.py --stimulus video.mp4 --output result.json
    python scripts/tribe_v2_bridge.py --stimulus-dir data/stimuli --output results/
    python scripts/tribe_v2_bridge.py --mock --output mock_result.json

Requirements for real inference:
    numpy
    TRIBE v2 from https://github.com/facebookresearch/tribev2

References:
    - d'Ascoli et al. (2026), TRIBE v2.
    - Official API: from tribev2 import TribeModel
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


MOCK_BASE = {
    "Visual": 0.70,
    "Auditory": 0.50,
    "Language": 0.30,
    "Motor": 0.15,
    "Sensory": 0.10,
    "Prefrontal": 0.35,
    "Memory": 0.25,
    "Emotional": 0.30,
    "Social": 0.20,
    "Executive": 0.25,
    "Creative": 0.15,
    "Integration": 0.40,
}

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEXT_SUFFIXES = {".txt", ".md"}


def _stable_seed(text: str) -> int:
    """Derive a process-independent 64-bit RNG seed from text."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a NumPy/Torch-like prediction tensor to a CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _stimulus_argument(stimulus_path: str) -> dict[str, str]:
    """Map a file extension to TribeModel.get_events_dataframe arguments."""
    suffix = Path(stimulus_path).suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return {"video_path": stimulus_path}
    if suffix in AUDIO_SUFFIXES:
        return {"audio_path": stimulus_path}
    if suffix in TEXT_SUFFIXES:
        return {"text_path": stimulus_path}
    raise ValueError(
        f"Unsupported stimulus type {suffix!r}; expected video, audio, or text input"
    )


def run_tribe_v2(
    stimulus_path: str,
    model_path: Optional[str] = None,
    cache_folder: Optional[str] = None,
) -> dict[str, Any]:
    """Run released TRIBE v2 and return native fsaverage5 predictions.

    The released model predicts ``(n_timesteps, n_vertices)`` on fsaverage5.
    We temporally average for this bridge artifact but preserve the native
    cortical coordinate system. No atlas interpretation is attempted here.

    Raises:
        RuntimeError: if TRIBE v2 is unavailable or returns an invalid shape.
        ValueError: if the stimulus type is unsupported.
    """
    try:
        from tribev2 import TribeModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "TRIBE v2 inference was requested, but the 'tribev2' package is not "
            "available. Refusing synthetic fallback; use --mock explicitly for "
            "fixture data."
        ) from exc

    model_id = model_path or "facebook/tribev2"
    kwargs: dict[str, Any] = {}
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    model = TribeModel.from_pretrained(model_id, **kwargs)

    events = model.get_events_dataframe(**_stimulus_argument(stimulus_path))
    predictions, _segments = model.predict(events=events)
    predicted_bold = _to_numpy(predictions)

    if predicted_bold.ndim != 2:
        raise RuntimeError(
            "TRIBE v2 returned an unexpected prediction rank: "
            f"shape={predicted_bold.shape!r}; expected (timesteps, vertices)"
        )
    if predicted_bold.shape[0] == 0 or predicted_bold.shape[1] == 0:
        raise RuntimeError(
            f"TRIBE v2 returned an empty prediction array: {predicted_bold.shape!r}"
        )

    mean_surface = predicted_bold.mean(axis=0, dtype=np.float64)
    return {
        "surface_activations": [float(v) for v in mean_surface],
        "n_timesteps": int(predicted_bold.shape[0]),
        "n_vertices": int(predicted_bold.shape[1]),
    }


def generate_mock_activations(stimulus_path: str = "") -> dict[str, float]:
    """Generate deterministic synthetic 12-region fixture activations."""
    rng = np.random.default_rng(_stable_seed(stimulus_path))
    activations: dict[str, float] = {}
    for region, base_val in MOCK_BASE.items():
        noise = rng.normal(0.0, 0.1)
        activations[region] = float(np.clip(base_val + noise, 0.0, 1.0))
    return activations


def make_surface_output(
    surface: dict[str, Any],
    stimulus_id: str,
    model_id: str,
) -> dict[str, Any]:
    """Format real external-model output without pretending it is observed fMRI."""
    return {
        "surface_activations": surface["surface_activations"],
        "stimulus_id": stimulus_id,
        "source": "FmriPredicted",
        "evidence_authority": "ExternalSurrogate",
        "eligible_for_empirical_benchmarks": False,
        "eligible_for_surrogate_benchmarks": True,
        "timestamp_cycles": 0,
        "model": model_id,
        "coordinate_system": "fsaverage5",
        "aggregation": "temporal_mean",
        "n_timesteps": surface["n_timesteps"],
        "n_vertices": surface["n_vertices"],
    }


def make_mock_output(
    activations: dict[str, float],
    stimulus_id: str,
) -> dict[str, Any]:
    """Format synthetic fixture output in a schema distinct from empirical input."""
    return {
        "synthetic_region_activations": activations,
        "stimulus_id": stimulus_id,
        "source": "SyntheticFixture",
        "evidence_authority": "SyntheticFixture",
        "eligible_for_empirical_benchmarks": False,
        "eligible_for_surrogate_benchmarks": False,
        "timestamp_cycles": 0,
        "model": "symthaea-tribev2-mock",
        "coordinate_system": "symthaea12",
        "aggregation": "synthetic_fixture",
    }


def process_stimulus(
    stimulus: str,
    *,
    mock: bool,
    model_path: Optional[str],
    cache_folder: Optional[str],
) -> dict[str, Any]:
    stimulus_id = Path(stimulus).stem if stimulus else "mock"
    if mock:
        return make_mock_output(generate_mock_activations(stimulus), stimulus_id)

    surface = run_tribe_v2(stimulus, model_path, cache_folder)
    return make_surface_output(surface, stimulus_id, model_path or "facebook/tribev2")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TRIBE v2 bridge — provenance-strict neural prediction export"
    )
    parser.add_argument(
        "--stimulus", type=str, help="Path to video, audio, or text stimulus"
    )
    parser.add_argument(
        "--stimulus-dir", type=str, help="Directory of stimulus files (batch mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON path, or output directory in batch mode",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Optional explicit output directory for batch mode"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="TRIBE v2 HuggingFace model id or compatible local model reference",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="Optional TRIBE v2 model cache folder",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate explicitly synthetic fixture data; never used as fallback",
    )
    args = parser.parse_args()

    try:
        if args.stimulus_dir:
            stim_dir = Path(args.stimulus_dir)
            out_dir = Path(args.output_dir or args.output)
            out_dir.mkdir(parents=True, exist_ok=True)
            allowed = VIDEO_SUFFIXES | AUDIO_SUFFIXES | TEXT_SUFFIXES
            for stim_file in sorted(stim_dir.iterdir()):
                if not stim_file.is_file() or stim_file.suffix.lower() not in allowed:
                    continue
                print(f"Processing: {stim_file.name}")
                result = process_stimulus(
                    str(stim_file),
                    mock=args.mock,
                    model_path=args.model_path,
                    cache_folder=args.cache_folder,
                )
                out_path = out_dir / f"{stim_file.stem}.json"
                write_json(out_path, result)
                print(f"  -> {out_path}")
            return 0

        if args.stimulus or args.mock:
            stimulus = args.stimulus or "mock_stimulus"
            result = process_stimulus(
                stimulus,
                mock=args.mock,
                model_path=args.model_path,
                cache_folder=args.cache_folder,
            )
            out_path = Path(args.output)
            write_json(out_path, result)
            print(f"Output written to {out_path}")
            print(
                "Evidence authority: "
                f"{result['evidence_authority']} ({result['coordinate_system']})"
            )
            return 0

        parser.error("Either --stimulus, --stimulus-dir, or --mock is required")
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
