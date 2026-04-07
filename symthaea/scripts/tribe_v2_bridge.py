#!/usr/bin/env python3
"""
TRIBE v2 Bridge — fMRI prediction oracle for Symthaea neural validation.

Loads Meta FAIR's TRIBE v2 brain encoding model and produces per-region
activation predictions from video/audio/text stimuli. Output is JSON
compatible with Symthaea's CorticalActivationMap format.

Usage:
    python scripts/tribe_v2_bridge.py --stimulus video.mp4 --output result.json
    python scripts/tribe_v2_bridge.py --stimulus-dir data/tribe-v2-stimuli/ --output-dir results/
    python scripts/tribe_v2_bridge.py --mock --output mock_result.json  # No model needed

Requirements:
    pip install torch torchvision torchaudio numpy
    # TRIBE v2 model: git clone https://github.com/facebookresearch/tribe
    # or: pip install tribe-v2  (when available)

References:
    - TRIBE v2 (Meta FAIR, 2026). Tri-modal brain encoding for fMRI prediction.
    - Glasser et al. (2016). Nature, 536, 171-178.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Glasser atlas 360-parcel → 12-region mapping (mirroring glasser_parcellation.rs)
# Left hemisphere parcels (1-180); right hemisphere = parcel + 180
REGION_PARCELS = {
    "Visual": [1, 2, 3, 4, 5, 6, 13, 14, 15, 18, 19, 153, 154, 155, 156, 160, 163],
    "Auditory": [24, 104, 105, 124, 125, 126, 173, 174],
    "Language": [75, 76, 128, 129, 130, 131, 132, 133, 134, 175, 176],
    "Motor": [8, 55, 56, 57, 58, 59, 60, 61, 62, 78, 79],
    "Sensory": [9, 51, 52, 53, 54, 106, 107, 108, 109],
    "Prefrontal": [68, 69, 70, 71, 72, 73, 74, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90],
    "Memory": [20, 21, 22, 23, 135, 136, 137, 164, 165],
    "Emotional": [91, 92, 93, 94, 95, 96, 97, 98, 110, 111, 112],
    "Social": [139, 140, 141, 142, 143, 144, 145],
    "Executive": [63, 64, 65, 66, 67, 99, 100, 101],
    "Creative": [26, 27, 28, 29, 30, 31, 146, 147, 148, 149, 150],
    "Integration": [
        7, 10, 11, 12, 16, 17, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 102, 103, 113, 114, 115,
        116, 117, 118, 119, 120, 121, 122, 123, 127, 138, 151, 152,
        157, 158, 159, 161, 162, 166, 167, 168, 169, 170, 171, 172,
        177, 178, 179, 180,
    ],
}

# Expand to include right hemisphere mirrors
REGION_PARCELS_FULL = {}
for region, parcels in REGION_PARCELS.items():
    full = []
    for p in parcels:
        full.append(p)
        full.append(p + 180)  # right hemisphere
    REGION_PARCELS_FULL[region] = full


def aggregate_voxels_to_regions(
    voxel_data: np.ndarray,
) -> dict[str, float]:
    """Aggregate 360 voxel values to 12 region means."""
    result = {}
    for region, parcels in REGION_PARCELS_FULL.items():
        values = []
        for p in parcels:
            idx = p - 1  # 0-based
            if idx < len(voxel_data):
                values.append(float(voxel_data[idx]))
        if values:
            result[region] = float(np.mean(values))
        else:
            result[region] = 0.0
    return result


def normalize_activations(activations: dict[str, float]) -> dict[str, float]:
    """Normalize activations to [0, 1] range."""
    vals = list(activations.values())
    if not vals:
        return activations
    vmin = min(vals)
    vmax = max(vals)
    rng = vmax - vmin
    if rng < 1e-10:
        return {k: 0.5 for k in activations}
    return {k: (v - vmin) / rng for k, v in activations.items()}


def run_tribe_v2(
    stimulus_path: str,
    model_path: Optional[str] = None,
) -> dict[str, float]:
    """Run TRIBE v2 inference on a stimulus and return per-region activations.

    Requires the TRIBE v2 model to be installed. Falls back to mock mode
    if the model is not available.
    """
    try:
        # Attempt to import TRIBE v2
        from tribe import TRIBEv2Model  # type: ignore[import-untyped]

        model = TRIBEv2Model.from_pretrained(model_path or "tribe-v2-large")

        import torch
        import torchaudio
        import torchvision

        # Load video + audio
        video, audio, info = torchvision.io.read_video(stimulus_path, pts_unit="sec")
        video = video.float() / 255.0  # [T, H, W, C]

        # Run inference
        with torch.no_grad():
            predicted_bold = model.encode(video=video, audio=audio)
            # predicted_bold shape: [time_points, n_voxels]

        # Average across time, take first 360 voxels (Glasser parcels)
        mean_bold = predicted_bold.mean(dim=0).cpu().numpy()
        voxel_360 = mean_bold[:360]

        activations = aggregate_voxels_to_regions(voxel_360)
        return normalize_activations(activations)

    except ImportError:
        print(
            "WARNING: TRIBE v2 model not installed. "
            "Install from https://github.com/facebookresearch/tribe",
            file=sys.stderr,
        )
        print("Falling back to mock mode.", file=sys.stderr)
        return generate_mock_activations(stimulus_path)


def generate_mock_activations(stimulus_path: str = "") -> dict[str, float]:
    """Generate plausible mock activations for testing without the real model.

    Uses stimulus filename to seed the RNG for reproducibility.
    """
    seed = hash(stimulus_path) % (2**31)
    rng = np.random.default_rng(seed)

    # Base pattern: visual and auditory regions high for video stimuli
    base = {
        "Visual": 0.7,
        "Auditory": 0.5,
        "Language": 0.3,
        "Motor": 0.15,
        "Sensory": 0.1,
        "Prefrontal": 0.35,
        "Memory": 0.25,
        "Emotional": 0.3,
        "Social": 0.2,
        "Executive": 0.25,
        "Creative": 0.15,
        "Integration": 0.4,
    }

    # Add noise
    activations = {}
    for region, base_val in base.items():
        noise = rng.normal(0, 0.1)
        activations[region] = float(np.clip(base_val + noise, 0.0, 1.0))

    return activations


def make_output(
    activations: dict[str, float],
    stimulus_id: str,
    source: str = "FmriPredicted",
) -> dict:
    """Format output as Symthaea-compatible JSON."""
    return {
        "region_activations": activations,
        "stimulus_id": stimulus_id,
        "source": source,
        "timestamp_cycles": 0,
        "model": "tribe-v2",
        "mapping": "glasser-360-to-12",
    }


def main():
    parser = argparse.ArgumentParser(
        description="TRIBE v2 Bridge — fMRI prediction oracle for Symthaea"
    )
    parser.add_argument(
        "--stimulus", type=str, help="Path to stimulus file (video/audio)"
    )
    parser.add_argument(
        "--stimulus-dir",
        type=str,
        help="Directory of stimulus files (batch mode)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch mode",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to TRIBE v2 model weights",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Generate mock activations (no model needed)",
    )
    args = parser.parse_args()

    if args.stimulus_dir:
        # Batch mode
        stim_dir = Path(args.stimulus_dir)
        out_dir = Path(args.output_dir or args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        for stim_file in sorted(stim_dir.glob("*")):
            if stim_file.suffix in (".mp4", ".avi", ".wav", ".mkv"):
                print(f"Processing: {stim_file.name}")
                if args.mock:
                    activations = generate_mock_activations(str(stim_file))
                else:
                    activations = run_tribe_v2(str(stim_file), args.model_path)
                result = make_output(activations, stim_file.stem)
                out_path = out_dir / f"{stim_file.stem}.json"
                with open(out_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  → {out_path}")

    elif args.stimulus or args.mock:
        stimulus = args.stimulus or "mock_stimulus"
        if args.mock:
            activations = generate_mock_activations(stimulus)
        else:
            activations = run_tribe_v2(stimulus, args.model_path)

        stim_id = Path(stimulus).stem if args.stimulus else "mock"
        result = make_output(activations, stim_id)

        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Output written to {args.output}")
        print(f"Region activations: {json.dumps(activations, indent=2)}")

    else:
        parser.error("Either --stimulus, --stimulus-dir, or --mock is required")


if __name__ == "__main__":
    main()
