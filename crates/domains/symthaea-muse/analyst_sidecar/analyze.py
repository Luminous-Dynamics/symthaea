#!/usr/bin/env python3
"""Pinned, read-only external audio witness for Muse Analyst.

This sidecar emits measurements and limitations. It never changes symbolic
truth, assigns artistic quality, or adjudicates a disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import essentia
import essentia.standard as es
import numpy as np


SCHEMA_VERSION = 1


def measurement(name: str, value: float, unit: str, limitation: str) -> dict:
    return {
        "metric": name,
        "value": float(value),
        "unit": unit,
        "uncertainty": None,
        "limitations": [limitation],
    }


def analyze(path: Path) -> dict:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    sample_rate = 44_100
    audio = es.MonoLoader(filename=str(path), sampleRate=sample_rate)()
    if len(audio) == 0:
        raise ValueError("decoded audio is empty")

    onsets, onset_rate = es.OnsetRate()(audio)
    bpm, beats, beat_confidence, _, _ = es.RhythmExtractor2013(method="multifeature")(audio)
    dynamic_complexity, mean_loudness = es.DynamicComplexity()(audio)
    key, scale, key_strength = es.KeyExtractor()(audio)

    frame_size = 2048
    hop_size = 1024
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum(size=frame_size)
    centroid = es.Centroid(range=sample_rate / 2)
    centroids = []
    flatnesses = []
    flatness = es.FlatnessDB()
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        spectral = spectrum(window(frame))
        centroids.append(centroid(spectral))
        flatnesses.append(flatness(spectral))

    duration = len(audio) / sample_rate
    records = [
        measurement("duration", duration, "seconds", "Essentia resamples the decoded mono signal to 44.1 kHz."),
        measurement("onset_rate", onset_rate, "onsets/second", "Onset detection depends on the rendered timbre and mix."),
        measurement("estimated_tempo", bpm, "BPM", "Tempo estimation can select a metrical multiple or subdivision."),
        measurement("beat_confidence", beat_confidence, "unitless", "Confidence is algorithm-specific, not a probability."),
        measurement("dynamic_complexity", dynamic_complexity, "dB", "This is an audio-dynamics descriptor, not expressive quality."),
        measurement("mean_loudness", mean_loudness, "dB", "Essentia DynamicComplexity loudness is not an integrated LUFS claim."),
        measurement("spectral_centroid_mean", np.mean(centroids) if centroids else 0.0, "Hz", "Centroid is mix- and renderer-dependent."),
        measurement("spectral_flatness_db_mean", np.mean(flatnesses) if flatnesses else 0.0, "dB", "Flatness is a broadband texture proxy."),
        measurement("key_strength", key_strength, "unitless", f"Estimated key label was {key} {scale}; tonal estimation may be ambiguous."),
        measurement("detected_beat_count", len(beats), "count", "Beat tracking may be inappropriate for unmetered or elastic passages."),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_source": "external-cross-check",
        "analyzer": "Essentia",
        "analyzer_version": essentia.__version__,
        "artifact_sha256": digest,
        "input_filename": path.name,
        "sample_rate_hz": sample_rate,
        "measurements": records,
        "limitations": [
            "External measurements cannot overwrite composer assertions or symbolic verification.",
            "Mono decoding cannot verify stereo placement or channel swaps.",
            "No measurement is an artistic-quality, authenticity, or listener-preference score.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.wav)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
