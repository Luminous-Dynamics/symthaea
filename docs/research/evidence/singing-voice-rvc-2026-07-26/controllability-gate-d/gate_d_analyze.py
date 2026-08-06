#!/usr/bin/env python3
"""
Gate D analysis: transcribe all 30 renders (10 phrases x 3 seeds),
report per-phrase-category exact-match rate and per-seed consistency.
"""
import glob
import os
import re
from collections import defaultdict

from faster_whisper import WhisperModel

GATE_D_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_d_out"

TARGETS = {
    "positive_control":      "won't you sing along with me",
    "conversational":        "i love the summer breeze tonight",
    "repeated_syllables":    "bye bye bye bye baby",
    "rapid_letter_names":    "a b c d e f g",
    "phrase_final_stops":    "turn off the light and lock it",
    "fricative_heavy":       "she sells seashells by the seashore",
    "consonant_clusters":    "strong streams splashed strangely",
    "long_sustained_vowels": "moon over the blue lagoon",
    "short_unstressed":      "it is what it is to me",
    "semantically_unusual":  "the clock ate my umbrella",
}


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    paths = sorted(glob.glob(os.path.join(GATE_D_DIR, "*.wav")))
    print(f"Found {len(paths)} renders\n")
    whisper = WhisperModel("base", device="cpu", compute_type="int8")

    by_category = defaultdict(list)
    for path in paths:
        name = os.path.basename(path)[:-4]
        m = re.match(r"(.+)_seed(\d+)$", name)
        category, seed = m.group(1), m.group(2)
        seg_iter, info = whisper.transcribe(path, language="en")
        transcript = " ".join(seg.text.strip() for seg in seg_iter).strip()
        norm_transcript = normalize(transcript)
        target = normalize(TARGETS[category])  # bug fix: target was never normalized,
        # so apostrophes in e.g. "won't" caused false negatives against the
        # apostrophe-stripped transcript -- caught and fixed same session
        exact = target in norm_transcript
        by_category[category].append((seed, transcript, exact))
        print(f"{name}: exact={exact} transcript={transcript!r}")

    print("\n=== Per-category summary ===")
    for category, target in TARGETS.items():
        entries = by_category[category]
        n_exact = sum(1 for _, _, e in entries if e)
        print(f"{category} (target: {target!r}): {n_exact}/{len(entries)} exact")


if __name__ == "__main__":
    main()
