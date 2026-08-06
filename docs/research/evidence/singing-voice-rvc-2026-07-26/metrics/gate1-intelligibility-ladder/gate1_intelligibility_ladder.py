#!/usr/bin/env python3
"""Gate 1: intelligibility ladder. Builds 7 phrases of increasing length/
complexity, each rendered with GENEROUS, UNIFORM note durations (slow
tempo, one syllable per note, no compressed multi-phoneme syllables) --
directly testing whether Gate 0's documented "crude uniform 70ms
consonant duration" weakness was a real bottleneck, before touching RVC
or the vocoder. Uses the same trained checkpoint (model_ckpt_steps_2000,
csd-en-poc) as every other render in this bundle -- no retraining.
"""
import json
import os

OUT_DIR = "/var/lib/symthaea/training-runs/diffsinger/ladder"
os.makedirs(OUT_DIR, exist_ok=True)

# Generous per-phoneme durations (contrast with Gate 0's uniform 70ms):
# consonants get 150ms, vowels get 350ms -- roughly 2x-5x the original
# song's rushed timing, "slow tempo, generous note durations" per spec.
CONS_DUR = 0.15
VOWEL_DUR = 0.35
SP_DUR = 0.25

VOWELS = {"a","ae","ai","ao","au","e","ei","eo","er","i","ii","oi","oo","ou","u","uu"}

# Each phrase as a list of syllables (each syllable = list of phoneme tokens,
# already validated against the csd-en dictionary and this project's own
# CSD-derived phoneme conventions for these exact words).
PHRASES = {
    "01_me": [["m","ii"]],
    "02_sing_with_me": [["s","i","ng"], ["w","i","dh"], ["m","ii"]],
    "03_wont_you_sing_with_me": [["w","ou","n","t"], ["y","uu"], ["s","i","ng"], ["w","i","dh"], ["m","ii"]],
    "04_wont_you_sing_along_with_me": [["w","ou","n","t"], ["y","uu"], ["s","i","ng"], ["eo"], ["l","ou","ng"], ["w","i","dh"], ["m","ii"]],
    "05_now_i_know_my_abc": [["n","au"], ["ai"], ["n","ou"], ["m","ai"], ["ei"], ["b","ii"], ["s","ii"]],
    "06_full_closing_phrase": [["n","au"], ["ai"], ["n","ou"], ["m","ai"], ["ei"], ["b","ii"], ["s","ii"],
                                ["w","ou","n","t"], ["y","uu"], ["s","i","ng"], ["eo"], ["l","ou","ng"], ["w","i","dh"], ["m","ii"]],
    "07_alphabet": [["ei"], ["b","ii"], ["s","ii"], ["d","ii"], ["ii"], ["e","f"], ["j","ii"],
                    ["ei","ch"], ["ai"], ["j","ei"], ["k","ei"], ["e","l"], ["e","m"], ["e","n"], ["ou"],
                    ["p","ii"], ["k","y","uu"], ["a","r"], ["e","s"], ["t","ii"], ["y","uu"], ["v","ii"],
                    ["d","ao"], ["b","eo","l"], ["y","uu"], ["e","k","s"], ["w","ai"], ["eo","n","d"], ["z","ii"]],
}

# Ground truth text for each rung, for reference when scoring the Whisper
# transcripts (scoring itself is a separate manual/CLI step, not automated
# blindly here -- see gate1_transcribe.py).
GROUND_TRUTH = {
    "01_me": "me",
    "02_sing_with_me": "sing with me",
    "03_wont_you_sing_with_me": "won't you sing with me",
    "04_wont_you_sing_along_with_me": "won't you sing along with me",
    "05_now_i_know_my_abc": "now I know my ABC",
    "06_full_closing_phrase": "now I know my ABC won't you sing along with me",
    "07_alphabet": "A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z",
}

F0_TIMESTEP = 0.02
BASE_HZ = 220.0  # comfortable, singable, roughly A3 -- flat contour, not melodic realism


def build_ds(name, syllables):
    ph_seq = ["SP"]
    ph_dur = [SP_DUR]
    for syll in syllables:
        for tok in syll:
            d = VOWEL_DUR if tok in VOWELS else CONS_DUR
            ph_seq.append(tok)
            ph_dur.append(d)
    ph_seq.append("SP")
    ph_dur.append(SP_DUR)

    total_dur = sum(ph_dur)
    n_frames = int(round(total_dur / F0_TIMESTEP))
    f0_seq = [BASE_HZ] * n_frames

    entry = {
        "offset": 0.0,
        "text": name,
        "ph_seq": " ".join(ph_seq),
        "ph_dur": " ".join(f"{d:.6f}" for d in ph_dur),
        "f0_seq": " ".join(f"{f:.1f}" for f in f0_seq),
        "f0_timestep": str(F0_TIMESTEP),
    }
    path = os.path.join(OUT_DIR, f"{name}.ds")
    with open(path, "w") as f:
        json.dump([entry], f)
    print(f"wrote {path}: {len(ph_seq)} phonemes, {total_dur:.2f}s")
    return path


if __name__ == "__main__":
    for name, syllables in PHRASES.items():
        build_ds(name, syllables)
