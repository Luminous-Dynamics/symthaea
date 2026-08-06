#!/usr/bin/env python3
"""Gate 1 v2: intelligibility ladder, confound-corrected.

v1 confounded two variables: generous duration AND a flat/unnatural
220Hz pitch contour (out-of-distribution -- DiffSinger's training data
never had a monotone pitch). v2 isolates duration alone by reusing the
REAL CSD ground-truth pitch (MIDI note per syllable, converted to Hz)
for every syllable, held constant across that syllable's own
deliberately-lengthened duration. Same 7 rungs, same generous/uniform
duration policy (150ms consonants, 350ms vowels) as v1.
"""
import json
import os

OUT_DIR = "/var/lib/symthaea/training-runs/diffsinger/ladder_v2"
os.makedirs(OUT_DIR, exist_ok=True)

CONS_DUR = 0.15
VOWEL_DUR = 0.35
SP_DUR = 0.25
F0_TIMESTEP = 0.02

VOWELS = {"a","ae","ai","ao","au","e","ei","eo","er","i","ii","oi","oo","ou","u","uu"}

def midi_to_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

# (syllable_tokens, real_CSD_midi_pitch) -- real ground-truth pitches from
# en001a.csv, same rows used in the Gate 0 audit.
CLOSING_SYLLABLES = [
    (["n","au"], 61), (["ai"], 61), (["n","ou"], 68), (["m","ai"], 68),
    (["ei"], 70), (["b","ii"], 70), (["s","ii"], 68),
    (["w","ou","n","t"], 66), (["y","uu"], 66), (["s","i","ng"], 65),
    (["eo"], 65), (["l","ou","ng"], 63), (["w","i","dh"], 63), (["m","ii"], 61),
]
ALPHABET_SYLLABLES = [
    (["ei"], 61), (["b","ii"], 61), (["s","ii"], 68), (["d","ii"], 68),
    (["ii"], 70), (["e","f"], 70), (["j","ii"], 68), (["ei","ch"], 66),
    (["ai"], 66), (["j","ei"], 65), (["k","ei"], 65), (["e","l"], 63),
    (["e","m"], 63), (["e","n"], 63), (["ou"], 63), (["p","ii"], 61),
    (["k","y","uu"], 68), (["a","r"], 68), (["e","s"], 66), (["t","ii"], 65),
    (["y","uu"], 65), (["v","ii"], 63), (["d","ao"], 68), (["b","eo","l"], 68),
    (["y","uu"], 68), (["e","k","s"], 66), (["w","ai"], 65), (["eo","n","d"], 65),
    (["z","ii"], 63),
]

# Ladder rungs, sliced from the real closing-phrase syllable list (indices
# into CLOSING_SYLLABLES): "me"=[13], "sing with me"=[9,10?]... build
# explicitly by index for clarity.
c = CLOSING_SYLLABLES
RUNGS = {
    "01_me": c[13:14],
    "02_sing_with_me": c[9:10] + c[12:14],
    "03_wont_you_sing_with_me": c[7:10] + c[12:14],
    "04_wont_you_sing_along_with_me": c[7:14],
    "05_now_i_know_my_abc": c[0:7],
    "06_full_closing_phrase": c[0:14],
    "07_alphabet": ALPHABET_SYLLABLES,
}

GROUND_TRUTH = {
    "01_me": "me",
    "02_sing_with_me": "sing with me",
    "03_wont_you_sing_with_me": "won't you sing with me",
    "04_wont_you_sing_along_with_me": "won't you sing along with me",
    "05_now_i_know_my_abc": "now I know my ABC",
    "06_full_closing_phrase": "now I know my ABC won't you sing along with me",
    "07_alphabet": "A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z",
}


def build_ds(name, syllables):
    ph_seq = ["SP"]
    ph_dur = [SP_DUR]
    f0_by_phoneme = [None]  # SP has no pitch; filled with neighbor below
    for tokens, midi in syllables:
        hz = midi_to_hz(midi)
        for tok in tokens:
            d = VOWEL_DUR if tok in VOWELS else CONS_DUR
            ph_seq.append(tok)
            ph_dur.append(d)
            f0_by_phoneme.append(hz)
    ph_seq.append("SP")
    ph_dur.append(SP_DUR)
    f0_by_phoneme.append(None)

    # Fill SP pitch by nearest neighbor (SP frames just need *a* value).
    for i in (0, len(f0_by_phoneme) - 1):
        if f0_by_phoneme[i] is None:
            neighbor = next(v for v in f0_by_phoneme if v is not None)
            f0_by_phoneme[i] = neighbor

    total_dur = sum(ph_dur)
    n_frames_total = int(round(total_dur / F0_TIMESTEP))

    # Build frame-level f0 by walking phoneme durations.
    f0_seq = []
    t = 0.0
    ph_idx = 0
    cum = 0.0
    bounds = []
    for d in ph_dur:
        cum += d
        bounds.append(cum)
    for frame in range(n_frames_total):
        t = frame * F0_TIMESTEP
        while ph_idx < len(bounds) - 1 and t >= bounds[ph_idx]:
            ph_idx += 1
        f0_seq.append(f0_by_phoneme[ph_idx])

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
    print(f"wrote {path}: {len(ph_seq)} phonemes, {total_dur:.2f}s, "
          f"pitch range {min(f0_seq):.0f}-{max(f0_seq):.0f}Hz")
    return path


if __name__ == "__main__":
    for name, syllables in RUNGS.items():
        build_ds(name, syllables)
