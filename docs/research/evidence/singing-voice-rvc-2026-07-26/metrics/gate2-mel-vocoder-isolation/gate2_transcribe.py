#!/usr/bin/env python3
"""Gate 2: transcribe the 4-way comparison for two rungs:
1. real ground-truth audio slice (sanity check -- should transcribe correctly)
2. ground-truth mel -> Griffin-Lim (how much does GriffinLim itself degrade REAL speech?)
3. predicted mel -> Griffin-Lim (does the acoustic model's mel carry usable structure?)
4. predicted mel -> trained NSF-HiFiGAN vocoder (already rendered, ladder_v2_out/*.wav)
"""
from faster_whisper import WhisperModel

OUT_DIR = "/var/lib/symthaea/training-runs/diffsinger/gate2_out"
LADDER_V2 = "/var/lib/symthaea/training-runs/diffsinger/ladder_v2_out"

CASES = {
    "04": {
        "ground_truth_text": "won't you sing along with me",
        "vocoder_wav": f"{LADDER_V2}/04_wont_you_sing_along_with_me.wav",
    },
    "01": {
        "ground_truth_text": "me",
        "vocoder_wav": f"{LADDER_V2}/01_me.wav",
    },
}

print("Loading Whisper model (small, CPU)...")
model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(path):
    segments, info = model.transcribe(path, language="en", beam_size=5)
    return " ".join(seg.text.strip() for seg in segments)


for prefix, case in CASES.items():
    gt_text = case["ground_truth_text"]
    files = {
        "1_ground_truth_real_audio": f"{OUT_DIR}/{prefix}_ground_truth_slice.wav",
        "2_ground_truth_mel_griffinlim": f"{OUT_DIR}/{prefix}_ground_truth_griffinlim.wav",
        "3_predicted_mel_griffinlim": f"{OUT_DIR}/{prefix}_predicted_griffinlim.wav",
        "4_predicted_mel_trained_vocoder": case["vocoder_wav"],
    }
    print(f"\n########## rung {prefix}: {gt_text!r} ##########")
    for name, path in files.items():
        text = transcribe(path)
        print(f"=== {name} ===")
        print(f"Whisper says: {text!r}")
