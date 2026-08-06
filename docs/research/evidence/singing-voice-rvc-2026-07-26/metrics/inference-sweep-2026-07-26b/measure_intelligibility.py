#!/usr/bin/env python3
"""Decisive intelligibility diagnostic: transcribe the raw DiffSinger
source and both RVC conversions (untuned, tuned) with Whisper, and score
each against the REAL ground-truth lyrics for en001a (confirmed literally
the ABC song, from CSD's own lyric/en001a.txt). Answers: is DiffSinger's
output already unintelligible upstream, or does RVC specifically destroy
intelligibility that was otherwise present?
"""
import jiwer
from faster_whisper import WhisperModel

FILES = {
    "source (DiffSinger, untouched)":
        "/srv/luminous-dynamics/symthaea/audio_output/diffsinger_csd_poc_2026-07-25/en001a-step2000-final.wav",
    "untuned RVC (defaults)":
        "/srv/luminous-dynamics/symthaea/audio_output/diffsinger_csd_poc_2026-07-25/en001a_af_heart_FINAL_ep200.wav",
    "tuned RVC (rms_mix_rate=1.0, index on)":
        "/srv/luminous-dynamics/symthaea/audio_output/diffsinger_csd_poc_2026-07-25/en001a_af_heart_FINAL_ep200_TUNED.wav",
}

# Verbatim from CSD_extracted/CSD/english/lyric/en001a.txt, both repeated
# verse+chorus sections (A and B), normalized to plain text for scoring.
GROUND_TRUTH = (
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z "
    "Now I know my A B C Next time won't you sing with me "
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z "
    "Now I know my A B C won't you sing along with me"
)

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])

print("Loading Whisper model (small, CPU)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

results = {}
for label, path in FILES.items():
    print(f"\n=== {label} ===")
    segments, info = model.transcribe(path, language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    print(f"Transcript: {text!r}")

    wer = jiwer.wer(GROUND_TRUTH, text, reference_transform=transform, hypothesis_transform=transform)
    cer = jiwer.cer(GROUND_TRUTH, text)
    print(f"WER: {wer*100:.1f}%   CER: {cer*100:.1f}%")
    results[label] = {"transcript": text, "wer": wer, "cer": cer}

print("\n\n=== SUMMARY ===")
print(f"{'condition':<42} {'WER':>8} {'CER':>8}")
for label, r in results.items():
    print(f"{label:<42} {r['wer']*100:>7.1f}% {r['cer']*100:>7.1f}%")
