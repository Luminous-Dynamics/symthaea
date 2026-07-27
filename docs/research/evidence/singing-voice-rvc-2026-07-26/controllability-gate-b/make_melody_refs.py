#!/usr/bin/env python3
"""
Gate B: synthesize 4 unmistakable reference melodies as pure-tone sequences
for ACE-Step's audio2audio (SDEdit-style) conditioning. Each is a
5-note sequence, ~1.2s/note with a short fade to avoid clicks, ~12s total
(padded with the last note held) to roughly match the render duration.

NOTE (documented, not glossed over): audio2audio here is genuinely an
SDEdit img2img mechanism (partial noising toward a reference latent, not
a note-by-note score-following API) -- see pipeline_ace_step.py:1074,
sigma_max=(1-ref_audio_strength). This tests whether that coarse
mechanism has any measurable causal effect on realized F0 contour, not
whether ACE-Step exposes precise phoneme/note/duration control.
"""
import numpy as np
import soundfile as sf

SR = 44100
NOTE_DUR = 1.2
N_NOTES = 5
FADE = 0.02  # seconds, avoid clicks at note boundaries

# A3=220, C4=261.63, D4=293.66, E4=329.63, F4=349.23, G4=392.00
MELODIES = {
    "monotone": [220.0] * N_NOTES,
    "ascending": [261.63, 293.66, 329.63, 349.23, 392.00],
    "descending": [392.00, 349.23, 329.63, 293.66, 261.63],
    "leap": [261.63, 392.00, 261.63, 392.00, 261.63],
}

OUT_DIR = "/var/lib/symthaea/training-runs/ace-step/melody_refs"
import os
os.makedirs(OUT_DIR, exist_ok=True)


def synth_note(freq, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    tone = 0.3 * np.sin(2 * np.pi * freq * t)
    fade_n = int(FADE * SR)
    env = np.ones(n)
    env[:fade_n] = np.linspace(0, 1, fade_n)
    env[-fade_n:] = np.linspace(1, 0, fade_n)
    return tone * env


def main():
    for name, freqs in MELODIES.items():
        notes = [synth_note(f, NOTE_DUR) for f in freqs]
        audio = np.concatenate(notes)
        # pad to ~12s by holding silence-free repeat of last note (avoids
        # a long silent tail that would dominate the reference latent)
        target_len = int(12.0 * SR)
        if len(audio) < target_len:
            extra = synth_note(freqs[-1], (target_len - len(audio)) / SR)
            audio = np.concatenate([audio, extra])
        path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(path, audio.astype(np.float32), SR)
        print(f"wrote {path}  freqs={freqs}  dur={len(audio)/SR:.2f}s")


if __name__ == "__main__":
    main()
