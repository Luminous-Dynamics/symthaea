#!/usr/bin/env python3
"""
Gate B v1 closing control: build a genuinely VOCAL (not sine-tone)
melody reference, per user direction after the sine-tone pilot showed a
sharp lyrics-vs-conditioning-strength cliff. Uses a single sustained
"la" syllable (espeak-ng TTS, la_single.wav), pitch-shifted per target
note (librosa.effects.pitch_shift, phase-vocoder -- preserves timbre/
formants reasonably well unlike a naive resample), tiled to fill each
1.2s note window, for the "ascending" and "leap" melodies only (per the
user's explicit small-scope instruction -- not a full 4-melody redo).
"""
import os

import librosa
import numpy as np
import soundfile as sf

SR = 22050
NOTE_DUR = 1.2
N_NOTES = 5
LA_PATH = "/var/lib/symthaea/training-runs/ace-step/la_single.wav"
OUT_DIR = "/var/lib/symthaea/training-runs/ace-step/melody_refs"
os.makedirs(OUT_DIR, exist_ok=True)

MELODIES = {
    "ascending_vocal": [261.63, 293.66, 329.63, 349.23, 392.00],
    "leap_vocal": [261.63, 392.00, 261.63, 392.00, 261.63],
}


def main():
    y, sr = librosa.load(LA_PATH, sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr)
    base_f0 = float(np.nanmedian(f0[voiced]))
    print(f"base 'la' F0: {base_f0:.2f}Hz")

    # Trim to the sustained voiced region only (drop the /l/ onset transient).
    hop = 512
    frame_times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop)
    voiced_times = frame_times[voiced]
    t0 = float(voiced_times[0]) if len(voiced_times) else 0.0
    t1 = float(voiced_times[-1]) if len(voiced_times) else len(y) / sr
    sustained = y[int(t0 * sr):int(t1 * sr)]
    print(f"sustained region: [{t0:.2f}, {t1:.2f}]s, {len(sustained)/sr:.2f}s")

    for name, targets in MELODIES.items():
        notes = []
        for target_hz in targets:
            n_steps = 12 * np.log2(target_hz / base_f0)
            shifted = librosa.effects.pitch_shift(sustained, sr=sr, n_steps=n_steps)
            # tile/trim to exactly fill this note's window
            reps = int(np.ceil(NOTE_DUR * sr / len(shifted)))
            tiled = np.tile(shifted, reps)[: int(NOTE_DUR * sr)]
            fade = int(0.02 * sr)
            tiled[:fade] *= np.linspace(0, 1, fade)
            tiled[-fade:] *= np.linspace(1, 0, fade)
            notes.append(tiled)
        audio = np.concatenate(notes)
        target_len = int(12.0 * sr)
        if len(audio) < target_len:
            audio = np.concatenate([audio, np.tile(notes[-1], int(np.ceil((target_len - len(audio)) / len(notes[-1]))))])[:target_len]
        path = os.path.join(OUT_DIR, f"{name}.wav")
        sf.write(path, audio.astype(np.float32), sr)
        print(f"wrote {path}  dur={len(audio)/sr:.2f}s  targets={targets}")


if __name__ == "__main__":
    main()
