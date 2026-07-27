#!/usr/bin/env python3
"""Gate 0 realized-audio cross-check: for each planned consonant window
(from gate0_duration_audit.py's closing-phrase output), measure RMS
energy relative to surrounding context and high-frequency energy ratio
in the actual rendered DiffSinger waveform. Informal signal, not a
validated forced aligner -- see README.md for interpretation and caveats.
"""
import numpy as np
import soundfile as sf

path = "/srv/luminous-dynamics/symthaea/audio_output/diffsinger_csd_poc_2026-07-25/en001a-step2000-final.wav"
a, sr = sf.read(path)
if a.ndim > 1:
    a = a.mean(axis=1)

# Planned consonant windows (start_ms, end_ms, word, phoneme, class) from the
# Gate 0 audit -- closing phrase, absolute timeline (matches build_ds_file.py's
# real f0/timing, so these ms offsets correspond directly to this wav).
windows = [
    (55200, 55270, "Now", "n", "nasal"),
    (56400, 56470, "know", "n", "nasal"),
    (57000, 57070, "my", "m", "nasal"),
    (58218.8, 58288.8, "B", "b", "stop"),
    (58762.5, 58832.5, "C", "s", "fricative"),
    (60000, 60070, "won't(w)", "w", "glide"),
    (60272.5, 60342.5, "won't(n)", "n", "nasal"),
    (60342.5, 60412.5, "won't(t)", "t", "stop"),
    (61162.5, 61232.5, "sing(s)", "s", "fricative"),
    (61711.2, 61781.2, "sing(ng)", "ng", "nasal"),
    (62400, 62470, "along(l)", "l", "liquid"),
    (62930, 63000, "along(ng)", "ng", "nasal"),
    (63000, 63070, "with(w)", "w", "glide"),
    (63492.5, 63562.5, "with(dh)", "dh", "fricative"),
    (63562.5, 63632.5, "me(m)", "m", "nasal"),
]

def hf_energy_ratio(seg, sr):
    # crude high-frequency-vs-total energy ratio via FFT, useful for
    # fricatives/stops which concentrate energy above ~3kHz
    if len(seg) < 32:
        return None
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    freqs = np.fft.rfftfreq(len(seg), 1/sr)
    total = spec.sum() + 1e-12
    hf = spec[freqs > 3000].sum()
    return hf / total

def rms(seg):
    return np.sqrt(np.mean(seg.astype(np.float64)**2)) if len(seg) else 0.0

if __name__ == "__main__":
    print(f"{'word/phon':<14}{'class':<10}{'ph_rms':<10}{'ctx_rms':<10}{'rms_ratio':<11}{'hf_ratio':<10}{'note'}")
    for start_ms, end_ms, word, phon, cls in windows:
        s = int(start_ms/1000 * sr)
        e = int(end_ms/1000 * sr)
        seg = a[s:e]
        ctx_s = max(0, s - int(0.05*sr))
        ctx_e = min(len(a), e + int(0.05*sr))
        ctx = a[ctx_s:ctx_e]

        ph_rms = rms(seg)
        ctx_rms = rms(ctx)
        ratio = ph_rms / (ctx_rms + 1e-9)
        hf = hf_energy_ratio(seg, sr)

        note = ""
        if cls in ("stop",) and ratio > 0.7:
            note = "no clear stop-closure dip detected"
        if cls in ("fricative",) and hf is not None and hf < 0.15:
            note = "low HF energy for a fricative -- may be under-articulated"
        print(f"{word:<14}{cls:<10}{ph_rms:<10.4f}{ctx_rms:<10.4f}{ratio:<11.3f}{hf:<10.3f}{note}")
