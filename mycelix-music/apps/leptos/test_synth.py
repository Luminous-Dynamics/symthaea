# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#!/usr/bin/env python3
"""Analyze WAV files rendered by the consciousness synth engine."""
import struct, wave, sys, os, math

def analyze_wav(path):
    with wave.open(path, 'rb') as w:
        frames = w.getnframes()
        rate = w.getframerate()
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(frames)

    # Decode 16-bit PCM
    samples = struct.unpack(f'<{frames * channels}h', raw)
    duration = frames / rate

    # Basic stats
    max_val = max(abs(s) for s in samples) / 32767.0
    rms = math.sqrt(sum(s*s for s in samples) / len(samples)) / 32767.0
    nonzero = sum(1 for s in samples if abs(s) > 10) / len(samples) * 100

    # Frequency analysis (simple zero-crossing rate for pitch estimate)
    mono = samples[::channels]  # left channel only
    crossings = sum(1 for i in range(1, len(mono)) if (mono[i] > 0) != (mono[i-1] > 0))
    est_freq = crossings / 2 / duration

    # Dynamic range (split into 1-second chunks)
    chunk_size = rate
    chunk_rms = []
    for i in range(0, len(mono), chunk_size):
        chunk = mono[i:i+chunk_size]
        if len(chunk) > 100:
            cr = math.sqrt(sum(s*s for s in chunk) / len(chunk)) / 32767.0
            chunk_rms.append(cr)

    print(f"=== {os.path.basename(path)} ===")
    print(f"  Duration: {duration:.1f}s, Rate: {rate}Hz, Channels: {channels}")
    print(f"  Peak: {max_val:.4f}, RMS: {rms:.4f}, Non-zero: {nonzero:.1f}%")
    print(f"  Est. fundamental: {est_freq:.0f} Hz")
    if chunk_rms:
        print(f"  Dynamic range: min_rms={min(chunk_rms):.4f}, max_rms={max(chunk_rms):.4f}, ratio={max(chunk_rms)/(min(chunk_rms)+0.0001):.1f}x")
        print(f"  Per-second RMS: {' '.join(f'{r:.3f}' for r in chunk_rms[:20])}")
    print()

if __name__ == '__main__':
    for path in sys.argv[1:]:
        analyze_wav(path)
