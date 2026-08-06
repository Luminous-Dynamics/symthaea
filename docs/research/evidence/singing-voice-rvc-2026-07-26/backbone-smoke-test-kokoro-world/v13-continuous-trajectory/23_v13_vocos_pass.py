#!/usr/bin/env python3
"""Stage 2 of v13 (continuous-trajectory WORLD + one global Vocos pass):
run the WORLD-only outputs from 22_continuous_trajectory_world_vocos.py
through Vocos's own forward() (mel-extract + decode), same method as
17_vocos_resynthesis.py used for v12. Run inside `nix develop
.#voice-vocoder` (torch/torchaudio/vocos are nix-managed, not pip).
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vocos import Vocos

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
AUDIO_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_continuous_trajectory")

TARGET_SR = 24000


def main():
    results = json.loads((BASE / "v13_continuous_trajectory_results.json").read_text())

    print("Loading Vocos (charactr/vocos-mel-24khz)...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()

    updated = []
    for r in results:
        world_path = Path(r["world_out"])
        audio, sr = sf.read(str(world_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        wav = torch.from_numpy(audio).unsqueeze(0)
        if sr != TARGET_SR:
            import torchaudio
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

        with torch.no_grad():
            y_hat = vocos(wav)
        y_vocos = y_hat.squeeze(0).cpu().numpy()

        n_nan = int(np.isnan(y_vocos).sum())
        n_inf = int(np.isinf(y_vocos).sum())
        peak = float(np.max(np.abs(y_vocos))) if y_vocos.size else float("nan")
        clipped = int(np.sum(np.abs(y_vocos) >= 0.999))
        status = "NAN_OR_INF" if (n_nan or n_inf) else ("BLOWUP" if peak > 1.5 else "OK")

        out_path = AUDIO_DIR / f"{r['phrase']}_sung_v13_vocos.wav"
        sf.write(str(out_path), y_vocos, TARGET_SR)

        print(f"{r['phrase']:24s} peak={peak:.3f} clipped={clipped} nan={n_nan} inf={n_inf} [{status}] -> {out_path.name}")

        r = dict(r)
        r.update({"vocos_out": str(out_path), "vocos_peak": peak,
                   "vocos_clipped_samples": clipped, "vocos_n_nan": n_nan,
                   "vocos_n_inf": n_inf, "vocos_status": status})
        updated.append(r)

    (BASE / "v13_continuous_trajectory_results.json").write_text(json.dumps(updated, indent=2))
    n_ok = sum(1 for r in updated if r["vocos_status"] == "OK")
    print(f"\n{n_ok}/{len(updated)} Vocos passes OK.")


if __name__ == "__main__":
    main()
