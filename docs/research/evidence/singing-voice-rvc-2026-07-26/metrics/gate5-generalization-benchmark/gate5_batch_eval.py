#!/usr/bin/env python3
"""Gate 5 batch evaluation: for a given checkpoint step, render all 9
held-out phrases through the trained vocoder (one model load, not one
per phrase -- keeps the render volume tractable), plus Griffin-Lim mel
isolation for the two designated phrases (simple_chirp, the SIMPLE
control; cluster_windyspring, the CLUSTER control), matching Gate 2/3/4's
methodology on a bounded subset rather than all 9 x all render paths.

Usage: python3 gate5_batch_eval.py <step>
"""
import os
import sys

import numpy as np
import torch
import librosa
import soundfile as sf

DS = "/var/lib/symthaea/training-runs/diffsinger"
sys.path.insert(0, f"{DS}/DiffSinger")
os.chdir(f"{DS}/DiffSinger")

from utils.hparams import set_hparams, hparams
from inference.ds_acoustic import DiffSingerAcousticInfer
from utils.infer_utils import save_wav

SR, N_FFT, WIN_SIZE, HOP, N_MELS, FMIN, FMAX = 44100, 2048, 2048, 512, 128, 40, 16000

PHRASES = [
    "heldout_wontyou", "heldout_simple_chirp", "heldout_cluster_windyspring",
    "heldout_butterfly", "heldout_comeflyover", "heldout_yellowwait",
    "heldout_petalssmile", "heldout_singsong", "heldout_comedanceover",
]
GRIFFINLIM_PHRASES = {"heldout_simple_chirp", "heldout_cluster_windyspring"}


def griffinlim_from_log_mel(mel_log_T_by_nmels, out_path):
    mel_linear = np.exp(mel_log_T_by_nmels.T).astype(np.float32)
    aud = librosa.feature.inverse.mel_to_audio(
        mel_linear, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN_SIZE,
        fmin=FMIN, fmax=FMAX, power=1.0, n_iter=60,
    )
    sf.write(out_path, aud, SR)


def main():
    step = int(sys.argv[1])
    out_dir = f"{DS}/gate5_out"
    os.makedirs(out_dir, exist_ok=True)

    # Mirror scripts/infer.py's exact CLI path: it sets sys.argv then calls
    # set_hparams() with no args, so hparams load from the SAVED experiment's
    # own resolved config.yaml (checkpoints/<exp>/config.yaml), not by
    # re-resolving the source base_config chain.
    sys.argv = [sys.argv[0], "--exp_name", "benchmark01-gate5", "--infer"]
    set_hparams()

    infer_ins = DiffSingerAcousticInfer(load_vocoder=True, ckpt_steps=step)

    for name in PHRASES:
        ds_path = f"{DS}/benchmark_ds/{name}.ds"
        import json
        params = json.load(open(ds_path))
        batch = infer_ins.preprocess_input(params[0], idx=0)
        mel_pred = infer_ins.forward_model(batch)
        wav = infer_ins.run_vocoder(mel_pred, f0=batch["f0"])[0].cpu().numpy()
        save_wav(wav, f"{out_dir}/{name}_step{step}_vocoder.wav", hparams["audio_sample_rate"])
        print(f"[{step}] rendered {name} -> vocoder wav")

        if name in GRIFFINLIM_PHRASES:
            mel_log_np = mel_pred.cpu().numpy()
            if mel_log_np.ndim == 3:
                mel_log_np = mel_log_np[0]
            griffinlim_from_log_mel(mel_log_np, f"{out_dir}/{name}_step{step}_griffinlim.wav")
            print(f"[{step}] rendered {name} -> griffinlim wav")


if __name__ == "__main__":
    main()
