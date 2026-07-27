#!/usr/bin/env python3
"""Gate 3 (memorization gate): evaluate one overfit-training checkpoint.

Renders the training example itself ("won't you sing along with me",
overfit01.ds -- exact ph_seq/ph_dur/f0 the model was trained on) through:
  1. the trained NSF-HiFiGAN vocoder (native infer.py acoustic)
  2. the raw predicted mel -> Griffin-Lim (same isolation method as Gate 2)
and transcribes both with Whisper. Run once per checkpoint in the ladder
(1000/2000/4000/6000 steps) to see whether/when intelligibility emerges.

Usage: python3 gate3_eval_checkpoint.py <step>
"""
import subprocess
import sys
import os

import numpy as np
import torch
import librosa
import soundfile as sf

DS = "/var/lib/symthaea/training-runs/diffsinger"
sys.path.insert(0, f"{DS}/DiffSinger")

SR = 44100
N_FFT = 2048
WIN_SIZE = 2048
HOP = 512
N_MELS = 128
FMIN = 40
FMAX = 16000

LD_PATH = ("/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:"
           "/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:"
           "/run/opengl-driver/lib")


def render(step, mel_only):
    out_dir = f"{DS}/gate3_out"
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    title = f"step{step}_mel" if mel_only else f"step{step}_vocoder"
    cmd = [
        f"{DS}/venv/bin/python3", "scripts/infer.py", "acoustic",
        f"{DS}/overfit01.ds", "--exp", "overfit01-gate3", "--ckpt", str(step),
        "--out", out_dir, "--title", title,
    ]
    if mel_only:
        cmd.append("--mel")
    subprocess.run(cmd, cwd=f"{DS}/DiffSinger", env=env, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_dir, title


def griffinlim_from_predicted_mel(mel_pt_path, out_wav_path):
    pred = torch.load(mel_pt_path)
    mel_log = pred[0]["mel"]
    if mel_log.dim() == 3:
        mel_log = mel_log[0]
    mel_log = mel_log.numpy()  # [T, n_mels]
    mel_linear = np.exp(mel_log.T).astype(np.float32)  # [n_mels, T]
    audio = librosa.feature.inverse.mel_to_audio(
        mel_linear, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN_SIZE,
        fmin=FMIN, fmax=FMAX, power=1.0, n_iter=60,
    )
    sf.write(out_wav_path, audio, SR)
    return out_wav_path


def main():
    step = int(sys.argv[1])
    out_dir = f"{DS}/gate3_out"

    render(step, mel_only=False)
    vocoder_wav = f"{out_dir}/step{step}_vocoder.wav"
    print(f"rendered trained-vocoder audio: {vocoder_wav}")

    render(step, mel_only=True)
    mel_pt = f"{out_dir}/step{step}_mel.mel.pt"
    griffinlim_wav = f"{out_dir}/step{step}_griffinlim.wav"
    griffinlim_from_predicted_mel(mel_pt, griffinlim_wav)
    print(f"rendered Griffin-Lim from predicted mel: {griffinlim_wav}")


if __name__ == "__main__":
    main()
