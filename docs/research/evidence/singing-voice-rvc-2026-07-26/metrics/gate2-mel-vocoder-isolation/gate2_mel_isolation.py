#!/usr/bin/env python3
"""Gate 2: mel-spectrogram-vs-vocoder isolation.

Question: does the acoustic model's own predicted mel-spectrogram carry
usable phonetic structure, independent of the specific trained NSF-HiFiGAN
vocoder? Method: convert both the model's predicted mel AND a real
ground-truth mel (computed from the actual CSD singer recording, same
region/words) to audio via Griffin-Lim -- a deterministic, non-neural,
non-learned vocoder. If GriffinLim-from-real-mel is at least partially
recoverable but GriffinLim-from-predicted-mel is not, that implicates the
acoustic model's mel output itself, not the NSF-HiFiGAN vocoder, as the
bottleneck. If both are equally bad, Griffin-Lim itself may be too lossy
to be informative either way.

Mel params matched exactly to configs/acoustic.yaml + nvSTFT.py:
sr=44100, n_fft=2048, win_size=2048, hop=512, n_mels=128, fmin=40, fmax=16000,
log-compression: log(clip(x, 1e-5)).

Run for two rungs (2026-07-26): 04 (5-word phrase) and 01 ("me" alone).
"""
import sys
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

OUT_DIR = f"{DS}/gate2_out"
GT_WAV_PATH = f"{DS}/CSD_extracted/CSD/english/wav/en001a.wav"

CASES = {
    "04": {"t_start": 60.0000, "t_end": 63.8813},
    "01": {"t_start": 63.5625, "t_end": 63.8813},
}


def band_energy_stats(mel_log, name):
    mel_linear = np.exp(mel_log)
    frame_sums = mel_linear.sum(axis=0 if mel_linear.shape[0] == N_MELS else 1)
    print(f"  [{name}] mean linear energy/frame: {frame_sums.mean():.4f}, "
          f"std: {frame_sums.std():.4f}, "
          f"nonzero-ish frames: {(frame_sums > frame_sums.mean()*0.1).sum()}/{len(frame_sums)}")


def griffinlim_from_log_mel(mel_log_T_by_nmels, out_path):
    # mel_log_T_by_nmels: [T, n_mels] -> librosa wants [n_mels, T]
    mel_linear = np.exp(mel_log_T_by_nmels.T).astype(np.float32)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_linear, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN_SIZE,
        fmin=FMIN, fmax=FMAX, power=1.0, n_iter=60,
    )
    sf.write(out_path, audio, SR)
    print(f"  wrote {out_path} ({len(audio)/SR:.2f}s)")


def run_case(prefix):
    t_start, t_end = CASES[prefix]["t_start"], CASES[prefix]["t_end"]
    print(f"\n########## case {prefix} ({t_start}-{t_end}s) ##########")

    pred = torch.load(f"{OUT_DIR}/{prefix}_predicted_mel.mel.pt")
    pred_mel_log = pred[0]["mel"]
    if pred_mel_log.dim() == 3:
        pred_mel_log = pred_mel_log[0]
    pred_mel_log = pred_mel_log.numpy()  # [T, n_mels]
    print(f"Predicted mel shape: {pred_mel_log.shape}, "
          f"log-range [{pred_mel_log.min():.3f}, {pred_mel_log.max():.3f}]")

    gt_wav, _ = librosa.load(GT_WAV_PATH, sr=SR)
    gt_slice = gt_wav[int(t_start * SR):int(t_end * SR)]
    sf.write(f"{OUT_DIR}/{prefix}_ground_truth_slice.wav", gt_slice, SR)
    print(f"Ground-truth slice: {len(gt_slice)/SR:.2f}s")

    gt_mel_linear = librosa.feature.melspectrogram(
        y=gt_slice, sr=SR, n_fft=N_FFT, hop_length=HOP, win_length=WIN_SIZE,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=1.0,
    )
    gt_mel_log = np.log(np.clip(gt_mel_linear, 1e-5, None))  # [n_mels, T]
    print(f"Ground-truth mel shape: {gt_mel_log.T.shape}, "
          f"log-range [{gt_mel_log.min():.3f}, {gt_mel_log.max():.3f}]")

    print("--- structural comparison ---")
    band_energy_stats(pred_mel_log.T, "predicted")
    band_energy_stats(gt_mel_log, "ground-truth")

    print("--- Griffin-Lim reconstructions ---")
    griffinlim_from_log_mel(pred_mel_log, f"{OUT_DIR}/{prefix}_predicted_griffinlim.wav")
    griffinlim_from_log_mel(gt_mel_log.T, f"{OUT_DIR}/{prefix}_ground_truth_griffinlim.wav")


if __name__ == "__main__":
    for prefix in CASES:
        run_case(prefix)
