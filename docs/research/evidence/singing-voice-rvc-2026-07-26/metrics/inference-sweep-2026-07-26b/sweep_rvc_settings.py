#!/usr/bin/env python3
"""Cheap inference-only sweep over RVC settings plausibly relevant to the
gating/silence artifact documented in the evidence bundle. Uses the
already-trained final (epoch 200) af_heart model -- no retraining.
"""
import os
import sys

RVC_ROOT = "/var/lib/symthaea/training-runs/voice-conversion/rvc"
sys.path.insert(0, RVC_ROOT)
os.chdir(RVC_ROOT)
os.environ["weight_root"] = os.path.join(RVC_ROOT, "assets", "weights")
os.environ["rmvpe_root"] = os.path.join(RVC_ROOT, "assets", "rmvpe")
os.environ["index_root"] = os.path.join(RVC_ROOT, "logs")
os.environ["outside_index_root"] = os.path.join(RVC_ROOT, "logs")

from configs.config import Config
from infer.vc.modules import VC
import soundfile as sf
import numpy as np

config = Config()
vc = VC(config)
vc.get_vc("af_heart_final.pth", 0.5, 0.33)

SRC = "/var/lib/symthaea/training-runs/voice-conversion/en001a_clip12s.wav"
INDEX = "/var/lib/symthaea/training-runs/voice-conversion/rvc/logs/af_heart/added_IVF1389_Flat_nprobe_1_af_heart_v2.index"
OUT_DIR = "/var/lib/symthaea/training-runs/voice-conversion/sweep_out"
os.makedirs(OUT_DIR, exist_ok=True)

# name -> (protect, rms_mix_rate, file_index, index_rate)
CONDITIONS = {
    "baseline":            (0.33, 0.25, "", 0.0),
    "rms_mix_high":        (0.33, 1.00, "", 0.0),
    "protect_max":         (0.50, 0.25, "", 0.0),
    "protect_off":         (0.00, 0.25, "", 0.0),
    "index_on":            (0.33, 0.25, INDEX, 0.5),
    "rms_high_protect_max": (0.50, 1.00, "", 0.0),
}

for name, (protect, rms_mix_rate, file_index, index_rate) in CONDITIONS.items():
    status, (tgt_sr, audio_opt) = vc.vc_single(
        sid=0,
        input_audio_path=SRC,
        f0_up_key=0,
        f0_method="rmvpe",
        file_index=file_index,
        index_rate=index_rate,
        resample_sr=0,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )
    if audio_opt is None:
        print(f"{name}: FAILED\n{status}")
        continue
    out_path = os.path.join(OUT_DIR, f"en001a_{name}.wav")
    sf.write(out_path, audio_opt, tgt_sr)
    print(f"{name}: wrote {out_path} ({len(audio_opt)/tgt_sr:.1f}s)")
