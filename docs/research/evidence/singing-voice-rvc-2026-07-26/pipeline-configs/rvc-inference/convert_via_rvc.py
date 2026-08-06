#!/usr/bin/env python3
"""Export a usable inference model from the RVC training checkpoint and
run a real conversion test: take one of our existing CSD-trained
DiffSinger sung samples and convert its timbre to af_heart, preserving
the source's real sung pitch/rhythm (RVC's whole design point).
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
os.makedirs(os.environ["weight_root"], exist_ok=True)

from train.process_ckpt import extract_small_model

ckpt_path = "logs/af_heart/G_22200.pth"
model_name = "af_heart_final"
result = extract_small_model(ckpt_path, model_name, "40k", "1", "af_heart final (epoch 200)", "v2")
print("export result:", result)

from configs.config import Config
from infer.vc.modules import VC

config = Config()
vc = VC(config)
vc.get_vc(f"{model_name}.pth", 0.5, 0.33)

src = "/srv/luminous-dynamics/symthaea/audio_output/diffsinger_csd_poc_2026-07-25/en001a-step2000-final.wav"
status, (tgt_sr, audio_opt) = vc.vc_single(
    sid=0,
    input_audio_path=src,
    f0_up_key=0,
    f0_method="rmvpe",
    file_index="",
    index_rate=0,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33,
)
print("vc_single status:", status)
if audio_opt is not None:
    import soundfile as sf
    out_path = "/var/lib/symthaea/training-runs/voice-conversion/en001a_af_heart_final.wav"
    sf.write(out_path, audio_opt, tgt_sr)
    print(f"wrote {out_path}: sr={tgt_sr} dur={len(audio_opt)/tgt_sr:.1f}s")
else:
    print("CONVERSION FAILED -- no audio returned")
