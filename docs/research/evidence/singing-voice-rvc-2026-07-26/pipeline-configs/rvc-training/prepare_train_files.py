#!/usr/bin/env python3
"""Replicate webui.py's click_train() filelist/config generation (which
we bypassed by driving preprocess/extract_f0/extract_hubert_feature
headlessly instead of through the Gradio UI). train/train.py expects
logs/<exp>/filelist.txt and logs/<exp>/config.json to already exist.
"""
import json
import os
import random

RVC_ROOT = "/var/lib/symthaea/training-runs/voice-conversion/rvc"
EXP_DIR = os.path.join(RVC_ROOT, "logs", "af_heart")
SR = "40k"
VERSION = "v2"
SPK_ID = 0

gt_wavs_dir = os.path.join(EXP_DIR, "0_gt_wavs")
feature_dir = os.path.join(EXP_DIR, "3_feature768")  # v2 = 768-dim
f0_dir = os.path.join(EXP_DIR, "2a_f0")
f0nsf_dir = os.path.join(EXP_DIR, "2b-f0nsf")

names = (
    {n.split(".")[0] for n in os.listdir(gt_wavs_dir)}
    & {n.split(".")[0] for n in os.listdir(feature_dir)}
    & {n.split(".")[0] for n in os.listdir(f0_dir)}
    & {n.split(".")[0] for n in os.listdir(f0nsf_dir)}
)
assert names, "no matched names across gt_wavs/feature/f0/f0nsf dirs"

opt = []
for name in names:
    opt.append(
        "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
        % (gt_wavs_dir, name, feature_dir, name, f0_dir, name, f0nsf_dir, name, SPK_ID)
    )

fea_dim = 768
for _ in range(2):
    opt.append(
        "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|"
        "%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
        % (RVC_ROOT, SR, RVC_ROOT, fea_dim, RVC_ROOT, RVC_ROOT, SPK_ID)
    )

random.shuffle(opt)
filelist_path = os.path.join(EXP_DIR, "filelist.txt")
with open(filelist_path, "w", encoding="utf8") as f:
    f.write("\n".join(opt))
print(f"wrote {filelist_path}: {len(opt)} lines ({len(names)} real + 2 mute)")

# webui.py: 40k always routes to the v1 config regardless of version19.
config_path = os.path.join(RVC_ROOT, "configs", "v1", f"{SR}.json")
with open(config_path, encoding="utf8") as f:
    config = json.load(f)
config_save_path = os.path.join(EXP_DIR, "config.json")
with open(config_save_path, "w", encoding="utf8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4, sort_keys=True)
    f.write("\n")
print(f"wrote {config_save_path} from {config_path}")
