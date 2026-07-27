#!/usr/bin/env bash
# Exact training invocations used for this run's checkpoints.
set -euo pipefail

LD="/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:/run/opengl-driver/lib"
export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- DiffSinger acoustic model, 2000 steps, batch_size=2 (see
#     pipeline-configs/diffsinger/csd_en_acoustic.yaml for why max_batch_frames=16000
#     and random_time_stretching is disabled -- both fixes for real bugs
#     found during this run, documented inline in that file) ---
DS=/var/lib/symthaea/training-runs/diffsinger/DiffSinger
cd "$DS"
"$DS/../venv/bin/python3" scripts/train.py \
  --config configs/csd_en_acoustic.yaml --exp_name csd-en-poc --reset
# Checkpoint used: checkpoints/csd-en-poc/model_ckpt_steps_2000.ckpt
# sha256 in ../manifests/checkpoints.sha256

# --- RVC af_heart target-speaker model, 200 epochs, batch_size=4,
#     fine-tuned from RVC's own pretrained_v2 f0G40k.pth/f0D40k.pth ---
VC=/var/lib/symthaea/training-runs/voice-conversion/rvc
cd "$VC"
"$VC/../rvc-venv/bin/python3" -m train.train \
  -e af_heart -sr 40k -f0 1 -bs 4 -g 0 -te 200 -se 25 \
  -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth \
  -l 0 -c 0 -sw 0 -v v2
# Final checkpoint: logs/af_heart/G_22200.pth (epoch 200)
# Intermediate checkpoints saved every 25 epochs: G_2775 (25), G_5550 (50),
# G_8325 (75), G_11100 (100), G_13875 (125), G_16650 (150), G_19425 (175).
# sha256 for the ones sampled in this bundle are in ../manifests/checkpoints.sha256
