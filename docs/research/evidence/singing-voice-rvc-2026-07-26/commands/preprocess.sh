#!/usr/bin/env bash
# Exact commands used to prepare both training corpora for this run.
# Not idempotent-safe to copy/paste blindly -- read REPRODUCE.md first.
set -euo pipefail

LD="/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:/run/opengl-driver/lib"

# --- DiffSinger side: CSD -> raw dataset -> binarized ---
DS=/var/lib/symthaea/training-runs/diffsinger
export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
python3 "$DS/convert_csd.py" "$DS/CSD_extracted/CSD/english" "$DS/raw/csd-en"
ln -sfn "$DS/raw" "$DS/DiffSinger/raw"
cd "$DS/DiffSinger"
"$DS/venv/bin/python3" scripts/binarize.py --config configs/csd_en_acoustic.yaml

# --- RVC side: af_heart Kokoro corpus -> sliced/F0/HuBERT features ---
VC=/var/lib/symthaea/training-runs/voice-conversion
"$VC/venv/bin/python3" "$VC/generate_corpus.py"   # writes af_heart_corpus/*.wav (~20 min)

cd "$VC/rvc"
mkdir -p logs/af_heart
"$VC/rvc-venv/bin/python3" -m train.preprocess \
  "$VC/af_heart_corpus" 40000 4 "$(pwd)/logs/af_heart" False 3.0

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$VC/rvc-venv/bin/python3" -m train.dataset.extract_f0 cuda 1 0 0 "$(pwd)/logs/af_heart" True
"$VC/rvc-venv/bin/python3" -m train.dataset.extract_hubert_feature cuda 1 0 0 "$(pwd)/logs/af_heart" v2 True

# webui.py normally builds filelist.txt/config.json via its Gradio handler;
# this run bypassed the UI, so that step is replicated by a standalone script:
"$VC/rvc-venv/bin/python3" "$VC/prepare_train_files.py"
