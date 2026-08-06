#!/usr/bin/env bash
# Exact inference invocations used to produce this bundle's sample outputs.
set -euo pipefail

LD="/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:/run/opengl-driver/lib"
export LD_LIBRARY_PATH="$LD:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS=/var/lib/symthaea/training-runs/diffsinger

# --- Step 1: build a .ds inference file from held-out CSD test data
#     (real ph_seq/ph_dur from the corpus + real parselmouth-extracted F0
#     from the ground-truth recording -- see build_ds_file.py) ---
cd "$DS"
"$DS/venv/bin/python3" build_ds_file.py en001a

# --- Step 2: native DiffSinger acoustic inference -> sung waveform ---
cd "$DS/DiffSinger"
"$DS/venv/bin/python3" scripts/infer.py acoustic \
  ../en001a.ds --exp csd-en-poc --ckpt 2000 \
  --out /path/to/out --title en001a-final

# --- Step 3: RVC voice conversion (see pipeline-configs/rvc-inference/convert_via_rvc.py
#     for the exact Python driving the VC class directly, bypassing the
#     Gradio webui -- required os.environ["rmvpe_root"] etc. to be set
#     manually, a real gap found during this run) ---
VC=/var/lib/symthaea/training-runs/voice-conversion
"$VC/rvc-venv/bin/python3" "$VC/test_rvc_convert.py"
# vc_single() params used: f0_up_key=0 (no pitch shift), f0_method="rmvpe",
# file_index="" (no FAISS retrieval index was trained), index_rate=0,
# resample_sr=0 (keep 40kHz), rms_mix_rate=0.25, protect=0.33 (defaults).
# These were NOT swept -- see LICENSE_STATUS.md / CLAIMS.md for what a
# follow-up inference-settings sweep would need to test.
