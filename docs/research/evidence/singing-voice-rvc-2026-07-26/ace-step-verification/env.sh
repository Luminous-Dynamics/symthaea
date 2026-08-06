#!/usr/bin/env bash
source /var/lib/symthaea/training-runs/ace-step/venv/bin/activate
export LD_LIBRARY_PATH="/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:/run/opengl-driver/lib:"
export HF_HOME=/var/lib/symthaea/training-runs/ace-step/hf-cache
