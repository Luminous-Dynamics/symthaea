#!/usr/bin/env bash
# Source this before using the ctc-align venv (NixOS libstdc++ fix, same
# paths as diffsinger/env.sh — see SYMTHAEA_SINGING_VOICE_NEXT_STEPS_2026-07-27.md Step 0).
source /var/lib/symthaea/training-runs/ctc-align/venv/bin/activate
export LD_LIBRARY_PATH="/nix/store/8lahnh9pn3lrrnhax5nk7ibvjcbjmnkm-gcc-15.2.0-lib/lib:/nix/store/b2swxfi8srrbsafvh9iyyhd26mz9giwf-zlib-1.3.2/lib:/run/opengl-driver/lib:"
