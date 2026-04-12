#!/usr/bin/env bash
# Setup the audio benchmark environment (NixOS-compatible)
# Run once: ./setup.sh
# Then activate: source .venv/bin/activate

set -euo pipefail
cd "$(dirname "$0")"

echo "Creating Python venv with nix-provided base packages..."
nix-shell -p "python3.withPackages(ps: with ps; [numpy scipy librosa soundfile torch transformers virtualenv])" --run '
  python3 -m venv --system-site-packages .venv
  source .venv/bin/activate
  pip install --quiet laion-clap frechet-audio-distance mir_eval
  echo ""
  echo "Installed packages:"
  pip list 2>/dev/null | grep -i "laion\|frechet\|mir.eval"
  echo ""
  echo "Setup complete. Activate with:"
  echo "  source scripts/audio-bench/.venv/bin/activate"
'
