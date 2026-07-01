#!/usr/bin/env bash
# Lightweight lane checks for Symthaea development shells.

set -euo pipefail

cd "$(dirname "$0")/.."

lanes=()
if [[ "$#" -eq 0 ]]; then
    lanes=(core gpu python-research coding-validation)
else
    lanes=("$@")
fi

run_lane() {
    local lane="$1"
    case "$lane" in
        core)
            echo "== core =="
            nix develop .#rust -c /run/current-system/sw/bin/zsh -lc \
                'RUSTC_WRAPPER= SCCACHE_DISABLE=1 cargo check --workspace --all-targets'
            ;;
        gpu)
            echo "== gpu =="
            nix develop .#gpu -c bash ./scripts/gpu_smoke.sh --with-broca-test
            ;;
        python-research)
            echo "== python-research =="
            nix develop .#python-research -c /run/current-system/sw/bin/zsh -lc \
                'uv run --no-sync pytest tests/python -q && uv run --no-sync ruff check python/symthaea_research scripts/analyze_nixos_config.py tests/python'
            ;;
        coding-validation)
            echo "== coding-validation =="
            nix develop .#rust -c /run/current-system/sw/bin/zsh -lc \
                './scripts/run_coding_validation.sh'
            ;;
        *)
            echo "Unknown lane: $lane" >&2
            return 2
            ;;
    esac
}

for lane in "${lanes[@]}"; do
    run_lane "$lane"
done
