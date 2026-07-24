#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MUSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
cargo fmt --all --check
cargo test -p symthaea-muse --lib --features theory study_artifact
cargo test -p symthaea-muse --lib --features theory study_runner
cargo test -p symthaea-muse --lib --features theory study_collection
cargo test -p symthaea-muse --lib --features theory study_release
cargo test -p symthaea-muse --bin cognitive_study --features theory
cargo test -p symthaea-muse --bin cognitive_study_runner --features studio
cargo clippy -p symthaea-muse --all-targets --features studio -- -D warnings
bash "$MUSE_DIR/scripts/check-cognition-v82-reference.sh"
bash "$MUSE_DIR/scripts/check-cognition-v9-reference.sh"
git -C "$MUSE_DIR" diff --check
