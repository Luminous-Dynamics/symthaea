#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MUSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo fmt --all --check
cargo test -p symthaea-muse --lib --features theory external_review_protocol
cargo test -p symthaea-muse --lib --features theory external_review_package
cargo test -p symthaea-muse --lib --features theory external_review_response
cargo test -p symthaea-muse --lib --features theory external_review_resolution
cargo test -p symthaea-muse --lib --features theory external_review_completion
cargo test -p symthaea-muse --lib --features theory confirmatory_amendment_control
cargo test -p symthaea-muse --lib --features theory confirmatory_readiness
cargo test -p symthaea-muse --lib --features theory confirmatory_readiness_release
cargo test -p symthaea-muse --bin cognitive_study --features theory
cargo clippy -p symthaea-muse --all-targets --features studio -- -D warnings
python3 "$MUSE_DIR/scripts/verify_cognition_study_v11.py" --self-test
bash "$MUSE_DIR/scripts/check-cognition-v10-execution.sh"
git -C "$MUSE_DIR" diff --check
