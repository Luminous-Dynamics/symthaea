#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MUSE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cargo fmt --all --check
cargo test -p symthaea-muse --lib --features theory pilot_protocol
cargo test -p symthaea-muse --lib --features theory pilot_schedule
cargo test -p symthaea-muse --lib --features theory cohort_registry
cargo test -p symthaea-muse --lib --features theory pilot_collection
cargo test -p symthaea-muse --lib --features theory pilot_monitoring
cargo test -p symthaea-muse --lib --features theory pilot_report
cargo test -p symthaea-muse --lib --features theory study_orchestration
cargo test -p symthaea-muse --lib --features theory analysis_crosscheck
cargo test -p symthaea-muse --lib --features theory reproducibility_attestation
cargo test -p symthaea-muse --lib --features theory study_operations_release
cargo test -p symthaea-muse --bin cognitive_study --features theory
cargo test -p symthaea-muse --bin cognitive_study_runner --features studio
cargo clippy -p symthaea-muse --all-targets --features studio -- -D warnings
bash "$MUSE_DIR/scripts/check-cognition-v82-reference.sh"
bash "$MUSE_DIR/scripts/check-cognition-v9-reference.sh"
bash "$MUSE_DIR/scripts/check-cognition-v10-reference.sh"
git -C "$MUSE_DIR" diff --check
