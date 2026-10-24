#!/usr/bin/env bash
set -euo pipefail

# Keep this lane scoped to temporal evidence. Repository-wide formatting drift is
# outside this safety qualification and is exercised by the broader CI lane.
rustfmt --edition 2024 --check crates/core/nixward/src/action/temporal.rs
cargo check -p nixward --lib
cargo test -p nixward --lib action::temporal::tests
