#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Nixward authority focused qualification =="
echo "HEAD: $(git rev-parse HEAD)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "-- formatting --"
cargo fmt --all -- --check

echo "-- compile nixward library --"
cargo check -p nixward --lib

echo "-- authority identity tests --"
cargo test -p nixward --lib action::authorization::tests

echo "-- pre-service identity compatibility oracle --"
cargo test -p nixward --test action_identity_compat

echo "-- executor typed-command tests --"
cargo test -p nixward --lib action::executor::tests

echo "-- service manager typed-command tests --"
cargo test -p nixward --lib action::service_manager::tests

echo "Nixward authority focused qualification: PASS"
