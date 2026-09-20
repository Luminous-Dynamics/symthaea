#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Nixward ConfigWriter pre-image qualification =="
echo "HEAD: $(git rev-parse HEAD)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "-- formatting --"
cargo fmt --all -- --check

echo "-- compile nixward library --"
cargo check -p nixward --lib

echo "-- pre-image regression --"
cargo test -p nixward --test config_writer_preimage

echo "Nixward ConfigWriter pre-image qualification: PASS"
