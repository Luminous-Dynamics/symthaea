#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Nixward legacy approval replay qualification =="
echo "HEAD: $(git rev-parse HEAD)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "-- formatting --"
cargo fmt --all -- --check

echo "-- compile nixward library --"
cargo check -p nixward --lib

echo "-- legacy approval replay tests --"
cargo test -p nixward --lib action::legacy_approval_gate::tests

echo "Nixward legacy approval replay qualification: PASS"
