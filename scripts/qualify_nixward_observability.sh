#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Nixward observability boundary qualification =="
echo "HEAD: $(git rev-parse HEAD)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "-- formatting --"
cargo fmt --all -- --check

echo "-- compile observability feature --"
cargo check -p nixward --features observability --lib

echo "-- observability boundary tests --"
cargo test -p nixward --features observability --lib observability::tests

echo "Nixward observability boundary qualification: PASS"
