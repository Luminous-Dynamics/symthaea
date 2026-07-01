#!/usr/bin/env sh
set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"

run() {
    printf '\n==> %s\n' "$*"
    "$@"
}

run cargo fmt -p symthaea --check
printf '\n==> cargo metadata --format-version 1 --no-deps\n'
cargo metadata --format-version 1 --no-deps >/tmp/symthaea-hardening-metadata.json
run cargo test -p symthaea --lib storage_runtime --no-fail-fast --message-format short
run cargo test -p symthaea --test time_waterfall_regression --no-fail-fast --message-format short
run cargo check -p symthaea --features lancedb-backend --message-format short
run cargo check -p symthaea --features wasm-sandbox --message-format short

(
    cd crates/domains/symthaea-quantum-comp
    run cargo fmt --check
    run cargo test --all-features --message-format short
    run cargo run --bin symthaea-quantum-comp -- snapshot
)
