#!/usr/bin/env bash
set -euo pipefail

python scripts/validate-build-contract.py
python scripts/validate-postgres-plan.py
python scripts/validate-postgres-driver.py
python scripts/validate-monotonic-identities.py
python scripts/validate-fallible-runtime-paths.py
python scripts/validate-rust-api-shapes.py

cargo check --no-default-features
cargo check
cargo check --all-features
cargo check --no-default-features --features legacy-direct-startup
cargo check --features postgres-sync-driver
cargo test
cargo test --all-features
cargo test --features postgres-sync-driver
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
