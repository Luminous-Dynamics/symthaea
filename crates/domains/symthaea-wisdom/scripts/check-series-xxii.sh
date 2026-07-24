#!/usr/bin/env bash
set -euo pipefail

python scripts/validate-build-contract.py
python scripts/validate-rust-lexical.py
python scripts/validate-postgres-plan.py
python scripts/validate-postgres-driver.py
python scripts/validate-monotonic-identities.py
python scripts/validate-fallible-runtime-paths.py
python scripts/validate-hardened-runtime-api.py
python scripts/validate-panic-free-runtime.py
python scripts/validate-rust-api-shapes.py
python scripts/validate-state-encapsulation.py
python scripts/validate-authority-encapsulation.py
python scripts/validate-transition-journals.py

cargo check --no-default-features
cargo check
cargo check --all-features
cargo check --no-default-features --features legacy-direct-startup
cargo check --no-default-features --features legacy-fail-stop-apis
cargo check --no-default-features --features legacy-direct-state-mutation
cargo check --no-default-features --features legacy-fail-stop-apis,legacy-direct-state-mutation
cargo check --features postgres-sync-driver
cargo test
cargo test --all-features
cargo test --features postgres-sync-driver
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
