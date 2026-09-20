#!/usr/bin/env bash
set -euo pipefail

cargo fmt --all -- --check
cargo check -p nixward --lib
cargo test -p nixward --lib action::nix_mutation::tests
