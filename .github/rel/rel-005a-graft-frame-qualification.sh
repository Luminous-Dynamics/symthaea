#!/usr/bin/env bash
set -euo pipefail

phase="${1:-all}"
test_file="crates/core/symthaea-fep/tests/rel_graft_frame.rs"

run_fmt() {
  rustfmt --edition 2024 --check "${test_file}"
}

run_test() {
  cargo test --locked -p symthaea-fep --test rel_graft_frame -- --nocapture
}

run_clippy() {
  cargo clippy --locked -p symthaea-fep --test rel_graft_frame -- -D warnings
}

case "${phase}" in
  fmt)
    run_fmt
    ;;
  test)
    run_test
    ;;
  clippy)
    run_clippy
    ;;
  all)
    run_fmt
    run_test
    run_clippy
    ;;
  *)
    echo "usage: $0 [fmt|test|clippy|all]" >&2
    exit 64
    ;;
esac
