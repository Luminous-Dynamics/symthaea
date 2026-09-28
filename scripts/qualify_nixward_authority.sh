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

echo "-- authority tests --"
cargo test -p nixward --lib action::authorization::tests

echo "-- temporal tests --"
cargo test -p nixward --lib action::temporal::tests

echo "-- local approval tests --"
cargo test -p nixward --lib action::local_approval::tests

echo "-- daemon incarnation tests --"
cargo test -p nixward --lib action::daemon_incarnation::tests

echo "-- approver evidence tests --"
cargo test -p nixward --lib action::approver_evidence::tests

echo "-- local approval IPC peer-credential tests --"
cargo test -p nixward --lib action::local_approval_ipc::tests

echo "-- local approval submission/admission tests --"
cargo test -p nixward --lib action::local_approval_submission::tests

echo "-- local approval single-use store tests --"
cargo test -p nixward --lib action::local_approval_store::tests

if [[ "$(uname -s)" == "Linux" ]]; then
  echo "-- protected local approval socket tests --"
  cargo test -p nixward --lib action::local_approval_socket::tests

  echo "-- daemon-local approval runtime composition tests --"
  cargo test -p nixward --lib action::local_approval_runtime::tests
fi

echo "Nixward authority focused qualification: PASS"
