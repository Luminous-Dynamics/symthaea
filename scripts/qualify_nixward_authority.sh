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

echo "-- source boundary --"
bash scripts/check-nixward-observation-boundary.sh

echo "-- compile nixward library --"
cargo check -p nixward --lib

echo "-- authority tests --"
cargo test -p nixward --lib action::authorization::tests

echo "-- typed service-domain tests --"
cargo test -p nixward --lib action::service_domain::tests

echo "-- service-manager boundary tests --"
cargo test -p nixward --lib action::service_manager::tests

echo "-- executor typed-service tests --"
cargo test -p nixward --lib action::executor::tests

echo "-- daemon approval-binding tests --"
cargo test -p nixward --bin nixward_daemon

echo "-- TUI approval-binding tests --"
cargo test -p nixward --lib tui::app::tests

echo "-- temporal tests --"
cargo test -p nixward --lib action::temporal::tests

echo "-- local approval tests --"
cargo test -p nixward --lib action::local_approval::tests

echo "-- local approval projection tests --"
cargo test -p nixward --lib action::local_approval_projection::tests

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
