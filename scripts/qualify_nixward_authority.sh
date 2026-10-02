#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Nixward authority focused qualification =="
echo "HEAD: $(git rev-parse HEAD)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "-- Nixward formatting (workspace-independent) --"
# The repository workspace is intentionally broader than Nixward and can be
# structurally divergent on long-lived safety branches. Running cargo fmt
# --all therefore makes this focused lane depend on unrelated workspace member
# presence. Format the governed crate sources directly instead.
mapfile -d '' nixward_rust_files < <(
  find crates/core/nixward/src crates/core/nixward/tests -type f -name '*.rs' -print0
)
if ((${#nixward_rust_files[@]} == 0)); then
  echo "no Nixward Rust sources found" >&2
  exit 1
fi
rustfmt --edition 2024 --check "${nixward_rust_files[@]}"

echo "-- source boundary --"
bash scripts/check-nixward-observation-boundary.sh

echo "-- legacy custom mutation fence --"
if rg -n "fn approved_intent_digest|let approved_digest|approved_intent_digest\\(" crates/core/nixward/src/bin/nixward_daemon.rs; then
  echo "legacy verdict-file approval parser still present in daemon" >&2
  exit 1
fi
if rg -n "execute_confirmed\\(" crates/core/nixward/src/bin/nixward_daemon.rs; then
  echo "daemon still reaches legacy confirmed executor directly" >&2
  exit 1
fi
rg -n "NixOSCommand::ConfigPatch" crates/core/nixward/src/{action/executor.rs,bin/nixward_daemon.rs}
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

echo "-- config-writer currentness tests --"
cargo test -p nixward --lib action::config_writer::tests

echo "-- daemon approval-binding tests --"
cargo test -p nixward --bin nixward_daemon

echo "-- TUI approval-binding tests --"
cargo test -p nixward --lib tui::app::tests

echo "-- daemon snapshot IPC compatibility tests --"
cargo test -p nixward --test daemon_state_integration

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
