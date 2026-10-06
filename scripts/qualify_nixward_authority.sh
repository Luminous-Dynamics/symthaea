#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "== Nixward authority focused qualification =="
echo "HEAD: $(git rev-parse HEAD)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

echo "-- source-boundary scanner prerequisite --"
if ! command -v rg >/dev/null 2>&1; then
  echo "ERROR: ripgrep (rg) is required for CROSS-015/CROSS-022 source qualification" >&2
  exit 1
fi
echo "ripgrep: $(rg --version | head -n1)"

echo "-- focused Nixward formatting --"
# Validate formatting only for the authority surfaces maintained by this
# qualification lane. The wider Nixward tree contains historical formatting
# drift that is outside this safety PR and should not gate authority evidence.
rustfmt --edition 2024 --check \
  crates/core/nixward/src/action/authorization.rs \
  crates/core/nixward/src/action/executor.rs \
  crates/core/nixward/src/action/config_writer.rs \
  crates/core/nixward/src/action/post_state.rs \
  crates/core/nixward/src/action/systemd_observer.rs \
  crates/core/nixward/src/action/systemd_mutation.rs

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

echo "-- post-state evidence tests --"
cargo test -p nixward --lib action::post_state::tests
echo "-- read-only systemd D-Bus observer tests --"
cargo test -p nixward --features systemd-observer --lib action::systemd_observer::tests
echo "-- typed systemd D-Bus mutation transport tests --"
cargo test -p nixward --features systemd-mutation --lib action::systemd_mutation::tests

echo "-- executor typed-service tests --"
cargo test -p nixward --lib action::executor::tests

echo "-- config-writer currentness tests --"
cargo test -p nixward --lib action::config_writer::tests

echo "-- daemon approval-binding tests --"
cargo test -p nixward --features daemon --bin nixward-daemon

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
