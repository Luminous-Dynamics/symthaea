#!/usr/bin/env bash
set -euo pipefail

# Security smoke tests for the hardened network surfaces.
#
# This script is intentionally opinionated:
# - Uses `--locked` so CI/dev can't silently change the lockfile.
# - Disables sccache by default (repo often sets RUSTC_WRAPPER=sccache).
#
# Env:
# - SECURITY_SMOKE_OFFLINE=1      -> adds `--offline` to cargo commands
# - SECURITY_SMOKE_CARGO_ARGS=... -> extra args appended to each cargo invocation

note() { echo "• $*" >&2; }

main() {
  local -a cargo_args
  cargo_args=(--locked)
  if [[ "${SECURITY_SMOKE_OFFLINE:-}" == "1" ]]; then
    cargo_args+=(--offline)
  fi
  if [[ -n "${SECURITY_SMOKE_CARGO_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    cargo_args+=(${SECURITY_SMOKE_CARGO_ARGS})
  fi

  note "Network regression scan..."
  scripts/security/network-regression-scan.sh

  note "Syntax-check JS gun server..."
  node --check mycelix-core/gun-p2p-server.js >/dev/null
  node --check mycelix-core/signaling-server.js >/dev/null
  node --check mycelix-core/websocket-bridge.js >/dev/null

  note "Cargo check: symthaea-spore ssh-relay + eval-api..."
  (cd symthaea && RUSTC_WRAPPER= cargo check "${cargo_args[@]}" -p symthaea-spore --features server --bin ssh-relay)
  (cd symthaea && RUSTC_WRAPPER= cargo check "${cargo_args[@]}" -p symthaea-spore --features server --bin eval-api)

  note "Cargo check: symthaea API/demo/holon bins..."
  (cd symthaea && RUSTC_WRAPPER= cargo check "${cargo_args[@]}" -p symthaea --features api_module --bin symthaea-api)
  (cd symthaea && RUSTC_WRAPPER= cargo check "${cargo_args[@]}" -p symthaea --features api_module --bin symthaea-demo)
  (cd symthaea && RUSTC_WRAPPER= cargo check "${cargo_args[@]}" -p symthaea --features api_module --bin symthaea-holon)

  note "Cargo check: mycelix-finance sidecar..."
  (cd mycelix-finance/sidecar && RUSTC_WRAPPER= cargo check "${cargo_args[@]}")

  note "OK: security smoke checks passed"
}

main "$@"
