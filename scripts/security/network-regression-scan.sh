#!/usr/bin/env bash
set -euo pipefail

# Lightweight security regression scan for "Stop the Bleeding" guarantees.
# Intended to be run locally and in CI (no network access required).

die() {
  echo "ERROR: $*" >&2
  exit 1
}

note() {
  echo "• $*" >&2
}

require_rg() {
  command -v rg >/dev/null 2>&1 || die "ripgrep (rg) is required"
}

assert_no_match() {
  local pattern="$1"
  local path="$2"
  if rg -n "${pattern}" "${path}" >/dev/null 2>&1; then
    echo "FAIL: ${path} matches forbidden pattern: ${pattern}" >&2
    rg -n "${pattern}" "${path}" >&2 || true
    return 1
  fi
  return 0
}

assert_match() {
  local pattern="$1"
  local path="$2"
  if ! rg -n "${pattern}" "${path}" >/dev/null 2>&1; then
    echo "FAIL: ${path} is missing required pattern: ${pattern}" >&2
    return 1
  fi
  return 0
}

main() {
  require_rg

  local failures=0

  note "Checking Symthaea SSH relay (localhost bind, token auth, host-key verification)..."
  assert_no_match "0\\.0\\.0\\.0" "symthaea/crates/symthaea-spore/src/bin/ssh_relay.rs" || failures=$((failures + 1))
  assert_match "127\\.0\\.0\\.1" "symthaea/crates/symthaea-spore/src/bin/ssh_relay.rs" || failures=$((failures + 1))
  assert_no_match "ServerCheckMethod::NoCheck" "symthaea/crates/symthaea-spore/src/bin/ssh_relay.rs" || failures=$((failures + 1))
  assert_match "\"auth\"" "symthaea/crates/symthaea-spore/src/bin/ssh_relay.rs" || failures=$((failures + 1))

  note "Checking Symthaea eval-api (no wildcard CORS + localhost default)..."
  assert_no_match "allow_origin\\(\\s*Any\\s*\\)" "symthaea/crates/symthaea-spore/src/bin/eval_api.rs" || failures=$((failures + 1))
  assert_no_match "CorsLayer::permissive\\(" "symthaea/crates/symthaea-spore/src/bin/eval_api.rs" || failures=$((failures + 1))
  assert_no_match "0\\.0\\.0\\.0" "symthaea/crates/symthaea-spore/src/bin/eval_api.rs" || failures=$((failures + 1))

  note "Checking Symthaea Holon daemon (localhost default + token on LAN bind)..."
  assert_no_match "0\\.0\\.0\\.0" "symthaea/src/bin/symthaea-holon.rs" || failures=$((failures + 1))
  assert_match "HOLON_TOKEN" "symthaea/src/bin/symthaea-holon.rs" || failures=$((failures + 1))
  assert_match "holon_auth_middleware" "symthaea/src/api/holon.rs" || failures=$((failures + 1))

  note "Checking Mycelix gun server (localhost default + explicit CORS allowlist)..."
  assert_match "127\\.0\\.0\\.1" "mycelix-core/gun-p2p-server.js" || failures=$((failures + 1))
  assert_no_match "origin\\s*:\\s*['\\\"]\\*['\\\"]" "mycelix-core/gun-p2p-server.js" || failures=$((failures + 1))

  note "Checking Mycelix signaling server (localhost default + Origin gate + token-on-LAN)..."
  assert_match "SIGNAL_HOST" "mycelix-core/signaling-server.js" || failures=$((failures + 1))
  assert_match "127\\.0\\.0\\.1" "mycelix-core/signaling-server.js" || failures=$((failures + 1))
  assert_no_match "server\\.listen\\(PORT\\)" "mycelix-core/signaling-server.js" || failures=$((failures + 1))
  assert_match "Origin not allowed" "mycelix-core/signaling-server.js" || failures=$((failures + 1))
  assert_match "token" "mycelix-core/signaling-server.js" || failures=$((failures + 1))

  note "Checking Mycelix WebSocket bridge (localhost default + Origin gate + token-on-LAN)..."
  assert_match "BRIDGE_HOST" "mycelix-core/websocket-bridge.js" || failures=$((failures + 1))
  assert_match "127\\.0\\.0\\.1" "mycelix-core/websocket-bridge.js" || failures=$((failures + 1))
  assert_no_match "server\\.listen\\(BRIDGE_PORT\\)" "mycelix-core/websocket-bridge.js" || failures=$((failures + 1))
  assert_match "Origin not allowed" "mycelix-core/websocket-bridge.js" || failures=$((failures + 1))
  assert_match "token" "mycelix-core/websocket-bridge.js" || failures=$((failures + 1))

  note "Checking installer ISO (no static root password, no Avahi broadcast, firewall enabled)..."
  # The ISO builder imports the hardened module; validate both.
  assert_match "installer-iso-module\\.nix" "symthaea/nix/installer-iso.nix" || failures=$((failures + 1))
  assert_no_match "initialPassword" "symthaea/nix/modules/installer-iso-module.nix" || failures=$((failures + 1))
  assert_no_match "hashedPassword" "symthaea/nix/modules/installer-iso-module.nix" || failures=$((failures + 1))
  assert_no_match "services\\.avahi\\.enable\\s*=\\s*true" "symthaea/nix/modules/installer-iso-module.nix" || failures=$((failures + 1))
  assert_match "networking\\.firewall\\.enable\\s*=\\s*true" "symthaea/nix/modules/installer-iso-module.nix" || failures=$((failures + 1))

  if [[ "${failures}" -ne 0 ]]; then
    die "${failures} regression check(s) failed"
  fi

  note "OK: network regression scan passed"
}

main "$@"
