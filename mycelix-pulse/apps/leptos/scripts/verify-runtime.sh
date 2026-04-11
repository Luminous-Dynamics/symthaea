#!/usr/bin/env bash
set -euo pipefail

ROOT_URL="${1:-http://127.0.0.1:8117}"

html="$(curl -fsS "${ROOT_URL}/")"
config="$(curl -fsS "${ROOT_URL}/conductor-config.json")"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

pass() {
  echo "OK: $1"
}

[[ "$html" == *"Mycelix Pulse"* ]] || fail "root HTML does not contain Mycelix Pulse branding"
pass "root HTML contains Mycelix Pulse branding"

[[ "$html" == *"/conductor-config.json"* ]] || fail "root HTML does not reference the static conductor config fallback"
pass "root HTML references static conductor config fallback"

[[ "$html" == *"__HC_CALL_ZOME"* ]] || fail "root HTML does not expose the browser zome bridge"
pass "root HTML exposes browser zome bridge"

[[ "$config" == *"\"admin_port\":"* ]] || fail "conductor-config.json is missing admin_port"
[[ "$config" == *"\"app_port\":"* ]] || fail "conductor-config.json is missing app_port"
pass "conductor-config.json includes app and admin ports"

echo "Runtime verification passed for ${ROOT_URL}"
