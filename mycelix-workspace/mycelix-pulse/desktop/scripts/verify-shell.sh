#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/src-tauri/tauri.conf.json"
MAIN="$ROOT/src-tauri/src/main.rs"
PACKAGE_JSON="$ROOT/package.json"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

pass() {
  echo "OK: $1"
}

grep -q '"frontendDist": "../../apps/leptos/dist"' "$CONFIG" || fail "Tauri config does not point at apps/leptos/dist"
pass "Tauri config points at apps/leptos/dist"

grep -q '"devUrl": "http://127.0.0.1:1420"' "$CONFIG" || fail "Tauri config does not use the Leptos dev URL"
pass "Tauri config uses the Leptos dev URL"

grep -q '"identifier": "com.mycelix.mail"' "$CONFIG" || fail "Tauri config changed the compatibility bundle identifier"
pass "Tauri config keeps the compatibility bundle identifier"

grep -q 'frontend: "apps/leptos"' "$MAIN" || fail "Desktop runtime info no longer reports apps/leptos as the canonical frontend"
pass "Desktop runtime info reports apps/leptos"

grep -q '"dev": "cargo tauri dev --config src-tauri/tauri.conf.json"' "$PACKAGE_JSON" || fail "Desktop dev script is not using cargo tauri"
pass "Desktop dev script uses cargo tauri"

grep -q '"build": "cargo tauri build --config src-tauri/tauri.conf.json"' "$PACKAGE_JSON" || fail "Desktop build script is not using cargo tauri"
pass "Desktop build script uses cargo tauri"

echo "Desktop shell smoke verification passed"
