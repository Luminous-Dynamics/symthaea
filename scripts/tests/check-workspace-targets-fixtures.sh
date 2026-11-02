#!/usr/bin/env bash
# Adversarial fixtures for check-workspace-targets.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/scripts"
mkdir -p "$tmp/crates/core/good/src"
mkdir -p "$tmp/crates/core/symthaea/examples"
mkdir -p "$tmp/crates/core/symthaea-observation-attestation/src"
cp "$repo_root/scripts/check-workspace-targets.sh" "$tmp/scripts/check-workspace-targets.sh"

cat > "$tmp/Cargo.toml" <<'EOF'
[workspace]
members = ["crates/core/*"]
EOF

cat > "$tmp/crates/core/good/Cargo.toml" <<'EOF'
[package]
name = "good"
version = "0.1.0"
edition = "2024"
EOF
printf 'pub fn fixture() {}\n' > "$tmp/crates/core/good/src/lib.rs"
: > "$tmp/crates/core/symthaea/examples/unused.rs"
: > "$tmp/crates/core/symthaea-observation-attestation/src/lib.rs"

# These two examples/source-only directories are explicit exclusions. They
# have no Cargo.toml and must not prevent a valid workspace preflight.
bash "$tmp/scripts/check-workspace-targets.sh" > "$tmp/positive.log"
grep -F "MISSING MANIFEST" "$tmp/positive.log" && {
  echo "FAIL: an explicitly excluded directory was treated as a workspace member" >&2
  exit 1
}

# An arbitrary unmatched directory without a manifest must fail with its exact
# path instead of being silently skipped.
mkdir -p "$tmp/crates/core/unlisted-examples-only"
if bash "$tmp/scripts/check-workspace-targets.sh" > "$tmp/negative.log" 2>&1; then
  echo "FAIL: workspace-glob directory without Cargo.toml was accepted" >&2
  exit 1
fi
grep -F "MISSING MANIFEST: crates/core/unlisted-examples-only" "$tmp/negative.log" >/dev/null || {
  cat "$tmp/negative.log" >&2
  echo "FAIL: preflight did not identify the exact missing-manifest path" >&2
  exit 1
}

echo "OK: excluded source-only fixtures pass; unexcluded missing-manifest fixture fails closed."
