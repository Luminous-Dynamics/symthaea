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
mkdir -p "$tmp/crates/core/symthaea-iit/tests"
: > "$tmp/crates/core/symthaea-iit/tests/harness.rs"
cp "$repo_root/scripts/check-workspace-targets.sh" "$tmp/scripts/check-workspace-targets.sh"

cat > "$tmp/Cargo.toml" <<'EOF'
[workspace]
members = ["crates/core/*"]
exclude = [
    "crates/core/nixward-web",
    "crates/core/symthaea",
    "crates/core/symthaea-observation-attestation",
    "crates/core/symthaea-iit",
    "crates/core/symthaea-zkproof/methods",
    "crates/core/symthaea-zkproof/host",
    "crates/domains/symthaea-spore/fuzz",
    "crates/domains/spark-engine",
    "crates/domains/symthaea-lab",
    "crates/domains/symthaea-perception",
]
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

# A script-only exclusion must not hide an unexcluded directory from Cargo.
sed '/"crates\/domains\/symthaea-perception",/d' "$tmp/Cargo.toml" > "$tmp/Cargo.mismatch.toml"
cp "$tmp/Cargo.mismatch.toml" "$tmp/Cargo.toml"
if bash "$tmp/scripts/check-workspace-targets.sh" > "$tmp/mismatch.log" 2>&1; then
  echo "FAIL: mismatched exclusion mirror was accepted" >&2
  exit 1
fi
grep -F "WORKSPACE EXCLUSION MIRROR MISMATCH" "$tmp/mismatch.log" >/dev/null || {
  cat "$tmp/mismatch.log" >&2
  echo "FAIL: exclusion mirror mismatch was not reported" >&2
  exit 1
}

echo "OK: excluded source-only fixtures pass; unexcluded missing-manifest fixture fails closed."
