#!/usr/bin/env bash
# Adversarial fixtures for the workspace manifest preflight.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$tmp/scripts" "$tmp/crates/core/good/src"
cp "$repo_root/scripts/check-workspace-targets.sh" "$tmp/scripts/check-workspace-targets.sh"

# Mirror every current Cargo exclusion plus all nine source/docs/test-only paths
# from issue #13834. The latter remain research material, not Cargo packages.
mkdir -p \
  "$tmp/crates/core/symthaea/examples" \
  "$tmp/crates/core/symthaea-iit/tests" \
  "$tmp/crates/core/symthaea-observation-attestation/src" \
  "$tmp/crates/domains/symthaea-architecture-engineering/src" \
  "$tmp/crates/domains/symthaea-dishwasher-design/src" \
  "$tmp/crates/domains/symthaea-dishwasher-design/docs" \
  "$tmp/crates/domains/symthaea-experiment-design/docs" \
  "$tmp/crates/domains/symthaea-marine-geometry/src" \
  "$tmp/crates/domains/symthaea-rea-compat/src" \
  "$tmp/crates/domains/symthaea-rea-evidence-verifier"
: > "$tmp/crates/domains/symthaea-rea-evidence-verifier/README.md"

cat > "$tmp/Cargo.toml" <<'EOF'
[workspace]
members = ["crates/core/*", "crates/domains/*", "crates/bridges/*"]
exclude = [
    "crates/core/nixward-web",
    "crates/core/symthaea-zkproof/methods",
    "crates/core/symthaea-zkproof/host",
    "crates/domains/symthaea-spore/fuzz",
    "crates/domains/spark-engine",
    "crates/domains/symthaea-lab",
    "crates/domains/symthaea-perception",
    "crates/core/symthaea",
    "crates/core/symthaea-iit",
    "crates/core/symthaea-observation-attestation",
    "crates/domains/symthaea-architecture-engineering",
    "crates/domains/symthaea-dishwasher-design",
    "crates/domains/symthaea-experiment-design",
    "crates/domains/symthaea-marine-geometry",
    "crates/domains/symthaea-rea-compat",
    "crates/domains/symthaea-rea-evidence-verifier",
]
EOF

cat > "$tmp/crates/core/good/Cargo.toml" <<'EOF'
[package]
name = "good"
version = "0.1.0"
edition = "2024"
EOF
printf '%s\\n' 'pub fn fixture() {}' > "$tmp/crates/core/good/src/lib.rs"

# Positive control: a real package is admitted, all nine staging directories
# are explicitly excluded, and the seven pre-existing exclusions remain mirrored.
bash "$tmp/scripts/check-workspace-targets.sh" > "$tmp/positive.log"
if grep -F "MISSING MANIFEST" "$tmp/positive.log"; then
  echo "FAIL: an explicitly excluded directory was treated as a workspace member" >&2
  exit 1
fi

# Negative control: a new directory admitted by a workspace glob without a
# manifest must fail and identify the exact path.
mkdir -p "$tmp/crates/core/unlisted-examples-only"
if bash "$tmp/scripts/check-workspace-targets.sh" > "$tmp/missing.log" 2>&1; then
  echo "FAIL: workspace-glob directory without Cargo.toml was accepted" >&2
  exit 1
fi
grep -F "MISSING MANIFEST: crates/core/unlisted-examples-only" "$tmp/missing.log" >/dev/null || {
  cat "$tmp/missing.log" >&2
  echo "FAIL: preflight did not identify the exact missing-manifest path" >&2
  exit 1
}

# Negative control: changing Cargo's exclusion set without changing the script
# mirror must fail closed.
grep -Fv '"crates/domains/symthaea-perception",' "$tmp/Cargo.toml" > "$tmp/Cargo.mismatch.toml"
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

echo "OK: all nine staging exclusions, missing-manifest detection and exclusion-mirror drift controls are present."
