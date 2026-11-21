#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MYCELIX_PACKAGE="$ROOT/mycelix/mycelix-position/lib/positioning"
TMP_PACKAGE="$ROOT/.ci/mycelix-positioning"
HARNESS="$ROOT/.ci/soma-mycelix-adapter-harness"

[[ -f "$MYCELIX_PACKAGE/Cargo.toml" ]] || {
  echo "Pinned Mycelix Position package missing: $MYCELIX_PACKAGE" >&2
  exit 2
}

rm -rf "$TMP_PACKAGE" "$HARNESS"
mkdir -p "$(dirname "$TMP_PACKAGE")" "$HARNESS/src"
cp -a "$MYCELIX_PACKAGE/." "$TMP_PACKAGE/"

# Avoid colliding with Symthaea's offline local package named "positioning".
sed -i 's/^name = "positioning"$/name = "mycelix-positioning"/' "$TMP_PACKAGE/Cargo.toml"
grep -q '^name = "mycelix-positioning"$' "$TMP_PACKAGE/Cargo.toml" || {
  echo "Expected Mycelix package name 'positioning' was not found" >&2
  exit 2
}

cat > "$HARNESS/Cargo.toml" <<'EOF'
[package]
name = "soma-mycelix-adapter-harness"
version = "0.1.0"
edition = "2024"
publish = false

[workspace]

[features]
default = []
positioning = ["dep:positioning"]

[dependencies]
positioning = { package = "mycelix-positioning", path = "../mycelix-positioning", optional = true }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
blake3 = "1.5"
EOF

cat > "$HARNESS/src/lib.rs" <<'EOF'
// Compile the exact SOMA adapter source against the real checked-out Mycelix
// positioning package without resolving Symthaea's unrelated root workspace.
#[cfg(feature = "positioning")]
#[path = "../../../crates/domains/symthaea-soma/src/positioning_v2_adapter.rs"]
pub mod positioning_v2_adapter;
EOF

cargo fmt --manifest-path "$HARNESS/Cargo.toml"
git -C "$ROOT" diff --check
# Mirror the production feature's intended optional-dependency behavior:
# feature-off must not compile or require Mycelix Position; feature-on must
# compile and test the exact adapter source against the pinned real package.
cargo check --manifest-path "$HARNESS/Cargo.toml" --no-default-features --all-targets
cargo clippy --manifest-path "$HARNESS/Cargo.toml" --features positioning --all-targets
cargo test --manifest-path "$HARNESS/Cargo.toml" --features positioning --all-targets
