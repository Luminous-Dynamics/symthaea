#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOMA_MANIFEST="$ROOT/crates/domains/symthaea-soma/Cargo.toml"
HARNESS="$ROOT/.ci/soma-mycelix-adapter-harness"

# Extract the immutable facade revision from the actual production manifest.
MYCELIX_REV="$(sed -n 's/.*package = "mycelix-positioning-v2".*rev = "\([0-9a-f]\{40\}\)".*/\1/p' "$SOMA_MANIFEST")"
[[ "$MYCELIX_REV" =~ ^[0-9a-f]{40}$ ]] || {
  echo "Production SOMA manifest must pin a 40-character Mycelix Position facade commit" >&2
  exit 2
}

rm -rf "$HARNESS"
mkdir -p "$HARNESS/src"

cat > "$HARNESS/Cargo.toml" <<EOF
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
positioning = { package = "mycelix-positioning-v2", git = "https://github.com/Luminous-Dynamics/mycelix", rev = "$MYCELIX_REV", optional = true }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
blake3 = "1.5"
EOF

cat > "$HARNESS/src/lib.rs" <<'EOF'
// Compile the exact committed SOMA adapter source against the exact same
// pinned Git package that production symthaea-soma declares.
#[cfg(feature = "positioning")]
#[path = "../../../crates/domains/symthaea-soma/src/positioning_v2_adapter.rs"]
pub mod positioning_v2_adapter;
EOF

cargo fmt --manifest-path "$HARNESS/Cargo.toml"
git -C "$ROOT" diff --check

# Check the disabled feature path, then compile/lint/test the real Git package
# and the exact adapter source through the production dependency specification.
cargo check --manifest-path "$HARNESS/Cargo.toml" --no-default-features --all-targets
cargo clippy --manifest-path "$HARNESS/Cargo.toml" --features positioning --all-targets
cargo test --manifest-path "$HARNESS/Cargo.toml" --features positioning --all-targets
