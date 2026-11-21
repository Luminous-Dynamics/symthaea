#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MYCELIX_PACKAGE="$ROOT/mycelix/mycelix-position/lib/positioning"
TMP_PACKAGE="$ROOT/.ci/mycelix-positioning"
SOMA_MANIFEST="$ROOT/crates/domains/symthaea-soma/Cargo.toml"
BACKUP="${RUNNER_TEMP:-/tmp}/symthaea-soma-Cargo.toml.backup"
LOCKFILE="$ROOT/Cargo.lock"
LOCK_BACKUP="${RUNNER_TEMP:-/tmp}/symthaea-cargo.lock.backup"

[[ -f "$MYCELIX_PACKAGE/Cargo.toml" ]] || {
  echo "Pinned Mycelix Position package missing: $MYCELIX_PACKAGE" >&2
  exit 2
}

[[ -f "$SOMA_MANIFEST" ]] || {
  echo "SOMA manifest missing: $SOMA_MANIFEST" >&2
  exit 2
}

rm -rf "$TMP_PACKAGE"
mkdir -p "$(dirname "$TMP_PACKAGE")"
cp -a "$MYCELIX_PACKAGE/." "$TMP_PACKAGE/"

# Avoid colliding with Symthaea's offline local package named "positioning".
sed -i 's/^name = "positioning"$/name = "mycelix-positioning"/' "$TMP_PACKAGE/Cargo.toml"
grep -q '^name = "mycelix-positioning"$' "$TMP_PACKAGE/Cargo.toml" || {
  echo "Expected Mycelix package name 'positioning' was not found" >&2
  exit 2
}

cp "$SOMA_MANIFEST" "$BACKUP"
cp "$LOCKFILE" "$LOCK_BACKUP"

cleanup() {
  cp "$BACKUP" "$SOMA_MANIFEST"
  cp "$LOCK_BACKUP" "$LOCKFILE"
}
trap cleanup EXIT

TMP_MANIFEST="${SOMA_MANIFEST}.ci-tmp"
awk '
  /^\[dependencies\]$/ {
    print
    print "positioning = { package = \"mycelix-positioning\", path = \"../../../.ci/mycelix-positioning\", optional = true }"
    next
  }
  { print }
' "$SOMA_MANIFEST" > "$TMP_MANIFEST"
mv "$TMP_MANIFEST" "$SOMA_MANIFEST"

sed -i 's/^positioning = \[\]$/positioning = ["dep:positioning"]/' "$SOMA_MANIFEST"
grep -q '^positioning = \["dep:positioning"\]$' "$SOMA_MANIFEST" || {
  echo "Could not inject optional Mycelix dependency into SOMA positioning feature" >&2
  exit 2
}

cd "$ROOT"
cargo fmt --package symthaea-soma --manifest-path "$ROOT/Cargo.toml"
cargo clippy --package symthaea-soma --features positioning --all-targets --manifest-path "$ROOT/Cargo.toml"
cargo test --package symthaea-soma --features positioning --all-targets --manifest-path "$ROOT/Cargo.toml"
