#!/bin/bash
set -e
SYMTHAEA_ROOT="$(git rev-parse --show-toplevel)/symthaea"
STAGING_DIR=$(mktemp -d)

# Manifest and Core
mkdir -p "$STAGING_DIR/symthaea-core"
cp -r "$SYMTHAEA_ROOT/symthaea-core/src" "$STAGING_DIR/symthaea-core/"
cp "$SYMTHAEA_ROOT/symthaea-core/Cargo.toml" "$STAGING_DIR/symthaea-core/"
cp "$SYMTHAEA_ROOT/Cargo.public.toml" "$STAGING_DIR/Cargo.toml"

# Docs/License
cp "$SYMTHAEA_ROOT/LICENSE" "$STAGING_DIR/LICENSE"
cp "$SYMTHAEA_ROOT/README.md" "$STAGING_DIR/README.md"
cp -r "$SYMTHAEA_ROOT/docs" "$STAGING_DIR/docs"
cp -r "$SYMTHAEA_ROOT/papers" "$STAGING_DIR/papers"

echo "--- Export Audit ---"
cd "$STAGING_DIR"
cargo metadata --no-deps --format-version 1 > /dev/null
echo "Cargo metadata PASS"
cargo check -p symthaea-core --lib --offline
echo "Cargo check symthaea-core PASS"
