#!/usr/bin/env bash
# Normalizes Cargo.toml paths to the symthaea standard
echo "Normalizing workspace paths..."

# Fix common path errors (nesting)
sed -i 's|path = "../symthaea-|path = "crates/|g' Cargo.toml
sed -i 's|path = "../../symthaea-|path = "crates/|g' Cargo.toml

echo "Workspace manifest normalized."
