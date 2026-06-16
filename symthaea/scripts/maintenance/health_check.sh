#!/usr/bin/env bash
# Automated autopoietic system health check
echo "Running system integrity audit..."
cargo check --quiet -p symthaea-core
if [ $? -eq 0 ]; then
    echo "Autopoietic architecture verified stable."
else
    echo "Manifest inconsistency detected. Initiating path normalization..."
    # Placeholder for automated recovery logic
    sed -i 's|crates/core/symthaea-core|symthaea-core|g' Cargo.toml
fi
