#!/bin/bash
# Mycelix-Core Post-Start Script
# Runs every time the dev container starts

# Ensure Rust toolchain is up to date
rustup update stable --no-self-update 2>/dev/null &

# Check if any background services should be started
if [ -f ".devcontainer/services.conf" ]; then
    source .devcontainer/services.conf
fi

# Display helpful information
echo ""
echo "Mycelix-Core development environment ready."
echo "Run 'demo' or 'fl-demo' to see the system in action."
echo ""
