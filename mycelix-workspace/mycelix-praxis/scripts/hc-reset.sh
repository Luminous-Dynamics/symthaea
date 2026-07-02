#!/usr/bin/env bash
# Reset Holochain development environment (clears local chains and state)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${YELLOW}⚠  Holochain Development Environment Reset${NC}"
echo
echo "This will:"
echo "  • Delete all local Holochain sandbox data"
echo "  • Clear conductor state"
echo "  • Remove agent keys and source chains"
echo
echo -e "${RED}This action cannot be undone!${NC}"
echo
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo
echo -e "${YELLOW}▶ Resetting Holochain environment...${NC}"

# Kill any running conductors
if pgrep -f "holochain" > /dev/null; then
    echo -e "${YELLOW}  Stopping running conductors...${NC}"
    pkill -f "holochain" || true
    echo -e "${GREEN}✓ Conductors stopped${NC}"
fi

# Remove sandbox directories
SANDBOX_PATHS=(
    "$HOME/.config/holochain"
    "$HOME/.local/share/holochain"
    "$HOME/.cache/holochain"
    "./workdir"
    "./.hc*"
)

for path in "${SANDBOX_PATHS[@]}"; do
    if [ -e "$path" ]; then
        echo -e "${YELLOW}  Removing $path...${NC}"
        rm -rf "$path"
        echo -e "${GREEN}✓ Removed $path${NC}"
    fi
done

echo
echo -e "${GREEN}✓ Holochain environment reset complete${NC}"
echo
echo "To start fresh:"
echo "  make dev"
echo
