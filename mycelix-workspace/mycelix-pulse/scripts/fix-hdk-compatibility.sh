#!/bin/bash
# Fix HDK/HDI Compatibility Issues
# =================================
# This script fixes common compatibility issues when upgrading
# between HDK/HDI versions.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOLOCHAIN_DIR="$PROJECT_ROOT/holochain"

echo "Fixing HDK/HDI compatibility issues..."

# 1. Remove required_validation_type from all entry_type attributes
echo "Removing deprecated required_validation_type attributes..."
find "$HOLOCHAIN_DIR/zomes" -name "*.rs" -type f -exec sed -i \
    's/, required_validation_type = "[^"]*"//g' {} \;

# 2. Ensure all integrity zomes have holochain_serialized_bytes
echo "Updating Cargo.toml files..."
for toml in "$HOLOCHAIN_DIR"/zomes/*/integrity/Cargo.toml; do
    if [ -f "$toml" ]; then
        # Check if holochain_serialized_bytes is missing
        if ! grep -q "holochain_serialized_bytes" "$toml"; then
            # Add it after hdi
            sed -i '/^hdi = /a holochain_serialized_bytes = { workspace = true }' "$toml"
        fi
    fi
done

# 3. Fix WasmError::Guest -> wasm_error!(WasmErrorInner::Guest(...))
echo "Fixing WasmError patterns..."
find "$HOLOCHAIN_DIR" -name "*.rs" -type f -exec sed -i \
    's/WasmError::Guest(\([^)]*\))/wasm_error!(WasmErrorInner::Guest(\1))/g' {} \;

# 4. Fix EntryCreationAction type mismatches in validation
# (Create -> EntryCreationAction where needed)

echo "Done. Run 'cargo check' to verify fixes."
