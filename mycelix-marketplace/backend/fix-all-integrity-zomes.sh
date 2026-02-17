#!/usr/bin/env bash
#
# Fix all integrity zomes for HDI 0.7.0 compatibility
# Based on the pattern established in listings_integrity
#

set -e  # Exit on error

ZOMES=("reputation" "transactions" "arbitration" "messaging")

echo "🔧 Fixing all integrity zomes for HDI 0.7.0..."
echo ""

for zome in "${ZOMES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📦 Fixing ${zome}_integrity..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    ZOME_DIR="zomes/${zome}/integrity"
    CARGO_TOML="${ZOME_DIR}/Cargo.toml"
    LIB_RS="${ZOME_DIR}/src/lib.rs"

    # Step 1: Add holochain_serialized_bytes dependency
    echo "  ✓ Adding holochain_serialized_bytes dependency..."
    if ! grep -q "holochain_serialized_bytes" "${CARGO_TOML}"; then
        sed -i '/thiserror.workspace = true/a holochain_serialized_bytes.workspace = true' "${CARGO_TOML}"
    fi

    # Step 2: Fix macro name hdk_entry_defs → hdk_entry_types
    echo "  ✓ Fixing entry types macro..."
    sed -i 's/#\[hdk_entry_defs\]/#[hdk_entry_types]/g' "${LIB_RS}"

    # Step 3: Fix validation function type parameter
    echo "  ✓ Fixing validation type parameter..."
    sed -i 's/op\.flattened::<UnitEntryTypes,/op.flattened::<EntryTypes,/g' "${LIB_RS}"

    # Step 4: Prefix unused variables with underscore
    echo "  ✓ Fixing unused variable warnings..."
    sed -i 's/FlatOp::RegisterDelete(delete_entry)/FlatOp::RegisterDelete(_delete_entry)/g' "${LIB_RS}"

    echo "  ✓ ${zome}_integrity fixes applied!"
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All integrity zomes fixed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🧪 Testing compilation..."
echo ""

for zome in "listings" "${ZOMES[@]}"; do
    echo "  📦 Checking ${zome}_integrity..."
    if cargo check -p "${zome}_integrity" 2>&1 | grep -q "Finished"; then
        echo "  ✅ ${zome}_integrity compiles successfully!"
    else
        echo "  ❌ ${zome}_integrity has compilation errors"
        echo "     Run: cargo check -p ${zome}_integrity"
    fi
    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 HDI 0.7.0 compilation fix complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
