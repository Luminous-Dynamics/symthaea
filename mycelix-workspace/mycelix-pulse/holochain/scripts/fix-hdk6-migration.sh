#!/bin/bash
# HDK 0.6 Migration Fix Script
# Fixes common API changes from older HDK versions to HDK 0.6

set -e
cd "$(dirname "$0")/.."

echo "Fixing HDK 0.6 migration issues..."

# 1. Fix remote_signal to send_remote_signal with proper arguments
echo "Fixing remote_signal calls..."
for f in zomes/*/coordinator/src/lib.rs; do
    if [ -f "$f" ]; then
        # Replace remote_signal(ExternIO::encode(X)?, vec[...]) with send_remote_signal(X, vec[...])
        sed -i 's/remote_signal(ExternIO::encode(\([^)]*\))?, \(vec\[[^]]*\]\))/send_remote_signal(\1, \2)/g' "$f"
        # Replace let _ = remote_signal with let _ = send_remote_signal
        sed -i 's/let _ = remote_signal(/let _ = send_remote_signal(/g' "$f"
        # Replace any remaining remote_signal calls
        sed -i 's/remote_signal(signal,/send_remote_signal(signal,/g' "$f"
        sed -i 's/remote_signal(encoded,/send_remote_signal(encoded,/g' "$f"
    fi
done

# 2. Fix link.target.into_action_hash() - need to clone first
echo "Fixing link.target moves..."
for f in zomes/*/coordinator/src/lib.rs; do
    if [ -f "$f" ]; then
        sed -i 's/link\.target\.into_action_hash()/link.target.clone().into_action_hash()/g' "$f"
    fi
done

# 3. Fix BTreeSet to HashSet for GrantedFunctions
echo "Fixing GrantedFunctions..."
for f in zomes/*/coordinator/src/lib.rs; do
    if [ -f "$f" ]; then
        sed -i 's/BTreeSet::from(\[/HashSet::from([/g' "$f"
        # Add HashSet import if not present
        if grep -q "HashSet::from" "$f" && ! grep -q "use std::collections::.*HashSet" "$f"; then
            sed -i 's/use std::collections::BTreeSet;/use std::collections::{BTreeSet, HashSet};/g' "$f"
            # If no BTreeSet import, add HashSet
            if ! grep -q "use std::collections::" "$f"; then
                sed -i '1a use std::collections::HashSet;' "$f"
            fi
        fi
    fi
done

# 4. Fix LinkTag::new with string references
echo "Fixing LinkTag usage..."
for f in zomes/*/coordinator/src/lib.rs; do
    if [ -f "$f" ]; then
        # LinkTag::new(&str) -> LinkTag::new(str.to_string())
        sed -i 's/LinkTag::new(&\([^)]*\))/LinkTag::new(\1.to_string())/g' "$f"
    fi
done

echo "Done. Run 'cargo check' to verify."
