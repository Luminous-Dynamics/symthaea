#!/usr/bin/env bash
# Test feature profile compilation — validates that profile meta-features compile cleanly.
# Run from the symthaea/ directory or any location (uses script-relative path).
#
# Usage:
#   ./scripts/test-feature-profiles.sh              # Test all profiles
#   ./scripts/test-feature-profiles.sh voice reason  # Test profiles matching substring
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

profiles=(
    "profile-consciousness"
    "profile-voice"
    "profile-embodiment"
    "profile-distributed"
    "profile-reasoning"
    "profile-genesis"
)

# Filter to matching profiles if args provided
if [[ $# -gt 0 ]]; then
    filtered=()
    for arg in "$@"; do
        for p in "${profiles[@]}"; do
            if [[ "$p" == *"$arg"* ]]; then
                filtered+=("$p")
            fi
        done
    done
    profiles=("${filtered[@]}")
    if [[ ${#profiles[@]} -eq 0 ]]; then
        echo "No profiles matched: $*"
        echo "Available: profile-consciousness profile-voice profile-embodiment profile-distributed profile-reasoning profile-genesis"
        exit 1
    fi
fi

passed=0
failed=0
failed_names=()

echo "=== Symthaea Feature Profile Compilation Test ==="
echo "Crate: $CRATE_DIR"
echo "Profiles: ${#profiles[@]}"
echo ""

for profile in "${profiles[@]}"; do
    echo -n "  $profile ... "
    if cargo check --manifest-path "$CRATE_DIR/Cargo.toml" --lib --features "$profile" 2>/tmp/profile-check-err.log; then
        echo "OK"
        ((passed++))
    else
        echo "FAILED"
        ((failed++))
        failed_names+=("$profile")
        # Show last 10 lines of error for diagnosis
        tail -10 /tmp/profile-check-err.log | sed 's/^/    /'
    fi
done

echo ""
echo "=== Results: $passed passed, $failed failed ==="

if [[ $failed -gt 0 ]]; then
    echo "Failed profiles:"
    for name in "${failed_names[@]}"; do
        echo "  - $name"
    done
    exit 1
fi

echo "All profiles compile successfully."
