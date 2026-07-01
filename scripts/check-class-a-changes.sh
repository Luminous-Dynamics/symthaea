#!/usr/bin/env bash
# Class A Change Detection — Governance Charter Enforcement
#
# Detects changes to safety-critical files and requires ADR references
# in the commit message. See: symthaea/docs/compliance/GOVERNANCE_CHARTER.md §3.2
#
# Usage:
#   As commit-msg hook: check-class-a-changes.sh <commit-msg-file>
#   As CI check:        check-class-a-changes.sh --ci

set -euo pipefail

# Class A files: safety-critical parameters per GOVERNANCE_CHARTER.md §3.1
CLASS_A_FILES=(
    "symthaea/src/cognitive_loop/thresholds.rs"
    "symthaea/src/cognitive_loop/ethics_engine.rs"
    "symthaea/src/safety/agent.rs"
    "crates/mycelix-bridge-common/src/consciousness_profile.rs"
    "crates/mycelix-bridge-common/src/consciousness_thresholds.rs"
)

# Class B files: consciousness-affecting
CLASS_B_FILES=(
    "symthaea/src/cognitive_loop/consciousness_engine.rs"
    "symthaea/src/cognitive_loop/substrate_manager.rs"
    "symthaea/src/cognitive_loop/calibration/"
    "symthaea-core/src/hdc/substrate_independence.rs"
    "symthaea-core/src/hdc/substrate_validation.rs"
)

# Approved commit prefixes for Class A changes
# Match prefixes with optional scope: safety: or safety(scope):
CLASS_A_PREFIXES="safety(\\(.*\\))?:|ethics(\\(.*\\))?:|emergency-safety(\\(.*\\))?:|governance(\\(.*\\))?:"

check_staged_files() {
    local class_a_changed=()
    local class_b_changed=()

    # Get staged files
    local staged_files
    staged_files=$(git diff --cached --name-only --diff-filter=ACMR 2>/dev/null || echo "")

    if [ -z "$staged_files" ]; then
        return 0
    fi

    for file in $staged_files; do
        for pattern in "${CLASS_A_FILES[@]}"; do
            if [[ "$file" == *"$pattern"* ]]; then
                class_a_changed+=("$file")
            fi
        done
        for pattern in "${CLASS_B_FILES[@]}"; do
            if [[ "$file" == *"$pattern"* ]]; then
                class_b_changed+=("$file")
            fi
        done
    done

    if [ ${#class_a_changed[@]} -gt 0 ]; then
        echo "CLASS A (Safety-Critical) files changed:"
        for f in "${class_a_changed[@]}"; do
            echo "  - $f"
        done
        return 1
    fi

    if [ ${#class_b_changed[@]} -gt 0 ]; then
        echo "CLASS B (Consciousness-Affecting) files changed:"
        for f in "${class_b_changed[@]}"; do
            echo "  - $f"
        done
        return 2
    fi

    return 0
}

check_commit_message() {
    local msg_file="$1"
    local msg
    msg=$(cat "$msg_file")

    # Check for approved prefix
    if echo "$msg" | head -1 | grep -qE "^($CLASS_A_PREFIXES)"; then
        return 0
    fi

    return 1
}

# --- CI mode ---
if [ "${1:-}" = "--ci" ]; then
    echo "Checking for Class A/B changes in PR..."

    # Compare against base branch
    base="${GITHUB_BASE_REF:-main}"
    changed_files=$(git diff --name-only "origin/$base"...HEAD 2>/dev/null || git diff --name-only HEAD~1 2>/dev/null || echo "")

    class_a_found=()
    class_b_found=()

    for file in $changed_files; do
        for pattern in "${CLASS_A_FILES[@]}"; do
            if [[ "$file" == *"$pattern"* ]]; then
                class_a_found+=("$file")
            fi
        done
        for pattern in "${CLASS_B_FILES[@]}"; do
            if [[ "$file" == *"$pattern"* ]]; then
                class_b_found+=("$file")
            fi
        done
    done

    exit_code=0

    if [ ${#class_a_found[@]} -gt 0 ]; then
        echo ""
        echo "================================================"
        echo "  CLASS A (Safety-Critical) Changes Detected"
        echo "================================================"
        echo ""
        echo "The following safety-critical files were modified:"
        for f in "${class_a_found[@]}"; do
            echo "  - $f"
        done
        echo ""
        echo "Per GOVERNANCE_CHARTER.md Section 3.2, Class A changes require:"
        echo "  1. An Architecture Decision Record (ADR) in symthaea/docs/compliance/adr/"
        echo "  2. Scientific citation or empirical evidence"
        echo "  3. Proptest or soak test evidence"
        echo "  4. Commit prefix: safety: | ethics: | emergency-safety: | governance:"
        echo ""

        # Check if any ADR was added or modified
        adr_changed=false
        for file in $changed_files; do
            if [[ "$file" == *"compliance/adr/"* && "$file" == *.md ]]; then
                adr_changed=true
                break
            fi
        done

        if ! $adr_changed; then
            echo "WARNING: No ADR found in this changeset."
            echo "  Consider adding: symthaea/docs/compliance/adr/ADR-NNN-description.md"
            # Warning only in CI — don't block PRs, just flag
        fi
    fi

    if [ ${#class_b_found[@]} -gt 0 ]; then
        echo ""
        echo "Class B (Consciousness-Affecting) files modified:"
        for f in "${class_b_found[@]}"; do
            echo "  - $f"
        done
        echo "  Ensure documentation is updated per GOVERNANCE_CHARTER.md Section 3.3."
    fi

    if [ ${#class_a_found[@]} -eq 0 ] && [ ${#class_b_found[@]} -eq 0 ]; then
        echo "No Class A or B files changed."
    fi

    exit $exit_code
fi

# --- Commit-msg hook mode ---
if [ -n "${1:-}" ] && [ -f "${1:-}" ]; then
    # Check if Class A files are staged
    result=0
    output=$(check_staged_files) || result=$?

    if [ $result -eq 1 ]; then
        # Class A change detected — check commit message
        echo ""
        echo "================================================"
        echo "  CLASS A Change Governance Check"
        echo "================================================"
        echo ""
        echo "$output"
        echo ""

        if ! check_commit_message "$1"; then
            echo "NOTICE: Class A commit detected without approved prefix."
            echo ""
            echo "Per GOVERNANCE_CHARTER.md Section 3.2, Class A changes should use:"
            echo "  safety: | ethics: | emergency-safety: | governance:"
            echo ""
            echo "Example: safety(thresholds): adjust MORAL_CONCERN_THRESHOLD (Haidt 2012)"
            echo ""
            echo "Proceeding with commit. Consider adding an ADR for this change."
            echo ""
        else
            echo "Approved Class A commit prefix detected."
        fi
    elif [ $result -eq 2 ]; then
        echo ""
        echo "$output"
        echo "Consider using prefix: consciousness: | substrate:"
        echo ""
    fi

    exit 0  # Don't block commits — inform only
fi

echo "Usage: $0 <commit-msg-file> | --ci"
