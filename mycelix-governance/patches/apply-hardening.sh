#!/usr/bin/env bash
# Apply anti-tyranny governance hardening from known-good commits.
#
# Usage: cd /srv/luminous-dynamics && bash mycelix-governance/patches/apply-hardening.sh
#
# This script extracts the hardened versions of governance zome files
# from commits on previous branch points and applies them to the current
# working tree. Run from the repo root.
#
# After applying, the execution integrity will need the participation
# insurance patch applied separately (it's newer than the commits).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "=== Applying Anti-Tyranny Governance Hardening ==="
echo ""

# File -> commit mapping (latest hardened version of each file)
apply() {
  local commit="$1"
  local path="$2"
  local content
  content=$(git show "${commit}:${path}" 2>/dev/null || true)
  if [ -z "$content" ]; then
    echo "  SKIP $path (not in $commit)"
    return 1
  fi
  git show "${commit}:${path}" > "$path"
  echo "  OK   $path (from ${commit:0:7})"
}

# Round 1 commit: 2667196cf
apply 2667196cf "mycelix-governance/zomes/execution/integrity/src/lib.rs"

# Round 2 commit: 048354b7d
apply 048354b7d "mycelix-governance/zomes/execution/coordinator/src/lib.rs"
apply 048354b7d "mycelix-governance/zomes/councils/integrity/src/lib.rs"
apply 048354b7d "mycelix-governance/zomes/councils/coordinator/src/lib.rs"
apply 048354b7d "mycelix-governance/zomes/voting/coordinator/src/lib.rs"

# Round 3 commit: a6ebfb27f
apply a6ebfb27f "mycelix-governance/zomes/voting/integrity/src/lib.rs"

echo ""
echo "Verify:"
echo "  cd mycelix-governance"
echo "  cargo test -p execution_integrity -p execution -p councils_integrity \\"
echo "    -p councils -p voting_integrity -p voting --lib"
echo ""
echo "Note: Apply participation insurance separately (adaptive_override_threshold)"
echo "  See ANTI_TYRANNY_DESIGN.md section on Participation Insurance"
