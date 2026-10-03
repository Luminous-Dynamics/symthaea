#!/usr/bin/env bash
set -euo pipefail

# Security-sensitive source must never trust domains we no longer control.
legacy_regex='mycelix\\.net|mycelix\\.com|mycelix\\.dev|relationalharmonics\\.org|luminousdynamics\\.org'

# Keep the scan narrow so historical/archive material can retain provenance.
scan_paths=(
  .github
  nix/modules
  src/swarm
  crates/domains/symthaea-spore/src/bin
)

matches="$(git grep -nE "$legacy_regex" -- "${scan_paths[@]}"
  ':!**/archive/**'
  ':!**/_deprecated/**'
  ':!**/target/**' || true)

if [[ -n "$matches" ]]; then
  echo "ERROR: legacy/unverified domain reference found in security-sensitive paths:"
  echo
  echo "$matches"
  echo
  echo "Use a controlled canonical domain, or move a historical reference into archived/provenance material."
  exit 1
fi

echo "Domain trust check: clean."
