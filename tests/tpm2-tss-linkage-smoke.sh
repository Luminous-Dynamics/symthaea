#!/usr/bin/env bash
set -euo pipefail

# Build-environment readiness check for the future typed Rust tss-esapi adapter.
# This does not invoke the TPM and does not establish any trust claim.

for pkg in tss2-esys tss2-tctildr tss2-mu; do
  version="$(pkg-config --modversion "$pkg")"
  test -n "$version"
  echo "$pkg=$version"
  pkg-config --exists "$pkg"
  pkg-config --libs "$pkg"
done

echo 'TPM2 TSS linkage smoke test: PASS'
echo 'authority_claim=none (build-environment readiness only)'
