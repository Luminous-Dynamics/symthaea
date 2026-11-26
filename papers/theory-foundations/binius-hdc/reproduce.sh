#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Reproduce the standalone Binius HDC benchmark shipped in this public repo.
#
# Scope:
# - Runs the real Binius prover and verifier in crates/hdc-zkp-bench.
# - The benchmark also prints an analytical prime-field comparator.
# - It does NOT run a Winterfell prover, the historical health-attestation
#   pipeline, or sibling-repository crates. Those are not part of this checkout.
#
# Usage: bash papers/theory-foundations/binius-hdc/reproduce.sh
#
# Timing and proof-size observations are machine/toolchain dependent. Record
# the commit, rustc version, target, and hardware when publishing results.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
BENCH_MANIFEST="${REPO_ROOT}/crates/hdc-zkp-bench/Cargo.toml"
BENCH_LOCK="${REPO_ROOT}/crates/hdc-zkp-bench/Cargo.lock"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required (install the Rust toolchain first)" >&2
  exit 127
fi

if [[ ! -f "${BENCH_MANIFEST}" ]]; then
  echo "error: standalone benchmark manifest not found: ${BENCH_MANIFEST}" >&2
  exit 2
fi

if [[ ! -f "${BENCH_LOCK}" ]]; then
  echo "error: benchmark Cargo.lock not found: ${BENCH_LOCK}" >&2
  echo "Refusing to resolve a fresh dependency graph; restore the committed lockfile." >&2
  exit 2
fi

echo "=== Symthaea public Binius HDC benchmark ==="
echo "Repository: ${REPO_ROOT}"
if command -v git >/dev/null 2>&1 && git -C "${REPO_ROOT}" rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Commit:     $(git -C "${REPO_ROOT}" rev-parse HEAD)"
fi
rustc --version
cargo --version
echo
echo "Scope: standalone Binius benchmark plus analytical comparator."
echo "No Winterfell prover or cross-repository health pipeline is executed."
echo

cd "${REPO_ROOT}"
cargo run --locked --release --manifest-path "${BENCH_MANIFEST}"
