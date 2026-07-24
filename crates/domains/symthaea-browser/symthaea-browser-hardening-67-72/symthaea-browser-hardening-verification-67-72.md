# Symthaea Browser Hardening Verification — Patch Sets 67–72

## Scope

Campaign XII extends the Patch Set 66 tree with signed time consensus,
transitive authority expiry, staged configuration rollout, split-brain-resistant
control-plane leases, privacy-preserving forensic export, and release-evidence
schema v7 closure.

- Parent commit: `d218bf8837d2a8b3c2606c525e8d47dad102da4d`
- Final commit: `350263b4789c2a548cbad957c931befadf947260`
- Final Git tree: `374012d2f5313689648d7b69959cef07aab8b678`

## Patch sequence

1. `b2ca95be8c79d76b887963cbedff01f3596af947` — signed time authority
2. `d7df4a91f8ac8df931c82d0cc0914fbf9c58df94` — transitive authority liveness
3. `3368296b572755854bfdd999daf44bacbf301b9c` — staged configuration rollout
4. `a8a739ad8c7c81e8f563a89a4c93a26ac15f93bc` — split-brain control plane
5. `e32c6f785d2d7226f85c0c159f6f69117963c758` — privacy-safe forensic export
6. `350263b4789c2a548cbad957c931befadf947260` — profile/release-evidence closure

## Verification performed

### Patch integrity and replay

- Every Patch Set 67–72 mail patch passed
  `git apply --check --whitespace=error-all` in sequence.
- Patch Sets 67–72 replayed with `git am` from the exact Patch Set 66 parent.
- Patch Sets 01–66 replayed from the original uploaded source, followed by
  Patch Sets 67–72.
- Both replay routes reproduced the authored final Git tree exactly:
  `374012d2f5313689648d7b69959cef07aab8b678`.
- Re-extracting `symthaea-browser-hardened-72.tar.gz` and computing its Git tree
  reproduced the same tree exactly.

### Static structure

- `scripts/verify-static-structure.py` passed across 63 public Rust modules.
- Tree-sitter parsed all 67 Rust source, test, and example files without syntax
  errors.
- `Cargo.toml` parsed successfully with Python `tomllib`.
- `scripts/verify-browser-release.sh` passed `bash -n` shell syntax checking.
- `git diff --check` and per-commit `git show --check` passed.
- All generated gzip/tar archives passed integrity and listing checks.
- A conservative private-key and credential-assignment scan found no matches.

### Honest unavailable gates

The release helper exited with status `127` and the message
`cargo is required for release verification`. This is expected in the current
environment and is treated as an unavailable gate, not a pass.

The following were therefore **not executed**:

- Cargo compilation
- rustfmt
- Clippy
- Rust unit and integration tests
- live Chromium hostile-browser tests
- real signed time-source integration
- distributed coordinator-key and failover integration
- live authority-expiry propagation through the executor
- live forensic export and recipient verification

The standalone archive also lacks its workspace path dependency at
`../../core/symthaea-core`. These executable and host-integration gates must run
inside the complete Symthaea workspace before release claims are made.

## Change summary

Campaign XII changes 14 files with 3,104 insertions and 10 deletions. The final
snapshot contains approximately 28,273 Rust lines across source, tests, and
examples, with 221 `#[test]` markers.
