# symthaea-browser Campaign VIII verification

## Delivered series

Patch Sets 43–48 were authored from the Patch Set 42 source snapshot and cover:

1. TLS identity pinning against the active DNS pin and concrete peer address.
2. Browser/renderer crash epochs with bounded, single-use restart permits.
3. Resource-exhaustion throttling and quarantine.
4. Deterministic asynchronous CDP event ordering and causal chaining.
5. Cross-architecture reproducibility attestations requiring duplicate builds.
6. Security-profile and release-evidence schema v3 closure.

Patch Set 48 also corrects a malformed pre-existing test function declaration in
`src/release_evidence.rs` (`fn claimed execution...`), which would prevent
Rust parsing once a toolchain is available.

## Replay verification

- Authored final commit: `0448c055947cc11b129f438554276604294f0552`
- Authored final Git tree: `b2d8ec8315c80a5c981e8cc7ebd9dddeb126bc35`
- Patch Sets 43–48 replayed from `symthaea-browser-hardened-42.tar.gz`.
- Patch Sets 01–48 replayed from the original `symthaea-browser.tar.gz`.
- Both replay routes produced exactly `b2d8ec8315c80a5c981e8cc7ebd9dddeb126bc35`.
- Every Patch Set 43–48 patch passed
  `git apply --check --whitespace=error-all` before `git am`.
- The hardened Patch Set 48 source archive was re-extracted and reproduced the
  same Git tree.

## Static verification performed

- Cargo manifest parsed successfully with Python `tomllib`.
- All 43 public Rust module declarations resolve to source files.
- Rust-aware lexical delimiter scan passed across source, tests, and examples.
- Function-declaration scan passed after correcting the malformed legacy test.
- All shell scripts passed `bash -n`.
- The release helper exited with status `127` because Cargo is unavailable,
  correctly refusing to report incomplete executable checks as passed.
- All generated tar archives passed listing/integrity checks.

## Change size

- Campaign VIII diff:  12 files changed, 2831 insertions(+), 3 deletions(-)
- Insertions: 2831
- Deletions: 3
- Rust lines in the final standalone crate: 17302
- Rust test markers: 158
- Public modules: 43

## Gates not executed

Cargo, rustfmt, Clippy, Rust unit/integration tests, live Chromium tests, real TLS
socket integration, crash-injection tests, resource-pressure browser tests, and
actual duplicate builds on two architectures were **not run**. The environment
contains no Rust toolchain, and the standalone crate still references the absent
`../../core/symthaea-core` workspace dependency.

The cross-architecture release gate remains explicitly not run. Unit tests of the
reproducibility manifest cannot satisfy that gate; real repeated builds and their
artifact evidence are required.
