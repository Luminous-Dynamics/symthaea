# Roadmap

## Alpha.4 focus

Alpha.4 adds the first integration-grade surface: typed parameters, cognitive scan summaries, batch corpus generation, pairwise similarity matrices, and manifest helpers. The next milestone should be compile verification inside the actual Symthaea workspace, then a small Old Waterworks/Field Deck adapter.
: Symthaea Visual Compression Probe

## v0.1.0-alpha.2 — current

Purpose: establish a zero-dependency visual-memory packet that can reconstruct approximately, fingerprint structurally, and query without decoding.

Included:

- P2/P5 grayscale PGM support
- DCT-style sparse coefficient packets
- deterministic block HDC signatures
- thresholded topology summaries
- text `.svmp` packet format
- CLI: `inspect`, `encode`, `decode`, `compare`, `metrics`, `fingerprint`, `benchmark`, `query`
- JSON output for automation

## v0.1.0-alpha.3

Goal: improve experimental quality without adding heavy dependencies.

Planned:

- add cosine similarity over coefficient energy histograms
- add packet manifest/checksum field
- add corpus-building example
- add before/after repair delta report
- add explicit `SVMP 0.2` packet version with backward reader
- add test fixtures for query ranking

## v0.2

Goal: integrate with Symthaea/Symtropy types.

Planned:

- optional feature for native Symthaea HDC types
- Field Deck scan artifact struct
- Chronicle evidence metadata adapter
- JSONL packet index for corpus search
- false-green visual contradiction report

## v0.3

Goal: cognitive compression, not just structural compression.

Planned:

- FEP residual packet format
- topological anomaly deltas
- persistent-Laplacian-inspired summaries
- multi-resolution block pyramids
- local privacy-preserving evidence sharing

## v1.0 boundary

Do not call this production until it has:

- real corpus benchmarks
- deterministic replay tests
- baseline comparison against DCT/PCA/WebP-style reference outputs
- documented failure cases
- integration tests inside the monorepo
- clear claim boundaries in public docs

## Alpha.3 Follow-Up

- Add binary packet format once text format stabilizes.
- Add real baseline codecs as optional dev tooling, not core dependencies.
- Add corpus labels for retrieval-quality evaluation.
- Add confusion-matrix reports for query-without-decode tasks.
- Add structural change detection for before/after infrastructure scans.
- Integrate HDC signatures with existing `symthaea-hdc-store` or memory APIs once dependency boundaries are stable.
- Keep this crate experimental until it has reproducible corpus-level evidence.


## Alpha.5 Workspace Hardening

- Add `doctor` for correct workspace commands.
- Add `self-test` for fixture-level smoke verification without relying on broad workspace tests.
- Add `pipeline` for one-command corpus artifact generation.
- Keep claim boundary: this is a cognitive visual-memory probe, not a production image codec.

## Alpha.6 verification hardening

- Fix PSNR test semantics for perfect reconstruction (`+inf` is valid).
- Keep runtime codec behavior unchanged.
- Use alpha.8 as the clean baseline before adding real-image fixtures or dependency-gated image format support.


## Alpha.7 packet hash hardening

- Fix packet hash instability after `.svmp` text roundtrip.
- Treat `stable_hash64()` as an artifact/canonical-text hash, not a raw floating-point memory hash.
- Keep hash non-cryptographic and suitable only for regression fixtures, corpus manifests, and experiment comparison.
- Next: add external real-image fixture corpus and compare query-without-decode accuracy against ordinary image baselines.
