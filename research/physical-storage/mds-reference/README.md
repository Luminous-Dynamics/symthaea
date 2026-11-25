# Physical Storage Linear MDS Reference

This standalone crate is a deterministic byte-level research fixture. It is not a physical media model and does not claim to implement the Reed–Solomon standard.

## Model

- 48 physical tiles arranged as a synthetic 4 × 4 × 3 grid.
- 4 control tiles: replicated format header and calibration reference.
- 2 spare tiles.
- 42 active tiles arranged as seven interleaved stripes.
- Each stripe has four 8-byte data shards and two 8-byte parity shards.
- A systematic linear [6,4,3] MDS array code over GF(256) recovers any two known shard erasures in each stripe.
- SHA-256 verifies the final payload against an out-of-band expected digest captured when the payload is written; unknown substitutions are detected but not corrected. The digest is an integrity reference, not a signature or source-authentication proof.

The six generator rows are the four identity rows, [1,1,1,1] and [1,2,3,4]. Every choice of four rows is nonsingular under the implemented GF(256) arithmetic. The code is therefore described as a linear MDS code, not labelled Reed–Solomon without an explicit GRS equivalence proof. A standards-conformant RS baseline remains part of the future comparator tournament.

## Run

From the repository root:

```bash
cargo fmt --manifest-path research/physical-storage/mds-reference/Cargo.toml -- --check
cargo clippy --manifest-path research/physical-storage/mds-reference/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path research/physical-storage/mds-reference/Cargo.toml --all-targets
cargo run --manifest-path research/physical-storage/mds-reference/Cargo.toml --bin run_cases
```

## Initial spatial-interleaver comparison

A deterministic exhaustive geometry tournament compared the spatial-stride layout with a naive row-major assignment of each stripe's six shards to six consecutive available tiles. Under the same two-erasure-per-stripe MDS budget:

| Contiguous fault region | Spatial-stride failures | Row-major failures |
|---|---:|---:|
| 2 × 2 × 1 boxes | 0 / 27 | 13 / 27 |
| 3 × 3 × 1 boxes | 0 / 12 | 12 / 12 |
| 2 × 2 × 2 boxes | 0 / 18 | 14 / 18 |
| 3 × 3 × 2 boxes | 8 / 8 | 8 / 8 |
| Any one complete layer | 0 / 3 | 3 / 3 |

The spatial-stride map is materially better than row-major for smaller tested regions, but it does not improve the 3 × 3 × 2 class: both layouts exceed the two-erasure budget in all eight placements. This is deterministic combinatorial evidence about the declared toy geometry—not a physical defect probability or storage-device result. Matching Rust unit tests are authored; their Cargo execution remains pending.

## Deliberately limited claims

The 3-layer demonstration is a severe synthetic stress case: every complete layer contains two shards from every stripe, so any one of the three layers can be erased and reconstructed. The complete toy layout carries 224 payload bytes in 48 × 8 = 384 physical tile bytes, or 58.33% net payload efficiency after parity, controls and spare tiles. The low rate is intentional for a one-third-volume erasure proof; it is not an optimized capacity result.

A 301-layer storage medium should be evaluated with a different interleaver and higher-rate code. For example, a hypothetical 10-data + 1-parity stripe distributed across 11 distinct layers has 90.91% coding rate before calibration, metadata and spare-region overhead and can tolerate one known erasure per stripe. That is a design calculation, not a demonstrated glass result. Multiple-layer loss requires additional parity and reduces the maximum code rate.

Synthetic layer/cluster masks are not measured glass defects or failure probabilities. Physical readout, optics, laser–glass interactions, retention and manufacturing yield remain outside this crate.
