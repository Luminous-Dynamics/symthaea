# Parameterized Physical Layout Search — v0.1

**Programme:** PARAMETERIZED-PHYSICAL-LAYOUT-SEARCH-001  
**Tracking:** [Symthaea #13726](https://github.com/Luminous-Dynamics/symthaea/issues/13726)  
**Parent:** [Spatial interleaver optimization #13662](https://github.com/Luminous-Dynamics/symthaea/issues/13662)  
**Physical observation input:** [Symtropy #7534](https://github.com/Luminous-Dynamics/symtropy/issues/7534)

## Purpose

Replace the fixed 4×4×3 placement comparison with a parameterized, zero-dependency Rust layout scorer. This first increment compares row-major, shard-major interleaving, serpentine, Morton/Z-order and seeded-random placements, then exposes a deterministic greedy pair-swap search with a fixed candidate-evaluation budget.

The crate scores abstract tile geometry only. It is not an optical, magnetic, thermal, device, manufacturing or retention model. Fault masks are deterministic coverage tests; no probability is assigned to them.

## Input contract

A layout problem declares:
- grid dimensions (x,y,z);
- reserved physical tiles (control/spare regions);
- stripe count and shards per stripe;
- known-erasure budget per stripe;
- deterministic fault masks or parameterized cuboid families.

Each candidate must assign exactly one distinct, in-range, non-reserved tile to every logical shard. The validator rejects aliasing, reserved-tile use, out-of-range coordinates, code-geometry mismatch and insufficient active capacity.

## Candidate families

1. **Row-major contiguous:** consecutive physical cells hold each stripe.
2. **Row-major interleaved:** shard-major assignment distributes each stripe through the row-major physical order; reproduces the existing spatial-stride reference.
3. **Serpentine:** alternating scan direction across rows/layers, with contiguous and interleaved assignment modes.
4. **Morton/Z-order:** sort physical cells by 3D Morton key, then use contiguous/interleaved assignment.
5. **Seeded random:** Fisher–Yates permutation using a disclosed deterministic xorshift64* seed, then use both assignment modes.
6. **Greedy pair-swap:** bounded best-improvement search over swaps in the logical-to-physical shard map.

The random generator is for reproducible comparator permutations, not cryptographic randomness.

## Scoring contract

For each declared fault mask, count erased shards per stripe. A mask is unrecoverable when any stripe exceeds the declared known-erasure budget. Report:
- total fault sets;
- unrecoverable fault-set count;
- worst erased shards in one stripe;
- total erased shard observations;
- total excess erasures beyond the per-stripe budget;
- candidate evaluation count and deterministic layout fingerprint.

Search ranking is lexicographic: fewer unrecoverable fault sets, then fewer total excess erasures, then lower worst per-stripe loss, then fewer total shard erasures. This preserves exact-recovery priority rather than hiding it in a weighted scalar.

The fingerprint is FNV-1a-64 for stable comparison only. It is not cryptographic integrity or an authenticated receipt.

## Exact small-grid reference

The starter 4×4×3 problem reserves tiles [0,15,16,31,32,47], has seven MDS(6,4) stripes and a two-erasure budget. The test corpus covers:

| Shape | Placements |
|---|---:|
| 2×2×1 | 27 |
| 3×3×1 | 12 |
| 2×2×2 | 18 |
| 3×3×2 | 8 |
| 4×4×1 | 3 |
| **Total** | **68** |

The two principal reference mappings must reproduce the existing aggregate counts over all 68 masks:
- row-major contiguous: 50 unrecoverable masks;
- row-major shard-major interleaved: 8 unrecoverable masks.

These combined totals are compact regression assertions. Per-shape counts remain the primary report in the parent research fixture.

## Equal-budget benchmark protocol

- Baselines use one evaluation each (their fixed layout is scored once).
- Search methods declare a budget including the initial mapping score.
- All methods use the same physical grid, reserved cells, code parameters, fault masks and scoring function.
- Publish every seed and every fault-mask identity.
- Separate training, validation and held-out masks. When measured Symtropy maps become available, split by specimen/material batch/reader, not by neighbouring cells.
- Exhaustive enumeration is the small-grid oracle; heuristic results should be measured by optimality gap on those small cases before scaling.
- A scalar fitness cannot trade an exact-recovery failure for a modest density gain. Density/code-rate is a separate Pareto axis and must include parity, control, calibration and spare overhead.

## Prior art and novelty boundary

Blaum, Bruck and Vardy (1998) explicitly construct 2D and 3D interleavers for cluster errors and analyze graph-colouring and lattice constructions: https://d3ccyth396mz21.cloudfront.net/publications/interleaving-schemes-for-multidimensional-cluster-errors

Chou and Neifeld (1998) study ECC plus 2D interleaving in volume holographic memory under correlated optical errors: https://opg.optica.org/ao/abstract.cfm?uri=ao-37-29-6951

Therefore this project does not claim novelty for 3D interleaving, lattice interleavers or ECC/interleaver composition. Its hypothesis is to bind a reproducible optimizer to measured 3D channel maps, manufacturing constraints, calibration metadata, exact decoders and full service contracts.

## Run commands

```bash
cargo fmt --manifest-path research/physical-storage/layout-search/Cargo.toml -- --check
cargo clippy --manifest-path research/physical-storage/layout-search/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path research/physical-storage/layout-search/Cargo.toml --all-targets
cargo run --manifest-path research/physical-storage/layout-search/Cargo.toml
python3 research/physical-storage/layout-search/oracle/verify_layout_search.py
```

No Rust execution is claimed until these commands run against the exact branch head. The Python oracle independently checks only geometry and the two starter reference counts.
