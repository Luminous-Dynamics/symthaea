# LQCD-018G — independent block-size stability oracle

This evidence subject is standard-library Python and imports no Symthaea/Rust implementation code.

## Exact executed subject

SHA-256:

`6ce995feca98b5180de95c45e883a353470278c3fc872dc1dde4cfdfbd71900d`

Subject:

`scripts/lqcd-flow-scale-block-stability-oracle.py`

Frozen stdout:

`docs/release/evidence/LQCD_018G_BLOCK_STABILITY_RESULT.txt`

## Qualification theorem

The oracle recomputes the full nonlinear `t0`-like blocked-jackknife uncertainty on the frozen 12-configuration correlated trajectory for block sizes 1, 2, 3, 4 and 6. It does not infer a preferred block size.

Executed standard errors:

- block 1: `0.0042128883956444058`
- block 2: `0.0061045555275102302`
- block 3: `0.0080544204816276263`
- block 4: `0.0090863018951899219`
- block 6: `0.012732198384543647`

The central estimate remains `0.33750000000000002` because every candidate uses the same full retained ensemble; only the delete-one-block replicate geometry changes.

For the declared candidate plateau `[3, 4, 6]`, symmetric adjacent relative SE changes are:

- 3 → 4: `0.12040115823331843`
- 4 → 6: `0.33420230012236629`

Therefore the same numerical evidence gives:

- declared maximum relative SE change `0.30` → **FAIL**;
- declared maximum relative SE change `0.35` → **PASS**.

This is intentional: the oracle establishes arithmetic and semantics, not a universal plateau tolerance.

## Independent-chain boundary

The subject also proves a chain-local blocking rule. For two independent chains with six retained configurations each, flattened `N=12` is divisible by block size 4, but each chain length is not. A chain-aware resampler must reject block size 4 rather than create a block that crosses the chain boundary.

Block size 3 is valid and yields exactly:

`[(0,3), (3,6), (6,9), (9,12)]`

with every block contained inside one chain.

## Authority boundary

This is synthetic resampling-semantics evidence only. It does not establish a physically adequate block size, equilibrium, stationarity, ergodicity, topology tunnelling, physical `t0/w0`, lattice spacing, finite-volume adequacy or continuum control. A production stability claim must additionally bind qualified autocorrelation evidence and an explicit caller-declared policy.
