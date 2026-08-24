# HDC-LTC SIMD Batch Retrieval Handoff — 2026-08-24

## Status

Tranche 2 builds on `perf/hdc-ltc-simd-v1` and remains **opt-in**. It adds reusable retrieval/update primitives for the newer local `symthaea-broca-liquid` work without changing the existing scalar `ContinuousHV` methods or the frozen LH-001S reference path.

The public GitHub export does not yet contain the August 24 local `symthaea-broca-liquid` crate, so this tranche intentionally stops at the shared `symthaea-hdc-ltc` seam. Wire it into Broca only after local compilation, benchmarking, and semantic-equivalence validation.

## Provenance

- repository: `Luminous-Dynamics/symthaea`
- tranche-1 base/head: `aafd522282c94b15baa43937e492551f667868ee`
- branch: `perf/hdc-ltc-simd-v2`
- local research tree: `/srv/luminous-dynamics/symthaea` (newer than GitHub export)

## Added primitives

### `PreparedContinuousHvSet`

Immutable candidate hypervectors are copied once into contiguous row-major `f32` storage. Candidate inverse norms are cached once.

For each query:

1. compute the query inverse norm once;
2. score every prepared candidate with runtime-dispatched AVX2 dot products (portable scalar fallback otherwise);
3. multiply by the cached candidate inverse norm, avoiding a per-candidate square root;
4. write scores into a caller-owned buffer with no per-query heap allocation.

This removes repeated candidate-vector construction, repeated candidate norm reductions, and per-candidate square roots from the retrieval hot path.

### `ContinuousHvFusedSimdExt::bind_add_scaled_simd`

Computes:

```text
state += (a * b) * scale
```

in one traversal, avoiding a temporary bound hypervector and a second scale/add traversal. The AVX2 state-changing path deliberately does not use FMA.

### Engineering benchmark

`examples/simd_batch_bench.rs` compares repeated scalar cosine retrieval against prepared batch retrieval over:

- D = 256, 512, 1024, 2048, 4096, 8192, 16384;
- candidate counts = 32, 128, 512;
- 64 repeated queries per cell.

The benchmark reports winner agreement and does not make scientific claims. Its SIMD-only imports are feature-gated so default-feature builds remain valid.

## Scientific boundary

Do not use this tranche to rebuild, replace, or reinterpret the already frozen LH-001S confirmatory artifact. The scalar result remains the scientific reference.

Before accelerated Broca retrieval becomes an experimental execution mode, require a separate equivalence gate with at least:

- identical query winner for every evaluated query;
- identical ranking where score margins exceed the registered tolerance;
- bounded score error/ULP drift;
- identical exact-retrieval counts;
- identical `N95` capacity classifications;
- identical semantic-alignment verdict;
- identical state digests for pointwise state-changing kernels where operation order is preserved.

## Apply

If tranche 1 is already applied locally:

```bash
cd /srv/luminous-dynamics/symthaea

git status --short
git fetch origin perf/hdc-ltc-simd-v2

git cherry-pick \
  aafd522282c94b15baa43937e492551f667868ee..origin/perf/hdc-ltc-simd-v2
```

If neither tranche is applied, apply the whole SIMD series from the public export base:

```bash
cd /srv/luminous-dynamics/symthaea

git status --short
git fetch origin perf/hdc-ltc-simd-v2

git cherry-pick \
  8de0ca10e69c2da42844fd7a202e639bf21e32bc..origin/perf/hdc-ltc-simd-v2
```

Abort rather than broadening conflicts outside the scoped HDC-LTC files/handoffs. Preserve unrelated local research work.

## Required local validation

```bash
cargo test -p symthaea-hdc-ltc
cargo test -p symthaea-hdc-ltc --features simd
cargo clippy -p symthaea-hdc-ltc --all-targets --features simd -- -D warnings

cargo run -p symthaea-hdc-ltc \
  --example simd_continuous_bench \
  --release --features simd

cargo run -p symthaea-hdc-ltc \
  --example simd_batch_bench \
  --release --features simd
```

Run the benchmarks on a quiet machine before recording speedup numbers. The July SIMD audit already showed that handwritten intrinsics do not automatically beat LLVM on every simple operation.

## Local Broca integration seam

The August 24 LH-001S capacity loop currently reconstructs candidate-label collections inside every fact query. Do not change the sealed confirmatory code. For a new engineering/accelerated mode, prepare one immutable numeric candidate table per `(group/codebook candidate set)` and reuse a score buffer.

Conceptually:

```rust
#[cfg(feature = "simd")]
use symthaea_hdc_ltc::PreparedContinuousHvSet;

struct PreparedCandidates {
    labels: Vec<String>,
    vectors: PreparedContinuousHvSet,
    scores: Vec<f32>,
}
```

Build it once after the codebook is frozen, preserving exactly the same candidate ordering as the reference path. A query then becomes:

```rust
prepared.vectors.similarities_into(query_hv, &mut prepared.scores);
```

Map the winning numeric index back through `labels`; do not regenerate or reorder labels during the query.

For routed updates that currently materialize `bind(...).scale(gain)` and then accumulate it, use `bind_add_scaled_simd` only in the new accelerated execution mode after pointwise exactness is confirmed locally.

## Recommended Broca execution modes

```rust
enum HdcExecutionMode {
    Reference,
    Accelerated,
}
```

`Reference` must continue to call the frozen scalar path. `Accelerated` may use the prepared candidate and fused update primitives only after the equivalence campaign passes.

## Next tranche

After local v2 validation:

1. add the explicit `HdcExecutionMode` to `symthaea-broca-liquid`;
2. add a deterministic scalar-vs-accelerated equivalence executable over representative LH-001S cells;
3. record winner/ranking/error/state-digest hashes without exposing new scientific outcomes;
4. profile allocation counts and cache misses;
5. only then use accelerated mode for new, unfrozen large sweeps;
6. explore packed binary/ternary shadow indexing as a later representation/index experiment, not as a silent optimization of LH-001S.
