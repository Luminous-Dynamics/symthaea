# HDC-LTC SIMD Patch Handoff — 2026-08-24

## Status

Prepared on the public `Luminous-Dynamics/symthaea` export branch for application to the newer local monorepo after the frozen LH-001S confirmation is sealed.

This tranche is intentionally **opt-in**. Existing `ContinuousHV` scalar methods are unchanged. Enabling the Cargo feature alone does not alter them; a caller must explicitly import `ContinuousHvSimdExt` and call the `*_simd` methods.

## Provenance

- repository: `Luminous-Dynamics/symthaea`
- base: `8de0ca10e69c2da42844fd7a202e639bf21e32bc` (`main` at patch creation)
- branch: `perf/hdc-ltc-simd-v1`
- head before this handoff document: `0c0f2a88b3a8f2c16d35c70e5dfa9a9b489a3ca5`

The local `/srv/luminous-dynamics/symthaea` research tree is newer than this GitHub export. Apply by cherry-pick only after confirming the touched core files have not diverged incompatibly.

## Patch scope

1. `crates/core/symthaea-hdc-ltc/src/simd.rs`
   - runtime AVX2 detection on x86_64;
   - unaligned AVX2 loads/stores;
   - portable scalar fallback;
   - explicit extension trait for accelerated bind, dot, cosine, scale, add-scaled, and lerp;
   - fused dot/norm cosine reduction;
   - no FMA in state-changing kernels;
   - pointwise exactness and reduction/ranking tests.
2. `crates/core/symthaea-hdc-ltc/Cargo.toml`
   - adds opt-in `simd` feature; default remains empty.
3. `crates/core/symthaea-hdc-ltc/src/lib.rs`
   - exports SIMD module/trait only under the feature.
4. `crates/core/symthaea-hdc-ltc/examples/simd_continuous_bench.rs`
   - engineering-only scalar-vs-SIMD microbenchmark across D=256..16,384.

## Scientific boundary

Do **not** rebuild or reinterpret the already frozen/sealed LH-001S reference artifact with these kernels. The scalar result remains the scientific baseline.

A future Broca optimization tranche must establish a separate scalar-vs-accelerated equivalence gate before using SIMD for large sweeps or production cognition. Pointwise state-changing operations should be required to match exactly where practical. Reduction operations may use a preregistered numeric tolerance while requiring identical retrieval winners/rankings, identical capacity classifications, and identical scientific verdicts.

## Apply to the newer local tree

```bash
cd /srv/luminous-dynamics/symthaea

git status --short
git rev-parse HEAD

git fetch origin perf/hdc-ltc-simd-v1

git cherry-pick 8de0ca10e69c2da42844fd7a202e639bf21e32bc..origin/perf/hdc-ltc-simd-v1
```

If the local tree has unrelated changes, preserve them. Resolve only conflicts in the four scoped HDC-LTC files plus this handoff document; abort rather than broadening scope unexpectedly.

## Required validation after application

```bash
cargo test -p symthaea-hdc-ltc
cargo test -p symthaea-hdc-ltc --features simd
cargo clippy -p symthaea-hdc-ltc --all-targets --features simd -- -D warnings
cargo run -p symthaea-hdc-ltc --example simd_continuous_bench --release --features simd
```

Because the connected GitHub export cannot execute the user's local Nix/Rust workspace, these commands are required before calling the patch validated in the monorepo.

## Next tranche after validation

Do not immediately replace every call site. Profile first. The highest-value follow-up should likely be:

1. cache immutable candidate vectors/norms;
2. remove per-query candidate-label/vector allocations;
3. add `similarity_many` over contiguous candidate storage;
4. fuse bind+scale+accumulate kernels;
5. add a Broca scalar/SIMD semantic-equivalence campaign;
6. only then switch high-dimensional Broca retrieval/update paths to accelerated operations.

Packed binary/ternary shadow retrieval belongs in a later representation/indexing experiment, not this patch.
