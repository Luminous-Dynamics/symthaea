# Symthaea Rust toolchain qualification v0.2

**Status:** stacked on the workspace-integrity prerequisite PR #8674. The repository's declared default remains Rust/Cargo 1.96.0. Rust/Cargo 1.99.1 is a candidate only; no default toolchain promotion is proposed.

## Why this is stacked

The main branch's `crates/core/*` glob discovers three directories that do not contain Cargo manifests:
- `crates/core/symthaea`;
- `crates/core/symthaea-observation-attestation`;
- `crates/domains/symthaea-rea-evidence-verifier`.

PR #8674 proposes explicit exclusions for those source-only/incomplete paths, with no source deletion, placeholder crate, or weakened gate. Its updated exact head has a prior workspace metadata/buildability pass. This qualification is based on that workspace-integrity branch so it does not silently work around invalid workspace discovery.

## Toolchains and scope

- Baseline: Rust/Cargo 1.96.0, matching the root `rust-toolchain.toml`.
- Candidate: Rust/Cargo 1.99.1.
- Every command uses explicit `cargo +<version>` / `rustc +<version>` selectors. The repository toolchain file can otherwise override the version selected by a CI setup action.
- Both jobs run against identical source and unchanged Cargo.lock.

The focused packages are:
- `symthaea-core`;
- `symthaea-fep`.

Each toolchain runs locked workspace metadata, package formatting, library compilation, library unit tests, Clippy and rustdoc. The workflow intentionally avoids `--all-targets` and does not claim that all repository packages, target triples, feature combinations, CUDA, WASM, robotics or hardware integrations are qualified.

## Non-goals and safety

- No change to the default `rust-toolchain.toml`.
- No change to Cargo.toml, Cargo.lock, edition, package MSRV or dependencies.
- No change to default CI jobs or its large feature matrix.
- No Nixpkgs/flake update.
- No scientific model code, experiment seed, runtime parameters or generated artifact changes.
- No live host changes.

## Performance and SOTA programme

The first candidate gate establishes compatibility, not speed. The next opt-in phase should compare:

### Compiler/build
- Clean-target `cargo check`, debug and release builds.
- Incremental edit/rebuild with a fixed source edit.
- Peak RSS, CPU time, target-directory size and binary size.
- Cargo `--timings` critical path by package.
- Repeat-build output reproducibility.
- sccache hit/miss rate measured in a separate profile.

Use upstream `rustc-perf` for the general compiler ecosystem baseline; use Cargo timings to separate compiler cost from dependency graph/build-script/linker cost.

### Symthaea workloads
- HDC 16,384-dimensional bind, bundle, unbind and similarity throughput.
- Allocation count/bytes and memory footprint for representative batch sizes.
- SIMD/scalar equivalence and numerical error.
- CfC/LTC step throughput by hidden dimension and batch size.
- FEP update and Markov-blanket benchmark latency.
- Fixed-seed determinism and convergence under identical inputs.

Use existing Criterion targets `hdc_benchmarks`, `hdc_simd_compare`, `hdc_ltc_learning` and `markov_blanket_bench`. Preserve their source and input parameters across compiler versions.

### Statistical standard
- 5+ paired runs for exploratory build comparisons.
- 30+ independent runs for ordinary latency claims.
- 10,000+ observations per condition for p99 claims, or an explicitly justified alternative.
- Rotate compiler order to reduce thermal/order bias.
- Record exact compiler hash, Cargo identity, lockfile SHA-256, runner image, kernel, CPU, memory and filesystem metadata.
- Preserve raw Criterion output and failed/skipped trials.
- No compiler or runtime performance claim until matched results exist.

## Promotion rule

Keep Rust 1.96.0 as the default until both exact toolchains pass the named gates, candidate-only failures are understood, numerical differences are explained, and separate benchmark results support any claimed improvement. A later default pin change requires its own PR.

## References
- Rust compiler performance testing: https://rustc-dev-guide.rust-lang.org/tests/perf.html
- Cargo build timings: https://doc.rust-lang.org/cargo/reference/timings.html
- Criterion.rs: https://bheisler.github.io/criterion.rs/book/
