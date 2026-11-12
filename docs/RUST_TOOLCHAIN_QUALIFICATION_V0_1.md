# Symthaea Rust toolchain qualification v0.1

**Status:** research/qualification branch. The existing repository pin and main CI remain on Rust 1.96.0. Rust 1.99.1 is an isolated candidate only; no default toolchain promotion is proposed by this change.

## Scope

The qualification compares two exact toolchains over the same commit and unchanged Cargo.lock:

- Baseline: Rust/Cargo 1.96.0 (the repository's existing pin).
- Candidate: Rust/Cargo 1.99.1 (a stable patch release verified in the Symtropy project qualification run).

The workflow deliberately targets only:
- `symthaea-core` library targets and unit tests;
- `symthaea-fep` library targets and unit tests.

The packages are core dependencies of many other Symthaea domains. This focused tranche gives useful compiler coverage without pretending to validate all 200+ workspace packages or every feature/target combination.

It also runs full workspace `cargo metadata --no-deps` to catch manifest/workspace resolution errors, while avoiding dependency resolution changes.

## Non-goals and safety

- No change to root `rust-toolchain.toml`.
- No change to Cargo.toml, Cargo.lock, edition or package MSRV.
- No change to default CI jobs or their large feature matrix.
- No Nixpkgs/flake update.
- No scientific code, experiment seed or runtime configuration changes.
- No CUDA, WASM, Holochain or hardware-specific target promotion in this tranche.

A passing focused lane is evidence for the named packages and commands only. It is not a full-workspace compatibility statement.

## Qualification protocol

For both toolchains, with the same exact source and lockfile:
1. Record `rustc -Vv` and `cargo -Vv`.
2. Run `cargo metadata --locked --no-deps`.
3. Run rustfmt for the two selected packages.
4. Run locked library compilation.
5. Run locked library tests.
6. Run Clippy without converting the existing warning backlog into an unrelated hard-fail gate.
7. Build API documentation.
8. Require Cargo.lock to remain unchanged.

The jobs run separately to avoid shared target-directory or cache contamination.

## Performance and SOTA programme

Do not conclude that a compiler is faster from successful compilation or one timing. The next opt-in phase should measure both compiler costs and real Symthaea application behavior.

### Compiler/build measurements

- Clean-target `cargo check`, debug build and release build.
- Incremental edit/rebuild time with a fixed, scripted source edit.
- Peak RSS and CPU time.
- Critical path by package from Cargo `--timings`.
- Target-directory size and executable/library size.
- Repeat-build output reproducibility.
- sccache hit rate measured separately from no-cache compiler cost.

Use the Rust project's `rustc-perf` suite as the compiler ecosystem baseline. Use Cargo timing reports to distinguish compiler cost from dependency graph, build scripts and linker work.

### Application-specific measurements

For HDC:
- 16,384-dimensional vector create/bind/bundle/unbind/similarity operations.
- Throughput and latency distributions for representative batch sizes.
- Allocation count/bytes and peak resident memory.
- SIMD/scalar equivalence and numerical error.
- Determinism of fixed-seed workloads.

For temporal dynamics and active inference:
- CfC/LTC step throughput at representative hidden dimensions and batch sizes.
- FEP update / Markov-blanket workload duration.
- Warm/cold initialization costs.
- Numerical stability and convergence under identical seeds.
- Runtime p50/p95 plus throughput; p99 only with sufficient samples.

Use existing Criterion benchmarks such as `hdc_benchmarks`, `hdc_simd_compare`, `hdc_ltc_learning` and the FEP `markov_blanket_bench`. Preserve the current benchmark source until both compiler versions have a matched baseline.

### Statistical requirements

- At least 5 paired runs for exploratory comparisons.
- 30+ paired independent runs for ordinary latency-distribution claims.
- For p99, collect at least 10,000 observations per condition or use a justified alternative statistical design.
- Rotate compiler order to reduce thermal/order bias.
- Record CPU governor, kernel, memory, filesystem/mount, compiler hash, lockfile hash and environment.
- Preserve raw Criterion output and per-run metadata.
- Never call missing/skipped trials passes; report failed runs separately.

### Comparison standard

We should not claim to beat `rustc-perf` on unrelated crates. The fair competitive position is:
1. compiler-wide comparison against the upstream rustc-perf suite;
2. build-graph diagnosis with Cargo timings;
3. application-specific HDC/FEP performance on identical source and input;
4. correctness, determinism and resource efficiency as co-equal dimensions.

A newer compiler is promoted only when compatibility and correctness gates pass. Performance gains are optional evidence, not presumed.

## Promotion rule

Keep Rust 1.96.0 as the repository default until:
- both toolchains have exact-head hosted results;
- any candidate-only failure is understood and fixed or documented;
- targeted tests and docs pass;
- any numerical output difference has been explained;
- a separate PR proposes the actual default pin change.

## References

- Rust compiler performance testing: https://rustc-dev-guide.rust-lang.org/tests/perf.html
- Cargo build timings: https://doc.rust-lang.org/cargo/reference/timings.html
- Criterion.rs: https://bheisler.github.io/criterion.rs/book/
