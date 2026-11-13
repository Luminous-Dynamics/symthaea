# Symthaea Rust toolchain qualification v0.3

**Status:** stacked on the workspace-integrity prerequisite PR #8674 and the generated lockfile repair PR #11792. The repository's declared default remains Rust/Cargo 1.96.0. Rust 1.99.1 remains an intermediate stable comparison point; Rust 1.100.0 is the current stable candidate. No default toolchain promotion is proposed.

## Why this is stacked

The main branch's `crates/core/*` glob discovers three directories that do not contain Cargo manifests:
- `crates/core/symthaea`;
- `crates/core/symthaea-observation-attestation`;
- `crates/domains/symthaea-rea-evidence-verifier`.

PR #8674 proposes explicit exclusions for those source-only/incomplete paths, with no source deletion, placeholder crate, or weakened gate. PR #11792 repairs a distinct Cargo.lock edge for `sha2 0.10.9` in `symthaea-epistemic-types`, generated under Rust 1.96.0. This qualification stacks on those exact prerequisites rather than bypassing workspace or lockfile failures.

## Three exact toolchains

| Role | rustc / Cargo | Meaning |
|---|---|---|
| Baseline | 1.96.0 | Current root `rust-toolchain.toml` pin |
| Intermediate | 1.99.1 | Previous stable release reference |
| Candidate | 1.100.0 | Current stable release (Nov 12, 2026) |

Every command uses explicit `cargo +<version>` / `rustc +<version>` selectors. The repository toolchain file can otherwise override the version selected by a CI setup action. The intermediate and candidate jobs check out the exact same PR head commit as the baseline.

## Focused packages and qualification

The focused packages are:
- `symthaea-core`;
- `symthaea-fep`.

Each toolchain independently runs locked workspace metadata, package formatting, library compilation, library unit tests, Clippy and rustdoc, then checks Cargo.lock immutability. A disposable Git archive is used for an unlocked lockfile-drift diagnosis if a compile step fails; the checked-out lockfile is never modified by that diagnostic.

The workflow intentionally avoids `--all-targets` and does not claim that all repository packages, target triples, feature combinations, CUDA, WASM, robotics or hardware integrations are qualified. The two selected core libraries are a first, bounded qualification cell.

## Rust 1.100 changes to watch

1. **Never type stabilization and fallback change.** Never-to-any fallback changes from `()` to `!` across all editions, not just 2024. Older-edition generic code that accidentally depended on unit fallback can now fail or change inference. Core-package compilation under the exact candidate is necessary but does not prove all external/workspace crates are migrated.
2. **Cargo unused-dependency lint.** Cargo's own `cargo::unused_dependencies` lint reports dependencies not used by any relevant build target. It does not report unused dev-dependencies and may require all build targets before it can decide.
3. **Cargo publication-age policy.** Cargo now has stable dependency cooldown resolution; we should test it against a deterministic local sparse registry, not just assume a configuration parses.
4. **Cargo intermediate build layout.** Downstream tools should use Cargo JSON `compiler-artifact` messages, not private `target/debug/deps`/build paths.
5. **Sample-based PGO.** `-Cprofile-sample-use` is stabilized. This opens a measured optimization path for real HDC/LTC/FEP workloads: collect representative CPU samples, build with the profile, then compare output correctness and runtime. It is not automatically a performance win.

## Non-goals and safety

- No change to default `rust-toolchain.toml`.
- No change to Cargo.toml, Cargo.lock, edition, package MSRV or dependency versions in this qualification branch.
- No change to default CI jobs or the large workspace feature matrix.
- No Nixpkgs/flake update.
- No scientific model code, experiment seed, runtime parameter or generated artifact changes.
- No live host changes.

## Performance and SOTA programme

The first candidate gate establishes compatibility, not speed.

### Compiler/build
- Clean-target `cargo check`, debug and release builds.
- Incremental edit/rebuild with a fixed source edit.
- Peak RSS, CPU time, target-directory size and binary size.
- Cargo `--timings` critical path by package.
- Repeat-build output reproducibility.
- sccache hit/miss rate measured in a separate profile.
- PGO with and without sample profiles, only after a stable baseline workload has been frozen.

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

Keep Rust 1.96.0 as the default until all three exact toolchains pass the named core-package gates, candidate-only failures are understood, numerical differences are explained, and separate benchmark results support any claimed improvement. A later default pin change requires its own PR.

## References

- Rust 1.100.0 announcement: https://blog.rust-lang.org/2026/11/12/Rust-1.100.0/
- Rust compiler performance testing: https://rustc-dev-guide.rust-lang.org/tests/perf.html
- Rust PGO / sample PGO: https://doc.rust-lang.org/rustc/profile-guided-optimization.html
- Never-type fallback migration note: https://doc.rust-lang.org/beta/edition-guide/rust-2024/never-type-fallback.html
- Cargo build timings: https://doc.rust-lang.org/cargo/reference/timings.html
- Criterion.rs: https://bheisler.github.io/criterion.rs/book/
