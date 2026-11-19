# Plasma Memory Contract v0.1

**Status:** Dependency-free Rust contract checker with authored synthetic tests. No closure, kinetic solver or memory mechanism is implemented here. Rust compilation/hosted validation remain pending.

## Purpose

The contract distinguishes four model classes that can otherwise be incorrectly lumped together:
- **Markovian:** only instantaneous resolved fields;
- **TrailingHistory:** explicit samples of resolved fields over a declared physical-time horizon;
- **PrognosticLatent:** a compact evolving state with an explicit update-law identity, initialization and reset policy;
- **FourierHermiteReservoir:** closure of a velocity-space Hermite hierarchy, distinct from a configuration-space history window.

Every contract binds physics/regime, trajectory identity and digest, model artifact, solver build, RK tableau and stage semantics. These declarations make unlike experiments distinguishable; they do not prove that the mechanism has learned a physical quantity.

## Physical-time invariant

A sample count is not a physical memory horizon without its cadence and units.

The online FNO paper reports K=50 samples at Δt=0.1 ωp⁻¹, a 5 ωp⁻¹ horizon. This v0.1 contract encodes that relation exactly:

\`50 × (1/10) = 5\` in units of \`omega_p^-1\`.

If the cadence changes to 0.05 ωp⁻¹, 100 samples preserve the same horizon. Keeping 50 samples at 0.05 changes the horizon to 2.5 and is rejected when the profile expects 5. The check uses exact rational integer arithmetic; no floating-point tolerance or unit conversion is implied.

The benchmark papers define their own time conventions. A manifest must state the canonical time unit and the paper/solver-specific horizon semantics; do not silently convert between physical seconds, normalized plasma time, output cadence and solver substage time.

## Solver-stage semantics

The contract requires one explicit stage interpretation:
- \`accepted_state_only\`;
- \`stage_explicit\`;
- \`source_reported\`.

This field is an identity, not a validation that a given history is correct. Use #12910 for the detailed RK computational-order vs represented-physical-time checks. The contract binds the tableau digest so stage-semantic changes cannot be hidden behind the same model label.

## Synthetic fixture

The JSON fixture contains four illustrative, synthetic contracts. They are not reproductions of ADEPT, the 2026 FNO or Barbour et al.'s reservoir. Placeholder digests, state dimension and retained modes are deliberately marked synthetic; they are not source-pinned model details.

## Claim ceiling

This crate checks structural declarations and exact time arithmetic. It does not:
- authenticate issuers or verify that digests refer to real artifacts;
- establish that a latent variable represents trapped-electron population;
- establish that a reservoir faithfully closes the kinetic Hermite tail;
- measure autocorrelation or causal memory dependence;
- reproduce ADEPT, FNO or Fourier–Hermite papers;
- execute model inference, coupled rollouts or transfer benchmarks;
- prove numerical stability, conservation or physical validity.

The independent physics/solver referee and Symtropy #1685 run receipt remain the evidence and execution owners; Mycelix #5900 remains experiment provenance. This crate introduces no second solver or evidence ledger.

## Run

    cargo fmt --manifest-path tools/plasma-memory-contract-v0.1/Cargo.toml -- --check
    cargo test --manifest-path tools/plasma-memory-contract-v0.1/Cargo.toml --locked
    cargo clippy --manifest-path tools/plasma-memory-contract-v0.1/Cargo.toml --locked --all-targets -- -D warnings
