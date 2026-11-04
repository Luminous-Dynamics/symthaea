# SAM Reference Decision Harness v0.1

A small standalone, dependency-free Rust implementation of the interval-aware pairwise comparison core specified in Symthaea #10318.

## Run

    cargo fmt --manifest-path tools/sam-reference-decision-harness/Cargo.toml -- --check
    cargo test --manifest-path tools/sam-reference-decision-harness/Cargo.toml
    cargo clippy --manifest-path tools/sam-reference-decision-harness/Cargo.toml --all-targets -- -D warnings

This is intentionally a nested Cargo workspace, excluded from the large root workspace. It has no third-party dependencies.

## Implemented boundary

The evaluator compares two already-frozen feasible actions for one metric across context-qualified reference rows. It provides:
- interval-aware portfolio dominance and equivalence;
- reference-sensitive and unresolved outcomes;
- pointwise-only diagnostics for point estimates;
- explicit per-measurand applicability and qualification;
- a frozen expected-reference manifest that blocks omitted reference rows;
- scenario, action-set, reference, action, metric and unit identity checks;
- shared execution-context, action-payload, output and source-artifact digest format checks;
- exact action-payload matching across all admitted references;
- incomplete outcomes for missing, censored, timed-out, unsupported and failed cells;
- duplicate reference and receipt rejection;
- common-mode family counts as descriptive metadata only (never vote weights);
- canonical reference ordering so row permutation does not change output.

The caller must establish action-set completeness and hard-constraint feasibility before invoking this pairwise metric evaluator. Incomplete action sets and unresolved/infeasible feasibility are returned as incomplete; the evaluator never ranks a known-infeasible or unknown-feasibility action.

## Deliberate limits

- Digest fields are syntax-checked and matched to the supplied context, but the crate does not authenticate their source, recompute an external execution hash, or provide signature/non-repudiation verification.
- No reference independence proof; family IDs are caller-supplied dependency metadata.
- No scalarization of multiple objectives.
- No global optimum or population-probability claims.
- No physical simulator, human behavior, collision probability or vehicle safety claims.
- Results are conditional on the exact named references, metric, interval semantics and scenario context supplied by the caller.
