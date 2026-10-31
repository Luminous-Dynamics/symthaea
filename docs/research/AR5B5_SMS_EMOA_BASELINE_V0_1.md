# AR-5B5 deterministic SMS-EMOA baseline v0.1

**Status:** compact deterministic Rust reference variant committed; tests authored but not compiled or executed. No convergence, runtime or SOTA result is claimed.

## Purpose

SMS-EMOA adds a third competitive design point to the current baseline programme:
- NSGA-II: nondomination rank plus crowding-based diversity;
- MOEA/D-DE: decomposition into scalar subproblems;
- SMS-EMOA: hypervolume-contribution-based environmental selection.

This implementation uses the same ZDT1 and three-objective DTLZ2 fixtures, objective normalizations and Pareto indicator path as the other methods.

File:
\`crates/domains/symthaea-architecture-engineering/src/pareto_smsemoa.rs\`

## Implemented variant

- Random continuous initialization in [0,1].
- Binary tournament selection by nondomination rank then crowding distance.
- Simulated binary crossover (SBX).
- Polynomial mutation.
- One offspring is selected from the two SBX children and evaluated per iteration.
- The parent population plus offspring is sorted into Pareto fronts.
- The worst front is identified.
- If the worst front has one member, it is removed.
- Otherwise, compute exact exclusive hypervolume contribution of every member of that last front against the frozen reference point in transformed maximization-utility space. Remove the lowest-contribution point.
- Ties are resolved deterministically by removing the most recently created candidate (largest ordinal).

This is a compact SMS-EMOA-style implementation, not proof of equivalence to every reference SMS-EMOA codebase or parameterization. In particular, tournament selection uses rank/crowding, and the exact-HV archive selector is bounded to two through four objectives.

## Frozen run configuration

Every run records a SHA-256 digest of:
- population size;
- offspring evaluation budget;
- objective specification IDs/directions/scale anchors;
- hypervolume reference point;
- crossover and mutation probabilities;
- SBX and polynomial mutation distribution indices;
- RNG seed.

The exact objective evaluation count is:

\[
N_{\mathrm{eval}}=N_{\mathrm{pop}}+N_{\mathrm{offspring}}.
\]

The population is capped at 512 because each replacement computes total and leave-one-out hypervolume for points in the last front. This is a practical bound for the current exact-HV path, not a theoretical limitation of SMS-EMOA.

## Hypervolume semantics

The environment selector uses **exclusive contribution** within the worst front:

\[
\Delta HV_i = HV(F_{last}) - HV(F_{last}\setminus\{i\}).
\]

The reference point is frozen in the config and uses transformed maximization-utility coordinates. Objective values are first transformed using the same \`ParetoObjectiveSpec\` anchors as the final analysis. Values outside the anchors are not clipped.

Hypervolume contribution is sensitive to the reference point. The config digest binds the point; comparisons must use the same reference point and normalization anchors for all algorithms. Reference sensitivity should be reported separately with the shared Pareto evaluator. Do not select a point after examining which algorithm it favors.

## Replay and shared metrics

- \`verify_smsemoa_report\` reruns the exact problem/config/seed and compares the full report.
- \`analyze_smsemoa_report\` accepts only a replay-verified run, deduplicates repeated decision vectors only at the analysis boundary, and constructs the shared \`ParetoAnalysisPlan\`.
- The final population, nondominated archive, reference point, exact objective-evaluation count and config digest are included in the report.
- This optimizer does not report measured CPU, energy, GPU time or real memory. The external benchmark harness must measure those independently.

## Tests authored

- deterministic ZDT1 execution with fixed objective evaluation budget;
- three-objective DTLZ2 bounds, population and archive shape;
- replay verifier and shared Pareto indicator conversion;
- removal of a least-contributing point in a fixed four-candidate fixture;
- invalid population, objective direction and evaluation budget rejection.

These tests have not been run. Acceptance commands:

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_smsemoa -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_nsga2 -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_moead -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

## Fair comparison protocol

For every algorithm and problem instance:
- match objective evaluation budgets exactly;
- freeze objective anchors, feasibility rules, HV reference point and IGD+/epsilon reference front;
- retain every evaluation and final nondominated archive;
- compute HV, IGD+ and additive epsilon with the same code;
- report time/evaluations to fixed target indicator values, not only final values;
- report runtime and measured hardware costs separately;
- use the same seed set and independent problem instances;
- report all runs, failures and per-instance outcomes.

## Limitations

- This implementation has not been cross-checked against a pinned external SMS-EMOA implementation or PlatEMO 4.16.
- It supports only the current ZDT1 and three-objective DTLZ2 fixtures.
- The compact algorithm uses a particular rank/crowding tournament and one-offspring loop. External parity must compare every operator and parameter explicitly.
- Exact hypervolume and repeated leave-one-out contribution calculations are costly; the population cap is 512.
- No constraints, noisy/expensive evaluation strategy, parallel execution, or uncertainty-aware selection are implemented.
- No machine-level resource telemetry is recorded in the optimizer report.

No benchmark win or SOTA claim is made.

The test fixture includes four mutually nondominated points with known 2D exclusive contributions. Two points tie at the smallest contribution; the deterministic tie rule removes the most recently created one. This tests the selection decision rather than merely checking that population size decreases.
