# AR-5B5 deterministic NSGA-II baseline v0.1

**Status:** deterministic Rust reference implementation committed; test cases authored but not compiled or executed. No convergence, runtime or SOTA result is claimed.

## Purpose

A new Pareto evaluator is not a useful research benchmark until it is compared against a well-understood evolutionary baseline under the same functions, seeds, objective-evaluation budget and indicators.

This module implements a compact, deterministic NSGA-II reference optimizer for:
- ZDT1, two-objective continuous minimization;
- DTLZ2, three-objective continuous minimization.

File:
\`crates/domains/symthaea-architecture-engineering/src/pareto_nsga2.rs\`

## Algorithm components

The implementation contains:
- deterministic xorshift64* PRNG;
- random continuous initialization in [0,1];
- binary tournament selection by nondomination rank then crowding distance;
- simulated binary crossover (SBX);
- polynomial mutation;
- fast nondominated sorting;
- crowding-distance calculation;
- elitist environmental selection from parent + offspring union;
- deterministic candidate identities from decision vectors;
- complete final population and nondominated archive;
- config and report SHA-256 receipts;
- deterministic replay through \`verify_nsga2_report\`.

The optimizer minimizes native ZDT1/DTLZ2 objective values. It does not use the Symthaea Pareto evaluator to select individuals; this avoids using the new analysis primitive to choose the baseline's own offspring. The final archive is converted into a \`ParetoAnalysisPlan\` through \`analyze_nsga2_report\`, which uses the common indicators for post-hoc measurement. Repeated identical decision vectors are deduplicated only at this analysis boundary; the optimizer's evaluation count remains unchanged.

## Frozen parameters

Every run records the SHA-256 of:
- population size;
- generations;
- crossover probability;
- per-variable mutation probability;
- SBX distribution index;
- mutation distribution index;
- RNG seed.

The number of objective evaluations is exactly:

\[
N_{\mathrm{eval}}=N_{\mathrm{pop}}\,(1+G)
\]

for this generational implementation (initial population plus one offspring population per generation). The report records the measured evaluation count rather than inferring it later.

The test fixture uses population 20, 5 generations, crossover probability 0.9, mutation probability 0.1, SBX eta 15, polynomial mutation eta 20, and fixed seeds. These are deterministic smoke-test parameters, not recommended performance settings or a result.

## Fair comparison protocol

For each benchmark function, dimension, instance and seed:
- give NSGA-II and Symthaea the same objective-evaluation budget;
- use the same decision bounds and objective functions;
- retain every evaluated candidate and the final nondominated archive;
- use the same predeclared normalization anchors, reference point and frozen reference front;
- evaluate hypervolume, IGD+, additive epsilon, feasibility and target-attainment counts with common analysis code;
- report runtime, evaluation count, resource usage and failure rate separately;
- compare every seed and instance, not only the best run.

COCO's recommended fixed-target framing is a better primary comparison than a single final hypervolume: measure runtime/function evaluations to reach a set of preregistered indicator targets. COCO also preserves all nondominated points and versions reference values so results can be recalculated if the reference front improves: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/

## External reference coverage

PlatEMO 4.16, released July 19, 2026, reports 360 algorithms and 630 problems, including large-scale, expensive constrained and multimodal multi-objective methods. This baseline is intentionally a starting point, not a proxy for that platform's full breadth: https://github.com/BIMK/PlatEMO/releases/tag/v4.16

Next baselines, in order:
1. NSGA-II cross-check against PlatEMO 4.16 on ZDT1/DTLZ2 with pinned parameter parity.
2. MOEA/D and SMS-EMOA on two/three-objective cases.
3. NSGA-III and RVEA as objective count grows.
4. Constrained DTLZ and WFG, including deceptive/nonseparable fronts.
5. Expensive/noisy variants with explicit evaluation uncertainty and cost-aware acquisition.

For each external algorithm, pin the repository tag/commit, problem instance, population size, generation/evaluation budget, crossover/mutation parameters, seed set, hardware/runtime environment and the exact metric implementation. The Rust baseline should be checked against a pinned reference implementation; matching the algorithm name does not prove implementation equivalence.

## Tests authored

- same seed/config gives identical report and front;
- ZDT1 and DTLZ2 stay in decision bounds and emit correct objective dimensions;
- exact objective-evaluation budget;
- invalid configuration/dimension rejection;
- minimization dominance logic;
- report replay verification and common Pareto indicator conversion;
- crowding distance handles constant objectives without arbitrarily selecting extremes.

These tests have not been compiled or run. Exact-head acceptance commands:

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_nsga2 -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_benchmarks -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

## Limitations

- This is a compact reference implementation, not yet independently cross-checked bit-for-bit against the original NSGA-II implementation or PlatEMO.
- It covers only ZDT1 and 3-objective DTLZ2.
- It reports algorithm objective-evaluation counts, not machine CPU/GPU/energy telemetry. Benchmark harness measurement must supply those separately.
- No constraints, noisy evaluation model, parallel evaluation or many-objective reference-direction operator is implemented.
- The custom RNG is deterministic for reproducibility but is not presented as a cryptographic or statistical RNG.
- The test configuration is a smoke test, not a performance-tuned baseline.

No claim that this NSGA-II implementation beats SOTA is made. Its purpose is to create a reproducible, auditable baseline for fair evaluation.
