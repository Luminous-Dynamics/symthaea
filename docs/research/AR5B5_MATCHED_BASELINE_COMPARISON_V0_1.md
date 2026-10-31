# AR-5B5 matched-budget Pareto baseline comparison v0.1

**Status:** comparison harness committed; tests authored but not compiled or executed. It produces a fair, replay-verified comparison receipt for the current in-repo NSGA-II, MOEA/D-DE and SMS-EMOA references.

## Purpose

Three algorithms now expose replayable run reports on the same canonical problem fixtures. Comparing their raw report fields or independently computed metric values would invite subtle mismatches in seeds, objective normalization, reference points or evaluation budgets.

The comparison harness enforces one common contract before it produces a table.

File:
\`crates/domains/symthaea-architecture-engineering/src/pareto_baseline_comparison.rs\`

Public entry point:
\`compare_pareto_baselines(runs, objective_specs, reference_point_utility, reference_front_utility)\`

## Strict matching rules

Every algorithm run must:
1. pass its algorithm-specific deterministic replay verifier;
2. use the same benchmark problem ID (ZDT1 or three-objective DTLZ2);
3. use the same seed;
4. use exactly the same objective-evaluation count;
5. have the same objective dimensions;
6. be converted through the exact same ordered objective specifications and scale anchors;
7. use the same hypervolume reference point;
8. use the same archived reference Pareto front for IGD+ and additive epsilon;
9. have a unique algorithm ID in the comparison.

The harness rejects a mismatch rather than attempting to normalize away a different contract after results are available.

### Current algorithms

- NSGA-II, verified by \`verify_nsga2_report\`;
- MOEA/D-DE, verified by \`verify_moead_report\`;
- SMS-EMOA, verified by \`verify_smsemoa_report\`.

All three archive adapters feed the same \`ParetoAnalysisPlan\` and \`analyze_reference_front_indicators\` implementation.

## Comparison receipt

For every algorithm the report records:
- algorithm and problem ID;
- seed and exact objective-evaluation count;
- final population size and nondominated archive size;
- source run-report SHA-256;
- Pareto analysis plan and receipt digests;
- hypervolume, IGD+ and additive epsilon.

The comparison receipt binds:
- common problem ID and seed;
- matched evaluation budget;
- objective-specification digest;
- hypervolume-reference-point digest;
- reference-front digest;
- canonically ordered per-algorithm results.

This is a deterministic metric comparison, not yet a statistical multi-seed superiority analysis. It does not include measured runtime, energy, memory, tokens or machine-level compute. Those must be measured separately by the outer benchmark harness.

## Initial integration fixture

The unit test uses:
- ZDT1 with 10 decision variables;
- 20-individual population;
- 60 objective evaluations for each method;
- the same seed across all methods;
- common objective anchors f1 in [0,1] and f2 in [0,10];
- a fixed ZDT1 analytic reference-front sample;
- a hypervolume reference point below the normalized feasible utility range.

The test verifies all three methods appear in one receipt and produce finite HV, IGD+ and epsilon values. It also rejects a valid fourth run with 61 evaluations, a duplicate algorithm, an incomplete run set, an incompatible reference point, and reordered objective specs that conflict with SMS-EMOA's frozen config.

## Hypervolume reference point

Because the objective utilities are transformed to a maximization scale, a coordinate of zero would make exact boundary points contribute zero volume in that dimension. The benchmark fixture uses a reference point below the normalized range (for example -0.1 in each coordinate), so the boundary of the feasible Pareto front is included. All compared algorithms must share the same point; reference sensitivity remains a separate analysis.

## Tests authored

- three-way NSGA-II/MOEA/D-DE/SMS-EMOA comparison with the same 60-evaluation budget;
- valid unequal-budget run rejection (61 vs 60 evaluations);
- duplicate algorithm ID rejection;
- too-small comparison input rejection;
- incompatible reference point dimensionality rejection;
- mismatched objective ordering rejection;
- every run is replay-verified before indicator computation.

These tests have not been run. Exact-head acceptance commands:

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_baseline_comparison -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_nsga2 -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_moead -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_smsemoa -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

## What this does not establish

- It does not show that one algorithm wins; all test metrics are synthetic and the test only validates the comparison contract.
- It does not show cross-run statistical superiority; that needs multiple independently seeded runs and cluster-level inference.
- It does not show equal wall time or energy; only objective-evaluation budgets are matched here.
- It does not show external implementation equivalence; NSGA-II, MOEA/D-DE and SMS-EMOA must still be cross-checked against pinned PlatEMO 4.16 or other authoritative reference implementations.
- It does not compare Symthaea's native proposal-generation system to the optimizers as complete systems. This is an algorithmic baseline comparison, not a best-practical-system benchmark.

## Next steps

1. Run 30+ exploratory seeds per problem/instance; determine confirmatory sample size by power analysis.
2. Add independent cluster/bootstrap and paired randomization inference across seeds and problem instances.
3. Add time-to-HV-target and resource-cost target curves from external measured telemetry.
4. Cross-check each algorithm against the same pinned PlatEMO 4.16 algorithm/problem/parameter combination.
5. Add WFG, constrained DTLZ, noisy/expensive and multimodal benchmark families.
6. Add MOEA/D and SMS-EMOA to the mechanism-matched research direction-search comparison only after their evaluator and proposal streams can be frozen identically.

No algorithm superiority or SOTA claim is made by this harness.
