# AR-5B5 Pareto objective evaluator v0.1

**Status:** implementation candidate committed to the AR-5B4 research branch; tests authored but not compiled or executed. This is an analysis primitive, not a benchmark result.

## Why this tranche exists

The existing resource_pareto module performs single-step allocation under independent hard resource limits and exposes resource/capability trade-offs. The new pareto_objectives module addresses a different layer: evaluating candidate outcomes across a declared vector of objectives without collapsing them to one scalar reward.

A research candidate may improve quality while consuming more energy, improve robustness while reducing throughput, or be infeasible despite a high raw score. These outcomes must remain visible.

## Implemented API

File: crates/domains/symthaea-architecture-engineering/src/pareto_objectives.rs

- ObjectiveDirection::{Maximize, Minimize}: each raw objective has an explicit direction.
- ParetoObjectiveSpec: freezes objective ID and finite scale anchors.
- ParetoCandidate: carries a raw objective vector, explicit feasibility status, constraint-violation reasons and an evidence-artifact digest.
- ParetoAnalysisPlan: freezes objective order, campaign identity, reference point and candidate set. Candidate IDs are canonicalized.
- analyze_pareto: transforms objectives into maximization utilities, excludes infeasible candidates, computes strict Pareto dominance, retains the nondominated front, and returns exact hypervolume and a digest-bound receipt.
- analyze_reference_sensitivity: computes hypervolume under the primary reference point and each explicitly supplied alternative; reports min/max/range and a digest-bound sensitivity receipt.

### Objective transformation

For an objective with fixed lower and upper scale anchors L < U:

\[
u(x)=
\begin{cases}
(x-L)/(U-L), & \text{maximize}\\
(U-x)/(U-L), & \text{minimize}
\end{cases}
\]

Anchors determine units and orientation; values are not clipped. Values outside the anchor interval remain visible as utility values below zero or above one. This prevents silent saturation.

Feasibility is not another objective. Infeasible candidates remain in the receipt with violation reasons, but cannot enter the nondominated set or hypervolume.

### Hypervolume

The current implementation uses a recursive exact slicing algorithm for two through four objectives. “Exact” refers to the geometric algorithm; arithmetic uses IEEE-754 f64. The reference point is explicit and expressed in transformed utility coordinates. The method is intended for small-to-moderate archived fronts, not millions of points.

The API reports:
- feasible, infeasible and dominated candidate counts;
- nondominated candidate IDs and deterministic dominance witnesses;
- each candidate's transformed utility, feasibility reasons and evidence digest;
- hypervolume at the fixed reference point;
- plan and analysis receipt SHA-256 digests.

Reference sensitivity is a separate output. A single HV value is never presented without its reference point.

## Current limitations

1. This is a batch analysis primitive. It does not yet stream candidate events directly from EventTape.
2. The evidence SHA-256 is a content reference. This module does not fetch or independently verify artifact bytes; that remains the responsibility of the retained evidence/verifier path.
3. The current exact-HV plan supports at most four objectives. Higher-dimensional search needs separately validated approximation or non-HV indicators; it must not silently use an approximation under the same label.
4. The module does not yet calculate IGD+, additive epsilon, R2, attainment surfaces, uncertainty-aware dominance, or confidence intervals across independent campaign instances.
5. Reference-point sensitivity is computed, but the benchmark protocol still needs to freeze its primary point and sensitivity set before confirmatory results are examined.
6. The objective set is supplied by the evaluator. Policies must not choose objective transformations, constraints or reference points after observing their outcomes.

## Tests authored

- mixed maximize/minimize transformation and known dominance witness;
- exact two-dimensional and three-dimensional hypervolume fixtures;
- exact four-dimensional single-box volume;
- reference-clipped volume;
- infeasible-candidate exclusion from the front and hypervolume;
- reference-point sensitivity and receipt;
- candidate-order invariance of plan and receipt digests;
- malformed dimensions, feasibility metadata and evidence digests;
- duplicate reference-point rejection.

These tests have not been run in this environment. Exact-head acceptance commands:

    cargo fmt --all -- --check
    cargo test -p symthaea-architecture-engineering pareto_objectives -- --nocapture
    cargo test -p symthaea-architecture-engineering
    cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings

## Benchmark programme

### Track 1 — canonical synthetic multi-objective functions

Use:
- ZDT as a two-objective sanity suite;
- DTLZ1–7 for scalable objective counts and degenerate/convex/concave front shapes;
- WFG1–9 for non-separability, deceptive properties and varied front geometry;
- constrained DTLZ variants for feasibility handling;
- ZCAT/LSMOP only when extending beyond the current exact-HV dimensions and archive-size envelope.

At v0.1, run two-, three- and four-objective cases only. The three-objective campaign should contain distinct quality, robustness and resource/energy axes. Keep functions and instances separate, with at least 30 independent seeded runs per algorithm/instance as an exploratory floor; final sample sizes must be power-based.

### Track 2 — algorithms to compare

- NSGA-II: established general-purpose baseline.
- NSGA-III: reference-direction many-objective baseline for later 3+/many-objective work.
- MOEA/D: decomposition baseline.
- SMS-EMOA: hypervolume-based selection baseline.
- RVEA: reference-vector baseline.
- Random/uniform archive: sanity baseline.
- Symthaea DirectionBoard: mechanism under test.
- MAP-Elites/CVT-MAP-Elites: only when descriptor dimensions have domain meaning.
- ShinkaEvolve/CodeEvolve-style systems: best-practical-system track, not allocator-only comparison.

Pin implementations, versions, parameters and termination budgets. Do not compare a method's published score to a different test-suite instance as if conditions matched.

### Metrics and reporting

Primary:
- hypervolume at a preregistered, shared reference point;
- runtime/resource cost to reach fixed HV target levels;
- probability of reaching each target before budget exhaustion;
- quality-versus-spend and time-to-target curves.

Required companion metrics:
- IGD+ against a declared reference Pareto front;
- additive epsilon indicator;
- nondominated-set size and coverage;
- feasibility rate and constraint-violation distribution;
- worst-instance and worst-family performance;
- per-objective distributions and resource vectors.

Use multiple fixed reference points as sensitivity analysis, not as a post-hoc choice. Freeze objective normalization and reference sets, and retain all candidate archive points so analyses can be recalculated if a reference set changes. Report medians, dispersion, confidence intervals and per-instance outcomes; avoid winner-only or best-of-N reporting.

### Comparison protocol

Mechanism-matched comparisons share the exact candidate/offer stream, objective definitions, feasibility rules, evaluator/verifier versions, seed blocks, worker schedule and resource accounting. Best-practical-system comparisons allow each system its native proposer and orchestration, but charge total compute, energy, memory, token, tool and verifier costs. Publish the two tracks separately.

The goal is not simply a bigger hypervolume. The target is a better verified Pareto frontier per unit cost, with no hidden feasibility violations and no unacceptable worst-case regressions.

## External methodological references

- COCO's biobjective performance assessment uses the nondominated archive's hypervolume and measures runtime/function evaluations to reach prescribed quality targets. It also records versioned hypervolume reference values and every nondominated solution so results can be recalculated if reference sets improve: https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- Research on indicator reliability reports that HV, IGD, IGD+ and additive epsilon can disagree materially on real-world problems, and HV comparisons can change with reference-point choice. This motivates mandatory multi-indicator reporting and reference sensitivity: https://scholars.cityu.edu.hk/en/publications/reliability-of-indicator-based-comparison-results-of-evolutionary/
- DTLZ and WFG remain core suites for front-shape and many-objective testing; additional suites such as ZCAT and LSMOP target different structural and scale challenges: https://academic.oup.com/jcde/article/13/2/1/8407630
- CodeEvolve's 2026 Findings paper reports that its combined island-evolution, crossover, refinement and archive system matched/surpassed reported AlphaEvolve results on 5/9 tasks and outperformed OpenEvolve/ShinkaEvolve on 6/9 matched tasks. That belongs in end-to-end comparison, not isolated allocator claims: https://aclanthology.org/2026.findings-emnlp.1345/
- ShinkaEvolve reports a 150-sample circle-packing discovery and uses parent sampling, novelty rejection and bandit-based model selection. Include those mechanisms as end-to-end baselines with pinned task/model/budget conditions: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html

## Next implementation steps

1. Add an EventTape adapter that derives each candidate's objective vector and resource dimensions from independently verified candidate outcomes and complete resource receipts.
2. Retain candidate evidence bytes or content-addressed artifact locators, not only digest strings.
3. Add IGD+, additive epsilon and attainment-based analysis with frozen reference sets.
4. Add deterministic DTLZ/ZDT/WFG fixtures and baseline adapters in a separate benchmark module.
5. Add multi-run cluster inference and correction across the preregistered indicator family.
6. Extend beyond four objectives only with an explicitly versioned algorithm and approximation-error tests.

No SOTA result is claimed by this implementation.

## Next paradigm shift: uncertainty-aware Pareto fronts

The v0.1 API classifies dominance on point estimates. That is appropriate for deterministic fixtures, but noisy scientific and engineering measurements need a separate uncertainty-aware layer. Do not silently replace point estimates with confidence bounds in this API.

The next version should report:
- certain dominance when one candidate's conservative utility bounds dominate another's optimistic bounds;
- possible dominance when the uncertainty regions overlap enough that either ordering remains plausible;
- unresolved trade-offs when neither candidate dominates robustly;
- feasibility probability for uncertain constraints;
- distributions of hypervolume and Pareto-front membership across repeated trials;
- tail resource use and worst-case regressions, not only means.

A claim should not be called equivalent merely because confidence intervals overlap. Freeze the uncertainty model, practical effect margin, chance constraints and multiplicity treatment before confirmatory outcomes are examined. Point-estimate and robust fronts should be shown side by side.
