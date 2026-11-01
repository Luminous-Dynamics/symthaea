# Preregistered comparative benchmark protocol for dynamic e-closure

**Status:** protocol and manifest implementation draft. This document defines matched comparison contracts; it does not claim any method currently beats state of the art.

## 1. Why a shared protocol is necessary

The current tracks do not all make the same assumptions:

| Track | Statistical route | Principal limitation |
|---|---|---|
| #9538 | Global affine projective mergers plus persistent adjuster | Fixed global weight budget; finite exact optimizer |
| #9539 | Self-financing / predictable portfolio mixtures | Must be compared against AWAIRE-style prior art |
| #9648 | Predictable arrivals localized to finite deterministic envelopes | Requires a finite horizon envelope |
| #9684 | Countable prior over all finite intersections | Prior mass is split over intersections |
| #9695 | Countable global-projective affine closure | Requires global coordinate weights and neutral dormant padding |
| #9712 | History-dependent joint intersection betting | Requires a correctly specified common conditional model |

A method should not be declared more powerful because it was tested under a friendlier arrival process, a different filtration, or a different error-control target. The protocol therefore separates statistical guarantee classes from operational decision quality and single-intersection component diagnostics.

## 2. Comparison contract

A matched comparison must register and hash:

1. data generator and conditional law;
2. global filtration and observation-delay semantics;
3. predictable arrival policy;
4. null configuration;
5. alternative configuration;
6. reporting/monitoring schedule;
7. capacity and utility/cost contract.

The following must also agree between methods in one comparison cell:
- guarantee class;
- alpha;
- generator and all comparison-contract hashes;
- scenario ID;
- master seed and replicate index;
- sample count and horizon.

Method-specific assumptions (e.g. an assumed covariance upper bound) are recorded separately and stress-tested. They are not silently equalized or omitted.

## 3. Guarantee-class leaderboards

Run four distinct leaderboards:

### A. Stopped-FDR

The target is

\[
\mathbb E[\operatorname{FDP}(R_\tau)]\leq\alpha
\]

for every declared a.s.-finite global stopping time. Use the method's actual e-process and closure assumptions.

### B. SupFDR

The target is

\[
\mathbb E[\sup_t\operatorname{FDP}(R_t)]\leq\alpha.
\]

Time-monotone adjusted certificates and their integral budgets are required where the theorem needs them. Ordinary e-process validity is not sufficient.

### C. Component e-process crossing

This is a diagnostic benchmark for one fixed intersection test:

\[
P_{H_S}\left(\sup_{t\leq T}E_t^S\geq1/\alpha\right).
\]

It can evaluate joint betting, covariance calibration, and time-to-threshold. It is **not** a multiple-testing FDR result.

### D. Operational decision quality

Within a matched stopped-FDR or SupFDR guarantee, optimize the same value, cost, capacity, persistence, and queue rules. Report utility regret and false investigations as well as statistical metrics.

## 4. Scenario matrix

### Dependence
- independent coordinate increments;
- shared-control Gaussian dependence;
- block equicorrelation and common shocks;
- comonotone and adversarially coupled null e-process increments;
- covariance below, at, and above registered upper bounds;
- heteroskedastic and time-varying conditional covariance;
- heavy-tailed/sub-Gaussian conditional-mgf stress cases where a proof is available.

### Arrivals and filtration
- deterministic staggered arrivals;
- random predictable arrivals based on lagged observations;
- bursty arrivals and long dormant periods;
- explicit same-round lookahead negative controls;
- continuing evidence after activation;
- delayed, missing and asynchronous observations;
- local/coarsened evidence streams that require adjust-then-combine.

### Signal structures
- global null;
- sparse, dense and heterogeneous alternatives;
- one strong signal among many weak signals;
- clustered and dispersed signals;
- nonstationary effects and treatment/control drift;
- asymmetric utility/cost ratios and capacity saturation.

For the joint Gaussian method, report the model-matched covariance setting and robust covariance-bound settings separately. For arbitrary-dependence methods, preserve their guarantee under the same simulated dependence without granting a stronger model assumption.

## 5. Statistical metrics

Report:
- FDR and SupFDR estimates with Wilson or exact binomial intervals;
- probability of any false discovery;
- power and number of true discoveries;
- precision@B and recall@B;
- false escalations / false investigations;
- time-to-first discovery and time-to-useful certification;
- intersection log-wealth and calibration penalty;
- expected utility and regret;
- capacity utilization, queue delay and unserved actionable work;
- sensitivity to prior/weight allocation and index ordering;
- sensitivity to covariance misspecification and nuisance bounds.

Use 5,000 exploratory streams followed by 50,000 frozen confirmatory streams. Freeze generator versions, seed derivation, analysis rules, and confirmatory seed range before inspecting confirmatory outcomes. Simulations diagnose performance; theorem assumptions establish validity.

## 6. Optimization and solver scorecard

For #9695 and related solver-backed methods, benchmark n=2,4,8,16,32,64,128,256,512,1000.

For small n, compare with the exact exhaustive oracle. For larger n, preserve:
- time to first feasible incumbent;
- time to proven optimality;
- primal objective and dual bound;
- relative MIP gap and node count;
- presolve reductions and model nonzeros;
- p50/p95 end-to-end decision and exact-verifier latency;
- peak memory and certificate size;
- incumbent validity after time limits;
- utility regret against the best known/exact reference.

Compare at least three solvers where available (HiGHS, SCIP, CBC, and Gurobi/CP-SAT where licensed) under identical hardware, core/thread limits, time limits, solver tolerances, and generated instances. A feasible incumbent accepted by the exact verifier is not an optimality proof. Report solution quality and proof time separately.

The affine closure MILP has O(n) variables and rows, but the binary utility/capacity/persistence problem remains combinatorial. Do not equate compact formulation size with polynomial-time optimization.

## 7. Immutable manifest and seed derivation

The new \`validation/dynamic_eclosure_benchmark_manifest.py\` provides:
- a versioned JSON manifest schema;
- SHA-256 hashes for source, generator and comparison-contract dimensions;
- deterministic unsigned 64-bit replicate seeds derived from master seed, scenario ID and replicate index using SHA-256 (never Python's randomized \`hash()\`);
- a canonical JSON manifest digest;
- validation of the manifest and seed derivation;
- a comparator that rejects unlike guarantee classes, alpha values, generators, filtrations, arrivals, nulls, alternatives, reporting schedules, capacity/utility rules, scenarios, seeds, sample counts or horizons.

No wall-clock timestamps are included in a manifest. Method-specific assumptions and solver settings are retained separately so a comparison can acknowledge, rather than conceal, different assumptions.

## 8. Run artifacts

Every run should emit:
- the canonical JSON manifest and its SHA-256 digest;
- source and generator SHAs;
- PR/commit head SHA;
- alpha, guarantee class, scenario, seed, sample count and horizon;
- null/alternative, dependence, arrival, filtration and reporting configuration;
- utility, cost, capacity and persistence settings;
- environment, package and solver versions;
- raw per-stream outcomes or sufficient deterministic aggregates;
- exact verifier receipts and solver logs;
- a result status that distinguishes proven optimal, feasible incumbent, timeout, infeasible and invalid certificate.

## 9. Initial implementation and remaining work

Current files:
- \`validation/dynamic_eclosure_benchmark_manifest.py\` — deterministic seed derivation, canonical manifest builder, SHA-256 validation and matched-contract verifier.
- \`validation/test_dynamic_eclosure_benchmark_manifest.py\` — reproducibility, seed tampering, guarantee-class mismatch, filtration mismatch, invalid alpha/metadata and duplicate-method controls.

Next:
1. Add thin method adapters for each research track without forcing unlike procedures into one false abstraction.
2. Add common exact null increment and shared-control Gaussian generators.
3. Build the small-family cross-method oracle (n=2..12).
4. Run exploratory null/alternative sweeps.
5. Freeze confirmatory seeds and scorecard.
6. Execute solver comparisons under pinned hardware/software settings.

No SOTA or production-scale claim is justified until this protocol is executed with aligned guarantees and preserved artifacts.
