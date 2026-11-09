# Certificate-Sensitive Precision Allocation v1

Research owner: [PHYS-COMPUTE-PRECISION-001 #11189](https://github.com/Luminous-Dynamics/symthaea/issues/11189)

Source reference: `research/physical_compute/precision_allocation_v1.rs`

## Purpose

The reference allocator chooses a precision option for each preconditioner application by minimizing total declared local cost while respecting an additive task-error budget:

\[
\sum_k S_k\,\varepsilon_k\leq\varepsilon_{\mathrm{task}}.
\]

Here (S_k) is an externally supplied sensitivity upper bound and (arepsilon_k) is an externally supplied upper bound on the output error at one precision/read budget.

The module does **not** derive Krylov convergence, sensitivity, physical-noise or hardware-cost bounds. It is a small deterministic policy primitive that refuses to invent missing premises.

## Contract

Each operation provides:
- a unique operation ID;
- a sensitivity upper bound and evidence digest;
- a finite ordered list of precision options;
- each option's read count, output-error upper bound, complete local cost upper bound, evidence class and digest.

Options must have strictly increasing read counts, nondecreasing cost upper bounds and nonincreasing error upper bounds.

The allocator:
1. validates every input and evidence identifier;
2. abstains when sensitivity evidence is missing;
3. computes the complete Cartesian search size and abstains if it exceeds the hard cap;
4. exhaustively enumerates eligible combinations;
5. rounds positive products and sums upward by one representable binary64 value so accumulated bounds do not round down;
6. chooses minimum upper-bound cost among combinations within the task-error budget;
7. returns an explicitly exploratory disposition if any selected error or sensitivity bound is synthetic/modelled.

It never returns a heuristic plan after the search cap is exceeded.

## Qualification boundary

- `AnalyticBoundOnly` means the supplied bounds were labelled deterministic analytical bounds. The allocator does not prove their derivations or the additive decomposition; an independent method-level review must establish both.
- `ExploratoryModelled` means at least one selected bound is synthetic/modelled. It cannot qualify a production execution plan or establish a hardware performance claim.
- Missing sensitivity evidence produces `Abstain(MissingSensitivityEvidence)`.
- An infeasible error budget produces `Abstain(NoFeasibleAllocation)`.
- An excessive exhaustive search produces `Abstain(SearchBudgetExceeded)`.

Statistical confidence bounds, correlated stochastic error composition, runtime calibration and measured device cost models are deliberately out of scope for this V1. They require explicit confidence/dependence contracts before they can be combined with deterministic bounds.

## Standalone test command

When a Rust toolchain is available:

    rustc --edition=2021 --test research/physical_compute/precision_allocation_v1.rs -o /tmp/precision_allocation_v1_tests
    /tmp/precision_allocation_v1_tests

The source currently authors 12 tests covering:
- global minimum-cost feasible selection across multiple operations;
- fail-closed missing-sensitivity handling;
- synthetic-bound plan labelling;
- infeasible budgets and search-cap abstention;
- non-monotone profile rejection;
- duplicate IDs and empty evidence digests;
- non-finite task budgets;
- zero-error budgets and zero-sensitivity operations;
- upward rounding for positive subnormal products that would otherwise underflow to zero.

**Execution status:** tests are authored but have not been compiled or run in the current authoring environment. No Rust, CI, solver or hardware qualification is claimed.

## Next gates

1. Compile and run the 12 tests on the exact PR head.
2. Add property tests comparing allocator results with an independent exhaustive oracle on generated small option spaces.
3. Add rational/interval upper-bound inputs or rigorously specified statistical confidence profiles.
4. Derive FGMRES-specific sensitivity/error-propagation bounds before connecting this allocator to an actual solver.
5. Add paired per-seed fixed-budget and randomized-policy comparisons from #11189.
6. Include outer Krylov cost, FGMRES storage, calibration, verification, fallback and deadline cost in the complete plan cost.
7. Keep all synthetic profiles explicitly exploratory until replaced by qualified evidence.

## Claim ceiling

This module demonstrates deterministic selection over supplied finite error/cost profiles. It does not prove that a chosen read budget achieves the declared error, that the sensitivity model is valid for FGMRES, or that analog execution is faster or more energy efficient than digital execution.
