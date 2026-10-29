# AVIATION-ENTERPRISE-015 — Adaptive recovery screening contract

## Purpose and claim boundary

This module explores whether a finite aircraft-recovery candidate universe can be searched faster by screening candidates in a fixed order and expanding only when the best verified plan is not yet justified by admissible lower bounds.

The first implementation is a **generic bounded research contract**, not a complete airline-recovery optimizer. It does not yet enumerate operational aircraft swaps, encode crew/maintenance/airport rules, produce optimization bounds, or connect to dispatch. Its callback is a contract boundary: `VerifiedFeasible` must mean an independent checker has re-evaluated all hard constraints and recomputed the objective from frozen inputs. A caller must not return this variant merely because a ranking or solver says so.

The implementation does not prove the admissibility of caller-supplied bounds. The tiny-instance oracle and later bound-verifier implementation must independently test this premise. A false lower bound can make a false optimality claim possible; hashes alone do not make a bound mathematically valid.

## Paradigm shift

The conventional pipeline is:

1. Rank or prune recovery candidates.
2. Optimize the restricted set.
3. Return its best plan, sometimes described as optimal.

This is unsafe: optimality over the screened set does not imply optimality over the original set.

The adaptive contract instead requires:

1. Freeze the complete candidate universe and provenance.
2. Rank candidates deterministically; the ranking changes work order, not feasibility or objective semantics.
3. Evaluate a small batch using an independently checked feasibility/cost path.
4. Compare the best verified feasible cost to a lower bound over every unresolved candidate.
5. Stop only if the absolute gap is within the preregistered tolerance. Otherwise expand the set.
6. If the budget ends first, return `FeasibleUncertified`, never an optimality claim.
7. Preserve candidate IDs, bounds, exact outcomes, and unresolved reasons.

The implementation uses integer objective units for exact gap arithmetic. A future percentage tolerance must define a fixed nonzero normalization denominator before test evaluation; it must not be improvised from results.

## Core definitions

For a minimization objective:

- (U): objective cost of the best independently verified feasible candidate.
- (L_i): admissible lower bound for each unresolved candidate (i).
- (L = \min(U, \min_i L_i)) when a feasible candidate is known.
- Certified absolute gap: (G = U - L).
- Acceptable stopping condition: (G \leq \varepsilon), where (arepsilon) is frozen in the run policy.

A lower bound greater than a candidate's independently recomputed feasible cost is rejected. That check detects one class of invalid bound but does not prove the bound admissible for unexamined candidates.

If no verified feasible candidate exists, the module does not publish a feasible objective bound. If any candidate remains indeterminate and no feasible plan exists, the result is `IndeterminateEvidence`, not `NoFeasiblePlan`.

## Benchmark matrix

Every method must receive the same frozen scenario, full candidate universe, hard constraints, objective definition, and resource budget.

| Arm | Method | What it isolates |
|---|---|---|
| A | Full exact enumeration on tiny instances | Oracle for true feasible optimum |
| B | Full candidate evaluation, no screen | Screening overhead vs baseline |
| C | Stable-ID greedy / heuristic rank, no bound stop | Ranking quality only |
| D | Ranking plus hard candidate cap, no expansion | Unsafe ablation; optimality claims forbidden |
| E | Adaptive expansion with verified outcomes and admissible bounds | Proposed method |
| F | Recovery-aware features (swap savings and disruption proximity) | Feature-engineering contribution |
| G | ML-ranked candidates with the same verifier and bound protocol | Ranker contribution, not a separate safety guarantee |
| H | Hindsight/perfect-information plan | Upper/lower-bound context only; not deployable |

For larger instances, compare only solvers that use the same mathematical model, feasible action set, objective scaling, hardware class, thread budget, and timeout. A commercial or open MILP/CP-SAT result is not comparable when its formulation or constraints differ.

## Required correctness gates

- Exhaustively compare tiny instances with an independent raw-input oracle.
- Include zero-disruption controls and vary fleet size, overlapping flights, turnaround constraints, resource scarcity, and multiple simultaneous disruptions.
- Inject duplicate/missing candidate IDs, wrong scenario/policy digests, false-negative screens, non-admissible lower bounds, bound/objective contradictions, infeasible candidates, and indeterminate verifiers.
- Require zero hard-constraint violations among accepted feasible plans.
- Require zero false `OptimalWithinTolerance` reports across the frozen negative-control matrix.
- On timeout, return a feasible but uncertified plan only when one has been independently checked; otherwise state that no feasible plan was established within budget.
- Keep every screened-out candidate and its bound in the report; never silently discard a candidate.
- Retain failures and null results in the reproducibility record.

## Performance and quality targets

Targets, not achieved results:

- Research target: at least 66% lower mean end-to-end runtime than a matched full-model baseline while preserving the identical hard constraints and a preregistered absolute/normalized gap. This is inspired by a recent published recovery result; it is not a Symthaea result and must not be compared without matched instances and constraints.
- Report p50/p95/p99 end-to-end latency, time to first verified feasible plan, count of evaluated candidates, expansion count, full-search fallback rate, timeout rate, infeasibility-indeterminate rate, best cost, global lower bound, final gap, and feasibility failure rate.
- Report objective quality and safety/correctness separately. Do not compress delay, cancellation, actor fairness, carbon emissions, and cost into an opaque single metric.
- Bootstrap comparative uncertainty by disruption episode or another frozen dependence cluster, not individual correlated flights.

## Industry comparison and evidence tiers

A published October 2026 study reports an average 66% end-to-end recovery-runtime reduction at comparable solution quality using recovery-specific swap-cost and disruption-aware features across five major Chinese airlines (DOI: [10.1016/j.trb.2026.103566](https://doi.org/10.1016/j.trb.2026.103566)). That percentage is a benchmark target only until the task, constraints, hardware, runtime accounting, and gap are matched.

A separate 2026 study evaluates airline recovery coordinated with network-manager capacity/disruption decisions and a fairness mechanism. This motivates separate actor-level and network-level outputs, but its results are not directly comparable to the synthetic fixture (DOI: [10.1016/j.tre.2026.104977](https://doi.org/10.1016/j.tre.2026.104977)).

Keep evidence tiers explicit:
1. Unit and independent-oracle results on synthetic finite instances.
2. Reproducible public-data replay with frozen snapshot and terms.
3. Matched simulator policy comparisons with stated counterfactual assumptions.
4. Authorized operational evaluation with operator-reviewed constraints.
5. Safety/certification claims only under their applicable independent processes.

Evidence from an earlier tier does not imply evidence for a later one.

## Reproduction record

Every report must bind:

- Scenario/input snapshot digest and stable scenario ID.
- Complete candidate-universe digest over canonical candidate IDs, rank scores, and admissible lower bounds.
- Candidate universe and stable IDs.
- Screening policy digest and ranking artifact digest.
- Source commit IDs for screening and independent feasibility verification.
- Candidate ranking score and admissible bound for every candidate.
- Exact outcome and recomputed cost for every evaluated candidate.
- Best feasible cost, global lower bound, absolute gap, declared tolerance, and honest terminal status.
- Evaluation budget and any timeout, expansion, fallback, or indeterminate reason.

The current module validates digest encoding and echoes caller-supplied scenario, candidate-universe, policy, ranking-artifact, and source identities. The runner must calculate these digests from canonical frozen artifacts, and evidence-signing work must bind them to trusted receipts before any promotion gate consumes the report.

## Current scope and next integration steps

The Rust module `aviation_adaptive_screen.rs` provides a deterministic finite-candidate control loop with explicit bounds, audit records, and fail-closed terminal statuses. The separately compiled integration oracle uses an independent truth table to test expansion, stopping, budget exhaustion, indeterminate evidence, false bound witnesses, stable tie-breaking, and infeasibility.

Still required before calling this an optimizer:
1. Build the full candidate universe from a frozen recovery model and produce mathematically admissible per-candidate lower bounds.
2. Bind candidate outcomes to the independent hard-constraint/cost verifier.
3. Add deterministic scenario generators and exhaustive assignment checks against `aviation_enterprise` where the model semantics align.
4. Add per-stage runtime measurement and a matched baseline runner.
5. Bind the report into the evidence-bundle/readiness protocol and test mutation/rollback cases.
6. Run `cargo fmt --check` and `cargo test --locked -p symthaea-operations-research` on the exact final source SHA in a Rust-equipped environment.

No compilation, runtime benchmark, airline savings, SOTA superiority, operational-safety, dispatch-authority, or certification claim is made by this document.
