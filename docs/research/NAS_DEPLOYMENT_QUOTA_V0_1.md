# Success-driven verified-deployment quota sizing v0.1

Tracking: NAS-FIDELITY-001 / #9598  
Artifact identity: #9594 / PR #9595  
Physical readback: #9601

## Why a quota planner

A fixed top-K policy assumes that a stable fraction of selected candidates will survive full build, flash, readback and fleet qualification. A campaign that needs 20 verified deployable candidates should plan for failed builds and physical deployment rejection, not stop after evaluating 20 architectures.

This module computes a minimum number of attempts under an explicit success-yield contract.

## Contract

The floor manifest binds the benchmark and target cohort; frozen candidate proposal policy digest; held-out qualification cohort digest; lower bound on complete verified-deployment probability; and independent qualification receipt digest.

For X ~ Binomial(n, p_floor), the planner finds the smallest integer n satisfying:

P(X < quota) <= delta

The binomial CDF is evaluated in log space using the adjacent-PMF recurrence, without factorial overflow or an additional numerical dependency. Binary search finds the minimum attempt budget. The plan records failure probability at n and n-1, and the verifier recomputes the plan against the exact floor.

## Known-answer checks

| Required successes | Lower success probability | Maximum failure probability | Minimum attempts | Modeled failure probability |
|---:|---:|---:|---:|---:|
| 1 | 0.50 | 0.05 | 5 | 0.03125 |
| 10 | 0.50 | 0.05 | 28 | 0.0435792767 |
| 10 | 0.70 | 0.05 | 19 | 0.0325533569 |
| 25 | 0.80 | 0.01 | 39 | 0.0060484827 |
| 50 | 0.90 | 0.01 | 62 | 0.0075689487 |

These are mathematical fixtures under the stated probability model, not observed NAS yield results.

## Critical assumptions and non-claims

The guarantee is only as good as the success-probability floor and the attempt model.

- Attempts must be independent under the registered proposal policy, or have a valid conditional success-probability lower bound at each step.
- The success floor must apply to the exact task, target cohort, candidate-generation policy, toolchain and deployment/readback definition.
- An adaptive policy that changes candidate distribution after observing outcomes may violate the stationary Bernoulli model. Requalify the floor per policy, stratify cohorts, or develop a sequential conditional-bound model.
- Correlated candidates, board-level batch effects, compiler changes, target shift, time drift and retrying the same candidate as if it were independent are not covered automatically.
- The module only validates receipt shape and pins its digest. It does not independently inspect the referenced qualification dataset or prove the floor. Independent calibration/qualification is a prerequisite.
- Success must mean a complete artifact-bound deployment meeting the physical readback and fleet evidence gate—not merely a build or programmer exit code.

## Test plan

Eleven Rust tests are authored for known-answer cases, exact minimality, monotonicity in floor/confidence, scope binding, plan/floor tampering, invalid values and unknown-field rejection. Tests have not been compiled or executed in the current environment; no Rust test or CI pass is claimed.

## Next extension

A later tranche should add a deployment-attempt ledger that accepts only a separately verified readback capability, appends each success/failure with hash chaining and unique attempt identity, and stops when the quota is met or the attempt budget is exhausted. A caller-supplied success=true flag must never count as a verified deployment.
