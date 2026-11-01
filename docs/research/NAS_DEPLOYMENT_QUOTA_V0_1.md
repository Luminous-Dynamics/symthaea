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


## Finite candidate pools: hypergeometric quota mode

A fixed benchmark search space is not an iid stream. When drawing unique candidates uniformly without replacement from a finite population of N candidates, of which exactly K meet a predeclared deployment-feasibility definition, the number of successes in n draws follows a hypergeometric distribution:

X ~ Hypergeometric(N, K, n)

The module also supports a finite-population manifest binding the candidate universe, source dataset, target cohort, deployment-outcome definition, total candidate count and known deployable count. It computes the smallest sample size satisfying P(X < quota) <= delta using a log-space adjacent-PMF recurrence and binary search. The plan is re-computed against the exact population manifest during verification.

### NATS-Bench-MCU v2 illustrative finite-pool calculation

The official v2 record reports 15,625 candidates, of which 14,281 were successfully profiled and 1,344 were flash-infeasible under the nRF5340 pipeline and 746 KiB INT8 flatbuffer limit. Under **uniform random sampling without replacement** from that exact fixed population, a quota of 20 flash-deployable candidates at a maximum 5% probability of missing the quota requires a sample of 25 unique architectures.

- At 25 draws, modeled probability of fewer than 20 deployable candidates: 0.0170689547.
- At 24 draws, modeled probability of fewer than 20 deployable candidates: 0.0501382598.

This is a mathematical consequence of the published population counts, not a measured NAS algorithm result, a reproduction of the CSVs, or a probability floor for another board/toolchain/policy. It applies only to uniform sampling without replacement from that exact finite population and that specific definition of deployability.

### Model selection rule

- Use the **hypergeometric** mode for a known fixed pool sampled uniformly without replacement.
- Use the **binomial/conditional-floor** mode for independent deployment attempts or adaptive attempts whose conditional success probability is validly lower-bounded at every step.
- Use neither mode if the selection policy or success process satisfies neither premise. Stratified/adaptive search needs a corresponding conditional or stratified model.
- Do not use NATS-Bench-MCU's global 91.4% deployment yield as a success floor for quality-ranked, adaptive, different-target or different-toolchain deployment attempts.

Four additional Rust tests cover the NATS v2 known-answer sample, a small finite-population known answer, impossible quotas/population tampering and unknown-field rejection. An independent Python recurrence check matched direct hypergeometric summation across 88,244 small population/size/quota combinations.
