# MACE paired ITT estimator v0.1

A standard-library-only Go reference for paired agent-evaluation studies. It estimates the effect of **skill availability** across every preregistered assignment. Retrieval, selection, loading, invocation and execution are reported as mechanism diagnostics; they never determine whether an assignment enters the primary estimates.

## Four separate endpoints

1. **Task success:** pass=1, fail=0.
2. **Mandatory-control passage:** pass=1, not_passed=0. Missing mandatory evidence should be classified upstream as not_passed when the gate cannot be established; indeterminate is reserved for an oracle that cannot adjudicate the control result.
3. **Clean success:** 1 only when task passes AND mandatory controls pass. A known task failure or known control non-pass fixes clean success at zero even when the other axis is indeterminate.
4. **Unauthorized-effect incidence:** present=1, absent=0. A positive with-skill-minus-without-skill effect is worse on this endpoint.

These endpoints are not combined into one score.

## Primary estimand and post-treatment bias

The primary estimand is the paired intention-to-treat (ITT) difference in task-macro outcomes, with-skill minus without-skill, over all assignments in the supplied roster. Retrieval and later stages occur after skill availability has been assigned. Conditioning the primary estimate on those stages can select a different subset. MACE retains them as diagnostics only.

Repeated seeds are averaged within task; task clusters then receive equal weight. Each estimator invocation accepts one experiment ID and one configuration-stratum SHA-256 digest, and rejects mixed strata. Each pair records task revision, seed/block, and arm execution order. These fields improve comparability; they do not prove randomization or equal internal randomness.

## Unknown outcomes

Every endpoint accepts binary outcomes and indeterminate. Indeterminate outcomes remain as [0,1] and propagate through pair differences and task-macro aggregation. The estimator reports lower and upper partial-identification bounds; it does not silently replace them with a complete-case point estimate.

Task-cluster bootstrap percentile intervals are computed for the lower and upper identification endpoints. If fewer than two task clusters exist, the point bounds are reported and confidence intervals are null; a single task does not support between-task uncertainty estimation.

## Assignment roster

Supply a preregistered roster with every pair ID:

```sh
go run ./cmd/mace-itt -roster fixtures/assignment-roster.json < fixtures/pairs.jsonl
```

The roster check requires exact ID coverage and rejects missing, duplicate or substituted IDs. It does **not** authenticate the roster signature, prove it was preregistered before execution, prove randomization, or establish that the input run records are truthful. Those require an independently authenticated assignment manifest and run observer.

## Run verification

```sh
./verify.sh
```

The verification script runs unit tests, the race detector, go vet, coverage reporting, strict CLI parsing and a fixture report. It does not run an agent benchmark or connect to an actual agent runtime.

## Claim boundary

This estimator does not randomize treatment, validate task or control oracles, authenticate run records, prove hook coverage, establish external effect truth, or guarantee causal identification. Causal interpretation still requires valid randomization, no cross-arm interference, isolation of arm state, matched frozen controls and independently adjudicated outcomes. A configuration hash binds values; it does not authenticate their origin.
