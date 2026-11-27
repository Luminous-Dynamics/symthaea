# MON-039 — Counterfactual Fault-Set Reduction Contract v1

Profile: mon039-counterfactual-fault-reduction-input-v1. Status: deterministic replay-table minimizer and synthetic known-answer cases; live fault-scheduler adapter pending.

## Objective

Given a deterministic fault schedule whose independent recovery verifier returns a failure, identify a smaller subset of injected interventions that preserves the exact same verifier outcome. The minimizer narrows hypotheses; it does not prove physical or metaphysical causation.

## Unit of analysis

Each observation is keyed by the canonical sorted fault-ID set and binds:
- scenario SHA-256;
- recovery-boundary SHA-256;
- exact source code revision;
- observation time;
- verifier disposition;
- failure-signature SHA-256;
- outcome-receipt SHA-256;
- completeness assertion.

All subset observations must share the same scenario, recovery boundary and code revision. Missing, incomplete, duplicate or stale rows cannot be interpreted as absence of failure.

The target signature is the exact (verdict, failure_signature_sha256) pair observed for the complete initial fault set. A blocked result with a different signature is not treated as preserving the target failure.

## Search and guarantees

For n <= 12:
- enumerate all proper subsets in increasing cardinality and deterministic lexicographic order;
- stop at the first subset preserving the exact target failure, once all smaller cardinalities have complete observations;
- claim global_candidate_minimum only over the submitted fault universe and pinned replay conditions.

For n > 12:
- repeatedly attempt deterministic single-fault deletions;
- retain a deletion only if the exact target failure persists;
- claim one_minimal only when every single deletion from the final set has been evaluated and none preserves the target;
- do not infer global minimum or monotonicity.

If a replay row is missing, incomplete, exceeds the evaluation budget, or changes the code/boundary/scenario, minimality is not_established or the input is blocked. Missing evidence is not a no-failure result.

## Interpretation

A global candidate minimum is globally minimum only within the submitted, finite fault set and deterministic oracle. A one-minimal set is irreducible under any one-fault deletion, but a smaller multi-fault subset may still exist in a non-monotone system. Neither result alone proves causality outside the tested intervention space.

Symthaea is a hypothesis proposer and prioritizer. MON-037/MON-038 and the independently captured verifier outputs remain the adjudication oracle. The model must not generate the expected result and then certify its own generated explanation.

## Limits

The module consumes precomputed replay observations. It does not start chaos experiments, call Kafka/PostgreSQL, restore backups, authenticate receipts, or control a production system. Offline and runtime-candidate inputs remain campaign-indeterminate pending a trusted deterministic replay adapter and evidence custody. No production action is permitted.
