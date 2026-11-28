# MON-045 Benchmark Plan v0.1

All targets below are proposed, not achieved.

| Dimension | Proposed target |
|---|---:|
| Revoked required prerequisite propagation | 100% |
| Expired required prerequisite propagation | 100% held indeterminate |
| Required digest drift detection | 100% |
| Missing required parent handling | 100% indeterminate |
| Cycles and duplicate IDs/edges | 100% blocked |
| Optional-context non-propagation | 100% |
| Unrelated root claims preserved | 100% |
| Canonical graph digest across input order/time-zone representations | 100% |
| Deterministic topological order | 100% |
| Offline production authorization | 0 permitted |

## Scenario families

- Revoke a source-truth receipt used by one recovery claim.
- Revoke a shared capability used by multiple claims; all dependents invalidate.
- Expire one schema snapshot; claims requiring it become indeterminate.
- Change a pinned policy or artifact digest; dependents block.
- Leave one unrelated telemetry claim independent; it remains qualified.
- Add optional context with stale digest; report it without propagating failure.
- Remove a required source node; dependent becomes indeterminate.
- Create a required dependency cycle.
- Reorder graph input; canonical digest and topological result remain identical.
- Give a future revocation timestamp; do not apply it before evaluation time.
- Supply a revocation before issuance; block inconsistent history.
- Omit evaluation time; hold claims because expiry cannot be assessed.

## Comparative experiment

Compare:
1. Global campaign invalidation on any evidence change.
2. Flat per-claim dependency lists with no transitive graph.
3. MON-045 typed DAG evaluation.
4. MON-045 plus MON-041 authenticated evidence receipts and MON-044 capability scope.

Measure:
- impacted-claim precision and recall against an independently authored dependency oracle;
- unrelated-claim invalidation rate;
- missed invalidation rate;
- stale-claim exposure time after revocation;
- time to recompute affected claims;
- graph size, edge count, evaluation latency and memory;
- false blocked and false indeterminate outcomes.

The primary differentiator is exact downstream invalidation with zero unrelated claim damage and no dependent claim surviving a revoked required prerequisite.

## SOTA comparison boundaries

Build systems and content-addressed caches already use dependency tracking and invalidation; MON-045 applies that pattern to assurance claims with typed evidence, three-valued verdicts, expiry, revocation, exact digest pins and claim-level impact traces. The novelty claim must be established by literature review and matched implementation comparisons, not asserted from the architecture alone.
