# MON-045 — Evidence Dependency Graph and Scoped Invalidation Contract v1

Profile: mon045-evidence-dependency-input-v1. Status: standalone offline evaluator and synthetic regression corpus.

## Graph model

A graph contains typed nodes:
- evidence;
- capability;
- scope;
- policy;
- claim.

An edge points from prerequisite to dependent:
- requires: mandatory dependency, exact prerequisite digest required;
- context: optional context, visible in the trace but never used to satisfy or invalidate a mandatory requirement.

Every claim is evaluated from its own declared status plus all inbound required prerequisites. Required dependencies compose with three-valued logic:
- any blocked parent or digest mismatch => blocked;
- otherwise any indeterminate, expired, unauthenticated or missing parent => indeterminate;
- all required parents qualified and own node qualified => qualified.

A blocked state takes precedence over indeterminate. Optional context never changes the dependent verdict.

## Validity

Every node binds:
- stable node ID and kind;
- 32-byte lowercase SHA-256 digest;
- declared verdict;
- issuer ID;
- issue and valid-until timestamps;
- optional revocation timestamp;
- authenticated flag.

Expiry at or before evaluation time holds indeterminate. A revocation at or before evaluation time blocks. A revocation timestamp earlier than issuance blocks as inconsistent history. An issue time later than the evaluation time plus the configured clock skew blocks.

The authenticated flag is not itself trusted by this module; it is input for MON-041-authenticated adapters to populate.

## Determinism

- Node IDs, edges and root IDs are sorted for the graph digest.
- Node timestamps are normalized to UTC before digesting.
- Topological sorting uses lexicographically ordered ready nodes.
- Inbound edges, dependency paths, invalidation sets and reason codes are sorted.
- Duplicate node IDs and duplicate edge tuples block.
- Cycles block the graph.

## Scoped invalidation semantics

Only required edges propagate verdicts. If a source artifact is revoked or expires:
1. evaluate that node;
2. propagate the resulting state to direct required dependents;
3. propagate transitively through dependent capability/policy/claim nodes;
4. report only root claims in that dependency closure as impacted;
5. keep unrelated root claims unaffected and independently qualified.

Orphan invalid nodes remain in invalidated_node_ids for observability but do not downgrade unrelated root claims.

A required dependency digest mismatch blocks because a dependent claim is pinned to a different input. A missing required node is indeterminate, not silently removed from the dependency set.

## Non-claims

- GraphVerdict qualifies only the supplied root claims against supplied node states.
- It does not prove that the root-claim list is complete.
- It does not prove missing dependency edges were not omitted.
- It does not authenticate bytes, issuers, revocation authority, clocks or trust snapshots.
- It does not enforce a mutation boundary or authorize traffic promotion.
- CampaignVerdict remains indeterminate in v1; ProductionActionPermitted remains false.

MON-041 custody and MON-044 independent capability inventory are prerequisites for runtime use.
