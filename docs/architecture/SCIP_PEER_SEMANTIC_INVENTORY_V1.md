# SCIP Peer Semantic Inventory v1

Status: **experimental / draft session contract**

Crate: `crates/bridges/symthaea-interlingua`

This note defines the session-local prerequisite that lets SCIP safely choose
cached semantic references and exact graph deltas, plus the feedback path used
when previously acknowledged cache state is no longer available.

## Problem

The low-level SCIP transfer planner accepts optional byte counts for:

- a semantic reference, when the receiver already owns the exact target graph;
- an exact `GraphDelta`, when the receiver already owns the exact base graph.

Those options are intentionally caller-supplied at the low-level planning API.
Without a higher-level possession contract, a caller could optimistically expose
a reference or delta candidate without actually knowing that the peer owns the
required content-addressed graph.

Even a valid acknowledgement can later become stale because the receiver may
evict cached semantic state. Without explicit feedback, a stale possession claim
could repeatedly select a reference or delta that the receiver can no longer
resolve.

## Possession contract

`SemanticCacheAck` is a transport-neutral statement containing one canonical
semantic hash. It validates only that the value is a SCIP content address.

It is **not** proof that a particular peer sent the acknowledgement and it is not
a replay-prevention mechanism. The surrounding authenticated session must decide
whether an acknowledgement is attributable, fresh, and authorized before calling
`PeerSemanticInventory::record_ack` or `apply_cache_feedback`.

`PeerSemanticInventory` is bounded session-local state. It records canonical
semantic hashes that the authenticated peer has acknowledged possessing.

Default maximum inventory size: **4,096 hashes**.

Properties:

- malformed hashes are rejected;
- duplicate acknowledgements are idempotent;
- the inventory fails closed when its configured capacity is reached;
- individual hashes may be revoked;
- the whole inventory may be cleared on session reset or trust loss;
- the inventory is not serializable as authoritative peer state.

The inventory is runtime state, not a globally authoritative database and not an
epistemic claim about the graph itself.

## Safe transfer-input construction

`build_grounded_transfer_input` converts:

- the negotiated SCIP session;
- authenticated peer semantic inventory;
- the exact target graph;
- an optional exact `GraphDelta`;
- optional text/projection candidates;

into the existing low-level `TransferPlanningInput`.

The builder exposes candidates only when their prerequisites are satisfied:

| Candidate | Required state |
|---|---|
| full grounded graph | shared exact grounded representation |
| semantic reference | negotiated semantic references + target hash acknowledged |
| exact graph delta | negotiated exact graph deltas + exact base hash acknowledged + delta target matches transfer target |
| human-text fallback | `HumanText` is shared |
| HDC projection attachment | matching HDC profile is negotiated and `Hdc` is shared |

The builder computes grounded-graph, reference, and graph-delta byte sizes itself;
it does not accept caller-provided sizes for those exact semantic forms.

If the target graph itself is acknowledged, the downstream transfer planner may
select a semantic reference. If only the required base graph is acknowledged,
the planner may select an exact graph delta when it is smaller than the full
graph. With neither acknowledgement, the exact grounded graph remains the safe
baseline.

Acknowledgement controls **eligibility**, not **selection**. An acknowledged
`GraphDelta` may still be larger than the full canonical graph. `plan_transfer`
retains the authority to choose the smallest legal exact transfer for each
message.

## Authenticated cache feedback

SCIP v1 defines three session-local cache-feedback forms:

| Feedback | Meaning | Inventory effect |
|---|---|---|
| `Ack` | peer claims exact possession of one semantic hash | add hash |
| `Miss` | an attempted cached prerequisite was unavailable | remove hash |
| `Revoke` | peer proactively retracts an earlier possession claim | remove hash |

`SemanticCacheMiss` also records which prerequisite failed:

- `semantic_reference_target`: the referenced target graph was unavailable;
- `graph_delta_base`: the exact base graph required by a `GraphDelta` was unavailable.

A miss or revocation is idempotent. Repeating already-obsolete feedback does not
produce an error or mutate unrelated inventory entries.

`SemanticCacheFeedback` uses an explicitly tagged serialized shape and exposes
deterministic canonical JSON bytes for transcript binding and measurement.

As with acknowledgements, a bare miss or revocation does not authenticate its
sender. The surrounding session must authenticate, sequence/freshness-check,
transcript-bind, and authorize feedback **before** `apply_cache_feedback` mutates
`PeerSemanticInventory`.

## Recovery algorithm

There is intentionally no second recovery planner.

When a cached transfer prerequisite fails:

1. the receiver emits the appropriate authenticated `Miss`, or proactively sends
   `Revoke` when it knows content was evicted;
2. the sender's transport/session layer verifies identity, session binding,
   freshness/order, replay policy, and authorization;
3. `apply_cache_feedback` removes only the named semantic hash from
   `PeerSemanticInventory`;
4. the sender calls `build_grounded_transfer_input` again using the updated
   inventory;
5. the existing `plan_transfer` chooses the smallest remaining legal transfer;
6. the receiver performs the ordinary content-addressed verification.

This produces a natural fallback chain without duplicating semantic logic:

- missing reference target -> reference candidate disappears; a still-valid
  graph delta may remain eligible;
- missing graph-delta base -> delta candidate disappears;
- with neither prerequisite available -> full canonical grounded graph remains
  the exact baseline.

If the receiver later caches the graph again, a fresh authenticated `Ack` restores
eligibility.

## Trust boundary

A correct deployment sequence is:

1. an authenticated transport/session receives peer cache feedback;
2. transport policy verifies peer identity, freshness/replay rules, ordering,
   session binding, and authorization;
3. the accepted feedback is applied to `PeerSemanticInventory`;
4. SCIP builds transfer candidates from that inventory;
5. the ordinary transfer planner chooses the smallest permitted representation.

For Symthaea/Mycelix, Xenia is the intended layer for steps 1–2. A future Xenia
binding should bind at least the semantic hash, feedback kind, peer/session
identity, SCIP protocol version, and transcript position.

SCIP itself deliberately does not contain cryptographic identity, replay state,
transport ordering, or authorization policy.

## Failure behavior

The session helper fails closed when:

- no exact grounded representation is shared;
- an acknowledgement, miss, or revocation hash is malformed;
- the inventory is over capacity;
- a supplied delta declares a target hash different from the transfer target;
- a delta base has not been acknowledged;
- a representation candidate was not negotiated.

A missing acknowledgement never means “probably cached.” It removes the
reference/delta candidate and leaves the full grounded representation available.

A stale acknowledgement is therefore a liveness/performance issue, not a
semantic-integrity bypass. After authenticated miss feedback retracts the stale
claim, the next plan falls back to a representation whose prerequisites are
actually known.

## Bandwidth-amplification consideration

An authenticated but uncooperative peer can repeatedly miss or revoke cached
state and force larger exact transfers. Cache feedback therefore should not be
interpreted as a promise that recovery is cheap.

The Xenia/session binding should apply ordinary abuse controls such as rate
limits, recovery budgets, and repeated-miss telemetry. Those controls must not
change SCIP's semantic rule: when cache possession is uncertain, correctness
wins and the cached candidate is withdrawn.

## Non-claims

This contract does not claim that:

- bare cache feedback authenticates its sender;
- acknowledged content remains cached forever;
- graph deltas are always smaller than full graphs;
- cache possession establishes truth, provenance, or authorization to use the
  referenced knowledge;
- a cache miss proves malicious behavior or receiver fault;
- the inventory should persist across unrelated authenticated sessions.

Those concerns belong to transport/session policy and the canonical semantic
objects themselves.

## Next integration boundary

With possession, revocation, and miss recovery explicit, SCIP's semantic session
contract is sufficient for the first authenticated Xenia binding without
inventing hidden cache assumptions.

The next non-transport step remains Phase B language integration: a thin adapter
around `LLMOrgan` / `LLMBackend` that accepts a SCIP envelope and uses the existing
grounded text fallback for today's token-based providers, while keeping
`GroundedConceptGraph` canonical.
