# SCIP Peer Semantic Inventory v1

Status: **experimental / draft session contract**

Crate: `crates/bridges/symthaea-interlingua`

This note defines the session-local prerequisite that lets SCIP safely choose
cached semantic references and exact graph deltas.

## Problem

The low-level SCIP transfer planner accepts optional byte counts for:

- a semantic reference, when the receiver already owns the exact target graph;
- an exact `GraphDelta`, when the receiver already owns the exact base graph.

Those options are intentionally caller-supplied at the low-level planning API.
Without a higher-level possession contract, a caller could optimistically expose
a reference or delta candidate without actually knowing that the peer owns the
required content-addressed graph.

## Contract

`SemanticCacheAck` is a transport-neutral statement containing one canonical
semantic hash. It validates only that the value is a SCIP content address.

It is **not** proof that a particular peer sent the acknowledgement and it is not
a replay-prevention mechanism. The surrounding authenticated session must decide
whether an acknowledgement is attributable, fresh, and authorized before calling
`PeerSemanticInventory::record_ack`.

`PeerSemanticInventory` is bounded session-local state. It records canonical
semantic hashes that the authenticated peer has acknowledged possessing.

Default maximum inventory size: **4,096 hashes**.

Properties:

- malformed hashes are rejected;
- duplicate acknowledgements are idempotent;
- the inventory fails closed when its configured capacity is reached;
- individual hashes may be revoked;
- the whole inventory may be cleared on session reset, trust loss, or cache
  invalidation.

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

## Trust boundary

A correct deployment sequence is:

1. an authenticated transport/session receives a peer acknowledgement;
2. transport policy verifies peer identity, freshness/replay rules, session
   binding, and authorization;
3. the accepted `SemanticCacheAck` is recorded in `PeerSemanticInventory`;
4. SCIP builds transfer candidates from that inventory;
5. the ordinary transfer planner chooses the smallest permitted representation.

For Symthaea/Mycelix, Xenia is the intended layer for steps 1–2. A future Xenia
binding should bind at least the semantic hash, peer/session identity, SCIP
protocol version, and acknowledgement transcript position.

SCIP itself deliberately does not contain cryptographic identity, replay state,
or authorization policy.

## Failure behavior

The session helper fails closed when:

- no exact grounded representation is shared;
- an acknowledgement hash is malformed;
- the inventory is over capacity;
- a supplied delta declares a target hash different from the transfer target;
- a delta base has not been acknowledged;
- a representation candidate was not negotiated.

A missing acknowledgement never means “probably cached.” It removes the
reference/delta candidate and leaves the full grounded representation available.

## Non-claims

This contract does not claim that:

- a bare `SemanticCacheAck` authenticates its sender;
- acknowledged content remains cached forever;
- graph deltas are always smaller than full graphs;
- cache possession establishes truth, provenance, or authorization to use the
  referenced knowledge;
- the inventory should persist across unrelated authenticated sessions.

Those concerns belong to transport/session policy and the canonical semantic
objects themselves.

## Next integration boundary

Once this contract is validated, the next transport-focused step is to define an
authenticated Xenia acknowledgement frame/transcript binding. Alternatively,
SCIP can proceed to language integration because the semantic-reference and
exact-delta prerequisites are now explicit and testable rather than implicit
caller assumptions.
