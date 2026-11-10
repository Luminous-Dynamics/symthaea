# IG-COMM-TOPOLOGY-001 — deterministic CommunicationGraph fixtures

**Status:** synthetic graph profiles and fixture descriptors; no agent interaction executed  
**Owner:** [Symthaea #11337](https://github.com/Luminous-Dynamics/symthaea/issues/11337)

## Typed graph boundary

A communication graph is not a delegation graph.

- `AgentInstanceId` identifies a communicating process.
- `PrincipalId` is optional metadata; multiple AgentInstanceIds bound to one principal remain distinct nodes.
- A directed `MessageChannel` is a potential sender-to-recipient route. The reverse route is a separate edge.
- Reachability is not delivery. A delivered message must bind to a separate channel event and payload digest.
- A message edge transfers no voting weight, delegated mandate, legitimacy or action authority.

[Symthaea #11248](https://github.com/Luminous-Dynamics/symthaea/issues/11248) remains the owner of delegation-network semantics.

## Frozen topology profiles

The v0.1 manifest defines eight deterministic, eight-node profiles:

- chain;
- reciprocal star;
- balanced tree;
- complete directed control;
- sparse-random reciprocal network;
- stochastic-block communities with an explicit bridge;
- core-periphery;
- temporal churn with explicit revoke/restore/exit/rejoin events.

Each profile commits to node identities, principal metadata, directed channel IDs, message classes, timing/capacity, event order, RNG identity and seed. The graph digest covers the canonical profile payload. The centrality reference is separately recomputed from the realized graph.

The five centrality-relevant profiles used by #11320 now use the exact paired seeds:

| Profile | Seed |
|---|---:|
| chain | 1016 |
| star | 1017 |
| balanced_tree | 1018 |
| stochastic_block | 1019 |
| core_periphery | 1020 |

A complete graph and the checked-in sparse-random sample have tied centrality across all nodes; they are retained as unsupported min/max placement controls, not forced into an arbitrary low/high distinction.

## Centrality semantics

`normalized_betweenness_v1` uses:

- directed, unweighted simple graph;
- shortest-path distance measured in hops;
- strongly connected snapshot with at least three nodes;
- normalization by `(n-1)(n-2)`;
- full minimum/maximum tie sets, with lexical AgentInstanceId tie-break.

The Rust qualifier recomputes Brandes betweenness and checks every stored value/tie set. A graph without distinct extrema remains Unsupported.

## Thirty fixture descriptors

The manifest has 30 descriptors for:

- eight golden profiles;
- five centrality known-answer profiles;
- temporal churn ordering and explicit restoration;
- complete-graph tie control;
- delegation/message graph non-equivalence;
- delivery-event binding;
- principal multiplicity;
- self-loop, duplicate-channel, unknown-endpoint and duplicate-agent rejection;
- out-of-order event, stale channel replay and implicit restore rejection;
- explicit reciprocity and seed/identity controls;
- disconnected snapshot and deterministic tie-break behavior.

These are synthetic structural fixtures, not behavior runs. The Rust unit tests also mutate graph profiles directly to exercise selected negative controls.

## Run the qualifier

```bash
cargo fmt --manifest-path research/ig-comm-topology-001/qualifier/Cargo.toml --check
cargo test --manifest-path research/ig-comm-topology-001/qualifier/Cargo.toml
cargo run --manifest-path research/ig-comm-topology-001/qualifier/Cargo.toml -- \
  research/ig-comm-topology-001/COMMUNICATION_GRAPH_FIXTURES_V0_1.json \
  research/ig-comm-topology-001/COMMUNICATION_GRAPH_FIXTURES_V0_1.schema.json
```

The qualifier checks the bounded profile contract, graph commitments, endpoint/channel uniqueness, temporal event ordering, directed connectivity, centrality scores and tie sets, and fixture/profile references. It is not a full JSON Schema implementation or message-delivery simulator.

## Claim ceiling

These profiles describe synthetic communication topologies and structural properties only. They do not represent observed human networks, establish real-world influence, or infer that a reachable message was delivered. No production Mycelix delegation or voting semantics are changed.
