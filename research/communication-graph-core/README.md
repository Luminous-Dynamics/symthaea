# CommunicationGraph core v0.1

**Status:** typed graph and event-validation prototype; no model simulation executed.
**Owner:** [Symthaea #11337](https://github.com/Luminous-Dynamics/symthaea/issues/11337).
**Consumers:** #11320 (stance-shift experiments), #11434 (debate-effect security).

## Domain separation

- `CommunicationGraphProfile` nodes are unique `AgentInstanceId` values.
- Optional `PrincipalId` is metadata. Multiple agents may bind to one principal without implicit collapse.
- A directed `MessageChannel` is a possible sender-to-recipient path with allowed message classes, activation interval, per-round attempt capacity and payload limit.
- `DelegationGraph` is not represented in this crate. A delegation edge never creates a message channel; a message channel never transfers mandate, vote weight or action authority.
- Graph reachability is not message delivery. Every attempted delivery is an ordered event with a message, sender, recipient, channel, payload digest, round and explicit outcome.

## Deterministic semantics

- Channel activation uses `active_from_round <= round < active_until_round`; an absent upper bound means no scheduled expiry.
- Every attempt consumes per-channel/per-round capacity, including rejected attempts, so retries cannot bypass rate limits.
- Event sequences begin at 1, are contiguous and have nondecreasing rounds.
- Outcome precedence is inactive channel, disallowed message class, payload limit, capacity exhaustion, then delivery.
- Graph canonical bytes sort nodes by AgentInstanceId and channels by channel ID; enum and struct field order are fixed by this schema. SHA-256 identifies bytes, not truth or authenticity.
- Event-log canonical bytes preserve event order. Missing or incomplete traces are invalid, not proof of no delivery.

## Directed centrality

`normalized_betweenness_v1` uses directed, unweighted, hop-count Brandes centrality on the simple directed graph obtained by collapsing multiple channels between the same ordered pair.

Preconditions: at least three nodes, strong connectivity, reciprocal arcs for this v0.1 matched-comparison profile, and no self-loops. Normalize by `(n - 1)(n - 2)`. Record full minimum/maximum tie sets; select the lexicographically smallest stable AgentInstanceId in each set. If the same node is selected for min and max (for example, a complete graph where all scores are zero), the comparison is Unsupported rather than manufactured.

Centrality is descriptive exposure-path metadata, not influence proof, truth, credibility, legitimacy or authority.

## Included contracts

- `COMMUNICATION_GRAPH_PROFILE_V0_1.schema.json`
- `COMMUNICATION_EVENT_LOG_V0_1.schema.json`
- `GOLDEN_STAR_PROFILE_V0_1.json`: four AgentInstanceIds, with two bound to the same PrincipalId, and a reciprocal star topology.
- `GOLDEN_STAR_EVENT_LOG_V0_1.json`: two delivered messages and one explicit capacity rejection.
- Standalone Rust library and CLI qualifier.
- Exact-triggering-revision GitHub Actions workflow.

## Run

From the repository root:

`cargo fmt --manifest-path research/communication-graph-core/Cargo.toml --check`

`cargo test --manifest-path research/communication-graph-core/Cargo.toml`

`cargo run --manifest-path research/communication-graph-core/Cargo.toml -- research/communication-graph-core/GOLDEN_STAR_PROFILE_V0_1.json research/communication-graph-core/GOLDEN_STAR_EVENT_LOG_V0_1.json`

## Claim ceiling

The crate validates static graph profiles, ordered delivery-event records, reachability, SHA-256 commitments and a bounded centrality metric. It does not generate social networks, perform communication transport, prove event authenticity, run agents, establish human network behavior, infer influence causality or authorize governance actions.