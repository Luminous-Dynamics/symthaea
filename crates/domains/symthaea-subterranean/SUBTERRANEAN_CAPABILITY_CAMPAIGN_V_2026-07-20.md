# Subterranean Capability Campaign V

**Date:** 2026-07-20  
**Theme:** Multi-agent underground operational truth  
**Baseline:** Capability Campaign IV / patch 28  
**Scope:** Patches 29-38 plus protocol documentation

## Goal

Move `symthaea-subterranean` from a capable single-platform reference system to a bounded multi-agent substrate without claiming networking, authentication, geometric precision, or rescue authority that the crate does not possess.

The campaign treats team information as externally transported evidence. The crate owns deterministic ordering, replay rejection, freshness, conservative fusion, right-of-way, rescue consent, safety arbitration, and evidence. Cryptographic authentication and packet transport remain explicit external responsibilities.

## Patch sequence

### 29. Bounded peer directory

- Stable `AgentId`, team role, condition, and heartbeat types.
- Epoch and sequence ordering.
- Replay, self-message, malformed-message, and capacity rejection.
- Fresh/stale peer status and distress visibility.
- No cryptographic-authenticity claim.

### 30. Provenance-preserving shared tunnel map

- Latest observation retained per `(depth bin, source)`.
- Equal-version conflicting payloads rejected as equivocation.
- Conservative aggregation across peers:
  - minimum roof stability,
  - maximum water and slurry,
  - minimum localization and survey confidence.
- Order-independent merge tests.

### 31. Tunnel occupancy and right-of-way

- Bounded depth-interval reservations.
- Outbound, inbound, and holding directions.
- Routine, return, rescue, and emergency priorities.
- Stable agent-id tie breaking.
- Explicit yield assessment for opposed use of a narrow tunnel.

### 32. Relay mesh truth

- Explicit surface, agent, and relay nodes.
- Ordered, freshness-bounded link updates.
- Widest-path assessment maximizes the weakest link.
- Stale links are excluded from reachability.

### 33. Explicit rescue handoff

- Bounded rescue request and capability declaration.
- Feasibility includes travel cost, peer reachability, and the rescuer's own return budget.
- Offer and requester acceptance required before mission authority changes.
- Requested, offered, accepted, active, completed, and aborted states.

### 34. Team operations composition

- One coordinator owns peer directory, shared map, reservations, mesh, and rescue handoff.
- Produces only four directives: none, yield, maintain relay, assist peer.
- A distress heartbeat remains visible but cannot become `AssistPeer` by itself.

### 35. Causal runtime integration

- Adds `YieldTunnel`, `MaintainRelay`, and `AssistPeer` mission symbols.
- Adds `TunnelConflict` to the physical hazard portfolio.
- Collision conflict is latched through the existing safety supervisor.
- Verified planner emits `TunnelYield` and stops cutter, auger, and tracks.
- Physical hazards continue to override team directives.

### 36. Team evidence

Each retained command-level record now includes:

- peer counts and distress count,
- team directive,
- conflicting agent and severity,
- right-of-way result,
- surface mesh reachability, bottleneck, and hop count,
- shared-route coverage and obstruction risk,
- rescue handoff state.

Summary metrics count tunnel conflicts, mesh partitions, peer distress, and accepted/active rescue frames.

### 37. Multi-agent acceptance contracts

Deterministic gates verify:

1. shared-map convergence,
2. command-level collision arrest,
3. stale-mesh isolation,
4. rescue return-reserve protection,
5. distress does not bypass explicit acceptance.

### 38. Conservative peer route fusion

Shared route evidence may only make return feasibility more conservative. It may lower confidence, raise obstruction and energy cost, or make a route infeasible. It cannot improve local battery margin or clear a local failure.

## Validation performed in this environment

Using API-compatible local stand-ins for the missing `symthaea-core`, `symthaea-fep`, and serialization workspace crates:

- `git diff --check`: passed.
- Rust 1.85 rustfmt parse/check over every Rust source: passed.
- `cargo check --all-targets` with `-D warnings`: passed.
- Unit tests: **129 passed, 0 failed, 1 ignored**.
- The ignored test remains the controlled-hardware 200 Hz wall-clock benchmark.

The stand-in build validates Rust types, exhaustive matches, deterministic logic, and internal tests. The authoritative build still requires the complete Rust 1.94 workspace and real HDC/FEP/Serde implementations.

## Deliberate limitations

- No transport implementation.
- No signatures, MACs, certificates, or authenticated identity binding.
- No Byzantine-consensus claim.
- Occupancy is one-dimensional because the reference plant exposes depth rather than a 3-D tunnel pose.
- Shared map evidence is advisory and conservative; it never overrides local safety with a more optimistic claim.
- Rescue requires acceptance but does not yet model payload transfer, towing, docking, or human casualty handling.
- Relay deployment updates the plant's local resource state; network-topology updates still arrive through the integration API.

## Merge gates in the full workspace

```bash
cargo fmt --check -p symthaea-subterranean
cargo clippy -p symthaea-subterranean --all-targets -- -D warnings
cargo test -p symthaea-subterranean
cargo test -p symthaea-subterranean team_validation::tests::reference_team_contracts_pass
cargo test -p symthaea-subterranean runtime_budget::tests::reference_200_hz_control_loop_budget -- --ignored --nocapture
```

## Highest-value next campaign

1. Bind peer messages to Xenia/Mycelix authenticated identities and transcript epochs.
2. Replace depth intervals with a branching tunnel graph and metric pose covariance.
3. Add packet loss, delay, partition, duplication, reordering, and compromised-peer campaigns.
4. Model rescue payload transfer, towing energy, docking geometry, and abandonment criteria.
5. Add a deterministic multi-platform simulator to validate deadlock freedom and convoy throughput.
