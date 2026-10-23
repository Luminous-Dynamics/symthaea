# SPORE-FED-001A — Interplanetary Sovereign-Node Federation Showcase

**Status:** specification / implementation seed  
**Issue:** #6309  
**Claim ceiling:** reproducible software/networking semantics and illustrative systems behavior only

## Purpose

Turn the existing Spore + swarm infrastructure into a single, replayable story:

**germination → sovereign node → local federation → capability transfer → partition → interplanetary delay → healing → replay**

The demo is intentionally designed around one memorable invariant:

> **Capabilities can propagate without identity or authority propagating.**

This is the bridge between the biological metaphor of a Spore and the concrete systems architecture. The demo should feel alive while remaining explicit about what is simulation, observation, analysis, recommendation, authorization, and execution.

## Existing substrate

Prefer the existing implementation surface:

- crates/domains/symthaea-spore — Spore kernel and spore-mesh-daemon;
- crates/domains/symthaea-swarm — authenticated gossip/direct transport;
- src/swarm — higher-level bootstrap and peer state;
- Iroh endpoint/bootstrap identities;
- existing epistemic/provenance structures;
- existing browser/WebAssembly Spore portal.

Do not create a parallel networking protocol solely for the showcase.

## Demo state model

Each simulated node has an independent state:

| Field | Meaning | Transfer rule |
|---|---|---|
| node_id | stable node identity | **never transferable** |
| identity_generation | identity/key generation | changes only through explicit identity lifecycle |
| capabilities | locally available capabilities | may be learned/bootstrapped |
| capability_generation | generation of capability set | increments on capability change |
| local_state | local Spore/holon state | remains local unless explicitly published |
| peer_views | observations of peers | staleable, never authoritative by itself |
| analysis_artifacts | derived Symthaea outputs | advisory; never authorization |
| authority_state | local authorization state | **never propagated by capability gossip** |
| provenance | event/source lineage | travels with claims, not as authority |
| connectivity | peer/link state | affected by partition/delay |
| clock | local logical time | not assumed globally synchronized |

## Event envelope

The showcase event log should use a deterministic envelope with these semantic fields:

    EventV1 {
      sequence
      logical_time
      actor_node_id
      event_type
      subject_id
      source_generation
      payload
      provenance
      delivery
    }

### Event types

- NodeGerminated
- CapabilityDeclared
- PeerDiscovered
- PeerAuthenticated
- CapabilityShared
- CapabilityAccepted
- AnalysisProduced
- RecommendationProduced
- AuthorizationIssued
- ExecutionRecorded
- LinkDelayed
- PartitionOpened
- PartitionHealed
- MessageRejected
- CapabilityGenerationChanged
- ReplayStarted
- ReplayCompleted

The event vocabulary deliberately separates analysis, recommendation, authorization, and execution.

## State machine

    DORMANT
       │ Germinate
       ▼
    SPORE
       │ Node identity
       ▼
    SOVEREIGN NODE
       │ discover / publish capability
       ▼
    FEDERATING
       │ capability accepted
       ▼
    COOPERATING
       │ partition or interplanetary delay
       ▼
    DEGRADED
       │ heal
       ▼
    RECONCILING
       │ stale/replay checks + merge
       ▼
    COOPERATING

No transition may silently grant authority.

## The eight showcase assertions

### A1 — Germination is local

A new Spore begins with its own node identity and local state.

**Visible:** one node appears; its identity is unique.

### A2 — Federation is not identity collapse

When A authenticates B, B remains B.

**Adversarial check:** equal capabilities or equal model state must not cause node_id equality.

### A3 — Capability propagation is explicit

A receives capability information from B only through a named capability event.

**Adversarial check:** copying a capability set must not copy the source node's identity or authority state.

### A4 — Authority does not propagate

An advisory result may cross the network; an authorization cannot be minted merely because the result arrived.

**Visible:** the UI uses separate lanes for ANALYSIS, RECOMMENDATION, AUTHORIZATION, and EXECUTION.

### A5 — Partition does not imply global failure

During a partition, nodes continue local computation and queue bounded outbound state.

**Visible:** the interplanetary link becomes delayed while local node activity continues.

### A6 — Delay is first-class

The remote node receives events according to the configured link model, not instantaneously.

**Visible:** a message has a queued/delivery timestamp and a logical route.

### A7 — Stale/replayed information is not silently current

A delayed or replayed event is checked against sequence/generation/expiry rules.

**Visible:** rejected events appear in a refusal ledger rather than disappearing.

### A8 — Replay is deterministic

A frozen event log plus exact configuration reconstructs the same semantic dispositions.

**Visible:** RUN #N and REPLAY #N converge to the same event/disposition summary.

## Scenario timeline

A three-minute public run should follow this sequence.

### 0:00–0:25 — Spore

Start one edge/browser Spore.

Show:

    NODE A
    status: germinating
    identity: A
    capabilities: [core, local-analysis]
    authority: local-only

### 0:25–0:55 — Federation

Nodes B and C appear and authenticate.

Animate authenticated links rather than drawing a static network.

### 0:55–1:20 — Capability bootstrapping

Node A shares a capability with B.

B gains the capability generation.

B **does not** gain A's identity or authority.

This is the first "aha" moment.

### 1:20–1:50 — Symthaea analysis

A node produces an analysis artifact and recommendation.

The UI explicitly shows:

    OBSERVED → ANALYSIS → RECOMMENDATION → [authority boundary] → AUTHORIZATION → EXECUTION

Only the appropriate external authority lane can authorize execution.

### 1:50–2:20 — Earth/Mars partition

Introduce a large logical delay and intermittent delivery.

Earth nodes continue operating.

Mars queues messages.

The map shows latency rather than pretending the worlds share one synchronous state.

### 2:20–2:45 — Healing

The link returns.

Queued events drain.

Stale/replayed events are rejected visibly.

Fresh events reconcile.

### 2:45–3:00 — Replay

Freeze the run.

Replay from the immutable event ledger.

Show:

    original:  events=...
    replay:    events=...
    semantic dispositions: IDENTICAL
    rejected stale/replay events: IDENTICAL
    authority transitions: IDENTICAL

## Visual language

The dashboard should have five persistent zones:

1. **Federation map** — nodes, links, worlds, delay.
2. **Node inspector** — identity, capabilities, generations, local state.
3. **Event stream** — provenance-bearing event sequence.
4. **Epistemic lane** — observed → derived → recommended → authorized → executed.
5. **Replay/partition controls** — germinate, add node, share capability, partition, delay, heal, replay.

Use motion to communicate state changes, not decorative animation.

### Suggested status vocabulary

- LOCAL
- AUTHENTICATED
- CAPABILITY AVAILABLE
- DELAYED
- PARTITIONED
- STALE
- REJECTED
- RECONCILING
- REPLAY IDENTICAL

Avoid labels such as "consciousness proven", "sentient network", or "autonomous governance".

## Deterministic configuration

The first witness should be completely deterministic:

    seed = fixed
    node IDs = A/B/C/MARS
    event sequence = fixed
    capability generations = fixed
    partition interval = fixed
    interplanetary delay = fixed
    replay seed = original seed

Randomized variants can be added later, but the public baseline must always be replayable.

## Failure/refusal ledger

Failures are part of the demo, not hidden implementation details.

At minimum display:

- capability copied with wrong source generation → rejected;
- replayed event → rejected;
- stale event after generation change → rejected;
- analysis artifact presented as authorization → rejected;
- unknown capability → unresolved;
- delayed event within validity window → queued;
- delayed event beyond validity window → rejected.

## Relation to Integral / Revolution Now

The demo should communicate architecture rather than ideology:

- sovereignty is represented as independently held identity/state;
- federation is represented as authenticated coordination without identity collapse;
- resilience is represented through partition tolerance and local operation;
- reciprocity is represented through explicit capability sharing;
- epistemic humility is represented through provenance and refusal states;
- Integral-style coordination is represented through the separation between analysis and normative authority.

These are architectural correspondences, not empirical claims about social outcomes.

## Implementation sequence

### SPORE-FED-001A — this document

Lock the scenario, event semantics, state machine, visual contract, and claim ceiling.

### SPORE-FED-001B

Implement the deterministic witness as a small semantic simulator over the existing transport concepts.

### SPORE-FED-001C

Connect the witness to a browser visualization.

### SPORE-FED-001D

Add interplanetary delay/partition profiles.

### SPORE-FED-001E

Add provenance export, refusal ledger, and replay qualification.

### SPORE-FED-001F

Package the three-minute public demo script.

### SPORE-FED-001H

Qualify cross-layer evidence seams: identity → claim → verification → policy → execution, with transport delivery kept independent from domain execution. See `crates/domains/symthaea-spore/docs/FEDERATION_EVIDENCE_SEAMS_V1.md`.

## Non-goals

This showcase does not establish:

- actual consciousness;
- real-world deployment;
- governance legitimacy;
- economic productivity;
- social outcomes;
- physical interplanetary communications capability.

It demonstrates a reproducible software architecture and its failure/refusal semantics.
