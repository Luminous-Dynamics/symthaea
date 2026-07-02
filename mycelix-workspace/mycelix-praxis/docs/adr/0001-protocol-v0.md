# ADR-0001: FL Protocol v0 - Message Types and Phases

**Status**: Accepted

**Date**: 2025-11-15

**Deciders**: Luminous Dynamics core team

**Tags**: protocol, federated-learning, architecture

---

## Context

We need to define the initial federated learning protocol for Praxis. This protocol must:

- **Enable collaborative learning** without sharing raw training data
- **Resist poisoning attacks** from malicious participants
- **Provide provenance** for models and credentials
- **Scale** to hundreds of participants per round
- **Integrate** with Holochain's agent-centric architecture

### Constraints

- Holochain does not provide leader election or global consensus (agent-centric, not blockchain)
- Gradients can be large (MBs per update) and should not be stored on DHT permanently
- Participants may have heterogeneous data, compute, and network resources
- No central authority for validation (coordinator is semi-trusted)

### Requirements

Must-have:
- Phased protocol (discovery, join, update, aggregate, release)
- Robust aggregation (trimmed mean or median)
- Gradient clipping for privacy
- Commitment-reveal scheme for gradients
- On-chain provenance for models

Nice-to-have:
- Differential privacy support
- Multi-coordinator verification
- Asynchronous updates

---

## Decision

**We adopt a six-phase synchronous FL protocol with commitment-reveal and robust aggregation.**

### Phases

1. **DISCOVER**: Coordinator announces round (DHT entry)
2. **JOIN**: Participants signal intent (links to round)
3. **ASSIGN**: Coordinator selects participants, shares base model
4. **UPDATE**: Participants train locally, submit clipped gradient commitments (DHT)
5. **AGGREGATE**: Coordinator requests gradients (P2P), applies trimmed mean/median
6. **RELEASE**: New model published with provenance entry

### Key Design Choices

1. **Coordinator role**: Semi-trusted agent elected by DAO (future: rotation)
2. **Gradient storage**: Commitments on DHT, actual gradients P2P (not persistent)
3. **Aggregation**: Default to trimmed mean (10% trim); median as robust fallback
4. **Clipping**: Mandatory L2 norm clipping (default threshold: 1.0)
5. **Provenance**: `ModelProvenance` entry links model hash to round, participants, aggregation method

---

## Consequences

### Positive

- ✅ **Simplicity**: Six phases are easy to implement and reason about
- ✅ **Privacy**: Gradients not published to DHT (only commitments)
- ✅ **Robustness**: Trimmed mean tolerates up to 10% malicious participants
- ✅ **Provenance**: Every model traceable to training round
- ✅ **Flexibility**: Aggregation method configurable per round

### Negative

- ⚠️ **Coordinator trust**: Coordinator can censor or bias aggregation (mitigated by logging)
- ⚠️ **Synchronous**: All participants must complete within round deadline (no straggler tolerance)
- ⚠️ **Scalability**: Coordinator must fetch gradients from N participants (bandwidth bottleneck)

### Neutral

- ℹ️ **P2P gradient transfer**: Requires reliable P2P messaging (Holochain provides this)
- ℹ️ **Clipping threshold**: Must be tuned per model/dataset (default 1.0 may not fit all)

---

## Alternatives Considered

### Alternative 1: Asynchronous FL

**Description**: Participants update at their own pace; coordinator aggregates whenever K updates received

**Pros**:
- No straggler problem
- More flexible for heterogeneous devices

**Cons**:
- Complex to reason about (which model version is base?)
- Harder to provide round-level provenance
- Research-stage (less battle-tested)

**Why not chosen**: Too complex for v0; revisit in v1.0

---

### Alternative 2: Blockchain-based Coordination

**Description**: Use a smart contract (e.g., Ethereum, Cosmos) to coordinate FL rounds

**Pros**:
- Decentralized coordination (no trusted coordinator)
- Transparent, auditable

**Cons**:
- Requires bridge to Holochain (additional complexity)
- Gas costs for on-chain operations
- Not agent-centric (counter to Holochain philosophy)

**Why not chosen**: Adds dependency; Holochain's DHT sufficient for now

---

### Alternative 3: Secure Aggregation (MPC)

**Description**: Use multi-party computation to aggregate gradients without coordinator seeing individuals

**Pros**:
- Maximum privacy (coordinator can't infer data)
- Provably secure under crypto assumptions

**Cons**:
- High computational overhead (10-100x slower)
- Complex protocol (requires 3+ parties, threshold crypto)
- Research-stage implementations (not production-ready)

**Why not chosen**: Overkill for v0; roadmap for v2.0+ if strong demand

---

## Implementation Notes

### Affected Components

- `fl_zome`: Core protocol implementation
- `praxis-agg`: Aggregation algorithms (trimmed mean, median)
- `praxis-core`: `ModelProvenance` type

### Migration Path

N/A (initial protocol)

### Testing Strategy

1. **Unit tests**: Each phase transition in isolation
2. **Integration tests**: Full round with mock participants (3-10 agents)
3. **Adversarial tests**: Inject malicious gradients, verify trimming works
4. **Performance tests**: Measure latency/bandwidth with 50-100 participants

### Rollout Plan

- Alpha release (v0.1): Single coordinator, manual round creation
- Beta release (v0.2): DAO can create rounds, automated scheduling
- v1.0: Multi-coordinator verification, asynchronous option

---

## References

- [FL Protocol Specification](../protocol.md)
- [Threat Model](../threat-model.md)
- McMahan et al. (2017): *Communication-Efficient Learning of Deep Networks from Decentralized Data*
- Yin et al. (2018): *Byzantine-Robust Distributed Learning*

---

## Revision History

| Date | Changes | Author |
|------|---------|--------|
| 2025-11-15 | Initial draft | Luminous Dynamics |
