# Architecture Decision Records (ADR)

**Recording significant architectural decisions for Mycelix Protocol**

---

## What is an ADR?

An **Architecture Decision Record** (ADR) captures a significant architectural decision, its context, and its consequences. ADRs help us:

- **Understand why** decisions were made
- **Track evolution** of our thinking over time
- **Onboard new contributors** by explaining the "why" behind the "what"
- **Avoid revisiting** settled decisions
- **Learn from** past successes and failures

---

## ADR Index

### Active Decisions

| ADR | Title | Status | Date | Impact |
|-----|-------|--------|------|--------|
| [001](./001-multi-backend-architecture.md) | Multi-Backend Architecture (PostgreSQL + Holochain + Chains) | ✅ Accepted | Nov 2025 | High |
| [002](./002-epistemic-cube-3d-model.md) | 3D Epistemic Cube (E/N/M Axes) | ✅ Accepted | Nov 2025 | High |
| [003](./003-matl-composite-trust-scoring.md) | MATL Composite Trust Scoring (PoGQ + TCDM + Entropy) | ✅ Accepted | Nov 2025 | High |
| [004](./004-holochain-postgresql-hybrid.md) | Holochain + PostgreSQL Hybrid Approach | ✅ Accepted | Oct 2025 | Medium |
| [005](./005-45-percent-byzantine-tolerance.md) | Breaking the 33% BFT Barrier via Reputation-Weighting | ✅ Accepted | Nov 2025 | High |

### Superseded Decisions

| ADR | Title | Status | Superseded By | Date |
|-----|-------|--------|---------------|------|
| (none yet) | - | - | - | - |

---

## Creating a New ADR

### 1. Use the Template
Copy the [ADR template](./template.md) to create a new ADR:

```bash
cp docs/adr/template.md docs/adr/NNN-your-decision-title.md
```

### 2. Fill in the Sections
- **Title**: Short, descriptive title (< 50 chars)
- **Status**: Proposed | Accepted | Deprecated | Superseded
- **Context**: What problem are we solving? Why now?
- **Decision**: What did we decide to do?
- **Consequences**: What are the positive and negative outcomes?
- **Alternatives Considered**: What else did we evaluate?

### 3. Number Sequentially
ADRs are numbered sequentially (001, 002, 003...). Check the index above for the next number.

### 4. Submit for Review
1. Create a new ADR file
2. Submit a PR with the ADR
3. Discuss in PR comments
4. Update status to "Accepted" when merged

---

## ADR Lifecycle

```
┌─────────────┐
│  Proposed   │  ← Initial draft
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Accepted   │  ← Merged and active
└──────┬──────┘
       │
       ├─────────────┐
       ▼             ▼
┌─────────────┐  ┌──────────────┐
│ Deprecated  │  │  Superseded  │
│ (no longer  │  │  (replaced   │
│  followed)  │  │   by ADR-X)  │
└─────────────┘  └──────────────┘
```

---

## ADR Statuses

### 🟢 Proposed
Decision is drafted and under discussion. Not yet binding.

### ✅ Accepted
Decision is approved and should be followed. This is the "active" state.

### ⚠️ Deprecated
Decision is no longer recommended but not explicitly replaced. Use with caution.

### 🔄 Superseded
Decision is replaced by a newer ADR. See "Superseded By" field.

---

## Best Practices

### Keep ADRs Immutable
Once an ADR is accepted, **do not edit it** (except for minor typos/formatting). If a decision changes:
- Create a new ADR that supersedes the old one
- Update the old ADR's status to "Superseded"
- Link the old and new ADRs bidirectionally

### Be Concise
ADRs should be readable in < 10 minutes. If your ADR is getting long, consider:
- Breaking it into multiple ADRs
- Moving detailed analysis to a separate document

### Focus on "Why"
The code shows "what" we built. The ADR explains "why" we built it that way.

### Include Consequences
Be honest about trade-offs. Every decision has costs. Document them.

### Link to Related Documents
- Reference architectural specs
- Link to research papers
- Connect to implementation PRs

---

## Examples from Other Projects

- [Nix ADRs](https://github.com/NixOS/nix/tree/master/doc/adr)
- [Rust RFC Process](https://github.com/rust-lang/rfcs)
- [Ethereum EIPs](https://eips.ethereum.org/)

---

## Related Documentation

- [Architecture Index](../03-architecture/README.md)
- [Roadmap v5.3 → v6.0](../architecture/Mycelix_Roadmap_v5.3_to_v6.0.md)
- [Version History](../VERSION_HISTORY.md)

---

📍 **Navigation**: [← Architecture](../03-architecture/README.md) | [↑ Docs Home](../README.md) | [Template →](./template.md)

**Last Updated**: November 10, 2025
**ADR Count**: 5 active, 0 superseded
