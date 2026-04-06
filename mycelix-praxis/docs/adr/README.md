# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for Mycelix Praxis.

## What is an ADR?

An ADR captures an important architectural decision along with its context and consequences.

## When to write an ADR?

Create an ADR when making decisions that:
- Affect the protocol or core architecture
- Have significant long-term impact
- Involve trade-offs between multiple approaches
- Need to be understood by future contributors

## ADR Format

Each ADR should include:

1. **Title**: Short, descriptive name
2. **Status**: Proposed, Accepted, Deprecated, Superseded
3. **Context**: What problem are we solving?
4. **Decision**: What did we decide?
5. **Consequences**: What are the implications?
6. **Alternatives Considered**: What other options did we evaluate?

## ADR Template

See [0000-template.md](0000-template.md) for the template.

## Naming Convention

```
NNNN-short-descriptive-title.md
```

Where `NNNN` is a sequential number (0001, 0002, etc.).

## List of ADRs

| # | Title | Status | Date |
|---|-------|--------|------|
| [0001](0001-protocol-v0.md) | FL Protocol v0: Message Types and Phases | Accepted | 2025-11-15 |

## Lifecycle

```
Proposed → Accepted → [Deprecated / Superseded]
           ↓
        Rejected
```

- **Proposed**: Under discussion
- **Accepted**: Decision made and implemented
- **Rejected**: Considered but not adopted
- **Deprecated**: No longer recommended (but not removed)
- **Superseded**: Replaced by a newer ADR
