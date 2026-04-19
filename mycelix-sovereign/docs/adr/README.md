# Architecture Decision Records

Substantive architectural decisions for Mycelix Sovereign are recorded here as ADRs.

## Process

- ADRs are numbered sequentially (`0001-`, `0002-`, ...).
- Each ADR has a one-line title and a short body structured as **Context / Decision / Consequences / Alternatives**.
- ADRs are immutable once committed. If a decision is revisited, a new ADR is written that explicitly **Supersedes** the old one — the old one remains in the tree with a `Superseded by ADR-NNNN` header added.
- Component-specific decisions (internal to a single crate) may live in that crate's own `docs/adr/` directory. ADRs here cover **Suite-wide** decisions.

## Index

| ADR | Title | Status |
|---|---|---|
| [0000](0000-adr-process.md) | ADR process | Accepted |
| [0001](0001-screen-capture-backend.md) | Cross-platform screen capture backend | Accepted |
