# Institutional Evolution Independent Oracle Contract

**Status:** research contract v0
**Fixture corpus:** `institutional-evolution-fixtures-v0.json`

## Purpose

Provide a language-neutral conformance layer for the deterministic institutional-evolution kernel.

The oracle must not call the Rust implementation as its source of truth. It should consume the frozen fixture representation and independently evaluate the declared transition semantics.

## Required checks

### Identity

- institution identity remains distinct from profile identity;
- profile identity remains distinct from individual rule identity;
- parent profile identity is checked on every state-changing transition;
- a candidate rule cannot silently become the profile identity.

### Authority

- Operational changes require the current CollectiveChoice rule;
- CollectiveChoice changes require the current Constitutional rule;
- Constitutional changes require the current MetaConstitutional rule;
- authorizing rule hash must equal the current rule at the required level;
- claimed authorizing level must equal the required level.

### Transition separation

- Proposal does not mutate current state;
- Adoption does not mutate current state;
- Implementation requires successful Adoption;
- Rejection cannot be implemented;
- failed transitions leave state unchanged.

### Replay

- duplicate proposal identifiers reject;
- duplicate decisions reject;
- a proposal whose parent profile is no longer current rejects;
- an already implemented proposal rejects;
- the same valid event sequence produces the same current state and lineage.

### Constitutional baseline

- constitutional mutation is disabled in the deterministic baseline;
- meta-constitutional mutation is disabled in the deterministic baseline;
- enabling higher-order mutation is a separate experimental profile.

### Lineage

Every state-changing event must preserve:

- parent profile;
- candidate profile;
- candidate rule;
- mutation id;
- rule level;
- transition;
- authority identity;
- authorizing rule identity where required.

## Differential use

The intended qualification path is:

```text
frozen fixtures
    ├──> independent oracle
    └──> Rust kernel
             ↓
        compare observables
```

A green Rust test suite without matching independent-oracle observables is not sufficient for a qualification PASS.

## Claim ceiling

PASS establishes only conformance of the implementation to the frozen synthetic transition contract.

It does not establish that the institutional model is historically accurate, politically legitimate, socially desirable, or economically optimal.