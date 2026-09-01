# Bounded Agency Authority v0.1

## Purpose

This document freezes the first implementation boundary for shared Symthaea agency authority.
The authority layer is deliberately smaller than cognition: it does not learn, infer intent,
run tools, hold signing keys, or decide whether a proposed action is useful. It answers only
whether an already-described semantic operation is inside explicitly delegated authority.

## v0.1 invariants

1. **Confidence is not authority.** Phi, model confidence, utility, urgency, or predicted benefit never create permission.
2. **Delegation only attenuates.** A child grant may narrow resources, operations, audience, task, plan/world bindings, expiry, uses, delegation depth, and risk budget; it may not broaden any of them.
3. **Negative authority dominates.** Applicable revocation, tombstone, freeze, or minimum-epoch facts override otherwise valid positive grants.
4. **Epochs never resurrect.** A grant from a different authority epoch is denied; restoring old state does not restore old authority.
5. **Reserved uses are charged.** In-flight durable reservations count against use limits before dispatch so a crash cannot silently recreate authority.
6. **Exact resources in v0.1.** Resource matching is exact. Hierarchical or relationship-based scope must be introduced by an explicit future schema, never inferred from unsafe string-prefix matching.
7. **Integer security budgets.** Signed/committed budgets use integers; security semantics do not depend on floating-point canonicalization.
8. **Canonical commitments are domain separated.** Every security-relevant field of `CapabilityGrant` contributes to its deterministic BLAKE3 commitment.
9. **Unknown schemas fail closed.** Enforcement points reject capability schemas they do not understand.
10. **Authority and admission remain separate.** A valid grant is necessary but not sufficient for consequential execution; later admission layers may require fresh world state, safety evidence, information-flow checks, sandbox obligations, quorum, or requalification.

## Deliberate non-goals

v0.1 does not define signing, transport, persistence, policy languages, OAuth/MCP mappings,
OS sandbox compilation, trusted time, hierarchical resource scopes, multi-party quorum,
capability escrow, or cross-agent budget partitioning. Those belong in later layers and must
preserve the invariants above.

## Migration direction

The first intended consumers are:

- `symthaea-browser`, replacing domain-local authority checks while preserving browser-specific action semantics;
- Nixward, making capability authorization precede Phi-based caution and confirmation;
- Symthaea IPC, replacing coarse ambient `Execute`/`Admin` trust with explicit task/action grants;
- Xenia, signing and transporting authority commitments without moving policy into the wire layer;
- `symthaea-swarm`, eventually replacing duplicated lease/epoch primitives only after behavior-equivalence tests.

No domain-specific implementation should be deleted merely because this crate exists. Migrations
must run shared and legacy semantics against existing fixtures until equivalence and security
gates are demonstrated.
