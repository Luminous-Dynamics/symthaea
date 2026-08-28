# AI Assurance Kernel v0.1

**Status:** Research foundation

**Scope:** Project-neutral authority and action-lifecycle semantics for Symthaea and other autonomous systems.

## Purpose

The AI Assurance Kernel separates *cognition* from *authority*. A model may propose arbitrary actions, but execution authority is represented by explicit, scoped capabilities and typed state transitions that the model cannot synthesize from ordinary data.

The kernel does **not** claim to prove that an AI is wise, aligned, conscious, truthful, or correct about the world. It targets narrower properties that can be enforced mechanically: who may act, on what, under which scope, through which lifecycle, with which evidence.

## Threat model

Assume the proposing model can be mistaken, adversarially prompted, overconfident, compromised, or simply buggy. It may attempt to:

- execute before authorization;
- reuse one-shot authority;
- use expired authority;
- widen a delegated scope;
- substitute a different action after approval;
- resolve an outcome without external observation;
- self-grade success;
- turn read authority into write authority;
- bypass a policy gate;
- cause ambient filesystem, network, process, or shell access;
- activate an update that was only proposed or evaluated.

The trusted computing base for v0.1 is the Rust compiler plus the small assurance-kernel implementation. Cryptographic attestation, sandbox enforcement, information-flow analysis, and theorem-prover adapters are later layers.

## Core invariants

The initial research contract is:

1. **A1 — Explicit authority:** execution requires an explicit authority object.
2. **A2 — No widening:** delegated authority may be equal to or narrower than the authority from which it was derived, never broader.
3. **A3 — One-shot use:** one-shot authority cannot be reused through the safe API.
4. **A4 — Expiry:** expired authority cannot authorize an action.
5. **A5 — Scope binding:** authority is bound to an explicit scope.
6. **A6 — Resolution is observational:** resolution authority does not silently imply execution authority.
7. **A7 — No self-grading:** externally resolved outcomes require an observation/evidence transition.
8. **A8 — Typed lifecycle:** invalid action-state transitions are unrepresentable through the public API.
9. **A9 — Lineage:** evidence identifies the action and grant lineage that produced it.
10. **A10 — Fail closed:** failed validation does not produce an executable/authorized state.

## Action lifecycle

The first typed protocol is:

```text
Action<Proposed>
      |
      v
Action<RiskAssessed>
      |
      v
Action<Authorized>
      |
      v
Action<Executed>
      |
      v
Action<Observed>
      |
      v
Action<Resolved>
```

The public API intentionally omits shortcuts such as `Action<Proposed>::execute` or `Action<Executed>::resolve`.

## Authority model

Capabilities are affine Rust values: they are not `Copy`, and one-shot grants are not `Clone`. Each grant carries:

- a stable grant id;
- issuer and subject ids;
- an explicit scope;
- an optional expiry;
- a capability kind marker;
- a delegation depth.

The v0.1 kernel begins with data/resource scopes and does not grant ambient process authority. In particular, a generic shell string is not an assurance primitive.

## Relationship to existing Symthaea layers

- **MAGI / recursive improvement:** first consumer. MAGI should eventually require typed authority to move an action into an executable state.
- **symthaea-evidence-plane:** remains the shared research-evidence contract; assurance receipts can be adapted into it rather than replacing it.
- **symthaea-formal-safety:** remains the proof-obligation/safety-case layer. Assurance invariants can later discharge selected obligations through Kani/Aeneas/Lean evidence.
- **symthaea-hal / Xenia:** can later provide signed grant and receipt attestation without moving private-key custody into this crate.

## Non-goals for v0.1

- modifying rustc;
- inventing new Rust syntax;
- proving the whole Symthaea repository;
- granting unrestricted shell/network/filesystem access;
- cryptographic signing;
- policy-language design;
- claiming that type safety implies AI alignment.

## Qualification strategy

PR 1 is foundation-only: threat model, capability primitives, typestate actions, and adversarial/compile-fail tests. MAGI runtime behavior changes belong in a separate PR so architectural review and behavioral hardening remain independently reversible.

Later qualification should measure where attacks are stopped: compile time, assurance validation, sandbox admission, runtime-safe failure, or escape. Any escape is a critical failure for the tested invariant.
