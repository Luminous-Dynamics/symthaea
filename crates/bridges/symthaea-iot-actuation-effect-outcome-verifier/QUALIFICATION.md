# Physical-Effect Outcome Verifier v0.1 — Qualification Boundary

This crate is a **non-authorizing evidence verifier**.

It does not close the durable effect-attempt journal, dispatch a physical effect, mint a permit, or establish globally serialized currentness of outcome-verifier trust publication.

## Policy identity

Outcome policy generation is part of the policy's cryptographic identity and is retained explicitly by historical verified evidence.

This prevents policy ABA:

```text
generation 1 / policy A
        -> generation 2 / policy B
        -> generation 3 / policy A-like semantics
```

Generation 1 and generation 3 are distinct policy identities even if every semantic rule other than generation is byte-identical. A current fence requires the historical proof's exact policy generation and digest to equal the current guard's generation and digest.

A terminal reconciliation writer must additionally require those same values to equal the held reconciliation-trust publication root.

## Proof classes

`ExecutionAndPostcondition` requires both:

1. an exact tamper-evident execution-record commitment timestamped inside the original actuation window; and
2. a fresh postcondition-evidence commitment observed no earlier than the reconciliation challenge.

`NonExecution` requires both:

1. an explicit non-execution proof commitment; and
2. a tamper-evident execution-log-head commitment whose certified coverage spans the complete original wall-clock actuation validity window.

A current-state observation, command-sequence advancement, adapter acknowledgement, or missing postcondition is not sufficient to prove non-execution.

## Currentness

Historical verification produces `VerifiedPhysicalEffectOutcomeEvidence`.

A later `CurrentPhysicalEffectOutcomeFence<'a>` rechecks the exact current policy generation/digest, trust/key, fixed Ed25519 signature, retained commitments, and natural validity ceiling:

```text
min(
  evidence expiry,
  exact verifier-key expiry,
  outcome trust-snapshot expiry,
  reconciliation-challenge expiry
)
```

The fence still does not serialize concurrent publication of a successor outcome policy/trust generation. A terminal reconciliation writer must first hold a separate cross-process reconciliation-trust publication fence and must independently confirm that the rollback-protected effect-attempt journal head still equals the head named by the verified challenge.

## Qualification claim

No qualification claim exists until the exact candidate head passes its focused hosted workflow on Rust 1.94 (with Rust 1.96 formatting), all required upstream exact-head evidence is green, and stacked `Cargo.lock` changes are intentionally reviewed/frozen.
