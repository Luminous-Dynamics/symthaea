# Xenia witness-frontier verifier V0.1

## Purpose

Authenticate Xenia's typed qualification-witness anchor/currentness evidence independently inside Symthaea before that evidence is allowed to influence witness-frontier recovery or publication decisions.

This verifier mirrors the commitment-only protocol introduced by `xenia-peer` #293. It deliberately does **not** depend on Xenia source crates.

The trust chain is:

```text
Xenia durable witness anchor
        +
fresh challenge-bound Xenia observation
        +
trusted Xenia ledger public key
        +
trusted source epoch/policy/witness expectation
        +
trusted time interval
        ↓
VerifiedXeniaWitnessFrontierV1
```

The result is evidence chronology only. It is not an execution capability.

## Independent protocol reconstruction

Symthaea independently reconstructs the exact Xenia V1 objects and canonical bytes rather than accepting a Xenia-produced `verified=true` flag.

The verifier reproduces:

- Xenia source-ID derivation;
- Symthaea witness-frontier statement commitment;
- Symthaea #457 anchor-operation commitment;
- Xenia durable-anchor canonical message;
- Xenia anchor fingerprint;
- Xenia fresh-observation canonical message;
- Xenia observation fingerprint.

Any byte-level disagreement fails verification.

## Exact source identity

The expected source namespace is derived from the externally trusted Xenia ledger key and reviewed anchor-policy commitment:

```text
source_id = first16(BLAKE3(
    "xenia.witness-frontier-source-id.v1\0"
    || trusted_xenia_ledger_public_key
    || anchor_policy_digest
))
```

The caller does not get to label an arbitrary signed record as the trusted source.

## Durable anchor verification

`XeniaSignedWitnessFrontierAnchorV1` must:

1. have valid V1 structure;
2. reproduce the exact Symthaea frontier-statement digest;
3. reproduce the exact #457 operation ID;
4. derive the expected Xenia source ID;
5. have consistent predecessor-fingerprint shape for its source sequence;
6. use the V1 Ed25519 signature envelope;
7. verify under the exact trusted Xenia ledger public key.

The anchor fingerprint is computed only after its embedded Ed25519 signature verifies, matching the Xenia source semantics.

## Fresh observation verification

A durable anchor is history, not proof that it remains current.

`XeniaSignedWitnessFrontierObservationV1` must therefore verify under the same trusted Xenia key and bind the exact verifier-provided nonzero challenge.

The expectation also fixes:

- source epoch;
- anchor-policy digest;
- witness ID.

The verifier rejects a valid signature from the wrong Xenia key, epoch, policy, witness, or challenge.

## Trusted-time interval

The verifier does not read ordinary `SystemTime` and call it trusted.

It consumes a conservative externally established interval:

```text
earliest_now_unix_s <= actual trusted now <= latest_now_unix_s
```

and applies:

```text
oldest acceptable observation
    = latest_now_unix_s - max_age_s

latest acceptable observation
    = earliest_now_unix_s + max_future_skew_s
```

This intentionally fails closed as trusted-time uncertainty grows.

A production caller should derive this interval from Symthaea's reviewed authority-time/HAL path rather than an unconstrained caller clock.

## Exact current-anchor binding

A valid fresh observation must contain a current-anchor summary equal to the exact supplied durable anchor:

- Xenia anchor sequence;
- anchor fingerprint;
- deterministic operation ID;
- witness high watermark;
- reservation head;
- witness frontier-statement digest.

This prevents an old durable anchor from being paired with a fresh observation about another current anchor.

The observation timestamp must also be at or after the anchor's issue timestamp.

## Xenia ledger context

Both objects carry signed Xenia consent-ledger count/head context.

V0.1 rejects obvious regression:

```text
observation entry_count < anchor entry_count
    => reject

same entry_count + different head
    => reject
```

This is an internal consistency rule only.

**It is not proof that the consent ledger was durably persisted.** Xenia #295's durable-frontier witness is the source-side mechanism intended to establish that separate property before production anchor/currentness issuance.

## Verified result

`VerifiedXeniaWitnessFrontierV1` is only constructible through the verifier and carries:

- source ID;
- source epoch;
- Xenia anchor sequence;
- witness ID;
- witness high watermark;
- reservation head;
- exact frontier-statement digest;
- deterministic operation ID;
- exact signed anchor fingerprint;
- freshness-evidence digest;
- observation timestamp.

The freshness-evidence digest is the domain-separated fingerprint of the complete signed fresh observation.

These are the fields needed to translate into #452's generic `ExternalWitnessFrontierClaimV1` in a later glue layer.

## Authority boundary

This module does not expose `CapabilityGrant`, reservation, dispatch, retry, budget, executor, shell, systemd, browser, or mutation APIs.

Therefore:

```text
verified Xenia chronology
        ≠
execution authority
```

A verified witness frontier may alter rollback/currentness classification. It cannot authorize an external effect.

## Suppression model

A challenge-bound response prevents replay of a previously captured observation for a new challenge.

It does not force an adversarial network/source to answer.

Suppression therefore becomes an availability/currentness failure:

```text
no acceptable fresh response
        ↓
no VerifiedXeniaWitnessFrontierV1
```

It never becomes evidence that an old anchor is current.

## Tests authored

The source tests cover:

- exact signed anchor + fresh observation verification;
- challenge substitution rejection;
- stale observation rejection;
- re-signed current-summary substitution rejection;
- source relabelling rejection;
- signed ledger-context regression rejection;
- observation signature tampering rejection.

## Qualification

The dedicated workflow runs against the existing `symthaea-xenia-authority` package without adding dependencies or changing `Cargo.lock`:

- Rust 1.96.0;
- `cargo metadata --locked` with lock-byte equality;
- `cargo fmt --check`;
- `cargo test --locked`;
- Clippy `--all-targets -D warnings`;
- exact HEAD/tree/toolchain/lock/source digests retained as evidence.

No qualification claim exists until that exact-head workflow completes.

## Deliberate next boundary

After this verifier is compiler-qualified, the next small adapter should translate `VerifiedXeniaWitnessFrontierV1` into #452's transport-neutral external claim/verifier boundary and demonstrate:

```text
fresh authentic Xenia anchor
        ↓
#452 verified external frontier
        ↓
ancestry-aware local recovery classification
        ↓
#456 publication guard
```

That adapter must preserve the same rule: evidence chronology can constrain publication, but it cannot create or recover execution authority.
