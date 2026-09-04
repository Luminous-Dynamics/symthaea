# Xenia witness-frontier verifier V0.1

## Purpose

Authenticate Xenia's typed qualification-witness anchor/currentness evidence independently inside Symthaea before that evidence is allowed to influence witness-frontier recovery or publication decisions.

This verifier mirrors the commitment-only protocol introduced by `xenia-peer` #293. It deliberately does **not** depend on Xenia source crates.

The public trust chain is:

```text
Xenia durable witness anchor
        +
fresh challenge-bound Xenia observation
        +
trusted Xenia ledger public key
        +
trusted source epoch/policy/witness expectation
        +
exact reviewed witness-freshness policy
        +
subject-bound VerifiedAuthorityTime
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

## Trusted time is a typed prerequisite

The public verifier does **not** accept `SystemTime`, caller-supplied `now`, or a caller-constructed time interval.

It requires Symthaea's existing opaque `VerifiedAuthorityTime`, produced by the challenge-bound multi-authority time path.

That time fact must itself bind a deterministic subject commitment for this exact currentness check:

```text
BLAKE3(
    "symthaea.xenia-witness-frontier.time-subject.v1\0"
    || u16_be(1)
    || trusted_xenia_ledger_public_key
    || derived_source_id
    || u64_be(source_epoch)
    || anchor_policy_digest
    || witness_id
    || verifier_challenge
    || exact_signed_anchor_fingerprint
    || authority_time_policy_digest
    || u64_be(max_observation_age_s)
    || u64_be(max_future_skew_s)
)
```

`xenia_witness_frontier_time_subject_digest_v1` verifies the signed anchor and exact source bindings before returning this subject. The caller can then obtain `VerifiedAuthorityTime` for that exact subject and the reviewed time policy.

A valid time fact for another capability, anchor, witness, challenge, key, source policy, trusted-time policy, or freshness limit cannot be substituted.

The public verifier also checks:

```text
authority_time.policy_digest()
    ==
reviewed authority_time_policy_digest
```

before interpreting the time fact.

## Conservative freshness interval

After the subject and time-policy bindings are checked, the verifier derives time only from `VerifiedAuthorityTime`:

- the consensus lower bound at verification becomes the conservative earliest current time;
- `conservative_now_unix_s()` becomes the current upper bound and also enforces the time fact's short post-verification lifetime.

It then applies:

```text
oldest acceptable observation
    = conservative_current_upper_bound - max_observation_age_s

latest acceptable observation
    = consensus_lower_bound_at_time_verification + max_future_skew_s
```

This intentionally fails closed as time uncertainty or post-verification age grows. Greater uncertainty cannot make an old observation live longer.

The raw interval checker exists only inside the private protocol module for source-level unit testing and is not exported as a production verification boundary.

## Freshness policy binding

`XeniaWitnessFrontierFreshnessPolicyV1` binds all three reviewed inputs used by currentness admission:

- the exact `authority_time_policy_digest`;
- `max_observation_age_s`;
- `max_future_skew_s`.

All three are included in the authority-time subject commitment. Changing the trusted-time policy or either freshness limit therefore requires a new challenged time fact.

These settings must still be configured consistently with the trusted Xenia `anchor_policy_digest`. Symthaea treats that source policy digest as an opaque reviewed commitment; V0.1 does not claim it can reverse or reinterpret the complete Xenia source-policy document from the digest alone.

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

This module exposes no capability minting, reservation, dispatch, retry, budget, executor, shell, systemd, browser, or mutation path for witness evidence.

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

Protocol/source tests cover:

- exact signed anchor + fresh observation verification;
- challenge substitution rejection;
- stale observation rejection;
- re-signed current-summary substitution rejection;
- source relabelling rejection;
- signed ledger-context regression rejection;
- observation signature tampering rejection.

The public-boundary integration tests additionally prove:

- subject-bound multi-authority `VerifiedAuthorityTime` admits the exact currentness check;
- an otherwise valid `VerifiedAuthorityTime` for another subject cannot establish Xenia witness currentness;
- an otherwise valid `VerifiedAuthorityTime` produced under another time policy cannot satisfy the reviewed witness-freshness policy.

## Qualification

The dedicated workflow runs against the existing `symthaea-xenia-authority` package without adding a dependency family or hand-editing `Cargo.lock`.

The lane:

1. preserves the checked-in `Cargo.lock` bytes;
2. lets Cargo reconcile workspace metadata;
3. rejects any removed package, changed existing package, or newly added registry/Git package;
4. permits only additive local/path workspace nodes as a diagnostic candidate;
5. runs Rustfmt, tests, and Clippy against that Cargo-owned candidate;
6. records exact HEAD/tree/toolchain/source and the candidate lock/diff;
7. finally fails qualification unless the checked-in lock was already byte-fresh.

This keeps compiler diagnostics available without allowing lock staleness to become a qualified result.

The execution profile remains:

- Rust 1.96.0;
- `cargo fmt --check -p symthaea-xenia-authority`;
- `cargo test --locked -p symthaea-xenia-authority`;
- `cargo clippy --locked -p symthaea-xenia-authority --all-targets -- -D warnings`;
- exact HEAD/tree/toolchain/lock/source evidence retained.

No qualification claim exists until that exact-head workflow completes successfully, including the final checked-in lock freshness gate.

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
