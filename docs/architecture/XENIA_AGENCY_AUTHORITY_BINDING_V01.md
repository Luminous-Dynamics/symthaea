# Xenia ↔ Symthaea Agency Authority Binding v0.1

Status: **draft / unqualified**

## Goal

Give the Symthaea Agency Kernel cryptographic evidence that Xenia authorized one exact semantic capability for one exact executor workload, while keeping application payloads and secrets out of Xenia's signed authorization object.

The core rule is:

> A valid Xenia signature is necessary evidence of delegation, but it is not by itself proof of live authority.

Live admission requires agreement across the semantic grant, executor workload, Symthaea anti-rollback checkpoint, session provenance, and a fresh signed Xenia ledger frontier.

## Inputs

V0.1 verifies:

1. exact Symthaea `CapabilityGrant`;
2. exact `ExecutorWorkloadV1`;
3. exact current Symthaea `CheckpointHead`;
4. Xenia agent-capability attestation;
5. trusted Xenia ledger Ed25519 public key;
6. expected session id/transcript commitment/suite;
7. fresh signed Xenia `LedgerCheckpoint`;
8. trusted current wall-clock time and explicit freshness policy.

## Workload identity

`ExecutorWorkloadV1` commits:

- exact capability audience/executor principal;
- exact executable/build artifact digest;
- exact security-relevant configuration digest;
- exact host/workload identity digest.

For Nix deployments the artifact commitment should come from independently measured/qualified Nix output identity or an equivalent content commitment, not a human version string.

V0.1 defines the semantic commitment but does not itself measure the running executable. Measured boot, IMA/TPM, fs-verity, Nix store verification, or a supervisor-owned measurement source can supply stronger evidence later.

## Xenia signed authorization

The frozen cross-repository canonical object binds only fixed-size authority metadata:

- authorization id;
- Xenia session id;
- session transcript commitment + signature-suite id;
- exact `CapabilityGrant` digest;
- exact executor-workload digest;
- authority epoch;
- issuance + expiry;
- nonce;
- Xenia ledger entry count + head hash;
- exact prior Symthaea checkpoint anchor.

It does not contain plan text, journal content, browser content, secrets, credentials, or application payloads.

## Freshness / revocation behavior

An old cryptographically valid authorization is rejected when the supplied fresh signed Xenia checkpoint has a different ledger frontier.

This deliberately means *any* intervening Xenia ledger event invalidates the one-shot V0.1 authorization and requires fresh authorization. That is conservative but simple and safe for consequential short-lived effects.

A later protocol may use scoped revocation epochs so unrelated ledger events do not cause unnecessary reauthorization.

### Suppression limitation

A signed checkpoint proves the Xenia frontier as of its signed timestamp. It cannot prove that an attacker is not suppressing a newer checkpoint.

V0.1 therefore requires an explicit maximum checkpoint age and future-skew tolerance. Stronger deployments should add one or more of:

- authenticated online freshness challenge;
- independently retained checkpoint witness;
- threshold/witness quorum;
- Xenia revocation push with durable sequence;
- transparency/SCITT-style anchoring;
- TPM/supervisor freshness state.

Do not convert this limitation into a claim that short expiry equals instant revocation.

## Session provenance boundary

The authorization binds an exact Xenia session transcript commitment, but this Symthaea crate does not independently verify the handshake transcript signature. The expected session provenance must come from the Xenia-authenticated integration boundary.

Likewise, Xenia PR #232 currently binds the authorization to a `SessionTranscriptBinding`; its new signer does not itself replay the entire transcript-signature verification ceremony. Treat this V0.1 evidence as **session-bound Xenia delegation** until issuance consumes an explicit verified transcript-authentication witness.

## Symthaea anti-rollback binding

V0.1 requires `prior_checkpoint = Some(current_checkpoint)` for consequential authority.

This prevents an old Xenia authorization from being replayed after the Agency Kernel has advanced its execution/checkpoint lineage. A system starting from no checkpoint should first establish and externally retain a generation-zero grant/account checkpoint before requesting Xenia authority.

## Affine result

Successful verification returns `VerifiedXeniaCapability`, which intentionally is not `Clone`.

This is only a local affine proof token. It does not replace `symthaea-action-runtime` reservation accounting. The next integration should consume this token in the same transaction that reserves the exact capability use.

## Frozen interoperability vector

Xenia and Symthaea independently encode the same static authorization vector.

Expected canonical message length: **292 bytes**.

Ed25519 seed: 32 bytes of `0x03`.

Expected public key:

`ed4928c628d1c2c6eae90338905995612959273a5c63f93636c14614ac8737d1`

Expected signature:

`f34266c584aea26f8494f505e3fabac490ced192c604b04c9763e2d12dcbcea9f665249faad37d1eaef17b7b00118ac3d23d47d16c226636dbc7d20a05717c01`

Both repositories test this independently. No source dependency is used to make the tests agree.

## V0.1 signature profile

The Symthaea verifier accepts only Xenia ledger **Ed25519 / RFC 8032** attestations and checkpoints.

Xenia's evidence layer is signature-agile and already has ML-DSA support behind its policy/features, but V0.1 does not silently inherit that wider surface. PQ agent authority should receive an explicit V0.2/profile review and its own fixed interop vectors.

## Non-claims

V0.1 does not establish:

- instant revocation under total freshness-channel suppression;
- running-process attestation;
- Xenia consent-frontier persistence ordering;
- durable nonce consumption;
- distributed single-writer/checkpoint CAS;
- kernel/root compromise resistance;
- PQ agent-attestation acceptance;
- independent verification of the Xenia handshake transcript signature;
- production readiness.

## Next integration gate

Before using this for physical-host systemd mutation:

1. Xenia #232 and the Symthaea verifier must have exact-head compiler/test evidence;
2. an initial Agency Kernel checkpoint must exist before authorization;
3. Xenia authorization must be released only after its corresponding consent/frontier durability boundary succeeds;
4. the systemd wrapper must consume `VerifiedXeniaCapability` and reserve the matching grant atomically enough that the proof cannot be raced into two executions;
5. executor workload identity must come from a qualified measurement source;
6. fresh Xenia checkpoint retrieval/witness semantics must be defined operationally.
