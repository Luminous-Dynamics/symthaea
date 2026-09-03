# Xenia-Authorized Systemd Recovery Profile v0.1

Status: **draft / unqualified**

## Purpose

Compose the bounded Agency Kernel pieces into one consequential vertical slice without adding another effect type.

The only mutation remains:

`service.restart` for one exact `.service` resource.

## Composition

```text
exact CapabilityGrant
        |
        v
CAS generation-zero checkpoint          (#316)
        |
        +---- exact prior head -------------------+
        |                                         |
        v                                         v
Xenia session-bound capability attestation     workload identity
        |                                         |
        +-------------------+---------------------+
                            v
                independent verifier             (#315)
                            |
                   VerifiedXeniaCapability
                            |
                  recheck current checkpoint
                            |
                            v
                   typed systemd broker           (#305)
                            |
                   reserve one use/risk
                            |
                      CAS successor
                            |
                   typed systemctl restart
                            |
                  observe / verify / receipt
```

## Bootstrap order

`XeniaSystemdRecoveryProfile::bootstrap` establishes a generation-zero CAS checkpoint before external authority is requested.

The returned head is the exact checkpoint Xenia must bind in `prior_checkpoint`.

No restart authority is consumed by bootstrap.

## Effect-entry rule

`recover_verified_once` consumes `VerifiedXeniaCapability` by value and immediately rechecks:

- proof grant digest == profile grant digest;
- proof has not expired at effect entry;
- proof prior checkpoint == broker's exact current checkpoint.

Only then does it call #305.

#305 independently rechecks the semantic grant/plan/world/resource/operation/epoch/negative-fact constraints and then attempts to persist the use reservation before dispatch.

## Why an affine proof is not enough

The same signed Xenia attestation can be cryptographically verified twice in two processes. A non-Clone Rust proof token prevents accidental in-process reuse but cannot establish distributed single-use authority.

The durable rule is instead:

```text
both processes start at checkpoint H

A verifies proof(H)
B verifies proof(H)

A reserves -> CAS(H -> H1) succeeds -> may continue
B reserves -> CAS(H -> H1') fails    -> containment / no dispatch
```

The integration test deliberately verifies the same attestation twice and proves the losing process never invokes its backend restart.

## Receipt binding

On a successful #305 recovery the composed receipt adds privacy-minimized Xenia provenance:

- Xenia authorization id;
- Xenia session id;
- Xenia ledger frontier;
- executor workload digest;
- underlying `RecoveryReceipt`.

It does not include consent text, plan text, journal text, secrets, credentials, or arbitrary command output.

## Trusted-time boundary

The Xenia verifier and #305 both depend on a trustworthy current time for expiry/freshness semantics.

V0.1 receives time from its enclosing authority context/verifier call. It does not itself implement HAL-style wall-clock/monotonic integrity checks. A production profile should consume a trusted-time state that can enter containment on clock regression, suspicious suspend gaps, or loss of freshness authority.

Do not describe ordinary `SystemTime::now()` alone as a fully trusted authority clock.

## Unknown-outcome evidence gap

#305 conservatively persists `OutcomeUnknown` before returning on an unclassified post-dispatch backend failure, so duplicate authority does not reappear.

However, some error paths return `BrokerError` rather than a caller-facing durable `RecoveryReceipt`. The checkpoint contains the conservative accounting truth, but the composed profile cannot always return a complete Xenia-linked attempt receipt to its caller.

The next evidence tranche should introduce a result shape where every post-effect-entry path yields durable attempt evidence, including:

- known applied;
- known not dispatched;
- outcome unknown;
- postcondition unavailable;
- containment after persistence uncertainty.

This should not weaken the current fail-closed accounting just to make receipt construction easier.

## Cross-system atomicity boundary

The profile composes two durable domains:

- Xenia consent/authority ledger;
- Symthaea checkpoint CAS store.

V0.1 does not claim a distributed atomic transaction spanning both systems.

Safe ordering is asymmetric:

1. establish Symthaea checkpoint frontier;
2. durably establish Xenia consent/frontier;
3. release Xenia attestation;
4. verify attestation + fresh Xenia checkpoint;
5. CAS-reserve Symthaea use before effect.

Failure may leave unused authorization/evidence, but must not create an unrecorded effect.

## Non-claims

V0.1 does not establish:

- production CAS durability/linearizability for a concrete backend;
- running-workload TPM/IMA attestation;
- a trusted-time implementation;
- instant Xenia revocation under total freshness-channel suppression;
- a distributed transaction across Xenia and Symthaea;
- complete caller-facing receipts for every unknown/error path;
- PQ agent-attestation acceptance;
- physical-host production readiness.

## Next exit gates

Before physical-host activation:

1. exact-head compiler/test/Clippy evidence for #232, #315, #316 and this profile;
2. concrete CAS backend qualification;
3. executor workload measurement from Nix/attestation evidence rather than caller assertions;
4. trusted-time integrity integration;
5. complete post-effect attempt receipts;
6. Xenia issuance ordering that cannot release an attestation before the corresponding consent/frontier is durably acknowledged;
7. hostile crash injection at every transition from bootstrap through receipt.
