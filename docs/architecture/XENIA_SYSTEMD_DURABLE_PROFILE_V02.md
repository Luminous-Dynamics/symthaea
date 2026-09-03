# Xenia Systemd Durable Recovery Profile v0.2

Status: **draft / unqualified**

## Goal

Compose the bounded systemd recovery stack into the first profile with concrete local durability on both sides of the effect boundary:

- Xenia-authenticated capability provenance;
- an exact generation-zero Agency Kernel authority frontier;
- SQLite-backed linearizable checkpoint CAS;
- durable write-ahead attempt evidence before restart dispatch;
- #305's existing typed one-use systemd state machine;
- independent post-effect observation and accounting;
- a durable causal evidence key that remains queryable after process exit.

This V0.2 adds no new system action. The only mutation remains the exact typed `service.restart` operation from #305.

## Lineage

The implementation is a real merge composition:

- #317 — Xenia-authorized CAS-backed systemd profile;
- #320 — concrete SQLite `CheckpointCasStore`;
- #326 — durable `DispatchArmed` attempt evidence.

The V0.2 branch has a two-parent merge of #320 and #326 before adding the profile crate. It does not copy/reimplement either child state machine.

## Two-phase lifecycle

### Phase A — establish authority frontier

Before Xenia authority exists:

```text
CapabilityGrant
  -> empty GrantAccount
  -> checkpoint sequence 0
  -> SQLite BEGIN IMMEDIATE
  -> CAS(None -> head-0)
  -> durable head-0
```

`DurableXeniaSystemdBootstrap` is returned only after head-0 is installed.

The caller then obtains a `VerifiedXeniaCapability` whose authorization binds exactly that head.

### Phase B — consume verified authority

`recover_verified_once(self, verified, ...)` consumes the bootstrap and performs:

```text
recheck exact grant digest
recheck Xenia expiry
recheck exact prior checkpoint == head-0
  -> commit Xenia provenance to authority_evidence_digest
  -> construct attempt key/context
  -> open append-only SQLite attempt journal
  -> wrap CAS checkpoint store + service backend
  -> restore #305 broker at exact head-0
  -> reserve one use
  -> SQLite checkpoint CAS(head-0 -> reserved-head)
  -> #305 pre-dispatch world re-observation
  -> durable DispatchArmed(attempt-key, reserved-head)
  -> typed real service.restart
  -> durable Applied | ProvenNotDispatched | OutcomeUnknown
  -> #305 accounting checkpoint CAS
  -> independent post-effect observation/reconciliation
  -> #305 RecoveryReceipt
  -> append RecoveryCompleted when possible
```

The Xenia proof object is consumed by value, but the security claim does not depend on Rust affinity. Cross-process reuse is constrained by the durable checkpoint CAS.

## Xenia authority evidence commitment

The attempt journal does not store raw Xenia objects. V0.2 derives a domain-separated commitment over:

- authorization id;
- session id;
- exact Symthaea grant digest;
- exact executor workload digest;
- Xenia ledger entry count + head hash;
- exact prior Agency Kernel checkpoint;
- Xenia authorization expiry.

That digest is embedded in `AttemptEvidenceContext` and returned in the successful durable profile receipt.

A later auditor can therefore associate the attempt lineage with the exact verified Xenia delegation without exposing consent text, session transcript contents, journal text, secrets, or credentials.

## Durable evidence identity

Every admitted attempt receives an `attempt_key`, derived from committed execution/reservation/grant/plan/world/authority provenance.

The key is returned:

- in a successful `DurableXeniaSystemdReceipt`; and
- in `DurableRecoveryError::BrokerAttempt` when #305 returns an error after admission.

The error additionally exposes the latest durable attempt-evidence head, if one exists.

This enables recovery tooling to distinguish:

- no attempt evidence: failure occurred before the real restart effect frontier;
- durable `DispatchArmed`: restart may have occurred and must not be blindly retried;
- terminal evidence: the wrapper captured a dispatch classification;
- `RecoveryCompleted`: the #305 broker also completed its accounting/verification path.

## Final evidence append failure

If #305 returns a successful `RecoveryReceipt` but appending `RecoveryCompleted` fails, V0.2 does **not** rewrite the already-known system effect as unknown.

The returned receipt records:

`DurableAttemptEvidenceStatus::FinalizationIncomplete`

with the last durable evidence head and a diagnostic commitment.

Earlier `DispatchArmed`/terminal evidence and #305's successful accounting checkpoint remain authoritative evidence. This separates evidence-publication completeness from effect truth.

## Same SQLite file, separate tables

V0.2 intentionally uses the same SQLite database path for:

- `agency_checkpoint_frontier`; and
- `system_attempt_evidence`.

They remain separate transactions/tables and therefore do **not** become a falsely claimed atomic cross-subsystem transaction merely because they share a file.

The ordering is intentionally conservative: authority reservation is durable before attempt arming; attempt arming is durable before external dispatch.

## Reopen verification

The integration regression closes/reopens the database after a successful recovery and independently checks:

1. the checkpoint frontier equals the final #305 recovery checkpoint head;
2. the attempt evidence chain re-hashes as exactly:
   - `DispatchArmed`;
   - `Applied`;
   - `RecoveryCompleted`;
3. the first attempt record contains the same Xenia authority evidence commitment returned in the receipt;
4. the final evidence record contains the same broker recovery outcome and independent verification result.

A separate regression proves an expired Xenia proof causes zero backend restart calls and zero attempt-evidence rows.

## Important remaining gaps

### Trusted time

Xenia expiry/checkpoint freshness and the authority context still rely on externally supplied wall-clock time. V0.2 does not yet bind those decisions to HAL monotonic/time-integrity evidence.

### Workload measurement

`ExecutorWorkloadV1` commits to exact artifact/configuration/host identity, but V0.2 does not prove the running broker process was independently measured into those values. Nix output identity plus IMA/TPM/fs-verity or a supervisor measurement path remains future work.

### Xenia issuance persistence ordering

Xenia #232 cryptographically binds its current ledger frontier but does not by itself prove that the relevant approval/frontier was durably committed before the attestation escaped. That ordering remains an Xenia-side integration requirement.

### Restore / crash reconciliation

V0.2 deliberately exposes a fresh one-use bootstrap API. Production restart recovery should compose #309's conservative `Reserved -> OutcomeUnknown` normalization with the SQLite checkpoint and attempt evidence records rather than mint a new authorization path.

### Global rollback

If an attacker can roll back the entire SQLite file together with every trusted external copy of the latest checkpoint/evidence head, SQLite alone cannot detect it. Xenia/TPM/remote witness retention remains required for stronger rollback resistance.

## Non-claims

V0.2 does not establish:

- physical truth merely from a software receipt;
- trusted-time integrity;
- running-process attestation;
- instant revocation under freshness-channel suppression;
- Xenia/Symthaea distributed transaction atomicity;
- TPM/IMA measured boot;
- Byzantine storage resistance;
- global filesystem/VM snapshot rollback immunity;
- production readiness.

## Qualification gate

Promotion requires:

1. exact-head format/check/Clippy/tests for every composed crate;
2. the real SQLite two-connection CAS regression;
3. the `DispatchArmed` fail-closed regressions;
4. the V0.2 reopen/end-to-end evidence-chain regression;
5. explicit crash/fault injection around each persistence/effect boundary;
6. a NixOS deployment profile defining database location, ownership, permissions, backup/restore policy, and independent trusted-head retention;
7. later integration of trusted-time and workload-measurement evidence.
