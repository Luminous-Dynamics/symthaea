# Authenticated Updates, Runtime Policy, and Rollback

Symthaea HAL v9 separates five questions that are often collapsed into a single
"signature valid" check:

1. **What bytes arrived?** `UpdateArtifactManifest` verifies every chunk and the
   final opaque package image.
2. **What was authorized?** `SignedUpdatePlan` binds those bytes to a source
   release, deployment configuration snapshot, authority epoch, and immediately
   preceding update generation.
3. **What may boot?** `UpdateState` stages only into the inactive A/B slot and
   persists trial state before changing the boot target.
4. **What actually booted?** `BootEpochJournal` records an append-only start and
   outcome, anchored by a sidecar checkpoint that detects tail truncation.
5. **What may become permanent?** `SignedUpdateConfirmation` requires a bounded
   health window with successful ticks, watchdog and supervisor evidence, and no
   disallowed faults before `UpdateState::confirm_verified` advances the
   confirmed generation.

## Installation sequence

1. Root-verify and durably accept the current authority trust bundle.
2. Verify the signed update plan using an active key whose purpose is
   `deployment-update`.
3. Verify the source release manifest, configuration snapshot, artifact
   manifest, and full artifact bytes.
4. Receive remote chunks through `UpdateReceiver`; never execute directly from
   the receive spool.
5. Stage the candidate in the inactive slot and persist the update checkpoint.
6. Open a boot epoch before selecting the candidate slot.
7. Run normal startup and security admission against the complete v9 evidence
   set.
8. Record the signed startup seal in both the boot epoch and update checkpoint.
9. Observe the candidate for the configured health window.
10. Verify an `update-confirmation` signature and operator `ConfirmUpdate`
    quorum, then call `confirm_verified`.

## Failure and rollback

A trial failure consumes the persisted trial budget. Once exhausted,
`UpdatePhase::RollbackRequired` permits returning only to the previously
confirmed slot. The confirmed generation and plan digest are not decremented or
replaced by the failed candidate. A new candidate must still be the next linked
generation from the confirmed deployment.

The automatic fallback is intentionally narrower than a general downgrade. A
future emergency downgrade mechanism should use a separate signed break-glass
artifact, independent operator quorum, and explicit vulnerability review.

## Runtime policy changes

`RuntimePolicySnapshot` uses integer units so its identity is deterministic.
Every policy is linked to the previous digest and persisted in
`RuntimePolicyState` before use. Changes are classified as:

- `Equivalent`
- `Tightening`
- `Relaxation`
- `Mixed`

All applications require `ServoLifecycle::VerifiedDisabled` and an independently
verified disabled output gate. Relaxation or mixed changes additionally require
an `ApproveConfiguration` quorum bound to the exact review digest.

## Authority continuity

A new trust epoch can carry an `AuthorityContinuityProof`. Witnesses use active
`authority-witness` keys from the previously accepted epoch and must satisfy
node, signature-key, and fault-domain diversity. This supplements, rather than
replaces, the normal offline/root signature on the candidate trust bundle.

## Incident escrow

The HAL does not implement or claim encryption. An external audited provider
produces ciphertext. `IncidentEscrowEnvelope` binds that opaque ciphertext to the
exact audit export, recipient key IDs, declared encryption algorithm, sorted
redaction manifest, authority epoch, and prior escrow generation. Export policy
can require both an `incident-escrow` signature and `ExportIncidentEscrow`
operator quorum.
