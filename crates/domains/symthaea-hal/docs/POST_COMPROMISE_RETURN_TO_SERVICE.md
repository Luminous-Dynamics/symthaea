# Post-Compromise Return to Service

Hardened v13 treats recovery as a measured, replayable, and progressively authorized process. Closing an incident does not by itself restore motion authority.

## Evidence sequence

1. Measure containment, authority restoration, fleet rejoin, evidence loss, and unaccounted commands against a `RecoveryObjectivePolicy`.
2. Exercise the recovery path in a signed, production-motion-prohibited `RecoveryDrillPlan`.
3. Reproduce safety decisions using an `EvidenceReplayManifest` bound to source, toolchain, configuration, ordered inputs, expected outputs, runtime, and output-size ceilings.
4. Consume device-attestation challenges through `AttestationNonceJournal` before accepting their reports.
5. Exercise metadata-only authority escrow. The HAL records commitments and receipts but never key shares.
6. Produce a structured postmortem and an evidence-backed corrective-action closure report.
7. Assemble the complete chain into `RecoveryAssuranceBundle`.
8. Evaluate `PostRecoveryAdmissionReport` before advancing service state.
9. Advance through the durable service phases one step at a time.
10. Rejoin devices in bounded waves. Missing or failing devices stop the wave.

## Service phases

- `Disabled`: no motion authority.
- `RecoveryOnly`: diagnostic and recovery operations only; no motion.
- `RestrictedMotion`: hard ceiling of 25% torque and velocity.
- `CanaryMotion`: hard ceiling of 50% torque and velocity.
- `NormalOperation`: ordinary configured envelope, still subject to all existing HAL interlocks.

Every upward transition requires a short-lived signed permit bound to the exact previous checkpoint, recovery-assurance report, corrective-action closure, re-attestation report, incident, deployment, and authority epoch. The checkpoint is generation-linked and atomically persisted. Emergency disable may occur from any phase.

## Authority escrow

`AuthorityEscrowManifest` is intentionally metadata-only. It requires distinct custodian identities, organizations, fault domains, commitments, and storage attestations. `AuthorityEscrowExercise` proves that the threshold can reconstruct the expected public identity in an isolated recovery environment and that temporary destructive material was destroyed. Private shares remain outside the HAL.

## Corrective actions

Corrective actions are classified independently:

- Advisory
- Required before fleet expansion
- Required before motion

An unresolved advisory does not falsely block motion. An unresolved motion-critical action prevents return-to-service admission, while a fleet-expansion action blocks rollout growth without necessarily blocking a bounded already-admitted device.

## Operator tools

- `hal-recovery-drill-verify`
- `hal-evidence-replay-verify`
- `hal-post-recovery-admission`

All tools bound input sizes and reject future-dated, stale, mismatched, or failing evidence.

## Remaining physical work

The source artifacts do not replace a physical recovery exercise. Production acceptance still requires:

- real recovery-media boot and authority-loss drills;
- measured e-stop, output-enable, watchdog, and current-decay evidence;
- attestation-provider interoperability;
- external signer/HSM or TPM integration;
- multi-device restricted and canary motion campaigns;
- independent review of postmortem and corrective-action evidence.
