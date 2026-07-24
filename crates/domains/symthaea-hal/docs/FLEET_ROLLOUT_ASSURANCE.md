# Fleet Rollout Assurance

Hardened v10 extends the single-robot admission and authenticated update path into a fleet-safe rollout protocol. The design keeps the physical fail-stop boundary local to every robot while making fleet coordination deterministic, reviewable, and incapable of masking one unsafe device behind averages.

## Admission sequence

1. Verify a quorum of independently signed time samples for the current boot.
2. Verify the signed fleet inventory and its generation link.
3. Verify each active device's startup, update, rollback-rehearsal, time, and recovery evidence.
4. Assemble the fleet admission report. A missing, stale, quarantined, or mismatched device fails individually.
5. Verify the canary rollout plan and deterministic device assignment.
6. Activate a durable rollout checkpoint.
7. Collect one bounded health observation for every assigned device.
8. Advance only when the exact current health report passes every device threshold.
9. Halt or require rollback on any missing or failing device result.
10. Preserve recovery media and rollback rehearsal evidence in the portable fleet assurance bundle.

## Non-negotiable invariants

- Final rollout coverage is exactly 100 percent.
- Device assignment is deterministic from fleet, rollout, salt, and device identity.
- Rollout stages never advance on aggregate averages alone.
- Confirmed devices must first be admitted.
- Quarantined devices cannot be marked confirmed.
- Recovery media is bound to deployment, hardware identity, confirmed generation, source release, and exact file bytes.
- Recovery paths reject traversal, symlinks, byte changes, and missing physical write-protection evidence when policy requires it.
- Trusted time requires distinct signing keys, sources, and fault domains with overlapping uncertainty intervals.
- Rollback evidence proves the candidate remained unconfirmed, the confirmed generation was preserved, the confirmed slot returned, and physical outputs were disabled.

## Recommended production policy

Use at least two independent trusted-time fault domains, two independently stored recovery-media manifests, and all rollback scenarios listed in `deploy/examples/rollback-rehearsal-policy.json`. Require human review between stages and keep the first canary cohort small enough that every device can be physically observed.

The rollout controller is not a physical safety controller. Every robot must continue enforcing e-stop, OE, watchdog, command freshness, operator presence, and local fault-latching independently of fleet connectivity.
