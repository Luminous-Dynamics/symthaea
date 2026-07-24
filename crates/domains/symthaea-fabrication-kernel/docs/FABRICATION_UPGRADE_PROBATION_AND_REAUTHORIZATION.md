# Fabrication Upgrade Probation and Reauthorization

Series 14 adds a post-activation authority layer between secure handoff and finalization.
An upgrade is not considered operationally proven merely because the successor process
started, loaded durable state, and passed its migration ceremony.

## Authority pipeline

```text
AuthorizedUpgradeHandoff
        |
        v
FabricationUpgradeState::Activated
        |
        +--> UpgradeProbationObservation[]
        |       - exact handoff and successor state
        |       - machine and region identity
        |       - closed job accounting
        |       - telemetry evidence
        |       - containment and emergency-stop counts
        |
        v
AuthorizedUpgradeProbationClearance
        |
        +--> SignedHardwareReauthorization per required machine
        |       - hardware identity
        |       - machine profile
        |       - firmware
        |       - calibration
        |       - capability set
        |       - successor source and executable
        |
        +--> AuthorizedEvidenceRetentionPolicy
        +--> VerifiedKeyContinuity
        +--> VerifiedClockContinuity
        +--> no AutomaticRollbackTrigger
        |
        v
AuthorizedUpgradeFinalization
        |
        v
UpgradeHandoffTracker::Finalized
```

## Probation

Probation aggregates bounded observations across distinct machines and regions. The
clearance policy can require minimum elapsed time, successful jobs, machine diversity,
regional diversity, and strict budgets for failures, uncertain outcomes, emergency stops,
and containment actions.

Observation order does not change evidence identity. Concurrent observations from
separate machines are permitted; observation digests are canonicalized and duplicate
evidence is rejected.

## Automatic rollback

Health signals can represent emergency stops, containment escalation, failure rate,
uncertain outcomes, telemetry loss, state divergence, or clock discontinuity. Evaluation
is deterministic and age-bounded. A trigger does not contact hardware or mutate durable
state by itself. It becomes rollback authority only after the dedicated
`automatic-upgrade-rollback` threshold ceremony.

Once an automatic rollback digest enters `FabricationUpgradeOperationalState`, it cannot
be removed or substituted by a later state generation.

## Hardware reauthorization

A machine must sign a statement binding its exact identity and configuration to the
upgrade handoff. Verification requires a fresh trust snapshot and a key eligible for
`KeyUsage::HardwareReauthorization`.

The persistent tracker rejects:

- handoff substitution;
- sequence rollback;
- same-sequence statement substitution;
- issue-time regression;
- malformed persisted records.

The tracker retains the hardware identity, machine-profile, firmware, calibration, and
capability digests rather than preserving only an opaque statement digest.

## Retention and continuity

Evidence retention is policy-controlled by evidence class. Incident-bound legal holds
force hot retention regardless of nominal age. Deletion is possible only when the class
policy explicitly permits it and the total-retention and last-reference horizons have
both elapsed.

Key continuity requires bridge keys across predecessor and successor trust snapshots for
each configured authority usage, minimum successor quorum capacity, and optional
algorithm diversity.

Clock continuity requires a monotonic epoch, bounded forward gap and consensus jump,
shared sources, and optionally a shared signature algorithm.

## Finalization

`AuthorizedUpgradeFinalization` is deliberately short-lived. It requires:

- an activated upgrade state for the exact handoff;
- unexpired probation clearance;
- every required machine to have current reauthorization;
- an effective retention policy;
- validated key and clock continuity evidence;
- no recorded automatic rollback trigger;
- authorization before the original handoff finalization deadline;
- a dedicated `upgrade-finalization` threshold ceremony.

The finalization capability should be recorded as the evidence digest on the transition
to `UpgradeStage::Finalized`.

## Durable state and replay

`FabricationUpgradeOperationalState` is hash-linked and tracks the sequence or count that
corresponds to every mutable evidence digest. A digest cannot change without its
associated monotonic value advancing. Probation clearance and rollback authority cannot
be removed after insertion.

`UpgradeOperationalReplayContract` and `UpgradeOperationalEvidenceBundle` bind the source
tree, handoff, upgrade state, probation, machine reauthorization tracker, retention
policy, key continuity, clock continuity, optional rollback trigger, and operational
state. Bundle decoding replays every retained cross-binding before returning evidence.

## Remaining physical validation

This authority layer does not prove printer firmware correctness, calibration accuracy,
sensor authenticity, or the safety of an actual upgrade. Production use still requires
real cryptographic providers, controlled power-loss testing, multi-machine probation,
clock-source failure drills, rollback drills, and supervised hardware finalization.
