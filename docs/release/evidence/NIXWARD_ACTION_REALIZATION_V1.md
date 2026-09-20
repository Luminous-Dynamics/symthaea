# Nixward Action Realization V1 — Evidence Boundary

Status: implementation candidate; qualification required on the exact PR head.

## Theorem

Planner recommendation does not imply executable action.

Target string resemblance does not imply typed target provenance.

Realization admission does not imply authorization.

## V1 boundary

`ActionRealizerV1` evaluates an abstract `ActionCategory` against a source-typed, currentness-bearing `TargetEvidenceV1`.

The outcome is one of:

- `Admitted`
- `NeedsResolution`
- `NeedsParameters`
- `Inapplicable`
- `Unsupported`
- `StaleEvidence`

`Admitted` means only that the category and target are semantically compatible under `nixward-action-realizer-v1`. It does not construct a `NixOSCommand`, grant authority, execute anything, or prove a post-state.

## Provenance rules

- structured `SystemdObserver::UnitInfo` may produce a `SystemdUnit` target after conservative exact-name validation;
- a journal producer remains `JournalProducer`, even when its text ends in `.service`;
- predictive-monitor metric names remain `Metric` targets;
- user strings remain `UserSupplied` until an explicit resolver proves stronger semantics;
- observation-backed targets carry bounded currentness and become `StaleEvidence` outside that window.

## First admissibility rules

- Enable/Disable + current observed SystemdUnit -> `Admitted`;
- Enable/Disable + JournalProducer/UserSupplied -> `NeedsResolution`;
- Enable/Disable + Metric -> `Inapplicable`;
- GarbageCollect + explicit System -> `Admitted`;
- Rebuild + explicit System -> `Admitted`;
- Configure + schema-backed NixOption -> `NeedsParameters`;
- Install/Remove + inventory-backed Package -> `NeedsParameters`;
- Rollback + observed SystemGeneration -> `NeedsParameters`;
- Update + System -> `NeedsParameters`;
- Custom -> `Unsupported`.

## Nonclaims

This subject does not:

- create executable commands;
- integrate the daemon active-healing path;
- resolve journal producer IDs against live systemd inventory;
- authorize any action;
- replace `NixActionIntentV1`;
- establish machine identity;
- implement compound config-patch/rebuild authorization;
- claim currentness beyond the supplied observation validity window.

## Qualification

The checked-in focused qualifier is:

`bash scripts/qualify_nixward_action_realization.sh`

A PASS claim requires that exact qualifier to complete successfully on the exact PR head.
