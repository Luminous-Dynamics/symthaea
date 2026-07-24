# Helicopter Assurance Extension 94–101

This extension closes the gap between a large set of engineering mechanisms and a reviewable assurance package. It does **not** claim regulatory certification or flight readiness. Every module preserves explicit `Pass`, `Fail`, and `Incomplete` outcomes so absent evidence cannot be interpreted as success.

## Patch 94 — Hazard closure

`hazard_closure` requires safety objectives, severity-appropriate mitigations, verification evidence, independent review for high-severity hazards, and attributable residual-risk acceptance. Expired acceptance and failed verification are failures; missing artifacts remain incomplete.

## Patch 95 — Freedom from interference

`partition_assurance` declares software criticality, processor/memory/restart domains, resource budgets, communication channels, and aggressor/victim interference tests. Cross-criticality channels must be authenticated, bounded, and fail closed.

## Patch 96 — Trusted identity and time

`trusted_identity_time` binds aircraft, device, hardware root, deployment digest, secure boot, boot counters, monotonic counters, and multiple independent time sources. The module consumes externally authenticated evidence; it does not implement cryptography.

## Patch 97 — Rollback drills

`rollback_drill` verifies the update rollback path by exercise rather than declaration. Drills cover detection, trial rejection, disarming, bank selection, state restoration, restart, health verification, and restored authority under bounded deadlines.

## Patch 98 — Endurance campaigns

`endurance_campaign` evaluates long-duration operation across required phases. It gates memory growth, queue depth, CPU load, thermal exposure, deadline misses, watchdog resets, restart behavior, sample continuity, and terminal evidence sealing.

## Patch 99 — Fleet safety actions

`fleet_safety_action` represents scoped advisories, restrictions, inspections, grounding, and software or hardware recalls. It evaluates acknowledgement, operational state, remediation completion, deadlines, authenticity references, and per-aircraft compliance.

## Patch 100 — Return to service

`return_to_service` requires closed work orders, completed tasks, independent inspection for critical work, component traceability, life-limit reset evidence, functional tests, deployment/calibration identity, and an attributable release decision.

## Patch 101 — Certification dossier

`certification_dossier` binds the status, identity, digest, age, validity, and evidence of the complete assurance set. Required independent roles review the latest artifacts. The strongest output is `ReadyForExternalReview`, deliberately not “certified.”

## Full-workspace verification

Run from the Symthaea workspace root:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-helicopter --all-targets --all-features
cargo test -p symthaea-helicopter --all-features
cargo clippy -p symthaea-helicopter --all-targets --all-features -- -D warnings
```

The standalone archive does not include all path dependencies required for these commands.
