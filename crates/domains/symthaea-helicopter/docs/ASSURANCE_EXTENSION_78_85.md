# Symthaea Helicopter Assurance Extension 78–85

This extension closes eight operational evidence gaps that remain after the
70–77 series. The modules are deliberately conservative: they return explicit
`Incomplete` outcomes when observations, authenticity providers, freshness, or
identity bindings are absent.

## Patch 78 — Deterministic test oracles

Adds signal-tolerance and response-deadline oracles suitable for replaying the
same acceptance criteria over SIL, HIL, and flight-test records. Completion of a
scenario is not considered success without explicit oracle evidence.

## Patch 79 — Degraded-mode human factors

Adds prioritized, latched, acknowledgement-aware, phase-inhibited annunciation.
Contradictory terminal modes and alert-capacity overflow are classified as
unsafe rather than merely displayed.

## Patch 80 — Secure dual-bank updates

Adds staged verification, external authenticity verification, deployment and
hardware compatibility, anti-rollback counters, trial boots, health evidence,
commit, rollback, and lockout. No cryptographic algorithm is simulated by the
crate.

## Patch 81 — Fleet drift

Binds every aircraft snapshot to a fleet baseline and distinguishes permitted
variance from restricted and grounding drift. Stale snapshots and absent
evidence are incomplete.

## Patch 82 — Maintenance life accounting

Adds serial-number-bound hours, cycles, starts, fatigue damage, exceedances,
inspection thresholds, replacement thresholds, and hard life limits. This is
accounting evidence, not an unsupported claim of remaining-useful-life
prediction.

## Patch 83 — Command security

Adds trusted-origin, mission-authority, deployment, timestamp, expiration,
sequence-gap, replay, payload-digest, and external-authenticity gates before a
command can be accepted.

## Patch 84 — Envelope conformance

Audits what the aircraft actually did, including peak exceedance, cumulative
excursion time, longest excursion, and recovery deadline. It complements the
command-side envelope protectors.

## Patch 85 — Operational readiness

Combines release, deployment, configuration, maintenance, operational,
estimator, timing, envelope, command-security, update, human-factors, mission,
and governance evidence into one time-sensitive dispatch decision. Restricted
operations preserve their exact restrictions; missing evidence cannot become a
generic green status.

## Claim boundary

These additions improve deterministic software assurance and evidence
structure. They do not establish airworthiness, regulatory approval, validated
human-factors performance, cryptographic authenticity, or physical flight
safety without the corresponding independent evidence, qualified hardware,
trained operators, and external review.
