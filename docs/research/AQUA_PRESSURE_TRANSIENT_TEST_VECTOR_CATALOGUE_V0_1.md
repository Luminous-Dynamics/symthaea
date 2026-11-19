# Aqua Pressure-Transient Event Contract — Adversarial Test Vector Catalogue v0.1

**Status:** proposed conformance vectors; not executed against a software validator or physical rig  
**Schema:** [aqua-pressure-transient-event.schema.json](aqua-pressure-transient-event.schema.json)  
**Protocol:** [Aqua Pressure-Transient and Supply-Recovery Benchmark Protocol v0.1](../AQUA_PRESSURE_TRANSIENT_AND_SUPPLY_RECOVERY_PROTOCOL_V0_1.md)

## Purpose

These vectors define the minimum negative-control suite for a future event-manifest parser and trace-admission implementation. They deliberately separate JSON Schema validity from metrology admission, method comparability, task qualification and physical safety.

A schema-valid manifest is only structurally well-formed. It is not proof that a sensor is calibrated, a pressure trace is physically plausible, the apparatus is rated, the event was faithfully replayed or a bathing task succeeded.

## Vector catalogue

| ID | Mutation / scenario | Expected gate | Required disposition |
|---|---|---|---|
| PT-SYN-001 | Synthetic supply-sag event with generator identity, parameter digest, synchronized channels and raw trace receipt | Schema | Structurally valid fixture only; always labelled synthetic |
| PT-SITE-001 | Site-observed event with site profile/revision, site-condition digest, survey receipt and a bounded observation window | Schema + site evidence | Structural admission only; site authority must still authenticate the survey and trace provenance |
| PT-NEG-001 | Mark an event `site_observed` but omit site profile revision or survey receipt | Schema | Reject |
| PT-NEG-002 | Mark a synthetic generated waveform as site-observed without site evidence | Schema | Reject; generator metadata cannot substitute for a site observation |
| PT-NEG-003 | Declare dropped samples but omit the per-channel gap-report digest | Schema | Reject |
| PT-NEG-004 | Omit clock identity, timestamp basis or synchronization receipt | Schema | Reject |
| PT-NEG-005 | Declare a pressure channel in kPa but give it a flow or temperature datum | Semantic channel validator | Reject; role, datum and unit must agree |
| PT-NEG-006 | Put sample timestamps out of order, duplicate timestamps where the method forbids duplicates, or outside the event interval | Trace validator | Reject; JSON Schema alone does not prove monotonic time |
| PT-NEG-007 | Effective measurement bandwidth exceeds the justified Nyquist / anti-alias envelope for the declared sample rate | Trace validator | Reject or Indeterminate, according to the preregistered method; never silently downsample and pass |
| PT-NEG-008 | Mark pressure channel as clipped, saturated, out of range or unknown | Metrology admission | Indeterminate / invalid run; never a zero excursion or PASS |
| PT-NEG-009 | Compare terminal-fitting-inlet pressure against point-of-discharge or faceplate-centre pressure | Matched comparison | Reject comparison as cross-datum |
| PT-NEG-010 | Use UWL hot/cold inlet pressure as one averaged scalar | Method-specific evidence | Reject for UWL method admission; retain independent inlet channels |
| PT-NEG-011 | Replay same event ID but with different event waveform digest, branch-load schedule or fixture topology | Matched comparison | Reject as unmatched event conditions |
| PT-NEG-012 | Pressure uncertainty interval crosses the complete rig/test-article rating boundary | Apparatus admission | Hard stop / invalid run; do not truncate the uncertainty interval |
| PT-NEG-013 | Average pressure looks acceptable while the lower tail has an out-of-envelope excursion | Trace analysis | Failure remains visible; average cannot compensate for a hard excursion |
| PT-NEG-014 | Supply is restored but the flow/temperature never returns to the preregistered qualified band | Recovery assessment | Recovery failed or Indeterminate; no automatic return-to-service inference |
| PT-NEG-015 | Candidate has lower fresh-water use only because interrupted / aborted tasks, purge or cleaning water were omitted | Resource closure | Reject efficiency claim; require per-attempt and per-qualified-completed-task accounting |
| PT-NEG-016 | A second fixture is activated, but its branch flow or activation timing is missing | Event validity | Indeterminate; disturbance schedule is not reproducible |
| PT-NEG-017 | One time-series contains millions of samples but only one physical specimen | Statistical analysis | One specimen-level experimental unit; samples are not independent replicates |
| PT-NEG-018 | Synthetic event profile is presented as observed Gauteng / South African household behavior | Evidence classification | Reject field-prevalence or local-representativeness claim |
| PT-NEG-019 | Sensor calibration receipt is stale, revoked, wrong instrument or wrong range | Metrology admission | Indeterminate / invalid run |
| PT-NEG-020 | A valid pressure trace is used to claim spray comfort, hygiene, accessibility or product compliance without the relevant independent assessment | Claim qualification | Reject the unsupported claim; pressure response alone qualifies none of those domains |

## Minimum implementation sequence

1. Validate the JSON document against the published schema.
2. Validate cross-field semantics: role/datum/unit, profile-kind evidence requirements, event timing, sample monotonicity, channel-specific rates/bandwidth, missing-data intervals and trace identity.
3. Obtain Mycelix metrology admission for calibration, measurement uncertainty, clock synchronization, channel range and trace integrity.
4. Check apparatus rating and pressure envelope before any controlled replay; this is an independent physical authorization gate, not a property of a JSON file.
5. Compare only identical event profiles and method-compatible datums / configurations.
6. Pass service and resource outcomes to the existing independent task assessor and Symtropy closure authority.
7. Preserve invalid, failed, aborted and indeterminate runs in the evidence packet.

## Open implementation questions

- Select a canonical time-series container (for example CSV/Parquet with a strict sidecar manifest) after deciding long-duration versus high-rate storage and archival requirements.
- Decide event ID / waveform digest canonicalization, resampling policy and lossless compression.
- Define channel-specific time-alignment and drift acceptance with Mycelix, rather than hard-coding a universal microsecond threshold.
- Define when an event can be replayed across rigs with different line impedance / branch topology; identical commands do not guarantee identical hydraulic inputs at the fixture.
- Have Symtropy and a competent laboratory define safe transient generation and apparatus authorization. No destructive surge test is authorized by this catalogue.

## Claim ceiling

This catalogue is a proposed software/experiment conformance plan. It contains no measured pressure traces, no validated event generator, no physical test result, no task-completion result, no local pressure profile, and no product safety, compliance or superiority claim.
