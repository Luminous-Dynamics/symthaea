# CP-01 Metrology / Characterization Contract v1

Status: architecture contract / preregistration.

**Claim ceiling:** this contract establishes only software semantics for binding a claim to a declared measurand, physical/configuration subject, observation, calibration/currentness, uncertainty, and traceability references. It establishes no instrument accuracy, calibration validity, physical capability, safety, certification, or execution authority.

## 1. Purpose

CP-01 is the cross-platform metrology waist for Symthaea engineering and CIV infrastructure.

It composes existing owners rather than creating a second observation, calibration, physical-article, quantity, or authority system.

The contract answers:

> What exact physical claim is being measured, on which subject and configuration, under which measurement context, with which observation and calibration lineage, and what remains unresolved?

It does **not** answer whether the instrument is actually accurate or whether the measured property satisfies an engineering requirement. Those remain evidence questions owned by the appropriate metrology/observation/engineering systems.

## 2. Canonical ownership

CP-01 is a composition contract only.

| Concern | Canonical owner / predecessor | CP-01 role |
|---|---|---|
| physical article / as-built identity | ROB-REALIZE / SEMI-EQP as applicable | reference |
| quantity, unit, datum, frame | SE-SEM | reference |
| physical observation and raw/derived lineage | FIELD / SE-OBS | reference |
| calibration identity, epoch, currentness | calibration/currentness owners such as #4620 | reference |
| commissioning relation | #4623 where applicable | reference |
| measurement-plan semantics | ENG-MEAS / domain owner | reference |
| requirement / verification obligation | ENG-DESIGN / SE-VV | reference |
| custody / session provenance | SENSE / operational provenance owners | reference |
| engineering interpretation | relevant engineering owner | reference |
| operational event / asset state | Mycelix / CIV-Service where applicable | reference |

CP-01 MUST NOT introduce replacement `PhysicalArticleId`, `ObservationId`, `CalibrationId`, unit ontology, commissioning authority, or operational authority.

## 3. Stable contract

A claim-bearing measurement subject is represented conceptually as:

```
MetrologySubjectRefV1 {
  subject_id,
  subject_generation,
  measurand_ref,
  measurement_profile_ref,
  physical_article_ref,
  as_built_configuration_ref,
  instrument_ref,
  instrument_configuration_ref,
  calibration_ref,
  observation_ref,
  uncertainty_ref,
  datum_frame_ref,
  reference_artifact_ref?,
  campaign_ref,
  applicability_ref,
  traceability_refs[],
  dependency_refs[],
  currentness_ref,
  authority_disposition,
  claim_ceiling,
}
```

The payload contains references to canonical records. It does not copy authoritative observation or calibration payloads into a new CP-01 database.

### Required distinctions

```
measurand
!= measurement method
!= instrument
!= calibrated instrument state
!= observation
!= uncertainty statement
!= engineering interpretation
!= requirement satisfaction
!= operational authority
```

Likewise:

```
sensor exists
!= measurand observable
!= measurement capable
!= calibration current
!= traceable
!= uncertainty sufficient
!= requirement satisfied
```

## 4. Identity and generation law

The semantic identity of a claim-bearing measurement context changes when an evidence-relevant dependency changes.

At minimum, a new subject/configuration context is required when applicable:

- physical article identity changes;
- as-built configuration changes;
- instrument identity or evidence-relevant configuration changes;
- material optics/fixture/mounting context changes;
- acquisition firmware/parser/profile changes;
- calibration epoch changes where the measurement profile requires rebinding;
- datum/reference-frame identity changes;
- reference artifact identity/currentness changes;
- measurement method/profile applicability changes.

Historical observations remain attached to the context under which they were acquired.

A new context does not rewrite historical observations. It creates a new derived applicability/currentness decision.

## 5. Evidence planes

Keep these planes independently addressable:

1. **Intent** — what claim or requirement is being investigated.
2. **Measurand** — what physical quantity/state/event is intended.
3. **Subject** — exact article/configuration being observed.
4. **Method** — declared measurement/inspection method and applicability.
5. **Instrument** — exact instrument and configuration.
6. **Calibration** — calibration identity, epoch, status and traceability.
7. **Observation** — raw/derived observation lineage.
8. **Uncertainty** — uncertainty model/statement and assumptions.
9. **Reference** — datum/frame/reference-artifact dependencies.
10. **Interpretation** — engineering derivation from the observation.
11. **Decision** — requirement/qualification disposition.
12. **Authority** — any external operational/certification authority.

No later plane may be synthesized merely because an earlier plane exists.

Examples:

```
calibration current
-> observation validity context

NOT

calibration current
-> instrument accurate for every measurand
```

and:

```
observed value
-> evidence about the observed context

NOT

one observation
-> population distribution
```

## 6. Applicability and currentness

Every claim-bearing subject must preserve an explicit applicability/currentness disposition.

Minimum dispositions:

- `CurrentAndApplicable`
- `CurrentButApplicabilityUnresolved`
- `Stale`
- `ConfigurationMismatch`
- `CalibrationStale`
- `ReferenceMismatch`
- `ObservationMissing`
- `UncertaintyInsufficient`
- `TraceabilityIncomplete`
- `Unknown`

A stale or mismatched dependency narrows the claim; it must not silently become a PASS through recomputation.

## 7. Uncertainty and observability

CP-01 records uncertainty as a first-class dependency.

The contract must distinguish:

```
uncertainty model exists
!= uncertainty adequate for decision
```

A measurement may be numerically precise yet insufficient for a declared engineering threshold if the decision margin is not resolved.

The contract also preserves the difference between:

```
feature present
!= feature observable
!= observable with sufficient coverage
!= observable with sufficient uncertainty
```

Design-for-knowability concerns belong to ENG-DESIGN / #6124; CP-01 consumes those affordances and later physical evidence rather than replacing them.

## 8. Independence and common-mode dependencies

When multiple observations are used as corroborating evidence, CP-01 MUST preserve their dependency references.

Agreement between two channels is not independent corroboration when they share a declared common-mode root such as:

- one calibration reference;
- one ADC/reference chain;
- one encoder transform;
- one fixture datum;
- one fitted parameter source;
- one acquisition parser;
- one environmental correction.

```
numeric agreement
!= independent evidence
```

The qualifier may establish faithful dependency semantics; it does not infer physical independence from labels.

## 9. Back-action

Measurement can alter the subject.

Where material to the declared decision, the measurement context may reference back-action classes including:

- mechanical loading/compliance;
- added mass/inertia;
- thermal loading;
- electrical loading;
- optical illumination;
- airflow or fluid disturbance;
- cable/fixture forces;
- changed enclosure/sealing state;
- acquisition timing/software load.

```
measurement available
!= measurement non-perturbing
```

Back-action evidence remains an explicit dependency rather than an implicit quality flag.

## 10. Traceability

Traceability is a graph of references, not a prose assertion.

A claim-bearing path should be reconstructable as:

```
claim
 -> measurand
 -> subject/configuration
 -> method
 -> instrument/configuration
 -> calibration
 -> reference/datum
 -> observation
 -> uncertainty
 -> interpretation
 -> decision
```

A missing edge produces an explicit unresolved state.

A traceability receipt must identify the exact referenced generations used for the decision. It must not silently follow mutable "latest" aliases.

## 11. Negative evidence

Negative and inconclusive observations remain first-class.

Examples:

- instrument could not resolve the threshold;
- calibration transfer was not demonstrated;
- reference target was unavailable;
- fixture drift exceeded the declared bound;
- observation contradicted the model;
- required measurement coverage was absent.

```
negative evidence
!= no evidence
```

Recomputation may change a derived disposition, but it MUST NOT delete or overwrite the historical negative record.

## 12. Cross-platform projection

### Materials

Materials convergence may consume CP-01 observations as measurement references. A measured property does not become a material truth independent of specimen, process, interface and lifecycle context.

### Manufacturing

MFG-PROC capability and as-built records may reference CP-01 measurement evidence. Inspection planned/executed/result/accepted remain distinct.

### Compute / sensing

Sensor or compute state may be an instrument dependency, but operational telemetry is not automatically claim-bearing metrology.

### CIV-Service

A service event may reference qualified measurement state for asset/service currentness.

```
service event
!= measurement evidence
```

### Mycelix

Mycelix may preserve custody, identity, coordination and operational provenance around a measurement campaign.

```
provenance/attestation
!= engineering truth
```

## 13. Minimal synthetic qualification corpus

A future CP-01 independent qualifier should freeze at least these cases:

1. exact article + configuration + current calibration + observation -> complete trace;
2. design exists but physical subject identity is missing -> unresolved;
3. friendly instrument name without exact instance identity -> insufficient;
4. same design instantiated as two physical articles -> distinct subjects;
5. changed fixture/mount -> new measurement context;
6. changed acquisition parser/firmware -> new acquisition context;
7. stale calibration -> claim narrowed;
8. reference-target mismatch -> reject;
9. observation present but uncertainty missing -> unresolved;
10. two agreeing channels share a calibration root -> not independent;
11. one specimen observation -> no population inference;
12. measurement feature exists but cannot resolve required threshold -> insufficient capability;
13. back-action exceeds declared decision assumption -> applicability unresolved;
14. negative observation remains addressable after recomputation;
15. operational event references measurement but cannot create it;
16. synthetic PASS establishes zero physical execution authority.

The qualifier must derive these outcomes independently of any production implementation.

## 14. Qualification pattern

CP-01 follows the common platform qualification train:

```
architecture contract
 -> frozen synthetic corpus
 -> independent stdlib/reference oracle
 -> adversarial mutations
 -> deterministic replay
 -> qualification receipt
 -> exact-head hosted provenance
 -> production adapter
```

The independent oracle should validate at least:

- schema identity;
- exact case manifest and digest;
- canonical identity/generation behavior;
- dependency closure;
- currentness/applicability;
- uncertainty and traceability requirements;
- negative-evidence retention;
- common-mode dependency preservation;
- authority ceiling;
- deterministic replay;
- source immutability.

A repository PASS proves only the synthetic software semantics named in the claim ceiling.

## 15. Production adapter gate

No CP-01 production adapter should be treated as claim-bearing until:

1. the canonical observation and calibration owners are identified;
2. the exact subject/configuration generation is bound;
3. the measurement method/profile is resolvable;
4. traceability edges are explicit;
5. uncertainty/currentness are represented;
6. negative evidence survives recomputation;
7. the independent qualifier has an exact source snapshot;
8. the qualification receipt is reproducible;
9. hosted provenance identifies that exact source snapshot;
10. any physical execution remains behind existing equipment/HAL/operator authority.

## 16. Review gates

Reviewers must be able to answer:

1. What exact measurand is claimed?
2. What physical subject and generation was observed?
3. Which method/profile applies?
4. Which instrument/configuration was used?
5. Which calibration generation was current?
6. What reference/datum dependencies exist?
7. What uncertainty supports the decision?
8. What common-mode dependencies exist?
9. What negative evidence exists?
10. Which change would force rebinding?
11. What does the software PASS prove?
12. What physical claim remains external?

## 17. Existing-lineage reuse

CP-01 intentionally reuses the existing metrology/equipment architecture rather than opening a duplicate generic measurement stack.

In particular, the existing SEMI-EQP-MET-001E2 exact bench-subject work (#5941/#5942) and its independent corpus validator (#5943/#5944) provide a concrete reference for exact physical/as-built/configuration binding.

The broader metrology contract must remain domain-neutral; semiconductor bench semantics stay owned by their existing SEMI-EQP owners.

## 18. Nonclaims

This contract does not establish:

- instrument accuracy or repeatability;
- calibration validity;
- physical measurement capability;
- population statistics from sparse observations;
- material, component or system qualification;
- safety or regulatory approval;
- production capability;
- operational service sufficiency;
- physical execution authority.

Its purpose is to make those claims **decomposable, traceable, generation-bound, and explicitly qualified** rather than allowing them to arise accidentally from a measurement-shaped record.
