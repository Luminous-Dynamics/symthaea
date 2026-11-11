# MARINE-HULL-STATE-EVIDENCE-JOIN-001
## Cross-domain linkage of acoustic, hull condition, cleaning and propulsion observations

**Status:** research specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Implementation:** #11612

> The join contract establishes contextual linkage only. It is not source authentication, metrological qualification, causal inference, fuel-saving evidence, acoustic certification or ecological impact assessment.

## 1. The digital-thread gap

The marine programme now has separate evidence contracts for:
- ISO 17208 underwater acoustic measurements and their hydrophone/calibration context;
- hull-zone inspection, coating identity, fouling status and inspection coverage;
- cleaning authorization, containment, waste disposition and coating damage;
- before/after same-vessel propulsion observations and condition fingerprints.

A narrative report can combine these records without proving they belong to the same physical vessel revision, service contract, trace, operating condition or event sequence. The join layer exists to prevent that.

## 2. Required identity and provenance links

The v0.1.0 join binds:
- exact vessel identity and revision across acoustic and biofouling records;
- exact commercial service fingerprint;
- acoustic record ID and raw-trace digest;
- hull-inspection record ID, inspection ID and raw-data digest;
- cleaning-event ID and start/end times;
- before/after propulsion observation IDs and trace digests;
- acoustic measurement start/end and hull-inspection time;
- a declared maximum separation between acoustic and inspection observations;
- before/after event order around the cleaning intervention.

Every digest must match the referenced record's digest; a matching digest is not proof of sensor authenticity or the truth of the bytes.

## 3. Timing and condition checks

The first audit checks:
1. measurement window start precedes end;
2. hull inspection lies within the declared temporal tolerance of the acoustic window;
3. baseline acoustic measurement completes before cleaning begins;
4. pre-intervention propulsion observation aligns to the acoustic baseline start;
5. post-intervention propulsion observation occurs after cleaning completion;
6. speed-through-water, shaft RPM, shaft power, draft and trim match between acoustic baseline and pre-intervention propulsion record;
7. exact service and vessel identity are retained.

These are v0.1 context rules. A later reviewed version may allow declared condition tolerances or counterfactual profiles, but must never silently widen matching windows to obtain a result.

## 4. Claim semantics

Three separate statements must never be conflated:
- **Schema-valid:** each record conforms to its own declared data contract.
- **Context-linked:** identity, service, digest and timing references resolve across records.
- **Scientifically qualified:** calibration, measurement adequacy, uncertainty, trial design and causal analysis support the requested performance claim.

The join auditor can establish only the second statement after checking the first at the structural schema level. Synthetic/modelled source records must keep the overall result INCOMPLETE even when contextual links resolve. The join cannot upgrade a source record's disposition.

The initial claim mode is fixed to contextual_linkage_only. The report must not output an energy saving, dB reduction, ecological benefit, organism identity or ROI estimate.

## 5. Negative controls

- vessel revision mismatch -> INCOMPARABLE;
- service fingerprint mismatch -> INCOMPARABLE;
- source trace/inspection digest mismatch -> INVALID;
- acoustic and inspection time outside declared tolerance -> INCOMPARABLE;
- cleaning begins before acoustic baseline completes -> INCOMPARABLE;
- post-intervention observation precedes cleaning completion -> INCOMPARABLE;
- acoustic baseline speed/RPM/draft/trim/shaft power differs from the pre-intervention observation -> INCOMPARABLE;
- synthetic or modelled source records -> INCOMPLETE, even when links match.

## 6. Industry comparison and value

PLM/AAS/digital-twin systems can link product configuration and operational records. Specialist acoustic and hydrodynamic standards specify their own measurement procedures. The proposed advantage is a narrow, deterministic cross-domain evidence join that preserves each discipline's source contract, exact raw digest, service identity and time alignment without creating a universal model or pretending that joined records prove causality.

The join should eventually interoperate with class/owner data systems and ship/port AAS/OPC UA interfaces, while retaining source-specific rights, provenance and independent verification.

## 7. Claim ceiling

A context-linked record is not proof that a hull is clean, a hydrophone is calibrated, a cleaning action was authorized, a measurement is valid, or that cleaning caused a fuel/noise change. A causal claim still requires a pre-registered trial protocol, suitable metrology, matched conditions, uncertainty propagation, repeated observations and independent review.
