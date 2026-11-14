# PLASMA-METHOD-ONTOLOGY-002 — three distinct material evidence planes

**Date:** 2026-11-14  
**Issue:** [#12072](https://github.com/Luminous-Dynamics/symthaea/issues/12072)  
**Status:** S0 method/claim registry extension  
**Stack:** #12028 → #12051 → #12070 → this branch

## Research finding

A plasma-assisted bond qualification Passport needs to distinguish three evidence planes that are related but not interchangeable:

1. **Adhesive constitutive response** — how the adhesive material responds mechanically (for example, shear stress versus shear strain).
2. **Joint-level adhesion / strength** — what a declared adhesive + adherend + geometry assembly sustains under a specific test.
3. **Failure-mode classification** — how the joint failed, described under a classification practice. The label does not by itself say whether the failure is acceptable.

Surface wettability is a fourth independent observation plane, not a substitute for any of the three.

A single scalar `bond_quality` obscures these distinctions and invites invalid inference.

## Standards evidence

### ASTM D5656/D5656M-26 — adhesive constitutive response, not substrate adhesion

ASTM's active 2026 edition is *Standard Test Method for Thick-Adherend Metal Lap-Shear Joints for Determination of the Stress-Strain Behavior of Adhesives in Shear by Tension Loading*. It covers preparation and testing of thick-adherend lap-shear samples and data reduction of stress-strain curves. ASTM states that it can determine adhesive shear stress-strain properties, proportional limit and shear strain at failure, and explicitly says that it is **not intended to determine adhesion characteristics of an adhesive to a particular substrate**. ASTM further says it has been developed and applied using bonded aluminum adherends and that validity for non-aluminum adherends cannot be assumed.

Official source: https://store.astm.org/d5656_d5656m-26.html

The registry therefore pins:
- method ID: `ASTM_D5656_D5656M`;
- edition: `D5656/D5656M-26`;
- evidence kind: `measurement`;
- material pair class: `aluminum_thick_adherend_joint`;
- geometry: `thick_adherend_lap`;
- endpoint: `adhesive_shear_stress_strain_response`;
- claim: `adhesive_constitutive_response_observation`.

Its ceiling is adhesive shear stress-strain / proportional-limit / shear-strain-at-failure characterization for the stated aluminum thick-adherend specimen. It cannot establish substrate-specific adhesion, D3163/D5868 apparent single-lap strength, structural design allowables or product qualification.

### ASTM D5573-99R19 — FRP failure-mode classification, not acceptance

ASTM's active D5573-99R19 is a practice for classifying, identifying and characterizing failure modes in adhesively bonded FRP joints. Its stated development scope is glass-fibre / thermoset-polyester SMC. It may be used to describe failure modes observed using test methods such as D3163 and D5868. Crucially, ASTM expressly states that the practice **does not address the acceptability of any specific failure mode**.

Official source: https://store.astm.org/standards/d5573

The registry therefore pins:
- method ID: `ASTM_D5573`;
- edition: `D5573-99R19`;
- evidence kind: `classification_practice`;
- material pair class: `frp_smc_joint`;
- geometry: `failure_surface_classification`;
- endpoint: `frp_joint_failure_mode_classification`;
- claim: `failure_mode_classification_reference`.

Its ceiling is descriptive failure-mode classification for that scoped SMC FRP joint family. It cannot establish that a failure mode is acceptable, define an acceptance criterion, determine product disposition, establish strength or authorize safe use. Generalized CFRP/FRP application is BLOCKED pending a separate applicability bridge.

### Plasma surface state is not bond strength

A 2026 study comparing argon and nitrogen plasma surface treatment across PP, HIPS, PVC, PC and PET measured water contact angle separately from adhesive lap-shear performance. It reports that the lowest water contact angle did not always correspond to the highest bond strength; polymer, plasma gas, adhesive chemistry and process window all matter.

Source: https://link.springer.com/article/10.1007/s44245-026-00396-4

This supports maintaining a distinct `wettability_observation` claim and rejecting contact-angle-to-strength promotion. It does not establish a universal contact-angle/bond-strength relationship or replace product-specific testing.

## S0 registry changes

The shared registry now distinguishes:
- `measurement`;
- `preparation_practice`;
- `conditioning_practice`;
- `interpretation_guide`;
- `classification_practice`.

The D5656 and D5573 entries are exact-edition and endpoint-specific. Special handling blocks D5656 on non-aluminum adherends and D5573 outside the stated SMC FRP family unless an independently reviewed scope bridge exists. The general claim-overreach set also rejects failure-mode acceptability and constitutive-response-to-bond-strength escalation.

This is a versioned research registry. The method names, editions and claim ceilings must still be checked against controlled standards text and the exact specimen/procedure selected by the accountable laboratory.

## Frozen fixture and benchmark

Fixture: `data/research/plasma_claim_scope_fixture_v0_3.json`  
Command:

```bash
cargo run -p xtask -- plasma-claim-scope-audit
```

The v0.3 fixture extends the existing 38 cases to **52 synthetic known-answer cases**:

| Disposition | Cases |
|---|---:|
| SUPPORTED | 13 |
| BLOCKED | 12 |
| REJECTED | 27 |
| **Total** | **52** |

New cases cover:
- valid D5656/D5656M-26 aluminum thick-adherend constitutive-response measurement;
- D5656 promoted to substrate adhesion or bond strength;
- D5656 substituted for a D3163/D5868 single-lap endpoint;
- D5656 applied to non-aluminum adherends or an old/unknown edition;
- D5656 evidence-kind and geometry mismatch;
- valid D5573 SMC FRP failure-mode classification;
- D5573 promoted to failure-mode acceptability or lap-shear strength;
- D5573 generalized to FRP outside the scoped SMC material class;
- D5573 evidence-kind and edition mismatch.

Fetched the latest claim-scope source and fixture back from the exact branch and independently replayed the method/evidence/claim rules in JavaScript: **52/52** expected dispositions matched (13 SUPPORTED, 12 BLOCKED, 27 REJECTED).

Prospective benchmark targets:
- zero D5656-as-substrate-adhesion false PASS;
- zero D5656-as-single-lap-strength false PASS;
- zero D5573-as-acceptance false PASS;
- zero contact-angle-to-strength false PASS;
- 100% detection of old/unknown edition and unsupported material-scope cases;
- separate reports for valid observation, blocked applicability and prohibited claim escalation.

This 52-case regression corpus is not a statistically powered evaluation of physical plasma treatment or laboratory execution.

## Paradigm shift

A bond qualification record should be a **typed evidence bundle**, not an averaged quality score:

`surface state`
→ `adhesive constitutive response`
→ `joint-level strength / fracture response`
→ `failure-mode classification`
→ `environmental aging / retained performance`
→ `design-specific inference`

The arrows are proposed evidence dependencies, not automatic mathematical implications. Each downstream inference must cite its exact upstream observations, specimen and configuration scope, uncertainty and validity. An excellent wetting value cannot override a poor strength result; a constitutive shear curve cannot stand in for interfacial adhesion; and a failure-mode label cannot silently become an acceptance decision.

## Ownership boundaries

- #12026 / #12028 owns the shared exact-edition claim-scope evaluator and versioned method registry.
- #12033 / #12051 owns method-qualified endpoint composition.
- #12064 / #12070 owns cluster-aware design and result linkage.
- #12002 owns qualification transfer.
- #11971 owns aging and survival evidence.
- #6414 owns generic evidence-DAG and invalidation semantics.
- Statistics owns numerical uncertainty, TOST and model validity.

No new fracture solver, constitutive model, strength estimator, QMS, materials registry or statistical engine is introduced.

## Validation boundary

- Independent synthetic fixture replay: **52/52**.
- Eight Rust unit tests are authored but **not executed**.
- Rust formatting, compilation, Clippy and exact-head hosted CI remain unverified.
- Digest checks are syntactic; raw data, standard text, specimen identities and laboratory execution are not authenticated here.
- No adhesive mechanical property, substrate adhesion, failure acceptability, product performance, design allowable, service life, material equivalence or product release is established.
