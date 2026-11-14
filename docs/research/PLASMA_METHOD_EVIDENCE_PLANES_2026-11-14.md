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


## 2026-11-14 — full failure-mode composition is a separate evidence plane

Follow-on implementation: [PLASMA-FAILURE-COMPOSITION-001 #12097](https://github.com/Luminous-Dynamics/symthaea/issues/12097), branch `research/plasma-failure-composition-v0-1`.

A scalar `failure_mode_fraction` can conceal shifts among failure categories. The new S0 receipt preserves the complete ordered category vector per specimen, measurement basis, independent cluster, observer references/reliability, blinded state, D5573 claim scope, preregistered zero handling, per-category uncertainty and joint compositional-distance interval.

The current fixture has 40 known-answer cases and independently replays 40/40 (4 PASS, 15 BLOCKED, 21 REJECTED). A valid study may still return `COMPOSITIONAL_EQUIVALENCE_NOT_ESTABLISHED`. The audit does not decide whether a failure mode is acceptable.

D5573-99R19 is the SMC FRP-specific route and must remain unaided visual classification. Image hashes are traceability artifacts, not a license to substitute magnified image analysis for the standard's observation scope. ISO 10365:2022 is a broader bonded-assembly failure-pattern designation and is being treated as a separate follow-on ontology rather than blended into D5573.


## 2026-11-14 — ISO 10365 is a separate general failure-pattern ontology

The method registry now supports ISO 10365:2022 for bonded assemblies regardless of adherend/adhesive, with its own ontology identifiers: SF, CSF, DF, CF, SCF, SWCF, AF, ACFP and COR. Mixed patterns retain their declared percentage vector.

The failure-composition S0 auditor now accepts either:
- D5573-99R19 with the D5573 SMC-FRP ontology and direct unaided visual observation; or
- ISO 10365:2022 with the ISO 10365 general bonded-assembly ontology and a declared method observation protocol.

The baseline and candidate classifications must use the same exact method, edition, evidence kind, material class, geometry, endpoint, claim and ontology. Cross-ontology comparison is REJECTED until a separately reviewed mapping receipt is implemented. No silent category mapping is allowed.

Updated frozen fixtures: claim scope v0.4 (59 cases), failure composition v0.2 (47 cases). Independent exact-branch replay: 59/59 and 47/47.


## 2026-11-14 — profile-specific typed claim contract (#12105)

The current transfer evidence model had a global required endpoint list that applied the same outcomes to every material profile. This is too coarse. A PP rigid-plastic single-lap profile, an FRP-to-FRP profile, an SMC-FRP classification profile and an aluminum thick-adherend constitutive-response profile do not require the same methods or evidence types.

New S0 branch: `research/plasma-profile-contract-v0-1`.

Implementation:
- `xtask/src/plasma_profile_contract.rs`
- `data/research/plasma_profile_contract_fixture_v0_1.json`
- command: `cargo run -p xtask -- plasma-profile-contract-audit`

The profile contract binds a canonical SHA-256 to profile identity, material pair, joint geometry, intended use and a frozen list of required/optional claim families. Each claim names explicit method-evidence roles (method ID, exact edition, evidence kind, material class, geometry, endpoint and requested claim). The audit composes the shared method-to-claim registry and rejects a role whose scope does not match the typed claim family.

Initial claim families:
- `joint_strength`: admitted joint-strength test methods only, with method scope matching the profile's material pair and joint geometry;
- `failure_pattern`: ISO 10365 general failure-pattern or D5573 SMC-FRP classification only;
- `adhesive_constitutive_response`: D5656 only, on its aluminum thick-adherend scope;
- `surface_state`: wettability observation under its own protocol, never a joint-strength proxy;
- `conditioned_measurement`: separate conditioning-practice and post-conditioning measurement receipts tied to the same specimen IDs.

The fixture includes **27 synthetic cases** (8 PASS, 3 BLOCKED, 16 REJECTED), including:
- PP profile with required D3163 strength and optional ISO failure classification omitted without blocking;
- FRP-to-FRP profile using D5868 and ISO 10365;
- D5573 accepted only for the SMC-FRP-specific profile;
- D5656 accepted only as adhesive constitutive response;
- aged-strength evidence requiring both a conditioning protocol and a post-condition strength test on the same specimens;
- wrong method for material/geometry, failure-classification/strength proxy substitutions, missing required claims, optional claims, undeclared roles/claims, duplicate evidence, duplicate raw artifact, and profile digest mismatch.

Latest source and fixture were fetched from the exact branch and independently replayed in JavaScript: **27/27** expected dispositions matched, including recomputation of the canonical profile SHA-256 over the exact serialized profile content.

This is a profile-contract audit, not yet a replacement of the old global required-endpoint list inside the legacy #12006 transfer command. The next integration step is to make the transfer decision consume this profile receipt while preserving its existing bounded-scope, individual bridge and multi-factor interaction gates.

Validation boundary: seven Rust tests are authored but not executed; Rust formatting, compilation, Clippy and exact-head CI are not verified. No physical joint strength, transfer equivalence, aging performance, product acceptance or production release is claimed.


### 2026-11-14 follow-on — both sides of transfer must satisfy the profile

The profile receipt has been advanced from one evidence list to explicit `baseline_evidence` and `candidate_evidence` sets. Both cohorts are checked against the same frozen profile claim matrix. The audit now requires:
- all required roles present on both sides;
- optional claim roles either absent on both sides or supplied symmetrically;
- identical method ID, edition, evidence kind, material scope, geometry, endpoint, claim, protocol ID and protocol digest;
- distinct baseline/candidate evidence IDs and raw-data artifacts;
- non-overlapping specimen IDs across cohorts;
- for conditioned measurements, the conditioning receipt and post-condition strength receipt bind the same specimens within each cohort.

The v0.2 fixture contains **32 cases** (8 PASS, 4 BLOCKED, 20 REJECTED) and independently replays **32/32**, including candidate-side omissions and one-sided optional evidence. Eight Rust tests are authored but remain unexecuted.

The contract is now transfer-oriented, but the legacy #12006 audit still needs to consume it directly while preserving its existing bridge, interval and multi-factor interaction checks.
