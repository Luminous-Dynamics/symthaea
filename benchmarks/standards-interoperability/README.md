# MBD-INTEROP-001 — standards-based engineering semantic round trips

**Tracking issue:** [#10595](https://github.com/Luminous-Dynamics/symthaea/issues/10595)  
**Status:** research specification; fixtures not yet materialized; no benchmark score  
**Authority:** evaluation-only. This document does not authorize production adapters, fabrication, machine control, evidence admission, or physical testing.

## Objective

Measure whether engineering meaning survives exchange across independent product-definition, inspection/quality, and manufacturing-device systems.

The benchmark is deliberately not a CAD kernel, geometry authority, QIF/metrology authority, device telemetry authority, lifecycle graph, provenance database, or evidence/currentness engine. It tests adapters and toolchains against explicit semantic contracts owned elsewhere in Symthaea.

The central distinction is:

```
file parses
  != schema/profile conformance
  != semantic preservation
  != engineering correctness
  != evidence applicability
  != product qualification
```

A successful import/export is not sufficient. The evaluator must detect when a critical relationship has been dropped, changed, weakened, or rebound to the wrong design generation.

## Pinned standard baselines

| Surface | Baseline for this research profile | Use |
|---|---|---|
| Product definition | STEP AP242, ISO 10303-242:2025, Edition 4 | Geometry, product structure, assemblies, semantic PMI, supported electrical/product data |
| Quality/inspection | QIF, ISO 23952:2020 | Characteristics, inspection plans/results and quality workflow information |
| Manufacturing devices | MTConnect versioned profile (version matrix unresolved) | Device identity, data items, units, time and process context |
| Traceability principles | NIST IR 8536 (September 2026) | Interoperable product pedigree, verifiable linkage and selective disclosure principles |

References:
- [ISO 10303-242:2025, Edition 4](https://www.iso.org/standard/84300.html)
- [ISO 23952:2020 — Quality Information Framework](https://committee.iso.org/standard/77461.html)
- [MTConnect standard-download page (labels 2.5 current)](https://mtconnect.squarespace.com/standard-download20181); [normative SysML model releases (v2.7)](https://github.com/mtconnect/mtconnect_sysml_model/releases); [normative model documentation (v2.8 page)](https://model.mtconnect.org/Version2.8/); [generated XML schemas (v2.7)](https://schemas.mtconnect.org/)
- [NIST IR 8536 final](https://csrc.nist.gov/pubs/ir/8536/final)
- [NIST Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing)

**QIF identifier correction:** use ISO 23952:2020. One NIST project-page sentence currently labels QIF as ISO 23953; the ISO standard record identifies QIF as ISO 23952.

Standards and schemas must be pinned by edition, schema/profile digest, and exact implementation version for every run. “STEP”, “QIF”, or “MTConnect” without a version/profile is not a reproducible test declaration. STEP AP242 Edition 5 is under development (committee draft); the initial AP242 profile remains Edition 4, the latest published International Standard. QIF ISO 23952:2020 was reviewed and confirmed current on 14 June 2026. For MTConnect, pin the normative XMI/model revision, generated XSD bundle, agent/server implementation, protocol behavior, and implementer-license/profile separately. The 2.5 / 2.7 / 2.8 public-surface mismatch is an unresolved compatibility question; no MTConnect adapter may be scored until the exact tuple is recorded.

## Ownership and composition

| Existing owner | What this benchmark consumes |
|---|---|
| Systems Engineering [#3677](https://github.com/Luminous-Dynamics/symthaea/issues/3677) | Read-only standards projections and semantic graph references |
| Geometry [#4927](https://github.com/Luminous-Dynamics/symthaea/issues/4927) and cross-kernel benchmark [#10128](https://github.com/Luminous-Dynamics/symthaea/issues/10128) | Exact geometry subjects and independent geometric checks |
| CAD history [#10510](https://github.com/Luminous-Dynamics/symthaea/issues/10510) | Editable design intent and feature references |
| Assembly/ECAD [#10530](https://github.com/Luminous-Dynamics/symthaea/issues/10530) | Component identity, transforms, mating and net connectivity |
| Component catalogue [#4877](https://github.com/Luminous-Dynamics/symthaea/issues/4877) | Exact component, variant, supplier and document references |
| Tolerance/metrology [#4883](https://github.com/Luminous-Dynamics/symthaea/issues/4883) and measurement [#4919](https://github.com/Luminous-Dynamics/symthaea/issues/4919) | Characteristics, measurement equations, uncertainty and conformance semantics |
| Realization [#4859](https://github.com/Luminous-Dynamics/symthaea/issues/4859) | Fabrication, article, as-built and measurement lineage |
| Manufacturing process [#5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) | Process, machine and execution context |
| Change impact [#10543](https://github.com/Luminous-Dynamics/symthaea/issues/10543) | Claim/evidence applicability after a semantic change |
| Model–Fixture–Article [#6121](https://github.com/Luminous-Dynamics/symthaea/issues/6121) | Exact reference model, article, fixture and campaign identities |

ETK remains the owner of evidence admission, currentness/applicability, and downstream authority. Mycelix remains the owner of cross-organization custody, work orders, lots and shared production provenance where applicable. This benchmark must not reproduce either system.

## Initial semantic profile

### 1. STEP AP242 product definition

Start with a small declared subset:

- Units and dimensional quantity semantics.
- Coordinate systems, datums, reference frames and handedness.
- B-rep and tessellated representations, each tagged with its representation class.
- Part/product identity, assembly structure and instance transforms.
- Semantic PMI: dimensions, tolerances, datum references and associations to exact features.
- Graphical/presentation PMI kept distinct from machine-interpretable semantic PMI.
- Material, process and external document references only where their provenance and meaning survive the exchange.

A graphical dimension that looks correct is not a substitute for semantic PMI. Unsupported entities must remain explicitly unsupported; they must not be silently defaulted, dropped, or promoted.

### 2. QIF inspection and quality information

The first profile must preserve the relationship among:

- Exact design characteristic and feature reference.
- Exact design/configuration generation.
- Nominal value, tolerance zone/limits, quantity kind and units.
- Datum and measurement frame.
- Inspection plan, method and equipment.
- Raw observation and separately derived measurand.
- Measurement equation, uncertainty and common-mode/dependence references where relevant.
- Calibration identity/currentness reference.
- Exact physical article, lot and inspection event.
- Result, exclusions, deviations and claim limitations.

A result attached to a friendly feature name but not to the exact feature and generation is not a successful mapping.

### 3. MTConnect manufacturing context

The version tuple is intentionally unresolved in v0.1. Before execution, freeze: normative XMI digest + version, generated XSD bundle digest + version, exact agent/server build, supported protocol behavior, and the implementer-license/policy state. The public surfaces currently disagree on which version label is the latest release; do not infer compatibility from version numbers alone.

The initial device profile should retain:

- Exact device/asset and adapter identity.
- Data-item identity, units and source quality.
- Timebase, timestamp and event ordering.
- Process/job/operation association where supplied.
- Missing, stale, invalid and out-of-range data states.
- Links to exact lot/article and process-execution references.

Device telemetry is contextual process evidence. It does not independently establish geometry, conformance, process qualification, or functional performance.

## Runner-independent fixture plan

Freeze the first 6–10 cases before implementing a candidate translator.

### Positive fixtures

1. **PMI part:** one part with linear/diametric dimensions, datum references and semantic PMI attached to exact features.
2. **Assembly:** a small assembly with multiple component variants, external IDs, reference frames and exact instance transforms.
3. **Representation pair:** equivalent B-rep and tessellated representations with explicit approximation/tolerance declarations.
4. **Inspection plan:** QIF characteristics linked to the AP242 PMI and exact design generation.
5. **Inspection result:** measurements linked to an exact article, instrument/calibration reference, and uncertainty-bearing measurand.
6. **Manufacturing event:** one MTConnect device/process event linked to a job, lot and as-built article.
7. **Trace chain:** one bounded AP242 → QIF plan/result → manufacturing-event context → design feedback chain.

### Critical negative and mutation controls

- Semantic PMI downgraded to graphics-only presentation.
- Datum or feature association dropped or reassigned.
- Unit conversion, axis inversion, handedness, origin or frame error.
- Duplicate, colliding, or unstable external identifier.
- Component instance moved, mirrored or rebound to an incompatible variant.
- QIF result attached to the wrong characteristic or design generation.
- Nominal/tolerance changed while rendered annotation remains visually plausible.
- Measurement detached from calibration or uncertainty context.
- Device timestamp/timebase drift or asset-identity substitution.
- Partial/unsupported import silently accepted as complete.
- Source revision, content digest, signature or provenance link mismatch.
- Unsupported extension converted to zero/default instead of Unknown/Unsupported.

Each mutation must have an independent expected disposition. A control is useful only if the evaluator can show that it detects its planted defect without falsely rejecting the corresponding valid control.

## Independent evaluation contract

For each case freeze:

- Input artifact bytes and cryptographic digests.
- Source subject and generation identities.
- Standard edition, schema/profile digests and implementation versions.
- Expected semantic entities and typed relationships.
- Permitted transformations and declared lossy fields.
- Unsupported/unknown expectations.
- Evaluator-only expected results and mutation labels.
- Run configuration, resource limits and deterministic replay information.

Do not allow the candidate adapter to generate its own gold labels. Keep candidate workspaces separate from evaluator-only material. Use independent references for geometry and semantic mapping wherever feasible.

The first corpus should be public and fully inspectable. Later family-held-out variants should be generated from frozen source fixtures and kept outside candidate indexes, agent memory, retrieval stores and training/evaluation feedback.

## Metrics — report separately

1. Syntax/schema validity.
2. Declared standard/profile coverage.
3. Semantic field and relationship preservation.
4. PMI-to-feature association precision and recall.
5. Geometry/topology difference within declared tolerances.
6. Assembly identity, structure and transform preservation.
7. QIF characteristic, result, article and calibration lineage.
8. MTConnect device, data-item, timestamp and job/lot linkage.
9. Unsupported/lossy-field detection and correct Unknown behavior.
10. False-current and stale-evidence disposition errors.
11. Determinism, latency, memory and resource consumption.

Critical false passes are non-compensatory. A high mean score cannot hide a lost critical tolerance, wrong assembly transform, or inspection result attached to the wrong physical article.

## Semantic mapping matrix

`mapping-matrix.csv` provides the first 24-row conceptual mapping register. It names the canonical owner, criticality, fail/Unknown behavior, and an independent check for each semantic contract. These are conceptual mappings, not literal AP242/QIF/MTConnect XML/EXPRESS element paths. Bind each row to exact schema paths and tool-specific representations only after pinning the implementation/profile and independently reviewing the mapping.

## Initial acceptance gates (provisional, not frozen)

- Zero false passes on authored critical negative controls.
- 100% preservation of required critical identifiers and semantic associations on accepted cases.
- 100% explicit reporting of unsupported and lossy semantics.
- Deterministic replay of the frozen fixtures.
- Independent adjudication of every critical mapping and mutation.
- No evidence-currentness or engineering-approval authority minted by a parser, adapter, or benchmark receipt.

Do not freeze numeric thresholds until the gold protocol and pilot corpus have been independently reviewed. Report confidence intervals and per-family denominators; do not substitute aggregate averages for critical-case coverage.

## Staging

**M0 — Specification:** owner review, exact standard/profile pinning, semantic mapping matrix, claim ceilings.

**M1 — Fixtures:** 6–10 deterministic cases, critical positive/negative pairs, independent expected results.

**M2 — One exchange path:** a pinned AP242 ↔ QIF workflow with explicit MTConnect references but no live device connection.

**M3 — Independent implementation:** add a second translator/inspection implementation and cross-toolchain round trips.

**M4 — Held-out evaluation:** family-held-out and adversarial mutations, model × harness × tool matched runs, published per-case receipts and costs.

**M5 — Bounded physical campaign:** only as a child of existing Model–Fixture–Article and realization owners, once all required physical, calibration, safety and authority gates have separately qualified.

## Claim ceiling

A passing report establishes only that the pinned implementations preserved the declared semantic subset on the exact tested corpus and configuration. It does not establish full AP242/QIF/MTConnect certification, general interoperability, product safety, structural validity, code compliance, supplier qualification, process capability, physical conformity, or authorization to manufacture or operate.


## MTConnect release / compatibility gate (machine-readable)

**MTConnect is not yet eligible for execution or scoring in MBD-INTEROP-001.** At the current retrieval, public version labels do not identify one self-consistent compatibility tuple:

| Surface | Public label observed | What this does *not* prove |
|---|---|---|
| Official standard-download page | 2.5 | That the latest normative XMI, generated XSD and a given agent/adaptor all implement this release |
| Normative SysML model release | 2.7 (release dated 2026-02-27) | That a given downloadable XMI and published schema bundle have matching semantics |
| Normative model documentation | 2.8 documentation site | That the docs site release label corresponds to the published XSD bundle or a released agent |
| XML schema index | 2.7 XSD files | That the selected agent/protocol implementation is compatible with those files |

Sources: [official standard download](https://mtconnect.squarespace.com/standard-download20181), [SysML model releases](https://github.com/mtconnect/mtconnect_sysml_model/releases), [model documentation](https://model.mtconnect.org/Version2.8/), [schema index](https://schemas.mtconnect.org/).

The machine-readable `mtconnect_execution_gate` therefore locks both `eligible_for_candidate_execution` and `eligible_for_scoring` to false. Before enabling MTConnect, freeze the exact normative model/XMI digest, XSD bundle digests, agent/adaptor/protocol versions, capability profile and validator version, and provide an independent compatibility witness. Any mismatch stays Unknown/Unsupported.

**Phase boundary:** proceed with AP242 Edition 4 ↔ QIF / ISO 23952:2020 only for the initial executable tranche. MTConnect manufacturing-event cases remain public blueprint placeholders until the complete compatibility tuple is independently reviewed. There is no live device connection in the first tranche.
