# Body Armour Material-Demand Profile v0.1

**Programme:** BODY-ARMOR-MATERIAL-DEMAND-001 (#12332)  
**Related product/system programme:** [Symtropy BODY-ARMOR-001 #5377](https://github.com/Luminous-Dynamics/symtropy/issues/5377)  
**Related wearer engineering:** [APPAREL-ENG-TWIN-001 #11731](https://github.com/Luminous-Dynamics/symthaea/issues/11731)  
**Snapshot:** 2026-11-15  
**Status:** research profile and integration contract; not a canonical new ontology

## 1. Purpose

Translate a bounded human-worn torso-protection engineering demand into the existing Symthaea materials-discovery pipeline without creating another material ontology, requirement ontology, optimizer, uncertainty kernel, campaign scheduler or evidence authority.

The canonical semantic owners remain:

- MAT-ENG-001 #6185 — engineering demand and qualification bridge;
- MAT-PSPP-001 #6224 — process → structure → property → performance;
- MAT-UQ-001 #6225 — calibrated multi-fidelity uncertainty and discrepancy;
- MAT-014 #4345 — active discovery and value of information;
- MAT-016 #4991 — advisory model/evidence-authority firewall;
- APPAREL-ENG-TWIN-001 #11731 — garment geometry, fit, construction and lifecycle references.

This file describes a **body-armour profile over existing owner-native objects**. It does not define a new canonical `BodyArmorMaterial` type.

## 2. Core distinction

Keep four propositions separate:

```
material candidate
!= computed material property
!= process-qualified engineering material
!= independently tested complete protective article
```

And:

```
ballistic protection claim
!= one material property target
```

A torso-worn protective system includes material architecture, interfaces, panels, carrier, covers, closures, seams, size, configuration, wearer fit, conditioning and service history. No single material-level score can substitute for article-level evidence.

## 3. Map into existing EngineeringMaterialDemandV1

The demand must be an instance/projection of the existing MAT-ENG engineering-demand semantics, with exact owner-native references. The body-armour profile constrains how those fields are populated; it does not introduce an independent schema.

### A. Engineering origin

Bind:
- exact Symtropy protective-system subject and configuration revision;
- declared use profile, wearer population/anthropometric applicability and service context;
- exact protection proposition and product scope;
- applicable standards register reference, including edition/addenda/currentness;
- hard protection and wearer-safety constraints;
- the evidence refs that justify the claimed material bottleneck;
- system-level sensitivity or discrepancy evidence showing that the material/process axis can plausibly change the declared outcome.

A materials campaign is not justified merely because a material can be optimized. Compose MAT-OPPORTUNITY-003 / the existing bottleneck-attribution owner where available. Valid dispositions include: existing material likely sufficient, process/interface/architecture alternative preferred, evidence insufficient, material discovery justified, and unknown.

### B. Independent property constraints

Represent only properties relevant to the declared material form and operating conditions. Examples include:
- density and thickness-related constraints;
- tensile/compressive/shear response and failure envelopes;
- toughness, fracture and interfacial behaviour;
- rate/temperature/moisture-conditioned response where justified;
- fatigue, creep, aging and environmental degradation;
- processability, joinability, repairability and lot variability.

Every property binds:
- exact canonical quantity and unit;
- material state and form;
- temperature, moisture, orientation, loading regime and conditioning;
- measurement/model source class;
- uncertainty and applicability;
- calibration and provenance.

Do not turn ballistic limit, backface signature, injury risk or a protection level into a generic intrinsic material property. Those are system/article outcomes with their own test and model boundaries.

### C. Keep wearer/system requirements downstream and explicit

These remain distinct product/system constraints, not silently converted into material properties:
- protected area and coverage under movement;
- fit across the declared wearer population;
- mobility, range of motion and task performance;
- heat/moisture burden and ventilation;
- load distribution and equipment interaction;
- service, inspection, damage and repair.

Compose #11731 for garment/pattern/fit/construction references and Symtropy #5377 for system-level evaluation. A flexible material is not, by itself, a good fit. A fit model is not a thermal comfort measurement.

### D. Hard versus soft requirements

Use the existing MAT-ENG constraint classes:
- `HardRequired`
- `Target`
- `Desirable`
- `Exploratory`
- `UnknownNeeded`

The following are non-compensatory:
- declared protection floor;
- exact article/standard scope;
- minimum fit/coverage envelope for the declared use;
- any wearer-safety or operational floor approved by the competent domain owner.

Mass, cost, mobility, thermal burden, manufacturing yield and lifecycle impacts remain separately reported objectives. A lighter or cheaper candidate cannot compensate for a failed hard protection or wearer-safety constraint.

## 4. Process–structure–property chain

Compose MAT-PSPP-001. Preserve separate references for:

```
precursor/material subject
    -> proposed process window
    -> executed process lineage
    -> characterized structure state
    -> conditioned property evidence
    -> article/interface configuration
    -> independently observed system outcome
```

Important distinctions:
- proposed process is not executed process;
- a nominal weave/laminate label is not the full realized structure state;
- an inferred structure feature is not a direct characterization result;
- a computed property is not measured evidence;
- a coupon result is not a complete article result;
- a product claim does not transfer automatically to a different size, panel, carrier, seam, lot or repair state.

Any relationship inferred from observational data must remain associational unless controlled intervention or other appropriate causal evidence supports a stronger disposition.

## 5. Uncertainty and evidence profile

Compose MAT-UQ-001. Preserve separate sources rather than one anonymous confidence scalar:
- input/parameter uncertainty;
- material-lot and process variability;
- aleatory response variability;
- model/epistemic uncertainty;
- model-form discrepancy;
- numerical/convergence error;
- measurement and characterization uncertainty;
- cross-fidelity and scale-transfer discrepancy;
- environmental/service distribution shift;
- OOD/applicability state;
- unknown or unquantified uncertainty.

Required dispositions include explicit unknown, review-required and out-of-domain states. Low model variance in an unfamiliar material/process region is not evidence of confidence.

A calibration receipt must bind exact material family, process regime, structure state, condition, model/evaluator generation and held-out partition. Do not transfer calibration to a new architecture or process just because the nominal chemistry is similar.

## 6. Candidate families: hypotheses only

The literature supports investigating multiple broad architecture/process families, but does not justify treating any as a universal best solution. Possible research families include:
- conventional and unidirectional textile architectures;
- woven and stitched architectures;
- three-dimensional woven/through-thickness reinforcement;
- aramid/UHMWPE and other hybrid reinforcement families;
- interface, matrix and coating strategies;
- adaptive treatments where independently justified;
- graded/segmented system architectures.

These are literature-search categories only. Do not turn them into recipe-level instructions or imply that a family has passed a particular protection level. Candidate generation remains constrained by the declared use case, test scope, available evidence and process capability.

The 2026 mechanistic aramid review reports both opportunities and adverse effects from inappropriate hybridization, reinforcing the need for exact matched comparisons:
- Rana et al., *Aramid Fibre Composites for Ballistic Impact Protection: A Mechanistic Review of Recent Advances* (30 May 2026): https://doi.org/10.1007/s10443-026-10485-y

A September 2026 systematic review reports that no multiscale model has yet demonstrated broad, independently validated predictive capability across 3D-woven architectures and impact regimes; it also highlights gaps in dynamic interfacial fracture data and validated multi-hit models:
- Wei et al., *Three-dimensional woven fabric composites under ballistic impact* (first published 19 September 2026): https://doi.org/10.1177/00219983261480233

This is a benchmark challenge, not a reason to declare a locally developed model superior.

## 7. Discovery and benchmark policy

Use MAT-014's existing acquisition interface. Compare, under matched candidate space and budgets:
1. expert/rule-based material/process selection;
2. small DOE/grid or stratified search;
3. random search as a control;
4. a Pareto optimizer such as NSGA-II;
5. matched-budget Bayesian optimization;
6. Symthaea's uncertainty-, diversity- and hypothesis-discrimination strategies.

Account for:
- evaluation calls, retries, failed calculations and failed synthesis proposals;
- computation and measurement budgets;
- cost of high-fidelity confirmation;
- calibration and held-out data access;
- model/experiment rank reversals;
- cost to independently confirmed feasible evidence;
- worst-stratum and OOD behaviour.

Preserve a separate Pareto profile for protection scope, coverage, mass/bulk, wearer fit, mobility, thermal burden, durability, repairability and lifecycle cost. Any scalar objective used in a small exact oracle must be explicitly named and bounded; it must not replace the multi-dimensional scorecard.

The parent Symtropy programme already includes a NASA materials Bayesian-optimization crosswalk and a finite robust-design oracle. Reuse those implementations and their matched-baseline discipline; do not create a second optimizer.

## 8. Physical-action authority boundary

Symthaea may propose a computation, sample characterization request, coupon test or other physical evidence request. It does not authorize execution.

Any physical action requires a human authorization record that binds:
- exact sample/material lot and product/design revision;
- exact protocol, conditions, specimens and analysis plan;
- exact facility and its active competence/accreditation scope;
- required hazard controls and responsible professional;
- budget, stop rules and post-test data handling;
- expiry and mutation semantics so a changed sample/protocol invalidates stale approval.

No purchasing, synthesis, ballistic testing, destructive testing or wearer exposure follows automatically from an optimization result.

## 9. Immediate acceptance criteria

1. A body-armour demand can be projected through the existing MAT-ENG semantics with exact references and no new canonical type.
2. Hard protection and wearer-safety constraints remain non-compensatory.
3. Material/process properties preserve exact conditions, units, state, provenance and uncertainty.
4. Model-predicted and computed properties cannot be promoted to measured/tested evidence.
5. Candidate ranking cannot directly satisfy an engineering requirement.
6. Unknown, OOD, uncalibrated, uncharacterized and conflicting evidence states remain explicit.
7. Human authorization binds every physical protocol and exact material/sample identity.
8. The B0 record verifier and Symtropy BODY-ARMOR-001 corpus remain the claim/evidence boundary; Symthaea does not become the certifier.

## 10. Claim ceiling

This profile can support requirement mapping, search-space definition, synthetic decision tests and selection of informative evidence requests. It cannot establish a material's ballistic performance, an armour article's protection, an injury-reduction claim, a product's readiness, compliance, certification or superiority over existing armour.