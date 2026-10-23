# MFG-LIFE-000 — Sustainable lifecycle and circular manufacturing architecture

Status: architecture freeze for #5705. This document defines ownership and claim boundaries. It does not establish environmental superiority, circularity certification, legal compliance, real recycling yield, product qualification, or fabrication authority.

## Purpose

Make lifecycle and circularity first-class engineering concerns during design and manufacturing without creating a parallel provenance graph, a second reliability subsystem, or a universal sustainability score.

The target is bidirectional co-design:

```text
design
  ↕
materials
  ↕
manufacturing route
  ↕
service / repair strategy
  ↕
disassembly / remanufacture route
  ↕
recovery / secondary-feedstock route
  ↕
measured lifecycle evidence
```

A lifecycle-friendly design is a proposal until the corresponding physical/economic events and evidence exist.

## Core non-equivalences

```text
sustainable design intent
!= circularity plan
!= predicted lifecycle burden
!= actual lifecycle event
!= verified environmental impact
!= regulatory compliance
```

```text
recyclable in principle
!= recovered in practice
!= specification-grade secondary feedstock
!= closed-loop reuse
```

```text
repairable architecture
!= repair performed
!= repaired article qualified
```

```text
long predicted life
!= observed service life
!= population reliability
```

## Canonical ownership

### Symthaea MFG-LIFE owns

- lifecycle design intent and constraints;
- design-for-disassembly / repair / reuse / remanufacture / recovery planning;
- lifecycle route candidates and engineering tradeoffs;
- references to lifecycle scenarios, service-life assumptions and external impact assessments;
- co-optimization interfaces into engineering design and MFG-PROC;
- explicit claim ceilings and missing-evidence diagnostics.

### MFG-PROC owns

- manufacturing and lifecycle-process identity;
- process capability requirements;
- recipes/commitments;
- canonical process plans;
- repair/remanufacture/recovery operations when represented as manufacturing-process plans.

MFG-LIFE must not create a second process ontology.

### CRITMAT #5200 owns

- critical-material functional dependence;
- material intensity per declared functional unit;
- substitution/dependency-delta research;
- recovery/separation research;
- critical-material reuse/recycling/lifetime intervention evidence.

MFG-LIFE consumes these artifacts rather than repeating critical-material accounting.

### SE-VV #3697 / #4920 owns

- lifecycle exposure;
- degradation/wear evidence;
- observed failure/censoring;
- maintenance/rework segmentation;
- reliability/maintainability analyses and their uncertainty.

MFG-LIFE may reference a service-life or maintenance subject but cannot turn a design assumption into reliability evidence.

### ENG-MAT #5074 owns

- exact material-state subjects;
- process-conditioned material properties;
- lot/article applicability and material evidence.

A recovered feedstock must become an explicit qualified material state before it can substitute for a primary feedstock in strict engineering claims.

### FIELD / QIF / metrology own

- physical observations;
- inspection measurements;
- instrument/calibration provenance;
- article-level measured evidence.

MFG-LIFE does not manufacture measured energy, emissions, composition, wear or recovery yield from a model.

### Mycelix Economic Reality Graph and Circularity own

- actual economic/material/product/custody/transformation events;
- lot/batch/serial/article identity across organizations;
- return, service, repair, reuse, resale, refurbishment, remanufacture, parts harvest, recycle, material recovery and disposal events;
- secondary-feedstock lineage and chain-of-custody strategy;
- outward product-passport/circularity projections.

Symthaea predicts and plans. Mycelix records/coordinates actual distributed lifecycle events. Neither role implies the other's authority.

## Lifecycle graph

The desired semantic loop is not a one-way supply chain:

```text
resource / feedstock
      ↓
processing
      ↓
manufacturing
      ↓
assembly
      ↓
distribution
      ↓
use
      ↓
inspection / service
      ↓
return
      ├── reuse / resale ──────────────────────────┐
      ├── repair ──────────────────────────────────┤
      ├── refurbishment ───────────────────────────┤
      ├── remanufacturing ─────────────────────────┤
      ├── parts harvesting ────────────────────────┤
      ├── recycling / recovery                     │
      │          ↓                                 │
      │   qualified secondary feedstock ───────────┘
      └── disposal
```

Disposal must remain explicit. A projection may not hide residual waste to improve a circularity metric.

## Lifecycle design profile

A future `LifecycleDesignProfileV1` should bind references rather than copy canonical domain state. Candidate fields include:

```text
LifecycleDesignProfileV1 {
  subject_configuration_ref,
  scenario_refs,
  functional_unit_ref,
  service_life_assumption_ref,
  maintenance_strategy_refs,
  disassembly_plan_ref,
  repair_strategy_refs,
  reuse_strategy_refs,
  remanufacture_strategy_refs,
  recovery_route_refs,
  material_flow_account_ref,
  environmental_inventory_refs,
  impact_assessment_refs,
  circularity_assessment_refs,
  protected_lifecycle_constraints,
  evidence_limitations,
}
```

All external refs remain unresolved until their owning systems resolve them.

## Independent design axes

Never collapse lifecycle performance into one `sustainability_score` by default. Keep at least these axes independently inspectable when evidence exists:

- primary material demand;
- secondary/recycled input fraction and provenance;
- critical-material mass per declared functional unit;
- manufacturing yield and scrap;
- process energy demand;
- water/reagent/consumable demand;
- environmental inventory / impact-assessment refs;
- expected service life and its evidence class;
- maintenance burden;
- modularity / replaceable subassemblies;
- repair access and required tools/processes;
- disassembly time/process burden;
- reversible vs destructive joining;
- component reuse / parts harvesting potential;
- remanufacturing route availability;
- recoverable material mass and quality;
- recovery energy/process burden;
- hazardous/controlled-material handling burden;
- residual waste/disposal route;
- uncertainty, missing evidence and model discrepancy.

An improvement on one axis cannot silently erase regression on another.

## Design-for-circularity constraints

Examples of legitimate design constraints include:

- require reversible fasteners/joints for a protected service interface;
- preserve material separability when a recovery route requires it;
- provide access/clearance for a declared repair task;
- design modules for replacement without destroying unrelated subassemblies;
- preserve IDs/datums needed for later inspection and remanufacturing;
- bind adhesive/coating/potting choices to their future disassembly/recovery consequences;
- require a recovery/disposal path for controlled or hazardous materials;
- allow remanufacturing operations to create a new as-built/configuration/qualification lineage.

These are engineering constraints, not evidence that the product will actually be repaired or recycled.

## Material-flow and mass-balance closure

Use CRITMAT and the Mycelix material/economic graph for material accounting. Do not create another ledger.

Required principle:

```text
attributable recovered output
<= attributable input + separately evidenced added material
```

subject to the declared accounting profile.

Where mixing destroys item-level physical identity, preserve the chain-of-custody strategy explicitly rather than pretending every atom remained individually traceable.

Possible strategies are referenced from the Mycelix layer, e.g. identity-preserved, segregated, controlled blending, mass balance or book-and-claim where appropriate.

## Value-retention hierarchy is descriptive, not a universal ranking

Reuse, repair, refurbishment, remanufacture and material recovery often retain different forms of value, but MFG-LIFE must not hard-code one universal ordering as an engineering truth.

A specific lifecycle scenario may compare routes using independent evidence such as:

- retained function;
- retained component/material mass;
- additional processing burden;
- expected remaining service life;
- quality/qualification requirements;
- logistics burden;
- environmental impact profile;
- cost and availability;
- uncertainty.

The result is a profile-relative tradeoff, not a universal winner.

## Assessment adapters

External methods and standards are edge adapters and profile identities, never the internal truth model.

Initial adapter families may include:

- ISO 14040 / ISO 14044 lifecycle-assessment framing;
- ISO 59004 circular-economy vocabulary/principles;
- ISO 59010 circular value-network/business-model transition guidance;
- ISO 59020 circularity measurement/assessment;
- ISO 59040 Product Circularity Data Sheet projection;
- EU ESPR / Digital Product Passport projection where applicable;
- sector-specific EPD/PEF/environmental methods through explicit versioned profiles.

Required theorem:

```text
adapter mapping exists
!= assessment verified
!= certification
!= legal compliance
```

Dataset/database identity, geography, time window, allocation method, functional unit and uncertainty remain part of any strong assessment claim.

## Product-passport boundary

Product passports are projections from the shared economic/material graph, not a second product-lifecycle database.

```text
Economic Reality Graph
   ├── EU DPP projection
   ├── GS1 EPCIS projection
   ├── ISO 59040 Product Circularity Data Sheet projection
   ├── consumer provenance view
   ├── repair passport
   └── recycler/material passport
```

A private purchase passport may point to a product passport, but customer identity/payment/private purchase history must remain separately controlled.

## Privacy and proprietary manufacturing knowledge

Lifecycle evidence must compose the existing recipe/privacy direction. A manufacturer may prove or reference bounded lifecycle facts without publishing proprietary recipes/process details when the verification profile permits it.

Keep distinct:

```text
commitment exists
!= underlying fact disclosed
!= underlying fact verified
```

Selective disclosure should never weaken mass-balance, qualification or evidence-currentness requirements.

## Optimization integration

MFG-LIFE should later plug into ENG-OPT / manufacturing optimization as independent Pareto objectives and hard constraints.

Examples:

```text
candidate A:
  lower manufacturing energy
  shorter demonstrated service life

candidate B:
  higher manufacturing energy
  modular repair
  longer bounded service evidence
```

Do not let a scalar optimizer silently decide that one is universally more sustainable. Preserve the frontier and profile-specific constraints.

## Change and history semantics

Lifecycle history is append-only/evidence-bearing.

```text
original article
 -> service event
 -> repair event
 -> new as-maintained state
 -> later refurbishment/remanufacture
 -> new configuration / qualification obligations
```

Do not rewrite original design, as-built or historical measurement evidence after repair/recovery.

## Initial adversarial corpus

1. recyclable material + inseparable adhesive assembly does not establish a feasible recycling route;
2. `repairable` component inaccessible without destructive teardown weakens the repair strategy;
3. claimed recycled content without chain-of-custody/material-flow evidence remains unresolved;
4. recovered material mass exceeds attributable feed without added-material evidence -> reject;
5. one physical mass cannot be simultaneously double-counted as harvested component and recycled constituent;
6. long predicted service life without lifecycle evidence remains an assumption;
7. lower manufacturing energy + much shorter life remains a visible tradeoff;
8. remanufactured article creates new configuration/qualification lineage;
9. external LCA database result != measured factory emissions;
10. circularity metric != regulatory compliance;
11. product-passport projection != canonical lifecycle database;
12. residual disposal may not be silently omitted;
13. stale impact dataset/profile cannot support a current strong assessment without an explicit applicability theorem;
14. private evidence commitment cannot be interpreted as undisclosed values;
15. design-for-repair intent cannot be promoted to an observed repair event.

## First pilots

### LIFE-PILOT-001 — benign serviceable product

Use a low-consequence product or fixture with:

- replaceable wear component;
- reversible fasteners;
- explicit inspection/service step;
- one repair path;
- one parts-harvest/material-recovery path.

Synthetic phase first, then physical evidence only after qualification and explicit operator approval.

### LIFE-PILOT-002 — secondary feedstock loop

Synthetic chain:

```text
primary + secondary feedstock
 -> manufacturing
 -> article
 -> recovery
 -> characterized secondary feedstock
 -> new manufacturing input
```

Prove material accounting and evidence boundaries before making real recycled-content claims.

## Implementation sequence

1. `MFG-LIFE-000A` — this architecture/ownership freeze.
2. `MFG-LIFE-001` — lifecycle-design profile + strategy refs.
3. `MFG-LIFE-002` — disassembly/repair/remanufacture/recovery planning contracts.
4. `MFG-LIFE-003` — material-flow bridge into CRITMAT/Mycelix accounting.
5. `MFG-LIFE-004` — LCA/circularity assessment-reference adapters.
6. `MFG-LIFE-005` — ENG-OPT / manufacturing Pareto integration.
7. `MFG-LIFE-006` — Mycelix Circularity/Economic Reality Graph bridge.
8. `MFG-LIFE-007` — product-passport / ISO 59040 projection qualification.
9. `LIFE-PILOT-001` — synthetic then bounded physical lifecycle campaign.

## Exit criterion

Symthaea can co-design an engineered product and manufacturing route with explicit repair, disassembly, reuse/remanufacture and recovery strategies; preserve independent lifecycle/environmental tradeoffs and uncertainty; connect actual lifecycle events through Mycelix without duplicating provenance; and close recovered materials/components back into future manufacturing subjects without confusing design intent, model predictions, events, measurements, assessments or compliance.