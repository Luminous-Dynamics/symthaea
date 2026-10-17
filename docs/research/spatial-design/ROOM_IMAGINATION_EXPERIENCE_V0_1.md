# Room Imagination & Spatial Design Experience v0.1

**Status:** Research/design proposal; not a shipped capability claim.  
**Primary composition/intent owner:** Symthaea, reusing ENG-DESIGN-003 (#6091) and SWA-001 (#6624).  
**Built-environment realization owner:** BUILT-ENV-001 (#6032).  
**Spatial simulation/topology integration:** Symtropy SPATIAL-DESIGN-001 (#1766), PB-04 (#455), ACCESS-HOME-001 (#1687), and the shelter-to-home integration proof (#492).  
**Related cross-project intent work:** [Mycelix PR #5137](https://github.com/Luminous-Dynamics/mycelix/pull/5137).  
**Scope:** How a person imagines, explores, revises, and qualifies interior/room/building concepts—from vague feeling to an editable spatial proposal and an honest engineering handoff.

## 1. Thesis: Symthaea should help people imagine, not merely decorate

Yes, Symthaea can be a powerful part of a room/building design system. Its most distinctive role is to translate human goals into multiple spatial possibilities and make the trade-offs understandable. But this is a **target interaction architecture**, not a claim that all necessary CAD, BIM, daylight, acoustics, HVAC, structural, code, or construction integrations are already implemented.

The paradigm shift is:

> From “generate a pretty room” to “help me discover the place I want, explore different ways it could work, and progressively turn one concept into a measurable and reviewable design.”

A person should not need architectural vocabulary to begin. They may say:

> “I want a small room that feels quiet and warm in the morning, gives me somewhere to read and think, doesn't feel cramped, and is easy to keep cool.”

The system should help clarify, for example:
- what activities the room must support;
- what “quiet,” “warm,” and “not cramped” mean to this person;
- what is a hard constraint versus a preference;
- which facts are known about the existing room/site;
- what budget, rental, structure, maintenance, accessibility, and local context restrictions apply;
- what remains uncertain and is worth exploring.

It should then offer several genuinely different alternatives, not just recolor one generic room, while preserving the declared constraints and showing what each option trades away.

## 2. The core paradigm shifts

| Conventional design experience | Proposed spatial-intelligence experience |
|---|---|
| Start from a blank CAD/BIM canvas | Start from activities, feelings, needs, constraints, context, and rough references |
| Require precise terminology up front | Let people sketch, speak, choose examples, move objects, compare variants, and correct the system in ordinary language |
| Generate one polished hero render | Generate several divergent, explainably different options, including inexpensive, adaptive, restrained, and ambitious options |
| Treat a render as the design | Maintain a strict distinction between illustrative imagery, approximate concept geometry, editable scaled geometry, and qualified BIM/engineering evidence |
| Treat accessibility as a later checklist | Co-design around real tasks, circulation, reach, sensory access, egress, and adaptable needs from the beginning |
| Optimize one room in isolation | Model its relationship to the rest of the home/building: movement, sound transmission, daylight, privacy, air, thermal behavior, maintenance, utilities, and shared use |
| Collapse comfort/cost/beauty/energy to one score | Preserve a vector of goals, hard constraints, uncertainties, costs, and stakeholder disagreement |
| Stop at a photorealistic rendering | Progress from concept to dimensioned plan, editable scene, analysis request, open-format exchange, professional review, and eventual as-built evidence |
| Assume a single universal ideal | Let different occupants and communities define what “good” means in their context |

## 3. Experience flow: from imagination to engineering

Use a progressive-fidelity sequence. A user can move back and forth; they are not forced through one linear wizard.

### Stage A — Tell us about the experience you want

Inputs can include natural language, a moodboard/reference image, rough sketch, room photo/scan supplied voluntarily, a floor plan, an existing model, or a few visual examples. The system should never require every input.

Begin with open-ended exploration and optional prompts around:
- activities and daily routines;
- atmosphere and visual character;
- light, acoustics, thermal preferences, privacy and connection;
- storage, furniture, movement and maintenance;
- budget, materials and reuse;
- accessibility and assistive needs framed around actual tasks;
- constraints the user knows, plus things they do not know yet.

The system may ask one decision-relevant question at a time. “Not sure yet—show me possibilities” must be a first-class answer. If a missing measurement only affects a later engineering decision, the user should still be able to explore a clearly labelled conceptual sketch.

### Stage B — Reflect a revisable design brief

Before serious generation, show the user a short, editable brief with:
- **Goals:** what the room should support or feel like.
- **Hard constraints:** dimensions, openings, protected structure, budget ceiling, required clearances, or other non-negotiables explicitly identified by the user or a qualified source.
- **Preferences:** desired styles, materials, colors, and atmosphere, without silently turning taste into a rule.
- **Unknowns:** dimensions, site conditions, jurisdiction, materials, or performance inputs not yet measured.
- **Evidence:** where each fact came from and how current it is.
- **Conflicts:** incompatible objectives, with possible resolutions.
- **Privacy:** what personal information, photographs, drawings, or household details will be retained or shared.
- **Review needs:** which qualified human roles need to assess a later stage.

A correction must update the active brief and all dependent proposals. Model inference must remain labelled as a hypothesis, not become the user's “true intent.”

### Stage C — Generate genuinely different alternatives

Provide a small set (start with 3) whose differences are structural, not superficial. For a reading room this could be:
1. **Quiet alcove:** strong sense of enclosure and acoustic separation.
2. **Daylight room:** more open visual connection and carefully managed glare/solar exposure.
3. **Flexible room:** modular furniture/storage and reversible layout changes.

For each, show:
- what user-stated goals it serves;
- what it costs or complicates;
- what assumptions it depends on;
- what the model actually checked, if anything;
- what is still unknown;
- why it differs from the other options.

Do not label a design “best” without naming its objective and the constraints that make it preferable. Include “none of these—change direction” and “keep exploring.”

### Stage D — Let people imagine and inhabit the proposal

A person should be able to inspect the same alternative in complementary views:
- moodboard and reference images;
- atmospheric concept render;
- top-down plan with dimensions and furniture footprints;
- editable 3D view and simple walkthrough;
- sections/elevations and critical clearance views;
- day/night and selected season/time-of-day lighting previews, only where the inputs support them;
- task walkthroughs: enter, turn, reach, sit, work, rest, clean, maintain, and exit;
- acoustic, thermal, ventilation, daylight, energy, and circulation panels when the relevant model and inputs exist.

**The image is not the geometry authority.** Any render should identify the exact underlying concept/model revision. A beautiful generated image must never silently invent extra space, omit a column or window, change dimensions, or imply that a technically impossible arrangement fits.

For every edit, allow natural-language requests (“move the desk nearer to daylight”), direct manipulation, and numeric controls. Show the affected scope and preserve constraints such as doors, columns, measured boundaries, or user-locked elements. Substantial changes create a preview or new branch rather than destructively rewriting the approved concept.

### Stage E — Compare lived consequences, not only appearance

Symtropy can own the relevant spatial/simulation substrate where the geometry and solver contracts qualify. The comparison should retain separate dimensions:
- task completion and route usability;
- clearances and access requirements;
- daylight and glare;
- thermal conditions and cooling/heating loads;
- acoustic privacy and noise transfer;
- ventilation/air-quality requirements;
- visibility, privacy, and social/spatial gradients;
- energy/water/material/cost estimates;
- repair, cleaning, reconfiguration, and maintenance access;
- resilience under power/network loss where connected systems are proposed;
- explicit disagreements among occupants.

Not every room needs all analyses. A panel should say “not evaluated,” “inputs incomplete,” or “out of this model's scope” rather than manufacture a result.

Human comfort is not one scalar. For example, ASHRAE 55 describes thermal comfort as dependent on environmental and personal factors; it does not cover every other indoor-quality issue such as lighting, acoustics or all air contaminants. Separate, properly scoped models and standards are required for separate claims.

### Stage F — Convert the concept into an honest engineering handoff

When requested, generate a structured design package containing:
- room/zone and spatial relationships;
- the exact geometry/model version;
- dimensional units and coordinate/frame conventions;
- objects/material assumptions and product references;
- requirement-to-room/task traceability;
- selected and rejected alternatives;
- unresolved questions and potential conflicts;
- analysis inputs/solver versions/results and their limitations;
- maintenance/reconfiguration assumptions;
- local jurisdiction/context fields;
- professional review checklist;
- change-impact lineage.

Use open interoperability rather than inventing a proprietary BIM replacement. IFC 4.3.2.0 is the current official IFC 4.3 release identified by buildingSMART and is published as ISO 16739-1:2024. Support only the specific IFC model view/subset that an adapter actually validates; do not claim full IFC compatibility simply because a file exports.

The application can help prepare evidence and candidate designs. It cannot confer planning permission, code compliance, structural adequacy, fire/life-safety approval, accessibility certification, or construction authority.

### Stage G — Learn from the built space, with consent

In a later, separately qualified phase, compare predictions with voluntary occupant feedback, measured environmental data, maintenance events, and verified as-built conditions. Keep individual comfort, occupancy and personal routines private by default.

Do not turn home sensors into resident/employee surveillance or infer health, emotion, productivity, or relationship quality from ambient telemetry. Learn about the design only from data appropriate to the purpose, consented scope, and evidence quality.

## 4. Responsibility boundaries across the ecosystem

| System | Proper responsibility | Must not claim |
|---|---|---|
| **Symthaea** | Reflect design intent; ask useful questions; generate alternatives; explain trade-offs; plan which analyses are needed; preserve uncertainty and change lineage | That a render proves feasibility, an inferred aesthetic is universal, or a recommendation is certification |
| **Symtropy** | Exact spatial geometry integration, typed/faceted topologies, scenario execution and deterministic comparison under declared solver/model limits | That generic physics or a scene graph proves building-code compliance, comfort, or physical safety |
| **Mycelix** | Consent and sharing boundaries, stakeholder participation, design decisions, provenance and review receipts | That a governance decision proves engineering performance |
| **Sol Atlas** | Qualified site/context inputs such as orientation, terrain, solar/weather or local context when valid sourced adapters exist | That map existence guarantees data freshness, a site survey, or environmental qualification |
| **Qualified professionals / responsible authorities** | Applicable codes, structural/MEP/fire engineering, accessibility review, permitting and approval, site surveys, construction and commissioning validation | Their role should not be implied by an AI-generated report |

Reuse the existing owners and issue scopes:
- Symthaea [ENG-DESIGN-003 #6091](https://github.com/Luminous-Dynamics/symthaea/issues/6091), [BUILT-ENV-001 #6032](https://github.com/Luminous-Dynamics/symthaea/issues/6032), and [SWA-001 #6624](https://github.com/Luminous-Dynamics/symthaea/issues/6624).
- Symtropy [SPATIAL-DESIGN-001 #1766](https://github.com/Luminous-Dynamics/symtropy/issues/1766), [ACCESS-HOME-001 #1687](https://github.com/Luminous-Dynamics/symtropy/issues/1687), [PB-04 #455](https://github.com/Luminous-Dynamics/symtropy/issues/455), and [PB-06 #492](https://github.com/Luminous-Dynamics/symtropy/issues/492).
- Mycelix [intent exploration and human agency PR #5137](https://github.com/Luminous-Dynamics/mycelix/pull/5137).

This document is an experience-layer composition of those responsibilities, not a second geometry engine, spatial ontology, building-code verifier, or authorization system.

## 5. Initial benchmark plan

The first qualification corpus should be synthetic; no resident photographs, diagnoses, floor plans or household routines should be imported without explicit purpose-specific permission and review. See the companion [Room Imagination benchmark](ROOM_IMAGINATION_BENCHMARK_V0_1.json).

### Proposed gates (targets, not achieved results)

| Gate | Initial target | Evidence type |
|---|---|---|
| Intent preservation | 100% of explicitly marked hard constraints appear in the brief and are preserved through generated variants; zero silent relaxation | Deterministic trace checks against known-answer fixtures |
| Preference honesty | Zero inferred tastes silently promoted to hard requirements; every meaningful inference can be corrected | Property tests and user walkthroughs |
| Geometry fidelity | 100% of exact dimension/clearance predicates used by the test fixture are checked against the authoritative geometry revision, not an image | Independent deterministic geometry checker |
| Render/model lineage | 100% of published views identify their model/scene revision; zero view claimed to depict unbound or incompatible geometry | Manifest and revision checks |
| Unknown/evidence honesty | 100% of unknown, stale or unvalidated fields remain visibly unknown/unverified; zero model confidence presented as certification | Frozen state matrix and negative controls |
| Variant diversity | For the curated prompt set, at least 3 alternatives must differ on declared spatial/functional dimensions rather than color alone | Structured geometry/property comparison |
| Analysis discipline | 100% of performance claims cite applicable inputs, model/solver version, units, conditions and limitations; unsupported claims fail closed | Solver receipts and independent review |
| Determinism | 100% identical output receipts for identical frozen deterministic geometry inputs/versions where determinism is a declared contract; otherwise compare with an explicit tolerance and reason | Repeated seeded scenario runs |
| Accessible task coverage | Every critical task in the declared scope has an evaluated path; no critical blocked cohort hidden by an average | Task-chain suite plus paid participant co-design/manual AT testing |
| Human task success | Initial target ≥90% completion for a defined set of representative, non-high-stakes concept-design tasks; report baseline and subgroup confidence intervals | Usability study |
| Intent comprehension | Median ≥4.3/5 on “This brief captures what I wanted,” reported alongside actual constraint omissions | Voluntary survey plus trace review |
| Maintenance/change impact | 100% of changed requirements/geometry identify potentially stale dependent analyses and affected handoff artifacts | Change-impact fixture |
| Professional handoff | 100% of mandatory unresolved review items are visible; no transition to a “verified” state solely from rendering or model generation | Independent checklist / gate verifier |

These targets should be reviewed and calibrated before treating them as release gates. Hard constraints, required accessibility paths, authorization and epistemic honesty cannot be traded away for a higher aggregate score.

### First synthetic scenario pack

1. A compact reading room with daylight preference and an explicitly fixed existing window.
2. A shared multigenerational living room with different privacy, noise, storage and activity needs.
3. A renter's room redesign where walls, plumbing and electrical circuits are off-limits; layout changes must be reversible.
4. A small home office with daylight versus glare trade-offs and task lighting.
5. A kitchen where work sequences, reachable storage, food preparation and circulation must be compared.
6. A bedroom with nighttime navigation, adjustable lighting and low-noise needs.
7. An accessible entrance and bathroom concept where site measurement and jurisdiction-specific review are required.
8. A hot-climate room comparing orientation/shading and cooling demand using a declared energy model, not a render guess.
9. A room with an uncertain measured boundary: permit concept exploration but block exact fit claims.
10. A beautiful render whose furniture or circulation overlaps in the authoritative plan: fail the fidelity gate.
11. A request for “traditional” design where the cultural context is not stated: ask or keep references unattributed, never stereotype.
12. Household members disagree about openness/privacy: preserve multiple user objectives and avoid claiming consensus.
13. A user changes a locked dimension after generating alternatives: invalidate dependent geometry and analyses.
14. A room variant removes necessary service/maintenance access: flag or reject it according to the declared requirement.
15. A design that looks accessible in a render but fails the task-based route/clearance checker: fail; show the concrete conflict.
16. A power/network-dependent smart-room proposal: present a manual fallback and degraded-mode implications.
17. A stale climate or product catalog input: expose freshness and block claims requiring current data.
18. Export to IFC with a known unsupported property: report projection loss explicitly rather than claiming lossless export.

For real-user research, include disabled people as paid co-design partners and test real tasks in the spaces they use. Do not use a simulated disability mode as a substitute for participation. ISO 21542:2021 is a relevant international reference for accessibility/usability of the built environment; South African projects should also use the applicable local regulations, standards and qualified professional review rather than assuming an international standard alone resolves jurisdictional requirements.

## 6. Implementation sequence

1. **Freeze the intent/evidence vocabulary and scenario manifest.** No frontend rewrite.
2. **Create one synthetic room brief and 3 materially different candidate layouts.** Keep exact dimensions, preferences, assumptions, and unknowns separate.
3. **Add structural validation for the brief/scene lineage.** Test hard-constraint retention, preference correction, stale evidence, and a render/geometry mismatch.
4. **Connect the selected prototype to Symtropy's qualified geometry/topology interface** only after confirming the exact provider contracts and model generation.
5. **Add one bounded analysis** (e.g. a simple, declared geometry/circulation check) with known-answer fixtures before claiming thermal/acoustic/light analysis.
6. **Test with newcomers and domain professionals.** Measure task completion, comprehension, revision effort, accessibility blockers, and misinterpretations; record failures as first-class outcomes.
7. **Only then expand to IFC/BIM exchange, richer environmental simulation, existing-building scans, materials/products, and professional workflows.**

### First vertical-slice definition of done

A person with no CAD experience can describe a room goal, correct the resulting intent brief, compare three materially different conceptual options, understand which facts are assumptions, inspect a dimensioned 2D/3D representation where geometry exists, and export a design brief. The slice can truthfully label analyses that have not been performed and does not claim professional approval.

## 7. Research and standards grounding

- buildingSMART International, [Industry Foundation Classes (IFC)](https://www.buildingsmart.org/standards/bsi-standards/industry-foundation-classes/?lang=en): open, vendor-neutral BIM interchange; IFC 4.3.2.0 is identified as the latest official IFC 4.3 release and published as ISO 16739-1:2024.
- U.S. Department of Energy, [About Building Energy Modeling](https://www.energy.gov/cmei/buildings/about-building-energy-modeling): describes physics-based building energy modeling from geometry, materials, lighting/HVAC/system configurations, schedules and weather inputs.
- ASHRAE, [Standard 55 — Thermal Environmental Conditions for Human Occupancy](https://www.ashrae.org/technical-resources/bookstore/standard-55-thermal-environmental-conditions-for-human-occupancy): explains the interaction of environmental and personal thermal-comfort factors; it does not substitute for separate air-quality, acoustics, lighting, or other discipline-specific criteria.
- ISO, [ISO 21542:2021 — Accessibility and usability of the built environment](https://www.iso.org/standard/71860.html): covers building access, circulation, usability and egress/evacuation considerations; ISO lists the edition as under review, so verify applicable current editions/status at project use.
- South African Government, [National Strategic Framework on Universal Design and Access](https://www.gov.za/documents/notices/national-strategic-framework-universal-design-and-access-promotion-universal): a local policy reference for universal design/access discussions.

These references inform the proposed interfaces; they do not certify a generated design or establish actual comfort, safety, accessibility, legal compliance, or constructability.

## Claim ceiling

This proposal defines interaction, representation, test and evidence requirements. It does not assert that Symthaea currently generates qualified architectural models, that an image proves geometry, or that a simulated room is safe or code-compliant. Those claims require the relevant integrated implementation, independent verification, professional review and, where appropriate, real-world measurement.
