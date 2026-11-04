# INVENTOR-LOOP-001 — Bridge / Truss Design Challenge (Learner Packet)

**Artifact status:** DESIGN_ONLY / learner-facing draft; no merged Symtropy solver integration  
**Programme:** [INVENTOR-LOOP-001](../INVENTOR_LOOP_001_PROGRAM_V0_1.md)  
**Tracking:** [Symthaea #10442](https://github.com/Luminous-Dynamics/symthaea/issues/10442)  
**Date:** 2026-11-04  
**Revision:** 0.1

> This is an educational design brief, not a construction drawing, engineering approval, physical test instruction or safety certification. The first activity is paper/digital modelling. Any physical prototype or loading exercise requires a separately approved educator-led risk assessment, guarded fixture and calibrated measurement plan.

## 1. The challenge

You have been asked to design a small model bridge that crosses a gap.

Your job is not to copy a famous bridge. Your job is to understand how a structure carries forces, create more than one possible design, explain why you chose one, and improve it when new evidence arrives.

**The big question:** How can we make a structure that uses its material thoughtfully while meeting clearly stated requirements?

There is no single required shape. You are allowed to invent different truss arrangements, explain trade-offs and discover that your first idea was not the best one.

## 2. What you will learn

By the end of the project, you should be able to:

- draw a free-body diagram for a simple pin-jointed structure;
- explain tension, compression, equilibrium and load paths;
- use geometry, vectors and trigonometry to describe member directions;
- explain how member length, cross-sectional area and elastic modulus affect idealized axial stiffness;
- distinguish a model prediction from a measured result;
- identify at least one important limitation in your model;
- compare two designs using requirements and evidence rather than appearance alone;
- explain one revision you made and what evidence motivated it.

The activity can be adapted by the educator for different ages and prior knowledge. More advanced learners can work with equations and sensitivity analysis; younger learners can begin with force arrows, manipulable diagrams and qualitative predictions. Until the owner-approved digital interface is available, all learners use the paper/hand-calculation path; no solver call is implied.

## 3. Design brief and requirements

The educator or benchmark author must provide a versioned task card containing the actual numerical values for the chosen digital case. The following are intentionally parameters, not hidden assumptions:

| Requirement | Task-card value | Unit / status |
|---|---|---|
| Clear span | To be assigned | length |
| Available envelope | To be assigned | length × height |
| Candidate material model | To be assigned | material identity + declared elastic properties |
| Member cross-section options | To be assigned | area |
| Joint idealization | Pin-jointed for the analytical exercise | model assumption |
| Applied load case | To be assigned for the digital reference case | force + direction + application point |
| Support conditions | To be assigned | constrained degrees of freedom |
| Mass/material budget | Optional, to be assigned | mass or consistent proxy |
| Design objective | Compare feasible alternatives; do not hide hard-constraint failures in a weighted score | decision rule |
| Analysis scope | 2D, static, small-displacement, linear-elastic, axial-only truss | explicit model limit |

Do not invent missing numerical values. Ask the educator or task owner for a clarification, or mark the requirement unresolved. A solver that accepts a request does not establish that the selected load, material or support assumptions are appropriate for a real structure.

## 4. Your design notebook

Keep a short record with these sections.

### A. What do you think will happen?

Before using a solver, draw the bridge and predict:

- which members you think will be in tension;
- which members you think will be in compression;
- where the largest movement might occur;
- which part you think will govern the design;
- which part of your prediction you are least sure about.

It is fine to be wrong. A useful prediction gives you something specific to compare with evidence.

### B. Draw at least two different ideas

Try different load paths rather than only changing the colour or dimensions of one drawing.

For each design, identify:
- nodes and members;
- supports and applied loads;
- material and section assumptions;
- the reason for each major diagonal or connection;
- one likely weakness;
- one question that analysis could answer.

### C. Explain the mathematics

Use the level of mathematics appropriate to your experience.

Useful relationships for the idealized axial member include:

```text
axial stiffness k = E A / L
```

where E is Young's modulus, A is cross-sectional area and L is member length.

This relationship is not a complete bridge model. It is one part of the idealized truss formulation. For an inclined member, its direction determines how its axial response contributes to horizontal and vertical stiffness.

Explore questions such as:
- What happens to idealized axial stiffness if the area doubles?
- What happens if the same material and area are used in a longer member?
- Why can two different truss layouts carry the same external load differently?
- Why are supports and load directions essential to the answer?

Do not apply a formula outside its declared assumptions.

### D. Analyse, then question the result

The digital analysis should be carried out using the existing Symtropy PHYS-TRUSS implementation and its supported contract, after its owner confirms the exact branch/API and fixture profile.

The current model family is intentionally narrow: static, small-displacement, linear-elastic, 2D pin-jointed axial bars. It does not model member bending, buckling, geometric nonlinearity, contact, fracture, material plasticity, dynamics, connection failure, fabrication defects or real-world structural safety.

Review:
- displacements and their directions;
- signed member axial forces and stresses;
- support reactions;
- force and moment balance residuals;
- energy/work consistency where the selected model reports it;
- exact input/model/solver identity;
- unresolved, unsupported or invalid conditions.

A PASS from a mathematical oracle or software test only applies to the case and model it actually checks. It is not a physical test result.

### E. Ask what would teach you the most

Write down two explanations for an unexpected result. Then ask:

- What observation would distinguish them?
- Can the question be answered from existing evidence?
- Would a different load direction or geometry expose a hidden assumption?
- What is the smallest safe, reversible next experiment?
- What result would make you change your mind?

An experiment proposal must identify its purpose, predicted outcomes, required instruments, risks, decision rule and stop conditions. Learners do not perform physical loading themselves as part of this packet.

### F. Revise and explain

Create a new design revision rather than overwriting the old one. Record:
- what changed;
- why it changed;
- which observation or calculation motivated the change;
- what you expect to improve;
- what might become worse;
- what evidence would show whether your prediction was right.

## 5. Helpful hints — not a solution

1. Start with supports and loads. A beautiful shape with an unsupported degree of freedom is not a valid structural model.
2. Draw force arrows before writing equations.
3. Think about the path a force takes from the load point to the supports.
4. Use symmetry as a useful check, but do not rely on symmetry alone: it can conceal coupling errors.
5. Compare a different geometry or an off-axis loading case only when the task definition explicitly provides it.
6. If the solver reports a mechanism, singular system, invalid input or unresolved outcome, do not add arbitrary supports just to obtain a number. Find out what assumption is missing or wrong.
7. If a result is sensitive to an uncertain input, show the uncertainty. Do not report a midpoint as though it were guaranteed.

## 6. Deliverables

A completed *digital learning exercise* contains:

1. A requirements table with units, source and unresolved items.
2. At least two distinguishable design concepts.
3. A prediction made before solver results are revealed.
4. A free-body diagram and a mathematical explanation.
5. An editable, versioned digital design/model.
6. An analysis record tied to exact inputs and supported solver scope.
7. A comparison of predictions with calculated results.
8. A documented revision and rationale.
9. A short reflection describing what the learner now understands and what remains unknown.

No physical prototype, fabricated component, structural test or safety assessment is implied by completing these deliverables.

## 7. Learner assessment

Assess the learner's reasoning separately from the design's numerical performance.

| Dimension | Evidence of progress |
|---|---|
| Requirements | Notices missing units, ambiguous limits and unsupported assumptions |
| Representation | Correctly identifies nodes, members, supports and loads |
| Mathematics | Explains force balance and the meaning/limits of the stiffness relationship |
| Prediction | Makes a specific prediction before seeing the analysis |
| Evidence use | Cites the result that supports a design conclusion and names contrary evidence |
| Model criticism | Identifies at least one physical effect omitted by the idealization |
| Revision | Makes a reasoned change and states what trade-off it may introduce |
| Independence | Solves a structurally different transfer problem without the same step-by-step scaffold |

Do not award mastery merely because the AI generated a correct answer or the learner submitted an AI-produced design. Retain separate records for unaided attempt, hints, explanations, AI assistance, educator support and independently assessed transfer. This must compose with Praxis #5286, #5455, #5461 and #5467 rather than changing its assessment/learner-state contracts.

## 8. Educator / benchmark-author checklist

Before releasing a numerical version of this challenge:

- freeze the task revision, requirements, numerical inputs, units and supported model class;
- confirm the task maps to the exact existing Symtropy PHYS-TRUSS interface and supported features;
- independently derive or identify a permitted reference oracle for the exact input case;
- keep solver-visible task inputs separate from evaluator-only labels and hidden variants;
- include positive, malformed, mechanism/singular and boundary cases;
- record licences and provenance for every reused external drawing, model or dataset;
- define how invalid, unattempted, interrupted and unresolved tasks remain in the denominator;
- confirm the task does not leak an answer through filenames, fixtures, tests, repository history or prompt metadata;
- obtain appropriate educator review and age/accessibility review;
- obtain a separate risk assessment before any physical activity.

Do not turn a synthetic task into a claimed industrial benchmark or use an unqualified result to recommend a real bridge or building design.

## 9. Physical extension — not part of v0.1

A later physical demonstration may be considered only after the digital learning packet and safety review are complete. It would need a supervised, guarded non-human-load-bearing fixture, characterized materials and joints, a calibrated reference load and displacement measurement chain, controlled run procedure, explicit stopping conditions, retained raw observations, and independent review.

The physical protocol must be designed by competent adults and must follow Symtropy's physical-trial/metrology owners (#3522, #3591, #3592). The digital PHYS-TRUSS result is not a substitute for this process.

## 10. Rights, privacy and learner agency

- This packet is a reusable educational brief; source assets and task fixtures still require their own rights review.
- Children should not be asked to publish their names, precise location, school, images, voice or private project history to participate.
- Default to local/private project work; sharing and attribution are opt-in and governed by the appropriate Mycelix mechanisms.
- Make hints and explanations accessible without forcing a rigid attempt-first flow where an accommodation or learner need warrants another path.
- AI should ask questions, explain and propose—not silently take ownership of the learner's design.

## 11. Current status

The packet is **DESIGN_ONLY**. The numeric task card, solver integration against an exact owner-approved Symtropy head, learner interface, evaluator custody, educator review, physical fixture, and any human learning study are not yet implemented or qualified.

This draft is intended for review by the Symtropy truss owner, Praxis educators/researchers, and the CEB/PhysicsIR owners before implementation.
