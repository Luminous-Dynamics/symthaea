# Atelier Lightshell — Topology-Bound Manufacturing Route Contract v0.1

## Purpose

Translate the garment topology graph into a revision-bound manufacturing-operation graph. Each panel, connection, articulation feature and service feature receives an explicit operation disposition; every operation preserves input/output component lineage, dependencies and references to externally owned process capability.

This is a Lightshell adapter to MFG-PROC #5686. It does not create a new machine registry, process capability model, scheduler, Knitout/DXF exporter, machine program or execution API.

## Research context

An integrated robotic sewing study published 21 September 2026 combines vision-based fabric contour extraction, learning-based grasp selection, motion planning and force-regulated seam control using a single robot arm and an unmodified industrial sewing machine. The article reports seam deviation within ±2.5 mm across its test configurations, but still relies on user placement and user-selected seam start/end points. This is a physical sewing-execution comparator, not a pattern/topology or robot-garment qualification benchmark. https://doi.org/10.1007/s41315-026-00590-3

A 1 September 2026 apparel-manufacturing study aggregates seam-level pattern information into workstation workloads and uses a genetic algorithm to allocate sewing-machine resources in a simulated modular production system. It supports explicit seam-operation modelling as a manufacturing input, but the reported production improvements are from simulation, not physical robot-garment trials. https://doi.org/10.1108/IJCST-01-2026-0029

A 2026 *Journal of Manufacturing Systems* paper proposes a physics-informed embodied-intelligence framework for smart garment manufacturing, joining task-aware planning and robot fabric manipulation. This is a relevant comparator for future integration, not proof that a generic garment manufacturing line can yet run autonomously. https://doi.org/10.1016/j.jmsy.2026.01.015

## Operation graph

Each operation node binds:
- operation ID and stable ordering index;
- typed operation kind;
- explicit predecessor dependencies;
- exact topology panel/connection/articulation/service/garment references;
- material-lot, input-component and output-component identities;
- generic process profile, capability owner, machine, tooling, process/program revision and capability evidence;
- optional evidence-gated inspection gate;
- rework predecessor identity where applicable.

The dependency graph must be acyclic. Every input component must have one producer, and the producing operation must be an explicit direct dependency. Every output component has exactly one producer. Rework creates a new component revision and points to the failed predecessor; it never overwrites the earlier article identity.

## Connection-to-operation mapping

| Topology connection | Required operation |
| --- | --- |
| sewn_seam | seam_join |
| knit_integrated | knit_integrate |
| bonded_seam | bond_join |
| removable_attachment | removable_attach |

Panel operations include textile_forming or panel_cut. Articulation and service features must be referenced by an operation, not omitted from the manufacturing plan.

## Review states

- `proposed`: incomplete and synthetic routes may be represented; no manufacturing readiness.
- `process_review`: candidate process profiles and owner references are attached; capability may still be pending.
- `route_reviewed`: topology is CAD-bound and preflighted; every operation has qualified machine/tool/process/program evidence; every material lot has receipt provenance; all topology entities are covered; and final inspection has a method/fixture/acceptance plan.

There is deliberately no `executable`, `released` or `manufacturing_authorized` state in P0. A route-review pass is not permission to run a knitting machine, sewing cell or robot.

Inspection `not_run` is an admissible planned state at route review. Recorded pass/fail/inconclusive results require exact method, fixture revision, acceptance rule and result evidence. Failed or inconclusive inspection blocks route review.

## P0 negative controls

Authored tests cover dependency cycles and dangling references; duplicate output ownership; unproduced inputs and missing predecessor dependencies; connection/operation-kind mismatches; missing topology panel or connection dispositions; route review with unqualified process capability; missing robot/topology binding; inspection plan omissions/failure; missing material receipts; and rework identity preservation.

## Benchmarks

- 100% panel/connection/articulation/service topology coverage;
- zero cyclic or dangling dependencies;
- zero duplicated component output ownership;
- 100% input components trace to an explicit producing predecessor;
- 100% process/machine/tool/program capability binding before route review;
- 100% declared material lots have supplier/receipt identity;
- 100% final inspections have applicable method/fixture/acceptance references;
- measure route-authoring time, process travel/handling time, first-pass yield, rework rate, material waste, replacement cost and time-to-requalify on actual equipment.

Compare the same topology against expert-authored manual work instructions and machine allocation strategies, but hold the machine, material lot, operation scope and production volume constant. Report simulation-only results separately from physically executed process records.

## Current example

The fixture is a synthetic three-panel elbow route with material receipt, textile forming, two panel cuts, seam joining, removable attachment and final inspection. Machine, tool, program and process capability are deliberately unknown. Final inspection is planned but not executed. No route-review, fabrication or product release is claimed.

## Claim boundary

Route graph validity is a software consistency result. It is not machine code, operator instruction, execution permission, manufacturing authorization, textile physical qualification, certification or robot-safety approval.