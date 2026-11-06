# ENG-KNOW-CO-DESIGN-001 — design/inspection co-optimization extension (proposal v0.2)

**Parent benchmark:** [ENG-KNOW-QUAL-001 #10652](https://github.com/Luminous-Dynamics/symthaea/issues/10652)  
**Current blueprint:** [engineering-knowability README](../../benchmarks/engineering-knowability/README.md)  
**Status:** research proposal only; no scored cases, no optimizer, no physical execution authority.

## Thesis

Move from:

`design → manufacture → discover inspection is difficult → redesign or accept weak evidence`

to:

`requirements + claims + geometry + process + service lifecycle → jointly design the article and the credible evidence paths that will be needed to verify and maintain it`

This is not the addition of another metrology engine. It is an evaluation of whether a design agent can use canonical engineering, measurement, tolerance, fabrication, lifecycle and evidence projections while preserving their existing owners.

The benchmark should reward a system that discovers *before design freeze* that a critical property cannot be observed on the final article, and then proposes a technically admissible redesign, a different inspection method, or a narrower claim. It must not reward a fabricated measurement path.

## Why the field is ready for this shift

### External engineering-agent benchmarks

The public comparators are now materially stronger than simple CAD-code-generation demonstrations:

| Benchmark | Current public reference | What it evaluates | What it does not establish |
|---|---|---|---|
| [Parametric CAD Bench V3](https://cadbench.ai/) | 100 FreeCAD tasks: 30 text creation, 30 create-and-edit, 40 drawing reconstruction. The retrieved leaderboard shows Claude Opus 5.5 + Claude Code at 61.03% mean reward (99/100 scored, 9/100 perfect; 95% interval displayed as ±7.53). | Geometry/specification consistency, native parametric history, spatial alignment, topology and workflow. | Post-assembly inspection access, calibrated measurement capability, physical conformity or service requalification. |
| [CADWorld](https://cad-world.github.io/) | 200 FreeCAD tasks, 11 workflow categories. Its public extended-run page reports 41.5% best evaluated-agent success against an 87.0% single-trial expert reference. | Long-horizon GUI execution and persistent CAD/FEM/CAM/measurement artifacts under task-specific executable checks. | A statistically characterized human baseline or universal engineering/measurement-system qualification. |
| [BenchCAD 2.1](https://benchcad.com/leaderboard) | Separate standard and customized-harness boards; the standard v2.1 board reports five passes over 100 Core cases. | Reconstruction, assemblies, placement and ECAD connectivity under its pinned scoring protocol. | Manufacturing readiness, tolerance capability, electrical safety or physical function. |

These are complete model-plus-agent systems, not model-only scores. Their scores use different task universes, execution surfaces and scoring functions; they must not be averaged or ranked against each other.

### Standards and industrial direction

- NIST's [Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing) describes design/product-definition data flowing to manufacturing and quality systems, with inspection and measurement results feeding back to design. STEP, QIF and MTConnect are part of the standards landscape.
- NIST's [2026 AI/ML Roadmap for Smart Manufacturing](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing) explicitly includes data-centric metrology, advanced digital twins, explainability and reliability among the research directions for industrial AI.
- The 2024 NORDDESIGN paper, [Design for Inspectability: Identifying Its Requirements and Underpinning Enabling Technologies](https://datainformed.designsociety.org/download-publication/47632/Design%2Bfor%2BInspectability%3A%2BIdentifying%2BIts%2BRequirements%2Band%2BUnderpinning%2BEnabling%2BTechnologies), defines DfI as a design-for-X concern and identifies nine requirements plus four enabling technology families. It is a conceptual/design-method contribution, not a public agent benchmark.
- An August 2026 review of [smart metrology in Industry 4.0–5.0](https://www.sciencedirect.com/science/article/pii/S2665917426000279) highlights interoperability, real-time uncertainty evaluation and trustworthy/auditable AI as outstanding challenges.
- [ISO 14253-1:2017](https://www.iso.org/standard/70137.html) supplies uncertainty-aware conformity decision rules for in-scope quantity-valued characteristics. The [2026 GUM amendment](https://www.bipm.org/documents/20126/2071204/JCGM_100_Amd1_2026.pdf/) addresses nonlinearity in measurement models.

The opportunity is not to claim that industry has no DfI methods. It is to make DfI *machine-evaluable across design, assembly and service states*, with exact references, independent grading and explicit uncertainty/authority boundaries.

## Co-design task contract

Each task should contain:

- A frozen requirement/claim set with criticality and acceptance semantics.
- A native CAD/assembly starting subject and exact feature/datum identities.
- A declared manufacturing/process family and its capability envelope.
- A final assembly state and a lifecycle sequence (inspection, closure, service, remount or replacement).
- Measurement method/instrument profiles, access constraints and permitted standards references.
- Explicit information gaps. The candidate must be able to return Unknown or request a missing owner projection.

The agent proposes one or more **design alternatives** plus their evidence plans. The output should bind:

1. Exact design and native feature-history revision.
2. Requirement/claim disposition for each critical characteristic.
3. Measurand and direct/derived observation class.
4. Datum, coordinate frame, article and lifecycle state.
5. Inspection access, instrument/method family and setup/fixture concept.
6. Calibration/reference dependencies and applicable scope.
7. Measurement equation, uncertainty/dependence model references and decision rule.
8. Requalification obligations after closure, remount, replacement or repair.
9. Unresolved assumptions, Unknown/Unsupported states and claim ceiling.
10. Resource/impact vector for the design and inspection plan.

The candidate does not issue a calibration PASS, create physical evidence, approve the design, select an unauthorized experiment, or authorize fabrication.

## New task families for a separately versioned extension

These are authoring blueprints, not gold labels. They must be independently adjudicated before use.

| ID | Design conflict | Correct system behavior |
|---|---|---|
| CDK-01 — inaccessible cavity | A closed enclosure hides a critical wall/floor characteristic. | Add a justified inspection affordance, preserve narrowly scoped pre-closure evidence, choose a valid alternative method, or narrow the claim. |
| CDK-02 — datum destroyed by service | A replaceable component carries a datum/registration surface that is removed during service. | Preserve an independent remount reference or reopen the affected registration and conformity obligations. |
| CDK-03 — inspection port competes with performance | An access opening improves inspection but weakens a load path, seal or environmental requirement. | Generate alternatives and preserve all critical engineering constraints; do not maximize access at the cost of function. |
| CDK-04 — CMM versus optical scope | The proposed instrument does not match the geometry, material, access or measurement standard scope. | Select a feasible method with applicable capability evidence, or report unresolved; never apply ISO 10360-2 as a generic optical-scanner rule. |
| CDK-05 — common-mode illusion | Two observations share a datum, calibration artifact or fixture. | Represent dependence; do not claim independent corroboration. |
| CDK-06 — uncertainty versus tolerance | A measurement can be obtained but cannot resolve the required decision boundary. | Change the method/design, improve the measurement chain, or return insufficient capability. |
| CDK-07 — logical versus physical testability | ECAD net connectivity is correct, but no routed board, test points or instrument/isolation profile is supplied. | Preserve logical correctness while returning physical testability as Unknown. |
| CDK-08 — destructive inspection trade-off | A destructive test provides strong evidence but consumes the article or interrupts service. | Make the trade-off explicit; identify sampling/lot inference assumptions and non-destructive alternatives. |

Each case needs a positive/negative pair, a hidden evaluator mutation and at least one lifecycle change. Split by base geometry, process family and measurement-chain family—not random row.

## Baselines and ablations

Use matched tasks, tool budgets, time limits and exact versions:

1. Design-first agent, with inspection considered only after design is frozen.
2. Conventional checklist/template DfI workflow.
3. Deterministic rule-based access/measurement planner.
4. Frontier multimodal CAD agent with the same CAD tool access but no canonical measurement/evidence projections.
5. Human mechanical-design/metrology team.
6. Symthaea proposal + canonical owner projections.
7. Symthaea ablations: provenance-aware retrieval, causal hypothesis maintenance and FEP value-of-information selection, only after their own contracts are qualified.

Do not compare a rich Symthaea configuration with a weaker frontier system that receives fewer tools or less information. Record model, harness, effort, tools, retries, task revision, runner, CAD kernel, time, cost and every terminal disposition.

## Scoring: no composite score

### Hard, non-compensatory integrity gates

- Zero critical false-observability passes on adjudicated negative controls.
- Zero false calibration-currentness, instrument identity, independence or physical-evidence claims.
- 100% of critical claims have an explicit disposition: admissible path, review, Unsupported or Unknown.
- Every asserted path binds the correct feature, datum, article/configuration generation and final lifecycle state.
- No score laundering through omitted, ungraded or unattempted cases.

A failure on any hard gate is reported separately and cannot be offset by geometry reward, speed or lower inspection cost.

### Continuous and operational metrics

- Requirement preservation and native feature-history validity.
- Claim-to-measurand/datum mapping accuracy.
- Final-assembly access and method-scope correctness.
- Calibration/reference applicability and independence correctness.
- Measurement model and uncertainty/decision-rule correctness.
- Service/remount requalification completeness.
- Unknown/abstention precision and recall.
- Late-discovered inspection blockers versus design-first/checklist baseline.
- Inspection setup time, expected inspection cost, destructive impact and service downtime.
- Design mass, material/process burden and functional performance—reported as distinct dimensions.
- Pareto-frontier coverage and dominated-design fraction.

Do not add unlike dimensions into one opaque “knowability score”. A Pareto improvement is meaningful only when the underlying engineering constraints and claim gates are preserved.

## Prospective targets

After a pilot establishes assessor agreement and baseline variance:

- **Integrity:** zero critical false-observability/calibration/independence passes on the sealed negative-control set.
- **Disposition:** 100% explicit disposition of critical claims.
- **Mapping:** at least 95% correct accepted-claim to measurand/datum/feature mapping on adjudicated pilot families.
- **Early prevention:** at least 30% fewer late-discovered inspection/requalification blockers than matched design-first/checklist workflows, with non-inferior critical-claim recall.
- **Efficiency:** improve the cost/time Pareto frontier without increasing critical false claims or weakening required functional constraints.
- **Transfer:** retain the integrity gates on held-out geometry and measurement-chain families, not just variants of the training/public fixtures.

These are research targets, not established industry standards or current results. Numeric efficiency thresholds should be preregistered only after a pilot and inter-assessor study.

## Staged execution

- **V0 — contract:** owner review; define the task and evidence-path projection without a new schema authority.
- **V1 — synthetic digital tasks:** geometry/access analysis and deterministic evaluator; no physical claims.
- **V2 — native CAD artifacts:** verify edits, feature history, assembly state, access and inspection plan in pinned FreeCAD/CadQuery/other declared CAD kernels.
- **V3 — fixture-level demonstration:** only through ENG-REF #6121 with actual fixture/article identities, approved instruments, calibration and measurement procedures.
- **V4 — production/field evidence:** only where process, supplier, quality and safety authorities admit the exact evidence.

## Claim ceiling

A successful V1/V2 result establishes only that the evaluated candidate produced better design/inspection proposals on exact synthetic/native CAD tasks. It does not establish measurement-system capability, calibration, physical observability, article conformity, product safety, regulatory compliance or design release authority.
