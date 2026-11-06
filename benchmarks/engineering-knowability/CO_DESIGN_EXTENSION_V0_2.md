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

It must also preserve the exact design/feature/datum/PMI identities and inspection-result links across declared exchange surfaces. Compose with [MBD-INTEROP-001 #10595](https://github.com/Luminous-Dynamics/symthaea/issues/10595), which owns the AP242 Edition 4 ↔ QIF / ISO 23952 ↔ MTConnect semantic round-trip benchmark. Do not duplicate its translator, schema, GUID or semantic-coverage implementation. A visually intact PMI annotation with a lost feature/datum association is a hard failure for the co-design task.

## Why the field is ready for this shift

### External engineering-agent benchmarks

The public comparators are now materially stronger than simple CAD-code-generation demonstrations. The comparisons below are deliberately separated by task split, scorer and execution surface.

| Benchmark | Current public reference | What it evaluates | What it does not establish |
|---|---|---|---|
| [Parametric CAD Bench V3](https://cadbench.ai/) | Claude Opus 5.5 + Claude Code: 61.03% overall (99/100 scored; displayed 95% interval ±7.53). GPT-6 Astra + Codex leads create-and-edit at 44.37%. | 100 FreeCAD tasks: text creation, create-and-edit, and engineering-drawing reconstruction; native CAD artifacts checked by a separate programmatic verifier. | Physical measurement capability, manufacturing approval, calibrated conformity or service requalification. |
| [CADWorld released v0](https://cad-world.github.io/) | GPT-5.4 CUA: 25.0% on 60 released tasks; human expert single-trial reference: 87.0%. | Long-horizon native FreeCAD GUI workflows across 11 categories, with all-checks-pass saved-artifact grading and a 100-step cap. | The 87% reference is not a population human mean. The 200-task universe is not the released 60-task v0 cohort. |
| [CADWorld extended-run snapshot](https://cad-world.github.io/) | Opus 5.5 EXT: 41.5%; Sonnet 5.5 EXT: 33.5%; GPT-6 Astra EXT: 33.0%. | Later indexed result snapshot with 14 listed agents. | Retrieved summary does not expose a sufficiently clear task denominator to compare with the v0 cohort. |
| [BenchCAD 2.0 Core](https://benchcad.com/blog/benchcad-2) | GPT-6 Astra: 0.59; Claude Opus 5.5: 0.57. | 100 held-out hardware-engineering cases under the official v2.0 harness. | Manufacturing readiness, calibrated metrology, lifecycle requalification or physical function. |

These are full model-plus-agent systems, not model-only scores. PLE's public-dev (29), hidden (56) and all-task (85) results are distinct. On the 29-task dev board, Claude Opus 5.5 ranges from 0.474 with pi to 0.551 with Claude Code; the benchmark reports three runs per pairing. <Cite refs={["turn348316search0","turn348316search2","turn181350search3"]}/>

BenchCAD's current official release is v2.0. Do not pool its 0.59 / 0.57 Core scores with vendor-reported Vision2Code scores on different subsets or with unverified later version labels. <Cite refs={["turn437429search1","turn437429search2"]}/>

### Standards and industrial direction

- NIST's [Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing) describes design/product-definition data flowing to manufacturing and quality systems, with inspection and measurement results feeding back to design. STEP, QIF and MTConnect are part of the standards landscape.
 - NIST's [2026 AI/ML Roadmap for Smart Manufacturing](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing) explicitly includes data-centric metrology, advanced digital twins, explainability, reliability, availability, maintainability and safety among industrial AI research directions.
- NIST's [2026 AI/ML Roadmap for Smart Manufacturing](https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing) explicitly includes data-centric metrology, advanced digital twins, explainability and reliability among the research directions for industrial AI.
- The 2024 NORDDESIGN paper, [Design for Inspectability: Identifying Its Requirements and Underpinning Enabling Technologies](https://datainformed.designsociety.org/download-publication/47632/Design%2Bfor%2BInspectability%3A%2BIdentifying%2BIts%2BRequirements%2Band%2BUnderpinning%2BEnabling%2BTechnologies), defines DfI as a design-for-X concern and identifies nine requirements plus four enabling technology families. It is a conceptual/design-method contribution, not a public agent benchmark.
- An August 2026 review of [smart metrology in Industry 4.0–5.0](https://www.sciencedirect.com/science/article/pii/S2665917426000279) highlights interoperability, real-time uncertainty evaluation and trustworthy/auditable AI as outstanding challenges.
- [ISO 14253-1:2017](https://www.iso.org/standard/70137.html) supplies uncertainty-aware conformity decision rules for in-scope quantity-valued characteristics. The [2026 GUM amendment](https://www.bipm.org/documents/20126/2071204/JCGM_100_Amd1_2026.pdf/) addresses nonlinearity in measurement models.

The opportunity is not to claim that industry has no DfI methods. It is to make DfI *machine-evaluable across design, assembly and service states*, with exact references, independent grading and explicit uncertainty/authority boundaries.

## Co-design task contract

The eight paired authoring families are enumerated in [codesign-paired-task-matrix.csv](codesign-paired-task-matrix.csv). Each defines a matched design-first baseline and a knowability/co-design alternative, the lifecycle mutation, critical claim/evidence requirements, hard failure gate and canonical owner inputs. This is an authoring matrix only; it does not make the fixtures scored or gold-adjudicated.

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


## Current execution owner

The co-design benchmark track is now tracked in [ENG-KNOW-CODESIGN-001 #10702](https://github.com/Luminous-Dynamics/symthaea/issues/10702). This issue owns the matched design-first/checklist/knowability-conditioned/co-design experiment, non-compensatory evaluator mutations, lifecycle economics and Pareto-preserving report. This proposal remains design-only until those tasks have independent gold adjudication and owner review.


## Statistical preregistration boundary

The eight paired rows are a pilot/authoring tranche, not a powered confirmatory benchmark. The independent experimental unit is the base design family—not each CAD variant, mutation, seed or model call. Confirmatory tasks must hold out complete geometry/assembly, process and measurement-chain families. The co-design issue #10702 now requires matched arms, randomized order, sealed adjudication, exact task denominators, paired binary analysis (e.g. exact McNemar), family-cluster bootstrap for continuous cost/time outcomes, and multiplicity control for co-primary comparisons.

For zero false accepts on (n) independent negative-control families, the one-sided 95% exact binomial upper bound is (1-0.05^{1/n}): 59/59 is below 5%, and 299/299 is below 1%. Mutations/variants from one base family are correlated and do not increase (n). These bounds apply only to the declared sampling frame; no universal safety or reliability guarantee follows.

The eight-row matrix is only a pilot. No claim that co-design beats a frontier agent or reduces lifecycle cost is permitted until an independently adjudicated, preregistered confirmatory study is completed.


## Candidate / evaluator trust boundary

The machine-readable [candidate submission contract](candidate-output.schema.json) deliberately excludes candidate scores, hidden gold labels, evaluator mutations and hidden-case metadata. A candidate can return editable design alternatives, explicit critical-claim dispositions, proposed evidence paths, Unknown/Unsupported states, owner-evidence requests and resource usage. Its allowed evidence classes for this synthetic v0.1 task contract are `synthetic_reference`, `proposal` and `simulation`.

The separate [evaluator-only gold contract](gold-adjudication.schema.json) requires at least two independent assessor records, declared assessor independence, a third-adjudicator route when disagreement exists, independent mutation-oracle references and a sealed custody record. It is not candidate-visible.

Neither schema creates an evidence authority. The candidate proposes; independent scoring adjudicates; ETK/ENG-MEAS/ENG-TOL/FIELD/ENG-REF retain their existing decision authority.
