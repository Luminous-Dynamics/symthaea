# First-Aid Workflow Integrity: Industry Comparison Note

**Snapshot date:** 2026-10-21  
**Status:** research note; not clinical validation, certification, or deployment authorization.

## Executive finding

The relevant evaluation frontier is moving from "did the system produce a good answer?" to "did it finish the intended workflow, preserve state and evidence between steps, and leave the environment in the correct state?" This project can contribute a narrow, auditable safety-contract layer to that direction. It must not claim to outrank medical AI systems on clinical performance because it does not execute or evaluate the same task.

## Current external evidence

### Stanford HAI: task execution rather than answer scoring

Stanford HAI's October 19, 2026 synthesis, *Measuring What Medical AI Can Actually Do*, reports results from several task-oriented benchmarks:

- **MedHELM:** a taxonomy developed with 29 clinicians spanning 5 categories, 22 subcategories and 121 tasks, evaluated with 37 benchmark datasets. The reported model win-rate range for administration/workflow was 0.53–0.63, lower than the ranges reported for note generation and patient communication.
- **PhysicianBench:** 100 long-horizon clinical tasks across 21 specialties, with 670 checks for retrieval, reasoning, action execution and documentation. The article reports the best tested model completed 46% of tasks on a single attempt; only 28% of tasks were properly completed in all three runs.
- **HealthAdminBench:** 135 administrative tasks across simulated EHR, payer and fax systems. The best tested agent completed 36.3% end-to-end. GPT-5.4 reportedly passed 82.8% of subtasks but completed only 26.7% of whole tasks.

These percentages are **not interchangeable scores**. Stanford explicitly cautions that PhysicianBench and HealthAdminBench differ in environments, model sets, interfaces and grading. The useful conclusion is directional: high performance on individual steps does not guarantee end-to-end success or repeatability.

Source: https://hai.stanford.edu/news/measuring-what-medical-ai-can-actually-do (published 2026-10-19; accessed 2026-10-21).

### Clinical AI evaluation literature

- Ruhrberg Estévez et al., *How to benchmark medical AI agents*, PLOS Medicine, 2026-07-09, argues that agentic benchmarks should cover action appropriateness, process safety and resource stewardship, not only final-answer matching: https://doi.org/10.1371/journal.pmed.1005170
- *State of clinical AI in 2026*, BMJ Digital Health & AI, reports persistent weaknesses in uncertainty calibration, robustness to distribution shift and real-world workflow performance: https://bmjdigitalhealth.bmj.com/content/2/1/e000113
- CSEDB uses 30 metrics and 2,069 expert-developed open-ended items; its authors report an average performance drop in high-risk scenarios. It is a complementary clinical response benchmark, not a direct comparator for a state-machine checker: https://www.nature.com/articles/s41746-025-02277-8

## How this project should compare

Treat external benchmarks as design references. A like-for-like comparison requires shared task definitions, a common dataset, the same execution environment, pre-registered metrics, and reference outcomes adjudicated independently of the system being tested.

The project's present scope is **deterministic validation of synthetic process traces**: bundle provenance and revocation, session trust state, uncertainty handoff, freshness claims, connectivity assumptions, and consent-scoped egress. It does not assess first-aid knowledge, clinical recommendations, actual patient workflows, or outcomes.

Do not compare the current trace corpus's 12 cases to hundreds of clinical tasks or present a local pass rate as superiority over an industry model. The proper near-term contribution is a reproducible, inspectable, machine-checkable process-integrity contract that could later serve as one verifier inside a larger workflow benchmark.

## Proposed layered benchmark

| Layer | What is measured | Target before advancing |
|---|---|---|
| L0 — deterministic contract | Required schema, known transitions, frozen answers, exact-head receipts | 100% frozen-case agreement; zero false passes in declared receipt mutations |
| L1 — fault containment | Seeded mutations, benign controls, invariant-level detection and false positives | 100% detection for each declared safety-critical mutation; 0% false positives on independently labelled synthetic controls |
| L2 — replay reliability | Same checker and same trace executed repeatedly, including clean process restarts | 100% output/digest repeatability over 1,000 replays |
| L3 — independent verifier | A separate implementation or formal state model compared on held-out generated traces | 100% agreement on the defined contract; every disagreement retained and categorized |
| L4 — workflow execution | Environment changes, required artifacts, state reconciliation, omissions and recovery | 100% for hard-gate invariants; publish end-to-end completion and subtask rates separately |
| L5 — domain relevance | Qualified review of intended invariants and acceptable workflow variation | Report reviewer agreement and disagreements; no silent conversion of judgment into ground truth |
| L6 — clinical evaluation | Human factors, relevant ethics/governance, prospective outcomes and monitoring | Separate authorized study protocol and its predefined endpoints; never inferred from L0–L5 |

All targets are prospective. The current code work does not yet demonstrate L1–L6 completion.

## Metrics that avoid benchmark gaming

1. **End-to-end completion rate:** all required end-state checks pass, not the fraction of subtasks completed.
2. **Critical invariant violation rate:** report counts per invariant and consequence class. A critical violation cannot be offset by a correct answer elsewhere.
3. **Three-run consistency:** same task passed in all three repeated runs; report alongside single-run success rather than averaging it away.
4. **Evidence-state integrity:** percentage of actions backed by the exact current, unrevoked and verified evidence version.
5. **Uncertainty preservation:** rate of uncertainty lost, erased, or contradicted during transition and handoff.
6. **Recovery quality:** percentage of interrupted workflows that resume only from validated state, with no implicit restoration of stale authorization.
7. **Resource stewardship:** unnecessary calls, redundant actions, retries, cost, latency and irreversible actions attempted without required evidence.
8. **False-positive burden:** benign controls incorrectly blocked, separated by invariant and context.
9. **Reproducibility:** exact source/corpus/protocol/evaluator digests, runtime environment, raw receipts, and pinned revision for every score.

## Recommended research path

First finish Q0: ensure the latest PR head passes formatting, tests, Clippy, mutation-matrix integrity, and receipt gates. Then make the mutation inventory executable and add generated traces with independently specified outcomes. Next, write a separately structured formal model or second implementation and compare them on a held-out corpus. Only after domain experts adjudicate the invariants should any workflow-level clinical study be designed.

The innovation claim should remain narrow and testable: **process-state guarantees are explicit, fault-injected, replayable, and bound to evidence**. The work will be stronger if its public contribution is a dependable verifier and transparent failure inventory, rather than an unsupported claim of clinical state-of-the-art.
