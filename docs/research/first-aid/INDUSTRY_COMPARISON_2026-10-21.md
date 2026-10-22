# First-Aid Workflow Integrity: Industry Comparison Note

**Snapshot date:** 2026-10-22  
**Status:** research note; not clinical validation, certification, or deployment authorization.

## Executive finding

The relevant evaluation frontier is moving from "did the system produce a good answer?" to "did it finish the intended workflow, preserve state and evidence between steps, and leave the environment in the correct state?" This project can contribute a narrow, auditable safety-contract layer to that direction. It must not claim to outrank medical AI systems on clinical performance because it does not execute or evaluate the same task.

## Current external evidence

### NVIDIA: task completion plus process scoring

NVIDIA's September 21, 2026 engineering article, *How to Evaluate AI Agents From Tool Calls to Task Completion*, describes a practical evaluation stack with step-level process scoring and end-to-end environment-state verification. It recommends reporting success alongside consistency, cost and latency, and warns that tool-call scores alone do not establish goal completion. This independently converges with our design direction: score both the path and the final state, and make verification executable.

Source: https://developer.nvidia.com/blog/how-to-evaluate-ai-agents-from-tool-calls-to-task-completion/ (published 2026-09-21).


### Stanford HAI — October 19, 2026 task-execution update

Stanford HAI's latest benchmark synthesis reports three complementary levels of evaluation. MedHELM spans 121 clinical tasks across 22 subcategories and 37 datasets; its win-rate ranges are weakest for administration/workflow (0.53–0.63) and clinical decision support (0.63–0.77). PhysicianBench checks 100 long-horizon tasks using 670 task checks across 21 specialties; its best tested model completed 46% of tasks on one attempt and 28% across all three repeated attempts. HealthAdminBench's best tested agent completed 36.3% of administrative tasks end-to-end even though the best cited subtask rate was materially higher. These benchmarks use different systems and graders; compare directionally, not as a common leaderboard.

The central finding relevant to this work is that an agent can narrate an action it never actually completes, lose state while moving across documents/systems, or pass many subtasks while failing end-to-end. That supports measuring state transitions, externalized outcomes, retries, omissions, and repeatability—rather than grading only final text.

Source: https://hai.stanford.edu/news/measuring-what-medical-ai-can-actually-do (published 2026-10-19; accessed 2026-10-22).

### AgentClinic and tool-using clinical systems

AgentClinic is a multimodal simulated clinical-agent benchmark covering nine specialties and seven languages, explicitly designed around sequential patient interaction, incomplete information, and tool use: https://www.nature.com/articles/s41746-026-02674-7 (published 2026-04-27).

In its reported AgentClinic-MedQA setup, Claude 3.5 Sonnet reached 62.1% diagnostic accuracy with a GPT-4 patient agent, while results differ materially with patient-agent configuration, interaction budget and tool choice. The benchmark also reported average relevant-information coverage of 67% in sequential conversations; coverage was 72% on correctly diagnosed cases versus 63% on incorrect diagnoses. The authors caution that patient-agent configuration affects results and that benchmark tool actions are simpler than real clinical workflows. This is important context: accurate state tracking is necessary, but it is not a clinical success metric.

A separate 2026 study compared general-purpose agent systems across AgentClinic, MedAgentsBench, and Humanity's Last Exam. It reported up to 60.3% accuracy on AgentClinic MedQA and 28.0% on the MIMIC-based setting for the evaluated systems, with higher resource costs (>10× token usage and >2× latency) and residual hallucinations despite in-agent safeguards. These are study-specific findings, not a universal ranking of all current models: https://www.nature.com/articles/s41746-026-02443-6 (published 2026-02-18).


### Clinical AI evaluation literature

- Ruhrberg Estévez et al., *How to benchmark medical AI agents*, PLOS Medicine, 2026-07-09, argues that agentic benchmarks should cover action appropriateness, process safety and resource stewardship, not only final-answer matching: https://doi.org/10.1371/journal.pmed.1005170
- *State of clinical AI in 2026*, BMJ Digital Health & AI, reports persistent weaknesses in uncertainty calibration, robustness to distribution shift and real-world workflow performance: https://bmjdigitalhealth.bmj.com/content/2/1/e000113
- CSEDB uses 30 metrics and 2,069 expert-developed open-ended items; its authors report an average performance drop in high-risk scenarios. It is a complementary clinical response benchmark, not a direct comparator for a state-machine checker: https://www.nature.com/articles/s41746-025-02277-8

## Snapshot refresh: 2026-10-22

The [Clinical Benchmarks index](https://clinicalbenchmarks.ai/benchmarks), which reports data current to **October 12, 2026**, tracks ten benchmark boards with multiple model results. Its [PhysicianBench board](https://clinicalbenchmarks.ai/benchmarks/physicianbench) lists 28 result rows across 20 models, based on 100 long-horizon tasks, 21 specialties and 670 structured checkpoints. The latest listed row at this snapshot is a vendor-reported 68.4% Pass@1 for Claude Opus 5.5 (max, Anthropic harness; measured September 2026).

That number differs from the 46% headline in Stanford HAI's October 19 synthesis because the model set, harness, timing, grading and aggregation differ. **Do not average or directly compare these figures.** The correct research conclusion is that results depend materially on the evaluated agent configuration and benchmark conditions; leaderboard snapshots should name source, model version, harness, date, metric, and uncertainty.

The official [HealthAgentBench leaderboard](https://microsoft.github.io/HealthAgentBench/) reports 54 tasks across seven terminal-based healthcare environments and scores pooled task success over three attempts per task. Its current published table lists Claude Code (Opus 5) at 55% success and $3.3 per task sweep, and Codex (GPT-5.6-sol) at 45% and $5.2; other agent/harness combinations trade off cost and success. These are benchmark-specific results, not universal rankings or results for this project.

The same index also tracks **CHI-Bench** (75 long-horizon healthcare operations workflows involving prior authorization, utilization management and care management). It says the AgentClinic board has no public frontier-model results since 2025, even though the original 2026 paper remains useful for its sequential-interaction and bias-design lessons. These boards are more relevant end-to-end design references than saturated exam sets, but their completion rates cannot be compared directly with our synthetic state-machine tests. A fair adapter experiment should run a shared set of workflow histories through the target agent harness plus the same independent invariant verifier, then separately report workflow completion, critical-invariant violations, benign false positives, repeatability over multiple runs, latency and cost.

The index also lists separate boards for workflow agents, safety and clinical reasoning; high scores on near-saturated exam benchmarks are not evidence of robust long-horizon workflow execution. Relevant sources:
- Stanford HAI, 2026-10-19: https://hai.stanford.edu/news/measuring-what-medical-ai-can-actually-do
- Clinical Benchmarks index, last updated 2026-10-12: https://clinicalbenchmarks.ai/benchmarks
- PhysicianBench result rows and methodology: https://clinicalbenchmarks.ai/benchmarks/physicianbench

## How this project should compare

Treat external benchmarks as design references. A like-for-like comparison requires shared task definitions, a common dataset, the same execution environment, pre-registered metrics, and reference outcomes adjudicated independently of the system being tested.

The project's present scope is **deterministic validation of synthetic process traces**: bundle provenance and revocation, session trust state, uncertainty handoff, freshness claims, connectivity assumptions, and consent-scoped egress. The latest branch work changes connectivity from a trace-wide constant into a sequence of explicit `connectivity_changed` events, so an online-to-intermittent/offline transition is evaluated at the subsequent network-dependent action, and recovery affects later events only. The finite test model now accounts for 252 states, 5,040 transitions, and 100,800 bounded two-action suffixes; its state-model/production comparison is maintained in the same repository and is not independent ground truth. It does not assess first-aid knowledge, clinical recommendations, actual patient workflows, or outcomes.

The current machine-readable coverage inventory contains 44 entries (12 frozen traces, 32 mutation/control cases, 30 executed by the Rust matrix runner, and two tied to explicit workflow gates). These are test definitions and gate mappings, not a claim that the latest head passed: exact-head validation remains pending. Do not compare the corpus to hundreds of clinical tasks or present a local pass rate as superiority over an industry model. The proper near-term contribution is a reproducible, inspectable, machine-checkable process-integrity contract that could later serve as one verifier inside a larger workflow benchmark.

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
