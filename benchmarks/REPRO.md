# External benchmark reproduction guide

**Registry:** [`manifest.json`](manifest.json)  
**Schema:** [`manifest.schema.json`](manifest.schema.json)  
**Status:** metadata-only registry; no external suite is currently qualified for execution from this checkout.  
**Last updated:** 2026-11-05

## Rules

1. The registry is the single metadata index for external benchmarks. Do not create another per-project registry or duplicate CEB's task/receipt schemas.
2. A registry record is not a runnable adapter. Status must say whether a suite is metadata-only, access-gated, adapter-in-progress, runnable, qualified, or retired.
3. Pin the exact task/data revision, runner revision, environment/image, toolchain, model/harness/configuration, network policy, time/resource budget, grader revision, and repeat policy before a result is called reproducible.
4. Preserve split boundaries and the exact task universe. Do not train or tune on a hidden test set or copy hidden task/grader material.
5. Distinguish benchmark-maintainer leaderboard results, independent third-party evaluations and local reproductions. Never compare different splits, versions, harnesses, model effort, resource budgets or score definitions as though they were one experiment.
6. Report per-task outcomes, all eligible/attempted/graded denominators, abstentions, refusals, timeouts, tool failures, ungraded cases and repeated-run variability. Missing or ungraded tasks do not silently disappear from a denominator.
7. Use the benchmark maintainer's official evaluator where available. Do not rewrite a grader and claim equivalent scoring without an independent equivalence study.
8. A schema-valid manifest does not authorize access, execution, security claims, engineering approval or performance claims.

## PLE-to-Luminous evidence mapping

**Do not force PLE deliverables into CEB's patch-centric v0 receipt.** The current CEB v0 schema requires a candidate patch, reference-patch hash and source/result tree subjects. PLE grades engineering deliverables such as calculations, technical reports, drawings, CAD/BIM artifacts and schedules; a source patch is not the semantic unit of result. Treating a PLE deliverable as a patch would misstate what was evaluated.

The adapter boundary must be resolved through the existing **benchmark-neutral run-accounting contract #10314** and **evidence-convergence issue #10370**. CEB #10115 / DSSE #10167/#10212 remains authoritative for Symthaea code-change tasks that actually have source patches. Do not add a third receipt schema or alter the meaning of the CEB patch subject.

| PLE evidence | Canonical Luminous mapping | Invariant |
|---|---|---|
| Official benchmark/split/task universe | Generic benchmark identity + complete task-universe commitment (#10314) | Every eligible PLE task has exactly one terminal disposition per run |
| Dataset revision + PLE task ID + task bundle bytes | Task-row input identity and digest | Record the Hugging Face dataset commit and exact task ID; compute a separate digest over the permitted materialized task bundle |
| Agent/harness/model/configuration | Candidate/agent/harness identity in the generic run record | Pin model, provider, effort, prompt, harness, tool policy and retry policy |
| Task/serve/verifier images and Harbor/runner | Environment/execution identity | Pin immutable image digests, official PLE runner commit, Harbor 0.20.0, platform and resource limits |
| Official `checks`, `rubric`, `checks-judge` grader | Grader identity, grading protocol and raw grader output digest | Use official verifier unchanged; preserve the grader type and judge model/revision |
| Deliverable files, trajectory, logs and resource data | Evidence-artifact inventory | Hash every artifact; do not invent a patch hash for a report, drawing or STEP file |
| Official per-task reward and hard-check outcomes | Official score plus separately derived coverage/yield/feasibility metrics | Never replace official score with a local approximation; never average away hard-gate failure |
| Repetition policy | Published dev leaderboard: three-run average; competition-entry dev score: committed artifacts plus live sample check; hidden competition: three-run average | Keep official and local repetition policies separate; do not confuse `--trials` parallelism or evaluator retries with independent repetitions |

The adapter may use a small **benchmark-profile mapping** to project PLE task metadata into generic accounting fields. That projection must not duplicate PLE task prompts, tests, graders or hidden labels, and it must not change official scoring semantics. A CEB task ID may identify the Luminous adapter/profile work, but the external PLE task ID remains a separate source identifier.

Before execution, #10370 must decide normative ownership for benchmark/task identity, accounting rows, artifact digests, official score provenance, and interaction with CEB's DSSE receipts. If those contracts are not yet reconciled, the PLE score remains an external source observation attached to its own official artifacts—not a qualified CEB run.

## PLE — metadata-only procedure

The first registered suite is Physical-Engineering's Last Exam (PLE).

Authoritative public references:

- Official site / leaderboard: https://ple-bench.com/
- Official Apache-2.0 runner kit: https://github.com/google-research/ple
- Official CC-BY-4.0 development dataset (access-gated): https://huggingface.co/datasets/google/physical-engineerings-last-exam
- Paper, arXiv v2 revised 2026-10-30: https://arxiv.org/abs/2610.30537

### Current metadata

The benchmark contains 85 tasks: structural 10, civil 18, electrical 27 and mechanical 30. Its public development split contains 29 tasks: 4 structural, 9 civil, 9 electrical and 7 mechanical. The hidden test split contains 56 tasks and must remain in the official competition trust boundary.

The dataset card identifies 12 `checks`, 14 `rubric` and 3 `checks-judge` dev tasks. Seventeen tasks require the official evaluation runner to build task-specific verifier images; standalone Harbor is not a full substitute. The official runner requires Harbor 0.20.0, uses `eval/run.py`, and enforces a six-hour per-task limit. The published public development leaderboard reports model results averaged over three runs. The competition submission dev score is different: it grades committed artifacts and runs the harness live on a sample to check reproducibility. The official hidden competition leaderboard also averages three harness runs. A Luminous matched dev comparison should use three independently identified full-set runs to align with the published dev baseline.

### Required gates before any run

- [ ] PLE dataset access is granted to the authorized account and its exact CC-BY-4.0 terms are accepted.
- [ ] Exact dataset commit is pinned and recorded. The official runner downloads the latest dataset revision by default; do not rely on a moving `latest` for reproducible comparison.
- [ ] Exact official `google-research/ple` source commit, Harbor 0.20.0, task/verifier image digests and all dependencies are pinned.
- [ ] CEB task/receipt mappings are reviewed against the exact CEB v0 schema; do not introduce a second task or receipt format.
- [ ] Agent-visible task files and evaluator/grader contents are separate trust domains under CEB #10436.
- [ ] The Luminous execution boundary is independently qualified under Symthaea #10064. A host-opt-in runner, Nix development shell or container label alone does not establish isolation.
- [ ] Model endpoint and judge endpoint are explicitly configured, secrets are held outside source control, and outbound networking follows the official allowlist.
- [ ] Run IDs, task IDs, exact inputs, harness/model identity, grader/image identity, outputs, per-task status, resource usage and all failure dispositions are retained.

### Official execution after qualification

Follow the official procedure in the upstream [evaluation README](https://github.com/google-research/ple/blob/main/eval/README.md). Do not use a locally invented replacement grader.

The upstream dev command is structurally:

```bash
uv run python eval/run.py <submission> --env-file <private-env-file>
```

This is documentation of the official interface, **not authorization to execute it in the current environment**. It invokes Python, Harbor, Docker images, task-provided software, local services and an agent harness; it must not run until all isolation/custody gates are satisfied. Keep `HF_TOKEN`, model keys and judge keys in a secret store / private environment file outside the repository.

Run all 29 dev tasks. For a Luminous matched comparison, preregister three independent full-set development runs with distinct run IDs to align with the published PLE dev leaderboard's three-run average. Keep the competition-entry dev score from committed artifacts as a separate result. Do not confuse Harbor's `--trials` parallelism or per-task retry count with independent benchmark repetitions. Re-grade only through the official tool. Record official per-task scores and raw artifacts as the evaluator returns them. Preserve tasks that are unattempted, ungraded, interrupted or unresolved.

### Score interpretation

The official site reports Claude Opus 5.5 at 48.3% (±1.4%) and GPT-6 Astra at 45.8% (±0.6%) on the 56-task hidden test leaderboard. The paper abstract separately states that the strongest configuration reached a 50.6% mean score. The published page and paper describe different reported summaries; this registry stores both as context only and forbids treating them as a single matched baseline.

The Google Cloud / Mercor launch report (3 November 2026) reports 91% average individual-constraint satisfaction on co-authored tasks but only 41.3% of those deliverables passing every hard safety and feasibility check. The paper reports about 27M tokens and 63 minutes per task for the strongest configuration. These are cohort- and source-scoped measures and must remain separate. The hard-check gate is non-compensatory: a good cost score cannot rescue a critical constraint failure.

For CEB comparisons, first reproduce the official public dev leaderboard using the exact 29-task dev split and same scoring protocol. Only then estimate matched improvements. Never compare a local dev-set score to a hidden-set leaderboard score.

### Current claim ceiling

The PLE registry entry is **access_gated / metadata-only**. No PLE task has been downloaded, run, independently regraded or scored by Luminous in this work. No runner adapter is qualified. No SOTA, engineering competence, licensure, code-compliance, safety or physical-validation claim follows from the registry record.

### Live leaderboard snapshot (retrieved 2026-11-05)

The official Mercor page currently reports the following means on its 56-task hidden test set:

| Model | Native harness | Mean score | Displayed ± |
|---|---|---:|---:|
| Claude Opus 5.5 | Claude Code | 48.3% | ±1.4% |
| GPT-6 Astra | Codex | 45.8% | ±0.6% |
| Claude Sonnet 5.5 | Claude Code | 43.5% | ±1.4% |
| Gemini 4 Argon | Antigravity | 40.6% | ±1.3% |
| Claude Opus 5 | Claude Code | 40.0% | ±1.6% |

The page describes highest reasoning effort in each model's native harness, three runs per task, an isolated container without internet access, and a six-hour per-task limit. The page does not define the meaning of the displayed ± value. It is preserved verbatim in the manifest and must not be interpreted as a confidence interval, standard deviation or standard error without a source definition.

This dated hidden-set snapshot is descriptive context only. It is not a Luminous baseline, is not the public 29-task development leaderboard mean, and is not the separate competition-entry development score based on committed artifacts plus a live sample check. Reproduce the exact official public development protocol before making matched claims. Never use hidden-test scores as a target for tuning or claim a local improvement from a different split/harness.


## BenchCAD 2.0 — metadata-only / evaluation-gated

Official references:
- [BenchCAD 2.0](https://benchcad.com/benchcad-2)
- [Official v2 repository](https://github.com/BenchCAD-org/benchcad-2-main)
- [Public 100-case set and terms](https://huggingface.co/datasets/BenchCAD/benchcad-2.0-public)
- [Core access agreement](https://benchcad.com/access)

### Task coverage

BenchCAD 2.0 has six task families:
1. Dimensioned drawing → part.
2. Supplied STEP parts + assembly drawing → placed assembly.
3. Four rendered views → part.
4. Rendered views of a mechanism → individual parts + assembly.
5. Part drawings + supplied parts + assembly drawing → parts + assembly.
6. Six PCB renders → ECAD wiring/schematic graph.

The public set contains 100 cases (21, 14, 23, 27, 6 and 9 across those task families). A separate 100-case Core set is the held-out source of published scores; its task-family distribution is not copied into this registry. The public and Core sets do not overlap.

### Published v2.0 baseline

The official 2026-10-12 report gives GPT-6 Astra at max effort 0.59 on Core (about $11/case, 135k tokens/case and 62 minutes median) and Claude Opus 5.5 at max effort 0.57 (about $28/case, 1.3M tokens/case and 3.7 hours median). These are v2.0 results, not the current v2.1 protocol.

The v2.0 published setting uses 30 rounds per case and does not carry reasoning between rounds. Claude reference lines are single runs; GPT-6 Astra is a three-run mean except ECAD T6, which has one run. Preserve per-task/per-model repetition differences.

The repository's current v2.1 uses reasoning carried across rounds, 60 rounds and a 500,000 output-token budget per case. The maintainers explicitly say v2.1 results are not comparable with v2.0. Do not run from a floating `main` ref and call it a reproduction; pin the release tag and record all pinned revisions and scorer digests.

### What the benchmark reveals

Its automated reference scoring separates part geometry/topology, assembly placement, and PCB wiring connectivity. The report highlights that assembly outputs often have the right parts but wrong poses: 83–96% of zero-scoring part types in the evaluated assembly tasks had the correct shape in the wrong place. It also reports connection recovery below component recovery on boards; in the max-effort snapshot, GPT-6 Astra found 80% of board connections and Claude Opus 5.5 found 68%. The report describes serious power-rail merge/split errors and notes that the model runs did not test for shorts or opens.

These observations motivate Luminous work on exact assembly datum/pose constraints, component identity, connectivity graphs, and independent short/open checks. They do not mean that a high geometry score proves manufacturability, electrical safety, signal integrity or physical function. BenchCAD 2.0 does not score manufacturing readiness or physical performance.

### Access and execution restrictions

- The public 100-case set is available only after accepting evaluation-only terms. Scores on this set are explicitly not reportable.
- The public terms prohibit training, fine-tuning, distillation, RL-environment construction, pre-training/retrieval/search inclusion, redistribution/mirroring, and posting cases, drawings, renders or reference answers. Preserve the canary in any permitted local copy.
- Core's 100 held-out cases require a separate evaluation-only agreement. Do not retrieve them before access and the Luminous isolation/custody gates are qualified.
- The official repository README documents `tools/run_core.sh` and requires Docker, a Hugging Face account, and a model API key. It executes a sandbox, provider calls and task code; no such run is authorized here.
- Exact runner revision, dataset revision and scorer digest must be pinned from the chosen release before reproduction. Dataset revision remains null in the metadata registry because it has not yet been verified.
- The Git repository does not visibly declare a code licence in its top-level metadata. Do not assume a licence or redistribute runner code until maintainers' terms are verified.
- No Luminous CAD/assembly/ECAD task has been downloaded, run, or scored.

The initial registry entry is metadata-only / access-gated. It establishes a comparator and execution plan, not a local score or a qualification result.


### Additional vendor-reported result (OpenAI, 2026-11-03)

OpenAI's [engineering research update](https://openai.com/index/accelerating-engineering-for-the-physical-world/) reports an average BenchCAD 2.0 score of **0.875** for a newer internal model and **0.594** for GPT-6 Astra at its highest reasoning setting.

This is recorded in the manifest's separate `reported_results` array as `vendor_report`, not inserted into the pinned v2.0 leaderboard rows. The post does not disclose the internal model ID, BenchCAD harness version, per-task results, repetition count, uncertainty, resource budget, or enough protocol detail for independent reproduction. Do not call the newer model "SOTA" on a reproducible leaderboard basis, and do not infer which internal model it is. The reported 0.594 is not silently substituted for the 0.59 rounded figure in the earlier public v2.0 snapshot.

The same OpenAI post describes a physical prototype workflow in which manufacturing preparation required several review/iteration rounds because production limits, tolerances and assembly details were not consistently accounted for. This is a separate evidence class from BenchCAD scoring. It motivates an additional engineering gate: design/geometry success must not be promoted to manufacturability, assembly readiness or a conforming physical article without exact process-capability, tolerance, assembly and inspection evidence.



## Parametric CAD Bench V3 — public suite / metadata-only

Official references:
- [V3 benchmark and live leaderboard](https://cadbench.ai/)
- [V3 run recipe](https://cadbench.ai/benchmark)
- [V3 September 21 release report](https://cadbench.ai/news/cad-bench-v3)
- [V3 paper / evaluation findings (2026-11-02)](https://cadbench.ai/news/cad-bench-v3-paper)
- [arXiv paper](https://arxiv.org/abs/2611.02659)
- [Official Harbor task suite](https://hub.harborframework.com/datasets/gnucleus-ai/cad-bench/v3)

### Task and scoring contract

V3 has 100 public tasks:
- 30 text-to-CAD creation;
- 30 text-to-CAD create-and-edit;
- 40 engineering-drawing image-to-CAD.

It uses native FreeCAD artifacts and a separate programmatic verifier. The current published stack is FreeCAD 1.1.0, gnucleus-freecad-validator 0.6.0 with the v2 scorer, Harbor 0.23.0 and task schema 1.3. The official run page specifies one attempt per task, four-way concurrency, zero retries and a 9,000-second per-task limit. The agent's network access defaults to no network except the selected model provider; the held-back reference geometry/verifier stays outside the agent workspace. Failed/unscored tasks earn zero.

The score is mean continuous reward combining geometry similarity and spec consistency, not a binary geometry pass rate and not a manufacturing/safety qualification. The benchmark measures geometry, spatial alignment, parametric correctness, topology/structure, workflow completion and efficiency.

The official site says the task suite is free to run, while participants pay model API and execution costs. Exact source/dataset redistribution or training terms were not confirmed by this registry pass. Do not redistribute task files or include them in retrieval/training until the official terms are verified. The public V3 tag is not a substitute for the exact immutable dataset digest; capture that digest from the exact official Harbor run before reporting any reproduction. The metadata registry therefore keeps dataset_revision null and execution_authorized false.

### Current public leaderboard snapshot (retrieved 2026-11-05)

| Model | Agent | Overall | Create | Create + edit | Image-to-CAD | 100-task run cost |
|---|---|---:|---:|---:|---:|---:|
| Claude Opus 5.5 | Claude Code | 61.03% | 50.87% | 39.69% | 84.66% | $573.94 |
| Claude Opus 5.5 | mini-swe-agent | 57.96% | 42.79% | 41.38% | 81.77% | $498.28 |
| GPT-6 Astra | Codex | 56.87% | 52.26% | 44.37% | 69.70% | $317.81 |
| Claude Fable 5.1 | Claude Code | 56.75% | 55.48% | 36.75% | 72.71% | $1,056.04 |
| Claude Opus 5 | Claude Code | 50.86% | 49.48% | 36.52% | 62.64% | $1,012.21 |

Costs are publisher-reported whole-job totals for the 100-task cohort, not per-case costs; do not divide or compare without matching the complete run scope. The September 24 paper snapshot reports overlapping confidence intervals among the leading systems; do not infer a statistically decisive rank from point estimates. Use the live page for current rows and the dated paper for the specific September 24 cohort.

### Why this suite is not enough for Luminous claims

The suite grades native CAD artifact geometry/spec consistency, parametric creation/editing and drawing reconstruction. It does not certify future edit preservation across arbitrary downstream changes, measurement traceability, GD&T conformance, manufacturing process capability, assembly access, as-built state, or physical function. Our additional CAD-HISTORY-001 interventions and HARDWARE-TOPOLOGY-001 buildability controls are separate extensions, not replacements for V3 and not directly comparable scores.

Do not pool V3's continuous reward with PLE's professional engineering hard-constraint score or BenchCAD 2.0's Core geometry/assembly/ECAD score. Keep model, agent harness, exact task cohort, scorer and resource budgets separate.

### No Luminous execution

No V3 task image, FreeCAD model, Harbor job, validator, model, or external dataset was downloaded or executed by Luminous in this work. Candidate execution is blocked pending CEB task-custody #10436 and OS isolation #10064. Public availability does not bypass these controls.



## BenchCAD 2.1 — separate protocol and two separate leaderboards (2026-11-05)

Official release: https://benchcad.com/blog/benchcad-2-1  
Official leaderboard: https://benchcad.com/leaderboard

BenchCAD 2.1 keeps the same 100 Core case files but changes the protocol:
- 60 rounds per case (v2.0 used 30).
- 500,000 output tokens per case, thinking included.
- Model reasoning is carried from one round to the next.
- The prompt states rounds/tokens spent and says a non-submission scores zero.
- Task 6 component matching is stricter: each predicted component may only claim its nearest part; duplicate predictions at the same location both score wrong.

**Standard 2.1 board:** Anthropic's published mean of five full passes over all 100 Core cases. At max effort: Claude Opus 5.5 0.490 ±0.013 ($8.1/case); Sonnet 5.5 0.468 ±0.014 ($6.1); Fable 5.1 0.363 ±0.022 ($20); Haiku 5.5 0.186 ±0.016 ($1.4).

**Customized-harness board:** distinct from the standard board, with a common 500,000-token-per-case budget, no round cap/retries, eight-hour wall-clock stop, API-only network access, and one isolated scoring pass. Claude Code/Anthropic self-reported Opus 5.5 0.653, Sonnet 5.5 0.609, Fable 5.1 0.444, and Haiku 5.5 0.229. OpenAI's unreleased internal model 0.875 is displayed on this board but scored using the v2.0 scorer; its model ID/repeats are undisclosed and its approximate $38/case is estimated by BenchCAD. These rows are not standard 2.1 results and are not comparable to the standard board.

Never merge the standard and customized boards or compare v2.1 numbers to v2.0's 0.59/0.57. The current public data/Core access restrictions remain in place; no task content was downloaded and execution_authorized=false.

The BenchCAD Vision2Code 1,000-file subset is another separate evaluation. Kimi K3.1 Pro's October 23 result (0.927 with tools / 0.728 without) is a single-shot max-effort 1,000-file random subset voxel-IoU result, not BenchCAD 2.0 Core. It must not be compared to the Core part/assembly/ECAD score.
