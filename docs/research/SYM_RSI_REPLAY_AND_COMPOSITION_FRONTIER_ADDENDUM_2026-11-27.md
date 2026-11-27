# RSI Replay and Composition Frontier Addendum

**Snapshot:** 2026-11-27  
**Parent programme:** [#10497 — SYM-RSI-META-001](https://github.com/Luminous-Dynamics/symthaea/issues/10497)  
**Status:** Research protocol and implementation direction. This document reports no Symthaea RSI benchmark result, no SOTA result, and no qualification pass.

## Executive thesis

The next Symthaea RSI experiment should compare three distinct ways of improving a research agent, and then test whether their combination improves the *improvement process* itself:

1. **Evaluate-and-select:** run candidate descendants and retain them using task outcomes and an evidence-bound Pareto archive.
2. **Replay-policy evolution:** use recorded discovery histories as bounded replay environments to improve the policy that allocates future search effort.
3. **Explore-then-integrate:** generate independent same-parent implementation variants, identify complementary mechanisms from their actual diffs and interfaces, and compose a compatible descendant before downstream task evaluation.

These are not interchangeable methods. Replay spends prior execution to reduce repeated evaluation. Explore-then-integrate reduces candidate evaluation inside the search loop but cannot establish behavioural correctness by structural inspection. Evaluate-and-select supplies task-outcome feedback but can become expensive and overfit its selection distribution.

The research opportunity is to compare them under one evidence and cost contract, then test a hybrid. The outer-loop claim remains stricter: a successor improver must beat an earlier improver when both are applied to identical starting artifacts under matched *realized total cost*, with independently confirmed successors and transfer.

This addendum extends #10497. It does **not** create another candidate archive, receipt schema, evaluator, statistics engine, scheduler, isolation layer, or promotion authority.

## 1. External frontier snapshot

The figures below are reported by the cited papers or project authors. They are not Symthaea results and should not be pooled into one leaderboard.

| System | What changes | Reported evidence | Comparison boundary |
|---|---|---|---|
| [Dream-RSI](https://arxiv.org/abs/2609.14858) (Google / Google DeepMind co-authors; Sep 2026) | Exploration policy; base agent, task and evaluator remain fixed | On six held-out Lasso datasets, Gemini-3.1-Pro reached 2,931.0 ms average versus 3,587.1 ms for fixed exploration, using 317 vs 550 discovery-agent calls. Gemini-3.7-Flash reached 2,350.6 ms vs 2,516.7 ms, using 1,879 vs 3,200 calls. On KernelBench, the paper reports comparable VGG16/LayerNorm outcomes with 2.43x/1.79x fewer generations and 2.09x/1.44x higher performance on ConvDiv/ConvMax at comparable budgets. | Strong evidence for policy-level replay in these settings. Its mathematical results are mixed: it leads on Sum–Difference, ties the best listed Circle Packing value, but does not beat SimpleTES on Autocorrelation. Historical replay is limited to the realized discovery tree. |
| [AIDE²](https://arxiv.org/abs/2609.26457) (Weco; Sep 2026) | Research-agent harness/code, with fixed model weights | An eight-day run produced 99 rewrite proposals and seven accepted changes; its selection grade rose from 0.703 to 0.778. The authors report transfer to four held-out benchmarks spanning ML engineering, algorithm engineering and weather-forecasting optimization, and a reward-hacking proxy rate falling from 55% to 32%. | Strong product/lineage evidence. The same-start outer-improver comparison ended at 0.780 vs 0.782 over three seeds per arm and was reported as inconclusive; a better inner agent did not establish a better improver. |
| [ELITE](https://arxiv.org/abs/2611.21056) (Nov 2026) | Mechanism integration rather than downstream-score selection | On SWE-bench Pro it reports 85.6%, versus 86.7% for DGM and 88.3% for HGM. It reports 73.3% on SWE-bench Verified and 69.3% on Polyglot without further adaptation. The search loop uses zero downstream candidate-task evaluations and reports 8.8x/3.2x lower wall-clock evolution time than DGM/HGM under the evaluated configurations. | Important evaluation-cost baseline, not the highest SWE-bench Pro score. Structural/operational validity inside the loop is not evidence that the integrated descendant behaves correctly on unseen tasks. |
| [Recursive automated R&D](https://recursive.com/updates/first-steps-toward-automated-ai-research) (company report; Oct 2026) | Long-horizon, multi-threaded experiment proposal, implementation, evaluation and branching | Reports NanoChat validation BPB 0.9109 vs 0.9372 (about 1.3x speedup to the same loss), NanoGPT time 77.5 s vs 79.7 s, and SOL-ExecBench mean 0.754 vs 0.699 across 235 kernels. | Valuable applied R&D comparator; vendor-reported results on narrow, low-variance optimization tasks. Reproduce artifacts and include all system/evaluation costs before treating as a matched baseline. |
| [Nested Learning / Hope](https://arxiv.org/abs/2512.24695) (Google Research; NeurIPS 2025) | Nested optimization, self-modifying update mechanism and continuum memory | Proposes multi-timescale learning and memory; Hope is evaluated on language modelling, knowledge incorporation, few-shot generalization, continual learning and long-context reasoning. | A model/learning-mechanism comparator, not a direct same-start research-agent improver comparison. Keep it as a separate memory/learning track. |

### Evidence-quality warning

The October 2026 study [Auditing the Product, Not the Recursion](https://arxiv.org/abs/2610.27610) coded 114 studies, including 90 that describe their systems as RSI. In its search frame, only 22% report two or more full-loop reruns. Of 26 studies claiming a better improvement procedure, only four compare earlier and later improvers from identical starting artifacts at a matched budget. This is a bounded literature audit, not a census of all relevant work, but it confirms that same-start improver evaluation is a particularly valuable differentiator.

The [Forecasting Research Institute review](https://forecastingresearch.org/research/recursive-self-improvement), published 2026-11-24, characterizes AI as a strong research assistant while humans still provide most research judgment. It identifies uncertainty about whether automation will extend to the remaining judgment-heavy parts of R&D. We should benchmark judgment and direction selection, not only code-edit throughput.

## 2. Research questions

**RQ1 — Replay validity:** When does a recorded discovery history provide reliable off-policy feedback for a new exploration policy, and when does the candidate leave the recorded support?

**RQ2 — Evaluation-cost trade-off:** Does replay reduce expensive live executions while preserving or improving independently confirmed downstream quality?

**RQ3 — Mechanism composition:** Can independent same-parent variants reveal complementary implementation mechanisms that can be integrated with fewer full task evaluations than evaluate-and-select?

**RQ4 — Hybrid benefit:** Does replay-guided policy search combined with explore-then-integrate outperform either method alone at the same total cost?

**RQ5 — R4:** Does a successor meta-agent that inherits one or more of these methods produce more independently confirmed, transferable improvements than the earlier meta-agent when both start from the same inner agent, repository, memory, harness, task state and budget?

These questions form an evidence ladder. RQ1–RQ4 can establish a useful search mechanism. They do not establish RQ5.

## 3. Proposed treatment arms

Use a staged factorial rather than launching an expensive full combination immediately.

| Arm | Search/improvement method | Purpose |
|---|---|---|
| A — Frozen baseline | Incumbent exploration and meta-procedure; no policy evolution | Measures ordinary repeated search |
| B — Evaluate/select | Existing archive-based candidate and lineage selection with frozen meta-procedure | Isolates archive and task-outcome selection |
| C — Replay policy | Dream-RSI-style offline exploration-policy rewrite/replay, followed by live rollout | Tests whether replay improves the policy at lower live cost |
| D — Explore/integrate | ELITE-style focused same-parent variants, structural compatibility review and coherent integration; no downstream score used to choose exploratory children | Tests evaluation-free mechanism composition |
| E — Hybrid | Replay selects/updates search policy; explore/integrate composes compatible mechanisms; independent confirmation selects the retained descendant | Tests complementarity |
| F — Evolving meta-agent | The meta-procedure itself evolves and is inherited by the next generation | Tests the R4 claim, not merely inner-agent improvement |

Required ablations: remove replay, remove history diversity, remove branch/lineage diversity, remove mechanism integration, remove causal failure attribution, remove memory/context management, and remove independent confirmation. Keep negative controls separate from promotion-eligible treatments.

## 4. Replay-world contract: exact where supported, unknown elsewhere

A replay world must be a frozen record of a realized search environment, not a claim that the system knows the full counterfactual world.

Each replayable decision should bind:

- world/task-family ID, task and evaluator version/digest;
- starting artifact, model/backend, harness and policy identities;
- parent decision, branch and candidate identity;
- candidate-visible context and available action/branch identities;
- exact source/artifact digest for each realized child;
- evaluator-owned outcome and cost record, inaccessible to the candidate policy;
- seed, budget, execution environment and timestamps;
- lineage and receipt references from the existing evidence/attestation stack.

### Support and abstention

- A candidate policy may be scored against an action only when that action has an observed, eligible outcome at the corresponding replay state.
- If the policy selects an action absent from the recorded support, record **unsupported**. Do not synthesize an outcome, silently choose the nearest available action, or drop the decision from the denominator.
- Report per-policy supported-decision coverage, unsupported selection rate, complete replay trajectories and partial trajectories separately.
- A high replay score with low support coverage is not a promotion result.
- Replay outcomes are dependent within a shared world/trajectory. Statistical units must be independent worlds/seeds/task families or a cluster-aware analysis, not individual branches treated as independent samples.
- Run a preregistered sample of policy candidates live. Compare replay ranking with live ranking and report rank concordance, top-k recall, selection regret and uncertainty. This estimates replay's predictive fidelity; it does not make unobserved branches observed.

The candidate process must not receive hidden evaluator labels, future states or outcome records while generating policy decisions. A Rust type boundary is useful for ordinary implementations; adversarial candidates require process-level isolation and independent custody.

## 5. Explore-then-integrate contract

ELITE motivates a second evidence channel: source-level mechanisms can be compared before expensive task evaluation.

For each generation:

1. Freeze one parent source tree and exact build/environment identity.
2. Generate multiple focused variants independently from that same parent.
3. Preserve each complete diff, failed build, interface change, dependency change and declared mechanism hypothesis.
4. Compare variants for complementarity, redundancy and conflict using source structure and deterministic operational checks.
5. Integrate a declared compatible subset into one descendant while retaining every contributing parent/variant identity.
6. Run deterministic correctness, regression, security and operational checks.
7. Independently evaluate the integrated descendant on held-out task outcomes. Structural compatibility must never be treated as behavioral proof.

Multi-parent lineage is provenance, not causal attribution. If the integrated descendant improves, the result does not prove every source variant contributed. Preserve variants and run targeted ablations or leave-one-mechanism-out tests before attributing the gain.

## 6. Matched-cost R4 protocol

The primary R4 estimand is the difference between earlier and successor improvers when applied to identical starting artifacts or a declared distribution of paired starting states.

Hold constant:

- inner agent and model checkpoint;
- repository/source tree and build toolchain;
- task/research state and task-family manifests;
- prior verified memory, rejected hypotheses and open questions;
- harness, tools, prompts and context policy;
- evaluator epoch, task set, split custody and outcome adjudicator;
- stopping, retry, time-out and candidate-retention rules;
- total resource ceiling and actual cost accounting.

Count the full realized cost boundary: outer-proposer inference, inner-agent inference, replay generation and scoring, candidate builds, task execution, evaluator calls, retries, failed/invalid candidates, infrastructure/accelerator usage, human review and independent confirmation. An allocation ceiling is not realized-cost matching.

Use a pilot of at least five paired full-loop seeds per arm where feasible to estimate variance and set confirmatory sample size/precision. Five is a pilot floor, not a universal sufficient sample. Run at least two independently evaluated successor meta-generations, then repeat the same-start improver comparison after a meta-level change is inherited.

Preserve and report separately:

- every accepted, rejected, failed, invalid, unsupported and not-run candidate;
- best-so-far, final and predeclared selected checkpoint;
- product score, confirmed improvement yield, cost per confirmed improvement;
- time/evaluations to first confirmed improvement;
- capability-retention floor and worst protected regression;
- fresh-family transfer and full-loop replication spread.

A better inner agent is an R3/product result. A better meta-agent on a different starting state is not an identified improver gain. A single positive outer-loop run is not compounding or an intelligence-explosion result.

## 7. Scorecard

Do not collapse the following dimensions into one headline score.

### Search and replay

- independently confirmed discovery quality versus cumulative actual cost;
- live task executions and evaluator calls avoided;
- replay support coverage and unsupported-action rate;
- replay-to-live rank concordance, top-k recall and selection regret;
- replay/live policy-ranking stability across task families and rounds;
- diversity of validated directions, branch coverage and dead-end budget.

### Integration and code quality

- candidate build/operational validity;
- integrated regression and security failure rate;
- mechanism conflict and integration rejection rate;
- leave-one-mechanism-out attribution where feasible;
- protected capability regression vector;
- best-so-far versus final capability.

### Meta-improvement and transfer

- independently confirmed inherited improvements per full realized cost unit;
- cost per confirmed improvement and time to first confirmed improvement;
- paired earlier-vs-later improver difference from identical starts;
- transfer across SelfImprove-Bench G0–G4 levels and across new task/method/domain families;
- full-loop rerun variance and probability of a positive successor advantage;
- cold-restart and successor-migration recovery of verified research state, including rejected hypotheses and evaluator epochs.

### Integrity and adversarial controls

- false promotions on seeded incorrect/overfit candidates;
- hidden-label, future-state, evaluator-code and split leakage detection;
- dropped/duplicate task rows and denominator manipulation;
- evaluator-version drift and historical regrading;
- best-of-N selection on sealed confirmation;
- unsupported replay actions and replay/live rank inversion;
- reward hacking and proxy improvement with task quality unchanged;
- budget overrun, partial execution and unknown outcomes.

Critical leakage, source-integrity failures, false confirmations and protected-safety regressions are non-compensatory. Higher mean task score cannot offset them.

## 8. Benchmark portfolio and fair comparison

Use benchmark families for different questions; do not compare their raw scores as if they shared a scale.

| Track | Primary comparators | What it isolates |
|---|---|---|
| Software engineering / agent harness | SWE-bench Pro, SWE-bench Verified, Polyglot; DGM, HGM, ELITE, AIDE² | Coding-agent product quality, harness changes and cross-language transfer |
| ML engineering / training | MLE-bench, FML-Bench, NanoChat Autoresearch, NanoGPT Speedrun | Model/training engineering under fixed compute and outcome budgets |
| GPU/kernel optimization | SOL-ExecBench, KernelBench (including Dream-RSI's VGG16, LayerNorm, ConvDiv and ConvMax workloads) | Hardware-aware algorithm discovery and measured speed/correctness |
| Mathematical/algorithmic discovery | Dream-RSI Sum–Difference, Autocorrelation and Circle Packing; ALE-Bench | Discovery quality, exact objective correctness and search efficiency |
| Data-centric research | RSIBench-Data | Diagnosis of learning bottlenecks, data strategy and checkpoint retention |
| Scientific workflows | Autonomous Science Bench, FIRE-Bench, SciWorkBench and fresh Luminous domain tasks | End-to-end research decisions, execution, evidence and domain transfer |
| Transfer ladder | SelfImprove-Bench G0–G4 | Same instance, new instances, harder tasks, new tasks in a capability, and new capability categories |

Use pinned benchmark versions, harnesses, task-set IDs, metric definitions and official evaluation policies. Reproduce the official baseline before attempting to exceed it. Require identical starting artifacts and matched model, context, tool privilege and total cost for treatment comparisons. Hardware-specific results need the same device/driver/compiler and isolated timing runs. Public leaderboard values and company-reported claims must retain their source and status.

## 9. Integration with the existing Symthaea programme

Reuse current owners and authorities:

- **#10472** — evidence-context-bound Pareto candidate archive.
- **#10473 / #10496** — matched per-case evaluation and analysis.
- **#10498 / #10499** — variation operators and multi-parent lineage.
- **#10497** — matched-start R4 improver and ignition experiment; this addendum belongs here.
- **#10691** — algorithmic RSI / clean-start successor training; keep algorithm changes distinct from research-agent/meta-policy changes.
- **#10708 / #10573** — evaluator epochs and independent promotion authority.
- **#10756** — evaluator research against independent target judgments.
- **#10846** — harness-RSI reproduction; distinguish warm-started product lift from a superior improver.
- **#10877** — research taste, problem/approach selection and experiment-design quality.
- **#10314 / #10370 / #10167** — shared benchmark receipts, accounting, attestation and regrade.
- **#10064 / #10369** — execution isolation and trust roots.
- **#10630** — evidence lifecycle and selective revalidation.

Do not add a second source of promotion authority or a second evidence ledger. An archive digest is not a signature; a sealed-holdout enum is not data access control; a passing producer test is not independent semantic verification.

## 10. Execution sequence

### Phase 0 — pin and reproduce baselines

- Pin papers, official code, commit SHA, runtime/model versions, task manifests and licensing.
- Reproduce Dream-RSI's fixed-policy vs replay-policy comparison on one affordable task family.
- Implement ELITE's same-parent variant/integration comparison on a deterministic software task.
- Capture full realized cost, including the outer improver and failed proposals.
- Record whether each result is paper-reported, independently reproduced, structurally checked or still pending.

### Phase 1 — replay-support pilot

- Use a small deterministic search space with saved branches and a frozen evaluator.
- Freeze a common first-round exploration tree and branch all arms from it.
- Compare fixed exploration, offline replay policy improvement and a random-search/selection control.
- Audit candidate decisions against evaluator-owned support; force explicit unsupported outcomes for off-tree choices.
- Run a preregistered live-policy sample to estimate replay/live rank fidelity.

### Phase 2 — evaluate/select vs explore/integrate vs hybrid

- Use same-parent independent variants and preserve all diffs.
- Compare task-evaluation counts, source integration defects, independently confirmed task quality and transfer.
- Add replay-policy improvement as a separate factor.
- Keep evaluator versions frozen within an epoch; use the established evaluator-evolution and promotion owners at explicit boundaries.

### Phase 3 — R4 ignition and repeatability

- Compare earlier and successor improvers from identical starting artifacts at matched realized total cost.
- Repeat with at least two independent full-loop successor generations.
- Confirm on sealed task families and at least one fresh domain.
- Audit capability retention, reward hacking, cost and research-state continuity.
- Report inconclusive/negative outcomes without changing the metric or selecting on sealed confirmation.

## 11. Proposed gates and claim ceiling

Before the pilot, freeze the task family, cost boundary, metric definitions, split and evaluator. Use the pilot to estimate run variance, task-family dependence and practically meaningful effect size; then preregister confirmatory sample size and margins. Do not assign arbitrary numerical superiority targets before that baseline exists.

Minimum integrity gates:

1. All deterministic known-answer, leakage, denominator and support-boundary controls produce their expected dispositions.
2. Every candidate and every attempt has a traceable identity and terminal disposition.
3. No policy gets an outcome for an unsupported replay action.
4. Replay-to-live ranking fidelity is reported on a preregistered live sample.
5. Candidate selection and final confirmation use disjoint task-family custody.
6. Earlier and successor improvers begin from identical artifacts and matched total-cost budgets.
7. Independent transfer and protected capability floors pass before a stronger process claim is made.

Claim ladder:

- **Replay utility:** replay gives useful bounded feedback on recorded support.
- **Search efficiency:** replay or integration yields more independently confirmed task improvements per matched cost.
- **Product transfer:** the improved agent generalizes to held-out task families/domains.
- **R4 improver gain:** the later improver beats the earlier one from identical starting states and budgets.
- **Repeatable recursive progress:** repeated same-start comparisons show that inherited improvers continue to improve the improvement process itself.

No stage establishes unbounded self-improvement, general intelligence, consciousness, or safety by itself.

## Sources

- Dream-RSI paper: https://arxiv.org/abs/2609.14858
- Dream-RSI source and reproduction instructions: https://github.com/google-research/dream-rsi
- AIDE² paper: https://arxiv.org/abs/2609.26457
- ELITE paper: https://arxiv.org/abs/2611.21056
- Recursive automated R&D results: https://recursive.com/updates/first-steps-toward-automated-ai-research
- Nested Learning / Hope: https://arxiv.org/abs/2512.24695
- RSI evidence audit: https://arxiv.org/abs/2610.27610
- Forecasting Research Institute RSI review (2026-11-24): https://forecastingresearch.org/research/recursive-self-improvement
- RSIBench-Data: https://arxiv.org/abs/2607.25886
- SelfImprove-Bench: https://arxiv.org/abs/2611.02797

## 12. Existing low-cost replay substrate and first pilot

A bounded starting point already exists in the Futures Laboratory:

- `crates/domains/symthaea-futures-ensemble/examples/support/evolutionary_rescue_common.rs` defines `SerializedWorld`, `record_world`, `load_world`, the world format version, independent train/test seed sets and the deterministic world-recording procedure.
- `crates/domains/symthaea-futures-ensemble/examples/evolutionary_rescue_generate_worlds.rs` writes ten reusable world trajectories: five train seeds ([11, 22, 33, 44, 55]) and five test seeds ([66, 77, 88, 99, 111]); each world records 11,000 simulation ticks.
- `crates/domains/symthaea-futures-ensemble/examples/evolutionary_rescue_architecture_ablation.rs` already consumes serialized worlds and compares multiple forecast/observation conditions.
- `crates/domains/symthaea-futures-ensemble/src/evolutionary_rescue.rs` supplies the existing baseline ladder, including census-only and census-plus-trait FEP generators.
- `crates/domains/symthaea-futures-ledger/src/lib.rs` already defines a replayable forecast evidence record containing scenario family, world seed, observation policy version/cutoff, belief-state hash, model and generator identities, predicted distribution, actual continuation, score, calibration bucket and wall-clock cost.

### What this substrate is and is not

This is a useful **recorded-world and forecast-replay substrate**, not a complete Dream-RSI discovery tree. It stores trajectories and observation streams; it does not yet record a research agent's full sequence of candidate-generation decisions, available branches, off-policy choices, evaluator feedback and search-policy versions. Therefore it can validate replay data integrity, world/observation support, forecast scoring and some policy-allocation questions, but it cannot by itself establish agent-level recursive self-improvement or Dream-RSI equivalence.

At inspection time, the generator and ablation source were present on main, but the expected serialized JSON fixtures were not present at the checked paths. The generator must run to materialize them. This document does not claim that the fixtures were generated or that the pilot was executed.

### First executable tranche

1. Run the existing generator once to materialize the 10 train/test worlds; record toolchain, source SHA, fixture SHA-256 and exact configuration.
2. Independently parse and validate format version, seed membership, tick continuity, stream lengths, terminal outcomes and absence of train/test seed overlap.
3. Use the five training worlds to develop a small replay policy over *declared, actually available* observation/forecast actions. Keep evaluator truth and test-world trajectories out of the policy process.
4. Add explicit action-support states: observed-and-supported, unavailable, unsupported and indeterminate. Never fill unsupported actions with oracle values or drop them from coverage.
5. Score the frozen policy and fixed/random controls on the five test worlds; include no-information and shuffled-observation controls.
6. Report per-world outcomes, calibration, support coverage, unsupported selection, Brier/log score where contractually valid, total execution cost and clustered uncertainty. Five test seeds are a pilot only; they cannot establish a high-confidence SOTA claim.
7. Separately build the decision-level search-tree recorder required for Dream-RSI proper. Bind every candidate branch, parent, policy, evaluator output and actual cost to the established candidate archive and benchmark receipt/attestation owners. Do not retrofit forecast `EvidenceRecord` into a generic research ledger.

### Stop/go criteria for the pilot

- **Stop** on stale/malformed fixtures, train/test overlap, missing outcome rows, source/evaluator identity mismatch or unsupported actions being assigned fabricated outcomes.
- **Proceed to a larger confirmatory study** only if replay policy ranking shows useful held-out agreement with live evaluation, improves the cost/quality frontier against fixed and random controls, and has no material calibration or protected-capability regression.
- A positive forecast-replay pilot is evidence for a reusable replay substrate only. The discovery-tree policy result and R4 same-start improver result remain separate claims.

