# Scientific Decision Replay Contract

**Snapshot:** 2026-11-27  
**Parent programme:** [#10497 — SYM-RSI-META-001](https://github.com/Luminous-Dynamics/symthaea/issues/10497)  
**Related research protocol:** [PR #13879](https://github.com/Luminous-Dynamics/symthaea/pull/13879)  
**Status:** Contract prototype and implementation plan. No scientific RSI result, benchmark reproduction, or qualification pass is claimed.

## 1. Purpose

Extend replay from recorded search outcomes to the prospective decisions that shape scientific work: question selection, hypothesis formation, experimental design, prospective prediction, execution/observation, interpretation, and research-policy updates (continue, replicate, pivot, stop or escalate).

The hypothesis is that complete, evidence- and cost-aware decision histories can improve future research allocation beyond outcome-only replay. This must be tested against controls; it is not an assumed property of Symthaea.

## 2. Split observation from outcome custody

The current Futures Laboratory fixture in the evolutionary-rescue example support code bundles candidate-usable observation streams with privileged observations, the hidden trajectory, first-extinction labels and other evaluator-derived fields. This is workable for trusted offline analysis, but it is not a safe object to pass wholesale to a candidate research policy.

A first custody adapter is now authored at `examples/support/evolutionary_rescue_custody.rs`, with the materializer at `examples/evolutionary_rescue_split_capsules.rs`. It projects one observation view at a time and writes candidate observations/receipt to one output root and evaluator outcomes/manifest to a disjoint output root. The generator refuses overlapping roots and file overwrites, uses opaque UUID world IDs, computes BLAKE3 byte digests, writes the evaluator manifest last, and reads the files back to verify their digest joins.

The current observation record still exposes only simulation tick and sample values; it does not yet provide physical event time, availability time, per-observation source digests or a calibrated observation envelope. Those semantics must be supplied through the #12407 observation-event work before treating the pilot as an asynchronous/irregular-observation benchmark.

The pilot must materialize three logically separate artifacts.

### Candidate observation capsule

Contains only the one observation view permitted for that treatment, observation IDs, source digests, event and availability times, task-family identity, public scenario contract, permitted actions, resource/deadline constraints, and prior candidate-visible decisions.

It must not contain privileged observations, future trajectory values, extinction labels, post-run scores, or a seed mapping that reveals held-out assignments.

### Evaluator outcome capsule

Held by a separate evaluator process or custody domain. Contains hidden trajectories and ground truth, privileged/oracle observations, evaluator-only labels and scores, sealed task/seed mappings, and outcome-resolution/adjudication records.

A Rust module boundary is not access control against malicious candidate code. Pass only the candidate capsule into the candidate process and retain evaluator data elsewhere.

### Evidence manifest

Binds capsules without disclosing evaluator payloads: schema/simulator version, content digests, world/task-family identity and seed commitment, observation-policy version, generator/build/environment digests, lineage/receipt references and custody metadata.

A digest proves byte identity, not truth, independence, access control or correct custody. BLAKE3 here is an integrity identifier, not an authenticated signature. Separate directories are not a security boundary by themselves: the operator must ensure candidate credentials cannot read evaluator storage, and the candidate process must never mount the evaluator root.

## 3. Contract prototype

The new support module defines:
- CandidateDecisionContext: decision-time evidence, permitted action set, context digest, budget, deadline, and a digest-bound decision objective (valuation mode, declared utility unit and loss-table reference where applicable).
- DecisionProposal: selected action, prospective probability distribution, policy lineage, assignment mechanism and causal-design declaration.
- EvaluatorResolution: evaluator-owned execution status, outcome-resolution state, observed-action support, evidence references and actual-cost completeness.

Candidate-visible types do not contain ground truth or future outcome labels. EvaluatorResolution is separate and must not be serialized into the candidate process input stream.

Validation covers:
- no evidence became available after decision time;
- every decision is bound to a frozen downstream decision/loss contract; raw entropy or information gain is not substituted for decision value;
- selected actions belong to the permitted set;
- prospective probabilities sum exactly to one million parts per million;
- randomized assignment records a nonzero propensity and protocol reference;
- causal-design claims include identification-assumption digests;
- randomized-controlled claims require randomized assignment;
- unsupported counterfactuals cannot carry observed actions, outcome labels or outcome evidence;
- delayed, censored and indeterminate outcomes remain distinct;
- execution failure is not silently converted into a scientific negative result;
- complete cost accounting records every resource dimension, including measured zero.

This prototype does not replace the existing evidence ledger, candidate archive, receipt/attestation system, evaluator authority or execution-isolation owner.

## 4. Replay must be support-aware and causally qualified

A historical trajectory gives outcomes for actions actually taken. It does not reveal outcomes for every action that could have been taken.

For each decision, record exact candidate-visible context and evidence availability times, admissible action set, selected action and behavior-policy identity, assignment mechanism and the full probability distribution over every permitted action when randomized (including zero-probability actions), outcome/evidence for the action actually executed, cost, feedback delay, adjudication, task family, evaluator epoch and lineage.

Rules:
1. Score a replayed action only when a valid outcome for that exact action at that decision state exists.
2. For an off-support action, emit unsupported_counterfactual; never impute a nearest-neighbor or model-generated outcome and never drop the row from coverage denominators.
3. Use propensity-aware estimators only when logged propensities, positivity, consistency and other assumptions are satisfied. A propensity alone does not identify a causal effect.
4. For unsupported decisions, spend real experiment budget when the expected decision value justifies it.
5. Preregister a live audit sample and compare replay-predicted policy ordering to live ordering, reporting rank concordance, top-k recall, selection regret, uncertainty and support coverage.
6. Treat worlds/trajectories/task families as clustered sampling units. Tick checkpoints within a world are not independent replicates.

## 5. Separate execution status from outcome status

| Execution status | Outcome status | Meaning |
|---|---|---|
| not_started | not_run | No action/experiment was executed |
| not_started | unsupported_counterfactual | Candidate chose a branch with no observed outcome; no outcome is fabricated |
| started | awaiting_observation | Execution began; evidence has not resolved the question |
| completed | immediate_observed | Outcome observed and evidence admitted immediately |
| completed | delayed_observed | Outcome became observable later |
| completed | censored_at_deadline | Outcome not resolved before observation deadline |
| completed | indeterminate | Evidence exists but does not settle the outcome |
| failed / timed_out / aborted | invalid_execution | Execution did not produce a valid scientific result |

A censored result is not a failure, a pending result is not success, an invalid execution is not a scientific null, and a not-run proposal is not a negative experiment. Preserve all rows in denominators and reports.

## 6. Staged experimental design

Use the existing Futures Laboratory substrate and its current forecast evidence ledger. Do not create another generic ledger, scheduler, benchmark receipt service, evaluator or promotion authority.

### Pilot 0 — materialize and custody-check existing worlds

- Generate the ten existing evolutionary-rescue worlds once (five train and five held-out seeds).
- Freeze file identities and simulator/build provenance.
- Validate format version, tick continuity, observation timing, seed separation and observation-view semantics.
- Run `evolutionary_rescue_generate_worlds` once to materialize the ten source fixtures.
- Run `evolutionary_rescue_split_capsules` against those frozen fixtures, choosing one view (`census`, `noisy-trait`, or `shuffled-trait`) for each campaign.
- Give the materializer disjoint candidate/evaluator output roots and the exact source commit SHA.
- The generator assigns opaque UUID world IDs, writes candidate observations plus a candidate receipt to the candidate root, and writes hidden outcomes plus the joined custody manifest to the evaluator root.
- Verify capsule byte digests and receipt/manifest/world-ID linkage after disk readback.
- Never pass the current combined serialized-world object or evaluator root into a candidate process.

Ten worlds are a pipeline pilot, not a confirmatory scientific benchmark.

Example commands (replace paths and source SHA with the actual campaign values):

```sh
cargo run --release --example evolutionary_rescue_generate_worlds -p symthaea-futures-ensemble
cargo run --release --example evolutionary_rescue_split_capsules -p symthaea-futures-ensemble -- /secure/candidate /secure/evaluator <40-or-64-hex-source-commit> census
```

The materializer creates both roots if absent, rejects equal/nested roots, refuses to overwrite existing files, writes the manifest last, and reads all four output artifacts back before accepting the pair. It does not set filesystem ACLs, create separate OS identities or launch the candidate/evaluator processes. Those are required deployment controls, not provided by this fixture utility.

### Pilot 1 — compare decision policies

Freeze task generator, simulator, executor, evaluator and per-treatment resource ceiling. Branch policies from the same decision histories.

| Arm | Replay signal | Question |
|---|---|---|
| A — fixed | None | Baseline research allocation |
| B — random | None | Does a learned policy beat budget-matched chance? |
| C — outcome-only | Executed outcomes | Is ordinary outcome replay sufficient? |
| D — process-only | Decision process, no outcome labels | Does decision-process history add value? |
| E — evidence-and-cost | Process + evidence quality + delay + cost | Does evidence-qualified memory improve allocation? |
| F — full-decision | Process + admissible outcomes + cost, support constrained | Does full history outperform partial history? |
| Controls | No history, shuffled history, success-only history | Is gain due to coherent evidence rather than context volume or positive selection bias? |

Use the same fixed executor/coder in the primary decision-policy comparison. Add an end-to-end interaction lane after the isolated researcher effect is measured.

### Pilot 2 — live replay-fidelity audit

Preregister a sample of candidate policy decisions for live execution. Estimate replay-vs-live ranking agreement and top-k recall; report selection regret and policy coverage; stratify by task family and support; retain not-run, failed, censored and indeterminate outcomes. Do not promote from replay-only evidence.

### Pilot 3 — matched-start R4

Use #10497 and its existing owners. Earlier and successor meta-improvers must start from the same inner agent, repository, task state, verified memory, harness, model/backend and evaluator epoch. Match realized total cost, not allocated budget. Include outer-proposer inference, inner inference, tools, failed proposals, replay generation, builds, execution, evaluator, monitoring, human review and independent confirmation.

Use at least five paired full-loop seeds as a variance-estimation pilot where feasible, not as a universally sufficient confirmatory sample. Freeze confirmatory sample size and margins after estimating variance and practical effect size. Repeat inherited improvements through two independently evaluated generations. Report failures, invalids, rejected candidates, unsupported proposals, best-so-far/final checkpoints and worst protected regression.

## 7. Three-timescale research memory

1. **Episode memory:** immutable decision/outcome/evidence trace, including nulls, failures, delays, rejected hypotheses and costs.
2. **Candidate working memory:** provisional mechanisms and strategies with explicit applicability conditions. It may guide proposals but is not established knowledge.
3. **Validated research memory:** independently confirmed rules with effect estimates, uncertainty, task-family support, known contraindications, provenance and revalidation conditions.

Promotion from working memory to validated memory requires independent replication or a declared evidence standard. Contradictions narrow, suspend or revoke rules without rewriting their historical support. Retain negative-transfer evidence. Retrieval checks applicability and contraindications.

## 8. Paradigm shifts

- **Decision value, not entropy:** optimize improvement in downstream decisions per inclusive lifecycle cost. Use EVSI only where priors, likelihoods and utility units are defensible; otherwise report robust/minimax regret.
- **Causal replay, not counterfactual storytelling:** missing outcomes remain unsupported; randomized interventions and prospective live audits qualify causal claims.
- **Evidence-carrying memory, not success stories:** every reusable rule binds support domain, evidence, cost, uncertainty, contraindications and revalidation triggers.
- **Portfolio allocation, not one-step ranking:** allocate across tractable work, high-impact uncertainty, replication and deliberate disconfirmation.
- **Evaluator pluralism:** cross task generators, model backbones, graders and blinded human/domain-expert anchors. Candidate generation, benchmark admission and final adjudication remain independently controlled.
- **Researcher/coder separation:** compare research policies with one fixed executor before measuring end-to-end interaction.

## 9. Competitive benchmark portfolio

Do not pool different task units into one RSI number.

| Capability | Benchmark / comparator | Symthaea objective |
|---|---|---|
| Scientific transfer | RecurSciBench | Positive family-clustered transfer with a smaller regression tail; report every task/domain |
| Research-direction taste | IdeaTaste + blinded expert anchors | Improve actionable idea and novelty quality, not just paper matching |
| Experimental taste | TasteVal-style fixed-coder comparison | Improve independently confirmed final quality per inclusive total cost |
| Full-cycle rediscovery | FIRE-Bench | Improve rediscovery validity and reduce planning/analysis errors on matched tasks |
| Workflow breadth and trajectory stability | SCI-AGENT-EXAM | Improve executable + written artifact correctness, strict success rate and run-to-run stability across research task types |
| Executable literature and method reuse | Paper2Agent | Convert more papers into scoped, reproducible, independently validated research tools; keep tool validity and scientific validity separate |
| Open scientific follow-up integrity | Into the Unknown | Satisfy every expert-authored necessary condition with claims tied to executed artifacts; zero unsupported work/result claims |
| Autonomous science | Autonomous Science Bench | Beat reproducible baselines on quality/cost with independent replication; six-task pilot is diagnostic |
| Long-horizon work | METR TH1.1 / TH2 | Improve long-horizon success without hiding lifecycle costs or judgment gaps |
| Research engineering | SWE-bench Pro/Verified, Polyglot, MLE-bench, FML-Bench | Matched-backbone/cost comparison and capability retention |
| Algorithm/kernel discovery | KernelBench, SOL-ExecBench, ALE-Bench | Correctness-first speedup and verified discovery per total compute |
| Meta-improver | AIDE²-style identical-start R4 | Repeated positive paired successor advantage at matched realized cost |

### Provisional stretch targets from the existing decision-intelligence benchmark plan

These are ambitious targets to validate and power after a variance-estimation pilot; they are not achieved Symthaea results or immediate pass/fail gates.

| Benchmark | Public reference / scale | Proposed Symthaea stretch target |
|---|---|---|
| Autonomous Science Bench 0.1 | Six pilot campaigns, three trials each; best listed resolution rate 72.2% | At least 15/18 resolved trials (83.3%), with discovery and held-out learning gates both passing for every resolved campaign |
| FIRE-Bench | Best published F1 46.7 on 30 tasks / three trials | At least 55 F1 on the same release, while reporting claim precision, recall, trial variance and unsupported claims |
| SCI-AGENT-EXAM | 120 expert-curated tasks across 11 fields and 7 task types; GPT-6 Astra reports 80.9 overall, with 26.0% runs above the 0.95 success threshold | At least 85 overall and at least 50% of runs above 0.95, with executable/text artifact agreement, per-task-family reporting and exact-release reproduction |
| Paper2Agent agentification | 74/100 biology papers agentified; 593/599 proposed tools passed automated validation; 91.2% on 300 tutorial questions with Sonnet 4 | At least 85% end-to-end agentification on a stratified corpus, at least 98% tool-contract test pass, and 95% on matched tutorial/execution questions; scientific conclusion validity remains separately adjudicated |
| Into the Unknown | 0/135 follow-up submissions met all necessary conditions; 114 failed integrity checks in the reported study | Zero unsupported claims in the admitted subset, 100% claim-to-artifact binding for quantitative claims, and explicit not-performed/indeterminate/invalid dispositions |
| SciWorkBench | Best reported mean 57.38 | At least 65 on the same official harness/scorer, with domain-specific failure rates |
| RecurSciBench | 20 independent test tasks across four domains; evolved states show mixed mean transfer and broad per-task regressions | Mean transfer above +5%, nonnegative aggregate in every domain, and no more than 4/20 independent test tasks regressing; confirm on expanded task families |
| TasteVal-style fixed-coder lane | Reported experimental-compute multiplier up to 2.30x; researcher inference and full lifecycle costs are not included in that multiplier | Match or exceed 2.30x compute-to-target while preserving final-quality non-inferiority and reporting a separate inclusive-cost frontier |
| METR TH1.1 / TH2 | Long-horizon tasks with materially different lifecycle costs | At non-inferior success, reduce inclusive lifecycle cost by 30%, or extend successful task horizon by 1.5x at matched total cost |
| SWE-bench Pro (secondary engineering lane) | HGM reported 88.3% on its evaluated subset | Exceed 88.3% only on the same pinned subset/harness, while retaining Verified/Polyglot transfer and reporting full compute/evaluation cost |

The RecurSciBench regression-tail target is intentionally more demanding than merely beating its mean. The TasteVal target must not be claimed from compute-only accounting. METR cost/horizon comparisons must use matched task strata and uncertainty intervals. SCI-AGENT-EXAM is a recent preprint and its hybrid judge requires independent calibration; reproduce its exact task release before comparing. Paper2Agent's tool-extraction accuracy is not equivalent to independent scientific-validity accuracy; compare agentification coverage and tool execution separately from claims about correct science. Into the Unknown's integrity gate should be retained even when no reference numeric answer exists. Never compare numerical scores across different benchmarks.

### Three non-poolable lanes

- **Fast-verifiable:** deterministic tests, formal checks, simulators and numerical oracles.
- **Slow-verifiable:** delayed rollouts, independent replication, held-out deployments and long-horizon consequences.
- **Judgment/underspecified:** question selection, experiment quality, interpretation and escalation under uncertain feedback.

Report quality, reliability, coverage, calibration, lifecycle cost and uncertainty separately. A fast-lane gain must not stand in for better scientific judgment.

The primary success criterion is **more independently confirmed, transferable research progress per fully inclusive resource budget, with no hidden regression in protected capabilities.**

## 10. Compose with the existing scientific decision-intelligence programme

This is a shared decision-episode contract, not a new scientific scheduler or decision-value engine. The existing programme already has a mature Stage-A/Stage-B split: hypothesis discrimination is not decision value; robust minimax and Bayesian expected utility are distinct modes.

- #12546: finite-world decision evaluator, including Bayesian and prior-free robust decision-value modes and a typed covariance-to-robust adapter. Reuse it for loss/decision qualification.
- #13453: joint-covariance experiment design v0.2. Keep its Stage-B hypothesis-discrimination output separate from Stage-A decision value.
- #9817/#9821: active sensing and decision-focused evidence acquisition.
- #5476/#5477/#5478: prospective value-of-computation calibration, realized-utility receipts and causal metareasoning ablations.
- #11012: evidence-contingent adaptive research orchestration; this contract can supply its evidence-prefix-bound decision episode.
- #8721: prospective scientific tool/computation allocation; do not recreate its action/tool scheduler.
- #10877: research direction/approach/experiment/interpretation taste.
- #10756: evaluator research and independent target judgments.
- #9230: verifier-grounded direction-board allocation across research directions.
- #9332: sealed oracle and adaptive benchmark generation.
- #12407: observation-time, mask and modality semantics.
- #12502/#12505/#12506/#12624: dynamical references and predictor baselines.
- #10497: matched-start R4 and recursive-improver qualification.
- #10472/#10473/#10499: evidence-bound candidate archive, evaluation and multi-parent lineage.
- #10708/#10573: evaluator epochs and independent promotion authority.
- #10314/#10370/#10167: benchmark receipts, accounting, attestation and regrade.
- #10064/#10369: execution isolation and trust roots.
- #10630: evidence lifecycle and selective revalidation.

The current futures ledger records forecast provenance and scored continuations. This contract is a decision-episode interface, not a competing ledger. Persistent writing should reference existing evidence/receipt identities and be assigned to an existing owner. Adapt this contract to the decision objective and event semantics in #12546 and #12407 rather than duplicating their calculations or observation model.

## 11. External research

- [Dream-RSI](https://arxiv.org/abs/2609.14858): replay-based exploration-policy improvements; motivates a policy baseline, not automatic transfer.
- [AIDE²](https://arxiv.org/abs/2609.26457): recursive improver comparator; same-start comparison was explicitly inconclusive at three seeds per arm.
- [ELITE](https://arxiv.org/abs/2611.21056): evaluation-free mechanism integration; independent behavioral tests remain necessary.
- [RecurSciBench](https://arxiv.org/abs/2611.32508): mixed scientific transfer with substantial per-task regression.
- [METR Q4 2026](https://metr.org/blog/2026-11-20-q4-2026-capability-update/): judgment without fast feedback and lifecycle-cost measurement need separate lanes.
- [TasteVal](https://arxiv.org/abs/2610.06824): fixed-coder comparisons can isolate experimental-planning quality.
- [IdeaTaste](https://arxiv.org/abs/2611.03424): retrospective idea matching is a diagnostic, not a scientific-validity oracle.
- [FIRE-Bench](https://arxiv.org/abs/2602.02905): full-cycle rediscovery measures planning, execution and analysis, not new discovery by itself.
- [SCI-AGENT-EXAM](https://arxiv.org/abs/2610.32441): 120 expert-curated tasks across 11 fields and seven research task types; combines executable and textual outputs and analyses run stability.
- [Paper2Agent](https://www.nature.com/articles/s41586-026-11044-y): converts papers/repositories into structured resources and validated executable tools; its agentification and tutorial-question metrics are distinct from independent scientific truth.
- [SciAgentArena](https://www.microsoft.com/en-us/research/publication/benchmarking-ai-agents-for-addressing-scientific-challenges-across-scales/): roughly 200 tasks with stepwise verification; reports persistent difficulty in novel insight and sustained self-directed exploration.
- [Into the Unknown](https://arxiv.org/abs/2610.19159): open scientific follow-up benchmark with expert-authored necessary conditions and a demanding claim-integrity check.

These are paper/author-reported baselines unless independently reproduced under matched harness and total-cost boundaries. No claim that Symthaea beats them is made here.

## 12. Implementation status and gates

Implemented in this branch:
- candidate-visible decision context and prospective proposal, including a decision-objective reference;
- outcome distribution, full randomized action propensities and causal-design declarations;
- evaluator-owned resolution and distinct execution/outcome states;
- validation for look-ahead evidence, action support, unsupported counterfactuals, delayed/censored/indeterminate outcomes and cost completeness;
- nine authored decision-contract tests;
- candidate/evaluator capsule projection, BLAKE3 digests, opaque IDs, disjoint-root checks, no-overwrite writes, manifest-last materialization and disk-readback checks;
- seven authored custody tests and a capsule materializer example.

Not implemented or not yet verified:
- the ten worlds have not been generated or split in this environment;
- the capsule utility does not enforce OS/service access control or launch isolated candidate/evaluator processes;
- current fixture observation timestamps remain simulation ticks, not full #12407 event/availability-time semantics;
- integration with existing ledger/archive/receipts/attestation;
- executable replay policy, live-fidelity audit, external benchmark runner or R4.

Verification boundary: schema/validation prototype only. Rust compilation and unit tests have not been run in this environment. Exact-head CI must be checked; queued, skipped or missing status is not a pass. Do not merge or promote on schema tests alone.
