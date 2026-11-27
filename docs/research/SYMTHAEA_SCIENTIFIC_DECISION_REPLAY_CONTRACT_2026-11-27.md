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

A first custody adapter is authored at `examples/support/evolutionary_rescue_custody.rs`, with its materializer at `examples/evolutionary_rescue_split_capsules.rs`. It projects one observation view at a time and writes a **custodian-only full replay archive** to one output root and evaluator outcomes/manifest to a disjoint output root. The generator refuses overlapping roots, requires fresh empty output roots, refuses file overwrites, uses opaque UUID world IDs, computes BLAKE3 byte digests, writes the evaluator manifest last, and reads the files back to verify digest joins.

**Critical temporal boundary:** the replay archive contains the entire observation sequence, including observations that are in the future relative to early decisions. It is not candidate input and must never be mounted into the candidate process. A trusted replay driver reads the archive and emits only the observation prefix whose availability time is no later than the current decision time, packaged in a `CandidateDecisionContext`. The evaluator outcome capsule remains separately controlled. A full archive hash does not authorize exposing the full archive.

The current source observation record exposes simulation tick and sample values only. It does not yet provide physical event time, availability time, per-observation source digests or a calibrated observation envelope. Those semantics must be supplied through #12407 before this pilot can support asynchronous/irregular-time replay.

The custody module now includes `build_decision_prefix(archive, decision_tick)`. It validates tick continuity and emits a separate `DecisionObservationPrefix` containing observations only through the requested tick, plus a BLAKE3 digest of that prefix. It rejects out-of-range cutoffs. This is a tested projection helper, not yet a running replay service: an isolated driver still has to deliver the prefix to the candidate and keep the archive/evaluator roots inaccessible.

The AS-Bench action binder is designed to pass its SHA-256 directly into the decision proposal. For Nuclease the request payload identifies the exact stage and ordered variant IDs; for Propylene it identifies the exact campaign, design-ID field and submitted batch. #10947 remains authoritative for job-state/result-row validation: queued/running jobs remain pending, failed jobs do not advance the stage, and only completed jobs with exact result identity and full measurements may resolve an outcome.

The pilot materializes three separate artifacts and a runtime-only candidate view.

### Custodian-only replay observation archive

Contains one permitted observation view for the treatment and the complete time-ordered stream. This archive may contain future observations relative to any particular decision. It is accessible to the trusted replay driver only, not to the candidate process. The driver must produce monotonically advancing decision prefixes and enforce the decision-time availability cutoff.

### Candidate-visible decision context

For each decision, the driver provides only the currently available evidence prefix, task-family identity, public scenario contract, permitted action set, resource/deadline constraints and previous candidate-visible decisions. The context digest binds exactly that prefix. No future observation, seed mapping, privileged measurement, trajectory label or post-run score may enter this context.

### Evaluator outcome capsule and manifest

Held by a separate evaluator process or custody domain. Contains hidden trajectories and ground truth, privileged/oracle observations, evaluator-only labels and scores, sealed task/seed mappings, seed/split provenance and outcome-resolution/adjudication records. Its evaluator-only manifest binds source commit and the replay/evaluator capsule digests.

A Rust type boundary is not access control against malicious candidate code. The candidate process must receive only the time-valid decision context and must not be able to read either the full replay archive or evaluator storage. Separate directories are not sufficient without OS/service-level access controls.

A digest proves byte identity, not truth, independence, access control or correct custody. BLAKE3 here is an integrity identifier, not an authenticated signature. The ten fixed public seeds remain a plumbing pilot; they do not create a sealed held-out benchmark split.

## 3. Contract prototype

The new support module defines:
- CandidateDecisionContext: decision-time evidence, exact upstream offer IDs paired with their payload SHA-256 digests, context digest, budget, deadline, and a digest-bound decision objective (valuation mode, declared utility unit and loss-table reference where applicable). It carries exact serialized ResearchState and evidence-prefix SHA-256 values from #13665/#11015. For the offer-list SHA-256, this PR specifies a proposed canonical v1 encoding; upstream #11015 must adopt it before the field can be treated as a production AgentView digest. These are references, not a duplicated evidence graph or action-offer authority.
- DecisionProposal: selected upstream offer ID, exact payload digest (which must match the upstream offer projection, such as the submitted design batch), prospective probability distribution, policy lineage, assignment mechanism and causal-design declaration.
- EvaluatorResolution: evaluator-owned execution status, outcome-resolution state, recorded action ID plus exact payload digest, upstream execution/job receipt reference, evidence references and actual-cost completeness. `ActionSupport::RecordedAction` means the submitted action/payload was recorded; it does not imply that the scientific outcome has resolved. A recorded external action requires a nonempty upstream receipt reference; `not_run` and unsupported counterfactual states must not carry one.

Candidate-visible types do not contain ground truth or future outcome labels. EvaluatorResolution is separate and must not be serialized into the candidate process input stream.

**Cross-owner canonicalization boundary:** #11015 currently defines the policy-level obligation to use an exact permitted offer ID, and it requires admissible actions/resource costs, but its protocol document does not yet define a canonical serialized offer-list hash or a concrete AgentView offer payload schema. This PR therefore proposes PermittedActionOfferRef v1 and a concrete canonical digest for adoption by #11015; it does not claim that the upstream production offer list already emits this digest.

V1 canonicalization is compact serde JSON for a fixed field-order struct containing:
- domain tag symthaea.agentview.permitted-action-offer-list;
- schema version 1;
- the ordered array of { offer_id, action_payload_sha256 } pairs.

The array is order-preserving. Each offer's payload digest must cover the complete immutable offer, including action parameters, resource-cost information, permission/scope and relevant risk constraints. The SHA-256 over those exact bytes is permitted_action_offer_list_sha256. #11015 must adopt this encoding or introduce a versioned compatible canonical representation before production wiring.

Validation covers:
- the canonical ResearchState, evidence-prefix, permitted-offer-list and exact action-payload bindings are well-formed lowercase SHA-256 digests;
- referenced evidence artifact digests use the same explicit SHA-256 encoding;
- no evidence became available after decision time;
- every decision is bound to a frozen downstream decision/loss contract; raw entropy or information gain is not substituted for decision value;
- selected action ID and exact payload digest must match one of the upstream #11015 AgentView offers;
- randomized policy probability mass covers every permitted offer exactly once, including its exact payload digest;
- the exact selected action payload digest is bound through later result resolution;
- prospective probabilities sum exactly to one million parts per million;
- randomized assignment records a complete probability distribution over permitted action identities and exact payload digests, with nonzero propensity for the selected action;
- recorded external execution binds to an upstream executor/Lab receipt reference; unsubmitted and unsupported actions cannot carry an execution receipt;
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

| Execution status | Outcome status | Action support | Meaning |
|---|---|---|---|
| not_started | not_run | no recorded action | No action/experiment was executed |
| not_started | unsupported_counterfactual | no recorded action | Candidate chose a branch with no observed outcome; no outcome is fabricated |
| started | awaiting_observation | recorded action + payload digest | Execution began and the exact submission is known; its scientific result is still pending |
| completed | immediate_observed | recorded action + payload digest | Outcome observed and evidence admitted immediately |
| completed | delayed_observed | recorded action + payload digest | Outcome became observable later |
| completed | censored_at_deadline | recorded action + payload digest | Outcome not resolved before observation deadline |
| completed | indeterminate | recorded action + payload digest | Evidence exists but does not settle the outcome |
| failed / timed_out / aborted | invalid_execution | recorded or absent, explicitly represented | Execution did not produce a valid scientific result |

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
- The generator assigns opaque UUID world IDs, writes the full observation archive plus its replay receipt to the replay-custodian root, and writes hidden outcomes plus the joined custody manifest to the evaluator root.
- Verify capsule byte digests and receipt/manifest/world-ID linkage after disk readback.
- Never pass the current combined serialized-world object, full replay archive or evaluator root into a candidate process. The trusted replay driver must construct and send only the time-valid evidence prefix.

Ten worlds are a custody/schema plumbing pilot only, not a test of adaptive research decisions or a confirmatory scientific benchmark. The evolutionary-rescue fixture is a passive recorded forecast world; it does not expose a sufficiently rich, independently observed set of alternative experimental branches. Do not use its ten seeds to claim off-policy decision quality, counterfactual discovery or research-taste superiority.

### Active research-decision experiment — use the existing AS-Bench adapter

The actual decision-policy comparison should run through #11012 and #10921, reusing [PR #10947](https://github.com/Luminous-Dynamics/symthaea/pull/10947)'s pinned Lab API v1 compatibility contract rather than creating another client or evaluator.

- **Nuclease replay:** four rounds of 96 chosen designs from a 20,000-design pool. It supplies multi-round experiment selection and observed replay outcomes. Discovery and held-out NDCG@100 learning gates remain separate.
- **Propylene digital twin:** three rounds of 20 designs from a 736-design pool. It tests asynchronous/twin compatibility and adaptive design ranking. Preserve the 96 published measurements versus 884 surrogate outputs; do not describe twin results as live measurements.

Use fixed script, fixed sequential/best-of-N, static threshold, state-contingent adaptive and matched-cost random-branch controllers from #11012. Keep model, available roles, coder, API/tools, context and total cost fixed; only the orchestration policy changes. Apply the decision episode to each actual batch submission, binding the research-state/evidence prefix, selected-design payload digest, predictions, resource estimate, next-action rule and actual lab job/result identity. Keep the benchmark-owned discovery and learning gates intact.

The pinned compatibility adapter PR #10947 has 38 authored tests. Its dedicated `Sym-RSI AS-Bench Adapter` GitHub Actions workflow passed on exact head `f23caa7b8a177f002ee25eaec34404d4e675946f`: formatting, all 38 adapter tests, strict Clippy, compatibility-manifest validation and Lab API fixture validation. Other general workflows on that head were skipped. This is offline contract qualification only; it has not executed a campaign or reached the upstream verifier.

The new `examples/support/asb_action_payload.rs` defines `AsBenchBatchPayloadV1`: a versioned, fixed-field representation of API version, compatibility-manifest SHA-256, campaign ID, stage, design-ID field and exact ordered design IDs. It validates stage batch size, catalog membership and uniqueness, then computes lowercase SHA-256 over the pinned serde JSON representation. This digest is the value to place in `DecisionProposal.selected_action_payload_digest`; the later evaluator resolution binds the same digest to the #10947 completed-job/result receipt. Reordering a submitted batch intentionally changes its digest. The helper does not reimplement the Lab API client, catalog parser or asynchronous job lifecycle.

Example custody-archive commands (replace paths and source SHA with actual campaign values; the first output is a custodian-only archive, not candidate input):

```sh
cargo run --release --example evolutionary_rescue_generate_worlds -p symthaea-futures-ensemble
cargo run --release --example evolutionary_rescue_split_capsules -p symthaea-futures-ensemble -- /secure/replay-archive /secure/evaluator <40-or-64-hex-source-commit> census
```

The materializer creates both roots if absent, rejects equal/nested roots, requires both roots to be empty before starting, refuses to overwrite existing files, writes the manifest last, and reads all four output artifacts back before accepting the pair. It does not set filesystem ACLs, create separate OS identities or launch the candidate/evaluator processes. Those are required deployment controls, not provided by this fixture utility.

### Pilot 1 — compare decision policies on AS-Bench

Use the Nuclease replay campaign as the first active choice environment; add the Propylene twin only as a separate backend/generalization lane. Freeze task generator, campaign version, executor, evaluator, role/tool/model set and per-treatment resource ceiling. Branch policies from the same initial evidence and task instance.

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

### Pilot 2 — replay-to-live / replay-to-twin evidence audit

Compare predicted action/policy rankings with actual selected-action outcomes only on supported task states. Mark unobserved branches unsupported. Distinguish replay-table outcomes, surrogate twin outputs and actual physical measurements as different evidence classes.

### Pilot 3 — live replay-fidelity and decision audit

Preregister a sample of candidate policy decisions for live execution. Estimate replay-vs-live ranking agreement and top-k recall; report selection regret and policy coverage; stratify by task family and support; retain not-run, failed, censored and indeterminate outcomes. Do not promote from replay-only evidence.

### Pilot 4 — matched-start R4

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
- **Evidence debt, not invisible uncertainty:** every active claim carries unresolved obligations, stale-support triggers, counterevidence and a due revalidation condition; unresolved debt blocks promotion rather than being hidden in a confidence scalar.
- **Event-time science, not timestamp flattening:** distinguish when an event occurred, when it was observed, when the result became available and when it was interpreted. A future result must remain unavailable to the policy even when its event timestamp is earlier.

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
| Scientific terminal workflows | Terminal-Bench-Science 0.1 / current release | Beat matched full-release results on independently re-run task artifacts; report domain-level and planning-stage outcomes |
| Project-level science | ASI-Bench | Improve full-project performance across the B1–B4 guidance ladder with repeated trials and pinned task-set identity |
| Long-horizon work | METR TH1.1 / TH2 | Improve long-horizon success without hiding lifecycle costs or judgment gaps |
| Research engineering | SWE-bench Pro/Verified, Polyglot, MLE-bench, FML-Bench | Matched-backbone/cost comparison and capability retention |
| Algorithm/kernel discovery | KernelBench, SOL-ExecBench, ALE-Bench | Correctness-first speedup and verified discovery per total compute |
| Meta-improver | AIDE²-style identical-start R4 | Repeated positive paired successor advantage at matched realized cost |

### Provisional stretch targets from the existing decision-intelligence benchmark plan

These are ambitious targets to validate and power after a variance-estimation pilot; they are not achieved Symthaea results or immediate pass/fail gates.

| Benchmark | Public reference / scale | Proposed Symthaea stretch target |
|---|---|---|
| Autonomous Science Bench 0.1 | Six tasks, three trials each; 10 configurations; GPT-6 Astra 72.2%, Opus 5.5 66.7% | At least 15/18 resolved trials (83.3%), with discovery and held-out learning gates both passing for every resolved campaign; repeat on a fresh campaign family |
| Terminal-Bench-Science 0.1 | 70 expert-curated tasks across five domains; three independent trials per task; Claude Opus 5 30.0%, GPT-5.6 Sol 22.4% on the initial release | At least 45% on the pinned 0.1 release as a historical floor; on the current 0.2/1.0 release, exceed the strongest independently reproduced contemporaneous system by at least 5 absolute points at matched cost and harness, with no integrity failures and no per-domain regression |
| ASI-Bench | 60 project-level tasks across 11 domains, B1–B4; leaderboard has 26 supplied results, 24 using three runs and two single-run exceptions; tool access disabled | Exceed 68/100 on the exact pinned task set with at least five paired runs and tool/compute parity; report each guidance level separately and do not claim leaderboard superiority until the official task-set identity and artifact-level scoring are reproducible |
| FIRE-Bench | Best published F1 46.7 on 30 tasks / three trials; 73.6% of categorized errors arise in planning | At least 55 F1 on the same release, while reducing planning-stage failure rate by at least 50% relative to the matched baseline and reporting claim precision, recall, trial variance and unsupported claims |
| SCI-AGENT-EXAM | 120 expert-curated tasks across 11 fields and 7 task types; GPT-6 Astra reports 80.9 overall, with 26.0% runs above the 0.95 success threshold | At least 85 overall and at least 50% of runs above 0.95, with executable/text artifact agreement, per-task-family reporting and exact-release reproduction |
| Paper2Agent agentification | 74/100 computational-biology papers agentified; 593/599 tools passed automated validation; Sonnet 4 achieved 91.2±1.6% on 300 tutorial questions | At least 85% end-to-end agentification on a stratified corpus, at least 99.5% tool-contract validation, and 95% on matched tutorial/execution questions; scientific conclusion validity remains separately adjudicated |
| Into the Unknown | 0/135 follow-up submissions met all necessary conditions; 114 failed integrity checks in the reported study | Zero unsupported claims in the admitted subset, 100% claim-to-artifact binding for quantitative claims, and explicit not-performed/indeterminate/invalid dispositions |
| SciAgentArena | ~200 tasks, 5 scientific domains, 4 task categories, 18 evaluated agents; representative best-agent results include hERG 68.1 AUROC, DEG detection 29.4 F1, spatial genes 47.6 Jaccard, diagnosis 70 accuracy and PRS 80 accuracy | At least 10 absolute points above the strongest reproduced baseline on 3/5 matched representative tasks, with no more than 5-point regression on the remaining tasks; then transfer to held-out task families |
| SciWorkBench | Best reported mean 57.38 | At least 65 on the same official harness/scorer, with domain-specific failure rates |
| RecurSciBench | 20 independent test tasks across four domains; evolved states show mixed mean transfer and broad per-task regressions | Mean transfer above +5%, nonnegative aggregate in every domain, and no more than 4/20 independent test tasks regressing; confirm on expanded task families |
| TasteVal-style fixed-coder lane | Reported experimental-compute multiplier up to 2.30x; researcher inference and full lifecycle costs are not included in that multiplier | Match or exceed 2.30x compute-to-target while preserving final-quality non-inferiority and reporting a separate inclusive-cost frontier |
| METR TH1.1 / TH2 | Long-horizon tasks with materially different lifecycle costs | At non-inferior success, reduce inclusive lifecycle cost by 30%, or extend successful task horizon by 1.5x at matched total cost |
| SWE-bench Pro (secondary engineering lane) | HGM reported 88.3% on its evaluated subset | Exceed 88.3% only on the same pinned subset/harness, while retaining Verified/Polyglot transfer and reporting full compute/evaluation cost |

The AS-Bench leaderboard reports ten configurations over six tasks with three trials per task; its own overview cautions that small between-configuration differences are not yet meaningful at this sample size. Terminal-Bench-Science 0.1 is the first 70-task five-domain baseline and is deliberately hard; its newer 0.2 calibration is against Claude Opus 5.5 and GPT-6 Astra, so the old 30% result is not a current-frontier claim. ASI-Bench is broader and project-level but its supplied aggregate leaderboard notes that it does not recompute scores from task artifacts and that the exact scored task-set identifier is unverified; it is a comparison anchor, not an independently audited truth source. The 15/18 target is a pilot-stage stretch threshold, not a statistically qualified superiority claim. Paper2Agent's 593/599 is automated tool validation, not scientific claim validity. SciAgentArena's task metrics are different units; its representative task values must not be averaged together. FIRE-Bench's planning-error share is a diagnosis of that benchmark's task set, not a universal breakdown of scientific failure.

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
- #11171: open-world research-policy adapter and production callable; this remains the owner of the missing state-contingent researcher.
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

### Production policy and deterministic baseline boundary

The production researcher is not yet the adaptive policy being proposed here. A source audit of `src/web_research/researcher.rs` shows heuristic query expansion, fixed fetch/extract/verify sequencing and static domain-targeting rules. `EpistemicConsciousness::research` still has a confidence-derived correctness shortcut; draft #13612 removes that feedback path. The researcher also seeds source support as a placeholder before verification; draft #13634 separates topical relevance from claim support and must not be bypassed by restoring automatic `supports=true`.

A deterministic public-state comparator already exists on branch `research/stateguard-b0-policy-stacked-v1` (StateGuard B0, 21 authored synthetic tests). It consumes typed ResearchState, upstream offers and componentwise budgets, and can only SELECT/ABSTAIN/STOP. It does not execute an action or adjudicate truth. Its first dedicated Rust 1.96 workflow failed at the formatting step; tests, synthetic replay and Clippy were skipped. Therefore B0 is an implemented comparator candidate, not yet a qualified baseline.

The next policy experiment belongs in #11171/#11015: freeze B0 as the rule-based comparator, then compare matched fixed, random, static-threshold, EVSI/robust-minimax and state-contingent policies. Use identical ResearchState bytes, upstream offer list, executor, model/tool availability, and inclusive budgets. The exact sequential oracle stays evaluator-only. Do not put another scheduler or research policy inside this episode-contract PR.

The current futures ledger records forecast provenance and scored continuations. This contract is a decision-episode interface, not a competing ledger or ResearchState authority. #13665 owns the typed public ResearchState projection and its exact serialized SHA-256; #11015 owns prospective policy traces and binds the ResearchState digest, evidence-prefix digest and upstream permitted-action-offer-list digest. #13933 must consume those three upstream identities unchanged and add only scientific episode semantics: decision phase, selected action plus exact payload digest, prospective outcome, assignment mechanism, causal-design declaration and evaluator resolution. It must not reconstruct the evidence graph, re-hash a second state, mint new action offers or replace the canonical action trace. Persistent writing should reference existing evidence/receipt identities and be assigned to an existing owner. Adapt this contract to #13663/#13665/#11015, the decision objective and event semantics in #12546/#12407 rather than duplicating their state, calculations or observation model.

## 11. External research

- [Dream-RSI](https://arxiv.org/abs/2609.14858): replay-based exploration-policy improvements; motivates a policy baseline, not automatic transfer.
- [AIDE²](https://arxiv.org/abs/2609.26457): recursive improver comparator; same-start comparison was explicitly inconclusive at three seeds per arm.
- [ELITE](https://arxiv.org/abs/2611.21056): evaluation-free mechanism integration; independent behavioral tests remain necessary.
- [RecurSciBench](https://arxiv.org/abs/2611.32508): recent RSI transfer benchmark with four scientific domains and 96 tasks; its paper emphasizes that aggregate gains can coexist with per-task regressions.
- [METR Q4 2026](https://metr.org/blog/2026-11-20-q4-2026-capability-update/): verifiable task horizons continue to change rapidly; judgment without fast feedback and full lifecycle-cost measurement need separate lanes.
- [TasteVal](https://arxiv.org/abs/2610.06824): fixed-coder comparisons can isolate experimental-planning quality; compute-only multipliers are not inclusive research-cost multipliers.
- [IdeaTaste](https://arxiv.org/abs/2611.03424): retrospective idea matching is a diagnostic, not a scientific-validity oracle.
- [SCI-AGENT-EXAM](https://arxiv.org/abs/2610.32441): 120 expert-curated tasks across 11 fields and seven scientific workflow task types; evaluates executable and textual artifacts and run stability.
- [Paper2Agent](https://www.nature.com/articles/s41586-026-11044-y): 100 computational-biology papers in its primary agentification cohort; 74 were agentified and 593/599 generated tools passed automated validation. Tool validation is distinct from independent scientific-claim validity.
- [SciAgentArena](https://sciagentarena.github.io/): approximately 200 stepwise-verified tasks across five scientific domains and four task categories; its published representative task scores use different metrics.
- [Autonomous Science Bench 0.1](https://asb.open-ssi.org/leaderboard): six pilot tasks, three trials each, with GPT-6 Astra at 72.2% and Opus 5.5 at 66.7%; the benchmark explicitly cautions that small differences are not meaningful at this run count.
- [FIRE-Bench](https://firebench.github.io/): 30 research rediscovery tasks, claim-level precision/recall/F1 scoring, and a published best F1 of 46.7 over three trials; the benchmark reports planning as the largest categorized error source.
- [Terminal-Bench-Science](https://www.tbench.ai/news/terminal-bench-science-0-1): 70 expert-curated scientific workflows across life, physical, Earth, mathematical and engineering sciences; 0.1 reports three trials per task and a 30% top resolution rate. The maintainers describe 0.2 as calibrated against Claude Opus 5.5 and GPT-6 Astra, and 1.0 is the next community task-contribution target.
- [OpenResearch agent-evaluation dashboard](https://openresearch.sh/evals): its current Terminal-Bench-Science aggregation includes newer vendor-reported values (e.g. 77.1% and 75.3%) that it explicitly says are not yet on the official leaderboard. Treat them as reported claims, not official or independently regraded results; only compare after task-version, harness, trial count and raw artifacts are reconciled.
- [ASI-Bench](https://asibench.apexin.ai/): 60 project-level tasks across 11 domains with B1–B4 guidance withdrawal; its supplied aggregate page lists 26 configurations, but 2 use one run and the page says task-set identity is not independently pinned in the results view.
- [Into the Unknown](https://arxiv.org/abs/2610.19159): open scientific follow-up benchmark with expert-authored necessary conditions and a demanding claim-integrity check.

These are paper/author-reported or benchmark-host-supplied baselines unless independently reproduced under matched task, harness and total-cost boundaries. No claim that Symthaea beats them is made here. Never compare numeric scores across different benchmarks.
 
## 12. Implementation status and gates

Implemented in this branch:
- candidate-visible decision context and prospective proposal, including a decision-objective reference and required exact ResearchState/evidence-prefix/action-offer-list SHA-256 bindings from #13665/#11015;
- versioned canonical candidate-context SHA-256 over all candidate-visible decision fields; the validator recomputes it and rejects time, phase, objective, evidence, offer, state or budget substitution without a matching digest;
- outcome distribution, full randomized action propensities over action/payload identities, exact selected-payload binding and causal-design declarations;
- versioned AS-Bench batch payload SHA-256 over campaign, stage, manifest, ID field and exact ordered design IDs;
- evaluator-owned resolution and distinct execution/outcome states;
- validation for look-ahead evidence, unique decision-evidence IDs, action support, unsupported counterfactuals, delayed/censored/indeterminate outcomes and cost completeness;
- nineteen authored decision-contract tests, including canonical candidate-context integrity and duplicate-evidence negative controls;
- custodian-only replay-archive/evaluator projection, BLAKE3 digests, opaque IDs, empty/disjoint-root checks, no-overwrite writes, manifest-last materialization and disk-readback checks;
- decision-time observation-prefix builder with digest and out-of-range rejection;
- nine authored custody/prefix tests, seven AS-Bench payload/offer-binding tests, two materializer output-root tests and a capsule materializer example (37 authored unit tests across the new Rust files; 35 are covered by the standalone harness and the two materializer tests still await the production-example workspace build).

Not implemented or not yet verified:
- the ten worlds have not been generated or split in this environment; the existing source seed split is public and is not a sealed confirmatory split;
- the capsule utility does not implement a time-prefix replay driver, enforce OS/service access control or launch isolated candidate/evaluator processes;
- current fixture observation timestamps remain simulation ticks, not full #12407 event/availability-time semantics;
- integration with existing ledger/archive/receipts/attestation;
- executable replay policy, live-fidelity audit, external benchmark runner or R4.

Follow-on candidate-context integrity gate:
- The candidate-visible context digest was added after the last confirmed green head `bddab8d62ebd9afb1df4a8863371385d60b9a386`; it raised the harness to 34 authored tests. A further guard rejects duplicate evidence IDs, raising the current harness to 35 authored tests (19 decision-contract, 9 custody/prefix, 7 AS-Bench).
- The first follow-on run passed rustfmt but reported 32/34 tests because two negative tests changed a context field while retaining the previous context digest. The digest guard correctly rejected those stale specimens before the tests reached their intended semantic assertions.
- Those tests now recompute the canonical context digest after the mutation, ensuring they test the look-ahead and decision-objective guards rather than the earlier digest guard.
- Duplicate evidence identifiers are now rejected before decision scoring, with a negative test that recomputes the context digest so the uniqueness guard is reached.
- The exact current PR head is `4dafba8e5ed6b6c7567a2cb9c33d3f0f9711c893`; the previously queued run 84 predates the duplicate-evidence change, and a fresh exact-head result has not yet been observed. No 35/35 or Clippy pass is claimed for this follow-on.

Verification boundary (exact tested code head `bddab8d62ebd9afb1df4a8863371385d60b9a386`):
- Focused Rust 1.96 PR and push workflow runs both succeeded after the standalone dependency lockfile was committed.
- `rustfmt --check` passed for all included production support modules and the standalone harness.
- Standalone harness ran with `--locked`, compiled the production support-module source against shape-matched domain shims, and passed **33/33 tests**: 17 decision-contract, 9 custody/prefix and 7 AS-Bench payload/offer-binding tests.
- Strict Clippy with `--locked` for all standalone harness targets passed.
- The canonical v1 permitted-offer-list digest was recomputed from the exact ordered offer ID/payload pairs; substitution and order-change negative controls passed.
- The lockfile is committed at `tools/sym-rsi-decision-replay-contract/Cargo.lock` (30 registry/workspace package entries); both Cargo test and Clippy ran with `--locked`.
- This is isolated schema/custody/adapter-contract qualification only. It does not establish integration with the real `symthaea-futures-symtropy` crate, a full workspace build, or compilation of the capsule-materializer example; its two output-root tests remain outside the standalone harness.
- The ten-world materializer has not been run. No OS/service-level access controls, separate evaluator process, live AS-Bench campaign, external benchmark reproduction, scientific-quality result, SOTA result or R4 qualification is claimed.
- Broad workspace CI and benchmark workflows were skipped by repository policy. The root workspace also currently cannot be enumerated by Cargo because its `crates/core/*` glob includes a missing `crates/core/symthaea/Cargo.toml`; the standalone harness intentionally avoids masking that repository-level problem.

Do not merge or promote on schema tests alone. The next technical gates are production-crate integration after workspace closure repair, actual filesystem/service custody, execution of the ten-world materializer, and an AS-Bench campaign.
