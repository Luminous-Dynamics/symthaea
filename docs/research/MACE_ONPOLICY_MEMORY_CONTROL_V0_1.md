# MACE On-Policy Memory Control v0.1

**Status:** research protocol specification. No AMemGym score, closed-loop policy gain, privacy qualification, or human-user result is claimed.

**Tracking issue:** [#13388 — qualify on-policy memory control under endogenous disclosure and feedback](https://github.com/Luminous-Dynamics/symthaea/issues/13388)

## 1. Research question

Does a memory controller improve long-horizon assistant outcomes when the assistant's own responses alter what information becomes observable in later interactions?

Most static memory evaluations replay a fixed transcript. That estimates performance conditional on an already-realized history. In deployment, the history is partly endogenous:

- memory affects retrieval;
- retrieval affects the assistant response;
- the response affects what a user chooses to reveal next;
- later observations affect memory updates and subsequent responses.

A system can therefore improve apparent recall by soliciting more information, eliciting information through leading questions, or causing users to disclose information that is not necessary for the task. A static replay score does not measure the total effect or cost of this behavior.

MACE should qualify both **off-policy evidence use** and **on-policy interaction policy** while keeping them separate.

## 2. Benchmark grounding

### AMemGym — primary interactive protocol

AMemGym (ICLR 2026) introduces on-policy evaluation for conversational memory. It uses structured user profiles, state-dependent questions and state-evolution trajectories; LLM-simulated users expose latent state through interactive conversations.

- Paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/0856bc553d3e3b9827e5140d0ad3bf8d-Abstract-Conference.html
- Project, code and dataset entry point: https://agi-eval-official.github.io/amemgym/

MACE must first reproduce the published AMemGym protocol and report its official metrics as a distinct benchmark lane. The MACE-specific measures in this document are additional diagnostics, not replacements for official scoring.

### Complementary protocols

- **MemoryAgentBench (ICLR 2026):** incremental multi-turn evaluation of accurate retrieval, test-time learning, long-range understanding and selective forgetting. https://proceedings.iclr.cc/paper_files/paper/2026/hash/fd1eff9dd295df50a41f2521942fa31d-Abstract.html
- **Beyond Recall:** multimodal temporal memory across images, videos and documents, with 11 fine-grained retrieval, reasoning and management tasks. Pin the exact dataset revision and its media/license card. https://beyond-recall.github.io/
- **MemoryArena / AMA-Bench:** later-task transfer, trajectory reasoning, state updating and causal/objective memory. These remain separate protocols.
- **MemSecBench / MemoryDojo / MemPoison:** poisoning, persistence, propagation, consequences and repair. These remain separate security lanes.

Do not pool percentages across these benchmarks.

### 2.1 AMemGym implementation snapshot (pinned 2026-11-23)

Repository: https://github.com/AGI-Eval-Official/amemgym/tree/ffcd18857a3e2b2c61f00730ebdec676e27d3e87

The current public repository snapshot exposes four directly comparable agent configurations and a separate evolution experiment. Pin the config bytes and runner code as well as the repository commit.

| Arm | Configuration | Important settings in the published file |
|---|---|---|
| Native | `configs/agent/native.json` | `gpt-4.1-mini`; no external memory module |
| AWI | `configs/agent/awi.json` | `gpt-4.1-mini`; in-context memory; `update_bsz=2`, `local_length=4` |
| AWE | `configs/agent/awe-2-4-30.json` | `gpt-4.1-mini`; Milvus; `text-embedding-3-small`; `top_k=30`; LLM memory policy enabled |
| RAG | `configs/agent/rag-2-4-30.json` | Same listed backend/model/top-k as AWE, with LLM memory policy disabled |

### Published AMemGym reference scores and MACE targets

The official AMemGym project page currently lists these memory-agent references under its published setup:

| Agent | Overall score | Memory score |
|---|---:|---:|
| AWE-(2,4,30) | 0.434 | 0.296 |
| Mem0-G | 0.426 | 0.284 |
| RAG-(2,4,30) | 0.384 | 0.223 |
| AWI | 0.353 | 0.177 |

Source: https://agi-eval-official.github.io/amemgym/ (public results displayed 2026-11-23). The site states that these memory-agent comparisons use `gpt-4.1-mini` as the shared base LLM. These are AMemGym-specific normalized scores; they are not comparable to AML, LongMemEval, AMA-Bench or other benchmark percentages.

**MACE prospective challenge:** first reproduce AWE-(2,4,30) and Mem0-G under the same dataset, simulator, model, metric and resource protocol. Then target at least **0.346 Memory Score** (five absolute points above the listed 0.296 AWE reference) while keeping Overall at least 0.434 and without worsening Write, Read or Utilization failures. A second target is a strictly better utility/cost Pareto point, such as at least 10% lower all-in lifecycle cost at non-inferior Memory Score. Treat these as stretch goals; the published figures must be revalidated under the exact pinned release before becoming confirmatory baselines.

The AMemGym paper/project also reports that off-policy replay can change configuration rankings by up to three positions. Therefore, never optimize MACE against the off-policy result and assume that the selected memory policy remains optimal on-policy.

Published baseline config blob SHAs at the inspected snapshot:

- AWI: `e665b6ad3d87c423ace125318598fb785eb8cf28`
- AWE: `c272de3cf40db86edfc40b11ea844776b4be7c2b`
- RAG: `9c1d9cd1b8ff47bd292ffe2fd6ecc9790acdcc59`
- Native: `6bd40ffc7029e993b89202394ab6209c6022af6b`

Environment configuration `configs/env/v1.base.json` (blob `0683d720b945fed55f472ec492ba66d7be014381`) specifies 20 user profiles, 10 questions, 10 periods, three months per period, four changes per period, one initial interaction round and two update rounds. Its user-simulation model is `gpt-4.1` at temperature 0.2; data generation uses `gpt-4.1` at temperature 1.0; the declared seed is 42. Record the actual generated dataset revision and all simulator/API fingerprints: a Python random seed alone does not make hosted LLM responses deterministic.

The released interaction generator (`src/amemgym/env/sample_interactions.py`, blob `dd308d4d6bbb74fa7bd7a20f2cf716d51337c69a`) includes the user profile, tracked state schema, current date and the last two conversation messages in the simulated-user follow-up prompt. That prompt instructs the simulated user not to reveal values for tracked schema variables and explicitly says early stopping is not allowed. Therefore:

- reproduce this behavior unchanged in the official AMemGym lane;
- do not interpret its official interaction count as a naturalistic question-burden or disengagement measure;
- test user refusal, early stopping, sensitive-data elicitation and over-questioning only in a separately labelled MACE extension.

The overall runner (`src/amemgym/eval/overall.py`, blob `c185fad8847ff0d8545429af001c4d8058dd6829`) has distinct on-policy and off-policy paths. On-policy generates new sessions through the interactive simulator; off-policy loads saved interaction transcripts. They must be run and reported separately. The current metric list contains `accuracy`. When answer JSON parsing fails, the runner substitutes a random choice and records a `json_error` internally; the top-level metric file does not expose the format-error rate. Preserve official scores for exact reproduction, but additionally report malformed-response count/rate as a separate diagnostic and never let random fallback be mistaken for valid model output in MACE-specific scoring.

The separate evolution runner can update the memory prompt after each period. Its feedback vocabulary includes `with_answer`, which inserts the gold answer into evolution feedback. **There is a source/configuration discrepancy to resolve before claiming exact reproduction:** at the inspected public repository snapshot, `configs/agent/awi-evolve/complete.json` sets `feedback_types` to `["vanilla"]`, while `collect_evolution_feedback` adds the gold `ground_truth` field only when `with_answer` is present. The official project page describes its Complete condition as including the ground-truth answer. Do not silently change the config or infer which artifact generated the published result; obtain the exact run config/code receipt or report a separate, clearly labelled `with_answer` ablation. The `question-only.json` config is a distinct feedback condition and `no-evolution.json` is a no-update control. For MACE held-out evaluation, policy adaptation must be trained on training/calibration profiles only and frozen before held-out episodes; no held-out answer or evaluator label may enter adaptation feedback.

### 2.2 MACE adapter requirements

- Implement the benchmark adapter through AMemGym's `BaseAgent` interface (`act`, `add_msgs`, `answer_question`, `save_state`, `load_state`, `reset`). Keep the official Python benchmark harness as an external evaluation dependency; it is not a reason to move production MACE memory logic out of Rust.
- Use the official `v1.base` release and pin the exact dataset revision, user personas, generated environment data, all config files, simulator model/API fingerprints, and scoring code. Do not regenerate the dataset for the primary reproduction unless separately reporting a newly generated environment.
- Preserve AWI, AWE, RAG and Native as the official reference arms. Add MACE as a new arm with the same outer agent backbone, prompt contract, user simulator and budget. Do not replace the official arms with internal HDC/KV variants.
- Report Write, Read and Utilization diagnostics separately from end-task accuracy. Preserve official JSON fallback behavior only for protocol reproduction; expose its error count and use strict validity accounting in the MACE extension.
- Run the official AMemGym evolution configurations as a distinct research experiment. Never adapt on sealed/held-out answer labels; bind every evolution step to a training/calibration feedback receipt and a new policy digest.

## 3. Formal interaction model

Let:

- (z): evaluator-owned latent user/task state;
- (h_t): candidate-visible interaction history through time (t);
- (m_t): candidate memory state;
- (a_t): assistant response or interaction action;
- (o_{t+1}): next user/environment observation;
- (pi): the complete agent-plus-memory policy.

The environment generates future observations conditional on the interaction so far:

`o_{t+1} \sim P(o \mid z, h_t, a_t)`

The memory controller updates from permitted observations:

`m_{t+1} = M(m_t, h_t, a_t, o_{t+1})`

The next response is generated from the resulting memory and visible history:

`a_{t+1} = A(m_{t+1}, h_{t+1})`

The evaluator retains (z), future state, simulator parameters and answer labels. Candidate code receives only permitted observations and public task/action/cost contracts.

### 3.1 Two distinct estimands

**Off-policy conditional performance**

Evaluate policies against the same frozen history (h). This measures retrieval and answer quality conditional on a fixed transcript:

`V_off(\pi) = E[Y(\pi; h)]`

It is useful for component diagnosis, but it does not measure how the policy shapes future observations.

**On-policy total closed-loop value**

Randomize the complete policy at the episode/profile level and allow each interaction to evolve naturally:

`V_on(\pi) = E_{z,U}[Y(\pi; z,U)]`

Here (U) represents the environment/user-simulator randomness. The primary policy contrast is:

`\Delta_on = V_on(\pi_{MACE}) - V_on(\pi_{baseline})`

This is the total effect of the assigned policy, including changes to future disclosure and interaction. Do not condition the primary total-effect estimate on post-assignment transcript length, number of disclosures, or later observations: those can be mediators of the policy effect.

### 3.2 Mechanism analysis

Report the interaction pathway separately:

`policy \rightarrow response \rightarrow disclosure \rightarrow memory\ update \rightarrow later\ decision`

Mediation estimates require additional assumptions and should be labelled exploratory unless the simulator supports a preregistered intervention on the mediator. Do not infer that increased disclosure caused better utility merely because both increased.

## 4. Experimental tracks

### Track A — official off-policy reproduction

1. Freeze the official AMemGym release, profile/task generators, user simulator, prompts, agent backbone, scoring code and environment.
2. Reproduce its declared baseline systems and metrics.
3. Preserve its official split and run settings.
4. Record deviations and method changes.
5. Do not adapt memory policy parameters using held-out profiles or answers.

### Track B — on-policy evaluation

1. Randomize complete memory policies across independent episodes, stratified by latent profile and task family.
2. Keep the agent backbone, system prompt, tools, simulator version and interaction budget matched within a comparison.
3. Allow conversations and observations to diverge naturally after policy-specific responses.
4. Use common random numbers only where the simulator contract guarantees that they do not reveal future state or invalidate the intervention.
5. Estimate policy value over independent episode seeds and cluster uncertainty by latent profile/task generator. Turns within one conversation are not independent samples.
6. Record every prompt/response, user observation, state transition, memory mutation, retrieval, tool action and cost with sequence numbers and source receipts.

### Track C — policy adaptation

1. Permit memory-policy adaptation only on declared training and calibration profiles.
2. Freeze the policy, prompts, retrieval configuration and thresholds before held-out evaluation.
3. Include a no-adaptation control.
4. Treat any test-time change to policy weights, rules or configuration as a new policy version with a new manifest.
5. Evaluate leave-one-profile-family-out, leave-one-task-family-out and simulator-version shift.

### Track D — MACE-specific controlled fixtures

Build small deterministic fixtures for mechanisms that can be exhaustively understood:

- stable preferences and constraints;
- explicit preference revisions;
- latent state changes without explicit announcement;
- conflicting statements with different source authority;
- missing or delayed information;
- useful information that is not volunteered;
- leading questions that induce irrelevant disclosures;
- excessive questioning that increases burden or refusal;
- privacy-sensitive information that is not needed for the task;
- stale memory that causes repeated or inappropriate questions;
- confirmation loops where a mistaken memory changes the assistant response and then elicits apparent confirmation;
- simulator or tool errors that must not be recorded as successful observations.

These fixtures are development controls. They do not replace official AMemGym data or consented human evaluation.

## 5. Matched baselines

Use one frozen agent backbone and tool harness per comparison. Report different model families and model revisions separately.

1. No external memory / current context only.
2. Full conversation history under the same context budget.
3. Fixed-window and recency memory.
4. Raw chunk BM25, dense retrieval and hybrid reciprocal-rank fusion.
5. Extracted facts with correction and supersession.
6. Raw evidence plus extracted facts.
7. Hierarchical abstraction with cue-anchor expansion.
8. Model-conditioned memory dose, including a valid zero-memory fallback (#13363).
9. MACE evidence-qualified controller with accept / retrieve / acquire / abstain behavior.
10. Evaluator-only latent-state oracle as a non-deployable upper bound. It must never enter candidate memory or candidate runtime.

Match model, prompt, tool, interaction, context, token and latency budgets. Where a budget cannot be matched exactly, show the full cost vector rather than hiding the mismatch in a scalar.

## 6. Outcome and safety scorecard

### 6.1 Primary utility

- Closed-loop task success and user/task utility.
- Regret against the evaluator-owned task optimum.
- Held-out utility difference against the strongest matched deployable baseline.
- Worst-profile and worst-task-family outcomes.
- Completion, refusal and attrition rates.

### 6.2 Memory correctness

- State acquisition precision/recall against evaluator-owned state.
- Preference and constraint update accuracy.
- Time to incorporate an explicit correction.
- Stale-state use after correction.
- Historical as-of answer accuracy.
- Unsupported inference rate.
- Provenance retention through write, consolidation, retrieval and use.

### 6.3 Interaction burden and disclosure

- Questions asked and interaction turns per successful task.
- Relevant state gained per question and per unit of interaction burden.
- Unnecessary question rate.
- Irrelevant or redundant disclosure elicitation.
- User refusal, disengagement and task abandonment.
- Leading-question and confirmation-loop incidence.
- Privacy-boundary violation and unauthorized inference.

Disclosure volume is not a standalone reward. More disclosure is not inherently better.

### 6.4 All-in cost

- Memory bytes, evidence sidecar bytes and index bytes.
- Input/output/context tokens.
- Write, update, consolidation, retrieval and verification calls.
- p50/p95/p99 latency.
- CPU/GPU time, peak memory and energy where measurable.
- User interaction cost and monetary cost.
- Recovery, replay and correction cost.

Publish utility-versus-cost and burden-versus-utility frontiers. Do not collapse unlike resource types without a declared application utility function.

## 7. Hard gates and negative controls

These are non-compensable qualification requirements; a higher task score cannot offset a failure.

1. **Latent-state isolation:** candidate cannot read simulator internals, future state, evaluator labels or random-state seeds.
2. **Hidden-state mutation invariance:** changing evaluator-only labels while holding the candidate-visible prefix fixed must not change candidate input bytes or candidate-visible logs.
3. **Future-state isolation:** candidate input at (t) must be invariant to mutations of future simulator state that have not yet been observed.
4. **No hidden-label feedback:** evaluator scores, task answer keys and intervention labels are not sent to candidate code during held-out execution.
5. **Privacy:** do not solicit, retain, infer or expose data outside the declared task purpose and consent contract.
6. **Correction:** an explicit correction invalidates stale derived state at current use; historical queries remain answerable as-of their original knowledge time.
7. **No resurrection:** retraction and erasure remain effective after restart, compaction, replay and backup restoration.
8. **Tool reconciliation:** a claimed external success is not stored as a successful outcome until an independently observed receipt reconciles it.
9. **No over-questioning reward:** extra irrelevant or sensitive disclosures must not improve the score.
10. **Policy freeze:** held-out test outcomes cannot silently update the evaluated memory policy.
11. **Cost integrity:** count the complete interaction and memory lifecycle, not just the final Search call.
12. **Independent evaluator:** held-out latent state and scoring remain under separate custody for confirmatory results.

## 8. Prospective targets

These are qualification targets, not current MACE results. Freeze a pilot, estimate variance and statistical power, then preregister confirmatory thresholds.

- Reproduce the official AMemGym baseline scores within the expected uncertainty of its frozen protocol before claiming a MACE extension.
- Demonstrate positive held-out on-policy utility against the strongest matched deployable baseline, with a cluster-level confidence interval whose lower bound exceeds zero.
- Use a stretch target of at least **5 percentage points** of closed-loop task-success improvement only where the task metric and sample size support that target.
- Achieve at least 95% accuracy on explicit state correction in the controlled fixture set, while separately reporting unresolved/ambiguous cases.
- Zero unauthorized latent-state access and zero evaluator-label leakage in all registered negative controls.
- Zero confirmed privacy/authority hard-gate violations in the qualification fixture suite.
- No increase in unnecessary questioning or privacy burden beyond a preregistered non-inferiority margin.
- Demonstrate positive transfer to held-out profile and task families; report any family where performance regresses.
- Publish the complete cost, interaction-burden and utility frontiers, including failed and abandoned episodes.

Do not set a universal numeric stale-use or privacy threshold before the event taxonomy and denominator are frozen. Report counts and denominators for every hard-gate failure.

## 9. Reproducibility manifest

Every result must bind:

- benchmark name, release, dataset revision and license/media card;
- profile/task generator and user-simulator version;
- evaluator and scoring implementation;
- agent model/provider/revision/reasoning mode;
- system/developer prompts and tool schemas;
- memory controller, extractor, retriever and consolidation digests;
- initial memory state and write/search protocol versions;
- episode/profile split and seed schedule;
- interaction, token, context, tool and wall-clock budgets;
- hardware/runtime/container and dependency lock;
- randomization assignment and cluster unit;
- raw candidate-visible interaction trace digest;
- evaluator-only outcome artifact digest, kept outside candidate custody;
- run, policy and result receipt identifiers.

Use immutable revisions. The Beyond Recall project specifically documents differences between paper and packaged dataset revisions; pin the actual media release and its license card rather than following a moving branch.

## 10. Machine-readable protocol contracts

Three strict JSON Schema artifacts accompany this specification:

- [`mace_onpolicy_run_manifest_v1.schema.json`](../../data/schemas/mace_onpolicy_run_manifest_v1.schema.json): public non-secret run configuration and artifact commitments. Assignment seeds are committed, not disclosed.
- [`mace_onpolicy_candidate_event_v1.schema.json`](../../data/schemas/mace_onpolicy_candidate_event_v1.schema.json): candidate-visible prefix events only, with an explicit visibility marker, source/rights profile references, payload digest and predecessor digest. Unknown fields are rejected.
- [`mace_onpolicy_candidate_action_v1.schema.json`](../../data/schemas/mace_onpolicy_candidate_action_v1.schema.json): candidate-emitted responses, user questions, memory mutations, tool requests and abstentions as explicit action records. The action type determines required fields, and unknown fields are rejected.
- [`mace_onpolicy_evaluator_outcome_v1.schema.json`](../../data/schemas/mace_onpolicy_evaluator_outcome_v1.schema.json): sealed evaluator-only latent-state, utility, burden, memory-quality, privacy/authority and resource outcomes. This record must never be serialized into the candidate request stream.

The schemas define wire shape, not authenticity or authorization. The runtime must separately verify digests, source/authority receipts, event sequence, append-only chain continuity, policy assignment and custody. A schema parse is not schema validation, runtime conformance or isolation qualification.

## 11. Execution sequence

### P0 — specification and fixtures

- Freeze the event vocabulary and episode manifest.
- Implement deterministic synthetic profiles, disclosure rules, explicit corrections, refusals and irrelevant-disclosure traps.
- Author hidden-state and future-state mutation negative controls.
- Write the causal estimand and cluster-analysis plan before held-out evaluation.

### P1 — official reproduction

- Reproduce AMemGym under its published protocol.
- Record its exact data/model/configuration and resource use.
- Preserve the official results separately from MACE-specific controls.

### P2 — matched policy comparisons

- Run static replay and on-policy interaction with matched backbones.
- Compare all baselines listed above.
- Separate the total policy effect from exploratory mediation analysis.

### P3 — shift and lifecycle qualification

- Hold out profile and task families.
- Introduce model/prompt/tool/simulator version shifts.
- Exercise correction, supersession, privacy, refusal, deletion and restart/restore controls.

### P4 — independent qualification

- Move evaluator custody outside candidate runtime.
- Run default-deny candidate isolation and external resource accounting.
- Obtain independent reproduction of the policy assignment, score and result receipts.
- Only then publish confirmatory claims.

## 12. Related MACE programmes

- [#13388 — on-policy memory control under endogenous disclosure and feedback](https://github.com/Luminous-Dynamics/symthaea/issues/13388)
- [#13261 — competitive hybrid, causal-graph and state-first baselines](https://github.com/Luminous-Dynamics/symthaea/issues/13261)
- [#13363 — model-conditioned memory dose under model-family shift](https://github.com/Luminous-Dynamics/symthaea/issues/13363)
- [#13284 — decision-sufficient memory and active evidence acquisition](https://github.com/Luminous-Dynamics/symthaea/issues/13284)
- [#9516 — causal experience transfer across dependent sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516)
- [#13319 — adversarial lifecycle, poisoning, repair and erasure](https://github.com/Luminous-Dynamics/symthaea/issues/13319)
- [#13287 — independent candidate/evaluator custody](https://github.com/Luminous-Dynamics/symthaea/issues/13287)

## 13. Claim boundary

- Static replay accuracy is not closed-loop policy value.
- More user disclosure is not inherently better.
- A simulated user is not a real human participant.
- An evaluator-only latent-state oracle is not a deployable baseline.
- A changed transcript is not automatically a retrieval failure.
- A candidate's reported support is not proof of causal influence.
- AMemGym reproduction is not proof of general real-world personalization.
- No SOTA, privacy, safety or causal claim is valid without matched protocols, independent evaluation and the hard gates above.


## 14. Executable analysis increment

The research branch now includes a Rust analysis primitive at
[`crates/domains/symthaea-psych-bench/src/benchmarks/memory_agent/on_policy.rs`](../../crates/domains/symthaea-psych-bench/src/benchmarks/memory_agent/on_policy.rs).
It adds:

- deterministic keyed pseudorandom allocation, balanced within profile strata;
- one fixed policy assignment per profile cluster;
- validation that terminal outcomes exist for every randomized profile and match the evaluator's frozen task list;
- validation against a frozen expected task schedule, including exact task identities for every randomized profile;
- profile-mean aggregation before treatment/control comparison, so repeated episodes do not inflate the independent sample count;
- a clearly labelled normal-approximation 95% interval and a 30-independent-profiles-per-arm confirmatory floor;
- separate privacy and unauthorized-action hard gates that cannot be traded against utility.

The primary estimator deliberately has no post-assignment disclosure, transcript-length or turn-count covariates. Those variables may be reported as outcomes or explored in a separately preregistered mediation analysis, but must not be used to adjust the primary total-effect estimate.

Seven unit tests cover allocation balance/repeatability, duplicate assignments, profile-cluster weighting, non-compensable privacy gates, arm mismatch, task-identity mismatch, missing outcomes on both arms, and invalid utility. **They are authored but not compiled or executed in this environment.** The Wald interval is an exploratory primitive, not a substitute for a preregistered stratified randomization test or independent confirmatory analysis.

See [MACE benchmark landscape and competitive qualification plan](MACE_BENCHMARK_LANDSCAPE_2026-11-23.md) for the current external benchmark crosswalk and prospective targets.
