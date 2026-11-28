# GameBench Transition Evidence v0.1

**Date:** 2026-11-28  
**Status:** research protocol and adapter draft; not yet compiled, executed, or qualified  
**Parent:** GameBench #14137  
**Causal-twin design:** #14215 / PR #14216  
**Implementation:** #14259 / draft PR to follow  
**Read-only decision-value bridge:** PR #14193

## 1. Research objective

Evaluate whether an agent can discover only the causal distinctions needed for a good decision, choose and execute state-changing experiments under explicit costs and safety constraints, update its internal model from the actual consequences, and transfer that knowledge to held-out worlds.

The benchmark should not reward a verbose explanation or a complete hidden-mechanism reconstruction when neither is necessary for the terminal decision. It must also prevent an agent from declaring a unique mechanism when multiple mechanisms remain compatible with its evidence.

## 2. Paradigm shifts

### 2.1 Mechanism identification → decision-sufficient causal understanding

For the candidate-compatible world set (M), compare the optimal action sets, not merely the hidden labels.

- **Action-equivalent:** the intersection of optimal action sets is non-empty. The agent may act correctly without distinguishing every mechanism.
- **Decision-sensitive:** no common optimal action remains. An experiment is valuable only insofar as it resolves a distinction that changes the action.
- **Unresolved:** an out-of-family world remains possible, the evidence support is incomplete, or the decision contract is inadequate. The agent must preserve uncertainty or abstain.

This is a different estimand from graph-recovery accuracy. Graph recovery is still useful on tasks where a graph is the target, but it should not be used as a proxy for decision correctness.

### 2.2 Passive observation → causal intervention with transition receipts

A world-mutating experiment is not a read-only sensor query. The system must bind the intervention to the actual transition:

- episode and invocation identity;
- intervention and action identity;
- pre-state and post-state observation digests;
- issue, application, and observation times;
- public result vocabulary;
- side-effect class;
- actual resource cost and declared ceiling;
- environment-event and transition-trace commitments.

Reject stale, duplicated, out-of-order, early, late, mismatched, unknown-outcome, over-budget, or wrong-side-effect events. Digest presence is not digest authenticity: the executor/evaluator must verify source authenticity in a separately permissioned boundary.

### 2.3 Information gain → decision value after intervention cost

Entropy reduction can be high while terminal loss remains unchanged. Report separately:
- mechanism-set coverage and false-unique rate;
- outcome calibration;
- action-equivalence and selected-action regret;
- intervention count, irreversible side effects, and consumed resources;
- terminal success/regret;
- transfer and model revision.

Use shared Bayesian EVSI only where priors and likelihoods are defensible. Use the existing prior-free robust evaluator/oracle when only support sets are defensible. Never relabel entropy, graph distance, or a mechanism score as decision value.

### 2.4 Single-shot answer → stateful causal trajectory

Test sequences of interventions, delayed observations, partial observability, memory retention, contradictory evidence, and held-out compositions. A candidate must update when an intervention disproves its hypothesis; the benchmark should separately score stale-memory persistence and unnecessary re-investigation.

### 2.5 Self-reported confidence → independently verified evidence

Candidate forecasts are pre-registered before interventions. Environment transitions are independently replayed. Evaluator truth and answer keys stay outside the candidate process. Rust type visibility is not a sandbox. A benchmark run needs canonical accounting and provenance through the existing #10314/#9735 authorities.

## 3. Current implementation slice

Standalone crate: `research/symthaea-gamebench-transition-bridge-v0-1/`.

Current scope:
- validates matched twins (same starting observation digest, nuisance digest, public action vocabulary);
- validates mechanism alternatives and terminal loss tables;
- computes action-equivalence by intersecting each world's optimal-action set;
- scores false-unique mechanism claims and retained-mechanism coverage;
- reports candidate action regret per compatible world;
- treats out-of-family support as unresolved;
- validates environment-produced world-mutation events and issues a domain transition receipt.

The implementation is intentionally not a planner or a generic receipt system. It does not replace the read-only PR #14193 or implement a separate EVSI/minimax/sequential oracle. It does not authenticate event digests or prove environment execution. The current source has 19 authored Rust tests, not yet run. No candidate campaign or performance result is claimed.

## 4. Challenge design

Retain the five causal-twin families from #14215:
1. switch-to-gate topology;
2. key/door/plate persistence;
3. resource consume/preserve/transform semantics;
4. action-triggered vs time-triggered hazards;
5. local reveal vs persistent memory.

For each family, include:
- at least two distinct causal mechanisms with identical initial candidate observations;
- nuisance-only surface changes;
- an action-equivalent pair whose mechanisms differ but best action does not;
- a decision-sensitive pair whose best action differs;
- a high-information but decision-irrelevant experiment;
- a decision-relevant experiment;
- a no-permitted-experiment observational-equivalence control;
- an out-of-family control;
- held-out two-mechanism compositions;
- positive, negative, replay, delayed-result, side-effect, and budget controls.

The five families × two variants × three roles (identify, decide, transfer) yield 30 initial task templates as a design target—not an existing task count.

## 5. Benchmark arms

On identical world instances, public observations, interaction budgets, and held-out split:
- no-probe reactive baseline;
- random legal intervention, resource matched;
- entropy-greedy;
- cost-aware entropy-greedy;
- shared Bayesian EVSI selector where probability assumptions are justified;
- prior-free robust selection/oracle from the existing decision evaluator / #13567 (evaluator-only upper reference);
- actual Symthaea FEP, spatial-HDC, HDC-LTC arms;
- memory on/off, active sensing on/off, and planning on/off ablations;
- external game/causal baselines under their published protocols where adapters are feasible.

Do not compare a one-step selector with a finite-horizon oracle as if they had equal information. The oracle is an upper-reference diagnostic and must be labelled as such.

## 6. Metrics: separate scorecard, not one aggregate

| Dimension | Primary measures |
|---|---|
| Causal support | mechanism-set coverage, false-unique rate, out-of-family precision/recall |
| Decision sufficiency | action-equivalent correctness, decision-sensitive regret, abstention quality |
| Experiment quality | gross/net EVSI where valid, guaranteed robust loss reduction where valid, unnecessary probe rate |
| Intervention integrity | unsafe/precondition failures, replay acceptance, timing violations, budget overrun, receipt completeness |
| Learning | posterior/belief revision quality, contradiction handling, memory retention/interference |
| Transfer | held-out surface-form accuracy, unseen mechanism compositions, new observation channels |
| Systems | wall-clock, latency, CPU/memory, measured energy at declared boundary, interactions and resource vector |

No weighted average should hide a safety or validity failure. Publish denominators, paired confidence intervals, worst-family and worst-regime outcomes, abstention/refusal rates, and all excluded cases.

## 7. Proposed targets (hypotheses, not achieved results)

- 0 false-unique mechanism claims on exact observational-equivalence controls.
- 0 accepted stale/replayed/invalid transitions in the frozen negative-control suite.
- 0 accepted intervention precondition or hard safety violations.
- 100% deterministic replay agreement on frozen deterministic fixtures.
- At least 90% correct terminal action on action-equivalent cases, without unnecessary interventions.
- At least 80% terminal success on held-out compositions of known mechanisms.
- At matched success, at least 20% fewer interactions than the strongest preregistered non-oracle baseline; alternatively, at matched budget, a statistically lower terminal regret.
- Calibration intervals and power analysis before setting confirmatory probability thresholds.

A smoke test is not a qualification result. Use paired seeds, preregistered training/development/holdout splits, sealed challenge keys, independent evaluation, and a power plan. Run multiple model seeds where stochastic candidate policies are compared.

## 8. Industry landscape and comparison strategy

The category is already active. The new work should be positioned as a complementary bridge, not a claim to be the first causal-game benchmark.

### CausalGame (ICML 2026)

CausalGame has 14 interactive scenarios across selection bias, hidden confounding, and noisy measurements. The project reports 29 frontier LLMs; none clears the scenario win threshold. Its strongest agentic result is 68.0% average survival against a 75% threshold, with analytical optima around 78–85%. Its rubric analysis reports 68.4% of agentic sessions with no causal engagement. This makes causal-process evidence—not just survival or end score—an important benchmark axis.

### Active Causal Discovery Benchmark (ACDB)

ACDB evaluates active graph recovery through budgeted hard interventions in linear-Gaussian SCMs. Its reported best non-oracle PC + greedy active-orientation method reaches 42.7% directed F1 (SHD 4.79), while raw Claude Sonnet 4.6 and GPT-5.4 active agents reach 31.7% and 22.9% directed F1. It also reports a 23.6% structure-blind random-DAG directed-F1 floor on its dense v0 ladder. The lesson: use strong algorithmic baselines and calibrated random floors; causal graph F1 is not interchangeable with decision regret.

### ACED-Bench

ACED-Bench has 180 ACED-Core tasks (60 set-valued ancestry/descendancy tasks and 120 constrained causal-decision tasks) plus 150 ACED-Qual questions. It explicitly measures intervention choice, evidence retention, answer commitment, and stopping using hidden probabilistic graphical models. This overlaps with our evidence lifecycle; interoperability and head-to-head comparison should be preferred over duplicating it.

### Heuresis

Heuresis uses anonymous simulated datasets across six mechanism families, separating gains from choosing a better causal-discovery algorithm from gains caused by actual data investigation. It reports GPT-6.1 Sol exceeding the best chosen algorithm on all six families, by 14 adjacency-F1 points on its hardest family; Claude Opus 5.5 gains 7.4 points there. Traceable edge changes and data-grounded improvement are strong design precedents.

### Game-playing benchmarks

BALROG's official experiment viewer includes GPT-6-Astra-Max at 68.26 ± 1.97 overall progress (255 completed episodes; 2026-09-18) and Claude-Opus-5-Max at 63.40 ± 1.85 (254 completed; 2026-09-20). Progress is not success. GameWorld covers 34 browser games and 170 tasks with state-verifiable outcomes; its standard generalist podium reports Gemini-3-Flash-Preview at 41.9% progress /21.2% success. Its novice-human progress baseline is 64.1%. GameWorld progress/success and BALROG progress are distinct metrics and protocols.

Do not compare the internal causal-twin score directly to BALROG progress, GameWorld success, ACDB graph F1, CausalGame survival, or Heuresis adjacency F1. Add external tracks with exact benchmark versions, actions, prompts, budgets, tool policies, seeds, graders, and scoring.

### Recommended competitive path

1. **First**, establish an exact reproducible causal-twin core: action-equivalence, decision-sensitive pairs, out-of-family abstention, transition receipts, and negative controls.
2. **Second**, crosswalk the same agent against CausalGame and ACED-Bench for causal experimentation/evidence lifecycle; ACDB and Heuresis for graph/data-driven discovery; BALROG/GameWorld for general interactive capability.
3. **Third**, publish paired component ablations for FEP, spatial HDC, HDC-LTC, memory, and active sensing under equal budgets.
4. **Fourth**, make superiority claims only on preregistered, independently graded comparisons with confidence intervals and a public replication package.

## 9. Ownership and integration

- #14137 / PR #14138: core GameBench tasks and candidate adapters.
- #14215 / PR #14216: matched causal-twin protocol and task families.
- PR #14193 / #14187: read-only observation bridge and shared evaluator adapter; world-mutating intervention is explicitly out of scope.
- #14161: joint experiment/intervention, robust test, stopping, and terminal decision co-design.
- #13567 / PR #13571: finite-horizon prior-free policy oracle.
- `research/decision-predictability-benchmark`: finite-world Bayesian EVSI and prior-free robust evaluator.
- PR #10314: canonical benchmark-run receipt schema and anti-score-laundering validation.
- #9332/#10756: sealed benchmark/evaluator research.

## 10. References

- CausalGame (ICML 2026): https://causalgame.github.io/ ; https://proceedings.mlr.press/v306/chen26l.html
- ACDB: https://arxiv.org/abs/2609.31675
- ACED-Bench: https://arxiv.org/abs/2610.14381
- Heuresis / Benchmarking and Enhancing Agentic Causal Discovery: https://arxiv.org/abs/2610.22846
- BALROG official experiment viewer: https://github.com/balrog-ai/experiments/blob/main/viewer.html
- BALROG leaderboard: https://balrogai.com/
- GameWorld: https://gameworld-project.github.io/

## 11. Claim boundary

This is a research protocol and draft adapter. It is not a completed or qualified benchmark, a causal-discovery result, a physical safety case, an externally reproduced leaderboard result, or SOTA. Exact-head Rust formatting, compilation, tests, strict Clippy, independent review, evaluator isolation, and a benchmark campaign remain pending.
