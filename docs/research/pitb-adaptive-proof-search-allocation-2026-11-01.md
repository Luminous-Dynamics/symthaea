# PITB research track: adaptive proof-search allocation (2026-11-01)

**Status:** deterministic controller, exact-byte calibration receipt verifier, and synthetic contract added; no real calibration dataset, proof-search allocation experiment, or Lean result has been executed. The synthetic calibration fixture is not empirical evidence.

## Research opportunity

Current Lean proof agents often improve a candidate proof by spending more attempts, decomposing a target, running tactics, retrieving premises, or parallelizing branches. The research problem is not just to find a successful tactic. It is to decide **which next computation has the highest conservative horizon-bounded success probability per shadow-priced resource, given the remaining budget and the exact current proof state**.

This is a control problem over partially observed, expensive actions. A failed tactic, failed elaboration, repeated typeclass search, stale premise route, or unproductive branch is not merely a negative outcome. It changes the posterior about which intervention is worth paying for next.

Two relevant external results make this an especially timely track:

- Rögnvaldsson et al., *Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean* (AI4Math@ICML 2026), introduce a data plane for decomposition/formalization/proof attempts and a control plane that estimates the likelihood and cost of another attempt. On a subset of PutnamBench they report a 25.8% average cost decrease versus a fixed-step baseline while preserving performance. Source: https://arxiv.org/abs/2606.04883 and https://eth-sri.github.io/publications/rognvaldsson2026optimizingcostqualitytradeoffagentic.
- *Smarter Spending: Dynamic Compute Allocation for Lean Theorem Provers* reports a 7.9% improvement in area under the cost-quality curve and a 28.9% reduction in budget needed to reach a given accuracy versus a fixed-step baseline, in simulation over 64 random seeds on an 85-problem PutnamBench subset. The authors also report a substantial remaining gap to oracle routing and identify richer trajectory features and a broader action space as future work. Source: https://matharena.ai/optimizing_agents/.

These are useful baselines, not directly comparable PITB results. They use PutnamBench subsets, different agent/model/harness configurations, and different cost accounting. PITB must reproduce a named baseline under the same task set and resource caps before claiming a head-to-head.

## PITB paradigm shift: marginal verified information per unit of compute

The controller should optimize the **incremental probability of obtaining an independently verified proof**, not the raw number of attempts, tokens, tool calls, or generated subgoals.

At each decision point it receives:
- exact target statement and proof-state identities;
- current remaining tokens, model calls, tool calls, wall time, Lean CPU and monetary budget;
- a bounded candidate action set;
- a prediction for each action's incremental verified-success probability and uncertainty;
- hashes identifying the prediction evidence and calibration dataset.

The v0 controller estimates the action-conditioned probability of verified proof success under a fixed post-action continuation policy and horizon, then computes a conservative lower confidence bound:

\[
LCB(a\mid s,\pi_c,H)=\max(0,\widehat{p}(\text{verified proof by }H\mid s,a,\pi_c)-1.96\,SE(a)).
\]

This is **not** a causal treatment-effect estimate or an incremental lift over a no-action counterfactual. It is a conditional success estimate for the chosen action followed by the registered continuation policy. The calibration receipt must use the same horizon and continuation-policy digest as the decision.

It then ranks feasible actions by:

\[
U(a)=\frac{LCB(a)}
{\sum_r w_r\,c_r(a)/b_r},
\]

where \(c_r(a)\) is the action's expected consumption of resource \(r\), \(b_r\) is the remaining resource budget for that dimension, and preregistered nonnegative shadow weights \(w_r\) sum to one. The action must fit **every** remaining resource cap. If no action has a positive LCB above the frozen threshold, the controller stops rather than spend compute on a prediction with inadequate conservative value.

The ranking is deterministic; ties are broken by action ID. It records the selected action-conditioned success probability, input digest, canonical policy digest, exact calibration-receipt digest, calibration dataset manifest, evaluation horizon, continuation-policy digest and excluded action IDs. It does not claim that the selected action will succeed. Lean plus independent replay remains the success authority.

This is a deliberately simple and auditable v0 policy, not the final algorithm. A future v1 should learn shadow prices from the observed constrained Pareto frontier, use calibrated survival/hazard models over attempt trajectories, and model the value of information from actions that fail but improve subsequent routing.

## Action space

PITB should route across a richer intervention set than continue-versus-stop:

| Action | When it may be worth paying for it |
|---|---|
| Continue current target | The current tactic/lemma route still has positive conditional probability of reaching a verified proof within the registered horizon |
| Retrieve context | Missing or ambiguous premises/instances likely explain failure |
| Change decomposition | Repeated local failure suggests the current proof breakdown is poor |
| Typed repair | A known blueprint node has a constrained, target-preserving correction |
| Tactic probe | A cheap, narrow Lean experiment can distinguish candidate proof paths |
| Spawn snapshot branch | More independent search is justified and a qualified R5 snapshot receipt is available |
| Semantic check | A proof or formalized statement needs a costly downstream/specification audit |
| Restart search | Existing trajectory has low marginal value but a different route remains promising |

R5 is an execution optimization, not a proof-quality signal. A snapshot branch receives no higher success prior simply because it is cheap to restore. Snapshot availability changes the resource estimate; only the verified outcome changes success probability.

## Causal benchmark arms

A first benchmark must hold the base model, prompts, target statements, Lean toolchain, proof checker, semantic gate and overall resource envelope fixed:

| Arm | Policy | Purpose |
|---|---|---|
| A0 | Fixed-step continuation / fixed number of attempts | Reproduce the standard baseline |
| A1 | Fixed success-threshold stop rule | Test the value of early stopping alone |
| A2 | Published-style cost router: continue current target or restart/decompose | Reproduce the narrow control-plane baseline |
| A3 | PITB multi-action LCB router with fixed shadow prices | Isolate action-space expansion and uncertainty-aware ranking |
| A4 | A3 plus qualified R5 live snapshot branch costs | Isolate post-elaboration snapshot reuse |
| A5 | A4 plus retrieval, support antichains, typed repair and transferable proof insights | Test the complete PITB system |

A2 must be a faithful reproduction of the published route policy where possible; a PITB reimplementation with altered action semantics is a new baseline. Record every adaptation.

Use paired target × model-seed assignments and identical total budget caps. Account for all parallel branch work, model calls, tokens, Lean CPU, memory, wall time and dollars. Do not let the adaptive arm obtain more total compute merely because its work is parallel. An oracle router that uses held-out outcomes is an upper-bound diagnostic only, never a deployable baseline.

## Benchmark programme

**Primary practical cohort:** SorryDB_2601, all 1,000 frozen obligations, in the original repositories/toolchains. This answers whether routing helps on real proof holes. Keep the reported SorryDB leaderboard separate: Aleph Prover 57.2% Pass@1, Gemini Flash 3 agentic 30.3% Pass@1, and a 63.2% union across currently listed approaches are not a matched PITB result.

**Secondary controlled cohort:** the 85-problem PutnamBench subset used by published cost-aware routing work, where task identity and eligibility can be reproduced. Report separately from SorryDB.

**Transfer test:** the frozen research-math family holdout tests whether the controller's routing policy generalizes, not only whether the base prover solves more tasks.

**Systems ablation:** on a fixed task subset, compare R4 cold branch reconstruction against R5 live post-elaboration snapshots. Report capture cost, restore latency, Lean CPU, memory, branch work and proof yield.

## Metrics and proposed qualification targets

All targets below are research goals, not measured results:

| Endpoint | Target |
|---|---:|
| Verified-success rate at equal total budget | Non-inferior to A0; target +5 percentage points |
| Cost per verified proof at matched success | ≥30% lower than A0 and ≥5 percentage points better than the published 25.8% cost reduction |
| Cost-quality AUC | ≥10% improvement over A0 and ≥2.1 points over the published 7.9% gain |
| Budget to matched accuracy | ≥30% lower than A0; surpass the 28.9% published simulation result |
| Calibration | ECE ≤5% on a frozen validation calibration set; publish Brier score and reliability diagrams |
| Selective risk | Zero false claims of success; 100% credited proofs pass independent Lean replay |
| Budget accounting | Zero cap violations; all action and branch costs reconcile |
| Generalization | Same-direction cost-quality gain on SorryDB and a temporally later cohort |
| Systems | R5 reduces branch restore/start p95 without increasing stale-cache observations or proof-state identity drift |

The five-percentage-point improvement over 25.8% is a stretch target. The first scientific success may be a replicated cost reduction at unchanged success rate rather than a new absolute proof-rate record.

## Required controls and anti-gaming rules

1. Keep an exploration/validation set distinct from the test cohort. The action predictor, calibration manifest and shadow weights are frozen before test evaluation.
2. Count non-attempts, timeouts, unavailable tools and rejected actions in their assigned denominators; do not condition results only on tasks that reached an agent.
3. The prediction of marginal success is not the success outcome. Only kernel checking plus applicable semantic/successor verification count as success.
4. Candidate proofs and subgoal claims are untrusted. Promotion binds exact bytes, exact target statement, checker receipt and independent replay.
5. Resource consumption is measured, not inferred from token count. Include wall time, Lean CPU, peak memory, concurrent workers, total branch work and cost.
6. Show task-level cost-quality curves, uncertainty intervals, reliability diagrams, per-family/per-project strata and the full Pareto frontier. Do not report only a scalar utility.
7. For public benchmarks, preserve frozen snapshots and record exposure/tuning history. A fresh rolling cohort is a separate result, not a silent update to the fixed snapshot.

## Current implementation

The current PR adds:
- `adaptive-allocation/proof-search-allocation-decision-v0.schema.json`;
- `adaptive-allocation/proof-search-calibration-receipt-v0.schema.json`;
- synthetic no-run decision and calibration fixtures;
- `allocate_proof_search_budget.rs`, a deterministic LCB/shadow-price decision engine;
- `proof_search_calibration.rs`, an exact-byte receipt verifier that recomputes per-action ten-bin ECE and Brier sufficient statistics;
- tests for uncertainty-sensitive ranking, single-resource infeasibility, calibration mismatch, receipt substitution, model mismatch, duplicate actions, state identity, shadow-weight normalization and conservative stopping.

The controller requires the exact calibration receipt file in production selection. It checks the receipt SHA-256 and dataset-manifest digest, verifies that each proposed action kind has its own calibration group, matches predictor ID/version, evaluation-horizon ID and continuation-policy SHA, recomputes ECE/Brier, and verifies the canonical policy hash. It is a deterministic selector, not a trained router, proof checker, or measured SOTA result.

Important limitation: the calibration bins are **self-consistent and content-addressed**, not independently authenticated scientific observations. The verifier checks sufficient-statistic arithmetic and the source outcome-receipt manifest digest, but does not open every underlying Lean checker and independent replay receipt. The checked-in 1,600-observation calibration fixture is synthetic; its ECE=0 and Brier values are only contract-test arithmetic. No empirical calibration result is claimed.

## Next milestones

1. Compile the Rust controller and calibration module; run the synthetic negative controls in the isolated PITB crate workflow and inspect exact-head CI evidence.
2. Implement a trajectory importer that authenticates each underlying Lean checker and independent replay receipt before producing calibration rows.
3. Measure horizon-bounded action-conditional calibration on a frozen validation split with a single pinned continuation policy; publish reliability diagrams, Brier score, ECE by action class, interval coverage and uncertainty intervals.
4. Add repeated-task/context strata so estimates are calibrated conditionally on failure history, proof-state features and remaining budget—not merely globally by action type.
5. Reproduce A0 and A2 on a small, pinned PutnamBench smoke cohort, then compare A0–A5 under equal resource caps.
6. Scale to all SorryDB_2601 tasks only after environment reconstruction, proof outcome receipts and calibration lineage are valid.
7. Freeze policy, calibration and shadow weights before the external challenge/test run.
