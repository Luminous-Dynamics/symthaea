# MACE Non-Regressive Learning Admission Gate V1

**Snapshot:** 2026-11-26  
**Status:** Rust policy/reference module, schema and deterministic tests authored. Compilation, test execution, external benchmark runs and production deployment are not claimed.

## 1. Why learning needs a retention constraint

An agent that improves on unfamiliar or changed tasks can still damage capabilities it already had. ServeLearnBench reports this negative-transfer pattern: the benchmark's hidden-task gains do not guarantee retention on Fully Specified tasks. Its public leaderboard is a useful cost/performance view, but its rows use different methods and backbones and must not be treated as matched comparisons.

MACE should therefore treat learning as a constrained state transition:

`improve changed-task behavior subject to a stable-capability floor`

Do not combine stable and adaptation scores into one average where a gain on new tasks can compensate for a serious regression on established tasks.

## 2. Implemented policy contract

Module:

`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/non_regressive_learning.rs`

Schema:

`data/schemas/mace_non_regressive_learning_gate_v1.schema.json`

The gate compares four independently verified run snapshots:
1. Stable-stratum, before the update.
2. Stable-stratum, after the update.
3. Adaptation-stratum, before the update.
4. Adaptation-stratum, after the update.

Each snapshot pins:
- benchmark ID, release digest and split;
- evaluation-protocol digest;
- grader digest;
- exact sorted task-set digest and per-task scores;
- comparison-group digest, including matched backbone/runtime family;
- candidate-state digest;
- raw run-receipt digest.

The gate requires:
- identical task identities in each before/after pair;
- identical benchmark, release, split, evaluator protocol and grader;
- identical comparison group across all four runs;
- one candidate state across both before-runs and a different shared state across both after-runs;
- a distinct run identity for all four executions;
- independent verification of every run receipt.

This prevents model-family, prompt, scorer, dataset, task-population or environment drift from being passed off as learning improvement.

## 3. Scoring and decision rule

Each task score is represented in integer basis points from 0 to 10,000. The gate retains exact sums and denominators; it does not round before making the admission decision.

Let:

`Delta_stable = Sum(stable_after) - Sum(stable_before)`

`Delta_adaptation = Sum(adaptation_after) - Sum(adaptation_before)`

For stable task count (N_s), maximum allowed regression (R) basis points, adaptation task count (N_a), and minimum adaptation gain (G):

`Accept iff Delta_stable >= -R * N_s and Delta_adaptation >= G * N_a`

Default proposed policy:
- maximum stable regression: 100 basis points = 1.00 percentage point;
- minimum adaptation gain: 0 basis points = no measured regression on the adaptation stratum.

These are MACE policy defaults, not official ServeLearnBench rules or an achieved result. Applications may choose stricter domain-specific constraints.

### Dispositions

| Disposition | Meaning |
|---|---|
| AcceptedWithinRetentionBudget | Both the stable retention floor and adaptation gain constraint were satisfied using independently verified runs |
| RejectedStableRegression | Stable performance regressed beyond the configured budget |
| RejectedInsufficientAdaptationGain | Adaptation performance failed its minimum gain constraint |
| InconclusiveUnverifiedRun | One or more run receipts could not be independently verified |

The gate does not automatically deploy, roll back or authorize an update. It produces an admission assessment for a separate consumer-owned deployment authority. An inconclusive result is not a pass.

## 4. Independent verification boundary

The module accepts an injected `LearningRunVerifierV1`. The verifier must independently bind:
- exact benchmark release, split and task IDs;
- grader and evaluation protocol;
- candidate state/checkpoint;
- raw execution and scoring receipts;
- per-task outcomes and denominator.

A missing or unverifiable receipt returns `InconclusiveUnverifiedRun`. The assessment digest is an integrity commitment, not a signature or proof that the external verifier is honest. Production must pin verifier identity, key material, revocation, trust policy and replay protection.

Skipped, invalid, timed-out or unavailable tasks must be represented according to the official scorer. If a task has no official score, the run must not silently drop it or shrink the denominator; the gate's exact task-set comparison should reject the mismatched population.

## 5. Deterministic test coverage

Nine authored tests cover:
1. acceptance at exactly the default 1 percentage-point stable regression boundary;
2. rejection beyond the stable regression budget even when adaptation improves;
3. rejection when adaptation does not improve under a zero minimum-gain policy;
4. missing verifier produces an inconclusive result;
5. changed task populations cannot be compared as paired learning;
6. protocol/grader drift is rejected;
7. duplicate run IDs and out-of-range task scores are rejected;\n8. one verification receipt cannot be reused to attest all four runs;\n9. one raw run receipt cannot be replayed across run IDs.

These tests are authored but not compiled or executed in this environment.

## 6. ServeLearnBench comparison protocol

Official project: https://infini-ai-lab.github.io/ServeLearnBench/  
Leaderboard: https://infini-ai-lab.github.io/ServeLearnBench/leaderboard.html  
Repository: https://github.com/Infini-AI-Lab/ServeLearnBench

MACE should reproduce an exact baseline under the official model, scenario, serving windows, hidden test tasks, judge, tool budget and cost accounting before comparing improvements. Then run the same stable and changed task populations before and after each learning window, preserving the paired task identity and all run receipts.

Report separately:
- Hidden/changed-stratum reward and learning curve;
- Fully Specified/stable-stratum reward and regression;
- AULC10 or the official learning-speed metric;
- exploration and knowledge-acquisition curves;
- API/token cost, tool calls and latency;
- rollback events, recovery time and post-rollback scores;
- task denominators and verifier coverage.

Compare systems on the cost/performance Pareto frontier. Different backbones and methods are not matched baselines. The MACE 1.00 pp stable regression budget is a proposed additional safety constraint, not an official benchmark metric.

## 7. Further paradigm shift: versioned skills with non-regression envelopes

The longer-term direction is to stop treating every memory update as a global change to one shared policy. MACE can maintain versioned, scoped skill/evidence bundles with:
- explicit applicability conditions;
- source and derivation provenance;
- confidence and validity envelopes;
- stable-task regression probes;
- independent activation and rollback receipts.

A new skill version should be promoted only when it improves its target stratum while preserving the already-qualified capabilities it is not intended to replace. Incompatible or uncertain skills should remain isolated, scoped or inactive rather than being merged into one potentially destructive global memory.

This is a research direction. The current module supplies only a matched-run admission rule, not skill routing or automatic rollback.

## 8. Production boundary

Production still requires:
1. trusted signed run receipts and pinned independent verifier roots;
2. reproducible paired task execution and protected hidden splits;
3. a durable candidate-version registry;
4. atomic promotion/rollback controlled by a separate authority;
5. post-deployment canary and regression monitoring;
6. explicit handling of non-comparable runs and model/provider changes;
7. a reconciliation workflow for unknown deployment outcomes.

The reference gate does not train, deploy or roll back a model.


## 9. Assessment evidence capsule

The final gate assessment carries the four exact run IDs, raw run-receipt digests, task-set digests and candidate-state digests, in stable-before/stable-after/adaptation-before/adaptation-after order. It also carries all four independent verifier identities and verification-receipt digests for any non-inconclusive result. The assessment digest commits to these fields and the exact score deltas and denominators.

An inconclusive assessment still records the four run identities and raw receipt digests, but contains no verifier attestations and cannot admit the update. This keeps an unavailable verifier distinguishable from an accepted run while retaining enough identity to reconstruct the blocked comparison.
