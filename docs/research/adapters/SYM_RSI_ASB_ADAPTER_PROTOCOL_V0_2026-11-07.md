# Symthaea research campaign adapter: AS-Bench v0.1

**Status:** pinned compatibility target; implementation not yet run.  
**Owner:** #10921.  
**Composes:** #10877, #9230, #8721, #8286, #9332, #10756, #10717, #10630, #10497.

## Purpose

Autonomous Science Bench (AS-Bench) is an interactive, budgeted design-build-test-learn benchmark, not a static question-answer dataset. Its agent chooses batches from a permitted design space, receives measurements through a stateful lab service, updates a predictor, and submits the predictor for verification. A campaign passes only if both the discovery gate (quality of what was measured) and learning gate (ranking quality on designs never measured) pass.

This makes AS-Bench a practical substrate for prospective research taste. A research policy has to make decisions whose consequences are visible in the campaign ledger and independent held-out evaluation.

The adapter must remain read-only with respect to upstream task instructions, data, gates and verifier. This project is an integration, not a fork of the benchmark.

## Pinned compatibility target

The v0 source pin is:

- Repository: [Yibo-Wen/as-bench](https://github.com/Yibo-Wen/as-bench)
- Commit: `b82ee550344e6055197b5eef3db47a44082cae4d`
- Task schema: 1.4
- Lab API: v1
- Pin and file identities: `SYM_RSI_ASB_COMPATIBILITY_MANIFEST_V0_2026-11-07.json`

A pin is a reproducibility anchor. It does not imply this is the latest upstream revision. Upgrades require a new manifest version and explicit compatibility review. Source task and data licenses and citations must be preserved.

## First two campaigns

| Campaign | Backend | Orderable pool | Budget | Discovery gate | Held-out learning gate |
|---|---|---:|---:|---|---|
| nuclease-active-learning | replay | 20,000 | 4 × 96 = 384 | ≥190 hits and ≥175 hit families | NDCG@100 ≥0.42 on 5,000 unmeasured designs |
| propylene-active-learning | twin | 736 | 3 × 20 = 60 | ≥7 measured designs above 2.0 mA cm⁻² | NDCG@20 ≥0.52 on 244 unmeasured designs |

The upstream reference solutions report 278 hits / 215 families / 0.488 NDCG@100 on the shipped nuclease split, and 8 hits / 0.626 NDCG@20 on the shipped propylene split. These figures belong to the upstream reference solutions. They are not Symthaea results, new SOTA claims or guaranteed targets across alternate splits.

The twin task returns exact published measurements for 96 of 980 source-study designs and surrogate-model outputs for the other 884. It is noiseless and returns means. Its output must never be called live laboratory measurement. Replay measurements and twin/surrogate outputs remain distinct evidence classes.

## Lab API mapping

| AS-Bench Lab API v1 | Adapter meaning | Required handling |
|---|---|---|
| `GET /v1/campaign` | Public campaign card | Bind the task identity, objective, design ID field and permitted design-space type |
| `GET /v1/status` | Campaign progress and resource status | Reconcile stage, completed measurements, spent budget and outstanding jobs |
| `GET /v1/catalog?stage=N` | Current eligible designs | Treat this as the agent-visible candidate pool; never join hidden truth into it |
| `POST /v1/experiments` | Request next batch | Preserve upstream stage order, exact batch size, eligibility, uniqueness and cost limits |
| `GET /v1/experiments/{job_id}` | Job status/result | Treat queued/running as pending; only completed jobs contribute measurements; handle failed jobs without claiming a completed stage |
| `POST /v1/deliverables` | Submit final model/report | Preserve upstream type, schema, size and required-stage checks |
| lab-owned `/state/ledger.json` | Campaign execution record | Bind the existing canonical receipt/evidence pipeline to the lab-provided ledger; do not trust agent-container copies |

The lab owns stage order, accepted batch size, eligibility, uniqueness and budget enforcement. Rejected requests must not be treated as consuming a stage. Re-submitting the latest accepted batch is idempotent. A failed asynchronous job is not a completed experiment. The offline Rust reference contract in `tools/sym-rsi-asb-adapter-reference/src/job_lifecycle.rs` rejects pending jobs with result rows, completed jobs without a full result batch, failed jobs with result rows, and repeated design IDs. It advances a stage only for a valid `completed` job with one result row per submitted design.

The verifier is a separate no-network boundary. Its hidden truth and evaluation data must never be exposed to the policy process, Coder, retrieval tools or direction board.

## Research-taste experiment

Use the upstream tasks to compare research policies, not to create another evaluator:

1. Random valid selection.
2. Campaign-provided static/expert prior.
3. A campaign-provided conventional adaptive strategy, where the task supports it.
4. Fixed Symthaea researcher with a static evidence packet and fixed Coder.
5. Fixed Symthaea researcher with active, budgeted literature/tool exploration and the same Coder.
6. Cost-matched random retrieval.
7. No optional retrieval/tool control where the campaign permits a valid comparison.

Use matched starting observations, seeds, model/backend, total resources, task identity, and sealed evaluator version. Freeze capability tiers before treatment. Do not assume retrieval benefits every capability tier.

Before each batch, record a prospective plan: candidate mechanisms, selected designs, predicted result distributions, what each outcome would change, gross EVSI, direct cost, opportunity cost, downside model/CVaR constraint, stop/continue/replicate/pivot rule. Use the separate-cost reference in #10879 as calculation logic only; it is not an authorization service.

## Score and claim separation

Report the following independently:

- discovery metric(s), raw values, thresholds and gate result;
- held-out learning metric, raw value, threshold and gate result;
- joint campaign pass;
- prospective EVSI forecast and realized decision regret;
- cost by model inference, Coder, experiment budget, compute/energy, evaluator, retries and human review;
- invalid/failed/not-run denominators and pending-job timeouts;
- performance by campaign, seed and family.

Do not allow a high discovery score to compensate for failed held-out learning, or vice versa. Do not normalize away worst-family regressions. Do not call replay/twin conformance original science. R4 remains governed by #10497 and needs same-start, inclusive-budget, repeated full-loop evidence plus fresh-family transfer.

## Progression

1. Validate the manifest offline.
2. Add a Rust Lab API compatibility client using frozen fixtures first.
3. Exercise idempotency, stage order, duplicate rejection, budgets, and pending/failed/completed job states.
4. Run the replay campaign end to end with public task artifacts and the official verifier in its isolated boundary.
5. Add the twin campaign, explicitly testing the distinction between published measurements and surrogate outputs.
6. Only after API/ledger and oracle boundaries pass, run matched-budget agent experiments.
7. Consider live-lab integration separately, only with explicit scientific, biosafety, authorization, and human governance; it is out of scope here.

## Current implementation boundary

The first artifacts are a pinned JSON manifest, its JSON Schema, an offline Rust manifest checker, and a typed Lab API job-lifecycle contract with negative controls. It does not yet make AS-Bench API calls or execute either campaign. The manifest remains `PINNED_UPSTREAM_COMPATIBILITY_TARGET_NOT_YET_RUN` until executable conformance evidence exists.
