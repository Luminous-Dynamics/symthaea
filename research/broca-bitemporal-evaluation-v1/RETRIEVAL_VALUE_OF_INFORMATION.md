# Broca retrieval value-of-information gate

**Policy version:** BROCA-RETRIEVAL-VOI-V1  
**Status:** deterministic decision module; Rust tests authored, not compiled/executed here.  
**Purpose:** choose whether another retrieval call is worth its expected benefit, risk and cost.

## The paradigm shift: retrieve for marginal proof value, not for available budget

Traditional RAG often retrieves a fixed top-k for every request. Agentic RAG may retrieve until a call limit or a language model's confidence threshold is reached. Neither policy alone answers the operational question: will the next retrieval call change the justified decision enough to justify its cost?

Broca's next step is a conservative value-of-information gate over the outstanding claim ledger. It evaluates:

**Net conservative utility = correctness-gain lower bound + lifecycle-risk-reduction lower bound − retrieval-harm upper bound − retrieval cost.**

All terms use integer micro-utility units defined by a pinned utility-schema digest. Retrieval continues only when:

- unresolved required claims remain;
- at least one retrieval call remains;
- the expected retrieval cost fits the remaining budget; and
- the conservative net utility is strictly above the configured minimum margin.

Equality stops. An early stop preserves unused call budget and is distinguished from budget exhaustion.

## Why claim coverage and utility must be combined

HALT (Roh & Han, Findings of EMNLP 2026) frames stopping as evidence coverage, not generator confidence: stop when cumulative evidence supports each expected hop claim, reducing redundant searches while largely preserving exact match. This is the right deterministic baseline for Broca because the proof/claim ledger already tracks what remains uncovered.

However, coverage alone is not always enough. If a required claim remains unresolved but no plausible source is left, or the next retrieval is expensive and unlikely to help, the system should stop and apply the lifecycle action policy. Conversely, a high-confidence reader must not stop if a required proof premise is still missing.

Active RAG evaluation work (Qian et al., 2026) formulates retrieval as budgeted utility estimation: compare correctness with and without retrieval, then evaluate ranking quality, threshold transfer to held-out data, realized evidence-use budgets, retrieval harm and total cost. Its experiments show that nominal usage targets may differ materially from held-out realized usage; on HotpotQA, common triggers exceeded a nominal 50% evidence-use target in 40–100% of splits when any overshoot was counted.

CONFRAG (Huang et al., EMNLP 2026 Industry Track) provides an industry reference for confidence-guided retrieval: its reported system reduced external retrieval calls by 5–39% and P90 tail latency by up to 800 ms while matching its always-on RAG answer quality. This is a useful efficiency comparator, but Broca should use confidence only as one estimator feature; lifecycle validity and proof coverage remain independent gates.

AgentAbstain (NeurIPS 2026 Evaluations & Datasets Track) provides a paired-action benchmark with 263 tasks, 42 executable environments and eight abstention scenarios. Its best reported paired accuracy across 17 models is 59.5%, illustrating that knowing when not to act is distinct from ordinary task success.

The Evidence Sufficiency Benchmark uses five evidence levels from full support through partial/irrelevant/absent to conflicting support. Its evaluated models over-answered 65–91% of L5 conflicting-evidence cases. That motivates a hard separation between an estimate that retrieval may improve coverage and the proof/action controller's authority to answer.

## Data contract

The input is a RetrievalUtilityEstimate bound to one decision, principal, scope, valid_at, known_at, exact EvidenceIndex state digest and unresolved claim set. A state change or newly ingested source invalidates the estimate and requires recalculation. It includes:

- exact evidence-index snapshot SHA-256, unresolved conclusion IDs and their deterministic digest;
- estimator identity/version and estimator artifact digest;
- held-out calibration-split digest, feature-schema digest and utility-schema digest;
- correctness-gain lower confidence bound;
- lifecycle-risk-reduction lower confidence bound;
- retrieval-harm upper confidence bound;
- measured/converted retrieval cost;
- remaining calls, remaining cost budget and minimum net-value margin.

The gate does not train the estimator. The upstream component must calibrate marginal effects on a held-out split and produce conservative confidence bounds. A SHA-256 receipt binds the estimate to its inputs; it is not a digital signature and does not prove that the estimator was honestly calibrated.

## Integration with lifecycle action selection

The gate returns RetrieveMore or Stop plus a deterministic decision receipt. Its to_lifecycle_progress() adapter converts that outcome into the existing lifecycle controller's progress field and carries the VOI decision digest into the lifecycle action receipt.

On RetrieveMore, the orchestrator must perform the next retrieval step, rebuild the evidence/claim ledger, rerun proof selection and produce a new value estimate. The previous estimate is bound to the old unresolved claim set and must not be reused.

On Stop, the gate does not invent an answer. The lifecycle action controller still decides whether the correct terminal action is ANSWER (only with the opaque independently verified proof capability), WARN_STALE, EXPLAIN_CONFLICT or ABSTAIN.

## Benchmark protocol

Compare at fixed and explicitly measured budgets. Required reports:

| Dimension | Metric | Why it matters |
|---|---|---|
| Trigger ranking | AUC / pairwise ranking of positive vs neutral vs harmful retrieval decisions | Does the estimate rank calls by actual marginal utility? |
| Probability calibration | Brier score, ECE, reliability curves and lower-bound coverage | Are predicted gains trustworthy? |
| Threshold transfer | Target-vs-realized retrieval-call rate across held-out splits and domains | Does a calibrated budget survive distribution shift? |
| Utility frontier | LifeEM / correctness / proof coverage versus realized calls, tokens, latency and cost | Is retrieval buying useful decisions? |
| Retrieval harm | Fraction of calls that lower correctness, introduce stale/conflicting distractors or increase unsupported claims | More context can make outcomes worse |
| Stopping quality | Correct stop, premature stop, redundant call, budget exhaustion, unresolved-claim count at stop | Does the agent stop for the right reason? |
| Operational overhead | Estimator latency/tokens, retrieval p50/p95/p99, total cost/query | The router itself is not free |

Required baselines:

1. Always retrieve at fixed k.
2. BM25 score threshold.
3. Reader confidence-only / CONFRAG-like gate.
4. Claim-coverage-only / HALT-like gate.
5. Broca proof-coverage plus calibrated VOI gate.

Use paired per-query outcomes and the same generator, reader prompt, corpus, proof verifier, budget accounting and held-out splits. Report the full budget frontier, not only one operating point. Do not compare LifeEM to ordinary answer accuracy or quote a retrieval-call reduction without measured answer/action quality.

## Prospective qualification targets

These are engineering targets, not measured Broca results:

- 100% proof coverage for all required claims at any ANSWER action.
- Zero retrieval-budget overshoot, measured by actual calls, tokens and cost rather than nominal thresholds.
- The lower confidence bound should be conservative: measured positive-utility coverage must meet the chosen calibration confidence level on held-out data.
- Reduce retrieval calls on already-covered or low-value cases without statistically significant loss in LifeEM or supported-answer quality. The CONFRAG 5–39% reduction is a contextual industry range, not a promised Broca result.
- Report and minimize retrieval-harm rate; any increase in stale, contradictory or distracting evidence is charged to retrieval harm, not hidden inside context length.
- Keep the claim-coverage-only and confidence-only baselines. Broca's claimed advantage must come from their combination, not from comparing against always-retrieve alone.

## Implementation and limitations

The reference module is memory_retrieval_value.rs. It uses deterministic integer arithmetic, conservative benefit/harm bounds, a strict margin comparison, snapshot/query binding and a SHA-256 decision receipt. It returns an explicit early-stop state and preserves unused retrieval calls.

The module does not train the value estimator, authenticate calibration artifacts, measure token/latency costs, or prove that the supplied lower/upper bounds are statistically valid. The current lifecycle action controller accepts a progress record with an optional VOI digest; orchestrators should use this adapter to populate it. A future release can require authenticated estimator receipts before permitting a production retrieval continuation.

No retrieval reduction, calibration score, LifeEM improvement or SOTA result is claimed by the reference implementation.

## References

- HALT, *Verification-Aware Stopping for Retrieval-Augmented Search Agents*, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.343/
- When Should Active RAG Retrieve? *A Budget-Aware Evaluation of Utility, Calibration, and Cost* (2026): https://arxiv.org/abs/2607.24010
- CONFRAG, *Confidence-Guided Retrieval-Augmented Generation*, EMNLP 2026 Industry Track: https://aclanthology.org/2026.emnlp-industry.171/
- AgentAbstain, NeurIPS 2026 Evaluations & Datasets Track: https://agentabstain.github.io/
- Evidence Sufficiency Benchmark, *Do LLMs Know When Evidence is Insufficient?* (2026): https://doi.org/10.32604/cmc.2026.086343
