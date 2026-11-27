# MACE evidence-span ablation v0.1

**Status:** Typed evaluator-side paired analysis implemented with deterministic tests. Diagnostic only: this module does not verify randomizer signatures, provider instrumentation, external evaluator custody, or production evidence-use links.

## Why this is a separate estimand

MACE already has three related but different intervention levels:

- Existing paired memory-to-decision and counterfactual runners remove events, roots, families, or memory portfolios and estimate effects of those memory interventions.
- #13390 estimates the total effect of an on-policy memory policy, including feedback into later user disclosure, memory writes, retrieval and subsequent episode histories.
- #13973 records whether a current, admissible evidence item was actually included in the exact serialized model context.

The new estimand is narrower: within one frozen episode history, what changes when one particular admissible evidence representation is exposed to the model versus removed or replaced with a neutral marker?

This is not the total policy effect. The total policy effect includes changes to future histories and disclosures. The span-ablation contrast holds the episode history and non-intervention inputs fixed and isolates the marginal effect of one context exposure under that frozen history.

## Observation contract

Each matched pair has exactly two evaluator-side records:

1. Exposed: the model receives the selected evidence span and the record carries a digest of the #13973 DecisionEvidenceUseLinkV1.
2. Ablated: the same request template is replayed with the target span removed or replaced by a neutral marker; no target-span use link is allowed.

The two rows must agree on run, profile cluster, episode, population stratum, intervention mode, randomization receipt, assignment manifest, policy/model revisions, prompt and tool contracts, task, frozen history, shared context projection, intervention specification, target evidence digest and outcome specification. Their exact request hashes must differ and randomized execution order must be 0/1.

The candidate must not receive assignment receipts, evaluator outcome receipts, utility, regret, violation counts, cluster identity or evaluator labels. Those remain evaluator-side only.

## Estimator

- Pair exposed and ablated outcomes within each pair_block_id.
- Utility effect = exposed utility minus ablated utility.
- Regret reduction = ablated regret minus exposed regret.
- Decision-change rate uses a canonical semantic decision fingerprint, not raw action JSON bytes.
- Added lifecycle cost = exposed cost minus ablated cost.
- Average episode-pair effects within each independent profile cluster first.
- Bootstrap profile-cluster means with a deterministic SplitMix64 stream, 2,000 percentile replicates, and 95% bounds.
- Report subgroup effects by population stratum.
- Report privacy violations, unauthorized actions and erasure resurrections as non-compensable hard gates. Utility cannot offset them; no composite score is formed.

A minimum of 30 independent profile clusters is the prospective confirmatory floor. Fewer clusters are exploratory; the code reports this floor rather than silently inflating sample size with repeated episodes.

## Trust and causal limits

This library validates structural pairing, metadata equality, digest shape, execution-order positions, exact request hash changes and presence/absence of the evidence-use-link commitment. It does not prove that the randomizer actually randomized, that the observer captured the true provider request, that the context projection removed only the target span, or that evaluator outcome receipts are independently valid.

A SHA-256 assignment receipt is a reference, not a signature. Before describing a result as a causal effect, an independent evaluator must verify the randomization receipt, sealed outcome custody, adapter instrumentation, exact request bytes, span transformation and missingness. The report is therefore hard-coded as diagnostic-only.

The ablation mode is homogeneous within one analysis. Do not pool deletion and neutral-marker results into one estimate. Run them as separate experiments with matched task/model/seed strata.

## Metrics and prospective targets

Keep conventional memory accuracy and latency separate from this intervention report. It adds paired utility and regret effects with profile-cluster intervals, decision-change rate, added end-to-end cost, per-stratum effects, independent profile-cluster count and privacy/authority/erasure hard gates.

Prospective target: positive utility and regret-reduction lower 95% cluster-bootstrap bounds, no worst-stratum regression, zero hard-gate violations, and lower user/compute burden than an equivalent active-reobservation strategy. These are targets, not achieved results.

## Run

```bash
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --lib
cargo clippy --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-evaluator --lib -- -D warnings
```

The library uses synthetic fixtures only. It has not been run against a live provider, an official benchmark, or a sealed independent evaluator.