# Connectome design-based residual-risk audit protocol V1

**Status:** executable design contract plus synthetic fixture; no biological sample has been drawn or evaluated.  
**Tracking:** [SCI-CONNECTOMICS-005 / issue #9187](https://github.com/Luminous-Dynamics/symthaea/issues/9187).

## Why this protocol exists

Queue-only evaluation suffers verification bias: candidate correctness is observed because a policy selected a candidate. It cannot estimate the error burden remaining among candidates not selected. Complete multi-rater adjudication across every item in a large pool can be too expensive. This protocol defines a probability audit that samples both queue-selected and unselected candidates through a frozen population frame.

This is a design-based estimator for the **finite frozen candidate pool**, not for every segmentation error in the entire connectome. Candidate recall against the whole connectome is not identified by this sample unless the candidate-generation process itself has a known sampling frame.

## Sampling design

V1 is a stratified one-stage cluster sample without replacement:

1. Freeze the complete candidate pool and create exactly one frame row per candidate: `candidate_id`, `stratum_id`, `cluster_id`.
2. A cluster is the declared primary sampling unit (PSU), chosen to capture dependence such as spatial block, neuron, or source-operation lineage. A cluster may not cross strata. Where dependencies span multiple candidate relationships, choose a conservative PSU that contains them or do not claim the variance estimate captures those dependencies.
3. Define strata before outcomes are observed. Strata may encode policy-selected status, rank decile, species, volume/region, candidate error family, or uncertainty band. Ensure every candidate has positive inclusion probability.
4. Within each stratum, select a fixed number of clusters without replacement. The selector sorts clusters by SHA-256 of the domain separator, revealed seed, stratum ID, and cluster ID, then selects the lowest hashes. This is reproducible pseudorandom selection; its design interpretation assumes the seed was generated from a cryptographically secure source and not chosen after seeing outcomes.
5. Audit every candidate in each selected cluster. Do not subsample candidates inside selected clusters in V1.
6. Require at least two sampled clusters in every non-census stratum so a sample variance is estimable. A full census stratum has zero design variance.

The seed hash must be recorded in a trusted, timestamped artifact (for example a repository commit or external preregistration record) **before outcome collection**. The seed itself is revealed in the audit file after collection. The executable checks seed/hash consistency but cannot prove when the commitment was timestamped; that chronology is an external governance requirement.

## Estimands and estimators

For stratum h, let N_h be the number of population clusters, n_h the number of sampled clusters, and y_hc the total of a candidate-level outcome across all candidates in cluster c.

For each metric, the estimated finite-population total is the stratified Horvitz–Thompson estimator for this equal-probability cluster design:

```text
T_hat = sum_h (N_h / n_h) * sum_(c in sample_h) y_hc
```

For a non-census stratum, the unbiased design variance estimator is:

```text
V_hat(T_hat_h) = N_h^2 * (1 - n_h/N_h) * s_h^2 / n_h
s_h^2 = sum_(c in sample_h) (y_hc - mean_h)^2 / (n_h - 1)
```

Strata are sampled independently, so total variances are added across strata. This cluster-total formulation respects within-cluster dependence; it does not pretend candidate rows are independent. The audit reports:
- estimated counts of correct, incorrect, and indeterminate proposals;
- estimated incorrect-candidate fraction using the known finite candidate-pool size as denominator;
- estimated positive isolated query-correction opportunity mass;
- estimated negative isolated query-harm mass;
- signed net isolated query-error reduction;
- estimated independent audit-panel person-minutes (sum of actual person time across annotators/adjudicators), kept separate from normal policy-review cost;
- design variance and standard error for every estimate.

Normal-approximation 95% intervals are emitted only for census strata or when each non-census stratum has at least 30 sampled clusters. This is a practical reporting guard, not a universal coverage guarantee. For smaller samples, the tool emits the design variance and standard error but withholds the interval; use design-specific randomization inference or a preregistered small-sample method before making a confidence claim.

The query-delta estimands are isolated candidate counterfactuals. Their totals are **not** a sequential graph replay and do not account for conflicts or interactions between mutations. Use conflict-connected PSU sampling or a prospective policy trial for sequential graph effects. An indeterminate adjudication has no oracle delta and is estimated as its own category.

## Data binding and integrity

The audit file binds to exact frozen proposal bytes and exact sampling protocol bytes. The complete population frame must exactly match the candidate IDs in the proposal input. The report emits a canonical frame digest, exact full audit-input SHA-256, canonical sampled-outcomes SHA-256, protocol SHA-256, proposal-input SHA-256, and seed commitment digest. Outcomes must include every candidate inside every deterministically selected cluster, and no candidate outside those clusters.

The audit must be generated independently from the ranker. The ranker cannot see the seed, sample selection, or outcome file. Do not tune model scores, query weights, strata, or sampling fractions after seeing audit outcomes.

The Rust tool is intentionally V1 narrow: stratified equal-probability cluster sampling without replacement. It does not accept arbitrary unequal-probability designs, probability-proportional-to-size sampling, systematic spatial samples, two-stage subsampling, or unknown inclusion probabilities. Those require separate estimators and design audits rather than silently reusing this formula.

## Collection and quality controls

- Select strata and the primary sampling-unit definition based on known dependence before drawing the sample.
- Freeze the candidate frame, policy ordering, graph revision, query bundle, reference graph, and protocol digest.
- Generate a high-entropy seed and commit its digest before reviewers see selected candidates.
- Reveal the seed only after outcomes are frozen.
- Blind reviewers/adjudicators to policy identity, model priority, and unreviewed/selected status wherever feasible.
- Preserve raw adjudicator votes and indeterminate outcomes; use the independent multi-rater protocol for sampled candidates where feasible.
- Record nonresponse and missingness separately. V1 rejects missing outcomes inside selected clusters; it does not infer missingness or silently replace a sampled cluster. A nonresponse adjustment requires a separately justified model or response-propensity design.
- Report person-minutes for all sample auditing separately from normal policy-review minutes.
- An independent auditor should recompute the selected cluster list and frame digest from the frozen seed and population roster.

## Promotion and kill criteria

- Every frame candidate has non-zero known inclusion probability.
- No missing or extra sampled outcomes; all sampled clusters receive a complete audit.
- Synthetic tests recover known finite-population totals under census, verify the finite-population correction, and reject tampered seeds, frames, protocols, and sample outcomes.
- Report residual incorrect-candidate fraction and harmful query-impact mass by stratum; do not hide an unreviewed-tail failure behind a pooled average.
- If a stratum has fewer than two sampled clusters and is not a census, fail closed because an unbiased within-stratum variance estimate is unavailable under this design.
- If the sampling frame is incomplete, cluster definition fails to capture relevant dependence, or the seed commitment was created after outcomes were inspected, invalidate the run.

This audit estimates finite-pool residual risk. Causal claims that one queue policy is better still require prospective randomized or blocked assignment to the review policies. No biological result or SOTA claim is currently available.

## Run

```bash
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome_sampling_audit -- \
  --proposals frozen-proposals.json \
  --sampling-protocol CONNECTOME_SAMPLING_AUDIT_PROTOCOL_V1.md \
  --audit blinded-probability-audit.json \
  --output residual-risk-report.json
```

The example fixture uses placeholder proposal/protocol digests and a synthetic fixed seed. Replace source digests and the population frame with the exact frozen study artifacts before use.
