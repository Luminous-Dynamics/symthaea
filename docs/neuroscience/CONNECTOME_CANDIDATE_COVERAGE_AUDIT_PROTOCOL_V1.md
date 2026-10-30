# Connectome candidate-generation coverage audit protocol V1

**Status:** executable design contract and synthetic example; no biological search sample has been collected or evaluated.  
**Tracking:** [SCI-CONNECTOMICS-006 / issue #9204](https://github.com/Luminous-Dynamics/symthaea/issues/9204).

## Purpose: measure what the model never proposed

The frozen proposal pool is not the universe of possible connectomic errors. A queue may have high precision and excellent review yield among its proposals while systematically missing an error family, region, species, or morphology. Auditing only proposed candidates cannot estimate this failure mode.

This protocol samples independently from a declared search frame and asks experts to find **all** eligible errors in each selected primary sampling unit (PSU), including PSUs for which the model proposed nothing. It estimates candidate-generation recall over that exact frame. It is separate from:

- proposal precision and review yield;
- the unreviewed-candidate-tail audit in #9187;
- whole-connectome recall outside the declared sampling frame;
- the causal effect of a queue policy, which still requires randomized policy assignment.

## Sampling frame and unit

V1 uses one-stage stratified simple random sampling without replacement of PSUs:

- A PSU is fixed before sampling (for example, a non-overlapping EM block, a canonical neuron, or another independently searchable unit). Do not mix incompatible PSU definitions within one stratum.
- The population frame contains every eligible PSU in the declared volume/revision, with a unique `psu_id` and a predeclared `stratum_id`.
- A PSU belongs to exactly one stratum. Strata are mutually exclusive and exhaustive over the claimed search frame.
- Every PSU has positive inclusion probability. Within each stratum, exactly `n_h` of `N_h` PSUs are selected with equal probability, without replacement.
- Every error inside a selected PSU is exhaustively searched and recorded. A selected PSU with no errors must still appear with an empty `errors` array. The complete `error_families` vocabulary is declared in the audit file before collection; it must be unique and non-empty. Errors outside that vocabulary invalidate the run. Declared families absent from the sample remain visible with zero estimated count and undefined recall when their denominator is zero.
- V1 does not permit within-PSU subsampling, unequal-probability sampling, PPS, systematic spatial sampling, or nonresponse substitution.

The seed must be generated from a cryptographically secure source and its SHA-256 commitment timestamped in a trusted preregistration artifact before outcomes are collected. The seed is revealed in the evaluator-only audit file. The executable checks the commitment but cannot establish the timestamp or prove that the search itself was blind.

## Independent error coverage adjudication

For every independently verified error, record:

- a stable `verified_error_id`, unique across the sampled frame;
- a predeclared `error_family`;
- an evaluator-only nonnegative `impact_weight` derived from the frozen scientific query contract;
- zero or more frozen proposal candidate IDs independently judged to cover that error.

An empty `proposal_candidate_ids` means candidate generation missed the error. Mapping correctness is itself an adjudication task: a nearby candidate is not coverage unless it proposes the relevant correction. Multiple proposal IDs may cover one verified error; the error is still counted once. Candidate IDs must resolve to the exact frozen proposal input.

Reviewers performing exhaustive search and coverage adjudication must be blind to policy identity, ranking, model scores, and sampling weights wherever feasible. Preserve disagreement and indeterminate cases in the raw adjudication records in the eventual multi-rater extension; V1 accepts only independently adjudicated determinate search results. A failure to exhaustively inspect a sampled PSU invalidates the run; do not replace it with a convenient PSU.

## Estimands

For each stratum (h), let (N_h) be the number of frame PSUs, (n_h) the sampled PSUs, and (y_{hi}) a PSU total (for example, all verified errors, candidate-covered errors, missed errors, impact mass, or audit panel person-minutes).

The estimated finite-frame total is:

[
widehat{T}_y=sum_h rac{N_h}{n_h}sum_{iin s_h} y_{hi}.
]

Primary metrics:

1. Candidate-generation recall: estimated number of independently verified errors covered by at least one frozen proposal divided by estimated total verified errors.
2. Impact-weighted candidate-generation recall: estimated impact mass covered by proposals divided by estimated total impact mass.
3. Estimated missed error count and missed impact mass.
4. Results by the complete predeclared error-family vocabulary, with sampled counts and denominators shown; an unobserved family has an undefined recall and does not imply that the family is absent from unsampled PSUs.
5. Estimated exhaustive-audit person-minutes, separate from normal policy-review time and compute cost.

The ratio variance uses a first-order Taylor linearization at the estimated ratio (R). Within each sampled PSU, form (z_{hi}=y_{hi}-R x_{hi}), where (y) is the covered count/impact and (x) is the total count/impact. Then:

[
widehat{V}(widehat{R}) =
rac{
sum_h N_h^2(1-n_h/N_h)s^2_{z,h}/n_h
}{
widehat{T}_x^2
}.
]

Census strata contribute zero sampling variance. V1 emits normal-approximation 95% intervals only when every non-census stratum has at least 30 sampled PSUs; otherwise it reports the point estimate, design variance, standard error, and denominators without a nominal 95% interval. This is a conservative reporting rule, not a universal coverage guarantee. Ratio intervals are bounded to [0,1].

The estimator is design-based for the **finite declared search frame**. It does not establish recall for tissue outside that frame, errors that the exhaustive-review protocol cannot detect, or other volumes/species.

## Integrity and information-flow boundary

The audit binds to exact bytes for the frozen proposal input, frozen ranking report, and sampling protocol. The ranking report must itself identify the exact proposal-input digest. The complete PSU frame, selected PSU list, seed commitment, raw sampled outcomes, and report digests are retained.

The executable recomputes the deterministic sample from the revealed seed and frame. It rejects duplicate or missing PSUs, unselected outcomes, duplicate verified-error IDs, unknown proposal IDs, missing exhaustive-search attestations, and digest mismatches.

The proposal generator and ranker must never receive the seed, selected sample, independent error labels, uncovered-error identities, or impact weights. Audit outcomes must not be used to tune rank scores, query weights, strata, or the lockbox.

## Synthetic validation requirements

Before biological use, validate known finite populations covering:

- census exactness and finite-population correction;
- all errors covered, no errors covered, and no errors present;
- empty PSUs and unequal PSU error counts;
- rare and heterogeneous error families;
- zero impact mass and skewed impact weights;
- seed/frame/protocol/proposal/ranking tampering;
- missing/extra/duplicate PSU and error records;
- invalid proposal references and non-exhaustive search flags;
- empirical design-based interval coverage under repeated synthetic seeds.

A synthetic pass validates implementation mechanics only. It does not show that experts can exhaustively identify all biological errors in a PSU.

## Promotion criteria and kill conditions

- Independent experts can execute a reproducible exhaustive search protocol with measured panel person-time.
- The declared PSU frame is complete, non-overlapping, and aligned to the scientific claim.
- Inclusion probabilities are positive and known; sample selection is reproducible.
- Candidate-generation recall and impact-weighted recall are reported separately from review precision and queue efficiency.
- If candidate coverage is low or sharply heterogeneous by region/error family, improve candidate generation before tuning the ranker.
- If exhaustive search is not feasible, narrow the estimand and report the limited frame; never label candidate-pool recall as whole-connectome recall.
- No biological performance, causal policy gain, or SOTA claim is made until a preregistered independent biological study is completed.

## Run

```bash
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome_candidate_coverage_audit -- \
  --proposals frozen-proposals.json \
  --ranking-report frozen-ranking-report.json \
  --sampling-protocol CONNECTOME_CANDIDATE_COVERAGE_AUDIT_PROTOCOL_V1.md \
  --spatial-frame spatial-frame.json \
  --spatial-frame-report spatial-frame-report.json \
  --audit independent-volume-search-audit.json \
  --output candidate-coverage-report.json
```

The example file contains placeholder source digests and synthetic records. Replace all placeholders and bind it to the exact frozen study artifacts before evaluation.


## Spatial frame prerequisite (SCI-CONNECTOMICS-013)

Before collecting or estimating candidate-generation recall, run [the spatial PSU lattice frame auditor](CONNECTOME_SPATIAL_FRAME_AUDIT_PROTOCOL_V1.md) and use its exported eligible `population_frame` verbatim. The coverage audit now requires `--spatial-frame-report` and binds the exact report bytes, complete spatial lattice digest, and eligible PSU population digest. It rejects a separately maintained population list or a failed spatial-frame report.

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-spatial-frame-audit -- \
  --frame spatial-frame.json \\
  --eligibility-mask eligibility-mask.bitset --output spatial-frame-report.json

cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin connectome-candidate-coverage-audit -- \
  --proposals frozen-proposals.json --ranking-report frozen-ranking-report.json \
  --sampling-protocol CONNECTOME_CANDIDATE_COVERAGE_AUDIT_PROTOCOL_V1.md \
  --spatial-frame spatial-frame.json \
  --spatial-frame-report spatial-frame-report.json \
  --audit independent-volume-search-audit.json --output candidate-coverage-report.json
```

The frame report only carries PASS after it verifies the exact eligibility-mask artifact digest, packed bit order, byte length, final padding, and consistency between mask foreground voxels and every PSU disposition. This proves no gaps/overlaps and mask-to-PSU consistency inside the declared integer box; it does not prove the box or mask selection rule covers all biologically eligible tissue.


## Verified spatial-mask attestation requirement

The candidate-coverage verifier rejects a spatial-frame report unless it declares `eligibility_mask_verified: true` and:
- its actual mask SHA-256 equals the declared artifact SHA-256;
- the mask encoding is one of the two protocol-defined LSB-first x-fastest representations;
- artifact byte count equals `ceil(declared_voxel_count / 8)` for packed bytes or twice that for ASCII hex;
- mask foreground count/fraction agrees with the declared voxel count;
- grid dimensions multiply to `total_grid_cells`, and mask tile rows cover that exact count;
- per-tile geometric volumes sum to the declared voxel count, and per-tile foreground counts sum to the global mask count;
- positive-mask PSU IDs exactly equal the exported eligible population, with the same cardinality and no duplicates.

A geometry-only status, missing mask fields, digest mismatch, count mismatch, duplicated PSU, or different population set blocks coverage estimation. This binds the coverage estimator to a mask-verified frame report. It still does not independently validate the scientific correctness of the mask-selection rule or prove that the declared volume bounding box is complete.
