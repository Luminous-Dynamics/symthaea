# Domain-shift calibration and abstention audit V1

**Research track:** [SCI-CONNECTOMICS-023 / #9403](https://github.com/Luminous-Dynamics/symthaea/issues/9403)  
**Implementation:** optional extension to `proofreading-score` in PR [#8883](https://github.com/Luminous-Dynamics/symthaea/pull/8883)  
**Status:** source implementation and synthetic fixture added; Rust compilation and execution remain unverified on the authoring environment.

## Research question

Does a frozen connectomics proofreading model remain calibrated and safely selective when the evaluated domain differs from the data used to train or calibrate it? Can a separate novelty detector identify held-out domains early enough to route uncertain cases to human or specialist review?

The evaluation must separate:
1. **Error probability:** `p_error`, the model's probability that a candidate is a genuine reconstruction error.
2. **Ranking priority:** `review_priority_score`, the frozen queue-ordering score.
3. **Domain novelty:** `ood_score`, the separate score for detecting a predeclared held-out domain.
4. **Grounded domain status:** `is_ood_domain`, fixed by the data split protocol, not by model output.

No one value substitutes for another.

## Input contract

Use JSONL rows conforming to [CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_INPUT.schema.json](CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT_INPUT.schema.json).

The existing eight prediction fields remain unchanged. The optional complete domain tuple is:
- `domain_id`
- `shift_axis`
- `is_ood_domain`

If novelty detection is included, provide `ood_score` for every row. Domain metadata and novelty scores must each be complete across the input; partial annotation fails closed.

A cluster_id must not span two domain_id values in one run; use globally unique/domain-qualified dependency identifiers. This prevents the OOD bootstrap from treating distinct domain blocks as one cluster.

One run contains exactly one shift axis:
- `species`
- `region`
- `source_volume`
- `source_archive`
- `acquisition_protocol`
- `segmentation_lineage`
- `modality_availability`
- `morphology_regime`

Run one audit per axis. A domain ID must map to one axis and one held-out status for the entire run. A held-out label is an evaluation-design fact: do not mark a sample OOD merely because its novelty score is high.

## Metrics

### Probability calibration

The scorer reports a ten-bin fixed-width probability-calibration table (last bin includes p_error = 1.0), including counts, mean predicted error probability, observed adjudicated error frequency and absolute gap. Separately, to align with ConnectomeBench2's confidence-calibration view, it reports a 15-bin adaptive confidence-versus-correctness ECE/MCE and a 12-bin equal-mass confidence reliability curve. Equal-confidence ties are never split across quantile bins. It also reports:
- Brier score and log loss;
- expected calibration error (ECE) and maximum calibration error (MCE);
- false-accept fraction among predictions at `p_error >= 0.90` and `>= 0.95`;
- count and impact weight of true errors assigned `p_error < 0.10`.

ECE/MCE depend on binning and support. Always publish the bin counts and proper scoring rules; never present ECE alone as proof of calibration. A sparse bin is not evidence of zero risk.

### Domain-stratified performance

Each domain receives its own classification metrics, reliability bins, high-confidence false-accept counts, sample count and distinct cluster count. Pooled metrics cannot override a failed or unsupported critical domain. The task/error-family grouping remains separate from the domain axis.

### Novelty detection

When every row carries `ood_score`, the report evaluates the score against the predeclared `is_ood_domain` flag using AUROC and tie-aware average precision. An optional `--ood-score-threshold` reports false-OOD rate on in-domain rows and held-out-domain recall at that frozen operating point.

Do not choose the threshold using confirmatory/lockbox labels. Select it on a separate development/calibration partition, then freeze and digest that choice. AUROC does not establish an operating point; AUPRC depends on the declared OOD prevalence.

## Invocation

Domain-stratified calibration and OOD ranking:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin proofreading-score -- \
  --input docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT.example.jsonl \
  --mapping-reference docs/neuroscience/CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json \
  --dataset-revision-commit-sha <exact-40-character-v2-commit> \
  --policy-id domain-shift-fixture-v1 \
  --prediction-profile-sha256 <exact-64-character-profile-digest> \
  --evaluation-partition development \
  --review-time-basis synthetic-proxy \
  --require-domain-shift-audit \
  --ood-score-threshold 0.50 \
  --output domain-shift-report.json
```

The placeholders above must be replaced with the exact pinned dataset and profile identities before a real run. The checked-in fixture uses synthetic IDs and is only a regression input.

## Split and statistical requirements

- Freeze train, probability-calibration, development, confirmatory and lockbox domain membership before outcome inspection.
- Keep source volume, neuron/root, operation, physical PSU and related reconstruction lineages within the declared split boundary.
- Distinguish held-out-domain transfer from covariate-shift stress within a represented domain.
- Do not treat image crops, skeleton windows, synapses or repeated pipeline variants as independent clusters.
- Report the independent domain, volume, neuron/root and PSU counts. The scorer now reports deterministic cluster-bootstrap 95% percentile intervals for fixed-bin ECE/MCE, adaptive 15-bin confidence ECE, 12-bin confidence reliability ECE and Brier score overall and per domain, plus OOD AUROC/AUPRC where a resampled replicate retains both OOD classes. OOD status is NOT_ESTIMABLE if the complete run contains no examples from either in-domain or held-out status. Clusters are resampled as complete units, never individual rows. An interval is withheld if fewer than two clusters exist or fewer than 80% of requested replicates are estimable. This is a minimum engineering safeguard, not a claim that two clusters provide adequate inferential power; promotion requires materially larger, independently justified cluster counts, coverage simulations and a source-volume-aware sampling design.
- Preserve indeterminate adjudications and reviewer disagreement. The label reference is an adjudication process, not infallible biological truth.
- If a domain/family has inadequate independent support, mark it NOT_ESTIMABLE and refer to human/specialist review.

## Required adversarial controls

1. In-domain calibrated model with a known reliability curve.
2. Overconfident held-out species/region/volume examples.
3. Good classification ranking but poor probability calibration.
4. Good probability calibration but poor novelty separation.
5. Novelty score constant, inverted, missing, out of range or spuriously high in-domain.
6. High-confidence false accepts concentrated in a rare error family.
7. Incomplete domain tuple, partial novelty scores, inconsistent domain holdout labels and multiple shift axes in one run.
8. Duplicate sample IDs and cross-role source lineage leakage.
9. Low-support and one-cluster domains.
10. Repeated-seed finite-population simulations for calibration error, referral rates and OOD detector sensitivity.

## Promotion gates (targets, not results)

- Zero false in-domain certifications on enumerated split/lineage leakage controls.
- For supported domains, calibrated probability estimates with design/cluster-aware uncertainty and preregistered tolerance.
- At least 95% held-out-domain sensitivity at no more than 5% false-OOD rate on a declared challenge frame, only when support permits valid intervals.
- At least 90% precision for any proposed automatic high-confidence correction, with family/domain-specific lower bounds and no pooled override.
- At matched expert-minute budget, a confidence-plus-novelty referral policy improves independently adjudicated impact-weighted query-error yield over confidence-only, novelty-only, and uncertainty-only controls.
- OOD, NOT_ESTIMABLE and integrity failures route to reversible specialist/human review. No automatic authoritative annotation mutation.

These are research targets. No biological OOD result, calibration improvement, efficiency gain or SOTA win is claimed.

## External comparison boundary

ConnectomeBench2's ViT and 2026 VLM transfer evaluations are operation-level classification comparisons. NEURD's morphology rules and Probe-EM's targeted tracing are different tasks. Compare calibration/transfer only on a common source cohort and task; do not combine classification balanced accuracy with tracing ERL or reviewer-time results into a single rank.
