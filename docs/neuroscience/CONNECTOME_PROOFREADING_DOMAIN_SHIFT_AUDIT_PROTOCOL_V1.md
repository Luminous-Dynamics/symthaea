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

If novelty detection is included, provide `ood_score` for every row. Domain metadata and novelty scores must each be complete across the input; partial annotation fails closed. Every domain-tagged run must provide `--domain-split-manifest <path>`. The scorer hashes the file bytes itself and records the SHA-256. The manifest binds the exact dataset revision, shift axis, seen/held-out domain lists, sample IDs, dependency-cluster IDs and split roles. It must contain every row in the requested evaluation partition, not a curated subset.

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

The scorer reports a ten-bin fixed-width probability-calibration table (last bin includes p_error = 1.0), including counts, mean predicted error probability, observed adjudicated error frequency and absolute gap. Separately, to align with ConnectomeBench2's confidence-calibration view, it reports a 15-bin adaptive confidence-versus-correctness ECE/MCE and a 12-bin equal-mass confidence reliability curve. Equal-confidence ties are never split across quantile bins. Each equal-mass reliability bin also carries cluster-bootstrap 95% intervals for observed accuracy and mean confidence, using fixed full-sample bin boundaries for the resampled curve; bins with inadequate valid replicate support have no interval. It also reports:
- Brier score and log loss;
- Normalized sharpness, `Var(p_error) / (mean(p_error) × (1 − mean(p_error)))`, to distinguish informative confidence spread from a flat predictor. It is reported with a cluster-bootstrap interval; when the denominator is zero or near machine precision, it is `NOT_ESTIMABLE`. Sharpness is not calibration quality: report it alongside proper scoring rules and calibration errors.
- A conditional-IID calibration noise-floor simulation: for each row, labels are resampled independently with correctness probability equal to predicted top-class confidence; median and 95th-percentile ECE are reported for fixed-probability, adaptive-confidence and equal-mass reliability views. This estimates finite-sample ECE expected under that idealized independent-label null. It is not cluster-aware and is not a hypothesis test; never subtract it from observed ECE to claim calibration or use it as a promotion gate.
- expected calibration error (ECE) and maximum calibration error (MCE);
- false-accept fraction among predictions at `p_error >= 0.90` and `>= 0.95`;
- count and impact weight of true errors assigned `p_error < 0.10`.

ECE/MCE depend on binning and support. Always publish the bin counts and proper scoring rules; never present ECE alone as proof of calibration. A sparse bin is not evidence of zero risk.

### Domain-stratified performance

Each domain receives its own classification metrics, reliability bins, high-confidence false-accept counts, sample count and distinct cluster count. Pooled metrics cannot override a failed or unsupported critical domain. The task/error-family grouping remains separate from the domain axis.

### Novelty detection

When every row carries `ood_score`, the run must supply `--ood-detector-id` and `--ood-detector-profile-sha256`. These identify the frozen novelty method and hash its representation, model and preprocessing. The report records both values and evaluates the score against the predeclared `is_ood_domain` flag using AUROC and tie-aware average precision. On a confirmatory/lockbox run, `--ood-threshold-selection-input <development-predictions.jsonl>` derives the operating threshold from a separate development file. The development file is checked against the same exact split manifest and detector/profile. The fixed rule maximizes held-out-domain recall subject to a 5% empirical development false-OOD-rate ceiling; it does not inspect reconstruction-error labels. The report records the development input SHA-256, counts, selected threshold and operating point. If no threshold satisfies the 5% constraint, the report records `NO_FEASIBLE_THRESHOLD` and does not apply an unsafe threshold. On the confirmatory/lockbox data, cluster-bootstrap intervals are also reported for false-OOD rate and held-out-domain recall at the selected threshold; resamples use the declared dependency clusters and are withheld below the minimum valid-replicate requirement.

Do not choose the threshold using confirmatory/lockbox labels. The scorer derives it from a separate development input and rejects reusing development as the evaluation partition. Development-only runs report AUROC/AUPRC but do not select a threshold. AUROC does not establish an operating point; AUPRC depends on the declared OOD prevalence.

## Invocation

Domain-stratified calibration and OOD ranking:

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin proofreading-score -- \
  --input docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT.example.jsonl \
  --mapping-reference docs/neuroscience/CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json \
  --dataset-revision-commit-sha 1111111111111111111111111111111111111111 \
  --policy-id domain-shift-fixture-v1 \
  --prediction-profile-sha256 <exact-64-character-profile-digest> \
  --evaluation-partition development \
  --review-time-basis synthetic-proxy \
  --require-domain-shift-audit \
  --domain-split-manifest docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SPLIT.example.json \
  --ood-detector-id mahalanobis-diagonal-v1 \
  --ood-detector-profile-sha256 <exact-64-character-ood-detector-digest> \
  --output domain-shift-development-report.json
```

The checked-in domain-split manifest and prediction file are synthetic regression fixtures. The all-ones dataset revision SHA is a synthetic identity, not a claim about the real ConnectomeBench2 commit. For a real run, pin the actual V2 commit in both the CLI and manifest, enumerate the complete split lineage, and never reuse the synthetic IDs.

For a later confirmatory/lockbox report, provide the full confirmatory/lockbox prediction JSONL as `--input`, set `--evaluation-partition confirmatory` or `lockbox`, and add `--ood-threshold-selection-input <development-predictions.jsonl>`. The scorer independently hashes and validates the development rows against the same split manifest, then selects the threshold under the fixed 5% development false-OOD-rate constraint. It rejects using the development partition as both selection and evaluation. Never pass confirmatory or lockbox labels back into threshold selection.

\`\`\`sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml \
  --bin proofreading-score -- \
  --input docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_CONFIRMATORY.example.jsonl \
  --ood-threshold-selection-input docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SHIFT_AUDIT.example.jsonl \
  --mapping-reference docs/neuroscience/CONNECTOMEBENCH2_V2_INTAKE_MANIFEST.schema.json \
  --dataset-revision-commit-sha 1111111111111111111111111111111111111111 \
  --policy-id domain-shift-confirmatory-fixture-v1 \
  --prediction-profile-sha256 <exact-64-character-profile-digest> \
  --evaluation-partition confirmatory \
  --review-time-basis synthetic-proxy \
  --require-domain-shift-audit \
  --domain-split-manifest docs/neuroscience/examples/CONNECTOME_PROOFREADING_DOMAIN_SPLIT.example.json \
  --ood-detector-id mahalanobis-diagonal-v1 \
  --ood-detector-profile-sha256 <exact-64-character-ood-detector-digest> \
  --output domain-shift-confirmatory-report.json
\`\`\`


The synthetic confirmatory fixture demonstrates why this separation matters: the development-selected threshold is 0.85 with 0% development false-OOD rate and 100% development OOD recall, but on the deliberately shifted confirmatory scores it catches only 1 of 8 held-out-domain rows (12.5% recall) at 0% in-domain false-OOD rate. This is a synthetic stress test, not a real transfer result.

For a later confirmatory/lockbox report, provide the full confirmatory/lockbox prediction JSONL as `--input`, set `--evaluation-partition confirmatory` or `lockbox`, and add `--ood-threshold-selection-input <development-predictions.jsonl>`. The scorer independently hashes and validates the development rows against the same split manifest, then selects the threshold under the fixed 5% development false-OOD-rate constraint. It rejects using the development partition as both selection and evaluation. Never pass confirmatory or lockbox labels back into threshold selection.

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
