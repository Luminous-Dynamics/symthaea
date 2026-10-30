# SCI-JUDGE-GEOMETRY-001 — Matched judge-human score-space diagnostics

**Status:** executable fixed-matrix analysis tool under development. No Symthaea judge-quality or human-alignment result is claimed.

**Tracking:** issue #9130. This work composes with [#9065 — judge robustness](https://github.com/Luminous-Dynamics/symthaea/pull/9065), [#9123 — rubric exploitability](https://github.com/Luminous-Dynamics/symthaea/issues/9123), and [#9124 — rubric measurement protocol](https://github.com/Luminous-Dynamics/symthaea/pull/9124).

## 1. Why score geometry is a distinct benchmark dimension

Pairwise rank or label agreement discards important information about the shape of the score distribution. A judge panel can agree with itself while:
- using a much narrower portion of the scoring scale than humans;
- varying primarily along a different criterion dimension;
- collapsing multiple human-relevant dimensions into one;
- sharing an error axis despite high inter-judge correlation.

The EMNLP Findings 2026 judge-geometry study reports these phenomena in its tested Indic community datasets, languages and subjective rubric settings. It reports judge/human score-spread ratios around 0.3–0.5 and judge principal axes around 87°–89° from human axes, despite judge–judge correlations exceeding judge–human correlations in some settings. Those are results in that study's domain and design, not universal expected values or Symthaea measurements.

Source: https://aclanthology.org/2026.findings-emnlp.1696/ and https://arxiv.org/abs/2606.03043

The ICML 2026 IRT work supplies a complementary perspective: intrinsic consistency under prompt changes and alignment with humans are separate latent properties. A geometric diagnostic is a descriptive first layer, not a substitute for a graded-response or hierarchical measurement model.

Source: https://proceedings.mlr.press/v306/choi26n.html and https://arxiv.org/abs/2602.00521

NIST AI 800-3 (2026) motivates keeping fixed-benchmark results separate from generalized accuracy, with uncertainty models that account for task/item variation.

Source: https://www.nist.gov/publications/expanding-ai-evaluation-toolbox-statistical-models

## 2. Executable v1

Tool: `research/science-bench-refresh/tools/judge_score_geometry.py`

Input contract: `schema/judge-score-geometry-input.schema.json`

Tests: `tools/test_judge_score_geometry.py`

The input is a matched item × criterion × rater score matrix. Each item has at least two judge ratings and two blinded-human ratings; each rating supplies integer score millionths over the same ordered criterion list. Rater identifiers are pseudonymous digest commitments. The analysis binds exact input bytes in its report.

CLI:
```sh
python3 research/science-bench-refresh/tools/judge_score_geometry.py \
  matched-score-matrix.json --output judge-score-geometry-report.json
```

The v1 tool uses Python's standard library only. It has a deterministic symmetric Jacobi eigensolver with a configured tolerance and a deterministic bootstrap seed derived from the exact input bytes.

## 3. Measurement definitions

### Pairwise agreement

For each rater pair, compare scores on the same items and the same criterion. Calculate Pearson and Spearman correlations per criterion, then average valid criterion correlations with equal criterion weight for that rater pair.

This deliberately avoids flattening every criterion into one long vector before correlation: different mean score levels between criteria can otherwise masquerade as within-criterion rater agreement. The report preserves rater-pair and criterion-comparison denominators, undefined zero-variance counts, and both Pearson/Spearman summaries.

Report the matched triple:
- judge–judge;
- judge–human;
- human–human.

A correlation does not establish calibration, absolute agreement, construct validity or annotator independence. A future layer should consider ordinal/weighted agreement and appropriate IRT or mixed-effects models.

### Score-spread ratios

Two distinct ratios are reported:
- **Pooled-rater spread ratio:** within-criterion standard deviation of all individual judge ratings divided by the corresponding human-rater standard deviation.
- **Judge panel-mean spread ratio:** within-criterion standard deviation of item-level panel means for judges divided by the same statistic for humans.

Each criterion is centered before pooling to prevent criterion mean offsets from driving the spread estimate. These ratios are diagnostic; there is no universal target ratio.

### Principal-axis angle

For each panel, compute the covariance matrix across the declared criterion dimensions using item-level panel means. Compute the leading eigenvector and principal angle:

[
	heta = cos^{-1}(|v_J cdot v_H|).
]

The absolute dot product makes the angle invariant to eigenvector sign. If a panel has zero variance or a non-identifiable leading eigenvector (top eigenvalue not separated from the second), the angle is null, not an arbitrary number.

This is a *leading-axis* diagnostic, not a full subspace equivalence test. A later release should add multiple principal angles between the first (k) judge/human score subspaces, with a preregistered (k) and adequate item/criterion counts.

### Effective rank

For nonnegative covariance eigenvalues (lambda_i), normalize (p_i=lambda_i/sum_jlambda_j) and report:

[
r_{mathrm{eff}} = exp(-sum_i p_ilog p_i).
]

Effective rank summarizes how many criterion directions materially contribute to the panel's score variation. The declared rubric criterion count is not an empirical dimensionality estimate.

### Uncertainty

The v1 bootstrap resamples **items as clusters**, preserving each sampled item's full judge and human panels and criterion vectors. It reports 95% percentile intervals for the spread ratio and leading-axis angle. If the item count is below the preregistered minimum, or no valid resamples exist, intervals are null with an explicit reason.

A bootstrap interval over a convenience/fixed score matrix does not by itself estimate population generalization. Generalized inference requires a declared sampling frame, task-family coverage and a suitable hierarchical model.

## 4. Synthetic controls

The test suite includes:
- compressed judge score range and an approximately orthogonal leading axis;
- aligned judge/human panels;
- a constant-scoring judge, for which correlations and axis angle must be null;
- deterministic item-cluster bootstrap;
- an undersized dataset with null intervals;
- task-family strata below the configured minimum;
- duplicate items/raters, cross-panel identity reuse, malformed score dimensions, out-of-range values, booleans, unknown fields and malformed purpose values.

These are protocol-level synthetic controls. They are not model outputs, blinded human judgments, empirical judge performance or benchmark scoring results.

## 5. Qualification targets and comparison

Targets are engineering properties, not empirical SOTA thresholds:
- 100% rejection of malformed/duplicate/out-of-range controls;
- exact pair and criterion denominator accounting;
- zero-variance correlation represented as null with explicit undefined counts;
- deterministic output and bootstrap resampling for identical bytes;
- item-cluster rather than rating-level resampling;
- no generalized-accuracy or construct-validity claim from a fixed score matrix.

The competitive comparison should reproduce the published study's score-spread, principal-axis, judge–judge / judge–human / human–human analyses where compatible, with matched task sets and blinded held-out human panels. The report should present distributions and uncertainty, not copy one paper's values as universal thresholds.

A meaningful improvement over current practice would combine:
1. high judge–human alignment;
2. score-spread and effective-rank behavior consistent with held-out human panels;
3. robustness to rubric paraphrase, surface-cue attacks and adversarial candidates;
4. calibration and reliable abstention;
5. quality-versus-cost comparisons under matched budgets.

No one axis can compensate for a critical evidence or correctness failure.

## 6. Explicit limitations and next steps

The v1 tool provides descriptive score-space diagnostics. It does not:
- authenticate the raters or prove their independence;
- establish the human panel is a valid construct reference;
- infer causality or generalized accuracy;
- fit IRT, GLMM or a latent-variable preference model;
- correct annotator severity, rubric drift or selection bias;
- certify the evaluator or candidate system.

Next, after independent review of v1, add a separately versioned IRT analysis with preregistered model structure, rater/item/criterion effects, identifiability constraints, missingness model, held-out validation and uncertainty. Do not retrofit that model into the descriptive v1 score geometry report.
