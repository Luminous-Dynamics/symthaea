# Symthaea Aesthetic — Empirical and Governance Patch Bundles

This series builds on the **Aesthetic Next-Wave Patch Series**. It closes the gap between a well-typed critic and a critic that can support reproducible modality integration, human studies, explainable decisions, and bounded release claims.

It intentionally does **not** add another universal beauty formula. The new layers declare measurement support, expose disagreement, preserve exact policy provenance, select grounded preference evidence efficiently, and fail closed when empirical evidence is missing.

## Required baseline

Apply the complete Aesthetic Truth and Aesthetic Next-Wave series first, or use the previously supplied `symthaea-aesthetic-next-wave-patched.tar.gz` tree.

The patch files are ordinary Git mail patches. They may apply to an equivalent source tree even when local commit IDs differ.

## Bundle 1 — Modality Evidence Adapters

Commits:

1. `feat: add auditable modality extraction contracts`
2. `feat: fuse independent extractors with disagreement diagnostics`
3. `feat: add conservative music visual and text adapters`

Adds:

- `EvidenceExtractor<A>` for downstream modality integrations;
- sparse `EvidenceFeatureSet` and per-feature `FeatureObservation` support;
- versioned `ExtractionReport` with structured issues;
- neutral values with zero reliability for unsupported channels;
- reliability-weighted `EvidenceFusion`;
- per-channel weighted disagreement and sample accounting;
- explicit modality mismatch and zero-weight failures;
- conservative normalized reference adapters for music, visual art, and text;
- calibrated Birkhoff derivation without hard saturation.

Important boundary:

- the reference adapters use only artifact-derived measurements;
- creator consciousness, reward state, and active harmony state are not treated as independent proof that the artifact is beautiful;
- the adapters consume normalized analyzer outputs and do not pretend to replace raw audio, image, score, or language analysis.

## Bundle 2 — Explainable Critic and Counterfactuals

Commits:

1. `feat: retain policy provenance for assessment replay`
2. `feat: add additive utility explanations and replay checks`
3. `feat: add counterfactual sensitivity and evidence priorities`

Adds:

- exact `AestheticPolicy` provenance on new assessments;
- backward-compatible policy reconstruction for older serialized assessments;
- additive utility contributions for intrinsic, context, novelty, and preference evidence;
- explicit supported and requested weights;
- missing-preference and low-grounding disclosures;
- utility replay mismatch detection;
- local counterfactual sensitivity across nine assessment variables;
- ranked recommendations for the next measurement likely to reduce policy uncertainty.

Counterfactuals expose policy sensitivity. They do not claim that changing one metric will causally improve human aesthetic response.

## Bundle 3 — Study Design and Active Learning

Commits:

1. `feat: add leakage-aware preference study splits`
2. `feat: select informative pairwise preference queries`
3. `feat: report preference calibration by context and provenance`

Adds:

- deterministic train/holdout splits by pseudonymous rater or artifact;
- rater-leakage warnings for anonymous observations;
- exclusion accounting for cross-partition artifact pairs;
- held-out-only model training and calibration evaluation;
- bounded active pair selection;
- exclusion of already judged pairs;
- acquisition scoring from model uncertainty, feature distance, and support gaps;
- reuse limits to prevent a few artifacts from dominating a study;
- segmented calibration by modality, context, or preference source;
- worst-segment calibration lookup.

For publication-quality claims, prefer rater-separated evaluation when generalizing to new people and artifact-separated evaluation when generalizing to unseen works. Report both when feasible.

## Bundle 4 — Evidence Governance and Release Gates

Commits:

1. `feat: add versioned aesthetic evidence manifests`
2. `feat: add fail-closed aesthetic release gates`
3. `fix: harden fusion selection and evidence graph boundaries`
4. `docs: document empirical aesthetic patch bundles`

Adds:

- versioned `AestheticEvidenceManifest` graphs;
- evidence records for extractors, datasets, protocols, ledgers, models, policies, diagnostics, and calibration reports;
- caller-supplied digests, licenses, parent evidence, and limitations;
- claims that must reference registered evidence;
- atomic manifest persistence;
- cycle detection across evidence-parent graphs;
- `AestheticEvidenceSummary` collection;
- development and publication gate presets;
- gates for extractor validity, confidence, coverage, measured disagreement, held-out support, Brier score, ECE, worst-segment ECE, grounded preference fraction, contradiction rate, policy provenance, and utility replay;
- `Pass`, `Conditional`, and `Fail` outcomes with metric-level findings.

The release gate is evidence governance, not a declaration that aesthetics has been solved. Passing means the configured empirical and audit thresholds were met.

## Recommended downstream flow

```text
raw modality artifact
    -> downstream analyzer
    -> normalized modality feature frame
    -> EvidenceExtractor
    -> ExtractionReport

independent extractors
    -> EvidenceFusion
    -> ArtifactEvidence + disagreement diagnostics

artifact evidence + context + novelty + preference
    -> AestheticAssessment::evaluate
    -> AestheticAssessment::explain
    -> counterfactuals / measurement_priorities

PreferenceStudyLedger
    -> StudySplit::create
    -> training ledger -> TasteModel
    -> holdout ledger -> calibration
    -> segmented calibration
    -> active pair query selection

extraction + held-out + explanation evidence
    -> AestheticEvidenceSummary
    -> AestheticReleaseCriteria
    -> release report
    -> AestheticEvidenceManifest claim graph
```

## Compatibility and migration

Existing constructors and evaluation methods remain available. `AestheticAssessment` adds an optional serialized `policy` field. Older assessments deserialize with `policy = None`; `effective_policy()` reconstructs the mode default and explanations emit `POLICY_RECONSTRUCTED`. Downstream code using direct `AestheticAssessment` struct literals must add the new field or switch to `AestheticAssessment::evaluate`.

New code should:

- persist `ExtractionReport`, not only `AestheticScore`;
- retain exact policy provenance;
- use held-out study ledgers for calibration claims;
- record manifests and release reports alongside published metrics;
- treat reference adapters as conservative starting points requiring modality-specific calibration.

## Required parent-workspace verification

The generation environment does not contain `cargo` or `rustc`. Run:

```text
cargo fmt --all -- --check
cargo test -p symthaea-aesthetic
cargo clippy -p symthaea-aesthetic --all-targets --all-features -- -D warnings
cargo test --workspace
```

Also run downstream integration tests for Muse, Canvas, voice, poetry, and Symtropy before enabling aesthetic feedback in production loops.

## Highest-value next research work

- raw-signal extractors with published feature definitions and domain limits;
- cross-cultural and expertise-stratified pairwise studies;
- confidence intervals or Bayesian posteriors rather than point calibration alone;
- artifact-family and creator-group leakage audits;
- causal intervention studies proving that critic-guided revisions improve blind human preference;
- drift monitoring when generators, extractors, or audiences change.
