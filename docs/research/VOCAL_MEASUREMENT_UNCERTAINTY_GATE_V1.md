# VIA-001 — Vocal Measurement Uncertainty Gate

**Date:** 10 November 2026  
**Status:** standalone interval-decision contract; synthetic tests only  
**Owner:** [Issue #11456](https://github.com/Luminous-Dynamics/symthaea/issues/11456)  
**Scope:** uncertainty-aware evaluation of requested vocal-control responses and collateral acoustic changes

## Research question

When an independent acoustic analyzer reports an interval rather than a single value, can Muse decide conservatively whether a requested control was realized without allowing uncertainty to become a hidden pass?

## Paradigm shift: from thresholding estimates to qualifying evidence

A point estimate is not the observed truth. It is an estimate from a finite, noisy, model-dependent measurement process.

Suppose a requested vibrato-depth intervention has an accepted target response of 8–12 cents. An analyzer returns a point estimate of 10 cents. If its uncertainty interval is 6–14 cents, the point estimate looks compliant but the available evidence does not establish that the true response lies inside the accepted region.

VIA-001 therefore returns one of three outcomes for each metric:

- **VERIFIED_COMPLIANT:** the entire supplied uncertainty interval satisfies the predeclared constraint.
- **VERIFIED_NONCOMPLIANT:** the entire interval is disjoint from the permitted region.
- **INDETERMINATE:** the interval overlaps a decision boundary and cannot establish compliance or definite failure.

The overall control trial is verified compliant only when the target and every collateral metric are verified compliant. One definite metric violation makes the trial verified noncompliant. Otherwise, any unresolved metric makes the trial indeterminate.

An indeterminate result is not a failure of the musician or renderer. It is a statement that the measurement evidence is insufficient for a decision. It must never count as a pass.

## Executable contract

The standalone Rust library at tools/vocal-interval-assurance consumes one interval per required metric. It requires:

- a signed, non-zero target response interval;
- an explicit absolute-delta tolerance for each collateral metric;
- exact metric coverage with no missing, extra or duplicate observations;
- finite, ordered uncertainty bounds and exact unit matches;
- common source-case, trial, input-manifest and interval-family bindings;
- a declared family-wise coverage meeting the requested minimum;
- a non-empty uncertainty method and calibration receipt for every metric.

The report preserves the metric-level decision, supplied interval, unit, uncertainty method and calibration reference. It emits no weighted composite.

The input-manifest digest is treated as an opaque identity string. The library does not recompute hashes or authenticate the underlying artifacts.

## Statistical protocol for producing the intervals

This library deliberately does not create uncertainty intervals. Doing so would couple estimator policy to decision policy and could let the evaluator silently choose assumptions favorable to its own result.

For a real Muse experiment:

1. Freeze source take, score, pronunciation plan, renderer/voicebank artifacts, preprocessing, metric definitions and analyzer versions.
2. Separate calibration from confirmatory source cases by singer, song and recording session.
3. Choose the inferential sampling unit before analysis. Overlapping frames and repeated crops of one take are not independent singer/song trials.
4. Derive simultaneous intervals for the target and all collateral metrics. A justified family-wise correction or joint/max-statistic cluster bootstrap is preferable to unrelated per-metric intervals whose family coverage is unknown.
5. Record interval method, source split, resampling unit, calibration receipt, family ID and declared simultaneous coverage.
6. Freeze those records before opening the confirmatory results.
7. Publish the three outcome counts and indeterminate fraction, along with per-metric error and coverage diagnostics.

The crate validates the structure and consistency of these claims. It cannot prove interval coverage, independence, representativeness, calibration quality or evaluator independence.

## Relation to existing work

- **VCRB-001 (#11401):** supplies one-factor control-response measurements and target/collateral metric intent. VIA adds conservative uncertainty-aware decisions.
- **VNOC-001 (#11420):** derives no-op collateral tolerances from repeated measurements. A tolerance bound and a confidence interval for an acoustic estimate are different statistical objects; neither substitutes for the other.
- **Observation integration (#11402):** is the planned producer of versioned acoustic observations. The adapter can later attach the uncertainty interval and its provenance.
- **Independent F0 audit (#11424) and qualification (#11425):** keep score pitch out of the primary estimator and require held-out vocal references. VIA can gate those estimates only after an upstream method supplies meaningful intervals.

## Competitive and research context

Hume's September 2026 Voice Controllability Leaderboard uses tens of thousands of human judgments for 17 voice models and forced-choice comparisons of contrasting instructions. Its central lesson for Muse is to evaluate directed changes independently of the generator and avoid making an LLM judge the sole authority.

MusicJudge (Interspeech 2026) couples block-aligned lyric correctness with pitch/rhythm fidelity. It informs a separate multimodal assessment track, but it does not supply a statistical uncertainty gate for collateral acoustic effects.

The current pitch-detection benchmark v2.1 compares 19 monophonic trackers across 10 corpora and eight noisy/reverberant/microphone variants. It reports SwiftF0 F1@50 cents of 0.781 (95% CI 0.768–0.795), RMVPE 0.768 (0.752–0.783), and FCPE 0.728 (0.712–0.742). This is a mixed monophonic-pitch benchmark, not a singing-only qualification. It should guide the reference-tracker shortlist for #11425, not be quoted as Muse singing accuracy.

The AES paper *Singing Voice Synthesis via Guided Diffusion* (10 November 2026) studies external phoneme and pitch classifier guidance and gradient conflict projection. This is a synthesis experiment track in the vocal-guidance lab, separate from measurement assurance.

## Qualification targets

These are contract targets, not measured outcomes:

| Measure | Initial target |
|---|---:|
| Synthetic positive/negative/boundary decision accuracy | 100% |
| Missing/extra/duplicate/stale evidence detection | 100% |
| Uncertainty/provenance fields on scored trials | 100% |
| Indeterminate outcomes silently accepted as pass | 0 |
| Composite score hiding a failed metric | 0 |

Real-data goals should be reported separately:
- empirical coverage of uncertainty intervals on held-out, independently annotated vocal references;
- calibration error/reliability by register, technique, SNR and phonetic context;
- false-compliant and false-noncompliant rates;
- indeterminate rate and remeasurement burden;
- singer/song/session-clustered confidence intervals;
- runtime and real-time factor.

Do not set a single target for empirical interval coverage until the reference protocol and acceptable error costs are preregistered. A nominal 95% interval does not guarantee 95% empirical coverage on unseen singing data.

## Explicit limitations

VIA is not a pitch tracker, audio analyzer, uncertainty estimator, conformal predictor, confidence-interval engine, renderer, human preference model or legal/rights system. Its metadata checks do not prove that upstream calibration receipts are truthful or that a supplied interval has the declared coverage.

The next evidence milestone remains a licensed/consented vocal reference set with manually reviewed F0/phoneme annotations and matched baselines. Synthetic tests are not a substitute.
