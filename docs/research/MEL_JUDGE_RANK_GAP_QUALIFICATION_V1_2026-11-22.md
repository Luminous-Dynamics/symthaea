# MEL Rank-Gap-Aware Judge Qualification v1

**Date:** 22 November 2026  
**Programme:** Melothaea / MEL-BENCH  
**Implementation:** \`tools/mel-judge-gap/\`  
**Status:** deterministic reference evaluator with synthetic known-answer fixture; no live judge or human-study result.

## Why this is a distinct workstream

MEL-PREF-ASSAY-001 (PR #13212) audits human preference-study presentation integrity: precommitted order, no repeated human exposure, counterbalanced strata, and identical-output controls. This new evaluator addresses a different protocol: a machine judge can be called twice on the same pair under independently isolated requests, once in each display orientation. It then tests whether that judge recovers an independent human ranking.

The distinction matters. Replaying a pair to the same human listener can contaminate judgment through memory; paired-order replay is instead an explicit machine-judge stress test. Human panels remain the independent reference, not the machine judge's self-generated labels.

## Current field and methodological findings

### Music leaderboards already use strong pairwise methods

Artificial Analysis' published Music Arena methodology says its human preference leaderboard is built from blind pairwise votes, Bradley–Terry maximum-likelihood estimation, 95% confidence intervals, standardized loudness/delivery, and separate vocal and instrumental leaderboards. Its published methodology describes validated private evaluators and a minimum listening requirement. This is a strong baseline; Melothaea should not claim novelty for simply using pairwise votes or Bradley–Terry.

MIREX 2026's CMI-RewardBench task evaluates music reward models against preference data. Its core set contains 249 examples; the published results include CloudEar at 73.9%, the CMI-RM baseline at 65.9%, and a 59.0% always-choose-B baseline. A SYSTASYS result also changes from 39.8% to 60.2% when its output orientation is flipped. Those figures justify orientation audits, but do not establish that every evaluator or human panel has the same bias.

References:
- Artificial Analysis, [Music Generation Benchmarking Methodology](https://artificialanalysis.ai/methodology/music).
- MIREX 2026, [Music Evaluation via CMI-RewardBench results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results).
- Ma et al., [CMI-RewardBench, ICML 2026](https://proceedings.mlr.press/v306/ma26am.html).

### Pairwise position invariance is not the same thing as ranking quality

The October 2026 ACL GroundLM paper *From Position Bias to Position-Stable Agreement* evaluates the same pair with reversed order and retains a directional verdict only when both orders select the same response. Across four judge models it reports strong second-position preference in its tested short A/B prompts. This is a useful protocol reference, but it concerns text LLM judging, not music preference.

The October 2026 EMNLP paper *Pair Difficulty Matters* adds an important qualification. Across the two human-rated text corpora it studies, aggregate consistency and position-bias proxies do not reliably track human-ranking recovery. Its reported correlation between far-pair consistency and ranking correlation is much stronger than for close-pair consistency; the authors argue that close-pair disagreement is often expected under the Bradley–Terry geometry. This finding also comes from text evaluation and must be validated in music rather than assumed to transfer.

References:
- Liu, Lin and Hao, [From Position Bias to Position-Stable Agreement](https://aclanthology.org/2026.groundlm-1.61/).
- Brocai and Becker, [Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency](https://aclanthology.org/2026.emnlp-main.2676/).

## Paradigm shift: measure ranking recovery, not just judge obedience

A judge can be perfectly stable and consistently wrong. A judge can be unstable on nearly tied items yet recover the global human ranking well. Therefore:

1. Use paired order swaps to detect identity instability.
2. Use a separate human-derived reference to determine whether the pair is clearly directional.
3. Report conditional and effective accuracy by human-reference gap.
4. Fit the machine judge's pairwise outcomes into an item ranking and compare that ranking directly with the human-derived ranking.
5. Never let the judge grade its own reference labels or authorize creative edits.

The reference uses a pair-specific standard error for the difference between two human Bradley–Terry item strengths. This allows the confidence bound to incorporate covariance from the shared reference fit instead of treating item estimates as independent. The evaluator excludes a pair from directional gold scoring when its conservative lower gap bound does not exceed the preregistered minimum.

## Implemented evaluator outputs

- committed, deterministic first/second display-order sequence per pair;
- validation that the second presentation reverses the first;
- stable directional, stable tie, order-inconsistent, and incomplete outcomes;
- left-choice rate as a diagnostic, not a standalone judge-quality score;
- near / medium / far reference-gap bins with coverage, conditional accuracy, and effective accuracy;
- ridge-regularized Bradley–Terry item skills from stable directional decisions;
- Spearman rank correlation against the independent human reference;
- fail-closed omission of the fitted ranking when the evidence graph is disconnected.

The fixture is five synthetic items, ten complete unordered pairs, and twenty order-swapped presentations. It proves only deterministic conformance and the known-answer computation.

## Prospective music-study targets

These targets are not achieved results. Preregister them against a rights-cleared, musician-annotated, item-held-out test set:

| Dimension | First serious gate | Stretch target |
|---|---:|---:|
| Order assignment | 100% exact replay; zero mismatches | External timestamp/signature receipt for every seed commitment |
| Far-gap swap stability | ≥95% on directionally clear pairs | ≥98%, with subgroup intervals |
| Far-gap effective accuracy | ≥90% | ≥95% |
| Human ranking recovery | Spearman ρ ≥0.85; lower 95% bootstrap bound ≥0.80 | ρ ≥0.90; lower bound ≥0.85 |
| Directional-pair coverage | ≥90% | ≥95% |
| Position-choice diagnostic | Report with interval and by judge/version; no hidden exclusions | Balanced-control rate within preregistered equivalence bounds |
| Gap reporting | All bins, counts, intervals and abstentions published | Replicated across at least three independent musician cohorts |

Use at least 100 held-out items and enough comparisons to populate every gap stratum (a planning floor of 500 unique pairs, with both order presentations, is reasonable for an initial study). Use bootstrap resampling over items and raters or cluster-aware hierarchical models; do not treat all pair presentations as independent observations. Report genre, vocal/instrumental mode, expertise, cultural familiarity and creator/artifact-family splits separately. Do not pool the two Artificial Analysis modalities into a single leaderboard.

## How to compete with industry without making an invalid comparison

- Reproduce official CMI-RewardBench splits and scoring before presenting any side-by-side percentage.
- Run an independent musician holdout with an independently estimated human Bradley–Terry reference.
- Match prompt, modality, generation budget, normalization, retry policy and audio delivery when comparing generators.
- Report the standard global ranking, then publish order stability and gap-conditional ranking recovery as additional judge diagnostics.
- Report uncertainty, coverage, abstention and subgroup results. A 90% score on a different sample is not automatically better than a 74% score on the MIREX core set.
- Keep generation quality, evaluator quality, musical agency, latency and rights/consent as separate axes. No single Elo can stand in for the whole collaborator.

## Claim ceiling and remaining work

This crate uses synthetic strengths and synthetic responses. It does not establish live judge quality, human agreement, audio quality, generalization to genres/cultures, or improvement over AA-Music or CMI-RewardBench. It does not authenticate API execution or prove context isolation between paired calls. A production adapter must capture actual model/provider/version, request IDs, prompt/response bytes, parameter settings, independent call contexts, and rights-cleared audio digests. The human reference must be independently collected and its uncertainty/covariance matrix independently estimated.
