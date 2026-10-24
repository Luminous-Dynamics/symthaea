# Symthaea Hi-Fi Research Agenda: Perception, Robustness and Active Experimentation

**Status:** research programme and falsifiable benchmark plan. No listener-preference model or Luminous hardware is claimed to be validated by this document.  
**Research snapshot:** 2026-10-17.  
**Tracker:** [HIFI-RESEARCH-001 #7888](https://github.com/Luminous-Dynamics/symthaea/issues/7888).  
**Product family:** [OPEN-AUDIO-000 #7826](https://github.com/Luminous-Dynamics/symthaea/issues/7826).  
**Evidence contract:** [OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md](OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md).
**Benchmark protocol:** [HIFI_BENCHMARK_PROTOCOL_2026-10-17.md](HIFI_BENCHMARK_PROTOCOL_2026-10-17.md) and [HIFI-BENCHMARK-001 #7903](https://github.com/Luminous-Dynamics/symthaea/issues/7903).

## Executive thesis

Symthaea should research not only how to produce smoother response curves, but how to design audio systems that are measurably accurate, physically robust, predictably preferred by real listeners, and honest about uncertainty.

The open research opportunity is the full loop:

**predict → choose the most informative test → measure physical hardware → model uncertainty → conduct controlled listening → validate on unseen products/listeners → publish raw evidence and failures.**

The target is not a neural network that says a speaker scores 8.9/10. The target is a model whose predictions survive unseen products, a declared measurement protocol, different listeners and independent experiments—and whose uncertainty tells us when not to trust it.

## Research already says we should not overfit a single target

### Headphone preference is not a solved single-curve problem

RTINGS' article updated March 19, 2026 reports a double-blind study with 35 participants, five curves and two short music excerpts. Four curves were statistically tied in that study; an intentionally low-bass anchor was less preferred. The authors explicitly state that the study is small and not a definitive characterization of listener preference. Its useful lesson is that small spectral differences can be audible while a unique universal target remains unjustified. Source: [One Curve Doesn't Fit All](https://www.rtings.com/headphones/learn/research/frequency-response-1-8).

A 2025 AES paper by Olive and Clark describes how a preferred target must be translated for different headphone fixtures because acoustic transfer impedance differs; measurements and targets from a GRAS 45CA-10 and B&K 5128 cannot simply be treated as identical. Source: [A Preferred Headphone Target Curve Defined for the B&K 5128 and GRAS45CA-10](https://aes.org/publications/elibrary/elibrary-page/?id=23068).

**Implication:** preserve fixture-specific calibration, model a population distribution of preferences and fit outcomes, and test whether personalization helps held-out listeners. Do not simply average several target curves into a fictitious universal truth.

### Loudspeaker preference models are valuable baselines, not infallible oracles

An AES loudspeaker preference model used anechoic measurements for 70 loudspeakers evaluated across 19 listening tests. It is a serious baseline to study and reproduce, not a reason to assume any score generalizes to a different product family, room, measurement chain or listening population. Source: [Olive, “A Multiple Regression Model for Predicting Loudspeaker Preference,” Part II](https://aes.org/publications/elibrary/elibrary-page/?id=12847).

Spinorama documents workflows that compute angular-response plots and preference estimates from speaker measurement data. This gives us useful established objective baselines before inventing a more complex Symthaea score. Source: [Computing a spinorama from your own measurements](https://www.spinorama.org/docs/08_compute_spin.html).

**Implication:** benchmark on product-held-out data, compare to simple measurements and published-model baselines, publish confidence intervals and prediction failures, and resist leakage from multiple captures of the same product into both train and test sets.

### Distortion needs a perceptual interpretation

RTINGS' March 19, 2026 follow-up reports controlled headphone-distortion listening tests: for the tested models and conditions, thresholds were observed at 104–112 dBA SPL, and most listeners did not reliably discriminate isolated distortion in ABX tests using recordings standardized at 94 dBA SPL. The work covered six headphones, with 13 participants in threshold tests and nine in the similarity/ABX study; these are useful findings, not a universal audibility law. Source: [The Surprising Truth About Headphone Distortion](https://www.rtings.com/headphones/learn/research/distortion-audibility-listening-tests).

**Implication:** quantify distortion physically, then test detection and preference separately. Don't assume a lower THD number is always audible or preferred, and don't substitute a score developed for electronic signal paths for a transducer-specific validation.

### Subjective evaluation has established methods

ITU-R BS.1116-3 is in force for controlled assessment of small audio impairments; ITU-R BS.1534-3 is in force for intermediate-quality subjective assessment. These are starting points for appropriate experimental design—not permission to claim standards compliance from a vaguely similar blind test. Sources: [BS.1116](https://www.itu.int/rec/R-REC-BS.1116) and [BS.1534](https://www.itu.int/rec/R-REC-BS.1534/en).

Bayesian experimental design is also a meaningful method precedent: adaptive experiments select the next measurement based on what prior results reveal, but benchmark studies show that model and acquisition choices materially affect efficiency and robustness. Source: [Bayesian experimental design in production engineering (2025)](https://www.frontiersin.org/journals/manufacturing-technology/articles/10.3389/fmtec.2025.1614335/full).

## The ten highest-value research tracks

### R1 — Perceptual predictor validity

**Question:** Which measurable features predict preference on genuinely unseen audio products?

Start with explainable baselines before learned models:
- headphones: response deviation under a fixture-specific target, peaks/dips, channel mismatch, fit variation, distortion by level and sensitivity;
- speakers: on-axis/listening-window response, early-reflection and sound-power-related metrics where available, directivity continuity, bass extension, distortion/compression, maximum linear output;
- electronics: electrical measurements first; test audible/detectable claims separately rather than treating tiny electrical differences as automatically important.

Compare the simplest baseline, known published predictors and any Symthaea model on identical held-out folds. Prefer a small model that generalizes over a sophisticated model that only memorizes the test corpus.

### R2 — Preference heterogeneity and conditional targets

**Question:** Can we predict a range of valid preferences for different listeners, uses and content instead of forcing one target?

Model listener, device, music excerpt and context as distinct effects. Separate the population-average prediction from uncertainty and potential listener-specific adaptations. Collect preferences for controlled EQ/profile alternatives with randomization, safe levels and counterbalanced order.

There is evidence worth testing rather than treating as a universal law: an AES 2025 method-of-adjustment study of 36 mostly trained listeners found four preference segments in bass/treble adjustment across three music tracks. The segment ranges were broad enough to motivate controlled low-dimensional profile adaptation, but the cohort and stimuli were limited. Source: [The Preferred In-Ear Headphone Target Curve Based on Method of Adjustments](https://aes.org/publications/elibrary/elibrary-page/?id=22928).

Candidate outcomes include preference probability, pairwise choice and expected utility with uncertainty. Personalization is only useful if it improves predictions for a held-out listener after a small amount of preference input. We should compare that against a static target and a transparent user-controlled bass/treble profile.

### R3 — Fixture transfer and measurement uncertainty

**Question:** Which findings survive a change of measurement fixture?

Characterize calibration identity, coupler/ear simulator geometry, reseating, pad/earpiece condition, orientation, microphone position and measurement processing. Where two fixtures are available, measure a matched subset with a declared bridge procedure.

Never apply one fixture's headphone target to another by visual alignment alone. A transfer model must report its domain, error distribution and products/listeners for which it was not validated. For DIY couplers, describe the result as fixture-specific until the appropriate transfer is demonstrated.

### R4 — Nonlinear distortion audibility

**Question:** At realistic safe playback levels, when do transducer nonlinearities become detectable, objectionable or preference-relevant?

Treat these as three distinct endpoints:
1. physical distortion magnitude by frequency and drive level;
2. detectability under a controlled comparison;
3. preference/annoyance conditional on detectability.

Use preplanned, randomized and safely level-limited tests. Match response or otherwise isolate the distortion variable so participants are not merely identifying a frequency-response difference. Evaluate single-tone metrics and music-like stimuli without assuming either is a complete perceptual proxy. Begin with replication on selected transducers before expanding the study.

### R5 — Robust acoustic design and manufacturing tolerance

**Question:** Which design is good across many plausible conditions, not just in a nominal simulation?

For headphones, perturb fit/reseat, pad seal, clamp/position, pad aging and driver sample. For speakers, perturb angular position, listening position, room assumptions, crossover component tolerance and driver-lot variation. For electronics, perturb load, temperature, supply tolerance and operating level.

Report median performance, lower-tail performance, worst declared operating case, distribution width and out-of-spec probability where data support it. Optimizer objectives should penalize fragility and unmeasured assumptions, not only nominal response error.

### R6 — Active experiment selection

**Question:** Given a limited budget, which next measurement reduces consequential uncertainty fastest?

Compare at least:
- a deterministic fixed plan;
- cheapest-first;
- uncertainty reduction per cost;
- expected improvement or Pareto-frontier relevance;
- random selection with fixed seed;
- an oracle upper bound in synthetic known-truth tests only.

Count physical setup, repeat measurements, calibration checks, analysis, rejected runs and deferred confirmatory validation in total cost. A policy must not win by selecting tests whose unresolved validation cost is hidden outside the budget.

### R7 — DSP/physical co-design and room robustness

**Question:** Which deficits can EQ safely fix, and which require a physical change?

Evaluate uncorrected and corrected response together. Include filter headroom, boost limits, clipping, output compression, seat-to-seat variation and off-axis behavior. Separate reducible tonal errors from directivity nulls, severe cancellations, distortion, thermal limits and inadequate displacement. A room-correction algorithm must not silently “fix” a null by applying a dangerous or ineffective boost.

For room work, optimize a declared set or distribution of seats rather than one microphone point. Publish the positions and correction limits, and keep an offline, inspectable filter profile.

### R8 — Listening-study methodology and replication

**Question:** Does an observed preference survive bias controls, appropriate sample size and independent replication?

Preregister hypotheses, primary endpoints, analysis and exclusion rules. Randomize product identity and order, level-match at a defined procedure, use safe playback levels and report participant/context coverage. Determine sample size from a pilot and a stated meaningful effect—not a desired headline. Report effect sizes, intervals and null results.

Use ITU-R BS.1116 / BS.1534 as applicable reference methods. Don't call a loosely structured listening demo a compliant test or conclusive study. A result can apply to one tested cohort without establishing a universal preference.

### R9 — Electronics: distinguish signal fidelity from perceived value

**Question:** Which amplifier/DAC differences are measurable; which are detectable under relevant conditions; and which are actually preferred?

Keep the evidence layers separate: response/noise/distortion/load measurements, blinded detectability, then preference if detectability is demonstrated. Establish audibility bounds for declared loads, output levels and signal conditions. Use a double-blind or otherwise bias-controlled test that avoids clipping/level mismatch and keeps source/device switching behavior from identifying the condition.

The research goal isn't to prove all electronics identical; it is to stop attributing an audible benefit to an electrical difference without evidence and to identify where the design needs more headroom, lower noise or better stability.

### R10 — Open data, provenance and reproducible evaluation

**Question:** Can a new researcher reproduce a published result from the declared raw data and software revision?

Every dataset needs explicit permission/license and provenance, device/build identifiers, fixture and calibration identity, raw measurements, processing version, uncertainty and exclusions. The Open Audio Evidence manifest and Rust integrity checker are the starting infrastructure. The checker is not a physical calibration system or a complete JSON Schema implementation; keep those as separate gates.

Never train on data with incompatible fixture metadata and pretend the model has learned a universal scale. Don't include third-party files in public artifacts until redistribution rights are clear. Open raw data, where licensable, and document unavailable or restricted inputs honestly.

## Benchmark programme

All numeric targets below are **proposed project gates**, not published-standard thresholds or achieved results. Establish feasibility first, preregister gates and hold out data before any confirmatory evaluation.

| Benchmark | Provisional success criterion |
|---|---|
| Measurement integrity | 100% of used samples have the required identity/protocol metadata; raw-artifact hashes match; failed controls cannot enter the accepted set silently |
| Predictor generalization | Evaluate on product-grouped held-out data; report Spearman/Kendall ranking, MAE/calibration where relevant, uncertainty intervals and error by product class; no acceptance based on random capture-level splits |
| Baseline value | Initial research ambition: ≥10% relative reduction in held-out pairwise log loss vs the strongest transparent baseline, with a listener-clustered uncertainty interval supporting improvement and no material calibration degradation; freeze the final practical threshold after pilot/power simulation but before confirmation |
| Preference model calibration | Report reliability/calibration plots and Brier/log loss for probabilistic choices; never report only rank correlation |
| Personalization uplift | A target/profile adapted from limited user feedback must improve held-out choices for unseen excerpts/products compared with the best static baseline; include users for whom it does not help |
| Fixture transfer | Report per-band transfer error and confidence intervals on held-out devices; never label the transfer universal if results remain device/fixture dependent |
| Distortion research | Measure at multiple frequencies/levels; report detection effect size and uncertainty separately from preference; use adequate trials and safe sound levels |
| Robust design | Report nominal and lower-tail performance under a declared perturbation distribution; show tolerance/fit sensitivity and failed cases |
| Active learning | On the same synthetic ground-truth tasks and budget, report cost to reach predeclared uncertainty/decision quality; compare fixed, cheapest-first, uncertainty, frontier, seeded random and synthetic oracle |
| Human study validity | Preregistered outcome, randomization, level matching, exclusion rule, sample-size rationale, confidence interval/effect size, null-result publication and reproducible analysis |
| Reproducibility | A second researcher can validate artifacts and reproduce the analysis from a revision-pinned bundle; missing files or changed bytes cause failure |
| Claims discipline | Separate “measured,” “audibly detected,” “preferred,” “robust,” “repairable,” and “standards-compliant”; each needs its own evidence |

Do not make a target such as “correlation ≥0.9” a universal gate without examining sample size, noise, baseline difficulty and domain shift. High in-sample correlation is cheap to obtain; held-out, calibrated and uncertainty-aware predictions are the hard benchmark.

## First concrete experiment: HP-PREF-001

**Question:** Does a preference distribution model outperform one fixed response target for predicting headphone-profile preference on new listeners and music excerpts?

The detailed preregistration scaffold is [HP_PREF_001_PREREGISTRATION_2026-10-17.md](HP_PREF_001_PREREGISTRATION_2026-10-17.md), tracked by [Issue #7889](https://github.com/Luminous-Dynamics/symthaea/issues/7889). It freezes the split rules, endpoint/analysis plan, masking, level-match procedure, human-study safeguards and stop/deployment gates; it contains no participant data or claimed result.

1. Freeze a small set of documented, safe EQ profiles. Include an unmodified baseline and profiles chosen to span a reasonable target space, not only profiles we expect to win.
2. Use randomized, double-blind, level-matched comparisons. A short pilot identifies protocol problems and variance; don't use pilot observations in the final confirmatory analysis.
3. Record listener IDs as pseudonyms, product/sample, fixture profile, excerpt ID, level matching procedure, condition, randomized order, preference response, confidence and exclusions. Keep sensitive demographic data out unless justified and consented.
4. Split evaluation by **listener and excerpt**; where products are involved, hold out product model/revision as well. Never let different reseats of the same product appear as independent examples across training and test.
5. Compare at minimum: static target-deviation baseline, population-average profile model, and hierarchical/personalized model. Fit hyperparameters only on training/development folds.
6. Report held-out pairwise log loss/Brier score, calibration, accuracy only with a naive baseline, ranking, confidence intervals, and how often personalization harms or fails to help.
7. Publish protocol, code, fixture limitations and null results. Do not infer universal preference from a small convenience sample.

This is an experiment plan, not a claim that participant data already exists or that a human-subject study has approval. Any participant study requires appropriate consent, ethics and safe listening procedure.

## How this connects to the current codebase

- **EAC response analysis / Pareto work:** becomes the transparent objective baseline and multi-objective optimizer layer. Do not replace it with learned scoring before the baseline is validated.
- **Acoustic experiment selection:** compare policies under synthetic known-truth tests before any real hardware study; include total cost and deferred validation.
- **Open Audio Evidence Contract:** attaches all captures, calibration, analysis and gate outputs to an exact run.
- **K0/K1/K2 instrument kits:** provide more repeatable room/speaker captures and headphone fit/reseat data.
- **H1 and S1 prototypes:** become the first hardware where robust response, fit/directivity, manufacturing variance and listening preference are jointly studied.

Symthaea can suggest experiments and profiles. Human study governance, hardware safety limits and instrument authority remain explicit boundaries; model confidence never substitutes for calibrated measurement or qualified review.

## Decision gates

**D0 — baseline.** Implement the simple metrics, data schema and known-answer controls. Validate artifact provenance and calculate baseline scores without model leakage.

**D1 — benchmark harness.** Create synthetic known-truth cases for grouped splits, preference calibration, target/fixture transfer and active selection. Make tests prove the evaluation catches leakage and a deliberately bad predictor.

**D2 — feasibility data.** Use already licensed/open data only after checking rights and metadata compatibility. Collect new physical results only when the instrument/fixture and protocol can be characterized. Mark fixture-incompatible records rather than forcing them into one space.

**D3 — pilot.** Test the randomized experiment procedure for usability, fit/reseating, level matching and variance; use pilot findings to revise and preregister the confirmatory protocol.

**D4 — confirmatory evaluation.** Evaluate fixed-baseline vs candidate model on untouched products/listeners/excerpts. Publish all metrics, confidence intervals, failure cases and calibration.

**D5 — product integration.** Only deploy a predictor/optimizer if it improves a prespecified, relevant metric over the baseline, preserves uncertainty, supports safe choices, and fails closed outside its validated domain.

## Sources

- RTINGS, [One Curve Doesn't Fit All: 350 Headphone Listening Tests Show That Many Sound Profiles Can Be Preferred](https://www.rtings.com/headphones/learn/research/frequency-response-1-8), updated 2026-03-19.
- Olive & Clark, AES, [A Preferred Headphone Target Curve Defined for the B&K 5128 and GRAS45CA-10 Test Fixtures](https://aes.org/publications/elibrary/elibrary-page/?id=23068), 2025.
- RTINGS, [Distortion, Audibility, and Preferences: What We Know, and What's Missing](https://www.rtings.com/headphones/learn/research/distortion-audibility), updated 2026-03-19.
- RTINGS, [The Surprising Truth About Headphone Distortion: Most Of It Is Inaudible](https://www.rtings.com/headphones/learn/research/distortion-audibility-listening-tests), updated 2026-03-19.
- Olive, AES, [The Preferred In-Ear Headphone Target Curve Based on Method of Adjustments](https://aes.org/publications/elibrary/elibrary-page/?id=22928), 2025.
- Olive, AES, [A Multiple Regression Model for Predicting Loudspeaker Preference Using Objective Measurements: Part II](https://aes.org/publications/elibrary/elibrary-page/?id=12847), 2004.
- Spinorama, [Computing a spinorama from your own measurements](https://www.spinorama.org/docs/08_compute_spin.html), updated 2026-04.
- ITU-R, [BS.1116 — small audio impairments](https://www.itu.int/rec/R-REC-BS.1116).
- ITU-R, [BS.1534 — intermediate-quality subjective assessment](https://www.itu.int/rec/R-REC-BS.1534/en).
- Frontiers, [Bayesian experimental design in production engineering](https://www.frontiersin.org/journals/manufacturing-technology/articles/10.3389/fmtec.2025.1614335/full), 2025.

## Claim boundary

No Symthaea preference model, listener-specific tuning model, perceptual distortion metric, or new hardware design is claimed to have passed this programme. The deliverable is a prioritized and falsifiable research agenda with established references, explicit uncertainties, proposed internal gates and a first experiment that can eventually produce real evidence.
