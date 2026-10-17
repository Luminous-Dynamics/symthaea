# HP-PREF-001 — Preregistration Scaffold for Headphone Preference Profiles

**Status:** planned study scaffold; no participant recruitment, ratings or results are represented here.  
**Tracker:** [HP-PREF-001 #7889](https://github.com/Luminous-Dynamics/symthaea/issues/7889).  
**Research parent:** [HIFI-RESEARCH-001 #7888](https://github.com/Luminous-Dynamics/symthaea/issues/7888).  
**Evidence record:** use the [Open Audio Evidence Contract](OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md) for physical captures and the versioned schema at [audio-evidence-manifest.schema.json](../../schemas/audio-evidence-manifest.schema.json).
**Split integrity:** use the [Open Hi-Fi Benchmark Protocol](HIFI_BENCHMARK_PROTOCOL_2026-10-17.md); the grouped split checker guards against declared listener, excerpt, product-revision, fixture and session leakage but does not validate model quality.

## 1. Research question and estimand

**Primary question:** For a new listener and unseen music excerpts, does a listener-adaptive headphone profile predict pairwise preference better than a fixed population profile?

The primary estimand is the held-out predictive loss difference between the preregistered personalized model and the strongest frozen non-personalized baseline, over a declared participant/excerpt test distribution. Positive benefit means lower held-out loss for the personalized model. **Provisional engineering ambition:** at least a 10% relative reduction in pairwise log loss, with a listener-clustered uncertainty interval supporting improvement and no material degradation in probability calibration. A separate pilot/power simulation must justify and freeze the final practical-effect threshold before confirmatory collection; it may revise this 10% ambition before—not after—confirmatory labels are inspected. This is a predictive study; it does not by itself prove that adapting a profile improves long-term enjoyment, hearing health or preference for all users.

## 2. Hypotheses

- **H1 (primary):** The personalized profile model lowers held-out pairwise log loss relative to the strongest frozen baseline by a preregistered practical margin.
- **H2 (secondary):** The personalized model improves Brier score/calibration and does not degrade performance materially on the subgroup of listeners for whom personalization is unhelpful.
- **H3 (secondary):** A population model improves on a naive majority/prevalence baseline on held-out listeners/excerpts.
- **Exploratory:** Preference interactions with excerpt, profile family and self-reported listening context.

Null and negative results are valid outcomes. If the fixed/population model performs as well as personalization, do not claim the more complex model is justified.

## 3. Freeze before confirmatory collection

Complete this table only after a separate pilot and power/simulation analysis; do not silently tune entries while inspecting confirmatory results.

| Preregistered item | Value to freeze |
|---|---|
| Protocol / version | Unique ID and immutable revision |
| Primary endpoint | Pairwise log loss on held-out trials |
| Practical-effect threshold | Numeric reduction vs strongest frozen baseline, justified by pilot/power simulation |
| Secondary endpoints | Brier score/calibration; pairwise accuracy and ranking with intervals |
| Profile set | Exact target/profile files and hashes; include baseline and reasonable contrasts |
| Listening excerpts | Licensed/owned stimulus IDs, excerpts/timestamps, levels and presentation rules |
| Population | Inclusion criteria and recruitment strategy, without overclaiming representativeness |
| Sample size | Power/simulation result under declared assumptions and expected attrition |
| Randomization | Generated seed, condition order and counterbalancing |
| Masking | What the participant, session operator and analyst cannot see |
| Level match | Measurement method and tolerated residual mismatch; verify at the listening chain |
| Exclusions | A priori technical, safety, fit and incomplete-session rules |
| Analysis | Model formulas, priors/regularization if applicable, software revision and uncertainty method |
| Missing data | Handling and sensitivity analysis |
| Data sharing | Consent, pseudonymization, stimulus licenses and permitted artifact release |

## 4. Experimental conditions

Compare at least the following approaches, fit only on the allowed training/development data:

1. **Fixed baseline:** one frozen population target/profile.
2. **Population model:** learns only shared effects without access to held-out listener ratings.
3. **Personalized model:** allowed a predeclared calibration subset from a listener; this subset must be disjoint from confirmatory trials and test excerpts.
4. **Naive baseline:** majority preference / prevalence model with no listener-specific information.

Use profile changes designed to cover a defensible and safe region of the response space. Do not select only profiles expected to win. Fix the profile files and their SHA-256 hashes before the confirmatory study.

The first study can use EQ profiles on a common headphone to isolate target preference from differences in physical construction. A later study is needed before claiming that the learned preference model generalizes to headphones with different transducers, pads, fit and fixture interactions.

## 5. Leakage-resistant split rules

- Split by **listener ID** and **music excerpt ID** according to the claimed use case; don't randomly split individual ratings while the same people and clips recur across all sets.
- All trials/repeats from the same listener stay in one outer listener fold when estimating cold-start performance.
- Test excerpts must not have informed model selection or the final model's hyperparameters.
- If a personalization calibration phase is part of deployment, explicitly allow only those designated ratings for that listener; hide all held-out responses.
- If claiming new-product generalization, group by physical product model/revision as an additional holdout. Same headphone with many EQ settings is not evidence for transfer to different hardware.
- Use nested development folds for model selection. Do not consult the confirmatory test to choose the profile family, features, thresholds or analysis.
- Publish the split construction code, IDs pseudonymized as necessary, assignment hashes and the exact trained model/source revision.

Report leakage tests that deliberately attempt to detect duplicate listener, excerpt, product, fixture-session or repeated-capture IDs across folds.

## 6. Listening procedure and safety

- Randomize presentation order and conceal profile/model identity from the participant; mask operators where technically feasible.
- Use a standardized switching path, identical source chain and verified level matching at a declared reference method. Document residual mismatch, especially where spectral changes affect perceived loudness.
- Use comfortable, conservative playback levels and participant-controlled stop/pause. Provide breaks and stop for discomfort or fatigue; do not use hazardous output levels as a preference condition.
- Keep the headphone fit procedure and fixture/response calibration documented. If the physical headphone moves or reseats during switching, record the event and use a frozen invalidation/retry rule.
- Obtain consent and appropriate ethics review before recruiting participants. Store participant records under pseudonymous IDs, minimize demographic data, and do not publish re-identifiable responses.
- Use owned or appropriately licensed music stimuli. Do not publish copyrighted audio files without permission; share stimulus identifiers/timestamps and hashes if redistribution is not licensed.

## 7. Analysis plan

**Primary analysis:** paired held-out loss difference between the personalized model and the strongest baseline, with a participant-clustered uncertainty estimate and the preregistered practical-effect threshold.

**Required reports:**
- pairwise log loss, Brier score and calibration plot;
- accuracy / ranking metrics with baseline and uncertainty interval;
- per-listener outcomes and fraction helped, unchanged or harmed by personalization;
- performance by unseen excerpt where sample size allows;
- missingness, exclusions, protocol deviations and failed technical controls;
- model complexity/calibration cost, including how many preference responses are required from a new listener;
- code/environment revision, split hashes, frozen target hashes and a reproducible report.

Do not use raw accuracy alone: a model can appear accurate when one choice dominates. Do not use only a correlation: it may conceal poorly calibrated probabilities or large errors on particular listeners.

## 8. Decision gates

- **P0 — pilot readiness:** profile files, level-matching procedure, randomized switching, fit procedure, consent and analysis simulation are ready. No pilot data are reused for the confirmatory endpoint.
- **P1 — pilot:** estimate variance, session usability, fatigue/fit failure rates and feasible effect thresholds. Revisions are recorded before confirmation.
- **P2 — preregister:** freeze all table entries, source/model revisions and the analysis/split code before any confirmatory labels are inspected.
- **P3 — confirmatory:** run the study, retain raw response records within the consent/privacy policy and publish the full planned analysis.
- **P4 — independent replay:** an independent analyst reproduces the held-out metrics from a permitted data package.
- **P5 — deployment decision:** adapt profiles only if the model clears the frozen primary threshold, calibration is acceptable, benefits generalize to the intended use, harmful regressions are characterized, and the added complexity is justified.

## 9. Existing research grounding

- RTINGS' 2026 study included 35 participants, five curves and two excerpts; four curves were statistically tied in that study, and the authors called it small-scale rather than definitive: [One Curve Doesn't Fit All](https://www.rtings.com/headphones/learn/research/frequency-response-1-8).
- AES 2025 work documents fixture-specific headphone targets and acoustic-transfer differences: [Olive & Clark](https://aes.org/publications/elibrary/elibrary-page/?id=23068).
- Apply ITU-R BS.1116 or BS.1534 as appropriate to the research question and actual method; do not claim compliance from superficial similarity: [BS.1116](https://www.itu.int/rec/R-REC-BS.1116), [BS.1534](https://www.itu.int/rec/R-REC-BS.1534/en).

**Boundary:** this preregistration is a planned design. No participants have been recruited and no result is claimed. Human study authorization, a power analysis and protocol freeze are prerequisites to confirmatory data collection.
