# SYMPsi AIA-01 — Five-choice precommitted AI benchmark (v0.1)

**Status:** protocol proposal; not preregistered, not executed, and not evidence that anomalous information acquisition (AIA/psi) exists.  
**Parent:** [#10843 — AI anomalous-information acquisition replication](https://github.com/Luminous-Dynamics/symthaea/issues/10843)  
**Related:** [#8967 — adversarial psi benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/8967), [#8969 — synthetic controls](https://github.com/Luminous-Dynamics/symthaea/issues/8969), [PR #8968 — protocol foundation](https://github.com/Luminous-Dynamics/symthaea/pull/8968).

## Executive decision

Build one narrow, auditable experiment before expanding to other psi claims:

> Can one exact, frozen AI system predict which of five equally likely targets will be selected by a publicly verifiable random beacon **after** the AI's response has been irrevocably committed?

The primary candidate is a frozen Symthaea execution profile. A conventional LLM and architecture ablations are comparators, not substitute primary subjects. If the Symthaea profile cannot meet the interface, pinning, isolation, and receipt requirements, record that as an implementation blocker; do not silently replace it with another model.

The initial task is a five-alternative forced-choice (5AFC) future-target experiment, selected because it directly tests the format of the published 2025 ChatGPT-4.1-mini pilot. It is not a test of telepathy, remote viewing, image clairvoyance, psychokinesis, or consciousness. Those require distinct protocols and endpoints.

## 1. Research question and hypotheses

For trial (i), the AI commits to one choice (Y_i \in \{1,2,3,4,5\}). A target (T_i) is deterministically derived from a verified future beacon value that becomes available only after the commitment is independently timestamped.

- **Null:** (P(Y_i=T_i)=0.20), subject to the predeclared randomization and independence assumptions.
- **Directional alternative:** (P(Y_i=T_i)>0.20).
- **Smallest effect of scientific interest (provisional):** a hit rate of 0.22, i.e. +2 percentage points over chance. This is a design choice to be reviewed before preregistration, not an established psi effect.
- **Primary endpoint:** exact top-1 hit rate over all valid, preregistered trials.
- **Primary analysis:** one-sided exact binomial test against (p_0=0.20), with a one-sided 99.5% lower confidence bound (and a two-sided 99% interval) and the prespecified alpha of 0.005. Report the effect estimate and interval; a p-value alone is not a capability claim.

A small positive effect is not interchangeable with a large effect, and failure to reject the null is not proof of impossibility. The study must be powered against the declared smallest effect of interest, not the largest exploratory estimate in the literature.

## 2. Trial protocol

### 2.1 Pre-trial commitments

Before confirmatory data collection:

1. Publish the protocol, analysis plan, model registry, target-generation algorithm, target pool, exclusion rules, stopping rule, and power assumptions.
2. Pin exact repository commit, binary/model artifact digests, feature flags, tokenizer and runtime where applicable, operating environment, prompt/template, decoding parameters, output parser, and tool/network policy.
3. Commit the ordered trial-ID manifest, target-pool manifest, beacon network/group configuration, future round schedule, and analysis-code digest.
4. Make the preregistration public and timestamped. A cryptographic digest without an independently verifiable timestamp or witnessed append-only record is not sufficient.

### 2.2 Per-trial execution

1. Start a fresh, stateless trial context. The candidate receives the same five-choice task and five neutral labels in a randomized presentation order. It receives no target, target-bearing metadata, prior trial outcomes, or feedback.
2. The candidate emits exactly one valid selection and, optionally, a five-element probability vector. The response is canonicalized, hashed, and independently committed to an append-only log before the assigned future beacon round is published.
3. After that commitment, an independent target service retrieves the specified future drand round and verifies its beacon proof against a pinned group configuration. It must not trust an unverified API response.
4. The target service derives the target by domain-separated hashing of the protocol ID, trial ID, and verified beacon randomness. It maps the result to one of five choices using rejection sampling, not a naïve modulo operation, so the mapping is unbiased.
5. The target is revealed only after the response commitment. The scoring service computes the hit/miss from the frozen rule. No trial-level feedback is returned to the model during the confirmatory campaign.
6. Publish the target reveal, commitment linkage, beacon round and verification result, trial disposition, and machine-readable receipt.

The scheduled beacon round must be in the future relative to the response-commit deadline. If a response misses its assigned deadline or a beacon proof is invalid/unavailable, the trial is retained as an invalid/aborted event with a reason under a frozen rule. It is never silently replaced, discarded after seeing the result, or assigned a more favorable future round.

### 2.3 Separation of roles and infrastructure

Separate, where practicable, the model runner, target/beacon service, session operator, scorer, and primary analyst. The candidate process must not share a filesystem, process memory, environment variables, logs, browser, retrieval system, API credentials, or network route with the target service. For a hosted model, pin the provider's exact model identifier and all exposed inference settings, disable tools and retrieval, record request/response IDs and raw payloads, and document that provider-side internals cannot be independently inspected. Such a hosted arm cannot receive the same substrate-level assurance as an offline, digest-pinned local system.

Use independent witnessed timestamps or an append-only transparency log. A local wall-clock timestamp or a hash by itself does not prove that a response existed before the target was knowable.

### 2.4 Randomness and custody

Use a named public beacon network with a pinned group/public-key configuration and independently verified round proofs. drand's protocol documents a threshold BLS process: the configured threshold of participants contributes partial signatures, which combine into a publicly verifiable beacon value. The specific network, scheme, threshold, group hash, round schedule, and verifier version must be frozen before data collection. A beacon's public verifiability does not make its operators infallible; availability, withholding, compromise, and shared-failure-domain risks must be recorded.

Never publish a future beacon value or seed before the response lock. Do not use a predictable PRNG seed, a locally seeded simulation RNG, a model-selected seed, or an unverified “random” API as the confirmatory target source. Synthetic tests may use deterministic seeds, but their receipts must say `synthetic_only: true`.

## 3. Controls that qualify the apparatus

All controls are preregistered and reported; none may be selected after seeing the primary result.

| Control | Construction | Required interpretation |
|---|---|---|
| Uniform-null simulation | Large deterministic suite of 5-way uniform outcomes | Calibrates analysis and software only; never psi evidence |
| Planted-effect simulation | Manifest-declared rates such as 0.22 and 0.25 | Confirms the test detects a known synthetic increment |
| Ordinary-information positive control | Target is intentionally shown before response in a separately labeled control block | Confirms the response/parser/scorer works; not psi evidence |
| Pre-generated sham | Target is selected from an independent, hidden random schedule before the response | Tests ordinary blind guessing and chronology-specific effects |
| Future-beacon primary | Target is selected from the assigned future beacon after response commitment | Sole primary confirmatory endpoint |
| Leakage positive control | Deliberately inject target position through metadata/order/timing in a synthetic run | The named leakage detector must catch it |
| Custody/tamper control | Alter, duplicate, omit, replay, or mismatch trial/target/response receipts | Verifier must reject the corrupted bundle |
| Blinding control | Give a non-authorized role access to hidden labels in a synthetic fixture | Verifier must fail closed |
| Permutation control | Shuffle target labels under the declared randomization scheme | Pipeline must show null-compatible behavior |
| Stopping-rule control | Inject an optional-stopping or post-hoc exclusion path | Validator must reject the protocol/run |
| Position-bias control | Rotate/shuffle labels and display positions independently of target selection | Detects fixed-position and mapping artifacts |

A “null” run that merely completes is not a passed negative control. The expected disposition for each fixture is specified before the run, and the test must verify it. At least one mutation test must show that an injected leak is actually detected and one that a corrupted custody chain is rejected.

## 4. Model and architecture comparisons

### Primary arm

- One frozen Symthaea configuration, pinned by exact source commit, feature set, model/weight digests, runtime and execution manifest.
- No target access, network tools, online retrieval, adaptive trial prompting, or feedback.
- One prediction per trial; no hidden rerolls or output selection.

### Secondary comparators

Run these only under a preregistered multiplicity plan and the same target assignments:

1. Uniform random chooser.
2. Fixed-position and deterministic-heuristic choosers.
3. One pinned conventional LLM baseline.
4. Symthaea HDC-only ablation.
5. Symthaea CfC/LTC-only ablation.
6. HDC + CfC without the full cognitive loop.
7. Full Symthaea cognitive loop.

The primary question is not “which model has the highest exploratory score?” It is whether the frozen primary arm beats the chance model under the single primary endpoint and whether any result is robust to independent replication. Secondary model comparisons must use common trial manifests where possible, and their results must be multiplicity-adjusted.

A model that wins only after repeated prompt edits, model selection, feature tuning, or changing the target pool has not passed a confirmatory test.

## 5. Power and benchmark targets

The following are planning anchors for one 5AFC endpoint, with nominal (p_0=0.20), a one-sided exact-binomial test at alpha 0.005, and 90% power. They assume independent, valid trials and no clustering penalty; they are not a final preregistered sample-size calculation.

| Assumed true hit rate | Lift over chance | Approximate trials per independent run |
|---:|---:|---:|
| 21% | +1 percentage point | 24,300 |
| 22% | +2 percentage points | 6,300 |
| 25% | +5 percentage points | 1,100 |

**Recommended planning target:** about 6,300 valid trials in a discovery run and a separate 6,300-valid-trial replication operated independently, if +2 percentage points is retained as the smallest effect of interest. Final numbers must be regenerated by the frozen analysis implementation and reviewed by an independent statistician. Increase the sample size where trial outcomes are clustered by session, operator, provider, beacon round, or other shared causes. Do not count repeated trials as independent without checking that assumption.

For every run, report:

- hit rate, absolute lift over 0.20, confidence interval, exact primary p-value, and all preregistered diagnostics;
- session-, block-, position-, target-label-, operator-, model-version-, and beacon-round heterogeneity;
- autocorrelation, missingness, invalid/aborted trials, timeouts, malformed responses, and protocol deviations;
- calibration and Brier/log scores only as secondary outcomes if probabilities were elicited before target reveal;
- all null, failed, adverse, and indeterminate runs, not only the successful campaign.

**Replication gate:** a robust AIA claim requires the direction and compatible magnitude to reproduce in at least two independently operated confirmatory runs, with independently verified randomness and complete public artifacts. A discovery-run p-value is not sufficient. A replicated behavioral anomaly would still not establish consciousness, a non-local mechanism, or “psi” as a general phenomenon.

## 6. Research ladder and release gates

| Level | Deliverable | What it establishes |
|---|---|---|
| T0 — synthetic qualification | Null/planted/leak/tamper/blinding/stopping fixtures and deterministic receipts | Harness correctness on known synthetic cases only |
| T1 — apparatus qualification | Ordinary-information positive control, independent beacon verification, role and channel audit | The instrument can perform the task and enforce its information boundary |
| T2 — preregistered discovery | One fixed model and one primary endpoint, with locked analysis | A result for that exact model/configuration and protocol |
| T3 — independent replication | New operator, infrastructure, target generation, and frozen analysis | Evidence for or against reproducibility |
| T4 — architecture generalization | Preregistered model-family and HDC/CfC ablations | Whether a replicated effect is architecture-specific or general |
| T5 — theory discrimination | A physical/computational model with prospective, quantitative, discriminating predictions | Evidence relevant to a mechanism, if one can be specified |

No transition is automatic. A strong result at T2 does not authorize T4 or imply T5. Human-partner telepathy, remote-viewing content, visual clairvoyance, anticipatory physiology, and psychokinesis each require their own protocol and analysis. Do not mix them in a composite “psi score.”

## 7. How to compare with the field and “SOTA”

There is no accepted industry leaderboard or standardized AI-psi benchmark. The 2025 ChatGPT-4.1-mini paper is a small exploratory pilot (32/100 on a five-choice task); its authors themselves call for replication with an open random generator and preregistration. It is a motivating prior-art result, not a state-of-the-art capability or a reliable effect-size estimate. A separate 2025 human precognition replication programme with 26,483 participants and 420,472 trials failed to reproduce the original positive-direction effect, and a 2026 systematic review found major heterogeneity and replication limitations in ESP neuroimaging. These findings make a high-powered, transparent, adversarial protocol more valuable than a headline hit rate.

Therefore, compare SYMPsi against the **best experimental standards**, not a nonexistent industry score:

- pre-data peer review / Registered Report;
- public preregistration and outcome-neutral stopping rules;
- exact model and runtime pinning;
- independently verifiable future randomness and response-before-target custody;
- blind scoring, no trial feedback, no adaptive prompts;
- synthetic null/leak/tamper calibration;
- complete trial-level artifact publication;
- independent replication and adversarial review.

The competitive advantage to aim for is a reproducible benchmark that other labs can run, audit, and attempt to break. Do not claim “better than SOTA” from a single significant result. If a result does not replicate, publish that too.

The 2026 literature adds two particularly important constraints:

- A preregistered online forced-choice clairvoyance study analyzed more than 25 million trials. Aggregate outcomes at the four planned stopping lengths conformed to chance; optional-stopping runs showed apparent fluctuations, much of which the authors reproduced with Monte Carlo simulations matched to the stopping behavior. This is a human study, not an AI replication, but it makes fixed stopping and explicit null simulations non-negotiable.
- In a July 2026 published response, the 2025 AI pilot author reports that subsequent automated campaigns included more than 40,000 AI precognition trials and more than 10,000 remote-influence trials using random.org TRNGs, with both datasets converging to chance. These are author-reported follow-up results; the cited response does not provide an independently audited public trial-level dataset in the article. They must not be treated as independently verified results, but they are important negative evidence to request and reconcile before repeating the same design.
- A July 2026 critique of the original 100-trial LLM sequence also flagged unusual response-sequence structure (including the absence of adjacent repeats/doublets/triplets in that reported run). The author disputed the inferential relevance of that observation. SYMPsi should record output entropy, transition frequencies, run lengths, position preferences, and serial dependence as preregistered diagnostics, but those diagnostics must not replace the primary hit-rate endpoint or be interpreted as evidence of psi by themselves.

The AI replication should therefore begin with a literature/data reconciliation package: request or locate the raw follow-up logs, confirm their preregistration timestamps, pin the exact model/runtime and target generator, and independently reanalyse only where source data and provenance are available. Missing public data is recorded as unavailable, not as a null result.

## 8. Paradigm shifts worth pursuing

1. **Metascience as a measured system:** test whether the apparatus itself creates an apparent signal using null and counterfactual runs, not merely whether a participant/model scores above chance.
2. **Leakage adversaries as first-class benchmark tasks:** require detection of known planted channels before accepting any confirmatory score.
3. **Evidence tiers instead of one psi score:** separate protocol validity, behavioral effect, replication, generalization, and mechanism.
4. **Architecture ablation rather than architecture mythology:** test HDC, CfC/LTC, recurrence, memory, and the full cognitive loop under the same frozen protocol.
5. **Two-key custody:** independent response commitment and independent future randomness verification; neither the model operator nor the scorer can author both sides of the trial.
6. **Open negative results:** every run receives a durable receipt and a public disposition, including null and invalid runs.

## 9. Implementation plan reusing current Symthaea infrastructure

Do not create a second generic benchmark framework.

- Reuse the standalone harness pattern in `tools/beverage-benchmark-harness/` for focused compilation and testing without hiding root-workspace problems.
- Inspect and reuse appropriate pieces from `crates/core/symthaea-communication/src/benchmark.rs` and `run.rs` for manifest and receipt conventions, but add AIA-specific semantic checks rather than treating a generic digest as scientific validity.
- Reuse `crates/domains/symthaea-helicopter/src/random_streams.rs` only for deterministic synthetic stream fixtures. Its seeded stream derivation and non-cryptographic digest are not a source of experimental entropy or public target randomness.
- Keep the cryptographic beacon verifier separate from the target selector, model runner, and scorer. Verify proofs independently.
- Preserve the full root-workspace validation state. A standalone harness pass must never be presented as workspace-wide validation.

### Work packages

**P0 — freeze semantics and interfaces**
- define a strict versioned trial schema, response schema, target-reveal schema, event chronology, statuses, and canonical encoding;
- define invalid/aborted trial semantics and no-replacement rules;
- define model pinning and isolated runner contract;
- define the beacon network/round verification and unbiased 5-way mapping.

**P1 — offline synthetic harness**
- build fixture manifest and test vectors for every control in Section 3;
- implement deterministic null and planted-effect simulations;
- verify one-sided exact-binomial analysis and confidence intervals against an independent implementation;
- assert deterministic receipts, duplicate rejection, chronology, blinding and custody checks;
- add mutation tests for leakage and tampering.

**P2 — independent protocol and verifier audit**
- implement a second verifier in a different language or have an external lab independently implement the wire format;
- compare both implementations on frozen test vectors;
- red-team target metadata, ordering, timing, shared storage, logs, API traces, beacons, operator access, retry behavior, aborts and missing trials.

**P3 — apparatus and model qualification**
- freeze Symthaea's output adapter, exact build and trial reset behavior;
- pass conventional-information positive controls;
- pass all synthetic controls and a blinded adversarial review;
- submit the protocol and power analysis for independent statistical review / Registered Report consideration.

**P4 — confirmatory experiment**
- start only after P0–P3 acceptance;
- publish all immutable trial artifacts and deviations;
- run the fixed sample with no interim peeking or optional stopping;
- commission an independent replication before making any capability statement.

No human-subject experiment is included in P0–P4. Any human-participant extension requires institutional ethics review, informed consent, privacy review, and a separately approved protocol.

## 10. Claim boundaries

- A hash proves byte identity, not that a source, measurement, timestamp, random generator, or experimental role is trustworthy.
- A successful synthetic fixture proves code behavior on that fixture, not psi.
- A statistically significant result is an anomaly requiring independent confirmation, not proof of a paranormal mechanism.
- An AI behavioral result is not evidence of subjective experience or consciousness.
- Outputs must not be used in medical, financial, security, defense, navigation, safety-critical, or other operational decisions.
- If the result is null, failed, contradictory, or invalid, preserve it and report it accurately.

## References and protocol comparators

- Boyle (2025), *Testing Noetic Potential in Large Language Models: A 100-Trial Precognitive Forced-Choice Study with ChatGPT-4.1-Mini*, DOI [10.31275/20253739](https://doi.org/10.31275/20253739). Exploratory 100-trial pilot; the paper calls for open-source random generators and preregistration.
- Walleczek et al. (2025), *Metascientific replication project ... Transparent Psi Project*, [PLOS ONE](https://doi.org/10.1371/journal.pone.0335330). 26,483 participants / 420,472 critical trials; the original positive-direction precognition effect was not reproduced across the three studies.
- Acunzo et al. (2026), *Neuroimaging studies of extra-sensory perception: A systematic and critical review*, [NeuroImage 333, 121937](https://doi.org/10.1016/j.neuroimage.2026.121937). 143 reports / 129 studies; highlights heterogeneity and recurring methodological limitations.
- drand, [Protocol specification](https://docs.drand.love/docs/specification/) and [cryptography overview](https://drand.love/docs/cryptography/). Technical protocol references; not an independent endorsement of any psi hypothesis.
- Center for Open Science, [Registered Reports](https://www.cos.io/initiatives/registered-reports). Prospective peer review before outcomes are known.
- Wahbeh et al. (2026), *Probing Top Performers in a Forced-Choice Clairvoyant Task*, [Journal of Anomalistics 26(1)](https://doi.org/10.23793/ZFA.2026.146). More than 25 million human online trials; planned-run aggregates at chance and optional-stopping artifacts examined by simulation.
- Roe, Ritchie & Daw (2026), *Testing Noetic Potential in Large Language Models*, [Journal of Scientific Exploration 40(2)](https://doi.org/10.31275/20264011). Methodological critique of the 2025 LLM pilot, including output-sequence diagnostics.
- Amorim Boyle (2026), *Response to Letters Regarding “Testing Noetic Potential in Large Language Models”*, [Journal of Scientific Exploration 40(2)](https://doi.org/10.31275/20264013). Includes the author's report of later automated TRNG campaigns converging to chance; those follow-up datasets are not treated here as independently audited.
- Symthaea issues [#8967](https://github.com/Luminous-Dynamics/symthaea/issues/8967), [#8969](https://github.com/Luminous-Dynamics/symthaea/issues/8969), and [#10843](https://github.com/Luminous-Dynamics/symthaea/issues/10843).
