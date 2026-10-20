# Melothaea Style-Interaction Benchmark v1

**Status: research protocol only.** This document defines a next-step benchmark for style-control composition. It adds no production behavior, does not claim a completed run, and is not evidence that any style sounds authentic or better.

**Proposed work item:** MUSE-STYLE-INTERACTION-001  
**Parent context:** [#7417](https://github.com/Luminous-Dynamics/symthaea/issues/7417), [#7422](https://github.com/Luminous-Dynamics/symthaea/issues/7422), [#7764](https://github.com/Luminous-Dynamics/symthaea/pull/7764), [#8081](https://github.com/Luminous-Dynamics/symthaea/pull/8081), [#8121](https://github.com/Luminous-Dynamics/symthaea/pull/8121).

## 1. Why this is the next gap

Melothaea's existing work separates several important questions:

- **Catalog and source inventory:** which named styles and declared profiles exist?
- **Single-axis response:** what changes when one input such as tonic, valence, arousal, energy, or requested length changes?
- **Performance/render contract:** which dialect, timing, room, and mastering settings are selected?
- **Blind listening logistics:** can a complete, byte-verified cohort be presented anonymously?

Those are necessary, but individually live controls do not establish that controls compose. A system may respond to groove and instrumentation separately while their combination causes a style override to be lost, changes a preserved motif, breaks a meter/cycle invariant, or becomes dependent on rule ordering.

The next question is therefore:

> **Can Melothaea combine two declared musical transformations while producing the intended joint effect, preserving every explicitly protected musical property, and disclosing conflicts or unsupported capabilities?**

This benchmark complements, rather than replaces, the existing style inventory, one-axis response-curve work, renderer-profile experiments, and Creative-QD work. It does not add style routes or a new quality scalar.

## 2. Paradigm shift: music recipes as typed interventions

Represent an experiment as:

`baseline composition + declared intervention(s) + preservation contract`

Each intervention names:
1. the target axis and versioned profile;
2. its expected observable effects;
3. explicitly protected properties;
4. permitted secondary effects and interactions;
5. unsupported or unresolved semantics.

The result is compared with a matched baseline. A declarative recipe must not silently resolve conflicting values through "last writer wins". Its resolver must either produce a deterministic plan with warnings or reject the conflicting combination with a structured reason.

A hybrid is a **new recipe**, not an assertion that a historical musical tradition contains that combination. Genre ancestry, instrumentation, musical grammar, mood, performance, and production remain distinct concepts. Similarity of feature vectors is not evidence of cultural lineage or authenticity.

## 3. Frozen initial experiment matrix

Start with symbolic score outputs, where motif, form, pitch and timing are inspectable. Add audio analysis only after symbolic outcomes are qualified.

The initial pairwise matrix should cover distinct mechanisms rather than many near-identical genre labels:

| Intervention A | Intervention B | Protected property examples | Observable joint-effect question |
|---|---|---|---|
| Groove / accent cycle | Harmonic or modal language | Frozen motif contour; requested cycle continuity | Does the harmonic treatment change without flattening the groove, or does a declared incompatibility trigger a clear result? |
| Formal plan | Motif development | Section order; theme identity and allowed transformations | Does the motif develop across the form instead of being reset or merely copied? |
| Ensemble / role assignment | Performance dialect | Voice/part identity; pitch content where protected | Are parts assigned and performed according to the selected recipe, rather than only renamed? |
| Phrase grammar | Tempo / energy profile | Phrase obligations and ending behavior | Does energy alter pace or density without destroying the phrase grammar? |
| Room / production profile | Performance dynamics | Symbolic score and performed-note timing when held fixed | Does the rendering intervention remain within its declared scope? |
| Modal or groove recipe | User-preserved melody | Exact protected pitches/rhythm or explicitly chosen transformations | Can a style change leave protected material untouched and mark what could not be reconciled? |

The matrix is a test-design proposal, not a statement that all combinations are valid. Define allowed transitions and protected fields per fixture. A musical property may be protected exactly, protected up to a named transformation (such as transposition), or intentionally left free; do not infer the protection policy after observing outputs.

### Conditions per matched fixture

- **C0 — baseline:** frozen composition, recipe and seed.
- **CA — A only:** one intervention.
- **CB — B only:** the other intervention.
- **CAB — A + B:** both interventions declared together in one recipe.
- **CAB-reordered — resolver metamorphic control:** reverse the serialization / input ordering of the same declarative recipe. This should resolve identically when the schema declares the axes independent and order-free. Ordered musical stages are not subject to that expectation.

Use the same source scaffold and seed across the five conditions. Do not compare a single CAB sample with a best-of-many baseline. All generation and selection budgets must be disclosed.

Initial development cohort: eight fixed seeds `[17, 53, 89, 144, 233, 377, 610, 987]` per eligible pair. Freeze these as the v1 development set; preserve a separate, unpublished or later-added held-out seed set for confirmatory claims. These seeds are a proposed protocol choice, not a measured result. Reuse the existing composer path where possible instead of creating another generation implementation.

The first symbolic pass is deliberately small: six pairs × eight seeds × five conditions = 240 condition runs before retries. Retries are recorded as failures plus attempts, never silently dropped. Audio listening can use a preregistered subset chosen before output identities are unblinded.

## 4. Measure effects as a vector

Do not collapse all behavior into one aesthetic or distance score. For each condition record:

### Target fidelity

For each intervention, predeclare which observations should change and in what way. Candidate observations include:
- rhythm: accent/cycle continuity, onset distribution, syncopation and subdivision;
- form: section order and transition obligations;
- motif: exact identity, named transformation type, and occurrence pattern;
- pitch/harmony: tonal or modal constraints, cadence/transition behavior, and voice-leading invariants;
- ensemble: persistent part identity, role assignment, playable register and unsupported-instrument disclosure;
- performance: articulation and timing dialect actually selected;
- rendering: exact profile identity and signal-health measurements.

Use only measurements whose semantics are defined for the relevant grammar family. A generic meter, pitch diversity, spectral, or fingerprint difference is not proof that the requested style behavior occurred.

### Preservation and edit locality

For every protected property, report:
- exact pass/fail or the declared transformation-aware check;
- before/after canonical identity where relevant;
- notes/events altered and alteration reason when the schema can expose it;
- unresolved or unsupported constraints.

A useful edit-locality diagnostic is the ratio of unintended changed protected fields to all protected fields evaluated. Report its denominator and every failure; never hide one serious violation inside an average. For exact-protection fixtures, the hard target is zero protected-field violations.

### Interaction, not only additive change

For numeric observations with a meaningful common scale, report the factorial interaction:

`I_AB = y_AB - y_A - y_B + y_0`

This is a diagnostic of non-additivity, not automatically a defect. Musical controls can legitimately interact. For categorical or symbolic observations, compare declared constraints and transition traces directly rather than forcing them into the equation.

A valid joint composition should satisfy the predeclared joint contract. It need not equal the sum of two independent changes. Publish main effects, interactions, exceptions and per-seed results.

### Repeatability and evidence binding

A report must bind:
- exact repository commit and source/profile digests;
- schema and benchmark-corpus digest;
- base score/scaffold digest and declared seed;
- resolved profile identities, warnings and fallback decisions;
- generation attempt counts and failures;
- canonical symbolic output digest, and separate audio hashes/renderer identity when rendered;
- toolchain and relevant asset identities;
- per-condition outcome rows, including failures.

A claimed commit SHA is identity metadata, not independent proof of how an executable was built. Keep artifact hashes, executable/build evidence and source identity separate.

## 5. Baselines

Compare the same frozen cases against these internal baselines:

1. **No intervention:** existing default recipe.
2. **Single-axis composition:** CA and CB independently.
3. **Sequential override baseline:** apply A then B and B then A where the legacy interface permits it; measure whether order unexpectedly changes supposedly order-independent axes.
4. **Joint declarative recipe:** CAB resolved under typed compatibility and transition rules.

This measures whether the new representation adds reliable compositional control, rather than merely adding code or changing an aesthetic profile. If a sequential baseline is unavailable, record it as unavailable; do not fabricate the result.

For industry context, maintain a *separate* human task track against current commercial tools when access and terms allow. Suno's official v6 release notes (9 September 2026) describe localized section editing and multi-source mashups. Such claims motivate evaluating preservation of unedited material, but product claims are not benchmark results. A commercial system may not expose symbolic scores or comparable controls; compare the user-facing task and blinded outcomes, not internal symbolic metrics. Fix the prompt, attempt budget, selected output policy, source assets/rights, and tool/model version. Report one-shot and best-of-N separately.

MIREX 2026 provides useful external anchors but not a direct substitute: the Symbolic Music Generation task focuses on piano continuation and rates prompt coherence, creativity, structure and musicality; the Music Audio Generation cover-song task rates melody, harmony, key consistency, style consistency, and arrangement/production; CMI-RewardBench studies prediction of pairwise human preference. These tasks answer different questions and must not be collapsed into a shared leaderboard.

## 6. Human evaluation

For perceptual claims, build randomized, anonymized, level-matched listening trials from frozen outputs. Ask separate questions for:
- intended target/style fit;
- whether the protected motif or structure remains recognizable;
- whether the requested change is perceptible;
- musical coherence / overall quality;
- audible rendering defects.

Use a pilot to identify misunderstood questions and estimate variance, then preregister the confirmatory sample-size/power analysis, exclusions, hypotheses and analysis. Report listener-level clustering, confidence intervals, rater disagreement and every predeclared style pair. Cultural or tradition-specific authenticity requires qualified practitioner review and appropriate provenance; listener preference alone does not grant authenticity.

Automatic metrics must be checked against held-out human responses for calibration and rank correlation. No single learned reward model, genre classifier, spectral statistic or aesthetic scalar is ground truth.

## 7. Proposed gates and targets (not achieved)

| Dimension | Target |
|---|---|
| Contract integrity | 100% of preregistered malformed/conflicting-recipe fixtures rejected or explicitly classified as allowed with reason |
| Preserved material | Zero violations on exact-preservation fixtures; transformation-aware fixtures must match their declared transform |
| Resolver determinism | Same recipe bytes, engine revision and seed resolve to the same canonical plan and symbolic result within declared reproducibility guarantees |
| Silent fallbacks | Zero; every unsupported field and fallback is visible in the receipt |
| Interaction coverage | Every frozen pair reports C0/CA/CB/CAB; no missing or failed cell is omitted from the report |
| Seed robustness | Report per-seed target fidelity and failures; after calibration, aim for at least 95% directional consistency on controls whose semantics explicitly predict a direction |
| Non-target invariance | Exact invariants remain exact; other non-target effects have declared tolerance bands and are reported independently |
| Perceptual superiority | Only claim superiority after a powered, blinded comparison against a strong matched baseline and replication on held-out briefs/seeds |
| Resource envelope | p50/p95 latency, peak RSS, render time, output duration and artifact size on pinned hardware/toolchain |

The 95% directional-consistency goal applies only to preregistered directional controls. A thresholded, non-monotonic or context-dependent control should be judged against its own declared semantics, not forced into monotonicity.

## 8. Failure taxonomy

Every failed cell should resolve to one of:
- invalid recipe / conflicting hard constraints;
- unsupported capability or declared fallback;
- composer contract violation;
- target effect absent or in the wrong direction;
- unintended protected-field change;
- expected non-additive interaction;
- renderer/audio integrity failure;
- missing provenance / incomplete evidence;
- perceptually unresolved result.

These are distinct outcomes. A render succeeding does not prove style identity; a symbolically valid score does not prove audible fidelity; listener preference does not prove historical authenticity.

## 9. Execution sequence

1. Freeze the v1 pair definitions, protected properties, effect semantics, condition budget and held-out split before inspecting comparative results.
2. Implement a small Rust benchmark runner and typed JSON receipt alongside the existing style response-curve / Creative-QD tooling; reuse the production composition path and validators.
3. Add independent oracle fixtures for exact preservation, allowed transformation, a true constraint conflict, and an unsupported axis.
4. Run symbolic tests and adversarial malformed-input tests locally or in an available Rust environment. Do not wait for hosted runners to write the fixture corpus or validate expected outcomes.
5. Qualify exact-head CI and artifact provenance; queued, skipped, absent or zero-matched test output is not a pass.
6. Render the preregistered subset and assemble a randomized listening pack; only then draw perceptual conclusions.
7. Use the observed interaction failures to prioritize changes to grammar/profile architecture. Do not add catalog entries to compensate for dead or incompatible controls.

## 10. Research references

- [Suno v6 release notes, 9 September 2026](https://www.suno.com/release-notes/introducing-v6) — localized editing and multi-source mashup product capabilities.
- [MIREX 2026: Symbolic Music Generation](https://www.music-ir.org/mirex/wiki/2026%3ASymbolic_Music_Generation) — continuation task and distinct subjective criteria.
- [MIREX 2026: Music Audio Generation](https://www.music-ir.org/mirex/wiki/2026%3AMusic_Audio_Generation) — cover-song task with melody, harmony, key, style and production ratings.
- [MIREX 2026: Music Evaluation via CMI-RewardBench](https://www.music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench) — pairwise human-preference prediction.
- Duan, Yu & Briot, [*Recent advances in music generation: methods, evaluation, and challenges*](https://doi.org/10.1007/s10462-026-11582-x), 2026 — documents fragmented evaluation and the limits of any single music-generation metric.
- Ibáñez-Martínez et al., [ICASSP 2026 work on music representation disentanglement](https://arxiv.org/abs/2602.10058) — motivates measuring whether semantic controls actually behave as intended.

**Evidence boundary:** this document is the protocol. It includes no generated score/audio results, no new compile/test/CI results, and no claim that Melothaea outperforms commercial or academic systems.
