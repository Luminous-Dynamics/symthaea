# MUSE-HUMAN-CONTROL-001 — musical relevance, preservation, and agency

**Status:** draft instrument; not preregistered, not executed, and not a validated psychometric scale.  
**Version:** 1.0.0-draft  
**Date:** November 19, 2026  
**Machine-readable registry:** [MUSE_HUMAN_CONTROL_PROTOCOL_V1.json](MUSE_HUMAN_CONTROL_PROTOCOL_V1.json)

## Why this exists

The factorial experiment now measures whether a control changes a symbolic score, including exact invariance for declared unsupported axes. That is only a structural result. A changed score may be musically irrelevant, directionally wrong, imperceptible, or damaging to material that should have remained unchanged.

A recent paper (November 16, 2026) analyzed questionnaire items from 150 ISMIR music-generation papers, reducing 284 items to 57 and identifying six candidate semantic groups: Rhythm, Melody, Audio, Composition, Structure, and Lyrics. These groups are a useful item taxonomy, but they were derived from semantic relations among questionnaire items and expert interpretation—not from a new participant-response dataset establishing a universal six-factor instrument. This protocol treats them as hypotheses to validate, not settled psychometrics. See [Silveira et al., arXiv:2611.21809](https://arxiv.org/abs/2611.21809).

## Measure different things separately

The study has two layers.

### General perceived qualities

The six candidate dimensions provide a starting point for item design:

- **Rhythm:** timing stability, groove, and temporal fit.
- **Melody:** melodic coherence, phrasing, and development.
- **Audio:** rendered sound quality and audible artifacts. Only ask this when rendered audio is present.
- **Composition:** overall coherence and liking; these must be separate items.
- **Structure:** phrase, section, and long-term flow.
- **Lyrics:** intelligibility and compatibility when lyrics are present.

Do not force every item onto every trial. A symbolic-only test cannot answer audio timbre questions; an instrumental trial cannot answer lyric questions.

### Task-specific control and agency

These are primary for editing/control qualification:

1. **Target attainment:** did the requested musical property change in the intended direction?
2. **Change detectability:** can listeners identify the requested difference in a randomized blind comparison?
3. **Protected-content preservation:** did the content that should stay unchanged remain acceptable?
4. **Instruction compatibility:** does the result follow the actual instruction?
5. **Contextual coherence:** does the change fit in the surrounding music?
6. **Agency and reversibility:** in interactive trials, did the collaborator retain veto, branch, revise, and undo control?
7. **Authorship and attribution:** can collaborators distinguish their contributions from system proposals?
8. **Confidence:** does confidence track correctness?

A high preference score cannot compensate for failed preservation or wrong-direction edits. A high score-feature response cannot substitute for perceptibility or task relevance.

## Experimental controls

Use matched source–instruction–candidate sets. Include intended changes, sham/no-op outputs, wrong-direction and wrong-axis changes, protected-region mutations, yoked candidates from another task, and multiple valid alternatives where one reference waveform is not the only acceptable answer.

Randomize and counterbalance trial order and candidate order. Hide system identity and condition labels. Match source material, duration, rendering chain, loudness normalization, candidate count, and inference budget. For audio, retain renderer versions and audio digests; for symbolic trials, do not infer timbre or listening quality.

The pilot should determine stimulus length and trial count, task difficulty, participant/item variance, and clustering. Use a randomized incomplete-block design or balanced crossover to reduce fatigue. Define rest and stopping rules before confirmatory recruitment.

## Participant cohorts

Recruit and report separately:

- experienced musicians, composers, arrangers, and performers;
- general listeners without formal musical training;
- tradition-specific musicians/practitioners for culturally situated tasks;
- production and performance-technology practitioners when rendering, sound, or latency is under test.

Do not use one general online panel as a proxy for tradition-specific expertise. Use voluntary self-description, consent, compensation, withdrawal, and data-use limits. A rights/consent gate must apply to every source, target, and audio stimulus before it is shown to participants.

## Scales and analysis

The machine-readable file defines five-point ordinal anchors, forced-choice-with-tie responses, and 0–100 confidence. Retain ordinal outcomes as ordinal in analysis rather than silently treating every rating as interval data.

Run a pilot first. Determine the confirmatory sample size and power after estimating task difficulty, participant variance, item variance, and clustering. Use mixed-effects or hierarchical ordinal models with participant, task, source lineage, and relevant style/tradition strata. Publish test–retest stability, inter-rater agreement, item distributions, floor/ceiling effects, and empirical factor-structure checks.

The six candidate general dimensions must be empirically validated in the actual recruited cohorts. Do not assume a single factor structure holds across tasks, modalities, languages, or musical traditions. Publish exact items, translations, randomization seeds, stimulus manifests, exclusions, missingness, abstentions, and adverse/null outcomes.

## Prospective targets

The JSON registry records these as *targets, not results*: at least 90% discrimination of unambiguous valid-change vs sham/no-op trials on held-out tasks; at least 95% detection of wrong-direction changes where direction is defined; at least 95% detection of protected mutations in the deterministic listening panel; at least 90% acceptance of independently adjudicated valid alternatives; confidence calibration that predicts correctness; and zero acceptance after veto/expiry controls. Refine thresholds and power during the pilot.

## Boundaries

This is not a universal scale of beauty, creativity, musicality, or consciousness. It does not replace relevant human judgment with a model-generated judge. It does not infer authenticity from a demographic label or one global average. Symbolic hashes, feature-response metrics, audio preference, instruction success, preservation, and agency remain separate evidence classes.
