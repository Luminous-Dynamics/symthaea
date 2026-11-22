# MEL Judge Grounding Probe Qualification v1

**Date:** 22 November 2026  
**Status:** Rust reference evaluator and synthetic conformance fixture; no live judge or human study result.

## Research motivation

The October 2026 study *What Do Audio-Language Judges of Generated Music Listen For? Four Probes with Controls from Expert Ratings* (https://arxiv.org/abs/2610.13247) asks whether audio-language judges are actually using the audio, rather than prompt priors, slot cues or rubric wording. It proposes audio ablation, audio replacement under a fixed prompt, pair-order swaps and instruction rewording. Its results on MusicEval and SongEval show why aggregate human correlation alone is insufficient to qualify a judge.

MEL-JUDGE-GROUNDING-001 extends this with semantic-preserving nuisance transformations and controlled semantic mismatches. The study motivates the tests; it does not establish their outcomes for Melothaea.

## What the evaluator checks

For each case, the baseline stores pair scores and exact prompt, instruction and audio digest references. Probe executions are frozen before the independent human reference is released. Only after that release is a scoring-time panel assembled with the post-freeze human-reference direction; this assembled panel must never be passed to the judge. The probe outcomes then test:

- prompt-only margin collapse after removing audio;
- preference reversal when the two exact audio assets are exchanged under a fixed prompt;
- rank/sign stability under a distinct, human-reviewed instruction paraphrase;
- bounded pair-margin drift under declared nuisance transformations;
- a minimum score drop when exactly one candidate's audio is changed to violate a declared semantic attribute.

The report keeps each axis separate and includes case/probe findings, coverage and skipped/unevaluable counts. It also reports baseline agreement with separately supplied human labels on directional cases.

The position/order reversal protocol remains owned by MEL-JUDGE-GAP-001 (#13218); this evaluator focuses on the audio/text/rubric causal probes and their metamorphic extensions.

## Transform governance

A transform can be called nuisance only after human review confirms that it preserves the task-relevant content. Gain, codec and channel changes may not be safe in all contexts. For example, channel conversion can remove meaningful spatial cues; a tempo transformation is not a nuisance for a prompt that explicitly specifies tempo. Every transform requires exact source and variant digests plus a preregistered tolerance.

Semantic mismatch probes change only the declared candidate's audio digest while preserving prompt and instruction digests. Examples include wrong instrument, absent vocals, tempo violation or section-order mismatch. The exact perturbation and attribute must be declared before collection.

## Prospective qualification gates (not results)

- Audio swap reversal on clear pairs: >=95%, stretch >=98%.
- Prompt-only pair-margin ratio: <=0.10, stretch <=0.05.
- Instruction-paraphrase pairwise sign stability: >=95%, stretch >=98%.
- Nuisance-transform margin invariance: >=95% within preregistered per-transform tolerance, stretch >=99%.
- Semantic-mismatch detection: >=90%, stretch >=95%.
- Baseline accuracy against independent human direction on clear pairs: >=90%, stretch >=95%.
- 100% identity and digest-shape coverage; no missing probes silently counted as passes.

These values are proposed gates. Determine transform-specific tolerances from a pilot, use rater/item-clustered intervals, and hold out at least one generator family.

## Claim ceiling

The fixture is synthetic. This evaluator validates supplied probe rows and digest-shaped identities; it does not open audio, independently verify that transforms preserve semantics, authenticate human labels, or prove the judge runtime lacked access to the human reference. Live musician and judge qualification requires the sealed release sequence and an independently controlled execution boundary.


## Follow-on: absolute score invariance and transform qualification

MEL-AUDIO-METAMORPH-001 (#13260) extends the nuisance-transform probe in two independent ways. First, it records the absolute score drift for candidate A and candidate B, and gates on both the maximum candidate-score drift and pair-margin drift. This prevents a common-mode shift in both scores from passing merely because their difference is unchanged. Second, it introduces a task-scoped transformation receipt that binds source/variant file hashes, transform implementation and parameter digests, task dimensions, rights/consent receipt digests, and a separate independent equivalence result. The equivalence confidence interval must fit wholly inside a preregistered margin using a listener/item-aware method. The follow-on fixture is synthetic; it does not report a live judge or completed listener study.
