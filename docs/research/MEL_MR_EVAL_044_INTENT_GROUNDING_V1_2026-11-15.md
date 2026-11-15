# MEL-MR-EVAL-044 — Grounded instruction-to-edit-plan contract v1

**Date:** 2026-11-15  
**Status:** Source, schemas and deterministic tests authored; exact-head Rust qualification pending  
**Issue:** [#12367](https://github.com/Luminous-Dynamics/symthaea/issues/12367)  
**Stack base:** EVAL-043 [PR #12245](https://github.com/Luminous-Dynamics/symthaea/pull/12245)

## 1. Research motivation

The newest directly relevant agent paper is Harmony (Tong & Tian, arXiv:2611.14724, 2026-11-10). On 420 eligible MMAE cases across addition, removal, replacement and volume adjustment, it reports pooled IFR 62.18%, CR 87.71% and EMR 22.62% (95/420 complete examples). Original WavCraft reports 13.57% EMR and the Audio Editing Challenge Agent baseline 8.33% on this same subset. Harmony uses collaborative planning with Qwen3-Omni and analysis tools to resolve targets and timing, then bounded corrections based on intermediate outputs. It retains an internal N=8 SAM Audio reranker.

Harmony's strongest lesson is that target grounding and intermediate evidence matter. Its remaining boundary is equally important: the paper describes a single-step system and lists sequential multi-edit instructions as future work. Its first full executions average 105.30 seconds per input (4.25 LLM calls); replacement averages 173.08 seconds and 7.04 calls. The timing excludes evaluation, queueing, service startup and initialization, and hardware/service load is not standardized. A small user study (16 participants, 16 curated cases) favoured Harmony, but is not a large confirmatory cohort.

Melothaea should not claim that no one has built an agentic planner. The differentiator should be a machine-checkable compiler boundary: raw natural language is not executable. The planner must emit a grounded, typed operation whose source, target occurrences, temporal scope, exact event support, units, preservation complement and ambiguity disposition all replay against a frozen source observation.

## 2. Industry positioning

- **Suno Studio 2.0** (announced 2026-08-13) provides a browser-based generative production workspace with MIDI tracks/piano roll, audio effects, wavetable synth, automation, generated plugins and chat-based arrangement/generation. It is a serious direct competitor in creative workflow and production UX, not only text-to-song generation. Official docs: https://help.suno.com/en/articles/13670529
- **Moises Studio** (announced 2026-09-02) offers shared multitrack browser sessions, MIDI transcription, real-time collaboration, agentic actions that are visible and reversible, DAWproject/export workflows and 30+ stem categories. Its product positioning explicitly emphasizes musician authorship and licensed training material. Official announcement: https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/
- **Google Lyria RealTime** provides low-latency bidirectional streaming generation, weighted-prompt steering and controls such as BPM and density. It is a useful generator/steering backend comparator, not evidence of exact edit correctness or rights-aware session history. Official docs: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- **AA-Music-Instrumental v1.1** is a generation-preference arena, not an editing benchmark. The 2026-11-15 visible snapshot lists Suno v6 at Elo 1148 (95% interval 1126–1170), followed by Suno v6-mini 1114, Mureka V9.5 1084 and Lyria 3 Pro 1080. Those ranks must not be reported as edit-success metrics. Source: https://artificialanalysis.ai/music/leaderboard/instrumental

The commercial baseline is therefore broader than a model leaderboard: musician collaboration, fast production, editable MIDI, stems, effects, agentic workflow and export interoperability already exist. Melothaea should interoperate with this ecosystem rather than rebuild every DAW primitive.

## 3. Benchmark limitations to preserve

SMEBench (arXiv:2610.36715, 2026-10-27) contributes 3,600 balanced synthetic-MIDI test cases over 12 operations and a separate 416-pair real-audio complement. Its human validation had 236 ratings from 17 participants: mean target quality 4.23/5, source alignment 4.65/5 and instruction alignment 3.99/5. Three operations (Legato, Remove Instrument and Swap Drums) did not reliably exceed the midpoint on instruction-following. Its own risk statement notes possible Western-centric bias. It is valuable for paired targets and operation-specific diagnostics, but does not replace real-audio, rights-cleared, musician-governed validation. Paper: https://arxiv.org/abs/2610.36715

MIREX 2026 CMI-RewardBench ranks automatic pairwise music preference predictors at 73.9% on its 249-pair core. This measures agreement with human preference between generated tracks; it is not exact edit correctness, preservation or temporal localization. Results: https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results

XACLE 2027 shifts audio-text alignment evaluation to unseen generation systems. Its test release is scheduled for 2026-11-19 and submission deadline for 2026-12-03. This is a useful near-term external metric-generalization opportunity, but it is not a music-edit benchmark. https://xacle.org/2027/

## 4. Stage 1 implementation

The standalone crate is \`tools/mel-music-edit-intent-grounding-harness\`.

### Frozen source observation

A source observation binds:
- source artifact and lineage digests;
- an upstream observation-attestation receipt digest included in the source commitment; each occurrence also binds its own upstream evidence-receipt digest;
- an upstream observation-attestation receipt digest included in the source commitment;
- source duration in integer ticks;
- canonical event inventory with stable event and part IDs;
- canonical temporal anchors (section, bar, beat, phrase, event or explicit time);
- observed occurrences with semantic class, half-open interval, event IDs and anchor IDs;
- a domain-separated SHA-256 commitment over the complete observation.

The observation verifier rejects malformed intervals, stale or unknown references, duplicate IDs, non-canonical arrays, and tampering after the commitment is created.

### Typed edit plan

A plan binds:
- instruction and source-observation digests;
- operation and exact typed parameters;
- selector semantics (unique, nth, all, or inside/after/before a temporal anchor);
- selected occurrence IDs and the union of their source event IDs;
- exact target windows;
- the exact complement of source event IDs as protected;
- one source-bound grounding-evidence record per selected occurrence;
- a terminal disposition: Ready, NeedsClarification or Abstain.

Parameters use integer values and explicit units. Stage 1 supports addition count, removal, replacement asset identity, gain in milli-decibels and transposition in semitones. This deliberately avoids silently treating dB as linear gain, cents as semitones, or bars as seconds.

### Non-executable uncertainty

A unique selector that matches multiple source occurrences must return NeedsClarification with the exact candidate list and a bounded question. A selector with no supported occurrence, or an out-of-range ordinal, must Abstain. Both dispositions are non-executable: they may not retain selected events, target windows, parameters or execution evidence. A confidence score cannot bypass this rule.

The verifier checks that a Ready plan's selection exactly matches its selector against the frozen occurrence inventory. A second entry point also compares the observation to expected source artifact, lineage, observation-receipt and observation digests supplied by the caller from its independently verified upstream context. Receipt digests are references, not cryptographic proof by themselves; this harness validates identity and binding but does not verify upstream signatures. A caller-supplied expected-source binding provides the independently verified source artifact, lineage, observation receipt and observation digest values for comparison. The receipt digests are references that bind into the content commitment; this stage validates their identity and shape but does not cryptographically verify the upstream observer or annotator signature. It recomputes event support, protected complement, target windows, parameter constraints and grounding-evidence bindings. A plan cannot gain execution authority merely by claiming high confidence.

## 5. Deterministic controls

Twenty-four unit tests are authored. They include:
- exact second-occurrence selection;
- all-occurrence selection and protected-complement computation;
- ambiguity-to-clarification and rejection of silent promotion to Ready;
- out-of-range ordinal abstention;
- temporal-anchor-constrained selection;
- stale source binding;
- expanded target interval rejection;
- omitted protected event rejection;
- event-support substitution rejection;
- parameter unit traps and valid gain units;
- replacement asset identity validation;
- missing/substituted grounding evidence, malformed receipt digests and occurrence-receipt substitution;
- source-observation commitment tampering;
- clarification-candidate substitution;
- non-executable disposition smuggling;
- confidence not overriding missing grounding;
- deterministic and scope-sensitive plan commitment;
- malformed source event-reference rejection.

The executable example demonstrates a ready "remove the second guitar entrance" plan alongside a non-executable "which guitar entrance?" clarification plan.

## 6. Prospective benchmark goals

These are targets, not measured results:

| Dimension | Target | Qualification |
|---|---:|---|
| Deterministic selector/anchor fixtures | 100% exact | Frozen source observations and known-answer plans |
| Ambiguous deterministic controls | 0 auto-executable guesses | Must clarify or abstain |
| Held-out target occurrence F1 | ≥95% | Rights-cleared musician-annotated cohort |
| Operation/parameter exact match | ≥95% | Units and operation-specific required parameters |
| Ambiguity recall | ≥95% | Adjudicated ambiguity requiring a question |
| Unnecessary clarification | ≤5% | Unambiguous controls at matched coverage |
| Unauthorized event support | 0 on deterministic controls | Exact source complement and event support |
| Risk/coverage | Better than confidence-only and audio-language-model-only | Paired held-out evaluation |

For the Harmony 420-example cohort, the broader Melothaea target remains ≥30% EMR, ≥65% IFR and ≥90% CR only on exact matched sample IDs, same judging rubric and comparable budgets. These targets are stretch goals, not results. Always report operation-level, duration-level, task-macro and pooled scores with uncertainty.

## 7. Next work

1. Connect the actual Symthaea/Melothaea planner to this schema; preserve raw instruction, parser/tool versions and candidate alternatives.
2. Add held-out compositional cases for "second occurrence", quantifiers, negation, exceptions, anchors, overlap, nested parts and unit conversions.
3. Compare tool-free, confidence-only, audio-language-model-only, evidence-acquisition and clarification-enabled policies at matched evidence-call budgets.
4. Bind verified plans into EVAL-037 validity envelopes; EVAL-040 remains the only terminal transaction authority and EVAL-041 remains the realization-evidence boundary.
5. Extend to rights-cleared real audio and musician-governed multilingual/cultural cohorts. Do not assume a Western instrument taxonomy or section vocabulary is universal.

## Claim boundary

This implementation verifies structured planner output against an explicit frozen source observation. It does not parse natural-language prompts, establish that an upstream audio model detected the occurrences correctly, render audio, measure musical quality, or authorize a transaction. Exact-head Rust qualification remains pending on the current source head; no test, format or Clippy pass is claimed yet.
