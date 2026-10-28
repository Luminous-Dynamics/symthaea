# MEL-MR-EVAL-004 — Edit locality and protected-property preservation

**Dataset:** `MEL-EDIT-LOCALITY-SMOKE-1.0`  
**Contract:** `edit-locality-v1.0`  
**Status:** prototype and synthetic smoke fixtures. Rust tests are authored, not executed in this environment.

## Research question

When a musician asks for one change, does the system make that change while preserving the properties the instruction did not authorize changing? This is distinct from whether the result sounds good, whether a motif is recognized, and whether the edit instruction was followed.

The contract produces two independent verdicts:

1. **Requested-property success:** did the requested change occur within its explicit tolerance?
2. **Protected-property preservation:** did every protected dimension remain within its tolerance?

A request passes only when both are true. These outcomes are not averaged into one score. A correctly transposed melody with a corrupted onset fails locality even though transposition succeeds.

## v1.0 operation scope

The Rust reference API currently covers five index-preserving monophonic operations:

- transpose all pitches by a signed semitone offset; protect onset and duration;
- scale every duration; protect pitch and onset;
- scale onsets relative to the source's first event and an explicit target start; protect pitch and duration;
- change one pitch at one event index; protect every other pitch, all onsets, and all durations;
- change one duration at one event index; protect pitch, onset, and every other duration.

Event count changes, reordered events, simultaneous notes, rests, instrumentation, velocity/dynamics, accompaniment, and rendered-audio preservation are unsupported by this contract and fail closed. New operations should be introduced as explicit contract versions with matching tests, not by broadening tolerances silently.

## Receipt and numerical rules

Each receipt records the exact edit request and thresholds, requested-property residuals, every protected-property residual summary, event-count status, verdict, and failure reasons. Pitch comparisons use MIDI semitones, onset comparisons use beats, and duration comparisons use the absolute log ratio `|ln(actual / expected)|`. Inputs must be finite; event onsets are strictly increasing; all pitch/duration transforms must remain in range. A fixed event-count and 4,096-event ceiling bound work.

## Smoke dataset

The machine-readable JSONL corpus contains 132 deterministic synthetic cases across 12 motif families. Each family has 11 cases:
- 5 valid edits: transposition, duration scaling, onset scaling, one-note pitch edit, one-note duration edit;
- 1 under-edit: the requested transposition does not happen;
- 5 collateral-change cases: timing damage, pitch damage, onset-scaling duration damage, non-local pitch leak, and non-local duration leak.

Expected verdicts are 60 accepts and 72 rejects. Families are held out by split: development 6 families / 66 cases, calibration 2 / 22, and test 4 / 44. The cases are synthetic and not human annotated. They validate contract mechanics, not generalization or listening preference.

## Benchmark protocol and targets

Report independently:
- requested-property success rate;
- protected-property violation rate, by property axis and operation family;
- false accept and false reject rates;
- unsupported/abstention rate;
- event count and alignment failures;
- p50/p95 runtime by event count;
- receipt serialization and deterministic replay success.

Proposed later targets, not achieved results:
- at least 95% requested-property success on family-held-out annotated edits;
- at most 5% protected-property violations;
- zero acceptance of predeclared critical collateral-change mutants;
- calibration only on the calibration split, with thresholds frozen before test;
- report confidence intervals and results per operation, not only an overall average.

Before generalization claims, expand to at least 50 independent motif families per split, document licensing/provenance, and obtain independent annotations for edits that lack a deterministic symbolic oracle.

## Industry/research comparison

This benchmark is a complement to, not a substitute for:
- **CMI-RewardBench (ICML 2026):** compositional multimodal instruction and music reward-model evaluation, with human-annotated preference judgments: https://proceedings.mlr.press/v306/ma26am.html
- **TimingBench (August 2026):** timestamped audio event and tempo control with detector-localization auditing: https://taan.ai/research/rushing-or-dragging/
- **MAGE (2026):** multimodal music generation and targeted mixture-grounded editing: https://research.google/pubs/mage-modality-agnostic-music-generation-and-editing/
- **Artificial Analysis Music Arena:** blind pairwise preference of normalized finished audio, with Bradley–Terry-derived ratings and confidence intervals: https://artificialanalysis.ai/methodology/music
- **2026 music-generation survey:** notes the lack of a single accepted metric for quality, control, structure and faithfulness: https://link.springer.com/article/10.1007/s10462-026-11582-x

Do not claim superiority to commercial systems from symbolic edit-locality accuracy. A fair industry comparison requires matched prompts, generation budgets and rendering settings; blind listener preference must be reported separately from symbolic correctness.

## Execution boundary

The fixture schema, family split and oracle consistency are checked independently. Rust unit tests replay the corpus and round-trip receipts, but have not been compiled or executed here because the environment lacks `cargo`, `rustc`, and `rustfmt`. Exact-head workflow records must be inspected before treating any CI result as a pass. A skipped workflow is not a pass.
