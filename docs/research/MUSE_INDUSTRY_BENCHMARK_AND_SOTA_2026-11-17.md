# MUSE — Industry, SOTA and Evidence-Gated Benchmark Plan

**Snapshot date:** 2026-11-17  
**Programme:** Melothaea / Muse  
**Status:** research plan and prospective targets; no new performance result is claimed  
**Related work:** #9535 (blind preference), #9581 (score-first capability benchmark), #10417 (external SOTA ledger), #12643 (style trajectories)

## Executive position

Do not try to beat commercial music generators on a single undifferentiated leaderboard. The frontier systems are optimized to produce compelling audio quickly. Muse should compete on a complementary, measurable axis: musical intent that remains editable and traceable from the symbolic plan through score, performance and rendered audio, with proof that the requested constraints and collaborative decisions actually affected the result.

The commercial preference comparison remains necessary. It is not the same task as exact symbolic control, style-grammar realization, preservation-aware editing, persistent part identity, reciprocal collaboration, or independently replayable provenance. These should be measured in separate panels with no composite score that hides a failure in one capability behind strength in another.

## Industry snapshot

The values below are a dated external snapshot, not permanent rankings. Record the exact page, access date, model version and sample counts in each run manifest.

### Blind audio preference

Artificial Analysis instrumental leaderboard, accessed 2026-11-17:
- Suno v6: 1148 ± 22 Elo (2,208 votes/samples as displayed on the page).
- Suno v6-mini: 1114 ± 21.
- Mureka V9.5: 1084 ± 21.
- Mureka V9: 1083 ± 21.
- Lyria 3 Pro: 1080 ± 21.
- Lyria 3.5: 1078 ± 21.

Artificial Analysis vocal leaderboard, accessed 2026-11-17:
- ACE Music v1: 1160 ± 21 Elo (2,119 samples as displayed).
- Suno v6: 1144 ± 18.
- Suno v6-mini: 1132 ± 18.
- Mureka V9.5: 1101 ± 18.
- Lyria 3.5 and Lyria 3 Pro: approximately 1050 Elo on the displayed snapshot.

The arena's method is pairwise blind preference with Bradley–Terry Elo estimation and confidence intervals. Its controls include same prompt/modality comparisons, a minimum listening duration, loudness normalization and a fixed delivery format. Muse's preference sidecar must follow the same principles, publish prompt/sample counts and confidence intervals, and avoid comparing unlike modalities or durations. These scores tell us who listeners prefer in that arena; they do not score symbolic correctness or proof-carrying edits.

### Interactive generation

Google Lyria RealTime is a relevant competitor for live musical interaction: the documented interface uses a persistent low-latency bidirectional stream and supports steering such as prompt weighting, BPM, density, brightness and scale. The documented control surface also has limitations: changes to some global musical parameters such as BPM or scale require a stop/restart and context reset. Muse should measure end-to-end response latency, turn-taking contingency, whether a requested edit is actually reflected in the next phrase, and whether musical identity survives the intervention. Do not claim to beat a realtime system on latency until measured on the same hardware/network conditions.

### Editing and long-form workflows

Suno's official v6 documentation describes generations up to eight minutes and editing workflows including Extend, Cover, Replace Section, Remix & Edit and custom models. That makes long-form generation and section-level revision table stakes for user expectations. Muse's defensible differentiation is not merely having an edit button: it is a score-native edit with exact before/after correspondence, protected-region invariants, reversible operation history and a verifier that rejects changes outside the requested scope.

### Symbolic benchmark coverage

MIREX 2026 symbolic generation includes a piano-continuation task with a four-measure prompt and a twelve-measure continuation in a sixteen-measure 4/4 output, quantized to sixteenth notes. Its evaluation includes coherence, creativity, structuredness and musicality; multiple candidates are generated, with candidate selection rules set by the task. This is useful as a shared symbolic baseline, but it is narrow: one piano-oriented continuation task does not evaluate multitrack ensemble identity, time-varying grammar, causal style control or collaborative editing.

The 2026 CMI-RewardBench results are evaluator results, not music-generator results. The leading reported overall pairwise evaluator accuracy on its core set is about 73.9%; instrumental-extension accuracy is around 75.4% for the leading submission, with vocal-extension accuracy notably lower. The core cohort is small (249 examples). Treat these values as an evaluator-design reference, not a generation SOTA score. Our reward/evaluator models must be held out from generator optimization and must never be allowed to self-authorize score validity.

## The paradigm shifts

### 1. A persistent shared musical world model

Represent a piece as a versioned graph of musical entities and relationships, not a prompt string or a flat list of notes:

- motifs, subjects, transformations and returns;
- parts, roles and temporal relationships;
- rhythmic cycles, meter, pulse phase and tempo;
- harmonic/pitch-centre/tuning contexts;
- formal regions, grammar obligations and transition windows;
- instrument, articulation, performance and renderer capabilities;
- provenance, rights and human-authored constraints.

An edit becomes a graph operation with explicit preconditions, effects and preserved identities. The graph is not allowed to invent stable identity from a voice label or a similar register. Unknown identity remains unknown.

### 2. Proof-carrying composition

Build a verifiable chain:

Intent → resolved capabilities → style/grammar plan → symbolic score → independent score verification → performance plan → rendered audio → independent acoustic verification.

Every stage binds to exact upstream digests and tool/model versions. A downstream success cannot repair an upstream failure. The user receives a compact receipt showing what was requested, what changed, what stayed invariant, what was not supported, and how the output can be replayed.

### 3. Style is a trajectory, not a global genre tag

A composition can move between grammars over score time. Each interval has one structural owner; each transition declares what is preserved, transformed or released. Discrete systems such as tala, tuning, meter or instrument technique are not numerically interpolated merely because a smooth morph would be aesthetically convenient. A composed bridge has its own grammar and obligations.

### 4. Musical editing becomes a reversible semantic operation

Operations such as “make this section more sparse without changing the theme,” “move the bass entry two beats later,” or “let the second part answer the first” should compile into constrained score edits. The verifier checks exact event correspondence and protected material. Undo restores the exact prior symbolic state; redo replays the same operation and digest. A change to event ordering must produce a new source-to-result correspondence rather than reuse stale role/index references.

### 5. Collaboration is measured through reciprocal contingency

The system should not merely accept a prompt and return audio. It should model a musical partner's proposals, decisions, uncertainty, consent and intent; offer alternatives; observe the partner's actual intervention; and respond in a musically contingent way. Measure whether the model reacts to the human's phrase, leaves space, reuses or transforms their material with permission, and accepts rejection without repeatedly reintroducing it. This is a different objective from maximizing one-shot prompt adherence.

### 6. Quality-diversity rather than one beauty scalar

Maintain a Pareto archive across listener preference, structural validity, controllability, originality, style identity, interaction quality, render cost and risk. Never optimize a single taste score that rewards model-specific shortcuts or suppresses legitimate musical plurality. Report the frontier and the tradeoffs.

### 7. Cultural knowledge with governance and abstention

A similarity score or a genre label is not cultural authority. Separate historical/source-grounded claims, newly invented hybrids and musician-reviewed tradition-specific work. Use rights-cleared or participant-authored material, disclose lineage and allow musician/community reviewers to reject attribution. Expert-review-required catalogue entries remain technically flagged until actual review evidence exists.

## Benchmark programme: separate scorecards

| Track | What it measures | Core metrics | Initial objective |
|---|---|---|---|
| Audio preference | Listener preference on matched prompts | Pairwise Elo, 95% CI, win/tie/loss | Reach statistical parity with current leader on a narrow, matched instrumental track before making a general claim |
| Symbolic control | Exact constraints and edit intent | Exact-match rate, boundary error, protected-event violations | 100% on deterministic fixtures; ≥95% holdout plan adherence |
| Grammar/form | Source, bridge and target grammar obligations | Obligation precision/recall, pass rate per family | 100% hard fixtures; ≥95% preregistered held-out realization |
| Identity continuity | Motif and persistent part identity | Precision/recall, transformation correctness, lineage validity | 100% accounting on accepted fixtures; no inferred identity |
| Causal control | Requested axis changes and non-target invariants | Target-axis effect, non-target disturbance, paired effect size | Every admitted gene has a same-seed counterfactual; zero inert genes counted as applied |
| Selective competence | Unsupported and ambiguous requests | Recall, false-abstention, coverage-risk curve | ≥95% recall on deterministic unsupported/ambiguous controls; report coverage separately |
| Evaluator reliability | Whether independent assessors agree with expert judgments | Held-out pairwise accuracy, calibration, subgroup gaps | ≥80% pairwise accuracy target on a separate held-out cohort; never authorize correctness alone |
| Real-time collaboration | Responsiveness and turn-taking | p50/p95 end-to-end latency, contingent-response rate, interruption recovery | Publish measurements on specified hardware/network; set SLA only after baseline |
| Audio engineering | Render quality/capability | LUFS, true peak, clipping, dropouts, FAD/CLAP as secondary | Zero deterministic clipping/dropout fixtures; acoustic metrics do not replace listening |
| Evidence and rights | Replayability and legal/ethical provenance | Digest replay, lineage completeness, rights-state coverage | 100% receipts replay; zero unlicensed items in benchmark manifests |

These are prospective targets. They are not current performance claims.

## Experimental protocol

1. Pin model/API versions, dates, prompts, source datasets, rights manifests, renderer builds, hardware, software, decoding settings, random seeds, retries, candidate budgets and costs.
2. Compare identical tasks with identical modality, duration, output format, prompt context and candidate budget. Report when any dimension cannot be matched.
3. Keep all attempts in the denominator: failures, timeouts, abstentions, unparseable outputs, rejected candidates and retries. No silent dropping or cherry-picking.
4. Separate development and confirmatory test sets. Split by complete work, composer, performer, edition, source lineage, tradition/region and session as appropriate; random bars from the same source must not leak across splits.
5. Preregister the primary metrics and analysis. Use blinded preference tests, confidence intervals and enough independent judgments to support the stated interval width. Report effect sizes, not only a rank.
6. Use original synthetic known-answer fixtures for hard invariants and rights-cleared/participant-authored material for external listening.
7. Keep generator, reward model, validator and expert adjudication independent. A generator-authored assertion is never evidence that its own constraint was met.
8. Publish per-example receipts and aggregate scorecards. Include negative controls and mutation tests so a broken verifier is detected.

## Near-term execution sequence

### M0 — evidence integrity (current PR #12645)
- Exact-rational trajectory envelope.
- Catalogue-backed style/grammar route resolution.
- Recomputed source-score digest.
- Recomputed built-in profile-configuration digest.
- Explicitly retain expert-review and unsupported status.
- Tests authored; compile/test evidence remains pending.

### M1 — two-family symbolic realization
- Start with Classical (PeriodSentence) → Minimalism (ProcessAdditive).
- Both have explicit typed grammar dispatch and 4/4 meter. Their configured tempo ranges overlap; the M1 runner must compute a common tempo from one frozen intent and assert exact equality instead of silently retiming one side.
- Realize each segment through its real typed engine; retain GrammarPlanEvidence and independently validate each segment score.
- Bind actual generated score durations to exact trajectory intervals. Concatenation must preserve event data and must not fabricate persistent PartId or motif identity.
- Until the Score schema supports time-varying tempo/meter, reject cross-segment tempo or meter differences. Do not encode them as if one global Score clock could express them.

### M2 — identity and edit correspondence
- Consume PartId through Listen using the existing ownership thread (#11351); do not invent a duplicate part ID.
- Add explicit motif/subject lineage, protected events/regions and source-to-result event correspondence.
- Treat unassigned part identity as unsupported for identity-dependent claims.

### M3 — independent score verifier and causal compiler
- Verify both segment grammars and the composed result independently.
- Run same-owner/same-seed one-factor counterfactuals to show which requested musical axis changed and which protected axes stayed fixed.
- Mutation-test the verifier and all digest/receipt boundaries.

### M4 — performance and real-time interaction
- Integrate the existing shared ensemble clock and part-local performance timing work (#11355).
- Measure audio output and live interaction separately from score correctness.
- Keep renderer and symbolic receipts linked but independently qualified.

### M5 — external evaluation
- Run the score-first suite (#9581), matched 180-second blind preference suite (#9535), external arena baselines and musician-governed held-out cohorts.
- Publish separate scorecards; do not combine Elo, structural correctness, cultural review and latency into a single rank.

## Stop conditions and claim discipline

- Do not claim “better than SOTA” until a preregistered matched test has sufficient samples and a confidence interval supporting that statement.
- A pass on synthetic fixtures does not establish musicality, authenticity or listener preference.
- A typed grammar plan does not prove the actual score fulfils all obligations.
- A profile digest binds configuration, not executable implementation or perceptual efficacy.
- A model-generated receipt is not independent verification unless checked by a separate verifier.
- Unknown capability, stale evidence or ambiguous musical identity must result in abstention/rejection, not a plausible generic fallback.

## Sources

- Artificial Analysis, instrumental leaderboard: https://artificialanalysis.ai/music/leaderboard/instrumental?mode=instrumental&tab=leaderboard
- Artificial Analysis, vocal leaderboard: https://artificialanalysis.ai/music/leaderboard/vocals
- Artificial Analysis, methodology: https://artificialanalysis.ai/methodology
- Google, Lyria RealTime documentation: https://ai.google.dev/gemini-api/docs/realtime-music-generation
- Suno, v6 model and editing documentation: https://help.suno.com/en/articles/5782977
- MIREX 2026, symbolic music generation task: https://www.music-ir.org/mirex/wiki/2026:Symbolic_Music_Generation
- MIREX 2026, CMI RewardBench task/results: https://www.music-ir.org/mirex/wiki/2026:CMI-RewardBench


## Additional enabling architecture — exact score-time maps

Issue #12722 now tracks a versioned score-time map for notated tempo and meter changes. This is intentionally separate from #11355's expressive performance clock. The current M1 experiment (#12721) remains same-tempo/same-meter; a broader tempo/meter transition is not claimed until exact score-beat-to-nominal-seconds mapping, meter boundaries, legacy Score migration and performance-clock composition are independently verified.

## 12. Industry snapshot clarification — 2026-11-17

The official Artificial Analysis v1.1 boards were rechecked on 2026-11-17. The displayed instrumental leaders remain Suno v6 (1148 ± 22; 2,208 samples), Suno v6-mini (1114 ± 21), Mureka V9.5 (1084 ± 21), Mureka V9 (1083 ± 21), Lyria 3 Pro (1080 ± 21) and Lyria 3.5 (1078 ± 21). The displayed vocal leaders are ACE Music v1 (1160 ± 21; 2,119 samples), Suno v6 (1144 ± 18; 3,063) and Suno v6-mini (1132 ± 18; 3,175), followed by Mureka V9.5 (1101 ± 18).

Do not overstate the vocal ordering: ACE Music v1's 95% interval (1139–1181) overlaps Suno v6's (1126–1162). It is the current displayed point-estimate leader, not a cleanly separated statistical winner from Suno v6 under the published intervals. Instrumental and vocal tracks are separate modalities and should remain separate Muse scorecards.

Artificial Analysis's methodology page specifies same-prompt/same-modality blind pairwise choices, at least ten seconds listening to each track, Bradley–Terry estimation, an updated October 2026 confidence-interval method, static gain normalization toward −16 LUFS with a −1 dBTP ceiling, and 320 kbps MP3 / 44.1 kHz delivery. Its standard tracks are approximately three minutes, or the closest supported length, with one track per prompt. These details should be copied into a matched Muse preference protocol rather than comparing results from unlike output budgets.

Suno has advanced beyond simple prompt-to-audio generation: Studio 2.0 documents MIDI import/record/edit, chat-driven clip and instrument creation, user-designed plugins, effects, a wavetable synth and automation; October 2026 release notes add downbeat detection improvements, clip time-stretching and editable fade curves. This makes it inaccurate to describe MIDI editing, chat or timeline control as an unoccupied niche. Muse's differentiation must be exact symbolic semantics, protected-content verification, reversible score transactions, causal attribution, persistent identity and evidence that requested changes occurred without unauthorized collateral changes.

Google DeepMind describes Lyria RealTime as a continuous real-time stream with direct key, tempo, density and brightness control and 48 kHz stereo output. Lyria 3.5 is described as capable of up-to-three-minute tracks and as improving prompt adherence, audio fidelity, vocals and musicality. Treat these as vendor-described capabilities; use matched latency and blinded musical tasks for quantitative comparisons.

### Source links added to the comparison record

- Artificial Analysis instrumental leaderboard: https://artificialanalysis.ai/music/leaderboard/instrumental?mode=instrumental&tab=leaderboard
- Artificial Analysis vocal leaderboard: https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/music
- Suno Studio 2.0 official feature overview: https://help.suno.com/en/articles/13670529
- Suno Oct 8, 2026 Studio improvements: https://suno.com/release-notes
- Google DeepMind Lyria RealTime: https://deepmind.google/models/lyria/lyria-realtime/
- Google DeepMind Lyria 3.5: https://deepmind.google/models/lyria/
- MIREX 2026 symbolic generation results: https://futuremirex.com/mirex/wiki/2026%3ASymbolic_Music_Generation_Results

**Claim boundary:** all model Elo values are preference results from the listed modality-specific arena. They do not imply exact-score control, edit fidelity, continuity, reciprocal contingency, rights compliance or source provenance. No Muse head-to-head preference result is claimed here.


---

## 2026-11-18 benchmark refresh and execution priorities

**Snapshot date:** 2026-11-18. These are external reference points, not Muse results.

### Blind audio preference

Artificial Analysis v1.1 currently displays:
- **Instrumental:** Suno v6 1148±22 (2,208 samples); Suno v6-mini 1114±21; Mureka V9.5 1084±21; Mureka V9 1083±21; Lyria 3 Pro 1080±21; Lyria 3.5 1078±21.
- **Vocal:** ACE Music v1 1160±21 (2,119 samples); Suno v6 1144±18 (3,063); Suno v6-mini 1132±18 (3,175); Mureka V9.5 1101±18 (3,141).

The vocal point-estimate leader is not cleanly separated from Suno v6 under the displayed 95% intervals (1139–1181 versus 1126–1162). Keep instrumental and vocal scorecards separate. Elo is preference under the Arena protocol, not symbolic adherence, source preservation, edit locality, authorship or provenance.

### Controlled editing is now an actionable external benchmark

SMEBench (27 October 2026) supplies paired source/target edits across twelve instrumentation, articulation, dynamics and pitch operations, with a 3,600-example synthetic test set and a smaller real-audio complement. Its key methodological lesson is to score **source preservation**, **target agreement** and **textual compatibility** independently: one scalar can reverse model rankings. The dataset is licensed CC BY-NC-SA 4.0, and its incorporated MIDI/soundfont resources retain their own licenses; this does not grant commercial-use permission. Reproduce its protocol and licensing constraints before using any dataset material.

The ICASSP 2027 Audio Editing Challenge's MMAE benchmark publishes 2,000 development examples and 17,741 atomic rubrics. The evaluation separates Instruction Following Rate (IFR), Consistency Rate (CR) and Exact Match Rate (EMR); the hidden Single Model and Agent tracks have 500 examples apiece. The official timeline places the final submission deadline at 25 November 2026, 11:59 p.m. US Pacific. Registration closed 8 October, so do not assume Luminous is eligible for leaderboard submission. The public development set is still useful for internal, clearly labelled evaluation. The organizers report EMR below 5% on the public challenge task; compare only after matching track, subset, model/tool access and rubric protocol.

### Long-range symbolic generation

The 2026 CAST paper reports structural error 0.22±0.02 and chord accuracy 0.96±0.01 on its held-out GiantMIDI-Piano setup, over five runs; its MuseFormer baseline reports 0.58±0.03 structural error and 0.64±0.02 chord accuracy. These are reproduction targets, not directly comparable universal scores. Reproduce the paper's exact split, tokenization, metrics, training protocol and five-seed evaluation before claiming parity or improvement. Its piano-focused setup is not a multitrack counterpoint, cross-cultural or live-collaboration benchmark.

### Product reality

Suno Studio 2.0 (13 August 2026) already advertises MIDI editing, automation, synths, effects and chat-assisted production. A basic editor or chat-to-audio wrapper is not a durable differentiator. Muse needs demonstrable reversible semantic edits, exact protected-region preservation, event correspondence and independently replayable receipts.

### Updated measurable priorities

1. **ProcessAdditive audit:** require 100% positive fixture acceptance and 100% mutation-control rejection on the pinned generator contract; retain the 4,096-case deterministic holdout, with all examples and seed/bar matrix published. This is a symbolic correctness objective, not an art-quality target.
2. **Trajectory semantics:** require each segment to retain its grammar-specific score audit. For the current M1 pair, the Minimalism segment must carry the independent ProcessAdditive audit report; Classical remains explicitly trace-integrity only until its own score-semantic verifier is wired.
3. **Edit benchmark:** establish a rights-cleared paired edit suite with per-operation source-preservation, target-agreement and instruction-following metrics; include untouched event/region invariants, exact undo/redo and adversarial over-edit controls. Prospective initial gates: 100% exact fixture acceptance/rejection, at least 95% held-out instruction and target agreement, and at least 99% preservation of explicitly protected symbolic events; publish per-operation confidence intervals and reject any benchmark result that hides over-edits.
4. **Causal controls:** run same-intent/same-seed one-factor interventions. A requested dimension counts as applied only when its intended symbolic or acoustic effect is detected while protected dimensions remain within preregistered tolerance.
5. **Human collaboration:** separately measure contingent response, human-material preservation under consent, acceptance/rejection recovery, edit reversibility and listener/musician preference. Publish confidence intervals and per-cohort differences.
6. **Engineering:** report p50/p95/p99 planning, audit, render and response latencies, plus resource/cost budgets. Never mix latency and preference into the same rank.

**Claim boundary:** these additions define targets and external comparison protocols. They do not establish benchmark wins, human preference, culture-specific authenticity, audio quality or executed test results.

### Sources (checked 2026-11-18)
- Artificial Analysis instrumental: https://artificialanalysis.ai/music/leaderboard/instrumental?mode=instrumental&tab=leaderboard
- Artificial Analysis vocal: https://artificialanalysis.ai/music/leaderboard/vocals
- SMEBench (27 Oct 2026): https://arxiv.org/abs/2610.36715
- ICASSP 2027 Audio Editing Challenge / MMAE: https://audio-editing-challenge.github.io/
- CAST (Scientific Reports, 14 Apr 2026): https://www.nature.com/articles/s41598-026-46750-0
- Suno Studio 2.0 official notes: https://help.suno.com/en/articles/13670529
