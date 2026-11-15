# Melothaea competitive research update — Harmony and the next benchmark frontier

**Date:** 2026-11-15  
**Scope:** Audio/music editing agents, edit-session correctness, collaboration and cost  
**Status:** Research synthesis and prospective targets; no Melothaea competitive result is claimed

## 1. The most relevant current research comparator

The newly released paper [Harmony: Bridging Audio Understanding and Editing through Agentic Collaboration](https://arxiv.org/abs/2611.14724) (Tong & Tian, submitted 2026-11-10) is now a particularly important comparator. It combines an LLM planner, Qwen3-Omni audio observation and external analysis tools for event grounding, followed by bounded, evidence-guided corrections to separated or generated audio.

On 420 eligible MMAE examples across addition, removal, replacement and volume adjustment, Harmony reports pooled:

| System | IFR | CR | EMR |
|---|---:|---:|---:|
| Harmony | 62.18% | 87.71% | 22.62% |
| WavCraft original | 58.31% | 78.11% | 13.57% |
| WavCraft matched backbone | 52.17% | 84.43% | 12.38% |
| Audio Editing Challenge Agent baseline | 44.36% | 74.74% | 8.33% |

Harmony reports 95 of 420 examples passing every instruction and preservation rubric. Its per-operation EMR is 31.30% addition, 22.82% removal, 22.78% replacement and 9.09% volume adjustment. The volume-adjustment result is a clear remaining weakness even in the current research leader.

This is an audio-editing benchmark subset containing examples that may include speech; it is not a music-only comparison and is not the same evaluation scope as the full 2,000-example MMAE development set or the new 500-example Audio Editing Challenge Agent test track. Melothaea must report exact sample IDs and cohort scope before comparing numbers.

## 2. What Harmony proves—and what it does not

The strongest architectural lesson is not “use more agents.” It is the coupling of target grounding, temporal evidence and bounded corrections. That confirms that a static text prompt sent to a powerful generator is not enough. The requested event, its boundaries, the operation, and the preserved context must remain linked through execution.

The paper also identifies several openings:

- **No multi-step session support.** Harmony explicitly lists sequential multi-edit requests as unsupported. It is therefore a useful single-step comparator, not a baseline for eight-step session integrity.
- **Cost accounting is incomplete for lifecycle comparisons.** Reported mean first execution is 105.30 seconds per input, and 173.08 seconds for replacement. These figures include unsuccessful first executions, but exclude subsequent offline repairs, evaluation, queueing, service startup and initialization. Hardware and service load are not standardized.
- **Reranking and selection are already present inside commercial/open model components.** Harmony reports SAM Audio's internal eight-candidate reranking. Evaluating only the final candidate hides a consequential source of selection pressure.
- **The user study is useful but small.** Sixteen participants judged 16 curated examples (128 overall-editing judgments and 120 preservation judgments). This is an early listener signal, not broad evidence of sustained musician preference.
- **Transition quality remains a limitation.** The paper notes that edits may not blend naturally and can create abrupt transitions.
- **Controller decisions are fixed-rule rather than learned.** The authors explicitly identify adaptive tool selection and correction as future work.

These limitations make EVAL-042 and EVAL-043 strategically complementary: EVAL-042 measures selection-conditioned evaluator exploitability; EVAL-043 measures whether a session remains correct as operations accumulate, commute, conflict, undo and merge.

## 3. The product landscape is broader than generation

[Suno Studio 2.0](https://suno.com/blog/studio-2), announced 2026-08-13, is a production workspace with timeline MIDI import/record/edit, a chat interface, audio effects, wavetable synthesis, plugin design and automation. [Moises Studio](https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/) launched 2026-09-02 as a browser-based collaborative workspace for multitrack recording, MIDI, arrangement and AI tools. Both are direct product competitors for creative workflow and collaboration, not merely text-to-song competitors.

Melothaea should not attempt to win by rebuilding every DAW feature or by promising a better foundation generator. The more defensible layer is the **verified musical change system** that can sit alongside these workspaces:

1. A musician's instruction becomes a typed musical operation with explicit target support, intent, preservation obligations and authority.
2. A candidate generator or DAW plugin is a replaceable execution backend.
3. Every candidate is bound to exact input, output bytes, backend/renderer version, event lineage and evidence.
4. Search is calibrated against hidden human/ground-truth evaluation rather than allowed to optimize the public proxy unchecked.
5. Multi-step sessions preserve ancestry, operation semantics, undo/redo and branch history.
6. Commit, export, publication and sharing remain musician-authorized decisions.

The product value is not “AI did something musical.” It is “the musician can see what changed, what did not change, what remains uncertain, and can reverse or branch the result.”

## 4. Matched single-step benchmark targets

Use the Harmony 420-example cohort as a comparator only if the exact same sample IDs, source assets, operation mix and evaluator rubric can be reproduced. Freeze model/renderer versions, prompts, compute budgets and judge versions. Report task-macro and pooled averages, duration cohorts, and each operation separately.

Prospective goals for a confirmatory run:

| Measure | Harmony reported | Melothaea target | Qualification rule |
|---|---:|---:|---|
| Pooled EMR | 22.62% | **≥30%** | Exact matched 420-example cohort |
| Pooled IFR | 62.18% | **≥65%** | Same rubric and sample weighting |
| Pooled CR | 87.71% | **≥90%** | No instruction-following gain purchased by collateral damage |
| Operation EMR | 31.30 / 22.82 / 22.78 / 9.09% | **At least Harmony in every operation; improve the weakest operations first** | Report addition, removal, replacement and volume separately |
| Listener preference | 69.53% overall edit; 56.67% preservation for Harmony in a small curated study | **≥60% pairwise preference with 95% CI lower bound >50%** | Blinded, randomized, preregistered study; powered sample size and separate preservation question |

These are research targets, not current results. A single pooled average is not sufficient; we must show no regression on longer recordings, replacement and volume adjustment. The current paper's per-operation results make volume adjustment and replacement priority areas.

The official [Audio Editing Challenge leaderboard protocol](https://audio-editing-challenge.github.io/leaderboard/) is a second track: it uses separate 500-example held-out sets for Single Model and Agent tracks, primary ranking by EMR, then IFR and CR. Do not mix its 500-example hidden cohort with the 420-example Harmony study or full 2,000-example MMAE development scope.

## 5. The new multi-step benchmark frontier

EVAL-043 now has a machine-checkable symbolic contract for edit sequences of length 1/2/4/8, including exact ancestry, support locality, hard-gate survival, operation commutativity, undo/redo, branch/merge and explicit conflict resolution.

That creates a measurable track which the current Harmony paper says it does not yet support. Proposed next-stage qualification:

| Measure | Prospective target |
|---|---:|
| Per-step hard-obligation success on confirmatory sessions | ≥98% |
| Complete eight-step session acceptance | ≥85% |
| Unauthorized symbolic event changes on deterministic fixtures | 0 |
| Hard-gate / rights / integrity false accepts on frozen negative controls | 0 at every step and tested candidate count |
| Exact undo/redo state restoration | 100% on deterministic symbolic fixtures |
| Declared commuting operation convergence | 100% on deterministic symbolic fixtures |
| Non-commuting conflict handling | 100% explicit conflict or resolution; no silent winner |
| Human-rated preservation and musical continuity | Preregistered non-inferiority versus matched single-step output, plus session-level preference study |

A session must be counted as accepted only if every required step passes. Report intermediate failures, repairs, abstentions and costs; do not allow a strong terminal result to hide a failed intermediate invariant.

The symbolic tests do not establish audible continuity. The next real-audio cohort must measure transient preservation, phase/time alignment, loudness, timbre, musical plausibility, edit-boundary artifacts and cumulative perceptual drift, with human review on licensed, consented recordings.

## 6. Cost and efficiency: change the denominator

Harmony's reported 105.30 seconds per input is an informative reference, but it is not a complete lifecycle cost because offline repairs and evaluation are excluded and hardware/service load is not standardized.

Melothaea's headline efficiency metric should be:

**Fully loaded cost per independently verified accepted edit/session.**

Record generation calls, candidate count, render calls, retries, failed and unknown outcomes, reconciliations, repair rounds, evaluation calls, human interventions, wall-clock p50/p95, GPU/CPU time, memory, energy and monetary service cost. Include abandoned sessions in the denominator and cost. Unknown provider outcomes must be reconciled before retrying.

Prospective goal: reduce fully loaded cost per verified accepted task by at least 30% against a matched baseline at equal or higher human-rated quality and identical hard-gate coverage. Compare N=1/2/4/8/16/32/64/128 search ladders; no best-of-N quality gain counts if it is purchased with unreported attempts or increased false acceptance.

## 7. Paradigm shifts to prioritize

### A. From audio blobs to semantic change sets

Represent edits as event-addressed patches over stable event IDs, parts, time spans and acoustic support. The patch declares what may change and what must remain invariant. This underpins support-local rendering, reversible operations and precise review.

### B. From linear edit stacks to an operation algebra

Classify operations by read/write sets, commutativity, preconditions and conflict semantics. Independent edits should converge in either order. Non-commuting edits should surface explicit conflicts. Undo/redo and branch merge should preserve both parent histories.

### C. From reranking to selection-aware assurance

Qualify the selector as part of the system. A metric that looks calibrated on random candidates may fail after repeated optimization. Preserve hidden labels, test increasing N and adaptive search, and measure selection-conditioned calibration and false acceptance.

### D. From “AI collaborator” to musician-governed agency

Support musician-owned branches, explicit approvals, reversible proposals, provenance and multiple valid aesthetic directions. The system should not collapse plural artistic preferences into one aggregate reward or treat a high proxy score as permission to commit.

### E. From one final score to evidence-carrying session receipts

Every operation should leave a replayable record of intent, authority, exact inputs, attempts, execution, observation, independent verification and resulting state. A session should be reviewable and reproducible rather than merely impressive in a demo.

## 8. Execution order

1. Finish EVAL-042 exact-head qualification and retain its frozen exploit controls.
2. Finish EVAL-043 symbolic operation algebra and verify the exact-head Rust workflow.
3. Add audio-boundary and perceptual continuity measurements to the EVAL-037 validity envelope.
4. Bind EVAL-041 actual-byte realization evidence to each EVAL-043 step; EVAL-040 remains terminal authority.
5. Re-run the same 420 MMAE IDs against Harmony-compatible operation scope, then submit to the Audio Editing Challenge Agent track using its official held-out test workflow where eligible.
6. Add the licensed/consented 1/2/4/8-step real-audio cohort and blinded musician panel.
7. Publish a reproducible cost-per-verified-edit comparison with failed attempts and repairs included.

## Sources

- Tong & Tian, [Harmony: Bridging Audio Understanding and Editing through Agentic Collaboration](https://arxiv.org/abs/2611.14724), 2026-11-10.
- [Audio Editing Challenge: leaderboard and evaluation protocol](https://audio-editing-challenge.github.io/leaderboard/).
- Suno, [Introducing Suno Studio 2.0](https://suno.com/blog/studio-2), 2026-08-13.
- Moises, [Moises Studio launch announcement](https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/), 2026-09-02.
