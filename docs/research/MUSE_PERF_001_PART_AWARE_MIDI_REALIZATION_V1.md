# MUSE-PERF-001 — Part-aware, provenance-bound MIDI realization

**Status:** Research/implementation draft  
**Date:** 2026-11-10  
**Implementation:** \`crates/domains/symthaea-muse/src/midi_performance.rs\`  
**Feature gate:** \`theory\`  
**Scope:** Symbolic Score → validated orchestration → MIDI 1.0 + provenance sidecar

## 1. Objective

Make symbolic realization preserve the identities and relationships that the composer authored. MIDI export must not reconstruct parts from register, pitch, velocity, VoiceRole, channel, or similarity. It must consume the persistent PartIds in the exact source Score and the digest-bound, time-indexed orchestration plan.

The output is two linked artifacts:

1. A deterministic Standard MIDI File (Type 1) carrying the performed note events and explicit General MIDI program changes.
2. A sidecar manifest binding the exact source Score, orchestration timeline, instrument-program map, MIDI bytes, tracks, note ordinals, and exact MIDI ticks.

This is an engineering integrity increment. It does not prove that an external synthesizer realizes the intended timbre, that a performer can play the part, or that the resulting audio is aesthetically successful.

## 2. Execution contract

The intended pipeline is:

\`Generator-authored Score\`  
→ canonical Score SHA-256  
→ \`OrchestrationTimelineV1\` validation  
→ explicit instrument-instance / renderer / patch bindings  
→ exact beat-to-tick conversion  
→ multi-track MIDI 1.0  
→ content-addressed realization manifest

The exporter computes the canonical Score digest using the existing score-demand serialization recipe: SHA-256 of the exact \`serde_json::to_vec(Score)\` bytes. It checks the caller's expected digest and validates the timeline against that exact Score.

### Hard invariants

- Every sounding ScoreNote has a real generator-authored PartId. Unassigned identities cause refusal; VoiceRole is not a substitute.
- Every note is contained in exactly one time-bounded instrument segment for its PartId.
- Handoffs must already be explicit and valid in the orchestration timeline.
- Every used instrument instance has exactly one explicit MIDI binding. Duplicate, missing, malformed, or unused bindings are rejected.
- GM program numbers are limited to 0–127.
- Different instrument instances receive different pitched MIDI channels in V1. Channel 10 (zero-based 9) is reserved for conventional percussion.
- Note onset and end times are converted to 960 PPQ only when the rational value is exactly representable. There is no silent rounding or quantization.
- Zero-velocity note events are rejected rather than turned into MIDI NoteOn events with velocity zero (which MIDI receivers interpret as NoteOff).
- Every source note receives one note receipt containing its original score-array ordinal, PartId, segment, instrument instance, track, pitch, velocity and exact start/end ticks.
- Source note ordinals are scoped to the Score digest. They are not persistent EventIds; event correspondence after edits must use the explicit event-remap contract.
- The manifest includes score, timeline, binding, MIDI and manifest digests. Frozen inputs must produce byte-identical MIDI and manifest output.

### Explicit V1 limits

- Standard MIDI 1.0 is used, at 960 ticks per quarter note.
- Fifteen pitched channels are available because channel 10 is reserved for conventional percussion.
- The legacy Score schema represents meter as quarter-note beats per bar; the MIDI time-signature denominator is therefore emitted as 4.
- Renderer and patch identifiers are explicit routing labels. Their presence is not evidence of sample coverage, articulation support, performer playability, or acoustic equivalence.
- The exporter is not yet a MIDI 2.0/UMP realization path and does not yet emit audio stems.

## 3. Implementation and test inventory

The Rust module currently authors 12 focused tests:

1. Two same-role parts are exported as distinct generator-authored parts.
2. A cello-to-horn handoff preserves one PartId across two tracks.
3. Overlapping parts receive distinct MIDI channels.
4. A stale score digest is rejected.
5. A missing instrument binding is rejected.
6. An unassigned PartId is not reconstructed.
7. A non-integral MIDI tick is rejected rather than rounded.
8. A zero-velocity note is not silently changed into an audible NoteOn.
9. Frozen inputs produce identical MIDI and manifest digests.
10. Every source note receives exactly one ordinal receipt.
11. A GM program outside 0–127 is rejected.
12. More than fifteen active pitched instrument instances is refused explicitly.

These are authored regression tests, not executed test results. No compilation, rustfmt, Clippy, local test, or CI pass is claimed in this draft.

## 4. Benchmark protocol

The objective is not merely to show that a MIDI file can be written. The benchmark must establish that a score-native performance export preserves declared musical identity and makes unsupported realization visible.

### B0 — Part identity fidelity

**Fixture:** Multi-part scores in which multiple parts share VoiceRole, pitch register, or overlapping onset patterns. Include counterpoint and unison/doubling cases.

**Metric:** Part Attribution Accuracy = correctly preserved PartId assignments / all source notes.

**Promotion target:** 100% for scores with complete generator-authored identity; zero inferred identities. Missing identity must abstain.

### B1 — Segment and handoff fidelity

**Fixture:** Static allocations, instrument handoffs, after-rest entries, reattacks, overlapping segments, notes at shared boundaries, and notes that cross handoff boundaries.

**Metrics:** Exact segment assignment accuracy; missed/ambiguous assignment rate; unauthorized handoff count.

**Promotion target:** 100% exact assignment; zero notes silently split, dropped, transposed, or reassigned.

### B2 — Symbolic-to-MIDI event parity

**Fixture:** Quarter, eighth, sixteenth, dotted, triplet, and arbitrary rational durations, including values that are not exactly representable at 960 PPQ.

**Metrics:** Note-on/off parity; pitch parity; velocity conversion error; onset/end tick error; refusal rate for unrepresentable timing.

**Promotion target:** Zero timing error for representable events. Zero silently quantized events. Unsupported time values must be refused with a typed reason.

### B3 — Provenance completeness

**Fixture:** Frozen Score, timeline, binding set and generated MIDI bytes. Include reordered score arrays, duplicate bindings, missing bindings, changed source digests, and mutated output bytes.

**Metrics:** Source-note receipt coverage; digest verification; exact replay parity; false source correspondence.

**Promotion target:** 100% note receipt coverage; 100% digest binding; zero false correspondences; deterministic replay for frozen inputs.

### B4 — Adversarial negative controls

Include missing PartIds, stale timelines, invalid segment scopes, notes crossing handoffs, out-of-range notes, polyphony overcommitment, duplicate instrument IDs, missing renderer mappings, malformed digests, non-integral ticks, invalid GM programs, oversized metadata and channel exhaustion.

**Promotion target:** Every planted invalidity is rejected by its intended contract. No negative control may pass because an unknown value was replaced with a plausible default.

### B5 — Renderer and DAW interoperability

Export a frozen test corpus through the same manifest into at least two independent playback environments. Record the exact synth binary, soundfont/patch, version, sample rate, mastering, and output digests.

**Metrics:** MIDI parser acceptance; track/program/channel parity; renderer-specific audio differences; audible clipping/dropouts; part stem separation.

**Promotion target:** 100% structural MIDI acceptance in supported environments. Report sound differences as renderer-specific observations rather than treating byte-identical MIDI as acoustically identical.

### B6 — Edit-and-replay preservation

Apply localized edits to one part or one time interval, generate a new Score and explicit event remap, rebind material lineage, and export again.

**Metrics:** Protected event survival; changed-event precision/recall; event-map completeness; material-lineage continuity; audio-region preservation.

**Promotion target:** 100% symbolic preservation of protected events; every inserted, modified, split, merged or deleted event explicitly accounted for. Audio preservation is measured separately and must not be inferred from the symbolic receipt alone.

## 5. Competitive evaluation

### YuE2 and WildSongBench

YuE2 is a direct comparator for symbolic planning and score-guided audio realization. WildSongBench's 12 September 2026 snapshot reports 192 prompts and 17 settings. YuE2 standard scored 6.7316 SongBench average and YuE2 best-of-8 scored 6.9632; the benchmark authors caution that the results are descriptive and not evidence of universal metric superiority. Best-of-8 is a different candidate-selection budget from standard generation.

Melothaea should report separate 1-, 2-, 4-, and 8-candidate arms. Disclose generation count, compute, selection metric, selection data, and whether the final candidate was selected by a human or a model. Use the same frozen prompts and candidate budget when comparing quality. Do not claim that provenance guarantees make a composition more musical.

**Competitive target:** Match or exceed YuE2 on controlled score-realization and musicality tasks only after measured, matched-budget experiments. Establish an independent advantage on event identity, edit locality, protected-content preservation and replay receipts.

### MAGE and audio-mixture editing

MAGE is a useful comparator for multimodal generation and mixture-grounded target-source editing. It is not the same task as exact score-native editing. Evaluate MAGE-like audio editing and Melothaea score-native editing on separate but linked tasks: one measures acoustic/source editing; the other measures exact symbolic intent, event provenance, and realized audio correspondence.

### NotaGen and symbolic orchestration

NotaGen's IJCAI 2025 evaluation used 92 music-college participants and considered melodic appeal, harmonic fluency, orchestral balance, counterpoint correctness and structural coherence. It is a useful human-evaluation reference for the symbolic-composition side. It does not qualify performer-specific range, renderer patch availability, or score-to-audio event receipts.

### MMAE / Audio Editing Challenge

The Audio Editing Challenge leaderboard page states that its submission window opens on 10 November 2026. It describes a public MMAE set of 2,000 examples and 17,741 atomic rubrics, plus separate hidden 500-example Single Model and Agent tracks. The published challenge schedule lists a 25 November 2026 submission deadline, subject to the organizer's Pacific Time convention.

Use the official track definitions and evaluator. Report the official scores separately from Melothaea's symbolic provenance metrics. The challenge measures audio-edit instruction following and preservation; a perfect MIDI event receipt cannot be substituted for its acoustic rubric results.

### Music-generation and audio-editing surveys

The 2026 music-generation survey groups open problems around musicality, consistency, controllability, representation and evaluation. It recommends unified representations that bridge score events and audio, and hybrid evaluation. The AACL-IJCNLP 2026 audio-editing survey separates acoustic, semantic and instance editing. These support a multi-axis evaluation rather than one blended musicality score.

## 6. Proposed Melothaea benchmark suite

The programme should maintain three linked but separately scored suites:

| Suite | What it measures | External anchors |
|---|---|---|
| MUSE-GEN | Musicality, structure, style control, diversity, long-range motif use | WildSongBench, SongBench, NotaGen human evaluation |
| MUSE-EDIT | Intent execution, target selection, edit locality, protected-content preservation, multi-turn consistency | MMAE, Audio Editing Challenge, MAGE |
| MUSE-REALIZE | Part identity, instrument feasibility, score-to-MIDI exactness, renderer realization, event provenance | MIDI/DAW interoperability, orchestration timeline, capability evidence |

Never combine these into a single headline result without publishing all component scores and their weighting. For preference experiments, use blind randomized A/B, loudness matching, preregistered analysis, listener-level clustering, confidence intervals, and matched candidate/compute budgets.

## 7. Research paradigm shifts

### 7.1 Treat audio as the trace of a musical execution

The Score is the authored intent. The orchestration timeline is the allocation plan. MIDI is a portable performance instruction stream. Audio is a renderer-specific physical realization. Each boundary needs its own receipt and its own error model.

This changes the product from score generation to a verifiable musical digital thread.

### 7.2 Build persistent event identity, not just ordinal identity

This V1 receipt uses digest-scoped ordinals. The next evolution should promote explicit EventIds at composition time and propagate them through edits. The existing event-remap contract is the bridge; do not infer correspondence after the fact.

### 7.3 Compile intent into a constrained edit or performance transaction

A user request should become a target, a candidate plan, hard preconditions, protected invariants, a deterministic transformation, and a receipt. Rejected or unsupported requests remain visible. No-op requests should not masquerade as successful edits.

### 7.4 Search for expressive alternatives only after feasibility

The current planner kernel enumerates hard-feasible instrument assignments, not aesthetic optima. The next planner should preserve hard feasibility while exposing a Pareto frontier over idiomaticity, timbral contrast, performer burden, register balance, density, and cultural/style-specific constraints. Soft scores must never compensate for an unplayable part.

### 7.5 Separate objective truth from human judgment

Exact score digests, event maps and channel constraints are objectively testable. Musicality, idiom, emotional interpretation, cultural appropriateness, and preference require human or appropriately validated perceptual evaluation. Never turn a symbolic validator's success into an aesthetic-quality claim.

## 8. Integration sequence

1. Stack this export implementation on the time-indexed orchestration timeline contract.
2. Feed it only generator-authored PartIds and a timeline validated against the exact Score digest.
3. Replace or clearly deprecate legacy role/frequency-based score MIDI export only after migration tests demonstrate parity for supported paths.
4. Connect material lineage and event remaps so localized edits can be exported without losing source event correspondence.
5. Integrate the evidence-bound capability materializer and renderer/patch qualification registry. The current exporter only consumes the lower-level structural capability contract and explicit MIDI bindings.
6. Add renderer-specific stem export and audio observation receipts.
7. Run the frozen B0–B6 suite and publish raw fixtures, commands, exact heads, environment fingerprints, and failure receipts.

## 9. Claim boundary

This PR authors a new part-aware export path and synthetic tests. It does not yet establish compilation, runtime correctness, DAW interoperability, audio quality, instrument playability, expert preference, benchmark leadership, or SOTA performance. All such claims require the stated evidence and exact-head verification.
