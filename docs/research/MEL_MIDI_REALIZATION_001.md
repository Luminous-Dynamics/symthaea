# MEL-MIDI-REALIZATION-001 — Part-aware score realization and preservation benchmark

**Status:** Research and implementation contract, v1  
**Owner:** Melothaea / Symthaea-Muse  
**Scope:** Symbolic score → orchestration timeline → Standard MIDI File → independently inspectable event correspondence

## 1. Why this track exists

The current legacy score exporter has a role-based path: it groups notes by `VoiceRole` and assigns a fixed lead/bass/harmony layout. That is not a valid realization boundary for a score whose generator-authored `PartId` identifies persistent musical lines. Two independent parts can both be melodic; one part can change role; a persistent part can move between instrument instances over time.

The orchestration work already supplies the missing source-side contracts:

- Score-demand digest and persistent part analysis.
- Static, evidence-informed instrument capability profiles.
- Time-indexed instrument assignment and explicit handoffs.
- Musical relationship graphs.
- Digest-bound, transactional score edits and source-to-candidate event remaps.

This track connects those contracts to an actual MIDI artifact without reconstructing part identity after the fact.

## 2. Implementation delivered in this branch

The feature is exposed under `midi_export::part_aware_v1`.

It accepts an exact symbolic `Score`, an `OrchestrationTimelineV1`, instrument capability inputs, explicit General MIDI program bindings, and the expected score digest supplied by the trusted score-demand boundary.

It then:

1. Runs the orchestration timeline validator before encoding.
2. Requires every note to have a real generator-authored `PartId`.
3. Allocates one MIDI track and stable MIDI channel per persistent part.
4. Emits explicit program changes at time-indexed instrument handoffs.
5. Preserves input-score note ordinals in a digest-scoped event map.
6. Preserves original rational onset/duration alongside quantized MIDI ticks.
7. Emits a receipt containing score, timeline and MIDI SHA-256 digests, per-part track information, per-segment program changes, and per-note source-to-MIDI correspondence.
8. Rejects cases it cannot faithfully encode instead of silently merging, clipping, stretching, or dropping musical events.

The MIDI program is explicit caller input. It is not inferred from instrument family. The `mapping_reference` is retained in the receipt but is not authenticated by this module.

## 3. Trust and scope boundaries

This is a MIDI realization contract, not yet a qualified acoustic renderer.

- The caller must supply the expected score digest produced by the trusted score-demand/provenance layer. The timeline validator checks that the timeline and caller-supplied expected digest agree with the score boundary.
- Instrument range and polyphony claims are validated as supplied inputs. This module does not authenticate their evidence registry or turn an unqualified claim into a qualified one.
- MIDI program bindings are explicit caller-provided mappings. Their references are retained, not independently verified.
- Event ordinals are scoped to the exact source score digest. They are not persistent EventIds. Persistent identity must come from the generator and the event-remap/material-lineage contracts.
- MIDI time is quantized to 960 ticks per quarter note using nearest-ties-to-even. The original rational times remain in the receipt. Notes that collapse to zero ticks are rejected.
- One stable channel is allocated per PartId. General MIDI channel 10 is reserved for percussion convention, so this v1 supports at most 15 simultaneously represented parts. Unsupported larger scores abstain rather than merge channels.
- Same-pitch overlaps on the same part/channel are rejected because a standard MIDI NoteOff identifies channel and pitch, not a unique overlapping source event.
- Standard MIDI variable-length event deltas are bounded; events beyond the supported delta range are rejected.

The next stage is renderer-backed audio realization: bind the exact renderer build, patch, asset bytes, and render environment, then produce stems and audio-to-score correspondence. A correct MIDI receipt does not prove that an audio renderer performed the notes correctly.

## 4. Acceptance tests for this implementation

The focused regression suite in the source currently covers:

| Test | Required behavior |
|---|---|
| Two parts share VoiceRole | They remain separate tracks and channels |
| Instrument handoff | Same PartId/channel; explicit program change at the boundary |
| Missing program binding | Reject |
| Missing PartId | Reject; never infer from role |
| Same-pitch overlap on one channel | Reject; never silently clip |
| Sub-tick note duration | Reject; never silently stretch or drop |
| MIDI delta beyond standard VLQ limit | Reject |
| Rational tie quantization | Nearest-ties-to-even |
| Frozen input replay | Byte-identical MIDI and receipt hashes |

**Verification status:** tests are authored but have not been compiled or executed in this environment. The focused CI workflow is included in this branch. A queued, skipped, or absent workflow is not a pass.

## 5. Melothaea identity-and-realization benchmark

We should evaluate this feature as a first-class benchmark, not only by listening to a few MIDI files.

### 5.1 Contract metrics

| Metric | Definition | Acceptance target |
|---|---|---:|
| Part identity coverage | Notes with generator-authored PartId / all source notes | 100% for accepted exports |
| Event-map coverage | Source notes with exactly one MIDI note-on/off mapping / all source notes | 100% for accepted exports |
| Event-map uniqueness | Duplicate or ambiguous source-to-MIDI mappings | 0 |
| Pitch fidelity | Exact source pitch preserved in MIDI | 100% |
| Assignment fidelity | MIDI channel/program agrees with the unique timeline segment | 100% |
| Handoff fidelity | Required program change is encoded at each segment boundary | 100% |
| Rational-time error | Absolute onset/offset quantization error | ≤ 0.5 MIDI tick per endpoint |
| Protected-event mutation | Protected events changed by an edit | 0 |
| Unrequested event mutation | Unrequested events changed by an edit | 0 |
| False feasibility acceptance | Unplayable or unqualified plans accepted as qualified | 0 |

The 100% targets are acceptance conditions for an individual export that claims success, not estimates of system-wide musical quality. When an input is outside the contract, the correct result is explicit rejection.

### 5.2 Editing metrics

For intent-directed edits, score the *requested transformation* separately from the preservation of everything else:

- **Instruction Following Rate (IFR):** fraction of requested edit rubrics actually satisfied.
- **Consistency Rate (CR):** fraction of unrelated-content and quality-preservation rubrics satisfied.
- **Exact Match Rate (EMR):** fraction of samples for which every instruction and preservation rubric passes.
- **Protected invariant violation rate:** count of protected parts/events/time regions changed, divided by protected items.
- **Edit locality precision/recall:** changed source events that were intended to change, and intended changes that were actually applied.
- **Abstention precision:** fraction of rejected requests that were genuinely unsupported or contradictory.

The public MMAE development set contains 2,000 examples and 17,741 atomic rubrics. The Audio Editing Challenge announced a held-out challenge set of 500 examples each for its Single Model and Agent tracks, with submissions opening November 10, 2026. Its primary ranking metric is EMR, with IFR then CR as tie-breakers. Use the challenge's official test split and SDK for any leaderboard claim; do not report development-set scores as held-out results.

Initial Melothaea engineering targets on the public development protocol:

| Metric | Initial target | Stretch |
|---|---:|---:|
| IFR | ≥ 0.90 | ≥ 0.95 |
| CR | ≥ 0.95 | ≥ 0.98 |
| EMR | ≥ 0.80 | ≥ 0.90 |
| Protected invariant violations | 0 | 0 |

These are targets, not current results and not claims about the current leaderboard.

### 5.3 Finished-music quality

Keep raw-generation quality separate from controllability and evidence quality.

Report:
- SongBench average and each component dimension.
- SongEval average and musicality separately.
- AudioBox production quality.
- MuLan and AllMusicCaps text/audio alignment.
- Q3O prompt control.
- Phoneme error rate (PER).
- Blind human preference, including no-preference responses.
- Long-range structure, motif return, harmonic/contrapuntal correctness, orchestration balance and performance playability.

WildSongBench's September 12, 2026 snapshot evaluates 192 prompts and 17 settings. It reports YuE2 standard at 6.7316 SongBench average and YuE2 best-of-8 at 6.9632. The best-of-8 selection procedure is different from the standard setting and other systems' selection budgets are not uniformly matched. Report single-generation and best-of-k results independently.

### 5.4 Orchestration and score quality

Add a deterministic synthetic suite and an adjudicated human suite:

- Distinct parts that share a VoiceRole.
- A single part that changes musical function.
- Imitation, canon, call-and-response, doubling and planned silence.
- Instrument changes at exact rational boundaries.
- Range, tessitura, polyphony and articulation conflicts.
- Conflicting evidence claims and renderer/patch mismatches.
- Source edits that split, merge, delete, insert and transform events.
- Protected motif/part/time-region counterfactuals.

Track:
- hard-constraint false-accepts (target 0);
- feasible-plan recall;
- assignment optimality only after feasibility is satisfied;
- blind preference for expressive alternatives among feasible plans;
- notation correctness and playability assessed by qualified musicians.

NotaGen is a useful symbolic-composition comparator: its 2025 work reports pretraining on 1.6M ABC pieces, fine-tuning on about 9K classical compositions, and subjective A/B assessment involving 92 music-college participants. It evaluates musical and orchestral properties such as melody, harmony, orchestral balance, counterpoint and structure. Melothaea should include those human dimensions rather than substituting a single embedding similarity score.

## 6. Fair comparison protocol

For every model and pipeline, record:

- exact model/checkpoint, renderer/decoder and version;
- prompt, score, conditioning input and fixed seed;
- number of generated candidates;
- candidate-selection method and metrics used for selection;
- inference time, accelerator, peak memory and total compute;
- post-processing, ASR passes, normalization and any human intervention;
- source and output artifact hashes.

Run two separate protocols:

1. **Single-pass:** one generation, no hidden candidate reranking.
2. **Matched best-of-k:** identical candidate counts and selection budgets for every system.

Use paired prompts and identical evaluation subsets. Publish per-prompt data, confidence intervals and failure cases. Small aggregate score differences must not be called statistically significant without an appropriate test.

## 7. Competitive positioning

| System family | What it establishes | Melothaea response |
|---|---|---|
| YuE2 / SheetSage2 | Symbolic planning, score editing, covers and audio realization | Preserve source identities, edit preconditions, protected invariants, assignment evidence and execution receipts end-to-end |
| NotaGen | Strong symbolic classical composition and human evaluation | Add performer/renderer feasibility, relationship realization and reversible editing |
| MAGE | Multimodal conditioning and mixture-grounded audio editing | Ground each edit in explicit score parts/events and report what changed and what remained invariant |
| Suno / Mureka | Strong finished-song quality and production | Compete on quality under matched budgets, while differentiating on inspectability and precise control |
| Audio Editing Challenge / MMAE | Instruction following plus preservation scored by IFR, CR and EMR | Use the same rubric discipline and add score-native identity and provenance checks |

The opportunity is not to assert that no competitor can preserve content. YuE2 already reports score-guided editing; MAGE demonstrates mixture-grounded editing. The opportunity is to make the entire transformation chain machine-auditable and independently replayable.

## 8. Next implementation sequence

1. **MIDI realization:** finish source-level review and run the exact-head focused Rust tests.
2. **Renderer qualification:** bind renderer build, patch, SoundFont/sample asset digest, license/consent and environment to each instrument realization.
3. **Stem realization:** render one stem per persistent part, preserving part/channel mappings.
4. **Audio event correspondence:** align score-note ordinals to actual rendered attacks/releases; report missed, extra, shifted and altered events.
5. **Transactional audio editing:** apply localized edits to selected parts/stems, preserve protected stems/regions, and produce before/after hashes plus event-remap receipts.
6. **Benchmark adapters:** MMAE/AEC editing rubrics, WildSongBench generation, and blinded orchestration/notation studies.
7. **Independent verifier:** a small read-only Rust tool that recomputes all digests and parses the MIDI without trusting the exporter receipt.

The paradigm shift is from “generate a plausible recording” to “execute a musical intention and prove, within stated limits, what was realized and what was preserved.”

## References

- [YuE2 project and technical updates](https://github.com/elvisxu/yue)
- [YuE2 project results and benchmark overview](https://map-yue2.github.io/)
- [WildSongBench results, September 12, 2026 snapshot](https://huggingface.co/datasets/m-a-p/WildSongBench/blob/main/benchmark/results.md)
- [Audio Editing Challenge leaderboard and evaluation protocol](https://audio-editing-challenge.github.io/leaderboard/)
- [MMAE / MAGE paper](https://arxiv.org/abs/2604.09803)
- [NotaGen, IJCAI 2025](https://ijcai.org/proceedings/2025/1134)
