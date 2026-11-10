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

1. Recomputes the canonical score-demand SHA-256 over the exact serialized score bytes (the schema ID labels the digest contract but is not prefixed into the hash), and rejects stale or mismatched expected digests.
2. Runs the orchestration timeline validator before encoding.
3. Requires every note to have a real generator-authored `PartId`.
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

The source currently contains twelve focused regression tests covering:

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
| Stale expected score digest | Reject a changed score paired with an old digest |
| Source overlap hidden by tick quantization | Reject using exact rational source-time overlap |
| Quantized handoff collision | Reject different instrument instances collapsing to one MIDI tick |
| Parsed MIDI-to-receipt correspondence | Independently parsed SMF note-on/off and program-change events exactly match the receipt |
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

The public MMAE development set contains 2,000 examples and 17,741 atomic rubrics. The Audio Editing Challenge released its held-out challenge set and SDK on November 10, 2026; each of its Single Model and Agent tracks has 500 examples. Its primary ranking metric is EMR, with IFR then CR as tie-breakers. The published schedule lists November 25, 2026 as the submission deadline. Registration closed October 8, and the Agent Track requires model-component versions and weights to have been public before November 1. Therefore, treat official participation as conditional on an already registered team and confirmation that newly written orchestration/tooling is permitted under the model cutoff; the public development benchmark remains useful regardless. Use the official test split and SDK for any leaderboard claim; do not report development-set scores as held-out results.

Separate the constrained music-native domain from general-purpose audio editing. The official AEC/MMAE overview reports existing systems below 5% overall EMR, with complex mixed-modality tasks reaching 0% in its reported evaluations. These figures describe the broad benchmark, not Melothaea's music-only slice.

**Music-native, score-grounded editing subset** (known source score, stable part identities, explicit event mappings, supported renderer, and edits expressible in the score graph):

| Metric | Initial target | Stretch |
|---|---:|---:|
| IFR | ≥ 0.95 | ≥ 0.99 |
| CR | ≥ 0.99 | ≥ 0.995 |
| EMR | ≥ 0.90 | ≥ 0.97 |
| Protected symbolic-event violations | 0 | 0 |
| Unrequested score-event changes | 0 | 0 |

These high targets apply only to the explicitly constrained, machine-checkable subset. They are not targets for arbitrary natural-language audio edits.

**General MMAE / AEC Agent Track:** first require a statistically defensible improvement over the published agent baseline on the same official split; use ≥10% overall EMR as an ambitious first milestone and ≥20% as a stretch. Report IFR and CR by audio modality and complexity; do not claim a general-audio win from music-only results. Use the official held-out test set and SDK for leaderboard claims.

All values above are engineering targets, not current results or leaderboard claims. The published baseline numbers are not guarantees of the next challenge release's ranking; rerun the official baselines on the exact evaluation split before claiming superiority.

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

**Melothaea target:** first exceed 6.7316 on the same single-generation protocol; separately target >6.9632 on a matched best-of-8 protocol, with the same candidate count and selection rules for every comparator. A stretch objective is >7.0 SongBench average plus a statistically significant blind human preference advantage. None of these thresholds alone proves general musical superiority; publish per-metric results and confidence intervals.

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

### 5.5 A Melothaea-native Musical Digital Thread benchmark

Build a benchmark that tests the differentiating claim directly: can the system execute a musical intention while preserving identity and proving what changed?

**Corpus design**

| Cohort | Starting material | Purpose |
|---|---|---|
| Contract-generated | Programmatically generated scores, PartIds, relationship graphs, capability profiles and timelines | Exhaustive deterministic positive and negative controls |
| Symbolic works | Licensed/consented MusicXML and MIDI with human-verified part and motif annotations | Generalization beyond Melothaea-generated scores |
| Realized performances | Score-aligned stems, full mixes, renderer/patch manifests and, where available, performer/performance metadata | Cross-modal score-to-audio realization and edit preservation |

Start with at least 300 contract-generated cases spanning all supported constraints and 100 adjudicated human-authored works. Expand toward 500–1,000 edit instructions after the annotation and scorer pipeline is stable. These are proposed dataset sizes, not existing results.

**Each benchmark record should carry**

- Immutable source score and score digest.
- Persistent PartIds and source EventIds where the originating format actually supplies them; otherwise clearly marked digest-scoped ordinals.
- Motif/theme lineage and relationship-graph version.
- Explicit orchestration timeline, capability evidence references and renderer/patch qualifications.
- Natural-language intent plus a normalized edit operation/plan.
- Protected parts, events, time windows, relationships and audio stems.
- Expected changed-event set, event-remap lineage and before/after symbolic constraints.
- MIDI bytes and receipt, and—when rendered—stem/mix bytes, renderer build, patch/asset digests and render receipt.
- Rights, consent, attribution and permitted-use metadata.

**Splits must prevent leakage.** Partition by composition/work and composer/arranger, not by randomly sampled bars or fragments. Hold out selected instrument combinations, renderer/patch versions and styles to test transfer. Keep all variants of one source work in a single split. Keep private rubrics and challenge test data out of training and prompt-ranking loops.

**Adversarial and negative-control families**

1. Two independent parts share the same VoiceRole, pitch range or rhythmic pattern.
2. One PartId changes function, instrument or register without becoming a new part.
3. An imitation/canon relationship is structurally authored; a similar-looking passage is not.
4. Planned silence versus missing/incomplete notation.
5. Conflicting capability claims, stale evidence, unqualified renderer fallback or wrong patch mapping.
6. Edit boundary lies on a rational subdivision, at a handoff, or just inside a protected region.
7. Source note overlap is hidden by MIDI tick rounding; two different handoffs quantize to one tick.
8. An edit changes one motif occurrence while all other occurrences must remain untouched.
9. Multi-turn requests depend on prior receipts, undo, redo or changed preconditions.
10. Audio edit requests alter a target stem while protected stems and their musical timing remain stable.

**Report selective-risk curves.** For requests that cannot be grounded, cannot be realized by the qualified renderer, or conflict with protected constraints, measure both coverage and false-accept rate. A correct abstention is preferable to a fabricated success. Report acceptance coverage at fixed risk thresholds, not just raw success rate.

**Ablations are mandatory.** Compare the full system against variants without (a) persistent PartIds, (b) relationship graph, (c) time-indexed orchestration/capability evidence, (d) digest-bound edit preconditions/event remaps, and (e) independent MIDI parsing/receipt checks. This will show whether each architectural layer produces measurable gains rather than merely adding metadata.

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
- [Audio Editing Challenge timeline, tracks and model release rules](https://audio-editing-challenge.github.io/)
- [Audio Editing Challenge leaderboard and evaluation protocol](https://audio-editing-challenge.github.io/leaderboard/)
- [MMAE / MAGE paper](https://arxiv.org/abs/2604.09803)
- [NotaGen, IJCAI 2025](https://ijcai.org/proceedings/2025/1134)
