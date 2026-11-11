# MUSE-STEM-001 — Part-isolated audio evidence contract (V1)

**Status:** research contract and schema; no runtime validator or renderer integration is claimed  
**Date:** 11 November 2026  
**Parent:** [#11389 — verify rendered audio against the symbolic score](https://github.com/Luminous-Dynamics/symthaea/issues/11389)  
**Builds on:** [PR #11541 — part-aware MIDI realization](https://github.com/Luminous-Dynamics/symthaea/pull/11541), [PR #11564 — provenance-bound audio pitch observations](https://github.com/Luminous-Dynamics/symthaea/pull/11564)

## Objective

Define a reproducible, machine-validatable bundle for independently rendered, part-isolated audio. Each audio stem must be traceable to exactly one source-bound MIDI track, persistent `PartId`, orchestration segment, instrument instance, renderer binary, patch asset, and render configuration.

The purpose is not to certify a renderer by looking at its own output. It is to create the smallest useful unit for independent acoustic inspection, so a simultaneous note in another part cannot silently supply the spectral energy used to support this part's expected note.

## Why this is the next increment

The current audio observation slice (#11564) measures expected-pitch and harmonic energy inside an event window. Its own protocol correctly withholds claims of note attribution: energy may come from another simultaneous note, an octave harmonic, resonance, or reverb. It recommends per-part bounces where practical.

The repository now also has a model-agnostic MIDI transcription evaluator (#11597) and a broader score-to-audio conformance programme (#11389). This contract connects those efforts without duplicating either scorer:

- #11541 provides persistent part/segment identity and a source-note-to-MIDI event map.
- #11564 provides a pitch-energy probe over exact WAV bytes.
- #11597 provides independent MIDI transcription scoring.
- #11389 owns the complete score-to-audio conformance decision.

MUSE-STEM-001 defines the artifact boundary between them. It does not claim that a separate WAV file automatically proves its own provenance or musical correctness.

## Paradigm shift: mix-first evaluation → part-addressable realization evidence

A stereo mix is a many-to-one projection. Once parts are mixed, a peak near a target frequency cannot reliably identify which contributor produced it. A high harmonic in one instrument can resemble the fundamental of another; simultaneous unisons are acoustically non-identifiable from the mixture alone.

For deterministic qualification fixtures, render each MIDI track as a separate full-length WAV against the same sample clock. Keep the original multi-part mix as a separate artifact. The verifier then has three different evidence objects:

1. **Symbolic intent:** the frozen Score, part-aware MIDI, and manifest.
2. **Part-isolated execution:** one full-length WAV per (PartId, segment, instrument instance).
3. **Combined realization:** the independently retained mix, where mix-level properties can be evaluated without being confused with per-part source evidence.

The isolated stem is not source separation. V1 requires a dedicated single-track render input. A neural separator, DAW extraction from a finished mix, or manually edited stem may be useful research data, but it is a different evidence class and must not be admitted as a V1 isolated render.

## V1 data contract

The machine-readable contract is `docs/research/schemas/muse-part-stem-evidence-v1.schema.json`.

### Bundle identity

The bundle binds:

- canonical Score SHA-256;
- exact part-aware MIDI byte SHA-256;
- exact MIDI manifest SHA-256;
- one common audio clock;
- the ordered set of isolated part/segment stem records;
- a domain-separated bundle commitment.

Every digest is a lowercase 64-character SHA-256 hex string. The digests detect substitution relative to a retained manifest; they are not signatures, trusted timestamps, or proof of who produced the artifacts.

The bundle commitment uses domain identifier `muse-part-stem-evidence-serde-json-sha256-v1`. Construct a typed payload containing, in schema field order, `schema`, `score_sha256`, `midi_sha256`, `midi_manifest_sha256`, `audio_clock`, and `stems`; omit `bundle_sha256`. Sort stems by `(part_id, segment_id, instrument_id, midi_track_index)`, serialize the payload with `serde_json::to_vec`, then calculate SHA-256 over `domain || 0x00 || payload_bytes`. The validator must recompute this value; a syntactically valid digest is not evidence that the bundle commitment is correct.

### One stem record

Each record names exactly one:

- `part_id`;
- `segment_id`;
- `instrument_id`;
- source MIDI `track_index`;
- renderer identity and renderer-binary digest;
- patch identity and patch-asset digest;
- render-configuration digest;
- canonical track-event-set digest;
- output WAV digest;
- complete, sorted source-note ordinal set.

A persistent part that changes instrument has one record per orchestration segment. Do not merge the segments into one record and lose the handoff boundary.

### Canonical track-event-set commitment

The track-event-set digest uses the domain identifier:

`muse-midi-track-event-set-json-sha256-v1`

It is computed over canonical UTF-8 JSON with these fields in this exact logical structure:

```json
{
  "schema": "muse-midi-track-event-set-v1",
  "score_sha256": "<score digest>",
  "part_id": 0,
  "segment_id": "segment-0",
  "instrument_id": "instrument-0",
  "midi_track_index": 1,
  "notes": [
    {
      "source_note_ordinal": 0,
      "pitch_midi": 60,
      "velocity_midi": 88,
      "start_tick": 0,
      "end_tick": 960
    }
  ],
  "program_changes": [
    { "tick": 0, "gm_program": 0 }
  ]
}
```

For digest calculation, serialize with the repository's declared canonical JSON procedure, sort `notes` by `source_note_ordinal`, sort `program_changes` by `(tick, gm_program)`, encode UTF-8, then calculate SHA-256 over:

`domain || 0x00 || canonical_json_bytes`

The verifier must parse the supplied MIDI bytes and recompute this event set from the actual selected track and its corresponding manifest receipts. It must not trust a caller-supplied event digest or infer PartId from pitch, register, channel, program, loudness, or audio similarity.

### Common sample clock

V1 uses full-length stems with the same:

- sample rate;
- channel count;
- sample format and bit depth;
- frame count;
- zero-time origin.

No implicit trimming, leading-silence removal, resampling, padding, alignment, or normalization is allowed during bundle admission. Such transformations change the measurement domain. A future format may encode explicit offsets and transforms, but those must be separate, versioned operations with their own receipts.

The common clock is a structural alignment contract, not evidence that the renderer placed every note at the correct sample. Event timing remains an independently measured property.

## Admission and verification procedure

The future Rust validator should perform these steps in order:

1. **Schema:** reject unknown fields, wrong schema IDs, malformed digests, empty identities, invalid ranges, and empty stem sets.
2. **Source binding:** verify the exact Score, MIDI bytes and MIDI-manifest digests; verify the MIDI artifact against its source manifest.
3. **Track identity:** derive the expected `(PartId, segment_id, instrument_id, track_index)` set from the trusted MIDI manifest. Require exactly one stem record for every expected tuple, with no duplicates, omissions or extras.
4. **Event coverage:** recompute the canonical event-set digest from the selected MIDI track; require the supplied sorted source-note ordinals to equal—not merely overlap—the manifest's ordinal set for that track.
5. **Renderer identity:** require non-empty renderer/patch identifiers and valid digests for the renderer binary, patch assets and render configuration. A missing digest is `UNQUALIFIED`, not a pass.
6. **WAV integrity:** hash the exact bytes, decode the WAV header and samples, reject malformed/non-finite audio, and compare decoded format and frame count with the common clock.
7. **Cross-stem clock:** require every stem to share the exact sample clock. Any mismatch is a hard rejection; never silently resample or align during verification.
8. **Receipt:** emit a deterministic, domain-separated commitment over the complete validated manifest and ordered stem records.

The verifier should produce typed outcomes such as `PASS`, `REJECT`, `UNQUALIFIED`, and `NOT_ESTIMABLE`. Do not turn missing renderer evidence, unsupported formats, or invalid acoustic attribution into a boolean success.

## Known-answer and adversarial fixture matrix

Start with small deterministic, rights-cleared synthetic Scores and a pinned renderer/patch.

| Fixture | Expected admission / evaluation result |
|---|---|
| Two parts, disjoint pitches, same clock | Both stems admitted; notes remain separately addressable |
| Two parts, simultaneous same pitch | Both stems remain separately addressable; mixed-only attribution marked non-identifiable |
| One part hands off from instrument A to B | Two segment records, one persistent PartId, exact handoff boundary retained |
| Missing part stem | Reject: incomplete track coverage |
| Duplicate (PartId, segment) record | Reject: duplicate identity |
| Swapped PartId labels | Reject when event-set / manifest binding disagrees |
| Wrong track index | Reject: event-set digest or ordinal coverage mismatch |
| Omitted or extra source-note ordinal | Reject: exact coverage mismatch |
| Stale MIDI, manifest, WAV or asset digest | Reject: digest mismatch |
| Same notes but wrong program-change event | Reject: track event-set mismatch |
| Different sample rate, channel count or frame count | Reject: common-clock mismatch |
| Truncated or malformed WAV | Reject: decode / clock validation failure |
| Neural source-separated stem | Not admissible as V1 isolated-render evidence; may enter a separate separator study |
| Reverb tail or same-part harmonic collision | Keep acoustic result ambiguous unless a separately calibrated method resolves it |

## Metrics and qualification targets

Keep artifact correctness separate from acoustic quality.

### Contract correctness

- 100% exact track-set completeness on deterministic fixtures.
- 100% exact source-note ordinal coverage.
- 0 accepted duplicate, missing, extra, swapped, stale, or clock-incompatible stem records in the frozen negative-control suite.
- 100% deterministic bundle commitment replay from frozen inputs.
- Unsupported or uncalibrated renderer/patch combinations remain `UNQUALIFIED`.

### Acoustic evidence

- Report per-note measured / abstained / ambiguous counts.
- Report fundamental salience and harmonic salience separately.
- Calibrate every renderer/patch/range/dynamic family independently.
- Report false support and missed support on unison, octave, fifth, dense-chord and reverb controls.
- Do not publish a universal dB threshold or call salience a probability.
- Add onset/offset claims only after a detector is calibrated against human-labelled boundaries and its uncertainty is reported.

### Human qualification

A musician/engineer review cohort should separately assess whether the stem sounds like the intended instrument/part, whether the articulation and dynamics are plausible, and whether the mix preserves the intended ensemble relationship. These judgements must not be inferred from the renderer's own patch label or the symbolic manifest.

No single weighted score may compensate for a failed identity, preservation, timing, or provenance gate.

## External comparison: what the numbers do and do not say

The relevant public frontier now includes different task families, and their headline scores must not be collapsed:

- **YuE2 (September 2026)** combines symbolic planning and audio generation. Its authors report 49.3% overall expert preference for the score-planned condition versus 34.6% for the unplanned condition under their matched two-candidate setup. On WildSongBench, the paper reports 6.7316 in its standard setting and 6.9632 best-of-8. Best-of-8 is not a single-pass comparison. YuE2 is the direct score-first generation comparator; it does not replace independent renderer-conformance testing. [Paper](https://arxiv.org/abs/2609.33757) · [Project](https://map-yue2.github.io/)
- **SheetSage2 / MERT2** provide strong audio-to-score and representation baselines within the YuE2 ecosystem. Their published benchmark results are relevant to transcription and feature extraction, not proof that a particular rendered WAV conforms to a particular source Score. [YuE2 project](https://github.com/hzwangming/YuE2)
- **TimingBench (August 2026)** contains 250 prompts and 399 timed events. Its detector evaluation and explicit limitations are a model for apparatus-aware event timing. It is a timing-control benchmark, not a part-stem integrity benchmark. [Method and results](https://taan.ai/research/rushing-or-dragging/)
- **SMEBench (October 27, 2026)** supplies paired source/target edits over 12 operation types and measures preservation, target agreement and text compatibility separately. It is a useful edit-preservation baseline, not a stem provenance standard. [Paper](https://arxiv.org/abs/2610.36715)

A fair comparison should therefore report separate scorecards for (a) full-song preference and quality, (b) symbolic edit correctness and preservation, (c) temporal controllability, (d) transcription/localization, (e) part-stem provenance and acoustic observation, and (f) musician-rated collaboration. Do not claim that a result on one track beats an unrelated system on another.

## Integration and sequencing

1. **This contract:** freeze schema, event-set digest recipe, common clock and negative controls.
2. **Rust validator:** validate bundle/schema/source/track/asset/WAV identities with bounded inputs and deterministic receipts.
3. **Stem-scoped observations:** reuse the existing pitch-energy observer on only the notes belonging to the selected stem; preserve its current caveat that spectral support is not note attribution.
4. **Independent transcription:** evaluate SheetSage2 or another permitted transcription model against the exact MIDI source, using #11597's model-agnostic evaluator and matched fixtures.
5. **Timing:** add separately calibrated onset/offset detection; retain `NOT_ESTIMABLE` when attack/release cannot be distinguished from patch envelope or overlapping same-part material.
6. **Mix consistency:** only after the renderer's routing/mastering path is pinned, compare the stem-derived mix with the published mix under an explicit summing/mastering contract.
7. **Human pilot:** validate a small, rights-cleared cohort with musicians and audio engineers before any broader instrument/patch qualification.

Adjacent programmes remain authoritative for their own concerns: #11389 for end-to-end score-to-audio conformance, #11564 for current pitch observations, #11597 for MIDI transcription evaluation, #11281 for orchestration handoffs, #11284 for part relationships, and #10417 for the external comparison ledger.

## Claim boundary

A valid bundle proves that its manifest consistently binds named, isolated WAV artifacts to a source-bound MIDI track set and a declared renderer/asset/configuration identity. It does **not** prove that the renderer actually used the declared binary or assets, that the acoustic signal is perceptually correct, that a patch sounds like its named instrument, or that a musician would approve the result. Those require independent execution observation, calibrated acoustic measurements, and human review.

**Exit condition:** a Rust validator admits a complete part/segment stem set only when source MIDI identity, track-event identity, renderer/asset evidence, WAV bytes and common-clock properties reconcile exactly; all frozen negative controls reject; and the admitted stems can be consumed by the downstream audio-observation and transcription evaluators without inventing part attribution.
