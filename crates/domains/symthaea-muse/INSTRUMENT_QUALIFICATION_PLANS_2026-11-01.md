# Melothaea instrument qualification plans — 2026-11-01

Status: **plan-only**. This document defines what must be measured and tested for every current Muse Instrument variant. It does not certify any range, technique, articulation, sample bank, renderer, or perceptual quality.

## Why this is a separate layer

The current 42-entry ontology is an identity map. The inventory in #7464 is source/asset discovery and opt-in rendering evidence. #7554 defines four qualification tiers. #9586 defines exact-cell, signed capability decisions. This document connects them by specifying the **instrument-specific experiments** that must exist before a capability can be advertised.

The unit of qualification is not an instrument name. It is an exact cell:

**(ontology identity, source model, setup/tuning, technique, articulation, sounding pitch or percussion key, dynamic/velocity region, expression state, renderer/build, asset revision)**

The result for every requested cell is one of:

- **Qualified** — exact, independently verifiable evidence exists for this cell.
- **Unsupported** — a trusted verifier has explicit negative evidence for this cell.
- **Unknown** — evidence is missing, ambiguous, stale, or not applicable to the requested realization.
- **Invalid evidence** — a receipt, manifest, identity binding, signature, or measurement is malformed or contradictory.

No neighboring pitch, velocity, articulation, setup, model, or renderer inherits a result. An unknown cell is not silently converted into a supported one by a procedural fallback or a GM patch.

## Paradigm shift: instruments as executable performance contracts

A conventional patch list maps a name to a sound. Melothaea should map a source identity to a versioned performance contract:

1. **Identity and lineage** — physical/model identity, taxonomy, aliases, source, cultural or regional context where evidenced, and rights.
2. **Configuration** — instrument size, tuning, strings/reeds/keys, register, ensemble size, mute, mallet, pedal, reed, or other setup as appropriate.
3. **Gesture vocabulary** — the instrument's own meaningful controls: bow speed/pressure/contact point, breath and tonguing, pluck position/force, mallet hardness/strike point, bellows pressure/direction, pedal/stops, slide position, vowel/formant, or synthesis parameters.
4. **Playable domain** — measured pitch/register/tessitura, overlap/polyphony, note transitions, dynamics, tuning, and microtonal/bend behavior.
5. **Realization** — measured model, source-derived model, approximation, procedural substitute, sample instrument, or unavailable path. These are different outcomes, not interchangeable labels.
6. **Exact capability evidence** — signed receipts over the exact model × configuration × gesture × pitch × dynamic × renderer/build tuple.
7. **Counterfactual validation** — vary one gesture or capability dimension while holding the score, seed, and other controls fixed. Confirm that the intended acoustic/structural feature changes and unrelated dimensions remain stable.

This creates an instrument-independent engine for evidence handling while preserving instrument-specific semantics. Shared validation infrastructure is desirable; shared capability claims are not.

## Qualification axes

Every plan below uses the relevant subset of these test families. “Test” means *measure and attempt to falsify*, not assume support.

| Axis | Required observation |
|---|---|
| Identity/setup | Source model, construction/setup, tuning, transposition, source lineage and license; do not resolve identity from register or voice role |
| Range/tessitura | Lowest/highest source-supported pitch, comfortable tessitura where a credible source exists, boundary behavior, register transitions and unsupported gaps |
| Excitation/onset | Bow, breath, reed, lip, pluck, hammer, mallet, drum stroke, voice onset or oscillator start; onset timing and transient shape |
| Sustain/release | Held-note behavior, decay, release samples, dampers, pedal, breath continuity, note-off and overlapping-note behavior |
| Dynamics | Velocity layers or continuous controls, measured response, layer boundaries, hysteresis and repeatability |
| Articulation | Source-specific articulation transitions and whether the renderer selects the intended articulation |
| Pitch/expression | Tuning, vibrato, bends, slides, portamento, microtonal intervals and per-note independence where meaningful |
| Polyphony/interaction | Chord overlap, voice stealing, sympathetic resonance, coupled strings, drum choke groups or ensemble interactions |
| Assets/rights | Immutable source revision, per-file content digest, mapping, decoded file facts, license and permitted use |
| Renderer | Exact patch/model, build, sample zones, fallback reason, output format, output digest and repeat-render behavior |
| Perceptual validation | Blinded, level-matched listening against a declared reference, with confidence intervals and instrument-specific task |
| Negative controls | Missing asset, unsupported pitch/technique, wrong model, corrupt/silent audio, invalid setup, wrong renderer and deliberately mismatched articulation |

Do not populate a numeric playable range from a generic instrument chart alone. A range is attached to an identified model/setup and a declared evidence source. Transposing instruments must retain written-versus-sounding pitch semantics rather than having a single ambiguous range field.

## Per-instrument plans — current 42 variants

Every row is a distinct plan entry. Related instruments may share a test harness, but not their source evidence or promotion status.

| Stable ontology ID | Current variant | Qualification track and priority experiments |
|---|---|---|
| **piano** | Piano | Acoustic keyboard. Measure register-dependent tuning/inharmonicity, hammer attack, velocity-to-spectrum response, decay, damper/release behavior, sustain and half-pedal where the source supports it, repedaling, repeated-note reset and polyphonic overlap. Compare measured partials only at their documented reference pitches; do not extrapolate them into a full-range certificate. |
| **piano.pp** | PianoPP | Piano realization/dynamic variant, not a second physical instrument. Test its actual relationship to Piano: soft-velocity response, timbral/dynamic mapping, shared or distinct sample/model provenance, release and pedal behavior. Must not inherit a full Piano qualification merely because the timbre is related. |
| **violin** | Violin | Bowed string. Test bow onset, sustain stability, bow changes, legato, détaché, tremolo, vibrato onset/depth, pizzicato, harmonics and register transitions as separate cells. Verify string/tuning/setup and avoid treating a static partial table as bow-technique evidence. |
| **cello** | Cello | Bowed string with its own low-register and string-response profile. Test bow and pizzicato paths independently, low-register onset/tail, string crossings, legato/détaché, vibrato, harmonics, dynamic response and note overlap. Do not copy violin range or articulation cells. |
| **flute** | Flute | Edge-blown aerophone. Test breath/chiff onset, tongued versus slurred starts, register/octave transitions, breath-noise relationship, dynamic stability at soft and loud levels, vibrato and pitch inflection. Separate flute models and headjoint/setup when source evidence provides them. |
| **organ** | Organ | Sustained keyboard source. Test registration/stop configuration, stable sustain, key release, attack/retrigger behavior, polyphonic chord load, manual/pedal assignment when modeled and whether dynamics come from registration or a continuous control. Do not infer pipe-organ ranks from an organ-like spectrum. |
| **electric_piano** | ElectricPiano | Electric keyboard. Identify the actual tine/reed/hammer and pickup/model family before testing. Measure velocity-dependent timbral transitions, decay, key release, repeated-note behavior, pickup/effect configuration and any tremolo or modulation only when it is actually implemented by the selected source. |
| **bell** | Bell | Struck idiophone. Test pitch map, inharmonic partial behavior, strike velocity/position where controllable, attack transient, long decay, overlap and damping. Record the physical bell/model identity; do not use a generic bell timbre as evidence for all bells. |
| **acoustic_guitar** | AcousticGuitar | Plucked string. Test string/course identity, pitch and tuning coverage, finger/pick attack, strum direction/voicing, damping, palm mute, harmonics, repeated plucks, fret/string transitions and release behavior. Treat nylon/steel and model-specific variants separately when they differ. |
| **harp** | Harp | Plucked multi-string instrument. Test string/pitch coverage, pluck and decay, overlapping resonance, glissando as a sequence of available strings, pedal/semitone changes only with a documented model, damping and repeated-note behavior. A glissando generator must not imply pedal-harp support. |
| **pad** | Pad | Electronic source category, not a physical instrument. Test oscillator/source identity, attack/sustain/release envelope, filter and modulation controls, polyphony, aliasing, note transitions and per-note pitch/timbre expression. Bind every test to a named patch/build and parameter preset. |
| **clarinet** | Clarinet | Single-reed aerophone. Test chalumeau/clarion register transition, attack/tonguing, legato, dynamic stability, breath/reed noise, pitch bends and vibrato. Preserve transposition and model/setup; do not inherit generic woodwind range. |
| **trumpet** | Trumpet | Brass/lip-reed. Test register and partial transitions, tongued/legato onset, breath pressure, dynamic response, pitch inflection and vibrato. Mutes, falls, shakes and other special effects are separate capabilities requiring explicit source assets or renderer evidence. |
| **saxophone** | Saxophone | Single-reed family. Bind subtype/model (alto, tenor, etc.) and transposition. Test register/altissimo boundaries, tonguing, subtone, vibrato, bends, breath noise, dynamic transitions and note releases. Do not silently equate the generic variant with a specific saxophone size or Saxello. |
| **marimba** | Marimba | Struck wooden idiophone. Test bar pitch map, mallet hardness/strike response, register-dependent decay, damping, repeated hits, rolls, multi-mallet overlap and velocity behavior. Track the actual bar range and instrument model rather than assuming the entire conventional range. |
| **sitar** | Sitar | Plucked/buzz-bridge string model. Test pluck attack, jawari/buzz behavior, sympathetic resonance if modeled, meend/bends, ornaments, tuning and string/course identity. Keep source-specific sitar evidence distinct from generic plucked-string or Oud/Koto presets. |
| **kalimba** | Kalimba | Plucked tine idiophone. Test each documented tine pitch, tine decay, overlap/resonance, pluck dynamics, damping and repeated-note behavior. Separate instrument variants and tunings; no extrapolation to unsupported tine notes. |
| **saw_lead** | SawLead | Electronic oscillator category. Test waveform/source and anti-aliasing, pitch tracking, glide/portamento, oscillator detune/unison, filter envelope, note-on/off, polyphony and per-note expression. A synth preset is a renderer identity, not an acoustic instrument identity. |
| **koto** | Koto | Plucked zither. Test string/tuning map, plectrum attack, string-pressure bends, glissando/ornaments, damping, repeated strokes and register coverage. Keep Koto source models and playing techniques distinct from generic harp or guitar mappings. |
| **oud** | Oud | Plucked fretless/lute-family instrument. Test risha attack, tremolo, fretless pitch movement/microtonality, string/course tuning, ornamentation and decay. GM export mapping is a compatibility alias only and cannot qualify Oud timbre or tuning. |
| **ney** | Ney | End-blown aerophone. Test breath/embouchure response, register transitions, breath noise, microtonal pitch inflection, ornamentation, vibrato and note connection. Source/model and regional tuning practice must be documented; do not treat the generic flute approximation as Ney evidence. |
| **upright_bass** | UprightBass | Double bass identity with distinct pizzicato and arco paths. Test low-register intonation/attack, string and tuning configuration, pizzicato decay, bow sustain, note overlap, string-crossing artifacts and register limits. Do not inherit Cello range or assume both articulations are rendered. |
| **viola** | Viola | Bowed string. Test C-string/low-register response, high-register transitions, bow changes, legato/détaché, pizzicato, vibrato and harmonics. The current derived spectrum is not measured Viola evidence; collect source-specific measurements before acoustic claims. |
| **oboe** | Oboe | Double-reed aerophone. Test reed onset, breath pressure, register/intonation transitions, tonguing, legato, dynamic control, vibrato and note releases. Identify model and transposition where relevant; distinguish a physical oboe from an oboe-like synthesis table. |
| **french_horn** | FrenchHorn | Brass/lip-reed with hand-stopping practice. Test register/partial accuracy, attacks, slurs, dynamic response, intonation, horn transposition and stopped/muted tone as independent source-supported articulations. Do not mark stopped horn supported from a normal horn sample. |
| **bassoon** | Bassoon | Double-reed aerophone. Test low-register onset, tenor/high-register transitions, reed response, tonguing, legato, dynamic control, vibrato and intonation. Record model/setup and avoid importing Oboe or generic woodwind bounds. |
| **accordion** | Accordion | Free-reed instrument. Test bellows pressure and direction changes, reed/register stops, attack/release, chordal polyphony, note overlap and dynamic response. Model left-hand/bass/chord semantics only where represented; a static accordion spectrum is not bellows control. |
| **mandolin** | Mandolin | Plucked double-course string. Test course tuning/unison behavior, plectrum attack, tremolo, repeated notes, damping, course resonance and playable pitch zones. Do not assume guitar articulation or range. |
| **banjo** | Banjo | Plucked string. Identify banjo type/model and tuning. Test picking rolls, single-note attack, strum, decay/bridge response, muting, repeated-note behavior and register. Bluegrass roll semantics should be a performance pattern, not an implied sample articulation. |
| **vibraphone** | Vibraphone | Struck idiophone. Test bar pitch/velocity coverage, mallet hardness, motor/vibrato rate/depth if implemented, sustain pedal/damper, rolls, repeated strikes and overlap. Distinguish motor-on/off and pedal states from generic pitch/velocity support. |
| **celesta** | Celesta | Keyboard-struck idiophone. Test actual key-to-bar/pitch mapping, attack, decay, register transitions, key release and repeated-note response. Do not confuse it with piano or glockenspiel merely because all are keyboard/mallet-like. |
| **choir.aah** | ChoirAah | Vocal realization with a specific vowel. Test Aah formant spectrum, onset/offset, breath/noise, vibrato, dynamic and pitch response, polyphonic overlap and sustain. This is not a complete human-voice or phoneme model; preserve vowel identity in every receipt. |
| **strings.ensemble** | StringEnsemble | Ensemble configuration, not a solo string instrument. Test section size/voice count, ensemble spread and detuning, correlated versus independent attacks, divisi/voice separation, sustained and detached textures, dynamic balance and polyphonic density. Solo violin/cello evidence cannot qualify an ensemble. |
| **electric_bass** | ElectricBass | Electric bass model. Identify pickup/string configuration and tuning. Test fingerstyle, pick and slap only as distinct source-supported techniques, muting, fret/register behavior, sustain, dynamic transitions and low-frequency rendering. Do not copy UprightBass or generic guitar claims. |
| **timpani** | Timpani | Tuned membranophone with dedicated drum semantics. Test drum-size/tuning map, pitch retuning/glissando, stroke hardness, roll, dampening, note transition and resonance. Represent hit/roll and drum identity separately from arbitrary chromatic note numbers; a pitched sample filename is not a tuning certificate. |
| **music_box** | MusicBox | Plucked/struck mechanical idiophone. Test cylinder/tine pitch map, attack and fixed decay, repeated-note limitations, overlap, release/tail and any winding/tempo behavior that is actually modeled. Identify the specific mechanism/model; do not infer general keyboard-like velocity. |
| **harpsichord** | Harpsichord | Plucked-keyboard instrument. Test manual/register/stop and coupling configuration, plectrum attack, limited touch-dynamic behavior, key release/damper, repeated-note reset and polyphonic overlap. Separate harpsichord models and manuals where the source evidence does so. |
| **tuba** | Tuba | Brass/lip-reed. Test low-register stability, breath/lip onset, partial transitions, tonguing, legato, dynamic response, pitch bends and transposition. Record model/size; do not assume one generic tuba covers all subtypes. |
| **trombone** | Trombone | Brass/lip-reed with continuous slide. Test slide-position/pitch mapping, glissando and portamento, legato between positions, tongued attacks, register transitions, breath, mutes and transposition. A discrete MIDI note grid alone cannot qualify slide behavior. |
| **duduk** | Duduk | Source-specific double-reed aerophone. Test reed/breath response, sustained tone, expressive vibrato, pitch bends, ornaments, register and tuning with culturally and geographically grounded sources. Do not relabel an Oboe/Clarinet approximation as Duduk or impose one regional practice as universal. |
| **choir.ooh** | ChoirOoh | Separate vocal realization from ChoirAah. Test Ooh-specific formants, vowel-specific sustain/offset, vibrato, dynamics, onset and overlap. It must have its own source or measured model evidence; Aah results cannot be copied across. |
| **xylophone** | Xylophone | Struck wooden idiophone. Test bar pitch mapping, mallet hardness/strike response, attack and decay by register, repeated hits, rolls and damping/overlap. Do not reuse Marimba or generic mallet range as qualification evidence. |

## Test execution strategy

The full Cartesian product of every configuration, technique, pitch, dynamic, expression control and renderer can be enormous. Use a **boundary-first, risk-weighted design**:

1. Exhaustively test identity resolution, receipt completeness, schema validation and all known hard boundaries.
2. Test every source-declared playable sample/model cell at least once. Do not invent missing cells to make a rectangular matrix.
3. Add boundary samples immediately below, at and above each declared pitch, velocity-layer, register and articulation transition.
4. Use pairwise/covering-array selection for lower-risk interactions, but retain exhaustive tests for safety-critical or known coupled dimensions (for example, instrument-change timing, slide pitch, drum mapping, choke groups, pedal/stop transitions).
5. Add repeated-note and overlap trials to expose voice stealing, stuck notes, missing releases, tail truncation and round-robin exhaustion.
6. Hold the musical phrase, seed, renderer build and all other controls fixed for one-gesture counterfactuals. Verify both the intended change and the absence of unrelated changes.
7. Repeat critical trials across independent source recordings, microphone positions, models, renderers and builds. Keep train/calibration sources disjoint from held-out sources.

Each run must bind the requested cell, source/asset manifest, exact sample digests, rights metadata, resolver/verifier version, renderer/build, fixture digest, output digest, signal diagnostics and decision. A file path plus size is not an audio content hash. A WAV decode is not proof of correct pitch. A non-silent render is not proof of the requested technique.

## Benchmark targets and industry comparison

Targets below are **engineering acceptance goals**, not claims about current performance:

| Benchmark | Initial hard target | How to compare fairly |
|---|---|---|
| Identity and coverage accounting | 100% of the declared 42 variants have exactly one plan and one explicit status; 100% of requested cells receive a decision receipt | Compare plans, inventory mappings, and runtime receipts; do not count variants as renderable instruments |
| False capability claims | 0 false-positive Qualified cells in the independent challenge set | Include near-neighbor controls: wrong setup, adjacent pitch, wrong articulation, different renderer, missing asset |
| Silent substitution | 0 unexplained fallback; every substitution carries source identity, selected path and reason | Compare with DAW/SoundFont/library paths using identical source notes and disclose each fallback |
| Render integrity | 0 NaN/Inf, malformed audio, unexplained silence or unintended clipping in successful fixtures | Pin sample rate, gain, effects, build and reference level; do not compare unmatched loudness |
| Coverage completeness | 100% of source-declared valid cells have a measured outcome; unknown gaps remain visible | Report pitch × velocity × articulation × model × renderer coverage, not one scalar instrument count |
| Reproducibility | Same-input/same-build rerender result is recorded for every qualification suite; exact hashes only where byte identity is promised | Compare content hashes under pinned environments, otherwise predeclare numeric tolerances |
| Acoustic fidelity | Establish an independent per-instrument reference baseline before setting numeric thresholds | Use instrument-appropriate acoustic measurements and source recordings; do not use one spectral score for all instruments |
| Perceptual quality | Blinded, level-matched, randomized trials with uncertainty and per-instrument reporting | Use ITU-R BS.1116 for small impairments and BS.1534/MUSHRA where the test question fits intermediate-quality comparison |
| Instrument recognition | Report per-class precision/recall and macro-F1 on source-disjoint data as a diagnostic only | MIREX 2026 uses clip-level multi-label recognition and macro-F1; recognition is not a proof that a renderer plays an instrument faithfully |
| Performance/runtime | Measure p50/p95 render latency, memory, startup cost and polyphony by renderer and instrument profile | Compare under the same sample rate, block size, polyphony, effects and hardware; publish cold and warm runs |

### Practical industry baseline

- **GM1** is a compatibility/program map, not a physical instrument ontology or a technique qualification.
- **MPE** provides per-note pitch/timbre/pressure control over MIDI 1.0 channel allocation. It is a useful baseline for expressive per-note control, not a substitute for instrument-specific gesture semantics.
- **MIDI 2.0 / UMP** adds higher-resolution messages, capability inquiry and profile negotiation while preserving MIDI 1.0 compatibility. Melothaea should plan an explicit export/interop adapter rather than silently quantizing its internal expression model.
- **MusicXML 4.0** can distinguish multiple score-instrument records within a part and supports instrument-change semantics. Preserve those identities and timed changes rather than flattening a part to a GM label.
- **VCSL** is a useful CC0 source library, but its own description emphasizes pitch fidelity and consistency over the many round robins, articulations and velocity layers expected from some professional libraries. Its presence is evidence of source material, not comprehensive instrument qualification.
- **MIREX instrument recognition** gives a reproducible external diagnostic for recognizing instruments in audio. It is not a benchmark of note-level playability, source-faithful rendering, articulations or musical quality.

The honest goal is not to claim a larger patch count than DAWs or commercial libraries. It is to outperform generic patch systems on **truthful capability resolution, provenance, compositional controllability, cross-renderer portability and zero silent substitution**, then demonstrate instrument-specific perceptual parity or improvement where we have real source evidence.

## Paradigm-shift research programme

### 1. Gesture-to-physics control layer

Introduce a typed control layer between symbolic notes and renderers. Examples include bow speed/pressure/contact point, breath pressure/embouchure, pluck position/force, mallet hardness/strike position, bellows pressure/direction, slide position, pedal state and vowel/formant controls. Each control is optional, source-specific, unit-bearing and tied to a known model. Do not create universal controls that pretend every instrument shares the same mechanism.

### 2. Evidence-driven active qualification

Use an active test selector to choose the next pitch/velocity/technique cell with the highest expected information gain about a stated uncertainty (for example, register boundary or articulation transition). Its output proposes experiments; the independent verifier decides whether resulting evidence is admissible. Record the selection rationale and preserve a fixed, non-adaptive held-out challenge set to prevent overfitting.

### 3. Counterfactual controllability benchmark

For each gesture axis, synthesize paired trials where only that axis changes. Measure the intended acoustic response and side effects. Examples: bow pressure should alter the declared bow-energy/timbre dimensions; a slide gesture should move pitch continuously; a vibraphone pedal should affect decay; choir vowel identity should alter formants. A control that changes no intended dimension is unconsumed; a control that changes unrelated dimensions is entangled and must be disclosed.

### 4. Cross-renderer instrument contract

Keep the symbolic performance contract stable while adapting it to sample libraries, procedural synthesis and external DAWs. Each adapter declares which controls it implements, approximates or cannot express. Use the same score and gesture trace across renderers. Compare preservation of the requested performance semantics, not just whether each renderer emits a WAV.

### 5. Source-faithful expansion beyond the 42 variants

After the current scope is measurable, add source-native identities (for example, Dan Tranh, Ocarina models, Baroque recorder variants, Saxello, harmonica models and dedicated world-percussion sources) without aliasing them to an existing Western instrument. Keep percussion drum IDs and release events in their own semantic lanes. A finite versioned catalogue is the goal; “every instrument in the world” is not a verifiable scope.

## Qualification gates

1. This document and the ontology must cover all 42 current variants exactly once.
2. Inventory records must distinguish physical identity, source model, procedural approximation, sample backing, SoundFont preset and actual renderer evidence.
3. Each profile must have at least one deterministic positive fixture and relevant negative controls before any capability is promoted.
4. Asset-backed claims require exact content hashes, source revision and rights/provenance; procedural claims require exact model/build/parameter bindings.
5. Every claimed range/technique/articulation cell must have a direct evidence receipt and independent verification.
6. Perceptual claims require a preregistered listening protocol and human-listening evidence.
7. Unknown remains a valid, visible answer. A fallback may provide a useful audition, but it never rewrites the requested source identity or upgrades the original cell.
8. Exact-head CI, compile and tests must pass before merge. Queued, skipped, absent or unrelated workflow runs are not passes.

## Existing implementation anchors and standards

- #9499 — finite, evidence-backed completion target for instrument identity and playable capability.
- #9565 / #9574 — stable ontology and mapping for the current 42 variants.
- #9586 — signed exact-cell capability decisions.
- #7464 — source-derived asset inventory and opt-in FluidSynth probes.
- #7554 — four-tier asset, render, acoustic and perceptual qualification.
- #7416 — source instrument identity and MIDI/MusicXML preservation.
- #7567 — source-native world instrument identities without misleading aliases.
- MIDI Association: [General MIDI Level 1](https://midi.org/general-midi-level-1), [MPE specification](https://midi.org/mpe-midi-polyphonic-expression), and [MIDI 2.0](https://midi.org/midi-2-0).
- W3C: [MusicXML 4.0 score-instrument](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/score-instrument/) and [instrument-change example](https://www.w3.org/2021/06/musicxml40/musicxml-reference/examples/instrument-change-element/).
- VCSL: [library overview and instrument/source notes](https://versilian-studios.com/vcsl/) and [CC0 repository README](https://github.com/sgossner/VCSL/blob/master/README.md).
- ITU-R: [BS.1116](https://www.itu.int/rec/R-REC-BS.1116/) and [BS.1534](https://www.itu.int/rec/r-rec-bs.1534/en).
- MIREX 2026: [Audio Instrument Recognition evaluation](https://www.music-ir.org/mirex/wiki/2026%3AAudio_Instrument_Recognition).

**Validation note:** this is a planning artefact on a draft branch stacked on the instrument ontology PR. It has not been compiled or test-run, and it makes no new capability qualification claims.
