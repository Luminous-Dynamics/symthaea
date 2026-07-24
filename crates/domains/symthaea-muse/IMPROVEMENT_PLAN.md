# symthaea-muse: Improvement Plan (2026-07-06)

From a 4-dimension comprehensive review (architecture/integration, DSP correctness,
learning stack, consciousness coupling) + test/clippy run. ~28K LOC, 70 modules,
450 lib tests.

## Where it stands

**Genuinely good and load-bearing:**
- Wired into the 31Hz cognitive loop for real: `MuseManager` (interval 1, snapshot→
  `MusicalState` every cycle, 8s WAV exports → `MusicPublisher` → Mycelix-Music) and
  `CreativeManager` batch path (`creative_bridge.rs`). Feature-gated (`muse`/`creative`),
  off by default.
- Real DSP: legitimate Freeverb (8 combs + 4 allpass), gated-integrated loudness
  normalization + real brick-wall limiter, band-limited additive/wavetable oscillators,
  clean cpal callback (no locks/allocs on the audio thread).
- Real musicianship code: SATB voice-leading (Ψ-gated polyphony), Just Intonation /
  Maqam tuning selection, chord accompaniment, motif memory, dramatic arc.
- Real TTS-singing bridge (`voice_bridge` → symthaea-voice Kokoro), wired into
  `StreamingSynth`.
- Strong property tests in `pitch`, `voice`, `creative_agency`, `mel_extractor`.
- `mycelix-music` leptos app consumes `StreamingSynth` directly (correct boundary).

**The honest problem:** the *learning* layer is mostly scaffold with overstated
provenance, and the lib test suite is currently red.

## Phase 0 — Stabilize (red → green) [hours]

- [x] Fix 3 failing lib tests: `synth::render_partials_affect_timbre`,
      `synth::render_fm_depth_affects_output`, `audio_analyzer::test_surprise_on_sine_sweep`.
- [x] Fix 2 clippy deny-level errors: `absurd_extreme_comparisons` (2 sites),
      `approx_constant` (hand-written SQRT_2).
- [x] **Fix `soft_clip` discontinuity** (`synth.rs:593-601`): for x just above 1.0 it
      returns ~0.5 while returning 1.0 at x=1.0 — a 0.5-amplitude jump that wavefolds
      any over-full-scale sample (applied to live reverb output at synth.rs:246-247 and
      Mono16 mix at :301). Use the knee-based `smooth_limit` directly above it (or tanh).
      NOTE: a naive `1.0 - 0.5*exp(-(x-1.0))` is *also* discontinuous — verify continuity
      at the boundary in the test. Delete the dead duplicate at `streaming.rs:127-135`.
- [x] Add the missing property test: full synth render stays within ±1.0 (this is the
      test that would have caught soft_clip).
- [x] `src/bin/train_spectral_vocoder.rs:104` uses `#[cfg(feature = "cuda")]` but `cuda`
      is not a declared feature — dead branch; declare or delete.

## Phase 1 — Honesty pass (labels must match code) [~1 day]

Provenance/labeling corrections (project principle: mark estimates as estimated):
- [x] `learned_melody.rs:3-8` claims "trained on 65M pairs (MAESTRO+Nottingham),
      93.2% direction accuracy" — the "model" is 40 hardcoded constants and **no code
      in the repo fits them**. Either re-derive via an in-repo fit script (preferred,
      pipeline exists in `midi_trainer`) or rewrite the header as hand-tuned/provenance-unknown.
- [x] `examples/train_deam_regressor.rs:63-88` — **target leakage**: synthesizes features
      as direct functions of the V/A labels it predicts; any R² is circular. Either wire
      real audio decoding (symphonia + the existing `mel_extractor`) or relabel as a
      calibration demo and stop calling it training.
- [x] `examples/trained_vs_random.rs:153` labels Neural mode "trained on MAESTRO" — it
      loads `data/midi-training/melody_projections.json` which **does not exist anywhere
      in the repo**, so it compares two untrained generators judged by the hand-coded critic.
- [x] `taste_bench.rs:5,56` "Derived from 5,972 Spotify liked songs" — hardcoded constants,
      no ingestion code. Relabel.
- [x] `hdc_mel_decoder.rs` — a real, well-built 17→128→128 MLP (SGD+Adam, tested), but
      contains zero HDC. Rename (`mel_mlp.rs`) or fix the docstring.
- [x] `creative_bench.rs:37` "melodic coherence vs trained corpus" — corpus is a 13-value
      hardcoded Huron-textbook histogram. Relabel.
- [x] `spectral_vocoder.rs` claims "Griffin-Lim-like" — it's a sine-bank resynthesizer,
      unused `mel_buffer`, not wired into StreamingSynth. Relabel or wire.
- [x] `consciousness_reverb.rs:212-215` "modulated combs" — `mod_phase` incremented,
      never read. Implement or delete.
- [x] `fingerprint.rs` — 3-line empty placeholder exported as public API. Delete.
- [x] Neural mode's silent fallback (`neural_melody.rs:92`, `lib.rs:401`): missing
      projections file must be **loud** (log + telemetry flag), not silent default weights.

## Phase 2 — DSP correctness & performance [1-2 days]

- [x] **Denormal protection**: no FTZ anywhere; Freeverb combs (feedback ~0.98), allpass,
      `OnePoleLP`, brightness LP all recurse toward zero → CPU spikes on silence/tails.
      Classic real-time hazard given cpal live output.
- [x] **Pre-delay stereo bug** (`consciousness_reverb.rs:205-206`): one mono delay line
      `process()`ed twice per frame — halves effective pre-delay and cross-contaminates
      L/R. Use two delay lines.
- [x] **Hot-loop perf** (`sample_player.rs:101-125` + `streaming.rs:581-628`): per-sample
      O(N) HashMap scan per active note + per-sample `Vec` allocation on the thread feeding
      the live ring buffer. Cache resolved sample/rate at note spawn; hoist the blend buffer.
- [x] FM instrument path (`instruments.rs:267-270`) has no band-limiting — sidebands fold
      at high notes. Nyquist-aware index scaling or oversampling.
- [x] `auto_master.rs` LUFS — FULLY BS.1770 as of 2026-07-07. First pass (07-06)
      relabeled it honestly + fixed the amplitude-averaging channel bug. Second pass
      (07-07) added the real two-stage K-weighting pre-filter (high-shelf + RLB
      high-pass, exact published coefficients at 44.1/48 kHz, RBJ fallback elsewhere)
      and the BS.1770 channel sum — `integrated` is now a genuine LUFS value. Tests:
      exact-coefficient guard + low-freq attenuation + high-shelf lift.
- [x] Cache the FFT plan in `MelExtractor` (`mel_extractor.rs:97` re-plans per call).
- [x] New property tests: reverb RT60 decay, click/discontinuity detection at note
      boundaries, output-bounded-under-hot-state stress test.

## Phase 3 — Make the learning real [the big one, 1-2 weeks]

The single highest-leverage gap: **no trained artifact reaches the generator.** The only
net with capacity (CfC, 64-dim) is initialized from genesis and never trained; ES training
optimizes exactly 6 decode scalars.

- [x] Pick ONE training path and land it end-to-end — DONE 2026-07-07 for the
      production `MelodyPredictor` (`train_melody_predictor` bin: ridge least squares on
      `midi_trainer` features, 1,136 train / 140 held-out MAESTRO files). Held-out
      direction accuracy 69.1% vs 38.2% (old hand-tuned constants) vs 51.5% (majority
      baseline); duration MAE 0.25 vs 2.18 beats. BONUS FINDING: the old header's
      mythical "93.2%" was explained — `melody_to_training_pairs` leaked the target
      interval into the context (off-by-one, now fixed + regression-tested); evaluations
      through that leak read ~93-100%. A higher-capacity CfC readout remains future work.
- [x] Trained artifact + embedded provenance committed: `data/melody_predictor_weights.json`
      (dataset, split, fit method, honest held-out metrics in its `provenance` block),
      embedded via include_str! with a parse-failure guard test. Still open: the separate
      `melody_projections.json` for Neural mode's 6-scalar CfC decode (low capacity —
      superseded in priority by the CfC-readout future work above).
- [ ] External grounding for evaluation (the current loop is closed: hand-coded critic
      judges, optimizers optimize against the same critic):
      - [x] FAD with a real reference set — DONE 2026-07-07 (`examples/fad_external.rs`):
        MAESTRO excerpts as reference, anchored. **Baseline: Symthaea 42.2, noise floor
        9.3 (MAESTRO-vs-MAESTRO), white-noise ceiling 136.8 → normalized 0.258.**
        24-band pseudo-MFCC embedding — not comparable to published VGGish FAD numbers,
        but the anchors make it a valid internal yardstick to beat.
      - [x] Real DEAM regressor — DONE 2026-07-07: `train_deam_regressor` rewritten to
        decode 1,744 DEAM MP3s (symphonia), extract 6 real signal features, fit on a
        song-level split (1,569/175). **Held-out: valence R² 0.410 (MAE 0.183 vs 0.245
        baseline), arousal R² 0.326 (MAE 0.240 vs 0.304 baseline)** — both positive,
        both beat the mean predictor. Weights + provenance in
        `data/deam/va_regressor_weights.json` (local artifact; DEAM data not committed).
      - [~] Blind A/B harness — BUILT 2026-07-07 (`examples/ab_melody_weights.rs`):
        renders the streaming engine trained-vs-fallback per scenario, writes blind WAV
        pairs + answer key to `audio_output/ab_melody/`. Pairs differ substantially
        (rel-RMS 0.8-1.3). Awaiting the actual human listening pass (Tristan's ears) to
        record preferences — that is the improvement evidence metrics can't provide.
      - [x] Wire the trained predictor into production — DONE 2026-07-07: until now
        `MelodyPredictor::predict()` had NO caller in the streaming path (record() fed it
        context but its output shaped nothing — a write-only predictor). Streaming now
        blends the trained prediction into pitch selection (Ψ-scaled, log-frequency,
        scale-snapped). Added `with_fallback_weights()` / `use_fallback_melody_weights()`
        for the A/B harness.

## Phase 4 — Architecture consolidation [2-3 days, parallelizable]

- [ ] **Two disjoint pipelines**: batch `compose()` (lib.rs:342) and `StreamingSynth`
      share almost no code and diverge on pitch/voicing logic; `compose_with_arc`
      (arc.rs:194) is a third near-duplicate of the compose loop. Extract a shared core
      (scale building, gestures, voicing) or make batch drive StreamingSynth.
- [ ] **Eight Harmonies dilution**: `HARMONY_INTERVALS` (pitch.rs:374) is principled and
      load-bearing in `compose()`, but the flagship streaming path picks pitches from
      taste/chord logic and only gesture-snaps to the harmony scale — and the JI tuning
      branch ignores it entirely (pitch.rs:325-341). Wire the signature mapping into
      streaming properly; dedupe the second copy in `wake_protocol.rs:47-56`.
- [ ] **Restructure lib.rs** (23 modules at top, 47 appended after the test block —
      pure accretion): fold into `midi/`, `taste/`, `melody/`, `hifi/` (the streaming DSP
      chain), `training/` submodules. Resolve the `voice` naming collision (voice.rs =
      SATB voice-leading, voice_bridge.rs = TTS).
- [x] **Delete dead weight**: `crates/bridges/symthaea-muse-wasm` (excluded from workspace,
      referenced by no frontend, superseded by mycelix-music's direct dependency);
      `temporal_hierarchy.rs` + `fingerprint.rs` (zero references); decide on
      `wake_protocol.rs` / `collaborative.rs` (example-only).
- [x] `symthaea-atelier`'s non-optional muse dep is used only by its examples — make
      optional. `bevy`/`voice-kokoro`/`voice-broca` features gate ~nothing at muse level —
      document or drop.
- [x] Fix stale `symthaea-atelier/examples/creative_demo.rs` (`composition.samples` field
      doesn't exist — likely doesn't compile).

## Phase 5 — Hygiene [ongoing]

- [ ] 97 clippy warnings → 0 (mostly mechanical: range-contains, Default impls,
      iterator-instead-of-index).
- [ ] Weak tests that assert nothing: `arc.rs:344` (`let _ = ...`),
      `musical_inference.rs:408` (`free_energy_decreases_with_learning` only eprintlns).
- [ ] Add a README (the crate has no docs at all).

## Suggested order

Phase 0 → Phase 1 are cheap and restore trust (green suite, honest labels). Phase 2 makes
the live path robust. Phase 3 is the strategic investment — it converts "consciousness-
driven music synthesis" from an architecture claim into a demonstrated capability. Phase 4
can interleave. Do not start Phase 3 before Phase 1: training against mislabeled
benchmarks reproduces the closed-loop problem.

## Phase 6 — Beat the industry where it can't follow (2026-07-08)

From a 2-agent deep review (render-path quality gaps; neural-infra inventory) + industry
comparison. Context: the composition brain (symthaea-music-theory: functional harmony,
voice leading, counterpoint, ternary/rondo form, 147 ground-truth tests) is now genuinely
good; what a listener hears is limited by the *renderer*, which is still at "MIDI preview"
quality. Meanwhile the two biggest quality assets sit unused on disk: VCSL
(`data/samples/vcsl`, 9.6GB, 4,231 multi-velocity CC0 instrument WAVs — zero code
references) and MAESTRO v3 (`/opt/datasets/maestro`, 121GB aligned score+performance —
mined only for melody intervals, velocity extracted then discarded).

**Strategy (honest):** we will not out-Suno Suno at photorealistic pop on an RTX 2070, and
shouldn't try — end-to-end audio-token generation is compute-impossible locally and
forfeits every differentiator. Muse's winning identity: **the best controllable, adaptive,
provenance-clean composer, with rendering good enough that nobody notices it isn't
neural.** Axes where industry systems (Suno/Udio/Stable Audio/MusicGen/Lyria) are
structurally weak and muse already wins: note-level editability (symbolic Score), real-time
adaptivity (LiveComposer, phrase-level reaction, 0.01s compute), licensing provenance (no
scraped training data; CC0 samples; textbook theory), on-device inference,
consciousness-state coupling, provable theory invariants. Rendering is the entire
remaining battle.

### Tier 1 — Performance & timbre honesty (days, no ML; transforms the listening experience)

Status: **ALL SIX LANDED 2026-07-08** (commits `29b87ff30f`..`Tier 1.5`), 493 lib tests
green (`--features theory`), each item committed+tested separately.

- [x] **Stereo field** (`29b87ff30f`): per-role pans via `pan_for_role` (lead +0.25,
      harmony −0.4, bass center; climax doubler −0.2, opposite the lead). Side/mid RMS
      test proves a real stereo image. Haas deferred (needs per-channel delay inside
      render, not per-voice offsets).
- [x] **Round-robin variation** (`31d6429371`): `excite_seeded` + per-note per-partial
      initial phases keyed by `note_seed()` (start/freq/duration bits — varies across
      notes, bit-deterministic per render). Tests prove windows differ at equal energy.
- [x] **Humanizer wired** (`ae56ccf0f0`): new `humanize_score_note` — a deliberate SUBSET
      (Φ-scaled meter-aware onset jitter + zero-mean ±8% velocity noise + optional
      legato); the full phrase-dynamics curve deliberately NOT applied (scores already
      carry structural dynamics — double-applying was the risk). `strum_chords` rolls
      block chords low-to-high (12ms plucked / 6ms keys), release still aligned; chord
      tones share one humanize seed so the roll survives jitter.
- [x] **Velocity→timbre + transients + inharmonicity** (`2de027d226`): per-note
      `NoteTimbre` — piano crossfades the measured FF/PP tables, others get velocity
      tilt; Conklin inharmonicity finally referenced; sustained instruments no longer
      pluck-decay; attack noise (bow/chiff/hammer) outside the envelope; FM index scales
      with velocity (zero-crossing verified). GOTCHA: KS loop filter is dullest at
      brightness 0.5 and brighter on EITHER side — naive velocity scaling DARKENED hard
      plucks; the ZC test caught it; fix pulls b toward 0.5 for soft notes only.
      Still open: fractional KS delay (~4-cent tuning quantization).
- [x] **VCSL sample mapper** (new `src/vcsl.rs`): real recorded instruments for
      piano/harp/recorder-as-flute/tenor-sax/pipe-organ/glockenspiel/balafon/mbira
      (VCSL has NO violin/cello/guitar/clarinet/trumpet — those honestly keep
      synthesis). Opt-in via `SYMTHAEA_VCSL_DIR` or `vcsl::init()`; inactive = bit-
      identical prior behavior. Metadata indexed up front (path-sorted for deterministic
      round-robin), audio decoded lazily+cached. Uppercase-note/lowercase-dynamic
      filename disambiguation (else `f1` parses as the note F1 — tested).
- [x] **Score→MIDI adapter** (`406dc30693`): `export_score_midi` — exact beat-arithmetic
      ticks, real VoiceRoles, score meter, GM programs from the same per-style ensemble
      the audio renderer uses (`Instrument::gm_program`). midly round-trip tested.
      Load into any DAW → production-grade rendering of the composed structure.

### Tier 2 — Arrangement & groove (1-2 weeks)

Status: **ALL THREE LANDED 2026-07-09** (`09b4bf11cd`, `9b2995ec1c`, `73cf12fab5`);
music-theory 158/158, muse 495/495 green.

- [x] **Accompaniment patterns** (`09b4bf11cd`, new `symthaea-music-theory/src/
      accompaniment.rs`): Block/Arpeggio/Alberti/OomPah/Comp. Safety property tested: a
      pattern only re-times the voice-led chord's tones — it can never introduce a wrong
      note. Waltz is ALWAYS oom-pah (its bass drops to quarter notes, walking disabled);
      Cinematic always Block; others pick by seed/2. Classical seed 0 stays Block
      (compat). LiveComposer keys the texture on a stable base_seed (piece identity,
      only retheme() changes it).
- [x] **Percussion in the theory path** (`9b2995ec1c`): style-gated drum_hits() on the
      SAME rubato timeline as the pitched voices (tested — a straight clock would drift
      within one ritardando). Classical/Waltz honestly get NO kit; Folk light pulse,
      Cinematic sparse barline kick, Playful full backbeat. Dry mix (outside reverb),
      pre-master, DrumColor from consciousness state.
- [x] **Vibrato** (`73cf12fab5`): true per-partial FM (phase-integral), 4.5-5.5Hz,
      onset-delayed; organ/struck/plucked honestly None. GOTCHA: carrier retains
      J₀(cf·d/rate) — at 220Hz that's 98% (test must measure in the 880Hz register where
      J₀≈0.66). Legato landed in Tier 1's humanizer; portamento still open.

### Tier 3 — The neural layer, where it actually pays (weeks; all infra proven)

- [x] **CLAP text tower + text-prompt steering** (`ec8d19ed5b`, 2026-07-09):
      `ClapTextEmbedder` (fp32 `text_model.onnx`, 501MB cached — fp32 deliberately, to
      avoid unauditable asymmetric drift vs the fp32 audio tower) + `steering::steer()`
      (compose N seed-varied candidates at 48kHz, rank by cosine, return best + full
      ranking — honest generate-and-rank framing in the API docs). LIVE-VERIFIED
      cross-modally: apt prompt outranks absurd prompt against real audio; end-to-end
      demo ran ("a gentle nostalgic waltz", 6 candidates, similarity spread
      0.397-0.499 — the prompt genuinely discriminates between seeds).
      GOTCHA: the text graph's sole input is `input_ids` (no attention_mask) — probe
      exported ONNX signatures, never assume from HF Python APIs.
      Still open from this item: swapping `param_tuner.rs`'s hand-coded critic fitness
      for CLAP reference-set distance (the preference-loop half).
- [x] **MAESTRO expressive-performance model** (`6818fe7490`, 2026-07-09): new
      `expressive.rs` + `train_performance_model` bin. HONEST SCOPE CHANGE from the
      original sketch: MAESTRO is performance capture, so grid-quantized onset
      deviations measure tempo drift, not expression — the model learns the two
      GRID-FREE dimensions instead: velocity deviation from a ±8-note local mean
      (composer's structural dynamics stay authoritative) and articulation
      (duration/IOI — drift cancels in the ratio). Grid micro-timing = future
      beat-tracking project. Trained on 4.35M pairs/1,136 files; held-out (522k
      pairs/140 files): accent-direction 73% (chance 50%), velocity-dev MAE .0676 vs
      .0759 zero-baseline, articulation MAE .424 vs .587 mean-baseline. Trainer
      REFUSES to write weights that beat no baseline; guard test fails the build on
      placeholder weights. Applied to the melody voice in `theory_realize` (additive
      velocity blend 0.7; articulation replaces flat legato for interior notes).
- [ ] **Pretrained neural vocoder via ONNX** (only if samples aren't enough):
      `spectral_vocoder.rs` confirmed dead end (genesis-random decoder, no-op trainer bin).
      Pragmatic route = pretrained HiFi-GAN/Vocos ONNX through the proven ort pattern.
      Do NOT train one on the 8GB 2070.

### Post-Tier extras landed 2026-07-09 (same push as Tier 3)

- [x] **Generalized applied dominants** (`5e936d3461`, music-theory): the thrice-deferred
      "chromatic melody fitting" rewrite turned out unnecessary — the ii→V7/V
      root+fifth-preservation safety property covers the whole family (vi→V7/ii,
      iii→V7/vi, seed-gated I→V7/IV). vii° (diminished fifth) and minor keys excluded
      with tests.
- [x] **Phrase breathing + climax grace note** (`ace82f32e2`): phrase-final notes release
      early (real silence, slot unchanged; final note rings); one acciaccatura leans
      into the piece's climax from below (diatonic neighbor scan reaches 3 semitones —
      harmonic minor's augmented second below the leading tone, found by a failing
      test). muse's learned articulation now respects symbolic gaps (written/slot <
      0.85 wins over the model).

### Artist-app wave, landed 2026-07-09 ("proceed with all")

- [x] **VSCO2-CE sampled LEAD instruments** (`4c1474d910`): 5.2GB CC0 clone at
      `data/samples/vsco2-ce` (sibling of the VCSL root, or `SYMTHAEA_VSCO2_DIR`).
      Solo violin, cello section, real flute, clarinet, trumpet — the most exposed
      voices no longer synthetic. Naming parses with the existing tokenizer unchanged.
- [x] **Muse Studio** (`49cb819034`, feature `studio`, bin `muse_studio`, port 8400):
      the sketch-partner MVP — intent controls → N candidates → listen inline →
      export MIDI (the product) + WAV (the preview); optional CLAP prompt ranking with
      graceful degradation. Smoke-tested live end-to-end (valid WAV + SMF served).
      Run: `SYMTHAEA_VCSL_DIR=data/samples/vcsl cargo run --release -p symthaea-muse
      --features studio --bin muse_studio`.
- [x] **Counter-melody + texture evolution** (`4089398d99`): new
      `VoiceRole::CounterMelody` — the return gains a voice-led, collision-avoided
      second line (cello answering the violin); the B section strips to melody+bass
      for its first half (the re-entry is an arrival; the full return lands as a
      return). Ground-truth tested; MIDI export gains a Counter track.

### The framework wave, landed 2026-07-09 ("proceed with all", round 2)

- [x] **Cadential harmonic rhythm + plagal coda** (`2a980271d3`): cadence-approach
      bars split triad → seventh (superset-only = melody-safe by construction); bass
      walks root→third into V; 2-bar IV→I coda under a held tonic, final ritardando
      follows automatically. Coda intensity bug caught by the existing tension-arc test.
- [x] **CompositionSpec** (`97b702e193`, new `symthaea-music-theory/src/spec.rs`): the
      answer to "architectural prison → user-owned framework". Every hard-coded choice
      (motif banks, progression, accompaniment/form/ensemble pools, meter, tempo,
      texture policies, drum policy) is now one validated, serializable spec; the five
      Styles are preset VALUES of it; the engine keeps the invariants (a spec cannot
      write a wrong note). Muse Studio: GET /api/spec/{style} + spec editor in the UI +
      compose-with-custom-spec (HTTP 400 with reasons on invalid). Live-verified: a
      mutated Folk spec (oom-pah, no drums, 4-bar coda, custom ensemble) rendered
      41.9s of audio; a broken motif was rejected with the exact reason.
      170/170 theory + 503/503 muse tests.

### The groove wave, landed 2026-07-09 ("proceed with all", round 3)

- [x] **Spec-controlled swing** (`979dcb9103`): TextureSpec.swing (off-beat position,
      0.5 straight ..= 0.75; validated; serde default). Pure performance-layer time map
      (swing_beat: monotonic, identity at barlines) composed before rubato in a shared
      Timeline so all voices AND drums swing together. Presets stay straight — swing is
      the user's dial. Swung-vs-straight render comparison tested.
- [x] **Articulation-aware samples** (`f8e0d854cf`): VSCO2 staccato/spiccato banks
      indexed for all five leads; notes <250ms play REAL short articulations, falling
      through to sustains then synthesis. Piano honestly has no short bank (tested).
- [x] **Studio ergonomics** (`17b57777e4`): named specs persist in data/specs/ (save/
      list/load endpoints + UI); per-candidate "More like this" (seed_stride=6 holds
      form/accompaniment/motif, varies orientation + progression). Live-verified.

### The ear-driven wave, 2026-07-09 (user listening reports → measured diagnosis → fix)

Loop: user listens to a Studio export, reports what's wrong; we measure (per-2s-window
peak/RMS/HF-ratio + note-level probes), fix at the right layer, add regression tests,
re-export, re-measure. Landed so far: voice-balance velocity formulas, register-fold
ceiling (horror-spike), treble-boost cap + Hermite interpolation. This round ("some
parts are still harsh"):

- **Diagnosis discipline paid off twice.** Hypothesis 1 (staccato threshold too loose)
  was DISPROVEN by a bit-identical re-export; kept the 250→130ms tightening as a
  defensive guard with an honest comment. Hypothesis 2 (sample-bank fallthrough to
  synthesis) was real but MINOR — only 3 counter-melody notes (cello susvib tops at D4;
  B4/C5 fell to additive synthesis); fixing it moved the HF measurement ~5%. The
  note-level window probe (`examples/probe_fallthrough.rs`) found the real cause:
  **every harsh window has the violin at MIDI 89-93 (`f` dynamic layer), and in the B
  section the harp doubles it at UNISON** — two instruments stacked on the ear's
  2-5kHz sensitivity peak.
- [x] **Two-stage sample pick** (`pick_with_window`): strict +5/−7 window first, then
      a relaxed +12/−24 retry — a far-shifted real recording beats a mid-line timbre
      switch to synthesis. Regression test pinned to the cello-at-B4 case the probe
      proved.
- [x] **Octave-below climax doubling**: the doubling voice's own comment said
      "underneath" but the code doubled at pitch; now doubles at −12 (adds weight,
      not glare).
- [x] **Equal-loudness velocity taper** above E6 (MIDI 84, −2.5%/semitone, floor at
      −30%): quiets the top register AND selects the softer recorded dynamic layer —
      most of the timbral relief comes from the layer switch.
- **Re-measured (v6 export)**: the worst windows collapsed to baseline — t=38s
  HF-ratio 0.177→0.012, t=20s 0.124→0.009, t=50-52s ~0.11→0.010, t=78s 0.153→0.022;
  piece-wide max 0.177→0.103. Remaining ~0.09 windows are violin `f`-layer notes at
  MIDI 76-84 (below the taper knee) — ordinary violin brightness, awaiting the next
  listening verdict before deciding whether to lower the knee.

### The reviewer wave, landed 2026-07-09 (external listening review of v6)

A detailed outside review ("lonely, walking-through-a-half-built-world-at-dusk"; "a
good seed... needs more human timing, clearer instrumental roles, one stronger
emotional arrival") triaged into four fixes:

- [x] **Performed MIDI export** (`export_performance_midi`): the review measured the
      MIDI straight-grid while the spec said swing 0.62 — the symbolic export bypassed
      the performance layer entirely. Extracted the renderer's voice-building into a
      shared `performance_voices()` (swing∘rubato timeline, learned expression,
      humanize, equal-loudness taper, doubling voice), so the WAV and `.mid` are now
      one implementation. Studio serves the performed export; `export_score_midi`
      stays as the clean-grid symbolic counterpart (its "expressive jitter is the
      DAW's job" stance was a deliberate design, not a bug). Regression test parses
      both exports and asserts swung off-beats land later.
- [x] **Distinct counter-melody voice**: bass and counter shared the cello. New
      `contrast_counter()` rule (first of clarinet/cello/flute differing from both
      lead and bass — "soft clarinet" was the reviewer's own first suggestion) plus
      `counter_instrument: Option<String>` on the spec for explicit choice.
- [x] **Mastering headroom + section leveling**: default target −16 LUFS (was −14)
      and ceiling −1.5dB (was −1.0) — the pop target pushed every climax into the
      limiter ("loud returns feel rendered hot"). New two-pass section leveler
      (~3s windows pulled halfway toward the piece mean, ±1.5dB cap, 2s smoothing,
      silence untouched) narrows section jumps gently instead of violently.
- [ ] **Stronger final arrival**: partially served by modal cadence work below +
      existing final-cadence ritardando (0.9·spb); revisit after the next listen.

### The enjoyability wave, 2026-07-10 ("honestly it's not really enjoyable")

The pivotal verdict: after every defect fix, the music still wasn't *enjoyable*.
Diagnosis shifted from defects to musicality — three structural absences, each fixed
at its own layer:

- [x] **Arrangement dramaturgy** (`TextureSpec.intro_bars` default 2,
      `staged_entrances` default true): nothing ever entered or left — melody,
      harmony, and bass all played wall-to-wall from bar 1. Now: an accompaniment-only
      intro (quiet, tonic chord, the pattern alone), the bass sits out the opening
      antecedent and arrives with the consequent, and the bar before every return is
      thinned (bass out, harmony out of its second half) so the reprise lands as an
      ARRIVAL. Implemented as post-realization passes over the score
      (`apply_staged_entrances`, `prepend_intro`) in section-bar coordinates derived
      from the form itself.
- [x] **Melody breathing + one big moment** (`TextureSpec.held_arrivals` default
      true): the motif machinery filled every beat of every bar — no rests, no held
      peak. Now: every mid-piece phrase-final bar reduces to approach + HELD cadence
      tone (the question hangs while the accompaniment answers; the next phrase's
      restatement is the reply), and the climax note absorbs the note after it so the
      peak is dwelt on, not passed through. The pre-return bar composes with this
      beautifully: held melody tone over thinned accompaniment = a real held breath.
- [x] **Legato crossfade** (renderer): every sampled note re-attacked — the recorded
      bow/tongue transient on each note of a slurred line is the "MIDI preview"
      sound. Consecutive notes ≤35ms apart on bowed/blown instruments now start
      playback ~80ms past the sample's attack with a 25ms fade-in, crossfading with
      the previous note's already-sounding 250ms release tail. Plucked/struck timbres
      are exempt (their sound IS the attack). Staccato notes never legato.

All three default ON (presets + serde defaults); old saved specs keep loading.

**Correction after the v8 listen ("the strings are not played properly")**: the first
legato cut slurred EVERY adjacent note pair — including the walking bass, whose
re-articulation IS the walk — and skipped a heavy 80ms of attack, erasing note
definition; meanwhile the new held notes rendered at constant gain for ~3 seconds
(a note parked, not played). Fixed: legato now gates on stepwise motion (≤3.5
semitones), never the bass voice, 40ms skip / 15ms fade; long bowed/blown notes
(≥1.2s) get a messa-di-voce swell (0.85 → 1.12 at 40% → 0.85), regression-tested
comparatively against the flat render.

### The damage pass, 2026-07-10 ("it has shape, but it lacks argument")

Review of v9: "less boring, but still not fully alive... the additive arrangement
formula (harmony → melody → bass → doubling → counter → louder → ending) is coherent
but predictable. It feels like the generator is saying 'now add the next layer,'
rather than 'something happened'... You improved the surface. You have not yet
solved the taste problem... What I would do next is not 'add more.' I would add a
**damage pass**." Implemented as `TextureSpec.damage` (0.0 pristine → 1.0 full,
default 0.5), a post-composition pass that deliberately injures the clean piece.
Every prior pass added ORDER; this one is the opposing force. Devices by rising
threshold, all placed by the piece's own structure (deterministic, no dice):

- **≥0.2 exposed climax**: harmony AND bass cut for the climax's whole bar — the
  peak stands alone instead of being carried.
- **≥0.2 transformed coda**: the opening motif returns CHANGED — augmented, an
  octave down, over a borrowed mixture subdominant (the quality-flip of the
  diatonic IV, keyed off IV's own quality so it's honest in every mode), resolving
  to I only in the final bar; fragment tones carrying the altered pc bend with the
  harmony. Replaces the held-tonic plagal fade the review called "it just recedes."
- **≥0.35 dark bass entrance**: the staged entry lands an octave low, slightly
  louder — "slightly too low, too dark."
- **≥0.35 the expectation hole**: the return's second-bar downbeat — the note the
  ear knows by now — is removed.
- **≥0.5 the "wait" tone**: one chromatic passing tone mid-departure.
- **≥0.5 counter disagreement**: the counterline's middle notes shift half a beat
  late — another will, not a decoration.

Test architecture note: the CLEAN contracts (plagal coda shape, counter half-note
gait/collision guard, zero-C#-in-Dorian) are now pinned at `damage: 0.0`; the
damage devices have their own comparative tests (exposed climax, −12 entrance,
counter shift, coda quote + borrowed-iv pc, determinism). 185 theory + 516 muse
tests green.

**v10 review verdict + follow-up (2026-07-10)**: "This is the first version where
I believe the system is composing with consequence... the next enemy is not
boredom anymore. The next enemy is memorability." Two actions taken from it:

- [x] **Climax-cut audit (review caught a real bug)**: "the bass is still present
      in those bars." Confirmed: the exposure removed notes by ONSET only, so a
      bass note sustaining in from the previous bar still carried the peak — and
      the test shared the blind spot. Fixed: the window now covers the climax
      note's full sounding span, removal is by overlap, and inbound sustains are
      truncated at the window edge; test strengthened to the overlap check.
- [x] **Damage PLANNER (was: fixed wounds)**: `plan_damage()` diagnoses the clean
      score — smoothness (stepwise × IOI uniformity), bar-rhythm repetition,
      climax velocity prominence, groove uniformity — and selects the injuries
      that answer them; `damage` sets how many fire (0.2 → 1 … 1.0 → all 5) and
      the seed jitters near-ties only, so two pieces don't share the same scars.
      Tests: determinism, wound-count scaling, seed variation (8 seeds → ≥2
      distinct plans), and diagnosis→selection (a synthetic maximally-smooth
      repetitive melody must draw interruption devices). Device mechanics
      re-pinned at damage 1.0. 187 theory tests green.

Named next enemy (deliberately not started in the same wave): **memorability** —
melody hook design (rhythmic identity + repetition schema + payoff in the motif
layer, "hook surgery, not arrangement damage"), plus a more melodically/
rhetorically assertive counterline, and more bite for Dorian.

### Hook surgery, 2026-07-10 ("can it make a theme worth injuring?")

v11 verdict: "materially better... has consequence now. But it still does not
quite have a face... generate a tiny memorable cell BEFORE generating the full
melody — a 3-5 note 'name' that can survive augmentation, inversion, silence,
and reharmonization." (The v11 bass audit was also answered: the planner
DESELECTED the dark-bass wound — plan was [ExpectationHole, WaitTone,
ExposedClimax] — working as designed, reviewer approved the softer result.)

- [x] **`hook.rs`**: `HookCell::generate(seed, meter)` builds the piece's name
      from identity-bearing rhythm skeletons (short-short-LONG, dotted call…)
      × contour skeletons (insistence-then-reach, reach-and-recoil…), with the
      identity PREDICATES (≥2 distinct durations at ≥2× ratio; leap-plus-recoil
      or repeated-note) enforced by construction and re-checked. 16 seeds → ≥6
      distinct names, deterministic.
- [x] **Grafting**: the hook becomes the HEAD of the bar-motif (after seed
      orientation, so the name is always stated upright first); the spec
      template's tail fills the remaining beats exactly. Survival comes free
      from existing machinery: sentence structure restates the head,
      development fragments it, and the transformed coda already quotes the
      opening — which is now the hook. `TextureSpec.hook_cell` (default true).
- [x] **Two real interaction bugs found by the wider contours**: (1)
      `apply_held_arrivals` could DELETE the climax when the piece's peak sat
      inside a cadence bar's interior — now hold-exempt; (2) phrases spanning
      past the fold window couldn't octave-fold whole, leaving notes above A6 —
      per-note fold added as documented last resort. Grace-note test now scans
      seeds (the ornament's long-note precondition isn't guaranteed under hook
      rhythms).
- E2E test: the first melody statement carries the hook's rhythm exactly, and
  hook-off changes the opening. 191 theory tests green.

Deliberately deferred from the same review: assertive counterline rhetoric
(answer/interrupt/mock — candidate: echo the hook during held-cadence bars) and
Dorian danger (♭6-biased wait tone / culturally specific instrumentation) —
next wave, after ears on v12.

### The comfort pass, 2026-07-10 ("kindness to the ear")

v12 verdict: "the composition got more interesting, but the instrument renderer
is now making it sound cheaper and harsher than the writing deserves... The next
patch should not be more drama. It should be kindness to the ear." Reviewer's
prescriptions, all landed:

- [x] **Register comfort**: `MELODY_CEILING_MIDI` A6→E6 (88); equal-loudness
      taper knee E6→E5 (76) and steepened (−3.5%/semitone → 0.58 at the
      ceiling) — "avoid naked violin above roughly E5–A5... soften velocity
      aggressively." Velocity reduction doubles as SOFT-sample-layer selection.
- [x] **Doubling shimmer cap**: the review measured doubling hits at MIDI
      velocity 100 ("should shimmer, not stab; cap ~55–70") — now ×0.7 capped
      at 0.5 (≈64), re-asserted AFTER humanize jitter so it's a guarantee.
- [x] **Room**: `ReverbConfig.wet_floor` (default 0.1 — every other render
      unchanged); the chamber path sets wet_floor 0.2 + room 0.68 — "dry GM
      violin is brutal; a little room helps the ear forgive the edges."
- [x] **Demo palettes** (the reviewer's exact prescriptions): wistful →
      clarinet lead / warm pad harmony / cello bass (counter auto-contrast:
      flute); dorian → wooden flute / soft marimba / upright bass, away from
      the "default MIDI orchestra fantasy-town" violin/harp.
- [x] **Timbre regression test**: doubling velocity cap pinned
      (`doubling_voice_shimmers_instead_of_stabbing`); taper knee/slope tests
      updated. 191 theory + 517 muse green.

### Modes — landed 2026-07-09 (was: scoped, deferred)

Implemented exactly along the scoping below, in one session, all ground-truth tested
(181 theory tests green):

- **Representation**: `Tonality::Modal(Mode)`; `Key::modal(tonic, mode)` normalizes
  Ionian→Major, HarmonicMinor→Minor, admits Dorian/Phrygian/Lydian/Mixolydian/Aeolian,
  and refuses the grammar-incompatible modes (Locrian, pentatonics, whole-tone,
  melodic minor) — `Option`, not panic. Old serde payloads unaffected (unit variants).
- **Triads generalized for free** as predicted: D Dorian i/IV(major!)/♭VII, G
  Mixolydian I/♭VII, A Aeolian minor-v/♭VII all fall out of
  `degree_pitch_class` + `classify_triad` — tests pin them.
- **Cadence grammar**: `Key::cadence_dominant_degree()` — 5 for functional keys and
  Lydian, 7 (♭VII) for Dorian/Mixolydian/Aeolian, 2 (♭II) for Phrygian.
  `Period::parallel_in`/`parallel_sentence_in` take the degree; `Form::ternary`/
  `rondo` pass each SECTION's own key's degree (modal home closes ♭VII→i while its
  functional B key keeps V→I). The V7-coloring and cadential harmonic-rhythm split in
  the composer key off the same degree.
- **Applied dominants**: stay Ionian-gated automatically (`!= Tonality::Major`).
- **Modulation**: modal `relative()` = the shared-pitch-class Ionian (D Dorian → C
  major — the no-chromatic-alteration contract holds exactly); modal `parallel()` =
  brightness flip to a functional key on the same tonic. Round-trip involution
  documented as Major/Minor-only.
- **Spec plumbing**: `mode: Option<Mode>` (serde default; overrides the
  valence→major/minor mapping), validation rejects incompatible modes with a
  didactic message. Studio needs no UI change — the spec JSON editor and the
  preset endpoint expose the field.
- **End-to-end honesty test**: a D Dorian ternary piece across 4 seeds contains ZERO
  C# anywhere (home sections Dorian, B section relative C major, applied dominants
  gated) — a single borrowed leading tone would fail it.
- **Deliberately untouched**: `LiveComposer` still cadences functionally (its keys
  come from valence, never modal); coda's plagal IV→I works verbatim in every mode
  (Dorian's major IV is a gift here).

### Modes — original scoping (kept for the record)

The remaining musical-universe item. Same discipline as the secondary dominants
(deferred 3× until the safe generalization was found — and that patience is why it
landed cleanly): `Key { tonic, tonality }` threads through the crate's most
invariant-sensitive seams. The scoping:

1. **Representation**: `Key` gains a mode (`Mode` enum + offsets already exist in
   `scale.rs`); `Tonality::{Major,Minor}` become the Ionian/HarmonicMinor cases of it.
   Spec: `mode: Option<Mode>` overriding the valence→major/minor mapping.
2. **Chord construction generalizes for free**: `diatonic_triad`/`seventh` already
   build from `degree_pitch_class` + `classify_triad` — mode-agnostic by construction.
3. **The careful seams** (each needs a decision + ground-truth tests):
   - Cadence grammar: Dorian/Mixolydian have no leading tone — v–i and ♭VII–I are the
     idiomatic closes; the V-forcing in `Period::parallel` must become mode-aware.
   - Applied dominants: already gated `Major`-only — keep that gate (modal music
     doesn't want them; honest exclusion).
   - Ternary/rondo modulation: `Key::relative()` assumes major↔minor; for modes the
     honest B-section move is the parallel or the subdominant — decide per mode.
   - Valence mapping: spec-mode overrides valence; document that interplay.
4. **Order of work**: representation + triads (tests) → cadence grammar (tests) →
   modulation choice (tests) → spec plumbing + Studio → listening pass.

## The Artist Plan (2026-07-10) — from generator to instrument

The FluidSynth A/B settled the renderer question ("if I had heard these renders
first, I would have gone straight to talking about melodic identity and
expression") and moved the bottleneck. The reviewer's ranking of what remains,
adopted as the roadmap — note synthesis is now LAST:

1. **Memorable motifs** (largest) — the hook cells exist; whether they produce a
   hummable face is a TASTE question that iterates through ears, not architecture.
   The Studio is the tool: compose candidates, mark keepers, evolve the
   rhythm/contour skeleton banks from what survives. (Candidate: a "keeper" flag
   in the Studio that logs the winning hook cells.)
2. **Phrase-level expression / dynamic articulation** (reviewer's new
   prescription: "not just dynamics — legato vs detached, accents on structural
   notes, longer phrase endings, short pickups, gentle delays into emotional
   notes"). Concrete next devices: metric accents (downbeat +velocity), a
   20–35ms lean INTO the climax, articulation ratio varying with phrase position
   (detached statement → legato cadence approach).
3. **Countermelody personality** — answer/interrupt/mock, not just support.
   Candidate: the counter echoes the HOOK during held-cadence bars
   (call-and-response with the piece's own name).
4. **Long-range orchestration** — "the palette stays constant... imagine the
   final return introduced a subtle new color." Candidate: `ReturnA` swaps or
   adds one voice's instrument (spec-controllable `return_color`).
5. **Synthesis** — only after all the above.

Artist-usability track (parallel, incremental):
- [x] **FluidSynth render backend** (this wave): the Studio renders every
      candidate's performed MIDI through the soundfont engine when available
      (launcher provides it via nix-shell), native render as fallback; each
      card badges which engine served it.
- [x] **Piano-roll visualization** (this wave): per-candidate canvas of the
      performed notes, voice-colored (melody amber, bass green, counter blue,
      doubling faint), velocity as opacity, playhead synced to the audio,
      click-to-seek. An artist can now SEE the arrangement, the wounds, and
      the hook.
- [x] **Keeper flag + hook-cell logging** (this wave): ♡ keep on every card →
      `data/taste/keepers.jsonl` (seed, spec, mode, ensemble, renderer, hook
      cell). The raw material for LEARNED taste — which motifs survive ears.
- [x] **State sliders under the soundfont backend** (this wave — user bug
      report "the music slider doesn't work" the day FluidSynth landed: the
      sliders drove the native renderer, which the soundfont path bypassed):
      `RenderColor::from_state` maps consciousness→reverb room/level,
      noradrenaline→dryness, serotonin→tail darkness, dopamine→chorus shimmer
      into fluidsynth `-o` settings. The labels tell the truth again.
- [x] **Intro variety** (this wave — "why do most songs start the same?"
      Because they did: one intro treatment): three seed-picked doors —
      pattern alone / pattern over a low tonic pedal / held chord swell.
- [ ] Per-voice mute/solo (auditioning the counterline alone).
- [ ] A/B toggle between two candidates at the same playhead position.
- [ ] Stem export (per-voice WAV) for DAW handoff.
- [ ] Provenance statement embedded in exports (CC0 samples, MAESTRO-fitted
      expression, no scraped audio) — the licensing story IS the product.

Adopted from the meta-review of this document (2026-07-10):
- [ ] **Split this file** — it now mixes architecture, roadmap, journal,
      listening diary, and evidence. Target: `docs/muse/ARCHITECTURE.md`
      (what Muse IS today), `ROADMAP.md` (what's next), `JOURNAL.md` (the
      chronological waves, moved wholesale), `LISTENING_NOTES.md`,
      `EVIDENCE.md` (measurements/A-Bs). "In six months, someone new will
      have to read a huge amount before finding what Muse is today."
- [ ] **Musical-memory evaluation** — the missing layer: reproducible human
      outcomes, not "objective catchiness." Protocol sketch: after one
      listen, can the listener tap/sing the opening motif? blind
      recognition of the motif 30 minutes later? which candidate gets
      voluntarily replayed (the Studio can measure replays TODAY)? which
      gets described with consistent words? Hook surgery gets the same
      discipline as DSP.
- [ ] **The self-curation loop ("can she hear what she makes?")** — she can,
      partially: CLAP audio embeddings (feature `clap-fad`) are her ears;
      the critic and the damage-planner diagnostics are her analysis. The
      missing piece is a TASTE VECTOR: bootstrap from keeper data (above) —
      embed kept vs. discarded candidates, learn the direction, then let
      Muse compose N, listen to her own renders, and present her favorites
      with reasons ("this one because the hook recurs cleanest"). That is
      the reviewer's Compose → Diagnose → Revise → Render → Critique loop
      made explicit.
- [ ] **More genres** — presets are data, not architecture: Nocturne,
      March, Tango, and a Pentatonic/Koto spec are each ~an hour of motif
      bank + progression + ensemble curation. But genre depth beyond that
      (swing feels, idiomatic bass lines, groove templates) belongs AFTER
      articulation, or every genre inherits the same stiffness.
- [ ] **Music-theory depth** — the theory is deliberately conservative
      (diatonic + applied dominants + one mixture chord + modes). Worth
      adding WHEN a musical need names it: secondary subdominants, real
      sequences (circle-of-fifths episodes), suspensions/retardations
      (huge expressiveness per unit of risk — a 4-3 suspension at every
      cadence would do more than another chord type), pedal points.
      Not worth adding: jazz extensions wholesale — no current genre asks.

## The Temperament Plan (2026-07-10) — escaping the emotional monoculture

The v15 review named the new bottleneck: "if I listened to ten Muse pieces in a
row... they all inhabit a similar emotional world: thoughtful, bittersweet,
careful, restrained... teach Muse to inhabit different emotional BEHAVIORS —
compositional attitudes, deeper than style presets." Adopted:

- [x] **Attitude framework** (this wave): `CompositionSpec.attitude:
      Option<Attitude>` — None keeps the native temperament exactly. Each
      attitude = dial bundle + a SIGNATURE DEVICE the default never uses:
      **Grief** — suspension chains at chord changes (the 4-3/6-5 grammar: the
      old chord's third held over the barline, resolving down by step; the
      first genuinely new music-theory device since modes), block
      accompaniment, tempo ×0.85. **Defiance** — notated syncopation (interior
      downbeats arrive an eighth early, held through; written into the score
      so MIDI carries it), assertive bass. **Joy** — tempo ×1.08, interior
      notes a tenth lighter. **Curiosity** — the final melody tone resolves UP
      to degree 2: the piece ends asking. All four ground-truth tested in one
      behavioral test (suspension pairs counted, syncopated onsets counted,
      bass/tempo/duration deltas, final-pc question vs. resolution).
- [ ] **Motif memory** (next): the hook's long-range trajectory — introduced
      confidently → fragmented → inverted → hidden in the BASS → finally
      complete. The reviewer: "where many great instrumental works become
      deeply satisfying." We have the pieces (development machinery, coda
      quote, counter echo); what's missing is a per-section treatment PLAN.
- [ ] **Return Color as a transformation layer** (next): not just a new
      instrument — register, articulation, harmonic reinterpretation, spatial
      placement as a bundled "the return is DIFFERENT" concept.
- [ ] Attitude-aware damage planning (Defiance→interruption, Curiosity→wait
      tones) — deferred from this wave to bound scope.

## The Style Roadmap (2026-07-10) — languages, not labels

Adopted from the echo-hook review, which also settled the philosophy: "most of
these are COMPOSITIONAL LANGUAGES, not just sonic aesthetics — each teaches
Muse a new way of organizing musical ideas."

**Next ten, in order** (Nocturne shipped in the style-expansion wave):
1. ~~Nocturne~~ ✓ 2. ~~**Theme & Variations**~~ ✓ (2026-07-11: `FormKind::
Variations` → `Form::variations` — theme (A) / minore (B, parallel key +
contrasting transform) / figuration (C, "division" variation via
`figuration_variation`: long notes split into passing/neighbor connecting
tones, duration-exact) / theme verbatim (ReturnA, earning the judgment
machinery's "finally complete" lift). THE invariant that makes it a
variation set and not more ternary: every section keeps the theme's
progression degrees — the ground never moves. In Nocturne + Lullaby
form pools; any saved spec can opt in via `"form_pool": ["Variations"]`.) 3.
~~**Fugue**~~ ✓ (2026-07-11: `fugue.rs` — a three-voice fughetta with
every load-bearing device real: exposition (real answer at the diatonic
fifth, derived countersubject = retrograde inversion up a third),
head-fragment episodes over a walking bass, inverted middle entry on the
submediant, true stretto (half-subject stagger, the intensity peak),
augmented final entry with its tail bent to the tonic.
`FormKind::Fugue` branches out of the period pipeline entirely;
`Style::Fugue` preset (organ/piano/cello). Documented limits = the
upgrade path: derived not species-checked counterpoint, real not tonal
answers, one diatonic collection. **Empirical Φ finding worth keeping**:
the fugue measured Φ≈0.003-0.02 vs Classical's 0.051 — the OPPOSITE of
the naive "fugues share the most material" prediction, and the metric is
plausibly RIGHT: its consonance-excess channel is detecting that the
derived countersubject is not voice-led (vertical intervals hover at/below
the independence baseline), i.e. it flagged exactly the documented
limitation. Species-checked counterpoint is now not just a craft upgrade
but a falsifiable one: doing it should move Φ. FOLLOW-UP 2026-07-11: it
was built, and the hypothesis was falsified TWICE — first draft bent
thematic material and LOWERED Φ; the redesign (themes sacrosanct, bass
bends, strong-beats-only) raised the consonance channel +25% with the
motif web intact and Φ didn't move: λ₂ is a bottleneck measure and the
fugue's min-cut is TEMPORAL. The insight redirected the roadmap to the
passacaglia, below.) 4. Tango (WAITS on real rhythm cells: habanera
accompaniment first) 5. ~~Celtic~~ ✓ (2026-07-12: Mixolydian, sextuple
meter with a real JigGait rhythm cell — FiveGait's 3+2 anchor mechanism
extended to 3+3 — a sustained tonic-fifth drone that deliberately IGNORES
the harmony above it, and unaccented "cut" ornaments distinguished from
appoggiaturas by construction: always quieter, always brief, never on the
beat. Live-verified against the running Studio server: the served Bass
voice alternates 65.4Hz/98.0Hz [C2/G2] every ~1.5s regardless of chord
motion, and 58 short-quiet-then-louder note pairs [cuts] appear in a
4-bar Melody voice.) 6.
~~Passacaglia~~ ✓ (2026-07-11: promoted out of order BY the Φ-bottleneck
result and shipped with a pre-registered in-tree experiment — invariant
identity-bearing ground vs anonymous ever-changing bass, CONFIRMED on
all seeds: Φ 0.038-0.048 vs 0.019-0.040, up to 2×, the highest Φ of any
contrapuntal form measured. Two lessons kept in-code: the first control
was contaminated (oriented() transforms of a subject-derived ground =
a RIVAL integration strategy — bass imitating the melody's family
integrates ~as well as strict invariance — not a null); and mean edge
weights ≠ λ₂ (seed 1's control had higher means on BOTH channels and
lower Φ — invariance fixes the WEAKEST link). The ground REMEMBERS per
the review's design: stated / walked / filled / ALTERED (deepest tone
lifted) / RESTORED as peak / FRAGMENTED / COMPLETE with tonic bend +
lift. Studio now also reports local_coherence + global_coherence per
candidate and logs both to keepers — "two kinds of coherence" as
observables.) 7. March ✓ (shipped; deepen) 8. Nordic
(sparse texture, open intervals, long silences) 9. Japanese (ma — space,
pentatonic restraint, asymmetry) 10. Maqam traditions (the tuning support
exists; build the compositional style).

**Deliberately postponed**: EDM/trap/dubstep/hyperpop/metal/big-band/bebop —
they need groove, production, and idiomatic phrasing that deserve dedicated
attention; shallow versions would be less convincing than the current focus.

**The deeper move — decompose "style"** (adopted as the eventual architecture):
style is becoming too coarse. Orthogonal dimensions the spec already mostly
has fields for: **Form** (binary/ternary/rondo/variations) · **Rhythm**
(waltz/tango/march/free) · **Harmony** (functional/modal/folk/impressionist) ·
**Counterpoint** (sparse/conversational/canonic/fugal) · **Attitude** (the
existing four) · **Palette** (chamber/piano/folk/ambient). A user then
composes combinations ("Nocturne harmony + Nordic palette + Curiosity +
Theme & Variations") — far more expressive than a hundred-genre dropdown.
Styles become curated PRESETS over these dimensions, which they already are
in embryo (CompositionSpec is the substrate). UPDATE 2026-07-11: the
Identity Grammar dropdown became the first shipped axis of this
decomposition (Grammar × Form × Style × Attitude × Palette).

**THE STYLE RULE (2026-07-11 review, adopted as the standing criterion):**
"Every new style must teach Muse one compositional habit that improves at
least one other style." Fugue taught imitation; Passacaglia taught
persistence; Nocturne taught lyricism; March taught rhythmic insistence;
~~Tango~~ ✓ taught RHYTHM CELLS (2026-07-11: `Accompaniment::Habanera` —
the first accompaniment whose identity is rhythm AND accent together, a
per-event accent table the bass locks to with a dotted anchor; blues
shuffle, baroque dance figures, and minimalist pulses are the same
mechanism with different tables. Style::Tango: harmonic minor, 4/4,
100-132 BPM, dotted/syncopated banks, i-iv-V-i, violin/piano/upright
bass).

**The reviewer's 3-group style roadmap (2026-07-11):**
- Group 1 (teach new grammar): ~~Tango~~ ✓, ~~Celtic~~ ✓ (2026-07-12:
  taught the drone — a genuinely static pedal texture, reusable by
  Nordic's open intervals and Ambient's near-stillness — and the
  unaccented cut, the first embellishment device that isn't a lean),
  ~~Blues~~ ✓ (2026-07-12: taught the shuffle — Accompaniment::Shuffle,
  the literal "blues shuffle" the Habanera doc predicted the rhythm-cell
  mechanism would generalize to — and the blue note: the first ornament
  that ALTERS an existing pitch instead of adding a new one, a deliberate
  melody/harmony scale mismatch reusable by Jazz Ballad. 12-bar chorus via
  the existing Archetype progression mechanism, no new form machinery;
  call-and-response left honestly attributed to the engine's existing
  period grammar rather than reimplemented as fake-new machinery),
  ~~Impressionism~~ ✓ (2026-07-12: taught PARALLEL PLANING —
  `TextureSpec.planing` — harmony that abandons functional root motion in
  the contrast section and instead rides the melody's exact contour, a
  struck chord shape re-centered per melody note; Lydian mode, dominant-
  free progression. Shipped in two commits: the device landed initially
  with a real bug found by its own test [the section-boundary window
  copied `apply_development_style`'s antecedent-only formula, landing
  exactly where `thin_departure` strips all harmony — 0 chords found in
  every run], root-caused via eprintln instrumentation and fixed by
  widening to the full section; live-verified via /api/notes scan, 43
  confirmed planing pairs in one piece).
- Group 2 (expand emotional range): ~~Minimalism~~ ✓ (2026-07-12: taught
  the additive process — `apply_additive_process`, the FIRST pass that
  wholesale REPLACES a voice within its window rather than decorating
  what's already there [every prior ornament adds/alters; drone/planing
  replace but only bass/harmony]. The piece's own hook cell grows one
  note at a time until it sounds whole, then shrinks back down, in the
  theme sections — the process substitutes for melodic argument entirely.
  Static mostly-tonic harmony under a pulsing Arpeggio ostinato; no coda
  [minimalist pieces stop, they don't resolve]. Aligns with identity
  grammars exactly as predicted — the additive process IS a kind of
  Memory grammar at the phrase level), ~~Sacred Choral~~ ✓ (2026-07-12:
  taught the harmonic
  suspension — `apply_suspensions`, the first ornament living in the
  HARMONY voice rather than the melody. Real voice-leading candidates
  [outgoing chord tone a diatonic step above an incoming one] get tied
  over as a prepared dissonance, resolving down by step — the actual
  4-3/7-6 mechanism, not appoggiatura wearing a different name. Phrygian
  mode [the ecclesiastical mode no prior style had used]; the plagal
  "Amen" coda needed ZERO new code — `append_coda`'s subdominant-then-
  tonic ending already existed, just switched on. Live-verified: bass
  ends F2→C2 [Amen], 11 confirmed suspension pairs in one piece), ~~Jazz
  Ballad~~ ✓ (2026-07-12: taught `TextureSpec.seventh_chords` — extends the
  existing dominant-only 7th coloring [already safe, already used at
  cadences] to EVERY chord in the progression, not just the cadential one;
  reusable by any future jazz-adjacent style [bebop, big band]. Aeolian
  mode, ii-V-I-vi turnaround via the existing Archetype progression
  mechanism, blue notes + appoggiaturas compounded onto Nocturne's Singing
  rhetoric. Two real bugs found by its own tests, not shipped blind: the
  Harmony voice excludes the chord root [`voicing::lead_upper` skips it],
  so triad→2 tones/seventh→3 tones, not 3/4 as first assumed; and the
  live-gate initially false-negatived because performed/served notes carry
  humanization jitter that splits simultaneous onsets across naive
  rounding buckets — fixed by clustering onsets within 0.05s instead of
  exact-matching them. Also forgot the front-end style-selector/palette
  entries on first deploy [`include_str!` compiles the Studio HTML in at
  build time, so this needed a second full release rebuild]. Live-verified
  via /api/notes scan: 41/53 real seventh-chord onsets in served audio).
- Group 3 (stress the engine): ~~Baroque Dance Suite~~ ✓ (2026-07-12:
  taught HARMONIC SEQUENCE — `TextureSpec.harmonic_sequence`, realized by
  `apply_harmonic_sequence` — the first pass to rewrite a section's
  harmonic PLAN itself [the stored scale-degree progression on the Form]
  rather than decorate or substitute already-realized chords. The B
  section gets a genuine descending-fifths circle
  [I-IV-vii°-iii-vi-ii-V-I], the quintessential Baroque/Pachelbel device;
  cadential tails left untouched so the harmony still resolves under the
  melody's already-baked cadence. Functional tonality [no exotic mode —
  identity comes from harmony/form], slow triple meter, broken-chord
  continuo [violin/organ/cello]. Two real bugs caught by its own isolated
  test before shipping: the antecedent/consequent seam skipped one step
  in the circle [fixed: resume from `ante_len - 1`, not `ante_len`,
  since the antecedent's cadential slot is diverted rather than spent];
  and the motif banks didn't total the style's own 3-beat meter, caught
  by `validate()`. 298 theory + 524 muse green), ~~Progressive Folk/
  Rock~~ ✓ (2026-07-12: taught a genuine mid-piece METER CHANGE — new
  `FormKind::ProgSuite`/`crate::prog_suite`, which bypasses the period
  pipeline (no single `meter_beats` scalar can represent more than one
  meter, so `realize_melody`/`harmony`/`bass` are called separately per
  section) and splices four independently-realized sections onto one
  timeline: theme in 4 → asymmetric riff in 7 (home key) → bridge in 5
  (relative key) → theme's return in 4 (home key), voice-leading carried
  continuously across every change. Reuses `Form::ternary`'s own
  `contrasting_transform` mechanism for the riff/bridge. This is exactly
  the pattern `live.rs` documents as the caller's job — "the crate gives
  you the pieces, not a scripted arc" — realized as that arc. Both new
  tests passed FIRST TRY, no debugging cycle, applying the Baroque wave's
  exact-sequence-assertion lesson from the start. 300 theory + 524 muse
  green. Live-verified: the served bass line's inter-onset gaps form
  three clearly distinct plateaus [~1.0s/~1.8s/~1.3s] — the meter change
  is real in rendered audio, not just the symbolic Score), ~~Ambient~~ ✓
  (2026-07-12: taught HARMONIC STASIS —
  `TextureSpec.harmonic_stasis`/`apply_harmonic_stasis` — when a voice
  [Harmony or Bass] repeats the exact same pitch across two consecutive
  chord onsets, tie the notes into one longer sustained note instead of
  re-striking; a sequential per-pitch sweep chains arbitrarily long runs
  into ONE note. Turns a static repeated progression into a genuine
  drone — repetition becomes duration, not re-attack. Slowest tempo of
  any style [32-52bpm], motifs never faster than quarter notes even at
  the busy tier, zero ornamentation, no damage, no coda [doesn't
  resolve, it stops]. 302 theory + 524 muse green, both new tests first
  try. Live-verified: Harmony sustaining up to 38s, Bass up to 47.75s,
  only 144 total note events in a 322-second piece — genuine "almost no
  events" stillness in real served audio), Film Score (leitmotifs =
  lineage in practice — still unscoped, outside the currently-planned
  roadmap).
- Still postponed: trap/dubstep/hyperpop/metal/EDM (identity lives in
  production/sound design, not symbolic structure).
- The reviewer's year-order: Tango✓ → Celtic✓ → Blues✓ → Impressionism✓ →
  Sacred Choral✓ → Minimalism✓ → Jazz Ballad✓ → Baroque Dance Suite✓ →
  Prog Folk/Rock✓ → Ambient✓. **The currently-scoped style roadmap is
  CLOSED — 15 styles shipped this session (Tango through Ambient).**
  Only Film Score remains unscoped in Group 3 if a future wave wants it.

**Echo confidence** (same review): echoes should have gradations — strong
(near-identical) / fading (shortened) / distant (rhythm kept, contour
altered) / false memory (interval believably changed) — "much like human
recollection." Pairs naturally with memory strength.

**Musical Φ** (user's question, implemented this wave): integration.rs —
the spectral-MIP idea over the score-as-system (voice×segment nodes; pc,
rhythm-grid, and interval-trigram sharing as edges; Fiedler value as Φ).
Honestly labeled: score integration, not consciousness. Validation: the
integration devices (echo, counter answer, hook) measurably RAISE Φ.
Next: surface per-candidate in the Studio; candidate use as a judgment/
taste signal alongside keeper data.

## The Judgment Plan (2026-07-10) — "which memories deserve to survive"

The motif-memory review named the frontier past devices: "teaching the system
JUDGMENT: which memories deserve to survive, which should disappear, which
should return transformed, and which should never come back. That's the layer
where pieces stop feeling generated and start feeling authored."

- [x] **Memory trajectory v1** (this wave): in rondo — two returns — memory
      has an ARC: every return but the last remembers through the attitude's
      ears (wounded), and the FINAL return says the name whole, verbatim,
      with a quiet confidence lift. "Finally complete." Ternary keeps the
      single wounded return (the coda's transformed quote carries its
      resolution). Neutral pieces unchanged.
- [ ] **Memory strength** (next): motifs carry strength that structure
      updates — stated often → stronger; wounded by damage → distorted;
      never restated → fades (drops from later sections). Fading is as
      important as returning: "humans don't remember every phrase equally."
- [ ] **Studio memory panel**: "This piece remembers / It forgets" per
      candidate — the artist literally watches the music evolve. Belongs to
      the explainability wave (/api/explain); the symbolic engine makes it
      true.
- [ ] **Motif beliefs** (horizon): motifs carrying metadata ("I am stable")
      that the music's own experience rewrites ("no, you're unstable") —
      change driven by the piece's history, not by dice.
- [ ] **The reinterpreting return** (from the rondo-arc review): judgment
      should eventually ask not just "should this memory survive?" but "was
      this memory RIGHT?" — one piece concludes the opening idea was true,
      another that it was naïve. The final return doesn't just complete; it
      reinterprets. (Candidate mechanics: mode-flip of the hook's harmony at
      the last return, or the misremembered version BECOMING the accepted
      one.)
- **Forgetting, promoted**: "a motif that slowly loses itself can be just as
  expressive as one that triumphantly returns" — treat fading as a
  first-class emotional device, not a cleanup rule.
- [x] **Hook surgery round 2** (same review: "the memory arc is now stronger
      than the hook itself"): the echo principle — every bar says the name
      twice (statement + varied echo a step lower) instead of anonymous
      template tail; reach alignment — the cell's longest note IS its
      signature note. ~10 aligned names in the pool.

**Decision recorded — NO bulk style/key expansion**: all 12 tonics × 7
tonalities already work; wide-and-shallow style walls multiply surface, not
depth ("every genre inherits the same stiffness"). New styles arrive ONE at a
time when the engine genuinely speaks their idiom (Nocturne now has the
suspension vocabulary it needs; Tango waits on rhythm cells).

## The Companion Plan (2026-07-10) — Listen and Create

Adopted from the Companion Mode review, reframed exactly as prescribed: "autoplay
sounds passive; companion sounds like there's another musician in the room."
Two top-level modes, two mindsets:

- **Listen** — Muse composes FOR you, continuously, evolving. Driven by three
  signals, never self-reported mood alone ("humans are surprisingly bad at
  accurately reporting their own mood"): (1) **intent** (working / walking /
  reading / relaxing / sleeping / creating — a prior over arousal, energy,
  attitude), (2) **attitude** (the existing framework: curious / reflective /
  defiant / joyful / grieving), (3) **RESONANCE** — the living taste model:
  what you keep, skip, replay; which hooks survive. "After six months it knows
  your musical taste better than you could describe it... composing inside
  your musical identity."
- **Create** — Muse composes WITH you: the candidate browser, hook editing,
  palettes, export. Today's Studio, inverted per the Workspace Plan.

Sequencing (data before intelligence):
1. [ ] **Resonance log** — extend the keeper flag to a full session signal:
       skip (<20s abandoned), replay, keep, dwell. Same jsonl, event-typed.
       (The ♡ keeper already exists; this is UI event wiring.)
2. [ ] **Taste vector v0** — frequency-weighted preferences over palette /
       attitude / mode / hook-rhythm-class from the resonance log; no ML
       until the log has real data.
3. [ ] **/api/companion** — a compose-ahead queue: while a piece plays, the
       next is composed under taste bias + gentle nudges (❤️ leans in, skip
       explores away, replay weights that motif family). Tiny nudges, never
       jumps.
4. [ ] **Listen page** — "no parameter wall": Continue Listening / Compose
       Something New; the living now-playing card (what it grew from, current
       attitude/palette, WHY you're hearing it — the symbolic engine makes
       the explanation TRUE).
5. [ ] **Musical memory as first-class** — every kept piece, hook, palette,
       and unfinished sketch as an evolving creative autobiography, not saved
       files. Long-term: Companion Mode as "a long-term musical relationship."
       (This is also where Muse's story reconnects to Symthaea proper — her
       memory systems remembering a shared musical history.)

Boundaries, adopted verbatim: **no biometric determinism** — "How should today
feel?" asked, never inferred; wearables/HR at most opt-in, never primary.

## The Workspace Plan (2026-07-10) — from control panel to musical workspace

A UX review of the Studio ("calm, intentional... but it currently feels like an
engineering control panel, not a musical workspace... help the user think in
music instead of parameters") + a singing/instrument vision. Adopted direction,
phased by leverage — the unifying principle from the review: **every feature
should make it easier to express an idea; the candidate browser is the heart of
the application, not the control panel.**

**Priority 2 — dynamic articulation: LANDED 2026-07-10** (`48a3da92dc`): metric
accents (beat one 1.06×, common-time mid-bar 1.03× — the meter FELT, the arch
still authoritative), the climax lean (25ms late arrival that eats its own
duration — intent, not noise, sized above the humanize jitter), and
phrase-position articulation (interior notes 0.88×→1.0× sounded length across
the phrase — legato is EARNED approaching the cadence; written-short notes
exempt). All in `performance_voices`, so audio + MIDI + piano-roll agree.
Tested: unit curves, an exact-isolation comparative test (deterministic jitter
seeds), and a statistical front-vs-back-of-phrase test on a real piece.
Artifacts: wistful v14 / dorian v8 (FluidSynth renders).

**UX Wave 1 — the workspace inversion (UI only, no engine work):**
- **Mode toggle**: **Artist** (default: prompt, style, mood, energy, length,
  Compose — nothing else visible) / **Studio** (today's full panel) /
  **Research** (spec JSON, consciousness sliders, diagnostics). **DONE
  (2026-07-13, commit `5046625c0f`)** — a single unduplicated `<form>`,
  fields tagged `data-tier="studio"`/`data-tier="research"` (untagged =
  always-visible Artist tier), 2 CSS rules do the hiding, each mode a
  strict superset of the one before. No duplicate inputs, so switching
  modes never loses a value. Compose drawer flipped to open-by-default
  so Artist's minimal set shows on load with no extra click. Live-
  verified on the Pixel 8 Pro: Artist mode shows exactly Style/Mood/
  Energy/Bars/Prompt/Compose; Research mode reveals every field in one
  screenshot. "Diagnostics" is already covered by the existing
  per-candidate Φ badge — no separate panel was invented for it.
- **Prompt-first layout** with rotating example prompts; candidates
  dominate the page after generation; controls collapse. **DONE
  (2026-07-13, commit `75672b8e04`)** — a hero prompt input is the true
  entry point of the Studio surface, above Today's Discoveries; its
  placeholder rotates through 7 example phrases every 3.5s, paused
  while focused. It's the sole source of the compose body's `prompt`
  field (the old duplicate in-drawer field was removed). The Compose
  `<details>` drawer now auto-collapses after every successful compose
  so candidates dominate the page; one click on "Compose" brings it
  back. This closes the workspace-inversion bullet in full. Verified
  live on the Pixel 8 Pro that the hero prompt renders first and its
  rotation matches the JS array; the manual collapse/expand toggle was
  confirmed via a real tap (same `<details>` DOM operation the auto-
  collapse uses) — but the auto-collapse firing after a live generate
  was not independently observed on-device this session, because the
  device happened to be under extreme *unrelated* system load
  (concurrent sessions' builds) during the verification window. See
  the commit message and `symthaea_muse_phase6_industry.md` for the
  full disclosure.
- **Musical language on the dials**: Dark•Melancholic•Reflective•Hopeful•
  Radiant instead of "-1 ↔ 1"; Sparse•Gentle•Flowing•Driving•Epic for
  energy. **DONE (2026-07-13, commit `080c5ae1d5`)** — a live word label
  above each of the Mood/Arousal/Energy sliders, updating on every drag
  via a plain `input` listener (`dialWord()` in `studio/index.html`).
  Added a matching 5-word arousal bank (Calm•Settled•Engaged•Restless•
  Electric) — not explicitly named in this bullet, designed to match the
  valence/energy banks' tone and granularity. Verified live on the actual
  Pixel 8 Pro device via adb: default sliders render "Mood Hopeful /
  Arousal Engaged / Energy Driving", exactly matching an independent
  Python dry-run of the bucketing logic. Could not get ADB touch-drag to
  reliably register on the native range-slider thumb to confirm the
  live-drag path device-side (tooling limitation, not a code gap) — the
  `input`-listener pattern itself is standard and low-risk.
- **Generated titles**: deterministic word-bank naming from seed/mode/hook
  ("The Long Return", "Ash Before Dawn") — no LLM, provenance-clean, people
  remember names, not "candidate 3". **DONE (2026-07-13, commit
  `ec1821d7f8`)** — see `describe::title_for`. `identity_card`'s title
  needed a premise to describe its features fairly, which left the
  Listen tab (never premised) showing "seed N" for every piece; `title_for`
  reads mode/valence/arousal straight off the resolved spec/intent
  instead, so it works with no premise. Shares its actual naming shape
  (`build_title`) with `identity_card`, so both use the same word banks
  and sentence form — just two ways of arriving at the two real inputs
  (color, motion) that pick from them.
- **Consciousness sliders re-labeled as a Creative State console**
  (Warmth/Focus/Wonder/Urgency) + presets (Dreaming, Storytelling, Flow…) —
  same parameters underneath, honest mental model on top. **DONE
  (2026-07-13, commit `c6ab956186`)** — Dopamine/Serotonin/Noradrenaline/
  Consciousness renamed to Wonder/Warmth/Urgency/Focus (API param names
  and [0,1] ranges unchanged), each keeping its existing plain-language
  badge as a secondary hint. Added the three named one-click presets;
  verified live on the Pixel 8 Pro — tapping "Flow" moved all four
  sliders to the exact designed values, a clean button-click interaction
  (unlike the dial-words wave's slider-drag, which ADB couldn't reliably
  simulate).

**UX Wave 2 — explainability (the true differentiator; the engine is symbolic
so every explanation is TRUE, not confabulated):**
- **"Why this piece"** panel per candidate from data we already have:
  hook cell (name + rhythm), damage plan, cadence grammar, mode, ensemble,
  form. `/api/explain/{id}`. **PARTIALLY DONE (2026-07-13, commit
  `667a9269e4`)** — see `describe::why_lines`. Shipped as an always-present
  `why: Vec<String>` field on `/api/compose`'s own response (no separate
  endpoint needed) covering grammar/development/accompaniment-rhythm-cell/
  texture-device facts, wired into both the Discover card and a new Listen-
  tab panel. Found and fixed a real, long-standing gap while building it:
  `card` (the identity title+traits) is only populated when `exploring` is
  true, which requires `n_candidates > 1` — the Listen tab's actual calls
  always use `n_candidates: 1`, so `card` has been `null` for EVERY single
  piece ever played there, the entire time this style roster has existed.
  `why` needs no premise so it's unconditional, fixing this for the
  highest-traffic surface. **Hook-cell character added (2026-07-13,
  commit `a9ddbd0f4c`)** — `why_lines` now takes a seed and describes the
  piece's own hook ("built from bold leaps" / "leans on a repeated note" /
  "moves by smooth, stepwise motion"), sharing one classification with
  `identity_card`'s trait via a new `hook_character()` helper (pinned by a
  test that the two renderings agree). NOT yet covered: the damage plan
  and per-note provenance (below) — still open.
- **Iteration verbs**: More adventurous (damage↑) · More peaceful (damage↓,
  arousal↓) · Different hook (reroll hook seed, hold the rest) · Better
  melody (N hook candidates, keep arrangement) · Different instrumentation
  (reroll ensemble, hold notes) — each a targeted spec/seed mutation the
  user never sees. THIS is "speed of iteration" made concrete.
- Note-level explanation (click a note → "hook statement / expectation hole
  / borrowed IV") needs per-note provenance tags through the damage pass —
  a real refactor, phase it behind the panel version.

**UX Wave 3 — palettes and voices:**
- **Per-voice instrument pickers** (Lead/Counter/Harmony/Bass dropdowns
  writing into the spec — the spec already supports all of it), grouped
  browser (Woodwinds/Strings/Pads) with preview, Randomize-Ensemble /
  Keep-Melody-Change-Instruments buttons. **Core pickers DONE
  (2026-07-13, commit `dc79e677c4`)** — 4 dropdowns (Melody/Harmony/
  Bass/Counter-melody) grouped into Keys & Mallets/Strings/Winds/
  Plucked/Pads & Synths (22 instruments), each defaulting to Auto.
  Writes directly into the existing "Advanced: edit style spec"
  textarea (`ensemble_pool` collapsed to the chosen triple,
  `counter_instrument` set or deleted for Auto) — no parallel state,
  same JSON round-trip `loadSpec`/`saveSpec` already use. Verified
  end-to-end via curl (patched spec → real compose → `/api/notes`
  confirmed Melody→Violin/Harmony→Cello/Bass→UprightBass/Counter→Oud
  exactly as picked) and live on the Pixel 8 Pro (native grouped
  picker renders correctly, selection updates the shown value) — see
  the commit message for what device verification could and couldn't
  confirm this session. **Randomize-Ensemble / Keep-Melody-Change-
  Instruments buttons DONE (2026-07-13, commit `2f1c866194`)** — both
  reuse the exact same load-then-patch path a manual picker change
  uses; "keep melody" pins melody to the spec's current pool value
  first if it was still on Auto, so there's something concrete to
  keep. **Audio preview per instrument DONE (2026-07-13, commit
  `8e5622b6b4`)** — a "▶" beside each voice select renders a ~3.5s
  scale on that instrument, entirely decoupled from Score/
  CompositionSpec (`midi_export::export_preview_midi` builds a
  hand-constructed MIDI directly from the `Instrument`'s GM program
  number), served via a new `GET /api/preview/{instrument}` that
  renders once and caches to `data/previews/{name}.wav` — same
  file-serving shape as the existing keeper-audio endpoint. Verified
  live against the running server (uncached render 1.4s, cached hit
  10ms, distinct instruments produce distinct files, real RIFF/WAVE
  bytes confirmed, unknown names 404) plus a new unit test. **Wave 3's
  per-voice-picker bullet is now fully done.** The other 2 Wave 3
  bullets (palettes-over-styles, wordless-voice-as-instrument) remain
  unstarted — both bigger, one needing a new data model and one
  needing engine-level singing-voice work.
- **Palettes over styles**: a palette = ensemble + articulation + room +
  humanization bundle ("Nordic Winter", "Desert Caravan" — we have Ney, Oud,
  Koto banks already). Styles remain as composition-structure presets;
  palettes own the SOUND. Clean split matching the two-crate architecture.
- **Wordless voice as an instrument** (singing Stage 1): Ah/Oo/Mm sustains
  and choir pads as just another Instrument the roles can pick — humanity
  without the NLP cliff. Lyrics (Stage 2: theme/perspective/vocabulary →
  proposals that fit the melody) and vocal direction (Stage 3: whisper/
  folk/ancient/ethereal) staged AFTER articulation, or the singer inherits
  the engine's stiffness.

**Long-term shape** (adopted as north star, not scaffolding to build now):
four connected spaces — Composition / Palette / Performance / Studio — "a
complete musical instrument where composition, orchestration, performance,
and production are all editable, explainable, and owned by the artist."
Build each space only when its engine layer is real (Performance space waits
on the articulation work).

### Explicitly not doing

- End-to-end neural audio generation (compute-impossible locally, off-brand vs the
  no-scraped-data stance, forfeits controllability/adaptivity/provenance moat).
- Full batch/streaming pipeline unification before the audible wins (real debt, but
  multi-day and risky — Phase 4 item, do after Tier 1/2).

Gate: the awaiting blind A/B listening artifacts (`audio_output/ab_melody/`,
`theory_ab/`, mastered `instrument_ensembles/`) should keep gating which improvements
get promoted to default paths.

## The Persistence Plan (2026-07-11) — from the passacaglia listening review

The review's verdict on the passacaglia artifacts: "the ground does not
feel like mere repetition. It feels like a persistent identity that the
upper voices keep having to reinterpret... restoration feels like
recognition, and completion feels like consequence." Seed 8 proved the
counterpoint: "ground suitability is a distinct compositional property —
a good melodic hook is not automatically a good ostinato."

**New primitive named by the review: PERSISTENCE** (distinct from memory:
memory asks "what returns?"; persistence asks "what refuses to
disappear?"). The contaminated-control finding adds a second long-range
strategy: **LINEAGE** (familial identity — Ground → Ground′ → Ground″,
kinship instead of literal repetition, "remarkably close to how human
composers work"). Keep them separate primitives.

**Next, in order (review-endorsed):**

1. **Ground-worthiness judgment** — score candidate subjects BEFORE
   granting them a ground: recurrence recognizability (hook identity
   predicates + trigram self-similarity), harmonic affordance (how few
   fitter-bends the uppers need over it — measurable with the existing
   species fitter), rhythmic durability, fragmentation survival (does
   the one-tone-per-bar skeleton retain the contour signature),
   transformation legibility (is the one-note alteration detectable).
   All five computable symbolically today. Studio composes N candidate
   subjects, ranks, grants the ground to the best — and logs the five
   scores into keeper entries so ♥ data eventually LEARNS the weighting.
   This answers "which subjects deserve to become grounds" empirically.
2. **Erosion** — the inverse arc: the ground loses confidence cycle by
   cycle (progressive, deterministic degradation) until the final cycle
   either reconstructs it or — the darker ending — never does. Persistence
   failing is as expressive as persistence winning. Small build: one
   degradation operator + an ending switch.
3. **Lineage ground** — formalize the contaminated control as a real
   mode: a chaconne-like form whose bass EVOLVES through the subject's
   transformation family while remaining kin. The experiment already
   measured it integrating ~as well as strict invariance.
4. **Cycle rhetoric** — give each cycle a bolder stated function
   (lyrical / imitative / agitated / sparse / dense / nearly-empty /
   reconciled); the current arc implies it, the uppers should declare it.

**Standing guard (review, adopted as policy):** Φ is an observable and an
experiment instrument, NEVER a fitness function. The moment Muse
optimizes for Φ it will exploit the metric instead of composing. Keeps
(♥) remain the only optimization target; Φ/local/global stay reported.

## The Music Knowledge Graph Plan (2026-07-11) — continuity beyond a composition

From the DKG review: build a **local-first, cryptographically verifiable
graph of musical identity, transformation, authorship, taste, and
provenance** — NOT "blockchain for compositions." Nodes: piece/version/
hook/motif/form/palette/attitude/memory-arc/damage-plan/render/artist/
sample/model/license/keeper-event. Edges: DERIVED_FROM / TRANSFORMS /
QUOTES / INVERTS / AUGMENTS / ECHOES / KEPT_BY / RENDERED_WITH /
TRAINED_FROM / LICENSED_UNDER...

Why Muse specifically: the symbolic engine KNOWS these facts at compose
time (exact hook, what the coda remembered, which palette rendered which
role, which sample bank sounded it) — provenance is recorded, not
narrated after the fact. And the engine is deterministic: (spec, seed,
intent) reproduces a piece bit-exactly, so a provenance record is tiny —
the recipe IS the content address.

**Sequence (adopted verbatim from the review):** local graph →
content-addressed musical objects → signed provenance records →
exportable bundles → optional federation → public DKG later. No
consensus/tokens/ledger at the start. The keeper jsonl is the embryo of
the taste subgraph and v0 can grow out of it: extend each Studio
generation record with parentage (parent piece/version), spec hash, hook
identity, memory transformations, damage plan, palette + sample/model
provenance — then the first queries ("show ancestors", "where else was
this hook used?", "which changes led to pieces I kept?") need no new
infrastructure at all.

**Ecosystem note (later, not a dependency):** when federation is actually
wanted, the rails already exist in-house — mycelix-attribution
(dependency registry, usage receipts, reciprocity) and mycelix-knowledge
(claims/graph/DKG) are built clusters. The bridge is a future wave;
nothing in v0 should wait on it.

**Main danger (adopted):** an ontology artists never use. The graph is
populated automatically by the engine; artists only ever see Branch /
Merge / Keep / Credit / Publish / Trace / Explain.

## The Melodic DNA Plan (2026-07-11) — "the melody still sounds like Muse"

The tango review isolated the remaining cross-style similarity: "I hear
tango accompaniment + Muse melody, not tango melody + tango accompaniment."
Root cause confirmed in code: `HookCell::generate(seed, meter)` is
STYLE-BLIND — one shared rhythm-skeleton pool × one shared contour pool
births every piece's name, whatever the style. The style dresses the hook;
it should own it.

**The reviewer's experiment (adopted as the acceptance test):** strip the
accompaniment and play only melodies — if you can't reliably tell which
style each came from, that's the bottleneck. Pinned in-tree at the hook
layer: per-style hook feature distributions (interval width, dotted
fraction) must separate.

**v1 (this wave):** `MelodicDna { hook_rhythms, hook_contours }` on
CompositionSpec (serde-default empty = classic shared pools — zero change
for existing styles/specs). Style presets: Tango (dotted calls, pickup
snaps; wide reaches, repeated-note insistence, descending tension),
Nocturne (long-short sighs; stepwise identities anchored by repetition or
gentle thirds), March (dotted-march cells; fourth/fifth reaches). All DNA
cells still pass the hook identity predicates — DNA changes WHICH
identities a style prefers, never whether the hook has identity.

**Later (the full decomposition the review sketched):** Style Rhythm /
Style Melody / Style Harmony / Style Counterpoint / Style Ornament as
separate axes; phrase rhetoric per style (tango: statement-interruption-
answer-stop; nocturne: breathe-continue-resolve); rhythm-first generation
(the melody born FROM the cell); melody arguing with the accompaniment.
Rule: hold new styles until the melodic layer exists (reviewer: the leap
in perceived diversity beats ten new styles).

## The Identity Explorer Plan (2026-07-11) — candidates, not renderings

The four-candidates screenshot review: seeds explore a NEIGHBORHOOD, not
continents — "four renderings of one compositional idea." Muse's
subsystems all ask "how do I preserve identity?"; almost none ask "how do
I invent a different one?" Planned subsystem (after Melodic DNA):

1. **Identity search**: generate ~dozens of symbolic hooks per intent,
   score, CLUSTER, select the N maximally-different that still match the
   intent — novelty as a first-class objective (distance from every other
   candidate), then compose.
2. **Composer personalities** (not attitudes, not styles): Minimalist /
   Architect / Storyteller / Wanderer / Dancer as reasoning biases — ask
   four composers once instead of one composer four times.
3. **Lineage BETWEEN pieces**: candidate ancestry (child/grandchild vs
   independent) as a diversification mechanism.
4. **UI language**: identities, not seeds ("The Lantern Keeper", not
   "seed 69"); "More like this" gains siblings: "Find distant cousin" /
   "Surprise me"; the Studio should read "discover music," not
   "configure a generator."

## The Discovery Plan (2026-07-11) — from control panel to place of discovery

The month's verdict, distilled from the card-browser reviews: "earlier, the
interface felt like a frontend for a composition engine. Now it feels like
the beginning of a musical studio. The next leap isn't adding another
dropdown — it's making the Studio feel like a place where you discover
ideas rather than configure them." The machinery is largely built (melodic
DNA, identity grammars, premises, meter families, worthiness, judgment,
Explorer); this plan makes it work together and VISIBLE.

**Phase 1 — Novelty Score + "why is this candidate different?"**
(the review's one engineering request). Decompose the Explorer's identity
distance into named channels — melodic (hook contour + motif head),
rhythmic (hook rhythm), premise (tempo/texture/length/mode/meter/form),
orchestration (ensemble + accompaniment), **harmonic (real chord-degree
sequence per seed — added 2026-07-12, commit `61b315437`, the channel
the original request named but the first pass left out; honestly near-
zero for Archetype-sourced styles across most seed pairs, genuinely
nonzero for Grammar-sourced ones)** — and report, per candidate, its
distance-to-nearest-batch-neighbor per channel. The card answers the
review's question directly: "differs most in: rhythm, premise." Keeper
logs the breakdown (taste data learns which KINDS of novelty get kept).
Novelty is an observable, never a fitness function — same policy as Φ.

**Phase 2 — Discovery-first Studio.** The candidate browser becomes the
hero: land on "Today's Discoveries" (a date-deterministic intent composed
on load), controls collapse into a "tweak after you pick" drawer.
"More like this" gains its siblings: "Find distant cousin" (re-explore
from the far end of the window) and "Surprise me" (new premise draw).

**Phase 3 — Development DNA.** The localized remaining convergence:
styles share the development machinery mid-piece. `DevelopmentDna`
(Sequential/Figural/Fragmenting, `apply_development_style`) already
existed and covers Tango/Nocturne/March/Blues/Celtic/Impressionism/
JazzBallad/BaroqueSuite/Cinematic — but as of 2026-07-12, **12 of 21
styles still share the plain `Classic` no-op** (Classical, Waltz,
Folk, Playful, Lullaby, ModalFolk, SacredChoral — some legitimately,
e.g. Fugue/Passacaglia bypass this pipeline entirely and Minimalism/
Ambient/ProgFolk have their own alternate development mechanism).
Triggered by a full-catalog listening review (2026-07-12, see
`memory/feedback_muse_style_ecosystem_review_jul12.md`) that named
this the #1 remaining gap ("not hook DNA, not accompaniment — the
middle. How does a style continue thinking?"). First move: new
`DevelopmentDna::Intensifying` (register climbs monotonically +
figuration accumulates + velocity itself crescendos — three axes
toward one peak, a real dramatic arc) replacing Cinematic's stale
Sequential assignment (commit `638924069`).

**DONE (commit `073abcc89`)**: assigned real identities to the 6
genuinely-undifferentiated styles — Classical→Figural, Waltz→
Sequential, Folk→Wandering (NEW variant: a genuine random walk, ±1/±2
diatonic degrees per bar, can reverse direction mid-passage — unlike
Sequential's single commitment), Playful→Fragmenting, Lullaby→Figural,
ModalFolk→Wandering. **Found and fixed a real pre-existing bug this
exposed**: `apply_development_style` could silently erase the note
marked as the piece's climax if it fell inside the rebuilt window —
latent since Tango/Nocturne/March adopted non-Classic development, but
never tripped a test until Classical (compose()'s own default)
finally exercised it. Fixed by exempting the climax's own bar from
clearing/rebuilding. 302 theory + 524 muse green.

Only 6 styles remain on `Classic`, all deliberately: Fugue/Passacaglia
(bypass this pipeline — their texture IS the counterpoint),
Minimalism/Ambient/ProgFolk (own alternate mechanisms), SacredChoral
(homophonic — "nothing to develop"). **Development DNA phase is
essentially complete** — every style that could meaningfully use it
now does. Possible future refinement (not urgent): reconsider whether
Tango/Blues/BaroqueSuite's shared Sequential assignment still fits
each individually now that Wandering/Intensifying exist as
alternatives.

**Phase 4 — Idiomatic meter cells + the formal listening test.** 5/4
currently generalizes mechanically; give Cinematic a real five-gait via
the rhythm-cell machinery. Then the randomized style-identification
harness — as diversity grows, the instrument that tells us the spread is
PERCEIVED, not just measured.

**Phase 5 — MKG v0 + site.** Keeper jsonl grows into the local music
knowledge graph (titles now give nodes human names); the site gallery
inherits identity cards.

**Liked Songs view — DONE (2026-07-12, commit `c32167631`)**, ahead of
its own phase number because it was a direct user request. New "Liked"
Studio tab reads `GET /api/keepers` and plays back the piece's ACTUAL
saved audio (`data/taste/audio/{ts}_{seed}.{wav,mid}`, written at keep
time) rather than recomposing an approximation — `Candidate` never
retained the original `MusicalIntent` (arousal/valence/energy/tonic/
bars), only spec+seed+state, so a later recompose from (style, seed)
alone couldn't reliably reproduce the exact heard piece. This is a
real precedent for MKG v0 above: keeper entries are already the
embryo, now also each pointing at real durable audio, not just
metadata.

**Audio/MIDI import — SCOPED, NOT STARTED** (2026-07-12 vision, see
`memory/feedback_muse_audio_import_vision_jul12.md` for the full
write-up). Explicit warning worth repeating here: do not build the
obvious version first (upload → clone/continue). Five ordered levels —
Listen (pure analysis: key/tempo/meter/phrase/form/motifs, no
generation), Explain (causal symbolic answers — "why does this feel
nostalgic" grounded in slow harmonic rhythm/descending thirds/etc, not
an AI summary), Learn (taste profile from imports, not model
training), Transform (rewrite in another style/grammar — generation
finally enters here), Converse (Muse RESPONDS to an imported piece
rather than continuing it). MIDI should be built before audio
(no transcription step needed). Audio pipeline must go Audio →
Transcription → Symbolic → everything else, never Audio → neural
continuation. "Explain this moment" (click a timestamp, get a named
symbolic event) is the single most-requested concrete feature inside
the vision.

## Future Style Roadmap v2 (2026-07-12) — capability over labels

A second full-catalog review, after the Tango→Ambient roadmap closed, gave a
new prioritized list — same discipline as the original Style Rule, restated
sharper: **"What new way of thinking does this style teach Muse?"** not
"what styles are popular?" The reviewer's own summary: "adding a Sonata
style is useful because it teaches exposition, development, and
recapitulation. Adding Viennese Classical after that might only be a
preset." Four tiers, ordered by how much NEW capability each teaches:

**Tier 1 — high priority (new capabilities Muse doesn't fully have yet):**
- ~~Sonata~~ ✓ **DONE (2026-07-12, commit `47cdbd5ac`)** — see `sonata.rs`.
  Tonal CONFLICT AND RESOLUTION: exposition states P (home) and S (a real
  foreign key), development compresses P through a third key,
  recapitulation restates BOTH home — S's return is provably the same
  scale-degree idea, only the key changed. New `Key::dominant()`.
- Theme & Variations — **note: `FormKind::Variations` already exists**
  (shipped 2026-07-11, in Nocturne/Lullaby's form_pool) and covers most of
  what this item asks for (transformation/inversion/augmentation/
  ornamentation/identity preservation). Not re-scoped as a fresh item;
  worth a quick audit of whether it deserves its OWN dedicated Style
  identity rather than living only inside two other styles' pools.
- ~~Opera/Art Song~~ ✓ **DONE (2026-07-13, commit `c03c12dd9b`)** — see
  `FormKind::Opera` / `opera.rs`. Theme A (Melody) and Theme B
  (CounterMelody) are genuinely unrelated material, not a transform of one
  into the other — solo statements, a bar-by-bar trading dialogue, then a
  literal interruption (B's phrase cut off mid-way, A enters early to
  resolve). Found and fixed a real `Period::parallel_in` doubling bug
  (antecedent+consequent built from the same progression silently doubles
  section length) while wiring the harmony/bass Form.
- Impressionist Orchestra — orchestral color, register painting, harmonic
  atmosphere, instrumental conversation (beyond Impressionism's existing
  planing device).
- ~~Renaissance Polyphony~~ ✓ **DONE (2026-07-12, commit `601d83dfd1`)** — see
  `renaissance.rs`. Three independent monophonic voices (soprano/alto/bass),
  deliberately NO `VoiceRole::Harmony` — real equal-voice polyphony, not
  melody-plus-accompaniment. Differs from Fugue at its foundation, not just
  its surface: imitation at the octave (not a fifth-transposed answer), two
  points of imitation with ROTATING voice entry order, a hand-built modal
  cadence (7-6 suspension + Landini under-third approach). Live-verified via
  `/api/notes`: served voices are Bass/Counter/Melody/Doubling with zero
  Harmony-role notes.

**Tier 2 — rhythm:** ~~Afro-Cuban~~ ✓ **DONE (2026-07-13, commit
`d28366c571`)** — see `Accompaniment::Montuno` in `accompaniment.rs`. Son
clave (3-2) as a genuine TWO-BAR cycle (the first pattern whose identity
isn't a single repeating bar), alternating a tresillo three-side with a
backbeat two-side; a tumbao bass interlocks with — never lands on — the
montuno's own onsets, making "rhythmic conversation" a checkable
non-overlap property. No new bypass form needed; the novelty is entirely
in the accompaniment/bass layer. ~~Bossa Nova~~ ✓ **DONE (2026-07-13,
commit `a9e768496d`)** — see `Accompaniment::BossaComp`. Syncopated onsets
(0,1.5,3.0) whose durations chain with ZERO silence (the first cell
defined by an absence of gaps, not a presence of accent) — "floating"
legato harmony instead of punctuated stabs, `seventh_chords` for the jazz
color, soft dynamics throughout. ~~Irish Traditional~~ ✓ **DONE
(2026-07-13, commit `cd94285b28`)** — see `apply_roll_ornaments` /
`TextureSpec::roll_ornaments`. The engine's first ORNAMENT CHAIN: not
Celtic's single grace-note "cut" but a full five-note roll (main, upper
cut, main, lower cut, main), the reel (meter 4, Dorian — distinct from
Celtic's jig/Mixolydian), a real session trio (flute/guitar/upright bass).
~~Hindustani-inspired~~ ✓ **DONE (2026-07-13, commit `f45f18da95`)** — see
`apply_full_drone` / `TextureSpec::full_drone`. The engine's first FULL
drone: not just Celtic's bass pedal under a moving harmony, but Harmony
too, replaced with a static tonic-fifth-octave pad tied into one
continuous sustain — no chord progression exists anywhere, so "tension
without modulation" is a structural guarantee, not a mood. Live-verified:
an entire ~5-minute composed piece's Harmony voice is 3 notes total, each
spanning the whole duration.

**Tier 3 — long-form architecture (forms that stress memory):**
Passacaglia (already done), Chaconne (variation over a harmonic
progression instead of a strict repeating bass — different enough from
Passacaglia to justify itself), Tone Poem (narrative composition).

**Tier 4 — modern, deliberately NOT rushed:** Progressive Rock (already
done — changing meter, thematic return, long arcs), Post-rock (texture,
patience, gradual growth), Neo-classical piano ("probably popular but
doesn't teach much new — I'd wait"), Lo-fi ("mostly production, not
theory").

**Deliberately skipped for now:** Dubstep/Trap/Hyperpop/Hardstyle/Brostep
— "the interesting work there is synthesis/sound design/production rather
than symbolic composition."

**Named as missing entirely — improvisational styles:** Jazz Combo, NOT
jazz harmony (already have that via JazzBallad) but real improvisation:
motif trading, call-and-response, spontaneous variation, accompaniment
REACTING. Flagged as "a huge leap," not yet scoped as a concrete
mechanism.

**The reviewer's "secretly most wanted" style: ~~Flamenco~~ ✓ DONE
(2026-07-13, commit `9df77060eb`)** — see `Accompaniment::CompasGait` in
`accompaniment.rs`. The 12-beat compás (3+3+2+2+2, accents only on counts
3/6/8/10/12, every other beat silent — the first cell defined as much by
its rests as its hits), Phrygian mode (the engine's first Phrygian style),
Andalusian cadence (progression `[4,3,2,1]`, a descending stepwise
tetrachord rather than the usual fifths motion), the roster's highest
ornament/appoggiatura rates, oud/guitar/cello ensemble. Full call-and-
response turn-taking between melody and accompaniment (as opposed to the
simultaneous non-collision Montuno/AfroCuban already covers) was
deliberately left for a future wave — this ship covers rhythm, mode,
harmony, and melody, not the full "conversation" ambition.

**Beyond style — musical dialects** (explicitly a LATER idea, once styles
feel sufficient): the same style rendered through different expressive
lineages (e.g. Nocturne → French → Russian → Late Romantic → Modern) —
not presets, different ways of expressing the same musical idea. Not
scoped; flagged for whenever the style list itself feels complete.

**Full "top ten" ranking as given:** 1. Sonata (done), 2. Theme &
Variations (mostly already exists), 3. Renaissance Polyphony (done), 4.
Afro-Cuban (done), 5. Flamenco (done), 6. Bossa Nova (done), 7. Opera/Art
Song (done), 8. Irish Traditional (done), 9. Hindustani-inspired (done),
10. Tone Poem.

**All ten items are now done or effectively covered** (Theme & Variations
already existed as `FormKind::Variations`). Remaining open items outside
the top ten: Tone Poem (#10, Tier 3, narrative composition — the list's
last un-shipped entry), Impressionist Orchestra (Tier 1, orchestral
color/register painting/instrumental conversation), Chaconne (Tier 3),
Post-rock/Neo-classical piano/Lo-fi (Tier 4, deliberately not rushed),
Jazz Combo improvisation and Flamenco-adjacent "musical dialects" (both
explicitly flagged as later ideas, not yet scoped as concrete mechanisms).

Standing question for every future addition, restated per the review:
**"If I build this style, what new musical habit will every other style
inherit?"**

## The Listening Test Battery (2026-07-12) — beyond Test A

Test A (style recognition, `listening_test.rs`) is live and produced a real
result: the reviewer scored the March clips wrong across the board (piano,
no drums, and — the deeper finding — March had no `PhraseRhetoric` of its
own and hook contours that passed the identity predicate only via a cheap
immediate repeat, not a genuinely march-shaped leap). Fixed: March now has
real bugle-call hook contours (triadic reaches to the octave, arpeggio
call-and-drop, upper call-and-answer, full descent — each predicate-valid
for a musical reason) and its own `PhraseRhetoric::Martial` (statement —
statement — STRIKE: no interruption, no silence — a march never stops —
but every cadence lands clipped and hard-accented, with its own pickup
accent on the note leading in). The fix deliberately did NOT lean on
drums — the test strips them on purpose, and re-adding rhythmic cues to
compensate would defeat its point. Test set regenerates once this lands;
worth re-running just the March-vs-others portion.

The reviewer's proposed extension — a battery, not a single test, each
probing a different claim the engine makes:

- **Test A — Style recognition** (built). "Which style is this?" Verifies
  a style's melodic thought is perceptible without its texture.
- **Test B — Identity-grammar recognition.** Play pairs from the same
  grammar (Memory/Persistence/Lineage/Erosion) vs. unrelated pairs. Ask:
  "are these the same idea, remembered/varied/eroding — or unrelated?"
  Verifies the identity-grammar layer is heard as continuity, not just
  computed as one.
- **Test C — Emotional-intention agreement.** Not a correctness test —
  there's no ground truth to score against. Play a clip, ask the listener
  to name the intended valence/arousal in their own words, compare to the
  `MusicalIntent` that generated it. Agreement (or a legible, explicable
  disagreement) is the signal.
- **Test D — Family resemblance / lineage.** Play a piece and its
  lineage-descendant (`crate::passacaglia` kinship chains) against a
  piece and an unrelated one. Ask: "which pair sounds related?" Verifies
  kinship is perceptible, following up on the lineage-experiment's
  quantified Φ result (figuration-kinship 0.150) with a perceptual one.
- **Test E — Continuation expectation.** Cut a phrase mid-cadence, offer
  two completions (the real one; a plausible-but-wrong one — e.g. a
  deceptive close where the piece actually resolves, or vice versa). Ask
  which continuation feels right. Verifies the cadence/expectation
  machinery (deceptive first close, evaded return close, pivot
  modulation) is shaping listener expectation, not just decorating notes
  that were already going to be there.

**Confidence scoring** (done, 2026-07-12): `listening_test score` now
accepts an optional `@N` confidence (0-100) per guess and reports average
confidence overall, when-right, and when-wrong, plus a per-style
breakdown and a confusion table (which truth gets heard as which guess —
the most direct signal for "which layer to work on next," sharper than a
bare accuracy number). A well-calibrated listener shows a real gap
between right-confidence and wrong-confidence; a listener who's *guessing*
confidently shows none.

**MKG v0 is deliberately deferred, not cancelled.** The reviewer's
argument: a knowledge graph encoding "commonly confused with…" or "this
identity-grammar chain reads as continuity to N/M listeners" needs
listening-test DATA to encode — building it first would mean guessing at
the very relationships it exists to capture. Sequence: accumulate Test
A-E results across enough listeners/sessions, THEN build MKG v0 with real
edges instead of placeholder ones. This reorders Discovery Phase 5 without
removing it.

**Research Mode (future, not started).** The listening test proved a
pattern worth productizing: ship a feature with a stated, falsifiable
hypothesis, then auto-generate a blinded experiment that tests it — not
after the fact, as part of landing the feature. Concretely: a new feature
lands with a one-line hypothesis (e.g. "Martial rhetoric makes March
melodically distinguishable without drums"), `listening_test` (or a
generalized harness sharing its shuffle/answer-key/scoring machinery)
auto-generates a before/after or A/B clip set targeting exactly that
claim, and the result — confirmed, falsified, or inconclusive — gets
recorded next to the feature the way the species-counterpoint and lineage
falsifications already are in code comments. This is the same
"observable, not fitness function" discipline the Φ/novelty/worthiness
policy already enforces, extended to melodic identity itself.
