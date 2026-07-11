# symthaea-music-theory: Generative Music Theory

**Status**: Layers 0–4 COMPLETE 2026-07-07 (135 tests, clippy clean, compiles
in seconds). The crate composes: `compose(MusicalIntent) -> Score`. muse-side
realizer (`symthaea-muse/src/theory_realize.rs`) is DONE. A four-part
improvement pass (voice leading, sentence-phrase type, dominant-7th harmony
color, ternary ABA form) is DONE. A [`live::LiveComposer`] for real-time
incremental composition (one phrase per call, reacting to a live-updated
intent — the shape a game needs) is DONE. A genre/style system
([`style::Style`]: Classical/Waltz/Folk/Cinematic/Playful) is DONE. A first
counterpoint rule (parallel fifths/octaves avoidance, `counterpoint.rs`) is
DONE. [`form::Form::rondo`] (ABACA, a form beyond ternary) is DONE.
Remaining/planned: true chromatic secondary dominants (needs a real
architecture change — see the note below), more counterpoint rules (voice
crossing between bass/upper, not just within upper voices), wiring `rondo`
into `compose()`'s own form-selection heuristic (currently only exposed as
a public building block, like the Progression archetypes).

Design doc — the plan for turning Symthaea's music from a stochastic
note-emitter into a composer.

## Progress

- **Layer 0** primitives (pitch/scale/chord) — DONE `b390c93605`
- **Layer 2** motif + development (transpose/invert/retrograde/augment/sequence,
  each with a proven property) — DONE `c8a0ff7174`
- **Layer 1** functional harmony + cadences (Key, diatonic chords, function,
  progression grammar, 4 cadence types) — DONE `87efbf6cb1`
- **Phrase + Period** (motif developed over harmony; antecedent question →
  consequent answer) — DONE `5b3cf8ce13`
- **Layers 3–4** Score + Composer (intent → structural choices → 3-voice score
  with climax/cadential emphasis) — DONE `f930af8a8a`
- **muse realizer** — DONE `5ddfdf51fe` (`theory_realize.rs`; expressive timing
  driven by `Emphasis` annotations, not random jitter). A/B render exists
  (`examples/theory_vs_engine.rs`); blind-listen verdict still pending.
- **Four-part improvement pass** (2026-07-08) — DONE: voice leading
  (`voicing.rs`, `aed82d2597`), sentence phrase type (`Phrase::build_sentence`,
  `Period::parallel_sentence`), dominant-7th harmony color (seventh chords on
  dominant-function measures), ternary ABA form (`form.rs`, `6ff7a65e0f`).
  See `memory/symthaea_music_theory_crate.md` for the honest gotcha found
  during this pass (the sentence continuation's "diminution" framing
  overclaimed — fixed to describe only the guaranteed bar-fill invariant).
- **`live::LiveComposer`** (2026-07-08) — DONE: incremental real-time
  composition. `realize_melody`/`realize_harmony`/`realize_bass` refactored to
  accept caller-owned state (`start_beat`, `prev_upper`, `prev_bass`) instead
  of initializing fresh each call, so both `compose()` and `LiveComposer`
  share the same realization logic with no duplication. Explicitly does NOT
  attempt a global climax or scripted large-scale form per call — see the
  module doc in `src/live.rs` for why that's an honest limit, not a gap.
- **Genre/style system** (2026-07-08) — DONE: `style::Style` enum
  (Classical/Waltz/Folk/Cinematic/Playful) biases meter, tempo range, motif
  shape bank, and progression archetype — deliberately NOT a new harmonic
  system (Key/Tonality/functional harmony untouched). `Style::Classical` is
  a verified-byte-identical passthrough to the pre-existing defaults, so
  `compose_styled(intent, Style::Classical) == compose(intent)` and adding
  style broke no existing caller. `compose_styled`/`LiveComposer::new_styled`
  expose the axis; `compose()`/`LiveComposer::new()` are unchanged
  one-line delegations. Waltz is the only 3/4 style — the realize_*
  pipeline was already meter-generic, so this needed no architecture
  change, just new bank content. 13 new tests (118 total).
- **Counterpoint: parallel fifths/octaves avoidance** (2026-07-08) — DONE:
  `counterpoint.rs` (`has_parallel_perfect`, `parallel_perfect_violations`),
  the oldest and most universally cited voice-leading rule — similar motion
  between two voices into another perfect fifth/octave. Wired as a soft
  cost-function penalty in `voicing::lead_upper`'s existing permutation
  search (500 per violation, vs. 1000 for the pre-existing crossing
  penalty). Verified with a constructed case (Cmaj7 -> Dm7) where a
  parallel-free alternative genuinely exists and gets chosen over the
  raw-cost-cheaper parallel one. **Honest scope limit**: only checks pairs
  WITHIN the upper voices `lead_upper` returns — `lead_bass` is computed
  independently with no shared cost function, so bass-vs-upper parallels
  (also classically important) aren't checked yet. 11 new tests (129 total).
- **`Key::parallel()`** (2026-07-08) — DONE: the mode-mixture counterpart to
  the existing `Key::relative()` (same tonic, opposite tonality — C major
  <-> C minor, vs. relative's same-pitch-classes-different-tonic). Added
  specifically to support `Form::rondo`'s second contrasting section.
- **`Form::rondo`** (2026-07-08) — DONE: ABACA form (the theme returns
  TWICE, each time framing a different episode) — B = relative key +
  inverted motif (same as ternary's B), C = parallel key + retrograded
  motif (a genuinely different modulation AND transformation, so the two
  episodes are distinguishable). Purely composes existing primitives
  (`Key::parallel`, `Motif::retrograde`) — no new machinery needed. 6 new
  tests (135 total). **Not yet wired into `compose()`'s own form choice**
  (currently ternary-only) — `rondo` is exposed as a public building block,
  the same status the individual `Progression` archetypes have.
- **Deferred, explicitly not attempted**: true chromatic secondary
  dominants. `Key::secondary_dominant()` already computes the correct CHORD
  (e.g. V7/V in C major = D7), but wiring it into a real progression would
  need the melody-fitting logic to handle a chromatically-altered scale
  degree — `Phrase`/`Motif` currently work entirely in abstract DIATONIC
  scale-degree space by design. This is a real architecture lever, not a
  quick add; flagged rather than rushed, same as prior passes.

## Why this crate exists

Tristan's verdict on the current output (after all the DSP/training work):
*"still sounds like a child with no understanding, or a robot without feeling
and a soul — it feels forced."*

That is a **correct** diagnosis, and no amount of synthesis polish fixes it.
A source-level audit of `symthaea-muse` found the cause: **every musical
dimension is an independent per-note random draw**, lightly shaped by phrase
position.

- Pitch (`taste_melody::next_freq`): a constrained random walk with an
  up/down direction bit that flips every 4–8 notes.
- Rhythm (`suggest_duration`): `hash % 100` → pick from {whole, half, quarter,
  eighth}. Independent every note.
- Dynamics (`suggest_velocity`): one fixed crescendo→diminuendo curve applied
  identically to every phrase, plus random jitter.
- Rests: another independent random probability.

There is no motif, no memory between phrases, no question-and-answer, no goal,
no relationship between melody and harmony. It is a stochastic note emitter,
not a composer. A child at a piano also produces locally-plausible but
globally-aimless sequences — hence the perception.

**The principle it violates** (Meyer, *Emotion and Meaning in Music*; Huron,
*Sweet Anticipation*; Narmour, *Implication-Realization*): musical feeling is
the play of **expectation over structure and time** — a pattern is established,
then confirmed, delayed, or broken, and *that play* is the affect. A random
walk establishes no pattern, so there is nothing to fulfil or subvert. No
structure → nothing to feel. "Soul" = structure + expressive *deviation* from
it. The current system has neither.

### The deepest technical reason: the wrong atom

muse's atom is `Note { frequency: f32, .. }` — a raw Hz value. You cannot do
real voice-leading, functional harmony, or non-chord-tone resolution on
frequencies. Those operations are defined on **symbolic** music: pitch classes,
scale degrees, chord functions, Roman numerals. The composition logic is
therefore trapped in a frequency-domain random walk because the representation
can't express anything better.

## Why a separate crate (not modules in muse)

1. **Ground truth → verifiable.** Music theory has textbook-correct answers: a
   major triad IS [0,4,7]; a V–I IS an authentic cadence; the inversion of an
   inversion IS the identity. Every rule is a unit test against a known fact.
   This is the exact opposite of the unverifiable scaffolds this project has
   spent days cleaning up — a theory crate is *anti-scaffold by construction*.
2. **Separation of "what to play" from "how it sounds."** The theory crate
   produces a symbolic `Score`; muse *realizes* it (tuning, voicing, timbre,
   expression, reverb — muse's genuine strengths). Their absence of separation
   is *why* the music is stuck.
3. **Dependency-light → compiles in seconds.** No wgpu, no symthaea-core. Under
   the current build contention (muse takes 3–90 min), a lean crate is
   iterable. This is a real practical multiplier.
4. **Reusable ground truth.** Praxis (music pedagogy), mycelix-music,
   analysis/critique, MIDI export — all want symbolic theory. There is none in
   the monorepo today (verified: no `PitchClass`/`Chord`/`RomanNumeral`/
   `Cadence`/`Motif` anywhere).
5. **Unifies muse's two pipelines.** The review flagged batch `compose()` and
   `StreamingSynth` as divergent duplicates. Both become *renderers of a
   theory-produced Score* — one composition brain, two realizers.

## Architecture (bottom → top)

The atom is **symbolic**. Nothing below Layer 4 knows what a Hz is.

### Layer 0 — Primitives  [SCAFFOLDED]
`PitchClass` (0–11), `Pitch` (class + octave, MIDI conversions), `Interval`
(semitones + quality). `Mode`/`Scale` (Ionian…Locrian, harmonic/melodic minor,
pentatonics), degree→pitch-class. `ChordQuality`, `Chord` (root + quality →
pitch classes), inversions. Every fact unit-tested (maj triad=[0,4,7],
V7/C=[G,B,D,F], …).

### Layer 1 — Functional harmony
`RomanNumeral` / `HarmonicFunction` (Tonic / Predominant / Dominant), diatonic
chords of a key, secondary dominants. `Progression` — Roman numerals in a key,
with a small archetype library (I–IV–V–I, ii–V–I, I–vi–IV–V, 12-bar blues,
Pachelbel, lament bass, Andalusian) AND a **functional-grammar generator**
(T→PD→D→T weighted transitions — a real grammar, not a fixed loop). `Cadence`
(Authentic / Half / Plagal / Deceptive) — detect + schedule. A per-chord
tension value → a progression tension curve.

### Layer 2 — Melody as theory  (the core of the fix)
- `Motif` — a symbolic germ: (scale-degree|interval, rhythmic value)+ with a
  contour. Ground-truth transformations, each a pure function with a testable
  property: `transpose`, `invert` (inv∘inv = id), `retrograde` (retro∘retro =
  id), `augment`/`diminish` (rhythm scaling), `sequence`, `fragment`,
  `ornament`.
- `Phrase` — develops a motif over a chord span: chord tones on strong beats,
  non-chord tones (passing, neighbor, suspension, appoggiatura, escape) on weak
  beats that **resolve correctly** (verifiable: a passing tone connects two
  chord tones by step; a suspension is prepared → sustained over the change →
  resolves down by step).
- `Period` — antecedent (ends on a half cadence, unresolved "question") +
  consequent (resolves to tonic, "answer"). This tension-and-release pair is
  the skeleton of tonal feeling.

### Layer 3 — Form & narrative
`Section`/`Form` (AABA, ABA, verse–chorus, theme-and-variations,
through-composed). Each section: a key, a tension target, a motif relationship
(same / transformed / contrasting). `Climax` — one phrase per section is the
goal; melodic contour arcs toward and away from it. Directionality.

### Layer 4 — Consciousness mapping  (the thesis bridge)
`compose(state) -> Score` maps the cognitive trajectory to **structural**
choices, not local nudges:
- motif character: arousal → rhythmic density; valence → mode (major/minor,
  or brighter/darker modes); consciousness → phrase complexity + harmonic
  richness.
- **tension-arc shape**: a *felt trajectory over time* (build → peak →
  release), not a static snapshot — this is where "soul as a changing inner
  state" lives, and it's what the current snapshot-in model cannot express.
- cadence choice: deceptive under uncertainty/surprise, authentic under
  resolution.
- **Eight Harmonies → harmonic color + modal choice**, meaningfully — finally
  closing the "harmonies diluted in the flagship path" gap the review found.

### Output: `Score`
Voiced symbolic notes (melody + harmony voices) with symbolic pitch, metric
onset/duration, dynamic, and **structural annotations** (which motif, phrase
role, cadence, is-climax). Pure data. No Hz, no audio.

## muse's new role: realize the Score

muse consumes `Score` → audio, its real strength:
- tuning (JI / 12-TET / maqam per consciousness — existing `pitch.rs`),
- chord voicing + register,
- **expressive timing tied to the score's annotations** (cadential ritardando,
  agogic accent on the climax, breath before a new idea) — driven by
  *structure*, not random jitter,
- dynamics tied to the tension arc,
- its (good) additive/FM synthesis, timbre, and reverb.

## Execution order (highest leverage first)

1. **Layer 0** — primitives + ground-truth tests. [DONE — scaffold]
2. **Layer 2 motif + transformations** — the single biggest lever on
   "understanding". Prove development turns aimless walks into recognizable
   ideas.
3. **Layer 1 functional harmony + cadences**, then **Layer 2 phrase/period** —
   question-and-answer over real harmony with NCT resolution.
4. **Score type + a muse realizer** behind a `theory` feature flag, A/B'd
   against the current engine (reuse `ab_melody_weights.rs` shape). Human ears
   decide.
5. **Layer 4 consciousness mapping** + **Layer 3 form** — the felt trajectory
   and long-range shape.
6. Expression realization in muse; unify batch/streaming as Score renderers.

## Honesty commitments (per project ethos)

- No "soul" claims. We claim *structure*: recognizable motifs, real cadences,
  correct NCT resolution — all verifiable. Whether it moves a listener is
  Tristan's ears, tested via blind A/B, recorded as evidence.
- Every theory rule ships a test against a textbook fact. If we can't state the
  ground-truth property, we don't ship the rule.
- The current engine stays the default until an A/B says the theory path wins.
