// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CompositionSpec: the composer's aesthetic space as USER-OWNED data.
//!
//! Everything that used to be a hard-coded architectural decision — motif
//! banks, progression source, accompaniment textures, large-scale forms,
//! ensembles, meter, tempo, texture/ornament policies — lives in one
//! declarative, serializable structure. The five built-in [`Style`]s are
//! nothing more than five preset values of this type ([`Style::spec`]);
//! a user can author their own spec (JSON in Muse Studio, or Rust) and get
//! the same engine with THEIR aesthetics.
//!
//! The division of power is deliberate:
//! - the SPEC owns every *choice* (which motifs, which textures, how the
//!   piece is allowed to breathe);
//! - the ENGINE owns every *invariant* (voice leading, cadence grammar,
//!   collision avoidance, the superset-only substitution rules). A spec can
//!   reshape the aesthetic space completely, but it cannot make the engine
//!   write a wrong note.
//!
//! [`CompositionSpec::validate`] enforces the boundary: malformed specs are
//! rejected with human-readable reasons, never silently "fixed".
//!
//! [`Style`]: crate::style::Style

use crate::accompaniment::Accompaniment;
use crate::harmony::Progression;
use crate::motif::Motif;
use crate::rhythm::Duration;
use serde::{Deserialize, Serialize};

/// One motif-template note: (scale degree, duration numerator, denominator).
/// Rational beats, exactly as [`Duration::new`] takes them.
pub type SpecNote = (i32, i64, i64);

/// Where the chord progression comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressionSpec {
    /// The functional-harmony grammar generator ([`Progression::generate`]),
    /// with the default (Classical) vocabulary.
    Grammar,
    /// The same grammar generator with a style's OWN harmonic vocabulary
    /// ([`Progression::generate_with`]).
    ///
    /// `Grammar` alone is style-agnostic: measured 2026-07-30, every style using
    /// it produced bit-identical progressions at a matched seed (96/96 across
    /// Classical/BaroqueSuite/March/Nocturne x 32 seeds). Converting a style to
    /// `Grammar` therefore bought within-style variety at the cost of making it
    /// indistinguishable from every other Grammar style. This variant is the fix:
    /// the T-PD-D-T grammar stays universal, the vocabulary becomes the style's.
    GrammarWithPalette(crate::harmony::HarmonicPalette),
    /// A fixed archetype of scale degrees, cycled to the requested length.
    Archetype(Vec<i32>),
    /// A seed-selected pool of archetypes, each cycled to the requested
    /// length. Unlike a single [`ProgressionSpec::Archetype`], this varies
    /// with `seed` — added to fix a real diversity-census finding: several
    /// styles that each hardcoded ONE archetype had zero within-style
    /// harmonic variety and, when those archetypes happened to share the
    /// same underlying chord-degree set (e.g. rotations of {I,IV,V,vi}),
    /// bled into each other as near-duplicates. Pool members should be
    /// chosen so the WHOLE pool stays harmonically distinct from other
    /// styles' pools, not just internally varied.
    ArchetypePool(Vec<Vec<i32>>),
}

/// Large-scale forms the seed may choose between.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormKind {
    /// A–B–A′: one departure, one return.
    Ternary,
    /// A–B–A–C–A: two contrasting episodes, each framed by the theme.
    Rondo,
    /// Theme–minore–figuration–theme: every section keeps the theme's
    /// harmonic skeleton (unlike ternary/rondo's fresh episode
    /// progressions); only the surface varies. Variation is structured
    /// remembering — see [`crate::form::Form::variations`].
    Variations,
    /// A three-voice fughetta: exposition (answer at the fifth, derived
    /// countersubject), sequential episodes, inverted middle entry,
    /// stretto, augmented final entry. Bypasses the period/accompaniment
    /// pipeline entirely — a fugue's texture IS its counterpoint. See
    /// [`crate::fugue`].
    Fugue,
    /// Variations over a remembering ground bass: seven cycles (stated,
    /// walked, filled, ALTERED, restored-as-peak, fragmented, completed).
    /// Bypasses the period pipeline like the fugue — the ground is the
    /// form. See [`crate::passacaglia`].
    Passacaglia,
    /// Persistence FAILING: the ground loses one tone per cycle while the
    /// melody persists above; the ending (recovery / acceptance / elegy,
    /// attitude-mapped) decides what persistence cost. See
    /// [`crate::passacaglia::realize_erosion`].
    Erosion,
    /// Identity evolving through kinship: each cycle's ground is ONE
    /// legible transformation step from its parent — "I become my
    /// descendants" instead of "I remain." See
    /// [`crate::passacaglia::realize_lineage`].
    Lineage,
    /// LONG FORM via a genuine mid-piece METER CHANGE: four sections
    /// realized separately, each in its own time signature (4 → 7 → 5 →
    /// 4) and the middle two in contrasting keys, spliced onto one
    /// continuous timeline with voice-leading carried across every
    /// change. Bypasses the period pipeline like the fugue/passacaglia
    /// family — no single `meter_beats` scalar can represent more than
    /// one meter. See [`crate::prog_suite`].
    ProgSuite,
    /// Sonata form: exposition (first subject home, second subject in a
    /// real foreign key), development (a third key, fragmented), and
    /// recapitulation (both subjects home — the second subject's return
    /// is the exact same idea, now resolved). The first form built
    /// around TONAL CONFLICT AND RESOLUTION rather than contrast alone.
    /// Bypasses the period pipeline like the fugue/passacaglia/prog-
    /// suite family — the key relationships across sections can't be
    /// expressed by the plain ternary/rondo B-key machinery. See
    /// [`crate::sonata`].
    Sonata,
    /// Renaissance polyphony: three EQUAL voices, no subject-entry
    /// hierarchy (unlike fugue's subject/answer/countersubject). Two
    /// points of imitation entering at the OCTAVE rather than the fifth,
    /// voice order rotating between them, closing on a real modal
    /// suspension-and-under-third cadence. Bypasses the period pipeline
    /// — the whole texture IS three independent species-fitted lines,
    /// not a melody-plus-accompaniment realization. See
    /// [`crate::renaissance`].
    Renaissance,
    /// Opera / Art Song: two genuinely INDEPENDENT melodic identities
    /// (Theme A in Melody, Theme B in CounterMelody — unrelated contour
    /// and register, not a transform of one into the other) in structured
    /// conversation — solo statements, a bar-by-bar dialogue trading the
    /// active voice, and a literal interruption (one theme's phrase is cut
    /// off mid-way and the other enters early) resolving to a cadence.
    /// Bypasses the period pipeline — no single continuous melody voice
    /// can carry a dialogue between two characters. See [`crate::opera`].
    Opera,
}

impl FormKind {
    /// True for the three forms realized by the shared period pipeline
    /// (`crate::composer::compose_with_spec_and_form`), which consumes
    /// `CompositionSpec::progression` (and therefore `MusicalIntent::bars`)
    /// to build its harmonic skeleton. False for every dedicated-engine
    /// form (`Fugue`/`Passacaglia`/`Erosion`/`Lineage`/`ProgSuite`/`Sonata`/
    /// `Renaissance`/`Opera`), which bypasses that pipeline entirely and
    /// builds its own harmony/length internally -- for those,
    /// `CompositionSpec.progression` and `MusicalIntent.bars` have no
    /// effect on the composed output.
    ///
    /// Exists so callers that compare or report on a spec's `progression`
    /// field (e.g. an A/B harness) can tell, without duplicating
    /// `compose_with_spec_and_form`'s own dispatch, when that comparison
    /// is actually meaningful for a given seed -- see
    /// `baroque_campaign.rs`'s `progression_intervention` field.
    pub fn uses_progression_pipeline(self) -> bool {
        matches!(
            self,
            FormKind::Ternary | FormKind::Rondo | FormKind::Variations
        )
    }
}

/// Drum policy, interpreted by the renderer (the symbolic layer only names
/// the intent; kick/snare/hat synthesis is muse's business).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrumPolicy {
    /// No kit (chamber music).
    None,
    /// Kick on the barline + soft hats on beats.
    LightPulse,
    /// A sparse kick on each barline only.
    BarPulse,
    /// Full backbeat: kick 1 & 3, snare 2 & 4, eighth-note hats.
    Backbeat,
}

/// A compositional ATTITUDE — an emotional posture deeper than a style
/// preset. The listening review that demanded it: "if I listened to ten
/// Muse pieces in a row... they all inhabit a similar emotional world:
/// thoughtful, bittersweet, careful, restrained... teach Muse to inhabit
/// different emotional behaviors. Those are deeper than style presets.
/// They're compositional attitudes." Each attitude bundles tempo/texture
/// dials with a SIGNATURE DEVICE the default temperament never uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Attitude {
    /// Little rhythmic questions, an unresolved ending: the final melody
    /// tone resolves UP to degree 2 — question intonation — and the
    /// piece leaves without answering.
    Curiosity,
    /// Notated syncopation (interior downbeats arrive an eighth EARLY,
    /// held through where they belonged), an assertive bass. Insistence,
    /// not reflection.
    Defiance,
    /// Lighter, detached articulation and a quicker pulse — lift.
    Joy,
    /// Suspension chains at chord changes (the classical grammar of
    /// grief: a tone held over from the old harmony, dissonant against
    /// the new, resolving down by step), sparse block accompaniment, a
    /// slower pulse.
    Grief,
}

/// How a style CONVERSES after its opening statement — the phrase-level
/// rhetoric the second melodic-DNA review demanded ("the opening sentence
/// differs; the conversation afterward still sounds recognizably Muse").
/// Applied as a late pass over cadences and the approaches to them; the
/// default is the classic shared behavior, byte-identical (pinned by
/// test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PhraseRhetoric {
    /// The existing shared phrase behavior, untouched.
    #[default]
    Classic,
    /// Statement — INTERRUPTION — answer — sharp stop (the tango's
    /// rhetoric): the melody stops dead before each cadence approach (a
    /// composed silence), and the cadence itself is short and accented
    /// rather than held.
    Declamatory,
    /// Question — expansion — SUSPENSION — quiet arrival (the nocturne's
    /// rhetoric): the tone before each cadence leans into the arrival
    /// (written-out suspension timing — it overstays, delaying the
    /// resolution), and the arrival lands softly.
    Singing,
    /// Statement — statement — STRIKE (the march's rhetoric, added after
    /// the melody-only listening test localized the failure to March: no
    /// rhetoric of its own, so stripped of drums it sounded generic). A
    /// march never stops — there is no interruption, no silence, no
    /// suspension. Instead every cadence lands clipped and hard-accented,
    /// and the note immediately before it (the drum-major's pickup) gets
    /// its own accent, so the metrical insistence survives even with the
    /// kit gone.
    Martial,
    /// Statement — statement — SUSTAINED, UNHURRIED close (the hymn's
    /// rhetoric — added after a melody-only listening test found
    /// SacredChoral and Nocturne collapsing into the same perceived
    /// identity: both slow, both stepwise, and SacredChoral had no
    /// rhetoric of its own, same root cause as March's original gap). The
    /// opposite of `Singing` on both axes that make `Singing` distinctive:
    /// no suspension-lean (the cadence arrives EXACTLY on the beat, not
    /// delayed) and no softening (the note holds at a steady, slightly
    /// firm dynamic — a congregation doesn't trail off). Instead the
    /// cadential note is HELD past its written value — the choral
    /// tradition of a fermata lingering on a phrase-final tone — a
    /// mechanism no other rhetoric uses (`Declamatory`/`Martial` clip
    /// short; `Singing` shortens the arrival note to make room for the
    /// suspension).
    Chorale,
}

/// How a style DEVELOPS its material in departure sections — the last
/// shared language the listening reviews localized ("the hooks are
/// different; the development grammar is still shared"). Applied to the
/// departure's antecedent: the home theme, put through the style's own
/// way of thinking, fitted against the section's real bass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DevelopmentDna {
    /// The existing shared behavior, untouched (pinned no-op).
    #[default]
    Classic,
    /// The classical sequence schema: the theme restated down (or up)
    /// the steps, one transposition per bar — insistence through motion.
    Sequential,
    /// The theme ornaments itself: progressive figuration (division),
    /// more connecting tones each statement — development as decoration.
    Figural,
    /// The theme compresses: each bar keeps a shorter head than the
    /// last, silence growing behind it — development as distillation.
    Fragmenting,
    /// The theme climbs: register rises a diatonic step every bar
    /// (monotonically, not direction-by-seed like Sequential) while
    /// figuration accumulates like Figural AND the velocity itself
    /// crescendos bar over bar — three axes all pointing the same way,
    /// toward a peak on the section's LAST bar. Development as a real
    /// dramatic arc (rising tension, a delayed climax) rather than
    /// decoration or insistence. Cinematic's habit — reusable by any
    /// future style whose identity is a build, not a statement.
    Intensifying,
    /// The theme wanders: each bar's transposition takes a small step
    /// (±1 or ±2 diatonic degrees) from where the LAST bar ended — a
    /// genuine random walk, not a directed one. Unlike Sequential (which
    /// commits to a single direction for the whole span), Wandering can
    /// reverse mid-passage; unlike Figural/Fragmenting it never touches
    /// note count, only where the line sits. Development as drift, not
    /// argument — the Folk/ModalFolk habit.
    Wandering,
}

/// A style's MELODIC DNA: the pools its hooks are born from. Empty pools
/// (the serde default) mean the classic shared pools — so every existing
/// style and saved spec is unchanged. Non-empty pools replace the shared
/// ones for that axis; every cell is still filtered through the hook
/// identity predicates (see [`crate::hook::HookCell::generate_with`]).
/// This is the layer the tango listening review demanded: "the style
/// should own the hook... not because of instrumentation — because tango
/// melodies naturally prefer different things."
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct MelodicDna {
    /// Hook rhythm skeletons, as (num, den) beat pairs per note.
    #[serde(default)]
    pub hook_rhythms: Vec<Vec<(i64, i64)>>,
    /// Hook contour skeletons (1-based scale degrees).
    #[serde(default)]
    pub hook_contours: Vec<Vec<i32>>,
    /// Appoggiatura rate [0,1]: how often a cadential melody tone is
    /// approached by an ACCENTED upper neighbor on the beat, resolving
    /// down onto the real tone — the leaning dissonance the tango review
    /// asked for by name ("dramatic appoggiaturas"). 0 (the default) =
    /// never; the ornament is style DNA, not a global habit.
    #[serde(default)]
    pub appoggiatura_rate: f32,
    /// Ornament ("cut") rate [0,1]: how often ANY melody note (not just
    /// cadential ones) gets a quick, UNACCENTED grace note a diatonic step
    /// above, borrowing a sliver of the main note's own time. The
    /// mechanism that distinguishes it from an appoggiatura: a cut never
    /// takes the beat, is always brief, and is always the quieter of the
    /// pair — a flick, not a lean. 0 (the default) = never. The reusable
    /// habit Celtic teaches the engine: a genuinely unaccented
    /// embellishment device, where appoggiaturas only had an accented one.
    #[serde(default)]
    pub ornament_rate: f32,
    /// Blue-note rate [0,1]: how often a melody tone landing on the major
    /// third or the leading tone (scale degrees 3 and 7) is flattened by a
    /// semitone — coloring the line WITHOUT touching the harmony
    /// underneath, which stays major/dominant throughout. The mechanism
    /// that distinguishes this from every prior ornament: appoggiaturas
    /// and cuts both ADD a note; a blue note ALTERS one that's already
    /// there. 0 (the default) = never. The reusable habit Blues teaches
    /// the engine: deliberate melody/harmony scale MISMATCH as an
    /// expressive device, reusable by Jazz Ballad later.
    #[serde(default)]
    pub blue_note_rate: f32,
    /// Opt-in (2026-07-24, Motif Foundry Phase 1): when true,
    /// [`crate::hook::HookCell::generate_with`] draws its hook from
    /// [`crate::motif_foundry`]'s procedural generator -- biased toward
    /// this DNA's own measured `hook_contours`/`hook_rhythms` character
    /// via `motif_foundry::config_for_dna` -- instead of the
    /// hand-authored `hook_contours`/`hook_rhythms` pools above. Default
    /// false, and no style preset in this crate sets it yet: which
    /// styles (if any) adopt procedural generation is a listening-test
    /// decision, not made here.
    #[serde(default)]
    pub use_procedural_foundry: bool,
}

/// Texture and ornament policies — the "how is this piece allowed to
/// breathe" switches.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextureSpec {
    /// Strip the departure (B) section to melody+bass for its first half.
    pub thin_departure: bool,
    /// Add the voice-led, collision-avoided second line in return sections.
    pub counter_melody: bool,
    /// One acciaccatura into the piece's climax.
    pub climax_grace: bool,
    /// Split cadence-approach bars into triad → seventh (4/4 only).
    pub cadential_harmonic_rhythm: bool,
    /// Plagal coda length in bars (0 = end on the final cadence).
    pub coda_bars: u8,
    /// Renderer drum policy.
    pub drums: DrumPolicy,
    /// Swing: the position of the off-beat eighth within its beat.
    /// 0.5 = straight (the default), 0.6 = light shuffle, 2/3 ≈ 0.667 =
    /// triplet swing. Applied at the PERFORMANCE layer (notation stays
    /// straight, the player swings — exactly how real jazz charts work),
    /// to melody, harmony, bass, counter-melody and drums together.
    #[serde(default = "default_swing")]
    pub swing: f32,
    /// Bars of accompaniment alone before the melody enters (0 = the
    /// melody starts cold on beat one). Part of the arrangement
    /// dramaturgy wave: a listener's first impression should be an
    /// invitation, not three voices already mid-sentence.
    #[serde(default = "default_intro_bars")]
    pub intro_bars: u8,
    /// Staged entrances and exits: the bass sits out the opening phrase
    /// and arrives with the consequent; the bar before every return is
    /// thinned (bass out, harmony out for its second half) so the
    /// reprise lands as an ARRIVAL. The single biggest "sounds like a
    /// piece, not wallpaper" lever found by listening.
    #[serde(default = "default_true")]
    pub staged_entrances: bool,
    /// Held arrivals: mid-piece phrase cadences hold their final tone
    /// through the bar (the question hangs while the accompaniment
    /// answers), and the climax note absorbs the note after it so the
    /// peak is DWELT ON, not passed through. Gives the melody the
    /// silence and the one big moment continuous motion never has.
    #[serde(default = "default_true")]
    pub held_arrivals: bool,
    /// The damage pass (0.0 = pristine, 1.0 = full): after the clean
    /// composition, deliberately injure it in musically meaningful ways —
    /// the listener-review verdict on the undamaged output was "it has
    /// shape, but it lacks argument; motion without consequence." Rising
    /// thresholds add devices: ≥0.2 exposes the climax bar (accompaniment
    /// cut, the peak stands alone) and transforms the coda (the opening
    /// motif returns changed — augmented, an octave down, over a borrowed
    /// mixture chord — instead of merely receding); ≥0.35 darkens the
    /// bass entrance an octave and removes the downbeat the ear expects
    /// in the return's second bar; ≥0.5 inserts one chromatic "wait"
    /// passing tone and makes the counterline rhythmically disagree.
    #[serde(default = "default_damage")]
    pub damage: f32,
    /// Hook surgery: generate a tiny memorable cell (3-5 notes with
    /// enforced rhythmic + contour identity — see [`crate::hook`]) BEFORE
    /// the melody, and graft it as the head of the bar-motif. The phrase
    /// gets a NAME the development machinery then restates, fragments,
    /// and (in the transformed coda) remembers. The review that demanded
    /// it: "the melody still feels more like a generated line than a
    /// remembered theme... can it make a theme worth injuring?"
    #[serde(default = "default_true")]
    pub hook_cell: bool,
    /// The DECEPTIVE FIRST CLOSE: the piece's first full cadence resolves
    /// V→vi instead of V→I — closure promised, then denied — so the final
    /// authentic cadence lands as something EARNED rather than routine.
    /// The harmonic member of the expectation family (the erosion elegy's
    /// false recovery is the same grammar in form). Off by default; on for
    /// styles whose drama wants it (Classical/Nocturne/Cinematic/Tango) —
    /// a lullaby should never deny closure.
    #[serde(default)]
    pub deceptive_close: bool,
    /// A subtle new instrumental color at the final return — sparse
    /// chord-top sparkles an octave up on a contrasting timbre. The
    /// listening review: "the palette stays fairly constant... imagine if
    /// the final return introduced a subtle new color — the listener
    /// subconsciously feels 'we're somewhere different now.'"
    #[serde(default = "default_true")]
    pub return_color: bool,
    /// A sustained tonic-fifth PEDAL replaces the walking bass for the
    /// whole piece — the bagpipe/hurdy-gurdy drone under a Celtic tune.
    /// Deliberately independent of the chord above it (the harmony still
    /// moves; the drone doesn't), which is the entire point: the drone is
    /// the fixed ground a modal melody floats over, not a bass line that
    /// follows the progression. Off by default; the reusable habit Celtic
    /// teaches the engine — a genuinely static pedal texture, reusable by
    /// Nordic's open intervals and Ambient's near-stillness later.
    #[serde(default)]
    pub drone: bool,
    /// PARALLEL PLANING: in the contrast (B/C) section, the harmony stops
    /// following functional root motion and instead RIDES the melody's
    /// own contour — the same chord shape, struck fresh at every melody
    /// note, shifted by the exact interval the tune just moved. The
    /// harmony no longer resolves anywhere; it's entirely defined by
    /// where the melody happens to be standing right now. "Color over
    /// function," the Impressionist habit — reusable by any future style
    /// that wants harmony untethered from cadential logic. Off by
    /// default.
    #[serde(default)]
    pub planing: bool,
    /// SUSPENSION rate [0,1]: how often a bar-to-bar chord change gets a
    /// classical suspension instead of striking cleanly — one harmony
    /// voice is tied over into the new chord (a prepared dissonance
    /// against the new bass/harmony), then resolves DOWN BY STEP a beat
    /// or two later. Candidates are found by real voice-leading (a tone
    /// in the outgoing chord sitting a diatonic step above a tone in the
    /// incoming one), not randomly placed. The Sacred Choral habit:
    /// harmony-voice suspension, distinct from every prior ornament
    /// (appoggiatura/cut/blue-note all live in the MELODY voice; this is
    /// the first one that lives in the harmony) — reusable by any future
    /// polyphonic/chorale-adjacent style. 0 (the default) = never.
    #[serde(default)]
    pub suspension_rate: f32,
    /// ADDITIVE PROCESS: in the theme (A/ReturnA) sections, replace the
    /// standard developed melody entirely with a Glass-style additive-
    /// then-subtractive process — the piece's own hook cell repeated with
    /// one more note each time until the full cell sounds, then one fewer
    /// each time back down to one. The Minimalism habit: the FIRST pass
    /// that wholesale replaces a voice within a section rather than
    /// adding to or altering what's already there. Off by default.
    #[serde(default)]
    pub additive_process: bool,
    /// SEVENTH CHORDS: extend the existing dominant-only 7th coloring
    /// (which was always safe — a superset of the triad, never clashing
    /// with the melody's triad-based snapping) to EVERY chord, not just
    /// the cadential one. The Jazz Ballad habit: real extended harmony —
    /// maj7/min7/dom7 throughout — as an opt-in style choice, reusable by
    /// any future jazz-adjacent style (bebop, big band). Off by default.
    #[serde(default)]
    pub seventh_chords: bool,
    /// HARMONIC SEQUENCE: replace the B section's chord-by-chord
    /// progression (except each phrase's cadential tail) with a real
    /// descending-fifths circle sequence — the quintessential Baroque
    /// development device (I-IV-vii°-iii-vi-ii-V-I). The Baroque Dance
    /// Suite habit: the first pass to rewrite a section's harmonic PLAN
    /// itself (upstream of chord realization) rather than decorate or
    /// substitute individual realized chords — reusable by any future
    /// style wanting a real sequence episode (Classical, Cinematic, Prog
    /// Rock). Off by default.
    #[serde(default)]
    pub harmonic_sequence: bool,
    /// HARMONIC STASIS: when a voice (Harmony or Bass) repeats the exact
    /// same pitch across two consecutive chord onsets, tie the notes into
    /// one longer sustained note instead of re-striking. The Ambient
    /// habit: repetition becomes duration — the mechanism that turns a
    /// static progression into a genuine drone rather than a re-attacked
    /// block chord. Reusable by any future drone/soundscape/pad style.
    /// Off by default.
    #[serde(default)]
    pub harmonic_stasis: bool,
    /// ROLL ORNAMENTS: replace the single-grace "cut" (Celtic's habit,
    /// `apply_grace_ornaments`) with a full five-note ORNAMENT CHAIN —
    /// main, upper cut, main, lower cut, main — filling exactly the
    /// original note's duration. The Irish Traditional habit: not one
    /// grace note but a genuine chain of them, reusable by any future
    /// style wanting a denser, more virtuosic decoration than a single
    /// cut. Mutually exclusive with the plain cut pass (a note gets one
    /// or the other, never both). Off by default.
    #[serde(default)]
    pub roll_ornaments: bool,
    /// FULL DRONE: an escalation beyond the plain `drone` flag above —
    /// Celtic's `drone` replaces only the walking BASS with a tonic-fifth
    /// pedal while Harmony still moves through a real chord progression
    /// above it. This flag replaces Harmony TOO, with a static tonic-
    /// fifth-octave pad tied by `apply_harmonic_stasis` into one
    /// continuous sustain for the whole piece. The Hindustani-inspired
    /// habit: no chord progression exists at all, so "tension without
    /// modulation" is a structural guarantee, not a mood — there is
    /// nothing TO modulate. Implies `drone` internally (calls
    /// `apply_drone_pedal` itself); setting both is redundant, not wrong.
    /// Off by default.
    #[serde(default)]
    pub full_drone: bool,
}

fn default_swing() -> f32 {
    0.5
}

fn default_intro_bars() -> u8 {
    2
}

fn default_damage() -> f32 {
    0.5
}

fn default_true() -> bool {
    true
}

/// The full declarative spec. See the module docs for the philosophy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositionSpec {
    pub name: String,
    /// The piece's emotional posture. `None` (the default) keeps the
    /// engine's native temperament — reflective, restrained. Setting an
    /// [`Attitude`] changes how the piece BEHAVES, not just its dials.
    #[serde(default)]
    pub attitude: Option<Attitude>,
    /// Mode override. `None` (the default) keeps the valence mapping:
    /// positive intent → major, negative → (harmonic) minor. Setting a mode
    /// pins the key's tonality REGARDLESS of valence — `Dorian`,
    /// `Phrygian`, `Lydian`, `Mixolydian`, and `Aeolian` give genuinely
    /// modal harmony (mode-native cadences, no borrowed leading tone);
    /// `Ionian`/`HarmonicMinor` pin plain major/minor. The grammar-
    /// incompatible modes (Locrian, pentatonics, whole-tone, melodic
    /// minor) are rejected by `validate()`.
    #[serde(default)]
    pub mode: Option<crate::scale::Mode>,
    /// Beats per bar (2–12 validated; 3 and 4 are the well-trodden paths).
    pub meter: u8,
    /// BPM at arousal 0 and arousal 1.
    pub tempo_range: (f32, f32),
    /// Motif templates by arousal tier (calm < 0.33 ≤ medium < 0.66 ≤ busy).
    /// Every template must total exactly `meter` beats.
    pub motifs_calm: Vec<Vec<SpecNote>>,
    pub motifs_medium: Vec<Vec<SpecNote>>,
    pub motifs_busy: Vec<Vec<SpecNote>>,
    pub progression: ProgressionSpec,
    /// Accompaniment textures the seed picks from (non-empty).
    pub accompaniment_pool: Vec<Accompaniment>,
    /// Forms the seed picks from (non-empty).
    pub form_pool: Vec<FormKind>,
    /// The style's melodic DNA (empty = the classic shared hook pools).
    #[serde(default)]
    pub melody: MelodicDna,
    /// The style's phrase rhetoric (default = classic shared behavior).
    #[serde(default)]
    pub rhetoric: PhraseRhetoric,
    /// The style's development grammar (default = classic shared behavior).
    #[serde(default)]
    pub development: DevelopmentDna,
    /// Meters the PREMISE layer may choose between per candidate (empty =
    /// the style's meter is fixed — e.g. the waltz IS 3/4). When the
    /// premise picks a meter differing from `meter`, it ADAPTS the motif
    /// banks exactly (see `premise::adapt_template_to_meter`) so the
    /// premise spec still satisfies the bank invariant.
    #[serde(default)]
    pub meter_pool: Vec<u8>,
    /// Modal centers the PREMISE layer may choose between per candidate
    /// (empty = the style's mode is fixed — its identity, e.g. tango's
    /// harmonic minor). Entries use the same rules as `mode`.
    #[serde(default)]
    pub mode_pool: Vec<Option<crate::scale::Mode>>,
    /// Instrument-name triples (melody, harmony, bass) the seed picks from.
    /// Names are resolved by the renderer; unrecognized names fall back to
    /// its defaults LOUDLY, never silently.
    pub ensemble_pool: Vec<[String; 3]>,
    /// The counter-melody's own voice. `None` (the default) lets the
    /// renderer pick a timbral CONTRAST to the melody and bass — a
    /// counterline played by the bass instrument doesn't separate as its
    /// own emotional line (listening-review finding). Set a name here to
    /// choose explicitly; unrecognized names fall back loudly.
    #[serde(default)]
    pub counter_instrument: Option<String>,
    pub texture: TextureSpec,
}

impl CompositionSpec {
    /// Reject malformed specs with every problem listed (not just the
    /// first). The engine's invariants (voice leading, cadences, collision
    /// rules) are enforced downstream regardless — validation covers the
    /// spec-shaped mistakes a hand-edited JSON can make.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if !(2..=12).contains(&self.meter) {
            errors.push(format!("meter {} outside 2..=12", self.meter));
        }
        let (lo, hi) = self.tempo_range;
        if !(20.0..=400.0).contains(&lo) || !(20.0..=400.0).contains(&hi) || lo > hi {
            errors.push(format!("tempo_range ({lo}, {hi}) not a sane BPM range"));
        }
        for (tier, bank) in [
            ("calm", &self.motifs_calm),
            ("medium", &self.motifs_medium),
            ("busy", &self.motifs_busy),
        ] {
            if bank.is_empty() {
                errors.push(format!("motifs_{tier} is empty"));
            }
            for (i, template) in bank.iter().enumerate() {
                if template.is_empty() {
                    errors.push(format!("motifs_{tier}[{i}] has no notes"));
                    continue;
                }
                let mut total = Duration::zero();
                let mut bad = false;
                for &(_deg, num, den) in template {
                    if den == 0 || num <= 0 {
                        errors.push(format!(
                            "motifs_{tier}[{i}] has a non-positive duration {num}/{den}"
                        ));
                        bad = true;
                        break;
                    }
                    total = total + Duration::new(num, den);
                }
                if !bad && (total.beats() - self.meter as f64).abs() > 1e-9 {
                    errors.push(format!(
                        "motifs_{tier}[{i}] totals {} beats, meter is {}",
                        total.beats(),
                        self.meter
                    ));
                }
            }
        }
        if let ProgressionSpec::Archetype(degs) = &self.progression {
            if degs.is_empty() {
                errors.push("progression archetype is empty".into());
            }
            for &d in degs {
                if !(1..=7).contains(&d) {
                    errors.push(format!("progression degree {d} outside 1..=7"));
                }
            }
        }
        if self.accompaniment_pool.is_empty() {
            errors.push("accompaniment_pool is empty".into());
        }
        if self.form_pool.is_empty() {
            errors.push("form_pool is empty".into());
        }
        if self.ensemble_pool.is_empty() {
            errors.push("ensemble_pool is empty".into());
        }
        if let Some(name) = &self.counter_instrument
            && name.trim().is_empty()
        {
            errors.push("counter_instrument is an empty string (use null for auto)".into());
        }
        if let Some(mode) = self.mode
            && crate::harmony::Key::modal(crate::pitch::PitchClass::new(0), mode).is_none()
        {
            errors.push(format!(
                "mode {mode:?} is not usable for composition (needs 7 degrees and a stable tonic \
                 triad — use Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, or \
                 HarmonicMinor)"
            ));
        }
        if self.texture.coda_bars > 8 {
            errors.push(format!("coda_bars {} > 8", self.texture.coda_bars));
        }
        if !(0.5..=0.75).contains(&self.texture.swing) {
            errors.push(format!(
                "swing {} outside 0.5 (straight) ..= 0.75 (hard shuffle)",
                self.texture.swing
            ));
        }
        if !(0.0..=1.0).contains(&self.texture.damage) {
            errors.push(format!(
                "damage {} outside 0.0 (pristine) ..= 1.0 (full damage pass)",
                self.texture.damage
            ));
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Motif for an arousal tier and seed, with the same 4-orientation
    /// variety multiplication the built-in banks always had.
    pub fn motif(&self, arousal: f32, seed: u64) -> Motif {
        let bank = if arousal < 0.33 {
            &self.motifs_calm
        } else if arousal < 0.66 {
            &self.motifs_medium
        } else {
            &self.motifs_busy
        };
        let idx = (seed % bank.len() as u64) as usize;
        let notes: Vec<(i32, Duration)> = bank[idx]
            .iter()
            .map(|&(deg, num, den)| (deg, Duration::new(num, den)))
            .collect();
        let picked = Motif::from_degrees(&notes);
        let orientation = (seed / bank.len() as u64) % 4;
        crate::form::oriented(&picked, orientation)
    }

    /// Force every arousal tier's motif bank to a single, caller-supplied
    /// motif — the minimal-pair listening-study control: presenting the
    /// SAME melodic material through several different grammar families
    /// isolates structural cues (form, harmony, phrase grammar) as what a
    /// listener actually recognizes, rather than "I remember this tune."
    /// Returns `false` (and changes nothing) if `source` doesn't fit one
    /// bar of this spec's own meter — installing a motif the engine would
    /// have to silently truncate or overrun defeats the point of a
    /// controlled comparison. Also disables the hook-cell graft (`texture.
    /// hook_cell`): grafting a DIFFERENT per-style hook onto the shared
    /// motif would reintroduce exactly the per-family melodic variation
    /// this method exists to eliminate.
    pub fn install_primary_motif_for_seed(&mut self, source: &Motif, _seed: u64) -> bool {
        let total_beats: f64 = source.notes.iter().map(|n| n.duration.beats()).sum();
        if total_beats > self.meter as f64 + 1e-9 {
            return false;
        }
        let spec_notes: Vec<SpecNote> = source
            .notes
            .iter()
            .map(|n| (n.degree.unwrap_or(1), n.duration.num(), n.duration.den()))
            .collect();
        self.motifs_calm = vec![spec_notes.clone()];
        self.motifs_medium = vec![spec_notes.clone()];
        self.motifs_busy = vec![spec_notes];
        self.texture.hook_cell = false;
        true
    }

    /// A progression `bars` measures long, from the grammar or the cycled
    /// archetype (same length contract as [`crate::style::Style`] had).
    pub fn progression(&self, bars: usize, seed: u64) -> Progression {
        let bars = bars.max(1);
        match &self.progression {
            ProgressionSpec::Grammar => Progression::generate(bars, seed),
            ProgressionSpec::GrammarWithPalette(palette) => {
                Progression::generate_with(bars, seed, palette)
            }
            ProgressionSpec::Archetype(degrees) => {
                let cycled: Vec<i32> = (0..bars).map(|i| degrees[i % degrees.len()]).collect();
                Progression::new(cycled)
            }
            ProgressionSpec::ArchetypePool(pool) => {
                let degrees = &pool[(seed as usize) % pool.len()];
                let cycled: Vec<i32> = (0..bars).map(|i| degrees[i % degrees.len()]).collect();
                Progression::new(cycled)
            }
        }
    }

    /// Accompaniment texture by seed (`seed / 2` decorrelates from the form
    /// pick, as the style system established).
    pub fn accompaniment(&self, seed: u64) -> Accompaniment {
        self.accompaniment_pool[(seed as usize / 2) % self.accompaniment_pool.len()]
    }

    /// Large-scale form by seed.
    pub fn form_kind(&self, seed: u64) -> FormKind {
        self.form_pool[(seed % self.form_pool.len() as u64) as usize]
    }

    /// Ensemble instrument names by seed.
    pub fn ensemble(&self, seed: u64) -> &[String; 3] {
        &self.ensemble_pool[(seed % self.ensemble_pool.len() as u64) as usize]
    }

    /// Tempo for an arousal, interpolated across `tempo_range`.
    pub fn tempo(&self, arousal: f32) -> f32 {
        let (lo, hi) = self.tempo_range;
        lo + arousal.clamp(0.0, 1.0) * (hi - lo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    #[test]
    fn every_builtin_style_spec_validates() {
        for style in [
            Style::Classical,
            Style::Waltz,
            Style::Folk,
            Style::Cinematic,
            Style::Playful,
        ] {
            let spec = style.spec();
            assert_eq!(
                spec.validate(),
                Ok(()),
                "{style:?}'s preset spec must validate"
            );
        }
    }

    #[test]
    fn validation_catches_the_hand_edit_mistakes() {
        let mut spec = Style::Classical.spec();
        spec.meter = 0;
        spec.motifs_calm = vec![vec![(1, 1, 1)]]; // 1 beat, meter says 0
        spec.accompaniment_pool.clear();
        spec.progression = ProgressionSpec::Archetype(vec![9]);
        let errors = spec.validate().unwrap_err();
        let text = errors.join("\n");
        assert!(text.contains("meter"), "{text}");
        assert!(text.contains("accompaniment_pool"), "{text}");
        assert!(text.contains("degree 9"), "{text}");
    }

    #[test]
    fn specs_round_trip_through_json() {
        let spec = Style::Playful.spec();
        let json = serde_json::to_string_pretty(&spec).unwrap();
        let back: CompositionSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    #[test]
    fn saved_specs_from_before_the_new_fields_still_load() {
        // Users have saved specs on disk (Studio's data/specs/) from before
        // `mode` and `counter_instrument` existed — those JSONs must keep
        // deserializing, with both defaulting to None.
        let mut v = serde_json::to_value(Style::Classical.spec()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("mode");
        obj.remove("counter_instrument");
        let spec: CompositionSpec = serde_json::from_value(v).unwrap();
        assert_eq!(spec.mode, None);
        assert_eq!(spec.counter_instrument, None);
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn spec_motifs_match_the_style_banks_they_transcribe() {
        // The preset spec's motif picker must agree with the original
        // style-bank picker for every tier and a spread of seeds — this is
        // the byte-identical-preset guarantee at the motif level.
        for style in [
            Style::Classical,
            Style::Waltz,
            Style::Folk,
            Style::Cinematic,
            Style::Playful,
        ] {
            let spec = style.spec();
            for arousal in [0.1f32, 0.5, 0.9] {
                for seed in 0..12u64 {
                    assert_eq!(
                        spec.motif(arousal, seed),
                        style.motif(arousal, seed),
                        "{style:?} arousal {arousal} seed {seed}"
                    );
                }
            }
        }
    }
}
