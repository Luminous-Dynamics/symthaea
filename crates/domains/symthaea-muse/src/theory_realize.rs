// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Realize a symbolic [`Score`](symthaea_music_theory::Score) as audio.
//!
//! This is the seam where the two crates meet: `symthaea-music-theory` decides
//! WHAT to play (motif, harmony, cadences, structure); muse decides HOW it
//! SOUNDS. Crucially, the realizer reads the score's structural `Emphasis`
//! annotations and applies **expressive timing** — a breath before each
//! phrase, an agogic dwell on the climax, and a ritardando into cadences —
//! tied to the music's structure rather than the old random jitter. That
//! structure-driven rubato is where "feeling" is supposed to come from.
//!
//! Each voice also gets its OWN real instrument (see [`instruments_for`])
//! instead of one synthesis patch shared by melody/harmony/bass — that
//! shared-patch design was the actual cause of the "80s retro game"
//! character early listening found: every voice differed only in pitch and
//! volume, never in timbre. `Style` (from `symthaea-music-theory`) selects
//! which ensemble plays: a string trio for Classical/Waltz, flute+guitar+
//! upright bass for Folk, strings+organ for Cinematic, clarinet+electric
//! piano+guitar for Playful.
//!
//! Feature-gated behind `theory`.

use crate::instruments::Instrument;
use crate::percussion::{self, DrumColor, DrumHit, DrumType};
use crate::voice::{Arrangement, Voice, VoiceRole};
use crate::{AudioData, Composition, MuseConfig, MusicalState, Note};
use symthaea_music_theory::score::{Emphasis, VoiceRole as TheoryRole};
use symthaea_music_theory::{MusicalIntent, Score as TheoryScore, Style};

/// Per-style-family PERFORMANCE INTERPRETATION — how a tradition's player
/// treats the clock, not what notes they play. Until this existed, ONE
/// classical-piano-derived performer interpreted every tradition: the same
/// rubato constants, the same humanize jitter, the same MAESTRO piano
/// expressive blend for a tango, a drone piece, and a nocturne alike.
///
/// v1 is deliberately a CONSTANTS TABLE, not a new model and not a spec
/// field: each variant is a set of scale factors applied to the existing
/// performance machinery ([`Rubato::from_score`]'s structural pauses,
/// [`crate::performance::humanize_score_note`]'s timing SD, and
/// [`apply_expressive_model`]'s blend). [`PerformanceDialect::ClassicalRubato`]
/// is the identity — every factor exactly 1.0 — so any render that derives
/// it is bit-identical to the pre-dialect behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PerformanceDialect {
    /// The historical behavior, exactly: full structural rubato, the
    /// standard Φ-scaled jitter, the full 0.7 MAESTRO expressive blend.
    /// All scale factors are 1.0 — the identity dialect.
    #[default]
    ClassicalRubato,
    /// Groove styles (tango, son montuno, flamenco, backbeat pop): the
    /// GROOVE is the clock. Structural rubato scaled way down (0.25x),
    /// timing jitter tightened (0.6x), expressive-model shaping reduced
    /// (0.4x) — a dance band does not breathe between phrases, it locks.
    DanceLocked,
    /// Process music (minimalism, ambient, drone): the piece is a process,
    /// not an utterance. Rubato nearly eliminated (0.1x), jitter tightened
    /// (0.4x), and NO expressive-model accent shaping at all (0.0x) — a
    /// pulse that leaned into climaxes would stop being a pulse.
    ProcessExact,
    /// Folk/celtic/irish session playing: the breath between phrases is
    /// kept (breath/climax 1.0x) but the big cadential ritardando is
    /// reduced (0.6x — a reel drives through its cadences), and jitter is
    /// slightly LOOSER (1.2x): session looseness, not concert precision.
    FolkLift,
    /// Blues/jazz: laid-back time. Jitter up (1.3x), phrase-final rubato
    /// and the expressive blend both kept — a jazz ballad leans harder
    /// into time than a classical player, never less.
    JazzLaidBack,
}

impl PerformanceDialect {
    /// Scale on the phrase-start breath (base 0.10 beats of held time).
    pub(crate) fn breath_scale(self) -> f64 {
        match self {
            PerformanceDialect::ClassicalRubato
            | PerformanceDialect::FolkLift
            | PerformanceDialect::JazzLaidBack => 1.0,
            PerformanceDialect::DanceLocked => 0.25,
            PerformanceDialect::ProcessExact => 0.1,
        }
    }

    /// Scale on the climax agogic dwell (base 0.18 beats).
    pub(crate) fn climax_scale(self) -> f64 {
        match self {
            PerformanceDialect::ClassicalRubato
            | PerformanceDialect::FolkLift
            | PerformanceDialect::JazzLaidBack => 1.0,
            PerformanceDialect::DanceLocked => 0.25,
            PerformanceDialect::ProcessExact => 0.1,
        }
    }

    /// Scale on the cadential ritardando (base 0.35 interior / 0.9 final).
    pub(crate) fn cadence_scale(self) -> f64 {
        match self {
            PerformanceDialect::ClassicalRubato | PerformanceDialect::JazzLaidBack => 1.0,
            PerformanceDialect::FolkLift => 0.6,
            PerformanceDialect::DanceLocked => 0.25,
            PerformanceDialect::ProcessExact => 0.1,
        }
    }

    /// Scale on the humanize TIMING jitter SD (base 7-22ms, Φ-scaled).
    pub(crate) fn jitter_scale(self) -> f32 {
        match self {
            PerformanceDialect::ClassicalRubato => 1.0,
            PerformanceDialect::DanceLocked => 0.6,
            PerformanceDialect::ProcessExact => 0.4,
            PerformanceDialect::FolkLift => 1.2,
            PerformanceDialect::JazzLaidBack => 1.3,
        }
    }

    /// Scale on the MAESTRO expressive-model blend (base 0.7). 0.0 means
    /// the model is skipped entirely (no accent shaping, no learned
    /// articulation) — the ProcessExact posture.
    pub(crate) fn expressive_blend_scale(self) -> f32 {
        match self {
            PerformanceDialect::ClassicalRubato
            | PerformanceDialect::FolkLift
            | PerformanceDialect::JazzLaidBack => 1.0,
            PerformanceDialect::DanceLocked => 0.4,
            PerformanceDialect::ProcessExact => 0.0,
        }
    }
}

/// Derive the performance dialect from the spec's own family markers.
///
/// A v1 HEURISTIC, pending the real grammar-interface work: the spec has no
/// explicit dialect field yet, so this keys off the properties that already
/// encode family identity — texture process flags, rhythm-cell
/// accompaniments, swing, seventh chords, drone/ornament habits. Checked in
/// priority order (a process piece with a pulse is still a process piece;
/// a groove with a drone is still a groove):
///
/// 1. `harmonic_stasis`/`full_drone`/`additive_process` → [`PerformanceDialect::ProcessExact`]
/// 2. a rhythm-cell accompaniment (Habanera/Montuno/BossaComp/CompasGait/
///    Shuffle), non-straight swing, or a full backbeat kit → groove: with a
///    jazz marker (Shuffle, swing ≠ 0.5, or `seventh_chords`) →
///    [`PerformanceDialect::JazzLaidBack`], else [`PerformanceDialect::DanceLocked`]
/// 3. `roll_ornaments` or `drone` → [`PerformanceDialect::FolkLift`]
/// 4. everything else → [`PerformanceDialect::ClassicalRubato`] (the identity)
///
/// Known v1 limits (documented, not hidden): March reads as ClassicalRubato
/// (no spec property marks its clock-insistence), and plain Folk/ModalFolk
/// read as ClassicalRubato (their specs carry none of the folk habit flags
/// — only Celtic's drone and Irish's rolls are visible at the spec level).
pub fn dialect_for_spec(spec: &symthaea_music_theory::CompositionSpec) -> PerformanceDialect {
    use symthaea_music_theory::Accompaniment as A;
    use symthaea_music_theory::DrumPolicy;
    let t = &spec.texture;
    if t.harmonic_stasis || t.full_drone || t.additive_process {
        return PerformanceDialect::ProcessExact;
    }
    let has_shuffle = spec.accompaniment_pool.contains(&A::Shuffle);
    let has_groove_cell = has_shuffle
        || spec
            .accompaniment_pool
            .iter()
            .any(|a| matches!(a, A::Habanera | A::Montuno | A::BossaComp | A::CompasGait));
    let swung = (t.swing - 0.5).abs() > 1e-6;
    if has_groove_cell || swung || t.drums == DrumPolicy::Backbeat {
        return if has_shuffle || swung || t.seventh_chords {
            PerformanceDialect::JazzLaidBack
        } else {
            PerformanceDialect::DanceLocked
        };
    }
    if t.roll_ornaments || t.drone {
        return PerformanceDialect::FolkLift;
    }
    PerformanceDialect::ClassicalRubato
}

/// The (melody, harmony, bass) instrument ensemble for a given [`Style`],
/// chosen from a small per-style POOL by `seed` rather than a single fixed
/// tuple. Deliberately real acoustic instruments (see `instruments.rs`'s
/// published-measurement partial tables) — this is what actually makes
/// styles sound like different genres, and now different PIECES within a
/// style, rather than the same three timbres forever.
pub(crate) fn instruments_for(style: Style, seed: u64) -> (Instrument, Instrument, Instrument) {
    let pool: &[(Instrument, Instrument, Instrument)] = match style {
        // String-trio family, with a couple of real substitutions: a wind
        // lead (flute) or a plucked harmony (harp) instead of the default.
        Style::Classical | Style::Waltz => &[
            (Instrument::Violin, Instrument::Piano, Instrument::Cello),
            (Instrument::Flute, Instrument::Piano, Instrument::Cello),
            (Instrument::Violin, Instrument::Harp, Instrument::Cello),
            // Viola instead of violin: the same string-trio identity, a
            // warmer/darker lead register (Fletcher & Rossing's "viola
            // problem" — a body genuinely too small for its pitch range —
            // is exactly the character that distinguishes it here).
            (Instrument::Viola, Instrument::Piano, Instrument::Cello),
        ],
        // Open, acoustic, folk-song-flavored: always plucked harmony/bass
        // (Karplus-Strong), but the lead voice varies.
        Style::Folk => &[
            (
                Instrument::Flute,
                Instrument::AcousticGuitar,
                Instrument::UprightBass,
            ),
            (
                Instrument::Clarinet,
                Instrument::Harp,
                Instrument::UprightBass,
            ),
            (Instrument::Flute, Instrument::Koto, Instrument::UprightBass),
            // Old-time/bluegrass color: banjo's bright plucked decay as
            // the harmony voice, genuinely idiomatic for this style's
            // "open, acoustic, folk-song" identity in a way guitar/harp/
            // koto don't cover.
            (
                Instrument::Clarinet,
                Instrument::Banjo,
                Instrument::UprightBass,
            ),
        ],
        // Sweeping and dramatic: a soaring lead over a sustained pad,
        // grounded by cello.
        Style::Cinematic => &[
            (Instrument::Violin, Instrument::Organ, Instrument::Cello),
            (Instrument::Trumpet, Instrument::Organ, Instrument::Cello),
            (Instrument::Violin, Instrument::Pad, Instrument::Cello),
            // The epic-orchestral variant: horn call over chorusing
            // strings, tuba anchoring the low end — the John Williams
            // register this style's "sweeping and dramatic" doc comment
            // describes but the existing entries (a solo violin/trumpet
            // over one pad) don't yet reach.
            (
                Instrument::FrenchHorn,
                Instrument::StringEnsemble,
                Instrument::Tuba,
            ),
            // The intimate-cinematic variant: duduk's narrow-dynamic,
            // breathy sustained character — a texture no other style uses
            // — over a pad, cello grounding it.
            (Instrument::Duduk, Instrument::Pad, Instrument::Cello),
        ],
        // Bright and bouncy: a reedy/brassy lead, an FM or plucked
        // harmony, plucked guitar bass throughout.
        Style::Playful => &[
            (
                Instrument::Clarinet,
                Instrument::ElectricPiano,
                Instrument::AcousticGuitar,
            ),
            (
                Instrument::Saxophone,
                Instrument::Marimba,
                Instrument::AcousticGuitar,
            ),
            (
                Instrument::Trumpet,
                Instrument::Kalimba,
                Instrument::AcousticGuitar,
            ),
            // Xylophone's brighter, drier, much faster-decaying mallet
            // character as the lead itself — distinct from Marimba/
            // Kalimba's existing role as harmony/accompaniment voices.
            (
                Instrument::Xylophone,
                Instrument::ElectricPiano,
                Instrument::AcousticGuitar,
            ),
        ],
        // The spec-first styles: palettes mirror their preset specs.
        Style::Nocturne => &[
            (Instrument::Clarinet, Instrument::Piano, Instrument::Cello),
            (Instrument::Flute, Instrument::Piano, Instrument::Cello),
            // Bassoon as the bass voice: a darker, more intimate register
            // than cello for the same lyrical wind-chamber texture —
            // genuinely idiomatic (bassoon-as-bass-voice is standard wind-
            // ensemble practice), not a forced substitution.
            (Instrument::Clarinet, Instrument::Piano, Instrument::Bassoon),
        ],
        Style::March => &[
            (Instrument::Trumpet, Instrument::Organ, Instrument::Cello),
            // The brass-band variant: trombone's own bugle-call register
            // over a tuba anchor — March's "rhythmic insistence" identity
            // played by the family that actually marches.
            (Instrument::Trombone, Instrument::Organ, Instrument::Tuba),
        ],
        Style::Lullaby => &[
            (Instrument::Flute, Instrument::Kalimba, Instrument::Cello),
            // The literal music-box sound — a genuinely new FM timbre for
            // a style whose whole identity is a bedtime music box.
            (Instrument::Flute, Instrument::MusicBox, Instrument::Cello),
        ],
        Style::ModalFolk => &[(
            Instrument::Flute,
            Instrument::Marimba,
            Instrument::UprightBass,
        )],
        // Three contrapuntal LINES, not lead+accompaniment: sustained,
        // even-toned voices so the counterpoint reads (the harmony slot is
        // empty in a fugue score — the middle voice arrives as the
        // counter-melody, defaulting to clarinet via `contrast_counter`).
        Style::Fugue => &[(Instrument::Organ, Instrument::Piano, Instrument::Cello)],
        // The ground must be UNMISTAKABLE: cello carries it; a reedy lead
        // varies above. (Harmony slot empty, as in the fugue.)
        Style::Passacaglia => &[(Instrument::Clarinet, Instrument::Organ, Instrument::Cello)],
        // Violin lead over piano habanera and a plucked anchor.
        Style::Tango => &[
            (
                Instrument::Violin,
                Instrument::Piano,
                Instrument::UprightBass,
            ),
            // The bandoneon stand-in: accordion is tango's own harmony
            // voice (no GM bandoneon exists; this is the nearest real
            // free-reed cousin), replacing the piano habanera comp.
            (
                Instrument::Violin,
                Instrument::Accordion,
                Instrument::UprightBass,
            ),
        ],
        // The jig's own palette: fiddle or whistle over harp/guitar, cello
        // standing in for the tonic-fifth drone (a real bagpipe/hurdy-
        // gurdy drone has no close GM equivalent, so the sustained low
        // string carries it).
        Style::Celtic => &[
            (Instrument::Violin, Instrument::Harp, Instrument::Cello),
            (
                Instrument::Flute,
                Instrument::AcousticGuitar,
                Instrument::Cello,
            ),
        ],
        // A reedy lead over electric piano's comping, upright bass
        // walking/shuffling underneath.
        Style::Blues => &[
            (
                Instrument::Saxophone,
                Instrument::ElectricPiano,
                Instrument::UprightBass,
            ),
            (
                Instrument::Clarinet,
                Instrument::Piano,
                Instrument::UprightBass,
            ),
        ],
        // Piano or flute over rippling harp, cello grounding the wash.
        Style::Impressionism => &[
            (Instrument::Piano, Instrument::Harp, Instrument::Cello),
            (Instrument::Flute, Instrument::Harp, Instrument::Cello),
            // Oboe's plaintive, reedy tone over the same rippling harp
            // wash — a French-woodwind color this style's harmonic
            // atmosphere is written for.
            (Instrument::Oboe, Instrument::Harp, Instrument::Cello),
        ],
        // The organ carries every voice, as a real hymn realization does;
        // bell stands in for a second manual/stop color.
        Style::SacredChoral => &[
            (Instrument::Organ, Instrument::Organ, Instrument::Cello),
            (Instrument::Organ, Instrument::Bell, Instrument::Cello),
            // Real choir voices, finally: two distinct vowel colors
            // (Aah lead, Ooh harmony — darker/more covered) instead of
            // organ standing in for the choir it's meant to accompany.
            (
                Instrument::ChoirAah,
                Instrument::ChoirOoh,
                Instrument::Cello,
            ),
        ],
        // Mallet/keys pulse — the Glass/Reich ensemble palette.
        Style::Minimalism => &[
            (
                Instrument::Marimba,
                Instrument::ElectricPiano,
                Instrument::UprightBass,
            ),
            // Vibraphone's shimmering, longer-ringing pulse — the other
            // half of the Glass/Reich mallet palette Marimba alone
            // doesn't cover.
            (
                Instrument::Vibraphone,
                Instrument::Marimba,
                Instrument::UprightBass,
            ),
            (
                Instrument::Piano,
                Instrument::Marimba,
                Instrument::UprightBass,
            ),
        ],
        // The torch-song trio: a reedy lead over piano/electric piano,
        // upright bass grounding it.
        Style::JazzBallad => &[
            (
                Instrument::Saxophone,
                Instrument::Piano,
                Instrument::UprightBass,
            ),
            (
                Instrument::Clarinet,
                Instrument::ElectricPiano,
                Instrument::UprightBass,
            ),
        ],
        // The trio-sonata continuo: violin lead, organ realizing the
        // harmony, cello grounding the bass line.
        Style::BaroqueSuite => &[
            (Instrument::Violin, Instrument::Organ, Instrument::Cello),
            // The actual period continuo instrument: this style's own
            // design docs call for "broken-chord continuo," and
            // harpsichord — not organ — is what that historically means.
            (
                Instrument::Violin,
                Instrument::Harpsichord,
                Instrument::Cello,
            ),
        ],
        // The prog rhythm section: an angular synth lead over electric
        // piano, upright bass grounding it.
        Style::ProgFolk => &[
            (
                Instrument::SawLead,
                Instrument::ElectricPiano,
                Instrument::UprightBass,
            ),
            // A rock/fusion variant: electric bass replacing the upright,
            // matching this style's synth-lead rhythm section rather than
            // an acoustic jazz one.
            (
                Instrument::SawLead,
                Instrument::ElectricPiano,
                Instrument::ElectricBass,
            ),
        ],
        // A soundscape: two pad voices framing a sparse bell.
        Style::Ambient => &[
            (Instrument::Pad, Instrument::Bell, Instrument::Pad),
            // Celesta's delicate, slow-decaying bell-like color as the
            // sparse center voice — a second stillness timbre distinct
            // from Bell's brighter attack.
            (Instrument::Pad, Instrument::Celesta, Instrument::Pad),
        ],
        // The classical trio: violin carrying both subjects, piano
        // realizing the Alberti-figured harmony, cello grounding it.
        Style::Sonata => &[(Instrument::Violin, Instrument::Piano, Instrument::Cello)],
        // Soprano/alto/bass: flute carries the top line, organ substitutes
        // for the choir's inner voice, cello grounds it.
        Style::RenaissancePolyphony => &[(Instrument::Flute, Instrument::Organ, Instrument::Cello)],
        // The son montuno trio: a horn line, piano voicing the montuno
        // stabs (the real instrument for the pattern's name), upright
        // bass walking the tumbao.
        Style::AfroCuban => &[(
            Instrument::Saxophone,
            Instrument::Piano,
            Instrument::UprightBass,
        )],
        // The Andalusian trio: oud carries the cante's Moorish-inflected
        // melisma (a plucked, sustained-decay voice no other style has
        // used yet), acoustic guitar realizes the compás's rasgueado
        // stabs, cello grounds the anchor.
        Style::Flamenco => &[(
            Instrument::Oud,
            Instrument::AcousticGuitar,
            Instrument::Cello,
        )],
        // The Getz/Jobim trio: saxophone carries the relaxed lead (bossa's
        // most famous non-Brazilian voice), acoustic guitar realizes the
        // floating comp, upright bass holds and anticipates.
        Style::BossaNova => &[(
            Instrument::Saxophone,
            Instrument::AcousticGuitar,
            Instrument::UprightBass,
        )],
        // Theme A's voice: bowed strings over a harp pit-reduction and
        // cello grounding. Theme B's own instrument (the "second
        // character") comes from `contrast_counter(melody, bass)` below,
        // which resolves to Clarinet for this (Violin, _, Cello) pair —
        // a genuinely different reed voice, not an override needed.
        Style::Opera => &[(Instrument::Violin, Instrument::Harp, Instrument::Cello)],
        // A real session trio — flute over guitar and upright bass —
        // distinct from Celtic's fiddle/harp/cello soundtrack cast.
        Style::IrishTraditional => &[
            (
                Instrument::Flute,
                Instrument::AcousticGuitar,
                Instrument::UprightBass,
            ),
            // Mandolin lead: a real session instrument in its own right,
            // its fast decay well suited to reel/jig ornament rolls.
            (
                Instrument::Mandolin,
                Instrument::AcousticGuitar,
                Instrument::UprightBass,
            ),
        ],
        // Sitar's engine debut — the quintessential Hindustani lead voice
        // — over a sustained pad standing in for the tanpura's continuous
        // drone, upright bass carrying the pulsing tonic/fifth pedal.
        Style::HindustaniInspired => {
            &[(Instrument::Sitar, Instrument::Pad, Instrument::UprightBass)]
        }
    };
    pool[(seed % pool.len() as u64) as usize]
}

/// The counter-melody's default voice: a timbral CONTRAST to the lead and
/// bass. It used to play the bass instrument ("cello answering the violin"),
/// but a listening review found the counterline doesn't separate as its own
/// emotional line when it shares the bass's timbre — the classic fix is a
/// wind voice against strings. Specs can override via `counter_instrument`.
pub(crate) fn contrast_counter(melody: Instrument, bass: Instrument) -> Instrument {
    [Instrument::Clarinet, Instrument::Cello, Instrument::Flute]
        .into_iter()
        .find(|i| *i != melody && *i != bass)
        .unwrap_or(bass)
}

/// A deterministic pseudo-seed derived from a [`TheoryScore`]'s own content
/// (tonic, note count, tempo) — used to vary the instrument ensemble for
/// callers (like [`realize`]) that only have a `Score`, not the original
/// [`MusicalIntent`] that composed it (which already carries a real seed;
/// see [`compose_and_realize_styled`], which uses `intent.seed` directly).
fn variety_seed_from_score(score: &TheoryScore) -> u64 {
    let tonic = score.key.tonic.value() as u64;
    let notes = score.notes.len() as u64;
    let tempo = score.tempo_bpm as u64;
    tonic
        .wrapping_mul(31)
        .wrapping_add(notes)
        .wrapping_mul(31)
        .wrapping_add(tempo)
}

/// Convert a MIDI note number to frequency in Hz (A4 = 69 = 440 Hz).
fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2.0_f32.powf((midi as f32 - 69.0) / 12.0)
}

/// De-overlap a monophonic performed voice's notes in place: sorted by
/// start time, any note whose end now runs past the next note's start is
/// truncated to end exactly there; a note with no usable room at all (an
/// exact-duplicate or negative-gap start, which humanize's onset jitter can
/// produce) is dropped rather than forced into a degenerate/overlapping
/// duration. Mirrors `symthaea-music-theory`'s own `composer::deoverlap_voice`
/// (same reasoning, seconds/f32 instead of exact rational beats — this
/// layer runs after humanize, which isn't exact-rational).
fn deoverlap_performance_notes(notes: &mut Vec<Note>) {
    let mut order: Vec<usize> = (0..notes.len()).collect();
    order.sort_by(|&a, &b| notes[a].start_time.total_cmp(&notes[b].start_time));
    let mut to_remove = Vec::new();
    for w in order.windows(2) {
        let (a, b) = (w[0], w[1]);
        let a_end = notes[a].start_time + notes[a].duration;
        let b_start = notes[b].start_time;
        if a_end > b_start + 1e-6 {
            let gap = b_start - notes[a].start_time;
            if gap <= 1e-6 {
                to_remove.push(a);
            } else {
                notes[a].duration = gap;
            }
        }
    }
    to_remove.sort_unstable();
    to_remove.dedup();
    for i in to_remove.into_iter().rev() {
        notes.remove(i);
    }
}

/// Roll the tones of each block chord low-to-high like a real strum (plucked
/// instruments) or hand roll (keyboards). Chord tones arrive from the score
/// with bit-identical start times (identical onset beats through the same
/// rubato map), so grouping by exact equality is reliable — and this must run
/// BEFORE humanization, whose jitter breaks that exact equality. Each tone's
/// duration is shortened by its offset so the chord still releases together.
fn strum_chords(notes: &mut [(f64, Note)], instrument: Instrument) {
    // A guitar/harp strum is a wider gesture than a piano roll.
    let step = if instrument.uses_karplus_strong() {
        0.012
    } else {
        0.006
    };
    let mut i = 0;
    while i < notes.len() {
        let group_start = notes[i].1.start_time;
        let mut j = i + 1;
        while j < notes.len() && notes[j].1.start_time == group_start {
            j += 1;
        }
        if j - i > 1 {
            notes[i..j].sort_by(|a, b| a.1.frequency.partial_cmp(&b.1.frequency).unwrap());
            for (k, (_, n)) in notes[i..j].iter_mut().enumerate() {
                let offset = k as f32 * step;
                n.start_time += offset;
                n.duration = (n.duration - offset).max(0.02);
            }
        }
        i = j;
    }
}

/// Per-note humanization seed. Keyed on the note's ONSET BEAT rather than its
/// index, so every tone of a block chord (same onset) shares one seed — the
/// whole chord lands with a single human timing error and its strum roll stays
/// intact, instead of each string getting independent jitter that could
/// scramble the roll order. Distinct `voice_tag`s keep the lead and its
/// climax doubling from being jitter-locked to each other.
fn humanize_seed(voice_tag: u32, onset_beats: f64) -> u32 {
    voice_tag.wrapping_mul(0x9E37_79B9) ^ (onset_beats as f32).to_bits()
}

/// Warp the fractional position within each beat so the off-beat eighth
/// lands at `offbeat` (0.5 = straight, 2/3 = triplet swing). Piecewise
/// linear, strictly monotonic, identity at integer beats — a pure
/// performance-layer time map: notation stays straight, the ensemble
/// swings, exactly how real charts work.
fn swing_beat(b: f64, offbeat: f64) -> f64 {
    if (offbeat - 0.5).abs() < 1e-9 {
        return b;
    }
    let whole = b.floor();
    let frac = b - whole;
    let warped = if frac <= 0.5 {
        frac * (offbeat / 0.5)
    } else {
        offbeat + (frac - 0.5) * ((1.0 - offbeat) / 0.5)
    };
    whole + warped
}

/// The complete beat→seconds pipeline: swing first (within-beat warp),
/// then the structure-driven rubato. ONE timeline shared by every voice
/// AND the drums — swung melody over straight hats would fall apart
/// within a bar.
struct Timeline {
    rubato: Rubato,
    swing: f64,
}

impl Timeline {
    fn seconds(&self, beat: f64) -> f32 {
        self.rubato.seconds(swing_beat(beat, self.swing))
    }
}

/// The style-gated drum track for a score, on the SAME rubato timeline as
/// the pitched voices — a straight-clock drum grid under rubato'd melody
/// and harmony would drift audibly out of sync within one ritardando.
///
/// `percussion.rs` (full kick/snare/hat synthesis + humanization) was wired
/// only into the streaming path; the theory path had no pulse at all —
/// conspicuous for the rhythmic styles. Style honesty: Classical and Waltz
/// get NO drums (a chamber ensemble has no kit — the waltz's rhythm lives in
/// its oom-pah accompaniment); Folk gets a light kick + brushed-hat pulse;
/// Cinematic a sparse low pulse on each barline; Playful a full backbeat.
fn drum_hits(
    score: &TheoryScore,
    policy: symthaea_music_theory::DrumPolicy,
    timeline: &Timeline,
    state: &MusicalState,
    seed: u64,
) -> Vec<DrumHit> {
    use symthaea_music_theory::DrumPolicy as P;
    let meter = score.meter as f64;
    let total_bars = (score.total_beats.beats() / meter).ceil() as i64;
    let mut hits: Vec<DrumHit> = Vec::new();
    let mut push = |hits: &mut Vec<DrumHit>, drum: DrumType, beat: f64, velocity: f32| {
        hits.push(DrumHit {
            drum,
            time: timeline.seconds(beat),
            velocity,
        });
    };
    match policy {
        P::None => {}
        P::LightPulse => {
            for bar in 0..total_bars {
                let b0 = bar as f64 * meter;
                push(&mut hits, DrumType::Kick, b0, 0.55);
                for k in 1..meter as i64 {
                    push(&mut hits, DrumType::HiHat, b0 + k as f64, 0.30);
                }
            }
        }
        P::BarPulse => {
            for bar in 0..total_bars {
                push(&mut hits, DrumType::Kick, bar as f64 * meter, 0.45);
            }
        }
        P::Backbeat => {
            for bar in 0..total_bars {
                let b0 = bar as f64 * meter;
                push(&mut hits, DrumType::Kick, b0, 0.80);
                push(&mut hits, DrumType::Kick, b0 + 2.0, 0.70);
                push(&mut hits, DrumType::Snare, b0 + 1.0, 0.65);
                push(&mut hits, DrumType::Snare, b0 + 3.0, 0.70);
                for k in 0..(meter * 2.0) as i64 {
                    let vel = if k % 2 == 0 { 0.40 } else { 0.25 };
                    push(&mut hits, DrumType::HiHat, b0 + k as f64 * 0.5, vel);
                }
            }
        }
    }
    percussion::humanize_hits(&mut hits, state.consciousness_level, seed as u32);
    hits
}

/// The style-gated drum hits on the FULL performed timeline (swing ∘
/// rubato — the same clock [`performance_voices`] uses), for consumers
/// outside the audio renderer. MIDI export needs this: the audio path
/// builds its `Timeline` privately inside `realize_core`, and without a
/// shared entry point the exported `.mid` carried no percussion at all.
pub(crate) fn performance_drum_hits(
    score: &TheoryScore,
    spec: &symthaea_music_theory::CompositionSpec,
    state: &MusicalState,
    seed: u64,
) -> Vec<DrumHit> {
    // The SAME dialect-scaled rubato as the pitched voices — a drum grid
    // on the unscaled map would drift against a DanceLocked ensemble.
    let timeline = Timeline {
        rubato: Rubato::from_score_with_dialect(score, dialect_for_spec(spec)),
        swing: spec.texture.swing as f64,
    };
    drum_hits(score, spec.texture.drums, &timeline, state, seed)
}

/// Mix a drum track into the rendered stereo frames. Drums stay DRY (no
/// reverb — they're added after the ensemble render) and punchy: kick/snare
/// anchored center, hats slightly right, modest level (accompaniment, not a
/// drum feature). Runs before mastering, whose limiter keeps peaks safe.
fn mix_drums(frames: &mut [[f32; 2]], hits: &[DrumHit], sample_rate: u32, state: &MusicalState) {
    let color = DrumColor {
        tightness: state.noradrenaline,
        brightness: state.dopamine,
        warmth: state.valence,
    };
    for hit in hits {
        let buf = percussion::render_drum_colored(hit, sample_rate, &color);
        let pan = match hit.drum {
            DrumType::HiHat => 0.25,
            _ => 0.0,
        };
        let theta = (pan + 1.0) * std::f32::consts::FRAC_PI_4;
        let (gl, gr) = (theta.cos(), theta.sin());
        let start = (hit.time * sample_rate as f32) as usize;
        for (i, &s) in buf.iter().enumerate() {
            let Some(frame) = frames.get_mut(start + i) else {
                break;
            };
            frame[0] += s * 0.35 * gl;
            frame[1] += s * 0.35 * gr;
        }
    }
}

/// The MAESTRO-trained expressive model, parsed once per process.
fn expressive_model() -> &'static crate::expressive::ExpressiveModel {
    static MODEL: std::sync::OnceLock<crate::expressive::ExpressiveModel> =
        std::sync::OnceLock::new();
    MODEL.get_or_init(crate::expressive::ExpressiveModel::from_embedded)
}

/// Apply learned expression (see [`crate::expressive`]) to the melody voice:
/// velocity accents layered ADDITIVELY on the score's structural dynamics
/// (the model was trained on deviation-from-local-mean, so the composer's
/// phrase arcs stay authoritative), and articulation replacing the flat
/// legato for interior notes (duration = learned fraction of the actual
/// inter-onset gap on the rubato timeline). First/last notes keep their
/// written durations — the model needs both neighbors for context.
fn apply_expressive_model(
    beat_notes: &mut [(f64, Note)],
    theory_notes: &[symthaea_music_theory::score::ScoreNote],
    blend_scale: f32,
) {
    // Dialect gate: 0.0 (ProcessExact) skips the model entirely — no
    // accent shaping, no learned articulation. 1.0 is the exact historical
    // path (0.7 * 1.0 == 0.7 bit-exactly; the articulation branch below
    // keeps the historical assignment rather than a lerp through it).
    if blend_scale <= 0.0 || beat_notes.len() < 3 || theory_notes.len() != beat_notes.len() {
        return;
    }
    let model = expressive_model();
    let total_beats = beat_notes.last().unwrap().0.max(1.0);
    for i in 1..beat_notes.len() - 1 {
        let ioi_prev = (beat_notes[i].0 - beat_notes[i - 1].0) as f32;
        let ioi = (beat_notes[i + 1].0 - beat_notes[i].0) as f32;
        if ioi_prev <= 1e-4 || ioi <= 1e-4 {
            continue; // simultaneous onsets — no melodic context
        }
        let features = crate::expressive::features_from_context(
            theory_notes[i - 1].pitch.midi(),
            theory_notes[i].pitch.midi(),
            theory_notes[i + 1].pitch.midi(),
            ioi_prev,
            ioi,
            (beat_notes[i].0 / total_beats) as f32,
        );
        let (velocity_dev, articulation) = model.predict(&features);
        // Blend at 0.7: full-strength learned accents on top of structural
        // dynamics read slightly overdone against non-piano timbres. The
        // dialect's blend_scale multiplies that base blend.
        let ioi_secs = beat_notes[i + 1].1.start_time - beat_notes[i].1.start_time;
        let note = &mut beat_notes[i].1;
        note.velocity = (note.velocity + 0.7 * blend_scale * velocity_dev).clamp(0.05, 1.0);
        // Symbolic articulation wins over learned articulation: when the
        // COMPOSER wrote a real gap (phrase breath, staccato — sounded
        // length well short of the slot), the model must not legato over
        // it. Otherwise the learned duration/IOI ratio applies — at full
        // blend via the exact historical assignment (bit-identical), at a
        // reduced blend as a lerp from the written duration toward it.
        if ioi_secs > 0.02 && note.duration / ioi_secs >= 0.85 {
            let learned = (ioi_secs * articulation).max(0.02);
            note.duration = if blend_scale >= 1.0 {
                learned
            } else {
                (note.duration + blend_scale * (learned - note.duration)).max(0.02)
            };
        }
    }
}

/// Equal-loudness velocity taper for the top register. The ear's
/// sensitivity peaks around 2-5kHz, so a violin note at A6 played at the
/// same written velocity as one at A4 reads as piercing, not merely high —
/// listeners flagged the register-ceiling notes as "harsh" even when fully
/// sampled. Tapering velocity above E6 (MIDI 84) both quiets the note AND
/// selects the softer recorded dynamic layer in the sample banks, which is
/// where most of the actual timbral relief comes from.
fn equal_loudness_taper(midi: f32) -> f32 {
    1.0 - 0.035 * (midi - 76.0).clamp(0.0, 12.0)
}

/// Metric accent: strong beats speak slightly louder. Beat one carries
/// the bar; in common time the mid-bar gets a half-accent. Multiplicative
/// and small — the phrase arch and learned expression stay authoritative,
/// this only makes the meter FELT ("slight accents on structural notes").
pub(crate) fn metric_accent(onset_beats: f64, meter: f64) -> f32 {
    let pos = onset_beats.rem_euclid(meter.max(1.0));
    if pos < 1e-6 {
        1.06
    } else if meter >= 4.0 && (pos - meter / 2.0).abs() < 1e-6 {
        1.03
    } else {
        1.0
    }
}

/// How long the climax note arrives after its grid position — the player
/// leaning into the emotional peak. Deliberately larger than the humanize
/// jitter's typical magnitude, so the lean reads as intent, not noise.
pub(crate) const CLIMAX_LEAN_SECS: f32 = 0.025;

/// Articulation as a function of position WITHIN the phrase: a statement
/// begins slightly detached and connects as it approaches its cadence —
/// legato is earned, not constant. Returns a duration multiplier.
pub(crate) fn phrase_position_articulation(pos: f32) -> f32 {
    0.88 + 0.12 * pos.clamp(0.0, 1.0)
}

/// Apply the phrase-position articulation curve to the melody's interior
/// notes ("legato vs. detached playing" — the review's point that real
/// players vary HOW a phrase speaks, not just how loud). Phrases are
/// segmented by the score's own `PhraseStart` emphases; each phrase's
/// first and last notes are exempt (they belong to the breath and
/// cadence devices), and deliberately short notes (< 70% of their IOI —
/// staccato, grace notes, written gaps) are never stretched or crushed.
fn apply_phrase_position_articulation(
    beat_notes: &mut [(f64, Note)],
    theory_notes: &[symthaea_music_theory::score::ScoreNote],
) {
    if theory_notes.len() != beat_notes.len() || beat_notes.len() < 4 {
        return;
    }
    // Phrase boundaries: [start, end) index ranges split at PhraseStart.
    let mut bounds: Vec<(usize, usize)> = Vec::new();
    let mut seg_start = 0usize;
    for (i, n) in theory_notes.iter().enumerate().skip(1) {
        if n.emphasis == Emphasis::PhraseStart {
            bounds.push((seg_start, i));
            seg_start = i;
        }
    }
    bounds.push((seg_start, theory_notes.len()));
    for &(lo, hi) in &bounds {
        let len = hi - lo;
        if len < 4 {
            continue;
        }
        for i in (lo + 1)..(hi - 1) {
            let ioi = beat_notes[i + 1].1.start_time - beat_notes[i].1.start_time;
            if ioi <= 0.02 {
                continue;
            }
            let ratio = beat_notes[i].1.duration / ioi;
            if ratio < 0.7 {
                continue; // written short — not ours to legato over
            }
            let pos = (i - lo) as f32 / (len - 1) as f32;
            beat_notes[i].1.duration =
                (beat_notes[i].1.duration * phrase_position_articulation(pos)).max(0.02);
        }
    }
}

/// Stage position for each ensemble role. Every theory voice used to render
/// at `pan: 0.0` — the whole mix was effectively mono, with melody, harmony,
/// and bass fighting for the same center image (the only width came from the
/// reverb tail). Chamber-ensemble convention: lead slightly right of center,
/// accompaniment left, bass anchored center (low frequencies don't localize
/// well and keep the mix balanced on any playback system).
/// `render_arrangement` applies these through an equal-power pan law
/// (`synth.rs`), so moving a voice off-center does not change its loudness.
fn pan_for_role(role: VoiceRole) -> f32 {
    match role {
        VoiceRole::Lead => 0.25,
        VoiceRole::Harmony => -0.4,
        VoiceRole::Bass => 0.0,
        _ => 0.0,
    }
}

/// An inserted rubato micro-pause: at beat `at`, hold for `delay` seconds.
/// A shared list keyed by beat keeps every voice on ONE timeline, so the
/// expressive stretches never desynchronize melody, harmony, and bass.
struct Rubato {
    events: Vec<(f64, f64)>, // (beat, extra seconds inserted from here on)
    spb: f64,                // seconds per beat (60 / tempo)
}

impl Rubato {
    /// Build the rubato map from the melody's structural emphases, with
    /// the classical (identity) dialect — the historical behavior.
    fn from_score(score: &TheoryScore) -> Self {
        Self::from_score_with_dialect(score, PerformanceDialect::ClassicalRubato)
    }

    /// Build the rubato map from the melody's structural emphases, scaled
    /// by the performance dialect. With [`PerformanceDialect::ClassicalRubato`]
    /// every scale is exactly 1.0, so the events are bit-identical to the
    /// pre-dialect map.
    fn from_score_with_dialect(score: &TheoryScore, dialect: PerformanceDialect) -> Self {
        let spb = 60.0 / score.tempo_bpm as f64;
        let mut events = Vec::new();
        let melody = score.voice(TheoryRole::Melody);
        let last_idx = melody.len().saturating_sub(1);
        for (i, n) in melody.iter().enumerate() {
            let onset = n.onset.beats();
            let end = (n.onset + n.duration).beats();
            match n.emphasis {
                // A breath BEFORE a new phrase begins.
                Emphasis::PhraseStart => events.push((onset, 0.10 * spb * dialect.breath_scale())),
                // Dwell ON the climax (agogic accent) — pause just after it.
                Emphasis::Climax => events.push((end, 0.18 * spb * dialect.climax_scale())),
                // Ritardando INTO a cadence; the final cadence relaxes more.
                Emphasis::Cadential => {
                    let amount = if i == last_idx { 0.9 } else { 0.35 };
                    events.push((end, amount * spb * dialect.cadence_scale()));
                }
                Emphasis::Normal => {}
            }
        }
        events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Rubato { events, spb }
    }

    /// Map a beat position to seconds, adding every rubato pause inserted at or
    /// before it. Monotonic in `beat`, identical for all voices.
    fn seconds(&self, beat: f64) -> f32 {
        let base = beat * self.spb;
        let extra: f64 = self
            .events
            .iter()
            .take_while(|(b, _)| *b <= beat + 1e-9)
            .map(|(_, d)| d)
            .sum();
        (base + extra) as f32
    }
}

/// Realize a symbolic score into a muse [`Composition`], using
/// [`Style::Classical`]'s ensemble (string trio) — see [`realize_styled`]
/// for a genre-specific ensemble.
///
/// `state` supplies muse's timbre/emotion parameters (used only as a
/// fallback for any voice with no instrument, and for reverb/dynamics); the
/// *notes themselves* come entirely from the score, and each voice's
/// *timbre* comes from its assigned instrument.
pub fn realize(score: &TheoryScore, state: &MusicalState, sample_rate: u32) -> Composition {
    let seed = variety_seed_from_score(score);
    realize_styled(score, Style::Classical, seed, state, sample_rate)
}

/// Like [`realize`], but every voice's instrument comes from `style`'s
/// ensemble (see [`instruments_for`]), picked by `seed` from a small pool
/// instead of one fixed tuple.
pub fn realize_styled(
    score: &TheoryScore,
    style: Style,
    seed: u64,
    state: &MusicalState,
    sample_rate: u32,
) -> Composition {
    let spec = style.spec();
    let (melody, harmony, bass) = instruments_for(style, seed);
    let counter = contrast_counter(melody, bass);
    realize_core(
        score,
        (melody, harmony, bass, counter),
        &spec,
        seed,
        state,
        sample_rate,
        None,
    )
}

/// Compose (via music theory) and return ONLY the performed melody voice's
/// notes — expressive timing (rubato, swing, dynamic accents, the learned
/// articulation model), but no harmony/bass/counter-melody, and no audio
/// rendering.
///
/// Built for [`crate::singing_bridge::sing`], which needs a single vocal
/// line, not a full arrangement: [`compose`](crate::compose)'s own
/// `Composition.notes` mixes the melody with bass + harmony-pad notes into
/// one flat `Vec<Note>` (see `generate_chord_accompaniment`), so feeding
/// that straight into a lyric-per-note binder would try to sing the
/// accompaniment too. This reuses the exact same performed-melody voice
/// [`realize_styled`] mixes into its audio — same rubato, same dynamic
/// accents, same climax lean — just without the other three voices or the
/// final render. See `SYMTHAEA_SINGING_PLAN_2026-07-18.md` Phase 1.
pub fn compose_and_perform_melody(
    intent: &MusicalIntent,
    style: Style,
    state: &MusicalState,
) -> Vec<Note> {
    let score = symthaea_music_theory::compose_styled(intent, style);
    let spec = style.spec();
    let (melody, harmony, bass) = instruments_for(style, intent.seed);
    let counter = contrast_counter(melody, bass);
    performance_voices(&score, (melody, harmony, bass, counter), &spec, state)
        .into_iter()
        .find(|pv| pv.name == "Melody")
        .map(|pv| pv.notes)
        .unwrap_or_default()
}

/// A performed voice as the outside world may see it — the notes exactly
/// as they will SOUND (swing∘rubato timing, expression, humanize, taper
/// all baked in). Powers the Studio's piano-roll; the internal
/// [`PerformanceVoice`] stays crate-private.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PerformedVoice {
    pub name: String,
    pub instrument: String,
    pub notes: Vec<Note>,
}

/// The full performed rendition of a score under a spec: every voice
/// (including the climax doubling), instruments resolved the same way the
/// audio renderer resolves them.
pub fn perform_with_spec(
    score: &TheoryScore,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &MusicalState,
) -> Vec<PerformedVoice> {
    let ensemble = resolve_spec_ensemble(spec, seed);
    performance_voices(score, ensemble, spec, state)
        .into_iter()
        .map(|pv| PerformedVoice {
            name: pv.name.to_string(),
            instrument: format!("{:?}", pv.instrument),
            notes: pv.notes,
        })
        .collect()
}

/// Resolve a spec's instrument names into the four performed voices
/// (melody, harmony, bass, counter). Unknown names fall back LOUDLY
/// (stderr); an unset `counter_instrument` gets the automatic timbral
/// contrast (see [`contrast_counter`]). Shared by the audio path and
/// MIDI export so both agree on who plays what.
pub(crate) fn resolve_spec_ensemble(
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
) -> (Instrument, Instrument, Instrument, Instrument) {
    let names = spec.ensemble(seed);
    let fallback = [Instrument::Violin, Instrument::Piano, Instrument::Cello];
    let mut resolved = [Instrument::Violin; 3];
    for (i, name) in names.iter().enumerate() {
        resolved[i] = Instrument::from_name(name).unwrap_or_else(|| {
            eprintln!(
                "[muse::theory_realize] unknown instrument name {name:?} in spec                  {:?} — falling back to {:?}",
                spec.name, fallback[i]
            );
            fallback[i]
        });
    }
    let counter = match &spec.counter_instrument {
        Some(name) => Instrument::from_name(name).unwrap_or_else(|| {
            let auto = contrast_counter(resolved[0], resolved[2]);
            eprintln!(
                "[muse::theory_realize] unknown counter_instrument {name:?} in spec {:?} — falling back to {auto:?}",
                spec.name
            );
            auto
        }),
        None => contrast_counter(resolved[0], resolved[2]),
    };
    (resolved[0], resolved[1], resolved[2], counter)
}

/// Realize a Score with a spec-authored ensemble and drum policy — the
/// user-controlled counterpart of [`realize_styled`]. Instrument names
/// resolve via [`Instrument::from_name`]; unknown names fall back LOUDLY
/// (stderr) to the Classical trio's member for that slot.
pub fn realize_with_spec(
    score: &TheoryScore,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &MusicalState,
    sample_rate: u32,
) -> Composition {
    realize_core(
        score,
        resolve_spec_ensemble(spec, seed),
        spec,
        seed,
        state,
        sample_rate,
        None,
    )
}

/// [`realize_with_spec`] with an explicit grammar-level [`symthaea_music_theory::
/// PerformanceDialect`] override — the controlled-comparison escape hatch a
/// listening study needs: e.g. force every arm to `ProcessExact` (neutral
/// performance) so timing/expression differences don't confound a
/// structural-recognition question, or force each arm to its own style's
/// dialect deliberately. Maps down to this crate's own smaller
/// [`PerformanceDialect`] via [`map_grammar_dialect`] before reaching the
/// render internals.
pub fn realize_with_spec_and_dialect(
    score: &TheoryScore,
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &MusicalState,
    sample_rate: u32,
    dialect: symthaea_music_theory::PerformanceDialect,
) -> Composition {
    realize_core(
        score,
        resolve_spec_ensemble(spec, seed),
        spec,
        seed,
        state,
        sample_rate,
        Some(map_grammar_dialect(dialect)),
    )
}

/// Map the grammar-level 8-variant [`symthaea_music_theory::
/// PerformanceDialect`] (declared per [`crate::style::Style::
/// grammar_profile`]/[`symthaea_music_theory::grammar::GrammarProfile`])
/// down to this crate's own 5-variant rendering-internals enum. The two
/// are deliberately separate types (this crate's is what `Rubato`/
/// `performance_voices_with_dialect` actually consume; the grammar crate's
/// is a higher-level per-style-family label) — `DramaticTiming`/
/// `ChoralBlend` have no closer existing analog than the general classical
/// rubato feel, disclosed here rather than silently picked.
pub(crate) fn map_grammar_dialect(
    d: symthaea_music_theory::PerformanceDialect,
) -> PerformanceDialect {
    use symthaea_music_theory::PerformanceDialect as G;
    match d {
        G::ClassicalRubato => PerformanceDialect::ClassicalRubato,
        G::DanceLocked => PerformanceDialect::DanceLocked,
        G::JazzLaidBack => PerformanceDialect::JazzLaidBack,
        G::FolkLift => PerformanceDialect::FolkLift,
        // Matches dialect_for_spec's own drone -> ProcessExact precedent.
        G::ProcessExact | G::DroneElastic => PerformanceDialect::ProcessExact,
        G::DramaticTiming | G::ChoralBlend => PerformanceDialect::ClassicalRubato,
    }
}

/// Compose from a user spec AND realize it — the full user-owned pipeline.
/// Validate the spec first when it came from user input.
pub fn compose_and_realize_spec(
    intent: &MusicalIntent,
    spec: &symthaea_music_theory::CompositionSpec,
    state: &MusicalState,
    sample_rate: u32,
) -> Composition {
    let score = symthaea_music_theory::compose_with_spec(intent, spec);
    realize_with_spec(&score, spec, intent.seed, state, sample_rate)
}

/// One fully performed voice: every timing/velocity transformation the
/// renderer applies (swing∘rubato timeline, learned expression, strum,
/// humanize, equal-loudness taper) already baked into its notes. This is
/// the SHARED performance layer: the audio path turns these into
/// [`Voice`]s, and MIDI export writes them directly — so an exported
/// `.mid` carries the same shuffle and phrasing you hear in the WAV
/// (a listening review found the MIDI was landing straight-grid while
/// the audio swung).
pub(crate) struct PerformanceVoice {
    pub(crate) name: &'static str,
    pub(crate) role: VoiceRole,
    pub(crate) volume: f32,
    pub(crate) pan: f32,
    pub(crate) instrument: Instrument,
    pub(crate) notes: Vec<Note>,
}

/// Build every performed voice for a score: the four theory roles, the
/// climax doubling voice, and (when the spec's `return_color` is on and a
/// counter exists) the return-color sparkle. Empty voices (e.g. no
/// counter-melody in this form) are skipped. Swing, return-color, and the
/// [`PerformanceDialect`] all come from the spec (see [`dialect_for_spec`]).
pub(crate) fn performance_voices(
    score: &TheoryScore,
    ensemble: (Instrument, Instrument, Instrument, Instrument),
    spec: &symthaea_music_theory::CompositionSpec,
    state: &MusicalState,
) -> Vec<PerformanceVoice> {
    performance_voices_with_dialect(
        score,
        ensemble,
        spec.texture.swing as f64,
        state,
        spec.texture.return_color,
        dialect_for_spec(spec),
    )
}

/// [`performance_voices`] with the dialect made explicit — the testable
/// seam (comparative tests force two dialects onto one score) and the
/// escape hatch for callers without a full spec.
pub(crate) fn performance_voices_with_dialect(
    score: &TheoryScore,
    ensemble: (Instrument, Instrument, Instrument, Instrument),
    swing: f64,
    state: &MusicalState,
    return_color: bool,
    dialect: PerformanceDialect,
) -> Vec<PerformanceVoice> {
    let timeline = Timeline {
        rubato: Rubato::from_score_with_dialect(score, dialect),
        swing,
    };
    let (melody_instrument, harmony_instrument, bass_instrument, counter_instrument) = ensemble;

    // Convert each theory voice into a performed voice of muse Notes. The
    // counter-melody (return-section second line) gets its OWN timbre (see
    // `contrast_counter`) in its tenor register — a wind voice answering
    // the strings — panned between harmony and the lead.
    let mut voices: Vec<PerformanceVoice> = Vec::new();
    for (theory_role, name, muse_role, volume, instrument, pan) in [
        (
            TheoryRole::Bass,
            "Bass",
            VoiceRole::Bass,
            0.85,
            bass_instrument,
            None,
        ),
        (
            TheoryRole::Harmony,
            "Harmony",
            VoiceRole::Harmony,
            0.5,
            harmony_instrument,
            None,
        ),
        (
            TheoryRole::CounterMelody,
            "Counter",
            VoiceRole::Harmony,
            0.6,
            counter_instrument,
            Some(-0.15),
        ),
        (
            TheoryRole::Melody,
            "Melody",
            VoiceRole::Lead,
            1.0,
            melody_instrument,
            None,
        ),
    ] {
        let is_melody_role = theory_role == TheoryRole::Melody;
        let mut beat_notes: Vec<(f64, Note)> = score
            .voice(theory_role)
            .iter()
            .map(|n| {
                let onset = n.onset.beats();
                let mut start = timeline.seconds(onset);
                let end = timeline.seconds((n.onset + n.duration).beats());
                let mut velocity =
                    (n.velocity * equal_loudness_taper(n.pitch.midi() as f32)).clamp(0.05, 1.0);
                if is_melody_role {
                    // Dynamic articulation (listening review: "slight
                    // accents on structural notes... gentle delays into
                    // emotional notes"): strong beats speak slightly
                    // louder, and the CLIMAX arrives a breath late — the
                    // player leaning into the note, not the grid hitting
                    // it. The lean eats its own duration (end unchanged),
                    // so nothing downstream shifts.
                    velocity =
                        (velocity * metric_accent(onset, score.meter as f64)).clamp(0.05, 1.0);
                    if n.emphasis == Emphasis::Climax {
                        start += CLIMAX_LEAN_SECS;
                    }
                }
                (
                    onset,
                    Note {
                        frequency: midi_to_hz(n.pitch.midi()),
                        start_time: start,
                        duration: (end - start).max(0.02),
                        velocity,
                    },
                )
            })
            .collect();
        if beat_notes.is_empty() {
            continue;
        }

        // Performance layer, in order:
        // 1. Learned expression (melody only): MAESTRO-trained velocity
        //    accents + articulation, applied BEFORE humanize so the random
        //    jitter textures the learned shape, not the other way round.
        // 2. Strum block chords (needs the bit-exact shared onsets — must
        //    run before jitter breaks that equality).
        // 3. Humanize: Φ-scaled onset jitter + velocity noise + legato for
        //    sustained lines (see `performance::humanize_score_note`).
        //    Chord tones share a seed (see `humanize_seed`), so a chord
        //    lands as one gesture and its strum roll survives intact.
        let is_melody = muse_role == VoiceRole::Lead;
        if is_melody {
            apply_expressive_model(
                &mut beat_notes,
                &score.voice(theory_role),
                dialect.expressive_blend_scale(),
            );
            apply_phrase_position_articulation(&mut beat_notes, &score.voice(theory_role));
        }
        // Strum applies to the CHORDAL accompaniment only — the counter-
        // melody also maps to muse's Harmony role but plays single tones
        // (nothing to roll), and it must jitter as its own player, not
        // locked to the chords: hence tags keyed on the THEORY role.
        if theory_role == TheoryRole::Harmony {
            strum_chords(&mut beat_notes, instrument);
        }
        let voice_tag = match theory_role {
            TheoryRole::Bass => 0,
            TheoryRole::Harmony => 1,
            TheoryRole::CounterMelody => 4,
            _ => 2,
        };
        // Melody articulation now comes from the learned model (interior
        // notes) — the humanizer's flat legato would fight it. Bass and the
        // counter-melody (a sustained singing line) keep the flat legato.
        let legato = theory_role == TheoryRole::Bass || theory_role == TheoryRole::CounterMelody;
        for (onset, note) in beat_notes.iter_mut() {
            crate::performance::humanize_score_note(
                note,
                *onset as f32,
                score.meter as f32,
                humanize_seed(voice_tag, *onset),
                state.consciousness_level,
                legato,
                dialect.jitter_scale(),
            );
        }
        let mut notes: Vec<Note> = beat_notes.into_iter().map(|(_, n)| n).collect();
        // The symbolic score is already overlap-free for every monophonic
        // theory role (`symthaea-music-theory`'s own debug validation gate
        // guarantees it) -- but humanize's per-note onset jitter runs AFTER
        // that guarantee, independently on each note, and can push two
        // once-adjacent notes into a real overlap (an audible double-attack
        // / phase-comb artifact, not just a data-format concern the way
        // MIDI export's overlap hazard was). `Harmony` is excluded: its
        // notes are legitimately simultaneous chord tones, not a single
        // monophonic line.
        if theory_role != TheoryRole::Harmony {
            deoverlap_performance_notes(&mut notes);
        }
        voices.push(PerformanceVoice {
            name,
            role: muse_role,
            volume,
            pan: pan.unwrap_or_else(|| pan_for_role(muse_role)),
            instrument,
            notes,
        });
    }

    // Timbral evolution: at the piece's genuine structural peak (the
    // highest `section_intensity` any melody note carries — see
    // `symthaea_music_theory::form::SectionRole::intensity`), the harmony
    // instrument JOINS the melody line, doubling it quietly underneath for
    // extra weight — a real "the ensemble intensifies at the climax"
    // device, not just a louder version of the same texture. Skipped
    // entirely when every note shares the same intensity (e.g. a single
    // LiveComposer phrase, which has no large-scale arc to peak within).
    let melody_notes_raw = score.voice(TheoryRole::Melody);
    if let Some(max_intensity) = melody_notes_raw
        .iter()
        .map(|n| n.section_intensity)
        .fold(None, |acc: Option<f32>, x| {
            Some(acc.map_or(x, |a| a.max(x)))
        })
    {
        let has_real_arc = melody_notes_raw
            .iter()
            .any(|n| (n.section_intensity - max_intensity).abs() > 1e-6);
        if has_real_arc {
            let mut peak_notes: Vec<Note> = melody_notes_raw
                .iter()
                .filter(|n| (n.section_intensity - max_intensity).abs() < 1e-6)
                .map(|n| {
                    let onset = n.onset.beats();
                    let start = timeline.seconds(onset);
                    let end = timeline.seconds((n.onset + n.duration).beats());
                    // An OCTAVE below the lead — standard climax
                    // orchestration: unison doubling at the melody's top
                    // register stacked a second instrument onto the ear's
                    // most sensitive band (2-5kHz), measured as the
                    // piece's worst HF-harshness windows. The lower
                    // octave adds weight instead of glare and sits
                    // comfortably inside every bank's range.
                    // Velocity capped at 0.5: the doubler should SHIMMER
                    // under the lead, not stab (a review measured doubling
                    // hits at MIDI velocity 100 — "very hot for a bright
                    // plucked instrument"). The cap also selects the SOFT
                    // recorded dynamic layer, which is most of the relief.
                    let mut note = Note {
                        frequency: midi_to_hz(n.pitch.midi().saturating_sub(12)),
                        start_time: start,
                        duration: (end - start).max(0.02),
                        velocity: (n.velocity * 0.7).clamp(0.05, 0.5),
                    };
                    // Its own voice_tag (3): the doubler gets DIFFERENT
                    // jitter from the lead it doubles — two players, not one
                    // sample-locked copy. That slight looseness is exactly
                    // what makes a doubled line sound like an ensemble.
                    crate::performance::humanize_score_note(
                        &mut note,
                        onset as f32,
                        score.meter as f32,
                        humanize_seed(3, onset),
                        state.consciousness_level,
                        true,
                        dialect.jitter_scale(),
                    );
                    // Re-assert the shimmer cap AFTER humanize's velocity
                    // jitter — the ceiling is a guarantee, not a suggestion.
                    note.velocity = note.velocity.min(0.5);
                    note
                })
                .collect();
            deoverlap_performance_notes(&mut peak_notes);
            if !peak_notes.is_empty() {
                voices.push(PerformanceVoice {
                    name: "Doubling",
                    role: VoiceRole::Lead,
                    volume: 0.5, // reinforcement, not a competing line
                    // Opposite side of the lead (which sits at +0.25, see
                    // `pan_for_role`) so the climax doubling genuinely widens
                    // the image instead of stacking onto the same position.
                    pan: -0.2,
                    instrument: harmony_instrument,
                    notes: peak_notes,
                });
            }
        }
    }

    // Return color: at the final return — located by the counter-melody's
    // own window, since that voice exists only there — the harmony's chord
    // tops ring an octave up on a CONTRASTING sparkle timbre, one per bar,
    // soft. The listening review: "the palette stays fairly constant...
    // imagine if the final return introduced a subtle new color — the
    // listener subconsciously feels 'we're somewhere different now.'"
    if return_color {
        let sparkle_voice = {
            let window = voices.iter().find(|v| v.name == "Counter").map(|c| {
                let lo = c
                    .notes
                    .iter()
                    .map(|n| n.start_time)
                    .fold(f32::MAX, f32::min);
                let hi = c
                    .notes
                    .iter()
                    .map(|n| n.start_time + n.duration)
                    .fold(0.0f32, f32::max);
                (lo, hi)
            });
            match (window, voices.iter().find(|v| v.name == "Harmony")) {
                (Some((w0, w1)), Some(harmony)) if w0 < w1 => {
                    let spb = 60.0 / score.tempo_bpm;
                    let min_gap = 0.85 * score.meter as f32 * spb;
                    let mut notes: Vec<Note> = Vec::new();
                    let mut last = f32::MIN;
                    let hn = &harmony.notes;
                    let mut i = 0;
                    while i < hn.len() {
                        // Chord group = notes sharing (strummed) onsets.
                        let s0 = hn[i].start_time;
                        let mut top = hn[i];
                        let mut j = i + 1;
                        while j < hn.len() && (hn[j].start_time - s0).abs() < 0.05 {
                            if hn[j].frequency > top.frequency {
                                top = hn[j];
                            }
                            j += 1;
                        }
                        if s0 >= w0 && s0 < w1 && s0 - last >= min_gap {
                            notes.push(Note {
                                frequency: top.frequency * 2.0,
                                start_time: top.start_time,
                                duration: top.duration.min(2.0 * spb),
                                // Soft and fixed: a shimmer over the return,
                                // never a competing voice.
                                velocity: 0.35,
                            });
                            last = s0;
                        }
                        i = j;
                    }
                    (!notes.is_empty()).then(|| PerformanceVoice {
                        name: "Return Color",
                        role: VoiceRole::Harmony,
                        volume: 0.4,
                        pan: 0.35,
                        instrument: if harmony_instrument == Instrument::Kalimba {
                            Instrument::Marimba
                        } else {
                            Instrument::Kalimba
                        },
                        notes,
                    })
                }
                _ => None,
            }
        };
        if let Some(v) = sparkle_voice {
            voices.push(v);
        }
    }
    voices
}

fn realize_core(
    score: &TheoryScore,
    ensemble: (Instrument, Instrument, Instrument, Instrument),
    spec: &symthaea_music_theory::CompositionSpec,
    seed: u64,
    state: &MusicalState,
    sample_rate: u32,
    dialect_override: Option<PerformanceDialect>,
) -> Composition {
    let dialect = dialect_override.unwrap_or_else(|| dialect_for_spec(spec));
    let swing = spec.texture.swing as f64;
    // Same beat→seconds map as performance_voices builds internally —
    // Rubato::from_score_with_dialect is deterministic, so the drum track
    // and total length stay sample-locked to the performed voices.
    let timeline = Timeline {
        rubato: Rubato::from_score_with_dialect(score, dialect),
        swing,
    };
    let voices: Vec<Voice> = performance_voices_with_dialect(
        score,
        ensemble,
        spec.texture.swing as f64,
        state,
        spec.texture.return_color,
        dialect,
    )
    .into_iter()
    .map(|pv| {
        let (lo, hi) = pv.notes.iter().fold((f32::MAX, 0.0f32), |(lo, hi), n| {
            (lo.min(n.frequency), hi.max(n.frequency))
        });
        Voice {
            role: pv.role,
            notes: pv.notes,
            pitch_range: (lo, hi),
            volume: pv.volume,
            pan: pv.pan,
            instrument: Some(pv.instrument),
        }
    })
    .collect();

    let arrangement = Arrangement { voices };

    // Total length: the last note end plus a reverb tail.
    let total_secs = timeline.seconds(score.total_beats.beats()) + 1.5;
    let total_samples = (total_secs * sample_rate as f32) as usize;

    let config = MuseConfig {
        sample_rate,
        duration_secs: total_secs,
        output_format: crate::OutputFormat::StereoF32,
        // A real room for the chamber ensemble: dry close-mic'd samples
        // read as synthetic ("dry GM violin is brutal" — listening
        // review). Bigger room + higher wet floor than the default
        // streaming path; state still adds its share of wet on top.
        reverb: crate::ReverbConfig {
            room_size: 0.68,
            damping: 0.45,
            width: 0.9,
            wet_floor: 0.2,
        },
        ..Default::default()
    };
    let mut audio =
        crate::synth::render_arrangement(&arrangement, sample_rate, total_samples, state, &config);

    // Style-gated drum track (see `drum_hits`): mixed in dry after the
    // ensemble render (so drums stay punchy, outside the reverb) and before
    // mastering (so the limiter keeps combined peaks safe).
    let drums = drum_hits(score, spec.texture.drums, &timeline, state, seed);
    if let (AudioData::StereoF32(frames), false) = (&mut audio, drums.is_empty()) {
        mix_drums(frames, &drums, sample_rate, state);
    }

    // Master the final mix: corrective 3-band EQ + loudness-normalize to
    // -14 LUFS (streaming standard) + brick-wall limit. `auto_master` is
    // real, tested BS.1770 mastering code (`auto_master.rs`) that already
    // existed in this crate but was never called from ANY generation
    // path before this -- render_arrangement's own normalization/limiter
    // is a simpler RMS-target pass, not a real mastering chain.
    if let AudioData::StereoF32(frames) = &mut audio {
        crate::auto_master::auto_master(
            frames,
            sample_rate,
            &crate::auto_master::MasteringConfig::default(),
        );
    }

    // Flatten notes for the Composition record.
    let notes: Vec<Note> = arrangement
        .voices
        .iter()
        .flat_map(|v| v.notes.iter().copied())
        .collect();

    Composition {
        audio,
        sample_rate,
        notes,
        duration_secs: total_secs,
        section: crate::structure::SectionType::Developmental,
    }
}

/// Compose (via music theory) AND realize (via muse) in one call — the whole
/// new pipeline. `intent` drives the structure; `state` drives the
/// dynamics/reverb; the instrument ensemble is [`Style::Classical`]'s
/// string trio. See [`compose_and_realize_styled`] for other genres.
pub fn compose_and_realize(
    intent: &MusicalIntent,
    state: &MusicalState,
    sample_rate: u32,
) -> Composition {
    compose_and_realize_styled(intent, Style::Classical, state, sample_rate)
}

/// Like [`compose_and_realize`], but `style` drives BOTH the composed
/// structure (meter, tempo, motif bank, progression — via
/// [`symthaea_music_theory::compose_styled`]) and the instrument ensemble
/// (via [`instruments_for`]) — a Waltz sounds like a string trio in 3/4,
/// a Folk piece sounds like flute-over-plucked-strings, etc.
pub fn compose_and_realize_styled(
    intent: &MusicalIntent,
    style: Style,
    state: &MusicalState,
    sample_rate: u32,
) -> Composition {
    let score = symthaea_music_theory::compose_styled(intent, style);
    realize_styled(&score, style, intent.seed, state, sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::pitch::PitchClass;

    fn note(start: f32, duration: f32) -> Note {
        Note {
            frequency: 440.0,
            start_time: start,
            duration,
            velocity: 0.8,
        }
    }

    #[test]
    fn deoverlap_performance_notes_truncates_a_humanize_induced_overlap() {
        // Two once-adjacent notes that now overlap by 0.05s (e.g. humanize
        // jittered the second note's onset earlier) -- the first must be
        // truncated to end exactly where the second begins.
        let mut notes = vec![note(0.0, 0.5), note(0.45, 0.5)];
        deoverlap_performance_notes(&mut notes);
        assert_eq!(notes.len(), 2);
        assert!((notes[0].duration - 0.45).abs() < 1e-6, "{:?}", notes[0]);
        assert!((notes[0].start_time + notes[0].duration - notes[1].start_time).abs() < 1e-6);
    }

    #[test]
    fn deoverlap_performance_notes_drops_an_exact_duplicate_start() {
        let mut notes = vec![note(0.0, 0.5), note(0.0, 0.3)];
        deoverlap_performance_notes(&mut notes);
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!((notes[0].start_time - 0.0).abs() < 1e-6);
    }

    #[test]
    fn deoverlap_performance_notes_leaves_non_overlapping_notes_untouched() {
        let mut notes = vec![note(0.0, 0.5), note(0.6, 0.5)];
        deoverlap_performance_notes(&mut notes);
        assert_eq!(notes.len(), 2);
        assert!((notes[0].start_time - 0.0).abs() < 1e-6);
        assert!((notes[0].duration - 0.5).abs() < 1e-6);
        assert!((notes[1].start_time - 0.6).abs() < 1e-6);
        assert!((notes[1].duration - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dialect_derivation_maps_every_style_preset() {
        use PerformanceDialect as D;
        // The complete v1 derivation table over all 29 style presets —
        // exercised against the REAL specs, so a preset change that moves
        // a style's family markers shows up here, not in a listening test.
        let cases = [
            (Style::Classical, D::ClassicalRubato),
            (Style::Waltz, D::ClassicalRubato),
            // v1 limit, documented on `dialect_for_spec`: plain Folk and
            // ModalFolk carry none of the spec-visible folk habit flags.
            (Style::Folk, D::ClassicalRubato),
            (Style::Cinematic, D::ClassicalRubato),
            (Style::Playful, D::DanceLocked), // full backbeat kit
            (Style::Nocturne, D::ClassicalRubato),
            (Style::March, D::ClassicalRubato), // v1 limit: no clock marker
            (Style::Lullaby, D::ClassicalRubato),
            (Style::ModalFolk, D::ClassicalRubato),
            (Style::Fugue, D::ClassicalRubato),
            (Style::Passacaglia, D::ClassicalRubato),
            (Style::Tango, D::DanceLocked),  // Habanera rhythm cell
            (Style::Celtic, D::FolkLift),    // tonic-fifth drone
            (Style::Blues, D::JazzLaidBack), // Shuffle cell + swung eighths
            (Style::Impressionism, D::ClassicalRubato),
            (Style::SacredChoral, D::ClassicalRubato),
            (Style::Minimalism, D::ProcessExact), // additive process
            (Style::JazzBallad, D::JazzLaidBack), // swing 0.58 + 7th chords
            (Style::BaroqueSuite, D::ClassicalRubato), // straight-swing dance
            (Style::ProgFolk, D::DanceLocked),    // backbeat kit
            (Style::Ambient, D::ProcessExact),    // harmonic stasis
            (Style::Sonata, D::ClassicalRubato),
            (Style::RenaissancePolyphony, D::ClassicalRubato),
            (Style::AfroCuban, D::DanceLocked), // Montuno rhythm cell
            (Style::Flamenco, D::DanceLocked),  // CompasGait rhythm cell
            (Style::BossaNova, D::JazzLaidBack), // BossaComp + 7th chords
            (Style::Opera, D::ClassicalRubato),
            (Style::IrishTraditional, D::FolkLift), // roll ornaments
            (Style::HindustaniInspired, D::ProcessExact), // full drone
        ];
        for (style, expected) in cases {
            assert_eq!(
                dialect_for_spec(&style.spec()),
                expected,
                "{style:?} must derive {expected:?}"
            );
        }
    }

    #[test]
    fn classical_dialect_is_the_exact_identity() {
        // Every scale factor exactly 1.0 — the guarantee that a render
        // deriving ClassicalRubato is bit-identical to pre-dialect output.
        let d = PerformanceDialect::ClassicalRubato;
        assert_eq!(d.breath_scale(), 1.0);
        assert_eq!(d.climax_scale(), 1.0);
        assert_eq!(d.cadence_scale(), 1.0);
        assert_eq!(d.jitter_scale(), 1.0);
        assert_eq!(d.expressive_blend_scale(), 1.0);
        assert_eq!(PerformanceDialect::default(), d);
        // The dialected rubato map degenerates to the plain constructor's
        // map exactly (same events, same seconds-per-beat).
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let plain = Rubato::from_score(&score);
        let dialected = Rubato::from_score_with_dialect(&score, d);
        assert_eq!(plain.events, dialected.events);
        assert_eq!(plain.spb, dialected.spb);
        // And the spec-driven entry point renders note-for-note identical
        // to the explicit ClassicalRubato path for a classical spec.
        let spec = Style::Classical.spec();
        let intent = MusicalIntent::default();
        let score = symthaea_music_theory::compose_with_spec(&intent, &spec);
        let state = MusicalState::default();
        let ensemble = resolve_spec_ensemble(&spec, intent.seed);
        let via_spec = performance_voices(&score, ensemble, &spec, &state);
        let explicit = performance_voices_with_dialect(
            &score,
            ensemble,
            spec.texture.swing as f64,
            &state,
            spec.texture.return_color,
            d,
        );
        assert_eq!(via_spec.len(), explicit.len());
        for (a, b) in via_spec.iter().zip(&explicit) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.notes.len(), b.notes.len(), "{}", a.name);
            for (x, y) in a.notes.iter().zip(&b.notes) {
                assert_eq!(x.start_time, y.start_time);
                assert_eq!(x.duration, y.duration);
                assert_eq!(x.velocity, y.velocity);
            }
        }
    }

    #[test]
    fn dance_locked_hugs_the_straight_grid_tighter_than_classical() {
        // Same score, same ensemble, same state — only the dialect
        // differs. The groove dialect's melody must deviate LESS from the
        // straight metronome grid than the classical one (its rubato is
        // scaled 0.25x and its jitter 0.6x), and the process dialect less
        // still (0.1x / 0.4x).
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let state = MusicalState::default();
        let ensemble = (
            Instrument::Violin,
            Instrument::Piano,
            Instrument::Cello,
            Instrument::Clarinet,
        );
        let spb = 60.0 / score.tempo_bpm;
        let grid_deviation = |dialect: PerformanceDialect| -> f32 {
            let voices =
                performance_voices_with_dialect(&score, ensemble, 0.5, &state, false, dialect);
            let melody = voices.iter().find(|v| v.name == "Melody").unwrap();
            let theory = score.voice(TheoryRole::Melody);
            assert_eq!(theory.len(), melody.notes.len());
            theory
                .iter()
                .zip(&melody.notes)
                .map(|(t, n)| (n.start_time - t.onset.beats() as f32 * spb).abs())
                .sum()
        };
        let classical = grid_deviation(PerformanceDialect::ClassicalRubato);
        let dance = grid_deviation(PerformanceDialect::DanceLocked);
        let process = grid_deviation(PerformanceDialect::ProcessExact);
        assert!(
            dance < classical,
            "DanceLocked ({dance:.3}s) must hug the grid tighter than ClassicalRubato ({classical:.3}s)"
        );
        assert!(
            process < dance,
            "ProcessExact ({process:.3}s) must hug the grid tighter than DanceLocked ({dance:.3}s)"
        );
    }

    #[test]
    fn midi_reference_pitches() {
        assert!((midi_to_hz(69) - 440.0).abs() < 0.01); // A4
        assert!((midi_to_hz(60) - 261.63).abs() < 0.5); // C4
    }

    #[test]
    fn counter_instrument_contrasts_with_lead_and_bass() {
        // The whole point: the counterline must not share the bass timbre.
        for style in [
            Style::Classical,
            Style::Folk,
            Style::Cinematic,
            Style::Playful,
            Style::Waltz,
        ] {
            for seed in 0..6 {
                let (melody, _harmony, bass) = instruments_for(style, seed);
                let counter = contrast_counter(melody, bass);
                assert_ne!(counter, bass, "{style:?} seed {seed}");
                assert_ne!(counter, melody, "{style:?} seed {seed}");
            }
        }
        // A spec's explicit choice wins over the automatic contrast.
        let mut spec = Style::Classical.spec();
        spec.counter_instrument = Some("trumpet".into());
        let (_, _, _, counter) = resolve_spec_ensemble(&spec, 0);
        assert_eq!(counter, Instrument::Trumpet);
    }

    #[test]
    fn the_return_gains_a_new_color() {
        // With defaults, the final return carries a sparse sparkle voice
        // (chord tops an octave up, contrasting timbre) confined to the
        // counter's window; with the flag off, it doesn't exist.
        let spec = symthaea_music_theory::Style::Classical.spec();
        let mut plain = spec.clone();
        plain.texture.return_color = false;
        let intent = MusicalIntent::default();
        let score = symthaea_music_theory::compose_with_spec(&intent, &spec);
        let state = MusicalState::default();
        let voices = perform_with_spec(&score, &spec, intent.seed, &state);
        let color = voices
            .iter()
            .find(|v| v.name == "Return Color")
            .expect("defaults must add the return color");
        assert!(!color.notes.is_empty());
        let counter = voices.iter().find(|v| v.name == "Counter").unwrap();
        let (w0, w1) = (
            counter
                .notes
                .iter()
                .map(|n| n.start_time)
                .fold(f32::MAX, f32::min),
            counter
                .notes
                .iter()
                .map(|n| n.start_time + n.duration)
                .fold(0.0f32, f32::max),
        );
        for n in &color.notes {
            assert!(
                n.start_time >= w0 - 1e-3 && n.start_time < w1 + 1e-3,
                "sparkle must live in the return window"
            );
            assert!(n.velocity <= 0.4, "a shimmer, not a competing voice");
        }
        // Contrasting timbre, and sparse — roughly one per bar, never the
        // full harmony stream.
        assert_ne!(color.instrument, counter.instrument);
        let harmony = voices.iter().find(|v| v.name == "Harmony").unwrap();
        assert!(color.notes.len() * 4 < harmony.notes.len());
        // Flag off → gone.
        let without = perform_with_spec(&score, &plain, intent.seed, &state);
        assert!(without.iter().all(|v| v.name != "Return Color"));
    }

    #[test]
    fn metric_accent_marks_the_strong_beats() {
        // Beat one carries the bar; common time gets a mid-bar half-accent;
        // everything else is untouched. 3/4 has no mid-bar accent.
        assert!(metric_accent(0.0, 4.0) > metric_accent(2.0, 4.0));
        assert!(metric_accent(2.0, 4.0) > metric_accent(1.0, 4.0));
        assert_eq!(metric_accent(1.0, 4.0), 1.0);
        assert_eq!(metric_accent(3.5, 4.0), 1.0);
        assert_eq!(metric_accent(4.0, 4.0), metric_accent(0.0, 4.0)); // wraps
        assert_eq!(metric_accent(1.5, 3.0), 1.0); // no mid-bar accent in 3/4
        assert!(metric_accent(3.0, 3.0) > 1.0);
    }

    #[test]
    fn phrase_articulation_earns_its_legato() {
        // Detached at the statement, connected into the cadence — strictly
        // rising, bounded, never stretching past the written length.
        let mut prev = 0.0;
        for i in 0..=10 {
            let f = phrase_position_articulation(i as f32 / 10.0);
            assert!(f > prev, "must rise monotonically");
            assert!((0.88..=1.0).contains(&f));
            prev = f;
        }
    }

    #[test]
    fn the_climax_leans_in_by_exactly_the_written_breath() {
        // Two IDENTICAL hand-built scores, except one marks its middle
        // note as the Climax. Humanize jitter is seeded on (voice_tag,
        // onset) — identical in both runs — so the performed onset
        // difference isolates the lean exactly.
        use symthaea_music_theory::pitch::Pitch;
        use symthaea_music_theory::rhythm::Duration as TDur;
        use symthaea_music_theory::score::ScoreNote;
        let build = |climax: bool| {
            let key =
                symthaea_music_theory::Key::major(symthaea_music_theory::pitch::PitchClass::C);
            let mut score = symthaea_music_theory::Score::new(key, 90.0, 4);
            for (i, (midi, beat)) in [(72u8, 0i64), (76, 2), (74, 4), (72, 6)].iter().enumerate() {
                score.push(ScoreNote {
                    pitch: Pitch::from_midi(*midi),
                    onset: TDur::new(*beat, 1),
                    duration: TDur::new(2, 1),
                    velocity: 0.6,
                    role: symthaea_music_theory::score::VoiceRole::Melody,
                    emphasis: if climax && i == 1 {
                        Emphasis::Climax
                    } else {
                        Emphasis::Normal
                    },
                    section_intensity: 1.0,
                });
            }
            let voices = performance_voices_with_dialect(
                &score,
                (
                    Instrument::Violin,
                    Instrument::Harp,
                    Instrument::Cello,
                    Instrument::Clarinet,
                ),
                0.5,
                &MusicalState::default(),
                false,
                PerformanceDialect::ClassicalRubato,
            );
            voices
                .into_iter()
                .find(|v| v.name == "Melody")
                .expect("melody voice")
                .notes
        };
        let plain = build(false);
        let leaned = build(true);
        assert_eq!(plain.len(), leaned.len());
        let diff = leaned[1].start_time - plain[1].start_time;
        assert!(
            (diff - CLIMAX_LEAN_SECS).abs() < 1e-5,
            "the climax must lean in by exactly {CLIMAX_LEAN_SECS}s, got {diff}"
        );
        // The note BEFORE the climax is untouched...
        assert!((leaned[0].start_time - plain[0].start_time).abs() < 1e-6);
        // ...and the notes AFTER it shift by the agogic rubato dwell that
        // marking a climax also creates (0.18 beats-worth at 90 BPM =
        // 0.12s) — the dwell is the point of the emphasis, not a leak.
        let dwell = 0.18 * (60.0 / 90.0) as f32;
        for i in [2usize, 3] {
            let shift = leaned[i].start_time - plain[i].start_time;
            assert!(
                (shift - dwell).abs() < 1e-4,
                "note {i}: expected the {dwell}s agogic dwell, got {shift}"
            );
        }
    }

    #[test]
    fn melody_articulation_connects_toward_cadences_across_a_real_piece() {
        // Across all phrases of a composed piece, interior notes in the
        // BACK half of their phrase sound longer (relative to their IOI)
        // than those in the front half — legato is earned. Statistical
        // over ~a dozen phrases, deterministic per seed.
        let mut spec = symthaea_music_theory::Style::Classical.spec();
        spec.texture.damage = 0.0; // no holes/wait-tones muddying the count
        let intent = MusicalIntent::default();
        let score = symthaea_music_theory::compose_with_spec(&intent, &spec);
        let voices = perform_with_spec(&score, &spec, intent.seed, &MusicalState::default());
        let melody = &voices.iter().find(|v| v.name == "Melody").unwrap().notes;
        let theory = score.voice(symthaea_music_theory::score::VoiceRole::Melody);
        assert_eq!(theory.len(), melody.len());
        // Phrase bounds from the score's own PhraseStart markers.
        let mut bounds = Vec::new();
        let mut lo = 0usize;
        for (i, n) in theory.iter().enumerate().skip(1) {
            if n.emphasis == Emphasis::PhraseStart {
                bounds.push((lo, i));
                lo = i;
            }
        }
        bounds.push((lo, theory.len()));
        let (mut front, mut back) = (Vec::new(), Vec::new());
        for &(a, b) in &bounds {
            let len = b - a;
            if len < 6 {
                continue;
            }
            for i in (a + 1)..(b - 1) {
                let ioi = melody[i + 1].start_time - melody[i].start_time;
                if ioi <= 0.02 {
                    continue;
                }
                let ratio = (melody[i].duration / ioi).min(1.5);
                let pos = (i - a) as f32 / (len - 1) as f32;
                if pos < 0.5 {
                    front.push(ratio);
                } else {
                    back.push(ratio);
                }
            }
        }
        assert!(front.len() > 8 && back.len() > 8, "need real phrase data");
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        assert!(
            mean(&back) > mean(&front),
            "cadence approach must be more connected: front {} back {}",
            mean(&front),
            mean(&back)
        );
    }

    #[test]
    fn equal_loudness_taper_quiets_only_the_top_register() {
        // Untouched through E5 — the working register keeps its dynamics.
        // (Knee lowered from E6 and slope steepened after a timbre review:
        // "avoid naked violin above roughly E5-A5... soften velocity
        // aggressively".)
        assert_eq!(equal_loudness_taper(60.0), 1.0);
        assert_eq!(equal_loudness_taper(76.0), 1.0);
        // Monotonically decreasing above, aggressively quieter at the
        // register-fold ceiling (E6 = MIDI 88).
        let mut prev = 1.0;
        for midi in 77..=92 {
            let t = equal_loudness_taper(midi as f32);
            assert!(t <= prev, "taper must not increase with pitch");
            prev = t;
        }
        let at_ceiling = equal_loudness_taper(88.0);
        assert!(at_ceiling < 0.75 && at_ceiling > 0.4, "got {at_ceiling}");
        // Clamped: never collapses to silence for out-of-range input.
        assert!(equal_loudness_taper(120.0) >= 0.4);
    }

    #[test]
    fn doubling_voice_shimmers_instead_of_stabbing() {
        // Timbre review measured doubling hits at MIDI velocity 100 —
        // far too hot for a bright plucked instrument over the lead. The
        // doubler's velocity is capped at 0.5 (≈ MIDI 64), which also
        // selects the SOFT recorded dynamic layer.
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let voices = performance_voices_with_dialect(
            &score,
            (
                Instrument::Violin,
                Instrument::Harp,
                Instrument::Cello,
                Instrument::Clarinet,
            ),
            0.5,
            &MusicalState::default(),
            false,
            PerformanceDialect::ClassicalRubato,
        );
        let doubling = voices
            .iter()
            .find(|v| v.name == "Doubling")
            .expect("a full piece has a climax doubling voice");
        assert!(!doubling.notes.is_empty());
        for n in &doubling.notes {
            assert!(
                n.velocity <= 0.5 + 1e-6,
                "doubling velocity {} must shimmer, not stab",
                n.velocity
            );
        }
    }

    #[test]
    fn rubato_is_monotonic_and_lengthens_total() {
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let r = Rubato::from_score(&score);
        // Monotonic in beat.
        let mut prev = -1.0;
        let mut b = 0.0;
        while b <= score.total_beats.beats() {
            let s = r.seconds(b) as f64;
            assert!(s >= prev, "rubato time must be monotonic");
            prev = s;
            b += 0.5;
        }
        // Expressive stretching makes the realized piece at least as long as
        // the strict-tempo length (cadences/climax add held time).
        let strict = score.total_beats.beats() * (60.0 / score.tempo_bpm as f64);
        assert!(r.seconds(score.total_beats.beats()) as f64 >= strict);
    }

    #[test]
    fn realize_adds_a_doubling_voice_at_the_pieces_peak_section() {
        // Any real compose() output has a genuine intensity spread (even
        // ternary: A=0.85, B=1.0, ReturnA=0.95) -- so the peak-section
        // doubling voice must add EXTRA notes beyond what a naive 1:1
        // theory-note -> muse-note mapping would produce.
        // Asserted on the named voice directly (the composition-level
        // note-count equation broke once the Return Color sparkle voice
        // started adding its own notes — a count can't name whose they
        // are; the voice name can).
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let voices = performance_voices_with_dialect(
            &score,
            (
                Instrument::Violin,
                Instrument::Piano,
                Instrument::Cello,
                Instrument::Clarinet,
            ),
            0.5,
            &MusicalState::default(),
            false,
            PerformanceDialect::ClassicalRubato,
        );
        let peak_notes_in_melody = score
            .voice(symthaea_music_theory::score::VoiceRole::Melody)
            .iter()
            .filter(|n| {
                let max = score
                    .voice(symthaea_music_theory::score::VoiceRole::Melody)
                    .iter()
                    .map(|n| n.section_intensity)
                    .fold(f32::MIN, f32::max);
                (n.section_intensity - max).abs() < 1e-6
            })
            .count();
        assert!(
            peak_notes_in_melody > 0,
            "test setup: there must be a peak section"
        );
        let doubling = voices
            .iter()
            .find(|v| v.name == "Doubling")
            .expect("a real piece gets its climax doubling");
        assert_eq!(
            doubling.notes.len(),
            peak_notes_in_melody,
            "the doubling voice should carry exactly the peak section's melody notes"
        );
    }

    #[test]
    fn roles_have_distinct_stage_positions() {
        // Lead right of center, harmony left, bass anchored center — and
        // lead/harmony genuinely on OPPOSITE sides, not just non-zero.
        assert!(pan_for_role(VoiceRole::Lead) > 0.0);
        assert!(pan_for_role(VoiceRole::Harmony) < 0.0);
        assert_eq!(pan_for_role(VoiceRole::Bass), 0.0);
        // Keep positions moderate: hard-panning a chamber voice sounds
        // artificial and breaks mono-compatibility.
        for role in [VoiceRole::Lead, VoiceRole::Harmony, VoiceRole::Bass] {
            assert!(pan_for_role(role).abs() <= 0.5);
        }
    }

    #[test]
    fn expressive_model_shapes_interior_melody_notes_only() {
        use symthaea_music_theory::pitch::Pitch;
        use symthaea_music_theory::rhythm::Duration as TDur;
        use symthaea_music_theory::score::{Emphasis as TEmph, ScoreNote};
        let mk_theory = |midi: u8, beat: i64| ScoreNote {
            pitch: Pitch::from_midi(midi),
            onset: TDur::new(beat, 1),
            duration: TDur::new(1, 1),
            velocity: 0.6,
            role: symthaea_music_theory::score::VoiceRole::Melody,
            emphasis: TEmph::Normal,
            section_intensity: 1.0,
        };
        let theory: Vec<ScoreNote> = [(60, 0), (64, 1), (67, 2), (64, 3), (60, 4)]
            .iter()
            .map(|&(m, b)| mk_theory(m, b))
            .collect();
        let mut beat_notes: Vec<(f64, Note)> = theory
            .iter()
            .map(|n| {
                (
                    n.onset.beats(),
                    Note {
                        frequency: midi_to_hz(n.pitch.midi()),
                        start_time: n.onset.beats() as f32 * 0.5, // 120 bpm
                        duration: 0.5,
                        velocity: 0.6,
                    },
                )
            })
            .collect();
        let before: Vec<Note> = beat_notes.iter().map(|(_, n)| *n).collect();
        apply_expressive_model(&mut beat_notes, &theory, 1.0);
        // Edges untouched (no melodic context on one side).
        assert_eq!(beat_notes[0].1.velocity, before[0].velocity);
        assert_eq!(beat_notes[4].1.duration, before[4].duration);
        // At least one interior note genuinely shaped, and all stay sane.
        let interior_changed = (1..4).any(|i| {
            beat_notes[i].1.velocity != before[i].velocity
                || beat_notes[i].1.duration != before[i].duration
        });
        assert!(interior_changed, "trained model must actually do something");
        for (_, n) in &beat_notes {
            assert!((0.05..=1.0).contains(&n.velocity));
            assert!(
                n.duration >= 0.02 && n.duration <= 0.8,
                "duration {} out of the ioi*1.3 envelope",
                n.duration
            );
        }
    }

    #[test]
    fn swing_warps_offbeats_and_nothing_else() {
        // Straight = identity everywhere.
        for b in [0.0, 0.25, 0.5, 1.0, 3.75] {
            assert_eq!(swing_beat(b, 0.5), b);
        }
        // Triplet swing: integers fixed, the off-beat eighth lands at 2/3.
        let r = 2.0 / 3.0;
        assert_eq!(swing_beat(4.0, r), 4.0);
        assert!((swing_beat(4.5, r) - (4.0 + r)).abs() < 1e-9);
        assert!((swing_beat(0.25, r) - r / 2.0).abs() < 1e-9);
        // Strictly monotonic (no time travel).
        let mut prev = -1.0;
        let mut b = 0.0;
        while b < 4.0 {
            let w = swing_beat(b, 0.7);
            assert!(w > prev, "swing must be strictly monotonic");
            prev = w;
            b += 0.01;
        }
    }

    #[test]
    fn swung_spec_renders_offbeat_notes_later_than_straight() {
        // Same spec twice, only swing differs: every note ON an off-beat
        // eighth must start strictly later in the swung render, and every
        // barline note must start at the same time.
        let mut spec = Style::Playful.spec(); // busy eighth-note motifs
        spec.texture.drums = symthaea_music_theory::DrumPolicy::None;
        // Damage OFF: the damage PLANNER deliberately reads swing (a
        // straight piece draws more timing damage), so a straight-vs-swung
        // comparison must hold the composition fixed to isolate the
        // performance-layer time map this test is about.
        spec.texture.damage = 0.0;
        // Dialect HELD CONSTANT: `dialect_for_spec` also reads swing (a
        // swung spec is a jazz marker), so without a stronger marker the
        // two arms would derive DIFFERENT performers (ClassicalRubato vs
        // JazzLaidBack) whose differing jitter scales can legitimately
        // move a note earlier. A Shuffle rhythm cell pins BOTH arms to
        // JazzLaidBack, restoring the all-else-equal comparison.
        spec.accompaniment_pool = vec![symthaea_music_theory::Accompaniment::Shuffle];
        // Return Color OFF: the sparkle voice groups harmony notes into
        // chord-onsets by an ABSOLUTE-SECONDS threshold (`0.05s`) and
        // gates inclusion on a real-time `min_gap` — both read the
        // performed timeline, so swing can legitimately shift which
        // groups qualify and change the sparkle voice's NOTE COUNT. A
        // fourth swing-sensitive parameter beyond damage/dialect that
        // this all-else-equal comparison must also pin, or the total
        // flattened note count (this test's own invariant) can differ
        // between arms for a reason that has nothing to do with the
        // off-beat timing claim being tested.
        spec.texture.return_color = false;
        let intent = MusicalIntent {
            arousal: 0.9, // busy tier → eighth notes
            ..Default::default()
        };
        let state = MusicalState::default();
        assert_eq!(dialect_for_spec(&spec), PerformanceDialect::JazzLaidBack);
        let straight = compose_and_realize_spec(&intent, &spec, &state, 44100);
        spec.texture.swing = 2.0 / 3.0;
        assert_eq!(dialect_for_spec(&spec), PerformanceDialect::JazzLaidBack);
        let swung = compose_and_realize_spec(&intent, &spec, &state, 44100);
        assert_eq!(straight.notes.len(), swung.notes.len());
        let mut checked_offbeat = 0;
        for (a, b) in straight.notes.iter().zip(&swung.notes) {
            if a.start_time > b.start_time + 1e-6 {
                panic!("swing must never move a note EARLIER");
            }
            if b.start_time > a.start_time + 1e-4 {
                checked_offbeat += 1;
            }
        }
        assert!(
            checked_offbeat > 10,
            "a busy piece must have many swung off-beats: {checked_offbeat}"
        );
    }

    #[test]
    fn chamber_styles_have_no_drums_and_rhythmic_styles_do() {
        let state = MusicalState::default();
        for (style, expect_drums) in [
            (Style::Classical, false),
            (Style::Waltz, false),
            (Style::Folk, true),
            (Style::Cinematic, true),
            (Style::Playful, true),
        ] {
            let score = symthaea_music_theory::compose_styled(&MusicalIntent::default(), style);
            let timeline = Timeline {
                rubato: Rubato::from_score(&score),
                swing: 0.5,
            };
            let hits = drum_hits(&score, style.spec().texture.drums, &timeline, &state, 0);
            assert_eq!(
                !hits.is_empty(),
                expect_drums,
                "{style:?}: drums presence mismatch"
            );
        }
    }

    #[test]
    fn playful_backbeat_has_the_right_anatomy() {
        let score =
            symthaea_music_theory::compose_styled(&MusicalIntent::default(), Style::Playful);
        let timeline = Timeline {
            rubato: Rubato::from_score(&score),
            swing: 0.5,
        };
        let state = MusicalState::default();
        let hits = drum_hits(
            &score,
            Style::Playful.spec().texture.drums,
            &timeline,
            &state,
            0,
        );
        let meter = score.meter as f64;
        let bars = (score.total_beats.beats() / meter).ceil() as usize;
        let count = |d: DrumType| hits.iter().filter(|h| h.drum == d).count();
        assert_eq!(count(DrumType::Kick), bars * 2, "two kicks per bar");
        assert_eq!(count(DrumType::Snare), bars * 2, "backbeat snares");
        assert_eq!(count(DrumType::HiHat), bars * 8, "eighth-note hats");
        // Drums live on the rubato timeline: the bar-2 kick must land where
        // the PITCHED voices' timeline puts that beat, not on a straight
        // clock (humanize jitter is ≤8ms; rubato offsets grow much larger).
        let second_bar_kick = hits
            .iter()
            .filter(|h| h.drum == DrumType::Kick)
            .nth(2)
            .unwrap();
        let expected = timeline.seconds(meter);
        assert!(
            (second_bar_kick.time - expected).abs() < 0.02,
            "kick at {} but rubato timeline says {}",
            second_bar_kick.time,
            expected
        );
        // Deterministic per seed.
        let again = drum_hits(
            &score,
            Style::Playful.spec().texture.drums,
            &timeline,
            &state,
            0,
        );
        assert_eq!(hits.len(), again.len());
        assert!(hits.iter().zip(&again).all(|(a, b)| a.time == b.time));
    }

    #[test]
    fn strum_rolls_chords_low_to_high_and_keeps_release_aligned() {
        let mk = |freq: f32, start: f32| {
            (
                0.0f64,
                Note {
                    frequency: freq,
                    start_time: start,
                    duration: 1.0,
                    velocity: 0.5,
                },
            )
        };
        // One 3-tone chord (written high-to-low to prove sorting) + a single note.
        let mut notes = vec![
            mk(660.0, 0.0),
            mk(440.0, 0.0),
            mk(220.0, 0.0),
            mk(330.0, 2.0),
        ];
        strum_chords(&mut notes, Instrument::AcousticGuitar);
        // Chord now ordered low-to-high with increasing onsets.
        assert_eq!(notes[0].1.frequency, 220.0);
        assert_eq!(notes[1].1.frequency, 440.0);
        assert_eq!(notes[2].1.frequency, 660.0);
        assert!(notes[0].1.start_time < notes[1].1.start_time);
        assert!(notes[1].1.start_time < notes[2].1.start_time);
        // All chord tones still END together.
        let end = |i: usize| notes[i].1.start_time + notes[i].1.duration;
        assert!((end(0) - end(1)).abs() < 1e-6);
        assert!((end(1) - end(2)).abs() < 1e-6);
        // The lone note is untouched.
        assert_eq!(notes[3].1.start_time, 2.0);
        assert_eq!(notes[3].1.duration, 1.0);
    }

    #[test]
    fn harmony_chords_no_longer_land_sample_simultaneous() {
        // Before the performance layer, every tone of a block chord had a
        // bit-identical start time. After strum + humanize, the first chord's
        // tones must be spread — by a small, roll-sized amount.
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let comp = realize(&score, &MusicalState::default(), 44100);
        let bass_len = score
            .voice(symthaea_music_theory::score::VoiceRole::Bass)
            .len();
        let harm_len = score
            .voice(symthaea_music_theory::score::VoiceRole::Harmony)
            .len();
        // comp.notes flattens voices in push order: bass, harmony, melody, …
        let harmony = &comp.notes[bass_len..bass_len + harm_len];
        let has_roll = harmony.windows(2).any(|w| {
            let d = (w[1].start_time - w[0].start_time).abs();
            d > 0.0 && d <= 0.05
        });
        assert!(
            has_roll,
            "expected at least one strummed/rolled chord in the harmony voice"
        );
    }

    #[test]
    fn humanized_realize_is_deterministic() {
        let intent = MusicalIntent::default();
        let a = compose_and_realize(&intent, &MusicalState::default(), 44100);
        let b = compose_and_realize(&intent, &MusicalState::default(), 44100);
        assert_eq!(a.notes.len(), b.notes.len());
        for (x, y) in a.notes.iter().zip(&b.notes) {
            assert_eq!(x.start_time, y.start_time);
            assert_eq!(x.velocity, y.velocity);
            assert_eq!(x.duration, y.duration);
        }
    }

    #[test]
    fn realized_mix_has_a_genuine_stereo_image() {
        // With per-role pans, the side signal (L-R) must carry real energy
        // relative to the mid — well beyond the faint decorrelation the
        // reverb tail alone used to provide when every voice sat at 0.0.
        let score = symthaea_music_theory::compose(&MusicalIntent::default());
        let comp = realize(&score, &MusicalState::default(), 44100);
        if let AudioData::StereoF32(frames) = &comp.audio {
            let (mut mid_e, mut side_e) = (0.0f64, 0.0f64);
            for [l, r] in frames {
                let m = (l + r) * 0.5;
                let s = (l - r) * 0.5;
                mid_e += (m * m) as f64;
                side_e += (s * s) as f64;
            }
            assert!(mid_e > 0.0, "silent render");
            let ratio = (side_e / mid_e).sqrt();
            assert!(
                ratio > 0.05,
                "side/mid RMS ratio {ratio:.4} — mix is still effectively mono"
            );
        } else {
            panic!("expected stereo output");
        }
    }

    #[test]
    fn realize_produces_audio_with_all_voices() {
        let intent = MusicalIntent {
            tonic: PitchClass::D,
            ..Default::default()
        };
        let score = symthaea_music_theory::compose(&intent);
        let comp = realize(&score, &MusicalState::default(), 44100);
        assert!(!comp.audio.is_empty());
        assert!(!comp.notes.is_empty());
        // The score's three voices all made it into the realized notes.
        assert!(comp.notes.len() >= score.notes.len());
    }

    #[test]
    fn compose_and_realize_is_finite() {
        let comp = compose_and_realize(&MusicalIntent::default(), &MusicalState::default(), 44100);
        if let AudioData::StereoF32(frames) = &comp.audio {
            assert!(frames.iter().all(|[l, r]| l.is_finite() && r.is_finite()));
            assert!(frames.iter().all(|[l, r]| l.abs() <= 1.0 && r.abs() <= 1.0));
        } else {
            panic!("expected stereo output");
        }
    }

    #[test]
    fn different_styles_pick_different_instrument_ensembles() {
        // Every seed in a small sample must land on a genuinely different
        // ensemble between Classical and Folk (the pools don't overlap).
        for seed in 0..3u64 {
            let (classical_mel, classical_harm, classical_bass) =
                instruments_for(Style::Classical, seed);
            let (folk_mel, folk_harm, folk_bass) = instruments_for(Style::Folk, seed);
            assert_ne!(classical_mel, folk_mel);
            assert_ne!(classical_harm, folk_harm);
            assert_ne!(classical_bass, folk_bass);
        }
    }

    #[test]
    fn waltz_shares_classicals_ensemble_pool() {
        // Waltz's distinctiveness is meter/tempo (handled by
        // symthaea-music-theory's Style), not a different ensemble --
        // a string trio family genuinely fits a ballroom waltz too.
        for seed in 0..5u64 {
            assert_eq!(
                instruments_for(Style::Waltz, seed),
                instruments_for(Style::Classical, seed)
            );
        }
    }

    #[test]
    fn instrument_choice_varies_by_seed_within_a_style() {
        // The whole point of the pool: NOT every Classical piece is the
        // same violin+piano+cello -- some seed must land elsewhere.
        let ensembles: std::collections::HashSet<_> = (0..3u64)
            .map(|seed| instruments_for(Style::Classical, seed))
            .collect();
        assert!(
            ensembles.len() > 1,
            "expected more than one distinct ensemble across seeds 0..3, got {ensembles:?}"
        );
    }

    #[test]
    fn realizing_the_same_score_with_different_styles_produces_different_audio() {
        // Same notes, different instrument ensembles -- if the timbre
        // wiring actually took effect, the rendered audio must differ (not
        // just the same synth patch relabeled).
        let intent = MusicalIntent::default();
        let score = symthaea_music_theory::compose(&intent); // one fixed Score
        let state = MusicalState::default();
        let classical = realize_styled(&score, Style::Classical, 0, &state, 44100);
        let cinematic = realize_styled(&score, Style::Cinematic, 0, &state, 44100);
        match (&classical.audio, &cinematic.audio) {
            (AudioData::StereoF32(a), AudioData::StereoF32(b)) => {
                assert_eq!(a.len(), b.len(), "same score -> same length");
                assert!(
                    a.iter().zip(b).any(|(x, y)| x != y),
                    "different ensembles must render different audio"
                );
            }
            _ => panic!("expected stereo output"),
        }
    }

    #[test]
    fn folk_style_exercises_the_karplus_strong_path_without_panicking() {
        // Folk's harmony and bass are both Karplus-Strong instruments
        // (AcousticGuitar, UprightBass) -- a real, different code path from
        // the additive engine. This must render finite, bounded audio.
        let comp = compose_and_realize_styled(
            &MusicalIntent::default(),
            Style::Folk,
            &MusicalState::default(),
            44100,
        );
        if let AudioData::StereoF32(frames) = &comp.audio {
            assert!(!frames.is_empty());
            assert!(frames.iter().all(|[l, r]| l.is_finite() && r.is_finite()));
            assert!(frames.iter().all(|[l, r]| l.abs() <= 1.0 && r.abs() <= 1.0));
        } else {
            panic!("expected stereo output");
        }
    }

    #[test]
    fn playful_style_exercises_the_fm_path_without_panicking() {
        // Playful's harmony is ElectricPiano (FM synthesis) -- a real,
        // different code path from both additive and Karplus-Strong.
        let comp = compose_and_realize_styled(
            &MusicalIntent::default(),
            Style::Playful,
            &MusicalState::default(),
            44100,
        );
        if let AudioData::StereoF32(frames) = &comp.audio {
            assert!(!frames.is_empty());
            assert!(frames.iter().all(|[l, r]| l.is_finite() && r.is_finite()));
            assert!(frames.iter().all(|[l, r]| l.abs() <= 1.0 && r.abs() <= 1.0));
        } else {
            panic!("expected stereo output");
        }
    }

    #[test]
    fn compose_and_perform_melody_produces_a_nonempty_melody_only_line() {
        let notes = compose_and_perform_melody(
            &MusicalIntent::default(),
            Style::Classical,
            &MusicalState::default(),
        );
        assert!(
            !notes.is_empty(),
            "a default intent must produce at least one melody note"
        );
        assert!(
            notes
                .iter()
                .all(|n| n.frequency.is_finite() && n.frequency > 0.0),
            "every melody note must have a positive, finite pitch"
        );
        assert!(
            notes.iter().all(|n| n.duration > 0.0),
            "every melody note must have a positive duration"
        );

        // The melody-only line must be shorter than the SAME score's total
        // notes across all four voices (Melody+Harmony+Bass+CounterMelody)
        // -- proof this genuinely excludes the accompaniment rather than
        // accidentally returning everything. (Comparing against a
        // DIFFERENT composer's output -- e.g. muse's own compose(), a
        // separate, much simpler melody generator with its own note-density
        // defaults -- is not a meaningful invariant: that composer can
        // legitimately produce more or fewer notes than this one for
        // unrelated reasons. The correct control is the same Score this
        // function itself composed internally.)
        let score =
            symthaea_music_theory::compose_styled(&MusicalIntent::default(), Style::Classical);
        assert!(
            notes.len() < score.notes.len(),
            "melody-only ({}) should be a strict subset of the same score's \
             total notes across all voices ({}) -- accompaniment must be excluded",
            notes.len(),
            score.notes.len()
        );
    }

    #[test]
    fn compose_and_perform_melody_is_deterministic_for_the_same_intent_and_seed() {
        let intent = MusicalIntent {
            seed: 42,
            ..MusicalIntent::default()
        };
        let a = compose_and_perform_melody(&intent, Style::Folk, &MusicalState::default());
        let b = compose_and_perform_melody(&intent, Style::Folk, &MusicalState::default());
        assert_eq!(
            a.len(),
            b.len(),
            "same intent+seed must produce the same note count"
        );
        for (na, nb) in a.iter().zip(b.iter()) {
            assert_eq!(na.frequency, nb.frequency);
            assert_eq!(na.start_time, nb.start_time);
            assert_eq!(na.duration, nb.duration);
        }
    }

    #[test]
    fn compose_and_perform_melody_reflects_contrasting_valence_arousal() {
        // Not a claim about WHICH direction pitch/tempo should move -- just
        // that a materially different consciousness state produces a
        // materially different melody, not the same tune every time
        // regardless of state (the exact failure mode /sing had before this
        // function existed: a fixed hardcoded arc).
        let calm = MusicalIntent {
            valence: -0.8,
            arousal: 0.1,
            seed: 7,
            ..MusicalIntent::default()
        };
        let excited = MusicalIntent {
            valence: 0.8,
            arousal: 0.9,
            seed: 7,
            ..MusicalIntent::default()
        };
        let notes_calm =
            compose_and_perform_melody(&calm, Style::Classical, &MusicalState::default());
        let notes_excited =
            compose_and_perform_melody(&excited, Style::Classical, &MusicalState::default());
        assert!(!notes_calm.is_empty() && !notes_excited.is_empty());

        // Deliberately NOT a claim about which specific property changes
        // (count, pitch, or timing) -- symthaea-music-theory's composer.rs
        // varies rhythm-bank choice, phrase archetype (period vs sentence),
        // AND tempo by arousal, any of which alone is sufficient evidence
        // the state genuinely reached the composer. Requiring a SPECIFIC
        // property (e.g. note count) to differ is over-constrained: two
        // structurally different phrases can legitimately land on the same
        // note count by coincidence for a given seed. The only thing that
        // would indicate a real bug is the two states producing a
        // byte-identical melody.
        let identical = notes_calm.len() == notes_excited.len()
            && notes_calm.iter().zip(notes_excited.iter()).all(|(a, b)| {
                a.frequency == b.frequency
                    && a.start_time == b.start_time
                    && a.duration == b.duration
            });
        assert!(
            !identical,
            "contrasting valence/arousal (calm vs excited) must produce a materially \
             different melody -- got a byte-identical note sequence for both states"
        );
    }
}
