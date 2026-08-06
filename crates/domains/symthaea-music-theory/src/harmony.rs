// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Functional harmony: keys, diatonic chords, harmonic function, and
//! progressions (archetypes + a functional-grammar generator).
//!
//! This is the harmonic scaffold a melody hangs on. A melody sounds like it
//! "means to go somewhere" only when it is heard *against* a progression that
//! itself moves with intention: tonic → predominant → dominant → tonic, the
//! engine of tonal tension and resolution.

use crate::chord::{Chord, ChordQuality};
use crate::pitch::PitchClass;
use crate::scale::{Mode, Scale};
use serde::{Deserialize, Serialize};

/// Major, minor, or modal tonality. (Minor uses the harmonic-minor scale
/// for chord construction so the dominant V is major and vii° is diminished
/// — the leading-tone harmony that makes functional minor work.)
///
/// `Modal` covers the diatonic church modes WITHOUT a leading tone
/// (Dorian, Phrygian, Lydian, Mixolydian, Aeolian). Their harmony is built
/// from the mode's own scale — no borrowed leading tone — so the idiomatic
/// closes differ (♭VII–i for Dorian/Mixolydian/Aeolian, ♭II–i for
/// Phrygian); see [`Key::cadence_dominant_degree`]. Ionian and
/// HarmonicMinor are NOT `Modal` values — [`Key::modal`] normalizes them to
/// `Major`/`Minor` so equality and the functional-harmony fast paths keep
/// working. Locrian and the non-heptatonic modes (pentatonic, whole-tone)
/// are excluded: the degree-based progression grammar assumes 7 usable
/// degrees and a stable tonic triad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tonality {
    Major,
    Minor,
    Modal(Mode),
}

/// A key: a tonic and a tonality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub tonic: PitchClass,
    pub tonality: Tonality,
}

/// The functional role of a chord in a key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HarmonicFunction {
    /// Home/rest (I, vi, iii).
    Tonic,
    /// Motion away, sets up the dominant (ii, IV).
    Predominant,
    /// Maximum tension, wants to resolve to tonic (V, vii°).
    Dominant,
}

impl Key {
    pub fn major(tonic: PitchClass) -> Self {
        Key {
            tonic,
            tonality: Tonality::Major,
        }
    }
    pub fn minor(tonic: PitchClass) -> Self {
        Key {
            tonic,
            tonality: Tonality::Minor,
        }
    }

    /// A modal key. Ionian and HarmonicMinor normalize to `Major`/`Minor`
    /// (they ARE those tonalities); Aeolian stays modal — natural minor
    /// with a ♭VII close is genuinely different music from functional
    /// harmonic minor. Returns `None` for modes the degree grammar can't
    /// support (Locrian's tonic is diminished; pentatonic/whole-tone lack
    /// 7 degrees; MelodicMinor's functional harmony is not implemented).
    pub fn modal(tonic: PitchClass, mode: Mode) -> Option<Self> {
        let tonality = match mode {
            Mode::Ionian => Tonality::Major,
            Mode::HarmonicMinor => Tonality::Minor,
            Mode::Dorian | Mode::Phrygian | Mode::Lydian | Mode::Mixolydian | Mode::Aeolian => {
                Tonality::Modal(mode)
            }
            _ => return None,
        };
        Some(Key { tonic, tonality })
    }

    /// The scale used to build diatonic chords in this key.
    pub fn scale(self) -> Scale {
        match self.tonality {
            Tonality::Major => Scale::new(self.tonic, Mode::Ionian),
            Tonality::Minor => Scale::new(self.tonic, Mode::HarmonicMinor),
            Tonality::Modal(mode) => Scale::new(self.tonic, mode),
        }
    }

    /// The scale degree whose chord plays the DOMINANT role at cadences.
    /// Functional keys (and Lydian, whose V is major) close V→I. The
    /// leading-tone-less modes close from their characteristic chord
    /// instead: ♭VII→i for Dorian/Mixolydian/Aeolian (degree 7 — a major
    /// triad in all three), ♭II→i for Phrygian (degree 2, the Phrygian
    /// cadence). Forcing V–I onto these modes would need a borrowed
    /// leading tone and erase the modal color.
    pub fn cadence_dominant_degree(self) -> i32 {
        match self.tonality {
            Tonality::Major | Tonality::Minor | Tonality::Modal(Mode::Lydian) => 5,
            Tonality::Modal(Mode::Phrygian) => 2,
            Tonality::Modal(_) => 7,
        }
    }

    /// The parallel key: same tonic, opposite tonality (C major <-> C minor).
    /// Unlike [`Self::relative`] (same pitch classes, different tonic), this
    /// keeps the tonic and changes the scale's pitch classes — the classic
    /// major/minor mode-mixture relationship, used by
    /// [`crate::form::Form::rondo`] for its second contrasting section (a
    /// genuinely different kind of modulation from the relative-key move
    /// `Form::ternary`'s B section already uses).
    pub fn parallel(self) -> Key {
        match self.tonality {
            Tonality::Major => Key::minor(self.tonic),
            Tonality::Minor => Key::major(self.tonic),
            // Modal keys flip toward the opposite brightness on the same
            // tonic — the same light/dark contrast the classic parallel
            // gives, resolved to a FUNCTIONAL key so the contrasting
            // section gets real leading-tone drive before the modal home
            // returns. (Not an involution for modal keys; the round-trip
            // guarantee holds for Major/Minor only.)
            Tonality::Modal(mode) => {
                if mode.is_minor_flavored() {
                    Key::major(self.tonic)
                } else {
                    Key::minor(self.tonic)
                }
            }
        }
    }

    /// The dominant key: same tonality, tonic transposed up a perfect
    /// fifth (7 semitones). The classic sonata-form second-subject key
    /// when the home key is MAJOR; a minor-mode sonata moves to the
    /// RELATIVE major instead (see [`Self::relative`]) — real
    /// common-practice usage, not a uniform fifth-transposition. See
    /// [`crate::sonata`].
    pub fn dominant(self) -> Key {
        Key {
            tonic: self.tonic.transpose(7),
            tonality: self.tonality,
        }
    }

    /// The relative key: major → its relative minor (scale degree 6 as the
    /// new tonic), minor → its relative major (scale degree 3). Relative
    /// keys share the same set of diatonic pitch classes, so this is a real
    /// modulation that needs no chromatic alteration — a genuine change of
    /// tonal center built entirely from primitives already in this crate,
    /// used by [`crate::form::Form::ternary`] for its contrasting section.
    pub fn relative(self) -> Key {
        match self.tonality {
            Tonality::Major => Key::minor(self.scale().degree_pitch_class(6)),
            Tonality::Minor => Key::major(self.scale().degree_pitch_class(3)),
            // Every church mode is a rotation of a major scale, so its
            // relative key is that Ionian collection: D Dorian → C major,
            // G Mixolydian → C major. Same pitch classes, different tonal
            // center — the contract "a real modulation with no chromatic
            // alteration" holds exactly. (Not an involution for modal
            // keys; Major/Minor keep their round-trip.)
            Tonality::Modal(mode) => {
                let rotation_offset = match mode {
                    Mode::Dorian => 2,
                    Mode::Phrygian => 4,
                    Mode::Lydian => 5,
                    Mode::Mixolydian => 7,
                    _ => 9, // Aeolian (Key::modal admits no other modes here)
                };
                Key::major(self.tonic.transpose(-rotation_offset))
            }
        }
    }

    /// The diatonic triad on scale degree `degree` (1..=7), built by stacking
    /// scale thirds (degrees d, d+2, d+4). The quality falls out of the scale.
    pub fn diatonic_triad(self, degree: i32) -> Chord {
        let scale = self.scale();
        let root = scale.degree_pitch_class(degree);
        let third = scale.degree_pitch_class(degree + 2);
        let fifth = scale.degree_pitch_class(degree + 4);
        Chord::new(root, classify_triad(root, third, fifth))
    }

    /// The diatonic seventh chord on `degree` (adds the scale seventh above
    /// the root: degrees d, d+2, d+4, d+6).
    pub fn diatonic_seventh(self, degree: i32) -> Chord {
        let scale = self.scale();
        let root = scale.degree_pitch_class(degree);
        let third = scale.degree_pitch_class(degree + 2);
        let fifth = scale.degree_pitch_class(degree + 4);
        let seventh = scale.degree_pitch_class(degree + 6);
        Chord::new(root, classify_seventh(root, third, fifth, seventh))
    }

    /// The harmonic function of scale degree `degree`.
    pub fn function(self, degree: i32) -> HarmonicFunction {
        match degree.rem_euclid(7) {
            1 | 6 | 3 => HarmonicFunction::Tonic, // I, vi, iii  (deg%7: 1,6,3)
            2 | 4 => HarmonicFunction::Predominant, // ii, IV
            5 | 0 => HarmonicFunction::Dominant,  // V, vii° (7 % 7 == 0)
            _ => HarmonicFunction::Tonic,
        }
    }

    /// The secondary dominant of a degree — V7 of that degree's chord (its
    /// dominant a fifth above). E.g. V7/V in C major = D7.
    pub fn secondary_dominant(self, of_degree: i32) -> Chord {
        let target_root = self.scale().degree_pitch_class(of_degree);
        let dom_root = target_root.transpose(7); // a fifth above the target
        Chord::new(dom_root, ChordQuality::Dominant7)
    }
}

/// Classify a triad from its three pitch classes.
fn classify_triad(root: PitchClass, third: PitchClass, fifth: PitchClass) -> ChordQuality {
    let t = root.interval_to(third);
    let f = root.interval_to(fifth);
    match (t, f) {
        (4, 7) => ChordQuality::Major,
        (3, 7) => ChordQuality::Minor,
        (3, 6) => ChordQuality::Diminished,
        (4, 8) => ChordQuality::Augmented,
        (2, 7) => ChordQuality::Sus2,
        (5, 7) => ChordQuality::Sus4,
        // Same reasoning as `classify_seventh`'s fallback: keep degrading rather
        // than failing a render, but do not do it silently.
        _ => {
            debug_assert!(
                false,
                "unclassified triad ({t}, {f}) from root {root:?}: add a ChordQuality \
                 variant instead of falling back to Major"
            );
            ChordQuality::Major
        }
    }
}

fn classify_seventh(
    root: PitchClass,
    third: PitchClass,
    fifth: PitchClass,
    seventh: PitchClass,
) -> ChordQuality {
    let t = root.interval_to(third);
    let f = root.interval_to(fifth);
    let s = root.interval_to(seventh);
    match (t, f, s) {
        (4, 7, 11) => ChordQuality::Major7,
        (3, 7, 10) => ChordQuality::Minor7,
        (4, 7, 10) => ChordQuality::Dominant7,
        (3, 7, 11) => ChordQuality::MinorMajor7,
        (3, 6, 10) => ChordQuality::HalfDiminished7,
        (3, 6, 9) => ChordQuality::Diminished7,
        // ♭III of every minor key: HarmonicMinor's degree 3 stacks to
        // (major third, augmented fifth, major seventh). This used to fall
        // through to the Dominant7 fallback below, silently turning
        // C–E–G♯–B into C–E–G–B♭ and destroying A minor's leading tone.
        (4, 8, 11) => ChordQuality::AugmentedMajor7,
        // Genuinely exotic stacks (non-heptatonic or synthetic scales) still
        // degrade to a dominant seventh rather than failing a render, but they
        // must not do so SILENTLY: the bug above hid here for as long as this
        // arm existed. `debug_assert` fails the test suite while leaving release
        // renders working.
        _ => {
            debug_assert!(
                false,
                "unclassified seventh stack ({t}, {f}, {s}) from root {root:?}: add a \
                 ChordQuality variant instead of falling back to Dominant7"
            );
            ChordQuality::Dominant7
        }
    }
}

/// A chord progression as a sequence of scale degrees in a key (1..=7). The
/// symbolic form; call [`Progression::chords`] to realize the chords.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Progression {
    pub degrees: Vec<i32>,
}

impl Progression {
    pub fn new(degrees: Vec<i32>) -> Self {
        Progression { degrees }
    }

    pub fn len(&self) -> usize {
        self.degrees.len()
    }
    pub fn is_empty(&self) -> bool {
        self.degrees.is_empty()
    }

    /// The diatonic triads for this progression in the given key.
    pub fn chords(&self, key: Key) -> Vec<Chord> {
        self.degrees
            .iter()
            .map(|&d| key.diatonic_triad(d))
            .collect()
    }

    /// The functional tension curve: each chord's tension weighted by its
    /// function (predominant and dominant raise tension, tonic releases it).
    pub fn tension_curve(&self, key: Key) -> Vec<f32> {
        self.degrees
            .iter()
            .map(|&d| {
                let base = key.diatonic_triad(d).quality.tension();
                let functional = match key.function(d) {
                    HarmonicFunction::Tonic => 0.0,
                    HarmonicFunction::Predominant => 0.4,
                    HarmonicFunction::Dominant => 0.8,
                };
                (base + functional).min(1.0)
            })
            .collect()
    }

    // ── Archetypes (the common progressions, by scale degree) ─────────────

    /// I–IV–V–I: the most basic authentic progression.
    pub fn authentic() -> Self {
        Progression::new(vec![1, 4, 5, 1])
    }
    /// ii–V–I: the jazz/tonal cadential core.
    pub fn two_five_one() -> Self {
        Progression::new(vec![2, 5, 1])
    }
    /// I–vi–IV–V: the "50s"/pop turnaround.
    pub fn pop_turnaround() -> Self {
        Progression::new(vec![1, 6, 4, 5])
    }
    /// I–V–vi–IV: the ubiquitous "four-chord" pop loop.
    pub fn four_chord() -> Self {
        Progression::new(vec![1, 5, 6, 4])
    }
    /// vi–IV–I–V: the same loop rotated to a minor-flavoured start.
    pub fn sensitive_female() -> Self {
        Progression::new(vec![6, 4, 1, 5])
    }
    /// I–V–vi–iii–IV–I–IV–V: Pachelbel's canon.
    pub fn pachelbel() -> Self {
        Progression::new(vec![1, 5, 6, 3, 4, 1, 4, 5])
    }

    /// Generate a progression by a functional grammar: T → PD → D → T, with a
    /// deterministic seeded walk. Always starts on tonic and ends on a
    /// resolution (authentic V→I or deceptive V→vi). `bars` chords total.
    pub fn generate(bars: usize, seed: u64) -> Self {
        Progression::generate_with(bars, seed, &HarmonicPalette::classical())
    }

    /// [`Progression::generate`] with a style's own harmonic vocabulary.
    ///
    /// # Why this exists
    ///
    /// `generate` is style-AGNOSTIC: same seed, same progression, whatever style
    /// asked. Measured 2026-07-30 over Classical / BaroqueSuite / March /
    /// Nocturne × 32 seeds, every `ProgressionSpec::Grammar` style produced
    /// **bit-identical progressions, 96/96**. Converting a style to `Grammar`
    /// therefore bought within-style variety (1 → 26 distinct) at the cost of
    /// making it harmonically indistinguishable from every other Grammar style —
    /// the exact cross-style bleed the diversity census fixed for
    /// Folk/Cinematic/Playful by giving each its own degree set. That blocked
    /// the harmonic-syntax rework (see
    /// `HARMONIC_SYNTAX_REWORK_SCOPE_2026-07-26.md`).
    ///
    /// A palette is the same idiom those three styles already use, applied to
    /// the generator instead of to a fixed archetype: the T-PD-D-T *grammar* is
    /// universal, the *vocabulary* is the style's.
    ///
    /// [`HarmonicPalette::classical`] reproduces the previous behaviour exactly
    /// — see `generate_with_classical_palette_is_bit_identical_to_generate`.
    pub fn generate_with(bars: usize, seed: u64, palette: &HarmonicPalette) -> Self {
        // The mid-phrase walk leaves a predominant ONLY via
        // `cadential_dominant`, and it leaves a dominant via the hardwired
        // D → T. So a `cadential_dominant` that is not Dominant-function makes
        // the predominant branch self-looping: a plagal palette (IV as the
        // "dominant") yields IV → IV → IV forever and never returns to tonic.
        //
        // That rules palettes out for genuinely NON-FUNCTIONAL styles — a
        // Cinematic-style suspended arc with no dominant at all, or
        // impressionist planing — which is a real limit, not an oversight.
        // Those belong on `ProgressionSpec::ArchetypePool` with their own
        // hand-chosen degree sets, which is exactly where they already are.
        debug_assert_eq!(
            degree_function(palette.cadential_dominant),
            HarmonicFunction::Dominant,
            "cadential_dominant must be Dominant-function (got degree {}); a predominant \
             here makes the walk loop on it forever. Non-functional styles need \
             ProgressionSpec::ArchetypePool, not a palette.",
            palette.cadential_dominant
        );
        if bars == 0 {
            return Progression::default();
        }
        let mut rng = SplitMix::new(seed);
        let mut degrees = vec![1]; // start at tonic
        for i in 1..bars {
            let prev = *degrees.last().unwrap();
            let prev_fn = degree_function(prev);
            // The last two chords form the cadence: force D then T.
            let next = if i == bars - 1 {
                // Final chord: authentic most of the time, deceptive sometimes.
                if rng.next_below(palette.deceptive_cadence_one_in) == 0 {
                    6
                } else {
                    1
                }
            } else if i == bars - 2 {
                palette.cadential_dominant // penultimate: set up the cadence
            } else {
                match prev_fn {
                    HarmonicFunction::Tonic => {
                        let from = &palette.from_tonic;
                        from[rng.next_below(from.len() as u64) as usize]
                    }
                    HarmonicFunction::Predominant => {
                        // Usually move on to the dominant; sometimes linger.
                        if rng.next_below(palette.linger_on_predominant_one_in) == 0 {
                            palette.predominant_linger
                        } else {
                            palette.cadential_dominant
                        }
                    }
                    HarmonicFunction::Dominant => 1, // D → T
                }
            };
            degrees.push(next);
        }
        Progression::new(degrees)
    }
}

/// A style's harmonic VOCABULARY for [`Progression::generate_with`]. The
/// T→PD→D→T grammar is universal; which degrees realize each function, and how
/// readily the walk leaves them, is what makes a march sound unlike a nocturne.
///
/// Deliberately narrow: it constrains *choices within* the existing grammar
/// rather than replacing it, so every palette still starts on tonic, still ends
/// on a real resolution, and still cannot emit a musically invalid sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarmonicPalette {
    /// Degrees reachable from a tonic-function chord, sampled uniformly.
    pub from_tonic: Vec<i32>,
    /// The degree that serves as the cadential dominant.
    pub cadential_dominant: i32,
    /// The predominant the walk lingers on instead of moving to the dominant.
    pub predominant_linger: i32,
    /// One-in-N chance of lingering on a predominant. Higher = more direct.
    pub linger_on_predominant_one_in: u64,
    /// One-in-N chance the final chord is deceptive (vi) rather than tonic.
    /// Higher = more decisive endings.
    pub deceptive_cadence_one_in: u64,
}

impl HarmonicPalette {
    /// The historical behaviour, exactly: T→{ii,IV,vi}, V as dominant, a 1-in-4
    /// lingering IV, a 1-in-4 deceptive close. Every value here is the literal
    /// that was inline in `generate` before palettes existed, so
    /// `generate_with(.., &CLASSICAL)` is bit-identical to `generate(..)`.
    pub fn classical() -> HarmonicPalette {
        HarmonicPalette {
            from_tonic: vec![2, 4, 6],
            cadential_dominant: 5,
            predominant_linger: 4,
            linger_on_predominant_one_in: 4,
            deceptive_cadence_one_in: 4,
        }
    }

    /// Sequential and driving — the harmony of a baroque suite movement.
    ///
    /// Baroque functional harmony chains predominants (the circle-of-fifths
    /// sequence is its signature gesture) and closes decisively; it does not
    /// wander to vi the way a classical period does. So: ii is the favoured
    /// departure from tonic, the walk lingers on predominants *often* rather
    /// than rarely (that lingering IS the sequence), and deceptive closes are
    /// uncommon.
    ///
    /// Added 2026-07-30 because
    /// `grammar_generated_styles_must_not_be_harmonically_interchangeable`
    /// caught BaroqueSuite and Classical producing identical progressions on
    /// 32/32 seeds — a real defect that shipped with the 2026-07-26 harmonic
    /// pilot, which converted BaroqueSuite to the style-agnostic `Grammar`
    /// generator and so made it a harmonic clone of Classical.
    pub fn baroque() -> HarmonicPalette {
        HarmonicPalette {
            from_tonic: vec![2, 2, 4],
            cadential_dominant: 5,
            predominant_linger: 2,
            linger_on_predominant_one_in: 2,
            deceptive_cadence_one_in: 8,
        }
    }

    /// Circling and songlike — the harmony of a waltz.
    ///
    /// Its fixed archetype was Pachelbel's [1,5,6,3,4,1,4,5], and the palette
    /// keeps that sequence's flavour rather than discarding it: the descending
    /// vi–iii–IV chain means a waltz leaves tonic toward *any* of the three
    /// non-dominant functions about equally, then walks down to the dominant
    /// rather than driving at it. Lingering is common (that is the descent);
    /// deceptive closes are rare, because a waltz phrase lands.
    pub fn waltz() -> HarmonicPalette {
        HarmonicPalette {
            from_tonic: vec![6, 3, 4, 2],
            cadential_dominant: 5,
            predominant_linger: 4,
            linger_on_predominant_one_in: 2,
            deceptive_cadence_one_in: 6,
        }
    }

    /// Wistful and unhurried — the harmony of a nocturne.
    ///
    /// The opposite of [`HarmonicPalette::march`] on every axis that matters:
    /// it leans on vi (the relative minor colouring that gives a nocturne its
    /// melancholy), reaches the dominant only reluctantly — lingering on ii
    /// rather than driving through — and closes deceptively far more often than
    /// Classical does, because a nocturne is characteristically reluctant to
    /// settle.
    pub fn nocturne() -> HarmonicPalette {
        HarmonicPalette {
            from_tonic: vec![6, 6, 2, 4],
            cadential_dominant: 5,
            predominant_linger: 2,
            linger_on_predominant_one_in: 2,
            deceptive_cadence_one_in: 3,
        }
    }

    /// Plain, decisive, unwandering — the harmony of a march.
    ///
    /// Drops vi from the tonic branch (no relative-minor colouring), uses IV as
    /// the only predominant, never lingers, and never closes deceptively. The
    /// result is I-IV-V-I motion with real per-seed variety in *where* those
    /// chords fall, rather than a fixed loop or Classical's wandering walk.
    pub fn march() -> HarmonicPalette {
        HarmonicPalette {
            from_tonic: vec![4, 4, 5],
            cadential_dominant: 5,
            predominant_linger: 4,
            linger_on_predominant_one_in: u64::MAX, // never lingers
            deceptive_cadence_one_in: u64::MAX,     // always closes authentic
        }
    }
}

fn degree_function(degree: i32) -> HarmonicFunction {
    match degree.rem_euclid(7) {
        1 | 6 | 3 => HarmonicFunction::Tonic,
        2 | 4 => HarmonicFunction::Predominant,
        _ => HarmonicFunction::Dominant,
    }
}

/// A tiny deterministic PRNG (SplitMix64) — keeps progression generation pure
/// and reproducible (same seed → same progression), no external rand dep.
struct SplitMix {
    state: u64,
}

impl SplitMix {
    fn new(seed: u64) -> Self {
        SplitMix { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    /// A value in `[0, n)`.
    fn next_below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_of_c_major_is_c_minor() {
        let c = Key::major(PitchClass::C);
        let par = c.parallel();
        assert_eq!(par.tonality, Tonality::Minor);
        assert_eq!(par.tonic, PitchClass::C); // SAME tonic, unlike relative()
    }

    #[test]
    fn parallel_round_trips() {
        let c = Key::major(PitchClass::C);
        assert_eq!(c.parallel().parallel(), c);
    }

    #[test]
    fn relative_minor_of_c_major_is_a_minor() {
        let c = Key::major(PitchClass::C);
        let rel = c.relative();
        assert_eq!(rel.tonality, Tonality::Minor);
        assert_eq!(rel.tonic, PitchClass::A);
    }

    #[test]
    fn relative_major_of_a_minor_is_c_major() {
        let a = Key::minor(PitchClass::A);
        let rel = a.relative();
        assert_eq!(rel.tonality, Tonality::Major);
        assert_eq!(rel.tonic, PitchClass::C);
    }

    #[test]
    fn relative_round_trips() {
        let c = Key::major(PitchClass::C);
        let back = c.relative().relative();
        assert_eq!(back.tonic, c.tonic);
        assert_eq!(back.tonality, c.tonality);
    }

    #[test]
    fn c_major_diatonic_triads_ground_truth() {
        // C major: I=C, ii=Dm, iii=Em, IV=F, V=G, vi=Am, vii°=B°.
        let k = Key::major(PitchClass::C);
        assert_eq!(k.diatonic_triad(1).quality, ChordQuality::Major); // C
        assert_eq!(k.diatonic_triad(2).quality, ChordQuality::Minor); // Dm
        assert_eq!(k.diatonic_triad(3).quality, ChordQuality::Minor); // Em
        assert_eq!(k.diatonic_triad(4).quality, ChordQuality::Major); // F
        assert_eq!(k.diatonic_triad(5).quality, ChordQuality::Major); // G
        assert_eq!(k.diatonic_triad(6).quality, ChordQuality::Minor); // Am
        assert_eq!(k.diatonic_triad(7).quality, ChordQuality::Diminished); // B°
        assert_eq!(k.diatonic_triad(5).root, PitchClass::G);
    }

    #[test]
    fn g_dominant_seventh_is_the_v7_of_c() {
        // V7 in C major = G7 = G B D F.
        let k = Key::major(PitchClass::C);
        let v7 = k.diatonic_seventh(5);
        assert_eq!(v7.root, PitchClass::G);
        assert_eq!(v7.quality, ChordQuality::Dominant7);
    }

    /// ♭III of a minor key is an AUGMENTED-MAJOR seventh, not a dominant seventh.
    ///
    /// `Key::minor` builds chords from harmonic minor (`Tonality::Minor` =>
    /// `Mode::HarmonicMinor`), whose degree 3 stacks to (major third, AUGMENTED
    /// fifth, major seventh) = (4, 8, 11). `classify_seventh` had no arm for that
    /// stack and fell through to `_ => Dominant7`, so in A minor
    /// `diatonic_seventh(3)` returned C–E–G–B♭ instead of C–E–G♯–B — a perfect
    /// fifth where the augmented fifth belongs, a minor seventh where the major
    /// belongs, and the key's LEADING TONE (G♯) replaced by G natural.
    ///
    /// It also broke the invariant `composer.rs:4146-4150` relies on ("a seventh
    /// is a strict superset of its triad, so the melody's triad-based chord-tone
    /// snapping never clashes"): `diatonic_triad(3)` correctly returned Augmented
    /// (C–E–G♯) via `classify_triad`'s own (4, 8) arm, so the triad contained G♯
    /// while the seventh contained G — a melody snapped to G♯ sounded a semitone
    /// against its own harmony. That superset property is asserted below.
    #[test]
    fn flat_three_of_a_minor_is_augmented_major_seventh_not_dominant() {
        let k = Key::minor(PitchClass::A);
        let chord = k.diatonic_seventh(3);
        assert_eq!(chord.root, PitchClass::C);
        assert_eq!(
            chord.quality,
            ChordQuality::AugmentedMajor7,
            "C-E-G#-B is +M7; Dominant7 would mean C-E-G-Bb and lose A minor's leading tone"
        );
        assert_eq!(chord.quality.intervals(), &[0, 4, 8, 11]);

        // The leading tone must be present: C + 8 semitones = G#.
        let tones: Vec<PitchClass> = chord
            .quality
            .intervals()
            .iter()
            .map(|&i| chord.root.transpose(i as i32))
            .collect();
        assert!(
            tones.contains(&PitchClass::GSHARP),
            "the augmented fifth must be G#, A minor's leading tone: got {tones:?}"
        );
        assert!(!tones.contains(&PitchClass::G), "G natural must NOT appear");

        // Superset invariant: the seventh contains every tone of its triad.
        let triad = k.diatonic_triad(3);
        assert_eq!(triad.quality, ChordQuality::Augmented);
        for i in triad.quality.intervals() {
            let t = triad.root.transpose(*i as i32);
            assert!(
                tones.contains(&t),
                "seventh must be a superset of its triad; missing {t:?}"
            );
        }
    }

    /// Every minor key, not just A — the defect was in the interval stack, so it
    /// applied to all 12 transpositions.
    #[test]
    fn flat_three_is_augmented_major_seventh_in_every_minor_key() {
        for pc in 0..12i32 {
            let k = Key::minor(PitchClass::new(pc));
            assert_eq!(
                k.diatonic_seventh(3).quality,
                ChordQuality::AugmentedMajor7,
                "minor key with tonic pc={pc}"
            );
        }
    }

    #[test]
    fn minor_key_has_major_dominant() {
        // A minor (harmonic): V = E major (with G#), vii° = G#°.
        let k = Key::minor(PitchClass::A);
        assert_eq!(k.diatonic_triad(1).quality, ChordQuality::Minor); // i = Am
        assert_eq!(k.diatonic_triad(5).quality, ChordQuality::Major); // V = E
        assert_eq!(k.diatonic_triad(5).root, PitchClass::E);
    }

    #[test]
    fn harmonic_function_classification() {
        let k = Key::major(PitchClass::C);
        assert_eq!(k.function(1), HarmonicFunction::Tonic);
        assert_eq!(k.function(6), HarmonicFunction::Tonic);
        assert_eq!(k.function(2), HarmonicFunction::Predominant);
        assert_eq!(k.function(4), HarmonicFunction::Predominant);
        assert_eq!(k.function(5), HarmonicFunction::Dominant);
        assert_eq!(k.function(7), HarmonicFunction::Dominant);
    }

    #[test]
    fn secondary_dominant_of_v_is_d7() {
        // V7/V in C major is D7 (the dominant of G).
        let k = Key::major(PitchClass::C);
        let v_of_v = k.secondary_dominant(5);
        assert_eq!(v_of_v.root, PitchClass::D);
        assert_eq!(v_of_v.quality, ChordQuality::Dominant7);
    }

    #[test]
    fn archetype_chords_realize() {
        // I–IV–V–I in C = C, F, G, C.
        let chords = Progression::authentic().chords(Key::major(PitchClass::C));
        let roots: Vec<_> = chords.iter().map(|c| c.root).collect();
        assert_eq!(
            roots,
            vec![PitchClass::C, PitchClass::F, PitchClass::G, PitchClass::C]
        );
    }

    #[test]
    fn tension_curve_peaks_on_dominant() {
        // In I–IV–V–I the tension should peak on V (index 2) and resolve on the
        // final I (lower than V).
        let curve = Progression::authentic().tension_curve(Key::major(PitchClass::C));
        assert!(curve[2] > curve[0]); // V more tense than I
        assert!(curve[2] > curve[3]); // resolves on final I
    }

    #[test]
    fn modal_keys_normalize_and_reject_correctly() {
        use crate::pitch::PitchClass;
        // Ionian/HarmonicMinor ARE major/minor.
        assert_eq!(
            Key::modal(PitchClass::C, Mode::Ionian),
            Some(Key::major(PitchClass::C))
        );
        assert_eq!(
            Key::modal(PitchClass::A, Mode::HarmonicMinor),
            Some(Key::minor(PitchClass::A))
        );
        // Aeolian stays modal — ♭VII closes, not harmonic-minor V.
        let aeolian = Key::modal(PitchClass::A, Mode::Aeolian).unwrap();
        assert_eq!(aeolian.tonality, Tonality::Modal(Mode::Aeolian));
        // The grammar-incompatible modes are refused.
        assert_eq!(Key::modal(PitchClass::C, Mode::Locrian), None);
        assert_eq!(Key::modal(PitchClass::C, Mode::MajorPentatonic), None);
        assert_eq!(Key::modal(PitchClass::C, Mode::WholeTone), None);
        assert_eq!(Key::modal(PitchClass::C, Mode::MelodicMinor), None);
    }

    #[test]
    fn dorian_triads_ground_truth() {
        // D Dorian: i=Dm, IV=G (major — THE Dorian sound), ♭VII=C.
        let k = Key::modal(PitchClass::D, Mode::Dorian).unwrap();
        assert_eq!(k.diatonic_triad(1).quality, ChordQuality::Minor);
        assert_eq!(k.diatonic_triad(1).root, PitchClass::D);
        assert_eq!(k.diatonic_triad(4).quality, ChordQuality::Major); // G
        assert_eq!(k.diatonic_triad(4).root, PitchClass::G);
        assert_eq!(k.diatonic_triad(7).quality, ChordQuality::Major); // C
        assert_eq!(k.diatonic_triad(7).root, PitchClass::C);
    }

    #[test]
    fn mixolydian_triads_ground_truth() {
        // G Mixolydian: I=G major, ♭VII=F major (the folk/blues close).
        let k = Key::modal(PitchClass::G, Mode::Mixolydian).unwrap();
        assert_eq!(k.diatonic_triad(1).quality, ChordQuality::Major);
        assert_eq!(k.diatonic_triad(7).quality, ChordQuality::Major);
        assert_eq!(k.diatonic_triad(7).root, PitchClass::F);
    }

    #[test]
    fn aeolian_has_no_leading_tone_harmony() {
        // A Aeolian: v = E MINOR (unlike harmonic minor's major V),
        // ♭VII = G major.
        let k = Key::modal(PitchClass::A, Mode::Aeolian).unwrap();
        assert_eq!(k.diatonic_triad(5).quality, ChordQuality::Minor);
        assert_eq!(k.diatonic_triad(7).quality, ChordQuality::Major);
        assert_eq!(k.diatonic_triad(7).root, PitchClass::G);
    }

    #[test]
    fn modal_cadence_degrees() {
        use crate::pitch::PitchClass as PC;
        assert_eq!(Key::major(PC::C).cadence_dominant_degree(), 5);
        assert_eq!(Key::minor(PC::A).cadence_dominant_degree(), 5);
        let deg = |m| Key::modal(PC::D, m).unwrap().cadence_dominant_degree();
        assert_eq!(deg(Mode::Dorian), 7);
        assert_eq!(deg(Mode::Mixolydian), 7);
        assert_eq!(deg(Mode::Aeolian), 7);
        assert_eq!(deg(Mode::Phrygian), 2); // the Phrygian cadence
        assert_eq!(deg(Mode::Lydian), 5); // Lydian's V is major — keep V–I
    }

    #[test]
    fn modal_relative_and_parallel_ground_truth() {
        // D Dorian shares its pitch classes with C major.
        let dorian = Key::modal(PitchClass::D, Mode::Dorian).unwrap();
        assert_eq!(dorian.relative(), Key::major(PitchClass::C));
        // G Mixolydian too.
        let mixo = Key::modal(PitchClass::G, Mode::Mixolydian).unwrap();
        assert_eq!(mixo.relative(), Key::major(PitchClass::C));
        // Relative keys really do share every pitch class.
        let a: Vec<_> = dorian.scale().pitch_classes();
        for pc in dorian.relative().scale().pitch_classes() {
            assert!(a.contains(&pc));
        }
        // Parallel flips brightness on the same tonic.
        assert_eq!(dorian.parallel(), Key::major(PitchClass::D));
        assert_eq!(mixo.parallel(), Key::minor(PitchClass::G));
    }

    /// The palette refactor must be a strict no-op for every existing caller.
    ///
    /// `HarmonicPalette::classical()` holds the exact literals that were inline
    /// in `generate` before palettes existed, so this is what makes the change
    /// safe to land: Classical and BaroqueSuite (the two shipped
    /// `ProgressionSpec::Grammar` styles, one of them the reference the
    /// BaroqueSuite pilot was A/B'd against) keep byte-identical output.
    ///
    /// A previous attempt at per-style character — salting the seed with the
    /// style name — failed exactly here: it silently changed Classical. Bit
    /// equality across a wide seed and length sweep is the guard against
    /// repeating that.
    #[test]
    fn generate_with_classical_palette_is_bit_identical_to_generate() {
        let classical = HarmonicPalette::classical();
        for bars in [0usize, 1, 2, 3, 4, 7, 8, 12, 16, 32] {
            for seed in 0..64u64 {
                assert_eq!(
                    Progression::generate(bars, seed),
                    Progression::generate_with(bars, seed, &classical),
                    "bars={bars} seed={seed}: the default palette must reproduce the \
                     pre-palette generator exactly"
                );
            }
        }
    }

    /// A palette must produce a genuinely DIFFERENT style, not just a different
    /// seed stream — that was the whole point of adding them.
    #[test]
    fn march_palette_differs_from_classical_yet_keeps_the_grammar() {
        let classical = HarmonicPalette::classical();
        let march = HarmonicPalette::march();
        let mut differ = 0;
        for seed in 0..64u64 {
            let c = Progression::generate_with(8, seed, &classical);
            let m = Progression::generate_with(8, seed, &march);
            if c != m {
                differ += 1;
            }
            // Both must still obey the universal grammar the palette only
            // supplies vocabulary for.
            for p in [&c, &m] {
                assert_eq!(p.degrees[0], 1, "must start on tonic");
                assert_eq!(p.degrees[6], 5, "penultimate must be the dominant");
                assert!(p.degrees[7] == 1 || p.degrees[7] == 6);
            }
            // March never wanders to vi and never closes deceptively.
            assert!(
                !m.degrees[..7].contains(&6),
                "march palette must not reach vi: {:?}",
                m.degrees
            );
            assert_eq!(
                m.degrees[7], 1,
                "march must close authentic: {:?}",
                m.degrees
            );
        }
        assert!(
            differ >= 60,
            "march should differ from classical on nearly every seed, got {differ}/64"
        );
    }

    #[test]
    fn generated_progression_is_deterministic_and_cadences() {
        let a = Progression::generate(8, 42);
        let b = Progression::generate(8, 42);
        assert_eq!(a, b); // same seed → same progression
        assert_eq!(a.len(), 8);
        assert_eq!(a.degrees[0], 1); // starts on tonic
        assert_eq!(a.degrees[6], 5); // penultimate is the dominant
        assert!(a.degrees[7] == 1 || a.degrees[7] == 6); // authentic or deceptive
    }
}
