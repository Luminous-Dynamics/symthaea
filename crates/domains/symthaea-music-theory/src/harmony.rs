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
        _ => ChordQuality::Major, // fallback for exotic scale spellings
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
        _ => ChordQuality::Dominant7,
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
                // Final chord: authentic (1) most of the time, deceptive (6) sometimes.
                if rng.next_below(4) == 0 { 6 } else { 1 }
            } else if i == bars - 2 {
                5 // penultimate: the dominant, to set up the cadence
            } else {
                match prev_fn {
                    HarmonicFunction::Tonic => {
                        // T → PD (ii/IV) or another T (vi)
                        [2, 4, 6][rng.next_below(3) as usize]
                    }
                    HarmonicFunction::Predominant => {
                        // PD → D (V) usually, sometimes another PD
                        if rng.next_below(4) == 0 { 4 } else { 5 }
                    }
                    HarmonicFunction::Dominant => 1, // D → T
                }
            };
            degrees.push(next);
        }
        Progression::new(degrees)
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
