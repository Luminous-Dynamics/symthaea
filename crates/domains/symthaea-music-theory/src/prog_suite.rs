// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progressive Folk/Rock (style-roadmap item, Group 3 "stress the
//! engine"): LONG FORM built from a genuine mid-piece METER CHANGE —
//! contrasting sections in different time signatures, not just different
//! keys or textures.
//!
//! Every prior style this session composed its whole piece in ONE meter
//! (`CompositionSpec.meter: u8`, one value, threaded unchanged through
//! [`crate::composer::realize_melody`]/`realize_harmony`/`realize_bass`,
//! all of which take a single `meter_beats: f64` applied uniformly across
//! every section of a [`crate::form::Form`]). A real meter change needs a
//! genuinely different mechanism: this module realizes each section
//! SEPARATELY, in its own meter, and splices the resulting sub-scores
//! onto one running timeline — precisely the pattern [`crate::live`]
//! documents as the caller's job ("the crate gives you the pieces, not a
//! scripted arc"). This is that arc, scripted: four sections, one
//! thematic idea, three meters.
//!
//! - **A** (home key, 4 beats/bar): the theme, as composed.
//! - **B** (home key, 7 beats/bar): an asymmetric riff — the theme's
//!   contrasting transformation (same mechanism [`crate::form::Form::
//!   ternary`] uses for its own B section).
//! - **C** (relative key, 5 beats/bar): a bridge — a second, DIFFERENT
//!   transformation, modulating too (long form travels harmonically as
//!   well as rhythmically).
//! - **ReturnA** (home key, 4 beats/bar): the theme restated verbatim —
//!   the payoff a listener can actually recognize after two meter
//!   changes and a key change.
//!
//! Voice-leading (`prev_upper`/`prev_bass`) carries continuously across
//! every section exactly as it does within one `compose()` call — the
//! harmonic thread never resets even though the meter and key both move.
//!
//! Known simplification: [`crate::score::Score::meter`] can only declare
//! ONE meter for the whole piece (it's a single `u8` field, used for MIDI
//! export and a few structural checks) — this module reports the OPENING
//! section's meter there, same as any consumer reading `.meter` would
//! expect the piece's "home" time signature to be. The audible meter
//! change is real in the note onsets themselves; only the single
//! declared metadata field can't represent more than one value at once.
//!
//! Every section's chord progression comes from the style's own
//! [`crate::spec::CompositionSpec::progression`] (re-seeded per section for
//! variety), not a generic classical grammar — ProgFolk's real "I-V-vi-IV"
//! four-chord loop identity would otherwise never actually sound.

use crate::MusicalIntent;
use crate::form::{Form, Section, SectionRole, contrasting_transform};
use crate::harmony::Key;
use crate::motif::Motif;
use crate::phrase::Period;
use crate::pitch::Pitch;
use crate::rhythm::Duration;
use crate::score::Score;

/// One section's plan: which key, which meter, and which motif
/// transformation (`None` = the theme unaltered).
struct SectionPlan {
    role: SectionRole,
    key_fn: fn(Key) -> Key,
    meter: f64,
    transform: Option<u64>,
}

fn identity_key(k: Key) -> Key {
    k
}

pub(crate) fn realize_prog_suite(
    home_key: Key,
    tempo: f32,
    motif: &Motif,
    seed: u64,
    intent: &MusicalIntent,
    spec: &crate::spec::CompositionSpec,
) -> Score {
    let pivot = motif.notes.iter().find_map(|x| x.degree).unwrap_or(1);
    let b_choice = seed % 3;
    let c_choice = (b_choice + 1) % 3;

    let plan = [
        SectionPlan {
            role: SectionRole::A,
            key_fn: identity_key,
            meter: 4.0,
            transform: None,
        },
        SectionPlan {
            role: SectionRole::B,
            key_fn: identity_key,
            meter: 7.0,
            transform: Some(b_choice),
        },
        SectionPlan {
            role: SectionRole::C,
            key_fn: Key::relative,
            meter: 5.0,
            transform: Some(c_choice),
        },
        SectionPlan {
            role: SectionRole::ReturnA,
            key_fn: identity_key,
            meter: 4.0,
            transform: None,
        },
    ];

    let opening_meter = plan[0].meter as u8;
    let mut score = Score::new(home_key, tempo, opening_meter);
    let mut prev_upper: Vec<Pitch> = Vec::new();
    let mut prev_bass: Option<Pitch> = None;
    let mut cursor = Duration::zero();
    let bars = 4usize;
    let pattern = crate::accompaniment::Accompaniment::Comp; // driving, active — a prog rhythm section

    for (i, sec) in plan.iter().enumerate() {
        let key = (sec.key_fn)(home_key);
        let section_motif = match sec.transform {
            Some(choice) => contrasting_transform(motif, pivot, choice),
            None => motif.clone(),
        };
        let dominant = key.cadence_dominant_degree();
        let seed_variant = seed ^ (0x51CE_u64.wrapping_mul(i as u64 + 1));
        // The style's own declared progression, not the generic classical
        // grammar -- ProgFolk's "I-V-vi-IV" four-chord loop is a real,
        // distinctive vocabulary (unlike e.g. Sonata/Opera's plain
        // [1,4,5,1], which the generic grammar already approximates), and
        // was previously discarded for every section of this form,
        // including its own theme.
        let progression = spec.progression(bars, seed_variant);
        let period = Period::parallel_in(&section_motif, &progression.degrees, sec.meter, dominant);

        let form = Form {
            sections: vec![Section {
                role: sec.role,
                key,
                period,
            }],
        };

        let mut phrase_score = Score::new(key, tempo, sec.meter as u8);
        crate::composer::realize_melody(
            &mut phrase_score,
            &form,
            intent,
            Duration::zero(),
            sec.meter,
            false, // no cross-phrase climax grace — each splice is its own local arc
        );
        // Bass is realized BEFORE harmony so `realize_harmony_measures` can read the
        // ACTUAL sounding bass from the score and voice the upper parts against it
        // (rootless chords + bass-vs-upper parallel fifths, both measured 2026-07-30).
        // Purely a reordering: the two use independent `prev_bass`/`prev_upper` chains
        // and never read each other's state, so the emitted NOTES are unchanged.
        crate::composer::realize_bass(
            &mut phrase_score,
            &form,
            sec.meter,
            intent,
            &mut prev_bass,
            pattern,
            true,
            false,
        );
        crate::composer::realize_harmony(
            &mut phrase_score,
            &form,
            sec.meter,
            intent,
            &mut prev_upper,
            pattern,
            true, // single-section form: the B-thinning gate never fires
            true,
            false,
        );

        for n in &phrase_score.notes {
            let mut shifted = *n;
            shifted.onset = shifted.onset + cursor;
            score.push(shifted);
        }
        cursor = cursor + phrase_score.total_beats;
    }

    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motif::Motif;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent::default()
    }

    fn spec() -> crate::spec::CompositionSpec {
        // ProgFolk is the sole real-world consumer of this form.
        crate::style::Style::ProgFolk.spec()
    }

    #[test]
    fn prog_suite_visits_three_distinct_meters_in_order() {
        let key = Key::major(PitchClass::C);
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let score = realize_prog_suite(key, 100.0, &motif, 5, &intent(), &spec());
        assert!(!score.notes.is_empty(), "a real piece must come out");

        // Reconstruct each section's boundary from the melody voice's
        // rhythmic grouping is fragile; instead confirm structurally via
        // the total length: 4 sections * 2 phrases * `bars` measures *
        // (that section's own meter) beats, using the SAME bars=4 the
        // implementation uses. If the meters were NOT actually applied
        // (e.g. a bug silently reusing 4 everywhere), this total would be
        // 4 * (2*4*4) = 128 beats instead of the real mixed-meter total.
        let uniform_4_only: f64 = 4.0 * (2.0 * 4.0 * 4.0);
        let real_total = score.total_beats.beats();
        assert!(
            (real_total - uniform_4_only).abs() > 1e-6,
            "meters must actually differ per section — got the same total as an all-4/4 piece: {real_total}"
        );
        // The real total: A(2*4*4=32) + B(2*4*7=56) + C(2*4*5=40) +
        // ReturnA(2*4*4=32) = 160 beats exactly.
        assert!(
            (real_total - 160.0).abs() < 1e-6,
            "expected exactly 160 beats from the 4/7/5/4 plan, got {real_total}"
        );
    }

    #[test]
    fn prog_suite_carries_voice_leading_continuously_across_meter_changes() {
        // Two independent runs with the same seed must be deterministic --
        // voice leading threading a mutable prev_upper/prev_bass across four
        // separately-realized sections is exactly the kind of state that's
        // easy to accidentally reset. A different seed must still produce a
        // genuinely different piece via its motif-transformation choice
        // (`b_choice`/`c_choice = seed % 3`) -- NOTE: for ProgFolk
        // specifically, the chord PROGRESSION itself is now seed-invariant
        // (ProgressionSpec::Archetype ignores its seed by design, unlike
        // the old Progression::generate grammar walk this replaced), so the
        // two seeds below are deliberately chosen to differ mod 3, and the
        // comparison uses full note content (pitches), not just count --
        // the duration-preserving transforms this module uses can leave
        // note COUNT identical while every pitch differs.
        let key = Key::major(PitchClass::C);
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let a = realize_prog_suite(key, 100.0, &motif, 9, &intent(), &spec());
        let b = realize_prog_suite(key, 100.0, &motif, 9, &intent(), &spec());
        assert_eq!(a.notes, b.notes, "deterministic for a fixed seed");
        assert_eq!(9 % 3, 0);
        assert_eq!(10 % 3, 1); // genuinely different b_choice/c_choice
        let c = realize_prog_suite(key, 100.0, &motif, 10, &intent(), &spec());
        assert_ne!(
            a.notes, c.notes,
            "a seed with a different motif-transformation choice must produce a genuinely different piece"
        );
    }

    #[test]
    fn every_section_reads_the_specs_own_declared_progression() {
        // Prove `realize_prog_suite` genuinely READS `spec.progression`
        // (rather than still silently generating harmony some other way)
        // by swapping ProgFolk's real "I-V-vi-IV" archetype for a
        // deliberately different one and confirming the piece changes.
        // Checking pitch-class membership against the rendered Bass voice
        // was tried first and is unreliable: the bass can voice-lead onto
        // any TONE of the current chord (not just its root), and the union
        // of {1,4,5,6}'s own chord tones in C major already covers 7 of 12
        // pitch classes -- not a meaningfully distinguishing test. Swapping
        // the whole declared progression and diffing full output sidesteps
        // that ambiguity entirely.
        let key = Key::major(PitchClass::C);
        let motif = Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ]);
        let mut foreign_spec = spec();
        assert_eq!(
            foreign_spec.progression,
            crate::spec::ProgressionSpec::Archetype(vec![1, 5, 6, 4])
        );
        foreign_spec.progression = crate::spec::ProgressionSpec::Archetype(vec![2, 3, 7, 2]);
        for seed in 0..6u64 {
            let real = realize_prog_suite(key, 100.0, &motif, seed, &intent(), &spec());
            let foreign = realize_prog_suite(key, 100.0, &motif, seed, &intent(), &foreign_spec);
            assert_ne!(
                real.notes, foreign.notes,
                "seed {seed}: swapping the spec's declared progression didn't change the \
                 output -- realize_prog_suite isn't actually reading spec.progression"
            );
        }
    }
}
