// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Opera / Art Song (style-roadmap Tier 1 item: "teaches dialogue,
//! character themes, emotional pacing, interruption, dramatic cadence —
//! not singing synthesis first"): the engine's first form built around TWO
//! genuinely independent melodic identities in structured conversation,
//! rather than one melody voice developing alone.
//!
//! Every prior style gives the Melody voice a single, continuous
//! identity — even Renaissance's three "equal voices" are all realized
//! from the SAME imitative subject. This form is different: Theme A
//! (Melody, confident, triadic, rising) and Theme B (CounterMelody,
//! searching, stepwise, falling) are UNRELATED material — different
//! contour, different register, different voice — and the piece is
//! structured as their conversation, not a transformation of one into
//! the other:
//!
//! - **Statement A**: Theme A alone, home key, Melody voice.
//!   CounterMelody silent — a clean solo establishes "character one."
//! - **Statement B**: Theme B alone, the RELATIVE key (real harmonic
//!   otherness, not just a different tune), CounterMelody voice. Melody
//!   silent — "character two," genuinely apart.
//! - **Dialogue**: the two trade, ONE BAR AT A TIME, back in the shared
//!   home key — Melody(A), CounterMelody(B), Melody(A), CounterMelody(B).
//!   Real turn-taking: never do both voices carry the theme in the same
//!   bar.
//! - **Interruption + cadence**: CounterMelody begins Theme B's phrase
//!   but is cut off — literally, notes past a fixed beat cutoff are never
//!   emitted — and Melody enters BEFORE B's phrase would have finished,
//!   with a short cadential tag that resolves the piece. The interruption
//!   is a checkable fact: B's last note in this bar ends at the cutoff,
//!   A's first note starts at the same cutoff, strictly before the bar's
//!   natural end.
//!
//! Harmony and bass are realized normally (`realize_harmony`/
//! `realize_bass` over a real multi-section `Form`, `Accompaniment::
//! Block` so the accompaniment stays out of the dialogue's way) — the
//! novelty here is entirely in how the two THEMES are placed, not in the
//! rhythm section.

use crate::MusicalIntent;
use crate::form::{Form, Section, SectionRole};
use crate::harmony::{Key, Progression};
use crate::motif::Motif;
use crate::phrase::Period;
use crate::rhythm::Duration;
use crate::score::{Emphasis, PartId, Score, ScoreNote, VoiceRole};

/// Character one: confident, triadic, rising to the octave.
pub(crate) fn theme_a() -> Motif {
    Motif::from_degrees(&[
        (1, Duration::quarter()),
        (3, Duration::quarter()),
        (5, Duration::quarter()),
        (8, Duration::quarter()),
    ])
}

/// Character two: searching, stepwise, falling from the leading tone — no
/// pitch-class overlap with Theme A's triad until the very last note (the
/// shared tonic both characters can still agree on).
pub(crate) fn theme_b() -> Motif {
    Motif::from_degrees(&[
        (7, Duration::eighth()),
        (6, Duration::eighth()),
        (4, Duration::quarter()),
        (2, Duration::quarter()),
        (1, Duration::quarter()),
    ])
}

/// The final resolving tag: an arrival on the dominant scale-step then a
/// quick resolve home — sized to exactly fill the 2 beats the interruption
/// leaves behind.
fn cadence_tag() -> Motif {
    Motif::from_degrees(&[(5, Duration::quarter()), (1, Duration::quarter())])
}

/// Alternative Theme-A identities: same rhythm (4 quarter notes, exactly
/// one bar — every timing-based test and `push_theme` call downstream
/// depends only on total duration, never on which degrees fill it) AND the
/// same degree multiset `{1, 3, 5, 8}` as [`theme_a`] — only the ORDER
/// varies, giving genuinely different contours (the census's motif-
/// contour fingerprint layer) while trivially preserving the "themes share
/// almost no scale-degree content" invariant against every Theme-B variant
/// (same set in ⇒ same shared-degree count as the original pair, checked
/// exhaustively in `every_theme_variant_pair_shares_almost_no_degree_
/// content` rather than assumed). Bank[0] is [`theme_a`] itself, so any
/// caller that doesn't go through [`theme_a_for_seed`] sees unchanged
/// behavior.
fn theme_a_variants() -> Vec<Motif> {
    use Duration as D;
    let q = D::quarter();
    vec![
        theme_a(),
        Motif::from_degrees(&[(1, q), (5, q), (3, q), (8, q)]),
        Motif::from_degrees(&[(3, q), (1, q), (5, q), (8, q)]),
        Motif::from_degrees(&[(1, q), (3, q), (8, q), (5, q)]),
    ]
}

/// Alternative Theme-B identities: same rhythm as [`theme_b`] (eighth,
/// eighth, quarter, quarter, quarter — one bar) AND the same degree
/// multiset `{7, 6, 4, 2, 1}` — only the order varies, for the same reason
/// (and the same exhaustively-checked invariant) as [`theme_a_variants`].
/// Bank[0] is [`theme_b`] itself.
fn theme_b_variants() -> Vec<Motif> {
    use Duration as D;
    let (e, q) = (D::eighth(), D::quarter());
    vec![
        theme_b(),
        Motif::from_degrees(&[(6, e), (7, e), (4, q), (2, q), (1, q)]),
        Motif::from_degrees(&[(7, e), (6, e), (2, q), (4, q), (1, q)]),
        Motif::from_degrees(&[(6, e), (7, e), (2, q), (4, q), (1, q)]),
    ]
}

/// Theme A for this piece's seed — the census (`c11cfa43b7`) found Opera's
/// two themes were 100% hardcoded constants with ZERO seed influence on
/// melodic identity (only the harmony/bass accompaniment varied), the most
/// severe of the four near-zero-diversity "bypass grammar" styles. Distinct
/// salt from theme_b's and from `section_of`'s progression reseeding
/// (`0xA91A...`) so the three decisions vary independently.
fn theme_a_for_seed(seed: u64) -> Motif {
    let bank = theme_a_variants();
    let idx = (crate::hook::scramble(seed, 0x0F0A_7EA5_0001) as usize) % bank.len();
    bank[idx].clone()
}

/// Theme B for this piece's seed — see [`theme_a_for_seed`].
fn theme_b_for_seed(seed: u64) -> Motif {
    let bank = theme_b_variants();
    let idx = (crate::hook::scramble(seed, 0x0F0A_7EA5_0002) as usize) % bank.len();
    bank[idx].clone()
}

/// Place `motif`'s notes into `role` starting at `start`, resolved against
/// `key` directly (not `score.key`, which `fugue::emit` would use — Theme
/// B's Statement section needs a genuinely different key than the score's
/// home). `cutoff` truncates the motif: any note whose onset would land at
/// or past `cutoff` beats from `start` is never emitted — the
/// interruption mechanism.
#[allow(clippy::too_many_arguments)]
fn push_theme(
    score: &mut Score,
    motif: &Motif,
    key: Key,
    start: Duration,
    role: VoiceRole,
    octave: i32,
    intensity: f32,
    cutoff: Option<Duration>,
) {
    let scale = key.scale();
    let mut t = Duration::zero();
    for n in &motif.notes {
        if let Some(max) = cutoff {
            if t.beats() >= max.beats() - 1e-9 {
                break;
            }
        }
        if let Some(d) = n.degree {
            score.push(ScoreNote {
                part: PartId::UNASSIGNED,
                pitch: scale.degree_pitch(d, octave),
                onset: start + t,
                duration: n.duration,
                velocity: (0.6 * intensity).clamp(0.1, 1.0),
                role,
                emphasis: Emphasis::Normal,
                section_intensity: intensity,
            });
        }
        t = t + n.duration;
    }
}

pub(crate) fn realize_opera(
    home_key: Key,
    tempo: f32,
    meter: f64,
    seed: u64,
    intent: &MusicalIntent,
) -> Score {
    let bar = Duration::new(meter as i64, 1);
    let relative_key = home_key.relative();
    let melody_octave = 5;
    let counter_octave = 4;

    // Real harmonic support across the whole piece, built from an
    // ordinary multi-section Form — melody content is discarded (harmony/
    // bass only ever read `Period::{antecedent,consequent}.progression`,
    // never `.line`), and the actual themes are pushed separately below.
    // NOTE: `Period::parallel_in` builds BOTH halves from the SAME
    // progression (antecedent + consequent, each `bars` long) — passing
    // it a `bars`-bar progression yields a `2*bars`-bar period. Since this
    // form wants EXACT bar counts (2, 2, 4, 1) rather than the doubled
    // classical-period shape, the Period is built directly here instead,
    // with the full progression in the antecedent and an empty consequent.
    let dummy = theme_a();
    let section_of = |role: SectionRole, key: Key, bars: usize, group: u64| -> Section {
        let seed_variant = seed ^ (0xA91A_u64.wrapping_mul(group + 1));
        let progression = Progression::generate(bars, seed_variant);
        let period = Period {
            antecedent: crate::phrase::Phrase {
                line: dummy.clone(),
                progression: progression.degrees,
                cadence: crate::cadence::Cadence::Authentic,
            },
            consequent: crate::phrase::Phrase {
                line: dummy.clone(),
                progression: Vec::new(),
                cadence: crate::cadence::Cadence::Authentic,
            },
        };
        Section { role, key, period }
    };
    let form = Form {
        sections: vec![
            section_of(SectionRole::A, home_key, 2, 0),
            section_of(SectionRole::B, relative_key, 2, 1),
            section_of(SectionRole::C, home_key, 4, 2),
            section_of(SectionRole::ReturnA, home_key, 1, 3),
        ],
    };

    let mut score = Score::new(home_key, tempo, meter as u8);
    let mut prev_upper: Vec<crate::pitch::Pitch> = Vec::new();
    let mut prev_bass: Option<crate::pitch::Pitch> = None;
    let pattern = crate::accompaniment::Accompaniment::Block;
    // Bass is realized BEFORE harmony so `realize_harmony_measures` can read the
    // ACTUAL sounding bass from the score and voice the upper parts against it
    // (rootless chords + bass-vs-upper parallel fifths, both measured 2026-07-30).
    // Purely a reordering: the two use independent `prev_bass`/`prev_upper` chains
    // and never read each other's state, so the emitted NOTES are unchanged.
    crate::composer::realize_bass(
        &mut score,
        &form,
        meter,
        intent,
        &mut prev_bass,
        pattern,
        false,
        false,
    );
    crate::composer::realize_harmony(
        &mut score,
        &form,
        meter,
        intent,
        &mut prev_upper,
        pattern,
        false,
        false,
        false,
    );

    let a = theme_a_for_seed(seed);
    let b = theme_b_for_seed(seed);
    let cadence = cadence_tag();

    // Statement A: bars 0-1, home key, Melody alone.
    for bar_idx in 0..2i64 {
        push_theme(
            &mut score,
            &a,
            home_key,
            bar.scale(bar_idx, 1),
            VoiceRole::Melody,
            melody_octave,
            SectionRole::A.intensity(),
            None,
        );
    }
    // Statement B: bars 2-3, RELATIVE key, CounterMelody alone.
    for bar_idx in 2..4i64 {
        push_theme(
            &mut score,
            &b,
            relative_key,
            bar.scale(bar_idx, 1),
            VoiceRole::CounterMelody,
            counter_octave,
            SectionRole::B.intensity(),
            None,
        );
    }
    // Dialogue: bars 4-7, home key, alternating one bar at a time.
    for (i, bar_idx) in (4..8i64).enumerate() {
        let onset = bar.scale(bar_idx, 1);
        if i % 2 == 0 {
            push_theme(
                &mut score,
                &a,
                home_key,
                onset,
                VoiceRole::Melody,
                melody_octave,
                SectionRole::C.intensity(),
                None,
            );
        } else {
            push_theme(
                &mut score,
                &b,
                home_key,
                onset,
                VoiceRole::CounterMelody,
                counter_octave,
                SectionRole::C.intensity(),
                None,
            );
        }
    }
    // Interruption + cadence: bar 8. CounterMelody(B) cut off at beat 2;
    // Melody(A's cadential tag) enters immediately, before B's phrase
    // would naturally have finished.
    let final_bar = bar.scale(8, 1);
    let cutoff = Duration::new(2, 1);
    push_theme(
        &mut score,
        &b,
        home_key,
        final_bar,
        VoiceRole::CounterMelody,
        counter_octave,
        SectionRole::ReturnA.intensity(),
        Some(cutoff),
    );
    push_theme(
        &mut score,
        &cadence,
        home_key,
        final_bar + cutoff,
        VoiceRole::Melody,
        melody_octave,
        SectionRole::ReturnA.intensity(),
        None,
    );
    if let Some(last) = score
        .notes
        .iter_mut()
        .filter(|n| n.role == VoiceRole::Melody)
        .max_by(|x, y| x.onset.beats().total_cmp(&y.onset.beats()))
    {
        last.emphasis = Emphasis::Cadential;
    }

    score
        .notes
        .sort_by(|x, y| x.onset.beats().total_cmp(&y.onset.beats()));
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent::default()
    }

    #[test]
    fn theme_a_and_theme_b_are_genuinely_different_material() {
        let a_degrees: Vec<i32> = theme_a().notes.iter().filter_map(|n| n.degree).collect();
        let b_degrees: Vec<i32> = theme_b().notes.iter().filter_map(|n| n.degree).collect();
        assert_ne!(a_degrees, b_degrees);
        // Barely any shared scale-degree content — two different ideas,
        // not a transform of one into the other.
        let shared = a_degrees.iter().filter(|d| b_degrees.contains(d)).count();
        assert!(
            shared <= 1,
            "themes should share almost no scale-degree content: a={a_degrees:?} b={b_degrees:?}"
        );
    }

    #[test]
    fn every_theme_variant_pair_shares_almost_no_degree_content() {
        // Same invariant as theme_a_and_theme_b_are_genuinely_different_
        // material, checked across the WHOLE bank each seed can draw from
        // — a real diversity fix must not accidentally introduce a pair of
        // themes that read as the same idea twice.
        let a_bank = theme_a_variants();
        let b_bank = theme_b_variants();
        for (i, a) in a_bank.iter().enumerate() {
            for (j, b) in b_bank.iter().enumerate() {
                let a_degrees: Vec<i32> = a.notes.iter().filter_map(|n| n.degree).collect();
                let b_degrees: Vec<i32> = b.notes.iter().filter_map(|n| n.degree).collect();
                let shared = a_degrees.iter().filter(|d| b_degrees.contains(d)).count();
                assert!(
                    shared <= 1,
                    "theme_a[{i}]={a_degrees:?} vs theme_b[{j}]={b_degrees:?} share too much"
                );
            }
        }
    }

    #[test]
    fn theme_variants_preserve_the_original_rhythm_exactly() {
        // Every variant must total exactly one bar (4 beats) with the SAME
        // per-note durations as the canonical theme — every timing-based
        // test below (statement bar occupancy, the interruption cutoff)
        // depends only on duration, never on which degrees fill it.
        let a_rhythm: Vec<Duration> = theme_a().notes.iter().map(|n| n.duration).collect();
        for (i, m) in theme_a_variants().iter().enumerate() {
            let r: Vec<Duration> = m.notes.iter().map(|n| n.duration).collect();
            assert_eq!(r, a_rhythm, "theme_a variant {i} changed rhythm");
        }
        let b_rhythm: Vec<Duration> = theme_b().notes.iter().map(|n| n.duration).collect();
        for (i, m) in theme_b_variants().iter().enumerate() {
            let r: Vec<Duration> = m.notes.iter().map(|n| n.duration).collect();
            assert_eq!(r, b_rhythm, "theme_b variant {i} changed rhythm");
        }
    }

    #[test]
    fn theme_selection_actually_varies_with_seed() {
        // The bug this fixes: realize_opera used to call theme_a()/
        // theme_b() directly, so EVERY piece in this style had byte-
        // identical melodic material regardless of seed (confirmed by the
        // census: ~0.0 median within-style nearest-neighbor distance).
        let mut a_seen = std::collections::HashSet::new();
        let mut b_seen = std::collections::HashSet::new();
        for seed in 0u64..40 {
            a_seen.insert(format!("{:?}", theme_a_for_seed(seed)));
            b_seen.insert(format!("{:?}", theme_b_for_seed(seed)));
        }
        assert!(
            a_seen.len() > 1,
            "theme A must vary across seeds, got {} distinct value(s)",
            a_seen.len()
        );
        assert!(
            b_seen.len() > 1,
            "theme B must vary across seeds, got {} distinct value(s)",
            b_seen.len()
        );
    }

    #[test]
    fn statement_sections_isolate_one_voice_each() {
        let home = Key::major(PitchClass::C);
        let score = realize_opera(home, 100.0, 4.0, 3, &intent());
        let bar = Duration::new(4, 1);
        let in_bar = |role: VoiceRole, bar_idx: i64| {
            score.notes.iter().any(|n| {
                n.role == role
                    && n.onset.beats() >= bar.scale(bar_idx, 1).beats() - 1e-9
                    && n.onset.beats() < bar.scale(bar_idx + 1, 1).beats() - 1e-9
            })
        };
        // Statement A (bars 0-1): Melody present, CounterMelody absent.
        for b in 0..2 {
            assert!(in_bar(VoiceRole::Melody, b), "bar {b} should carry Theme A");
            assert!(
                !in_bar(VoiceRole::CounterMelody, b),
                "bar {b} (Statement A) must not have CounterMelody"
            );
        }
        // Statement B (bars 2-3): CounterMelody present, Melody absent.
        for b in 2..4 {
            assert!(
                in_bar(VoiceRole::CounterMelody, b),
                "bar {b} should carry Theme B"
            );
            assert!(
                !in_bar(VoiceRole::Melody, b),
                "bar {b} (Statement B) must not have Melody"
            );
        }
    }

    #[test]
    fn dialogue_alternates_the_active_voice_one_bar_at_a_time() {
        let home = Key::major(PitchClass::C);
        let score = realize_opera(home, 100.0, 4.0, 3, &intent());
        let bar = Duration::new(4, 1);
        let in_bar = |role: VoiceRole, bar_idx: i64| {
            score.notes.iter().any(|n| {
                n.role == role
                    && n.onset.beats() >= bar.scale(bar_idx, 1).beats() - 1e-9
                    && n.onset.beats() < bar.scale(bar_idx + 1, 1).beats() - 1e-9
            })
        };
        let expect_melody = [4i64, 6];
        let expect_counter = [5i64, 7];
        for b in expect_melody {
            assert!(in_bar(VoiceRole::Melody, b));
            assert!(!in_bar(VoiceRole::CounterMelody, b));
        }
        for b in expect_counter {
            assert!(in_bar(VoiceRole::CounterMelody, b));
            assert!(!in_bar(VoiceRole::Melody, b));
        }
    }

    #[test]
    fn interruption_cuts_b_off_and_a_enters_before_the_bar_ends() {
        let home = Key::major(PitchClass::C);
        let score = realize_opera(home, 100.0, 4.0, 3, &intent());
        let bar8_start = Duration::new(4, 1).scale(8, 1).beats();
        let bar8_end = bar8_start + 4.0;

        let counter_in_bar8: Vec<f64> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::CounterMelody
                    && n.onset.beats() >= bar8_start - 1e-9
                    && n.onset.beats() < bar8_end - 1e-9
            })
            .map(|n| (n.onset + n.duration).beats() - bar8_start)
            .collect();
        assert!(!counter_in_bar8.is_empty());
        let counter_end = counter_in_bar8.iter().cloned().fold(0.0, f64::max);
        assert!(
            (counter_end - 2.0).abs() < 1e-6,
            "B's phrase must end exactly at the cutoff (2.0), got {counter_end}"
        );

        let melody_in_bar8: Vec<f64> = score
            .notes
            .iter()
            .filter(|n| {
                n.role == VoiceRole::Melody
                    && n.onset.beats() >= bar8_start - 1e-9
                    && n.onset.beats() < bar8_end - 1e-9
            })
            .map(|n| n.onset.beats() - bar8_start)
            .collect();
        assert!(!melody_in_bar8.is_empty());
        let melody_start = melody_in_bar8.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            (melody_start - 2.0).abs() < 1e-6,
            "A must enter exactly where B was cut off, got {melody_start}"
        );
        assert!(
            melody_start < 4.0 - 1e-9,
            "A's entry must be strictly before the bar's natural end — the interruption"
        );
    }

    #[test]
    fn realize_opera_composes_a_real_nine_bar_piece() {
        let home = Key::major(PitchClass::C);
        let score = realize_opera(home, 100.0, 4.0, 5, &intent());
        assert!(!score.notes.is_empty(), "a real piece must come out");
        let expected_beats = Duration::new(4, 1).scale(9, 1).beats();
        assert!((score.total_beats.beats() - expected_beats).abs() < 1e-6);
        let last_melody = score
            .notes
            .iter()
            .filter(|n| n.role == VoiceRole::Melody)
            .max_by(|x, y| x.onset.beats().total_cmp(&y.onset.beats()))
            .unwrap();
        assert_eq!(last_melody.emphasis, Emphasis::Cadential);
    }

    #[test]
    fn opera_is_deterministic() {
        let home = Key::major(PitchClass::C);
        let a = realize_opera(home, 100.0, 4.0, 7, &intent());
        let b = realize_opera(home, 100.0, 4.0, 7, &intent());
        assert_eq!(a.notes.len(), b.notes.len());
    }
}
