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
use serde::{Deserialize, Serialize};

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

/// One region in the current Opera/Art Song conversational form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperaSectionPlan {
    pub id: String,
    /// Inclusive start bar, zero-based.
    pub start_bar: u8,
    /// Exclusive end bar, zero-based.
    pub end_bar_exclusive: u8,
    /// `home` or `relative`; the engine currently uses a relative-key
    /// contrast for Statement B.
    pub key_context: String,
}

/// The dramatic contract emitted beside an Opera score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperaPlan {
    pub engine_version: String,
    /// The seed selects the immutable Theme A and Theme B motif variants.
    pub seed: u64,
    pub meter: u8,
    pub total_bars: u8,
    pub sections: Vec<OperaSectionPlan>,
    pub statement_a_role: VoiceRole,
    pub statement_b_role: VoiceRole,
    /// Four one-bar turns in the dialogue region.
    pub dialogue_voice_order: Vec<VoiceRole>,
    pub interruption_bar: u8,
    /// Offset inside the interruption bar at which Theme B is cut off.
    pub interruption_cutoff: Duration,
    /// Theme A's resolving tag enters at exactly the cutoff.
    pub cadence_entry: Duration,
}

/// Verifiable facts extracted from the emitted Opera score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperaVerification {
    pub nine_bar_timeline_matches: bool,
    pub statement_a_isolated: bool,
    pub statement_b_isolated: bool,
    pub statement_a_matches_home_key_theme: bool,
    pub statement_b_uses_relative_key: bool,
    pub dialogue_alternates_one_bar_at_a_time: bool,
    pub interruption_cuts_theme_b_at_exact_boundary: bool,
    pub cadence_enters_at_cutoff: bool,
    pub theme_identities_are_distinct: bool,
}

impl OperaVerification {
    pub fn all_verified(&self) -> bool {
        self.nine_bar_timeline_matches
            && self.statement_a_isolated
            && self.statement_b_isolated
            && self.statement_a_matches_home_key_theme
            && self.statement_b_uses_relative_key
            && self.dialogue_alternates_one_bar_at_a_time
            && self.interruption_cuts_theme_b_at_exact_boundary
            && self.cadence_enters_at_cutoff
            && self.theme_identities_are_distinct
    }
}

/// A score paired with the exact dramatic plan and score-side checks from the
/// same generation pass.
#[derive(Debug, Clone, PartialEq)]
pub struct OperaRealization {
    pub score: Score,
    pub plan: OperaPlan,
    pub verification: OperaVerification,
}

fn opera_plan(meter: u8, seed: u64) -> OperaPlan {
    OperaPlan {
        engine_version: "two-theme-dialogue-v1".into(),
        seed,
        meter,
        total_bars: 9,
        sections: vec![
            OperaSectionPlan {
                id: "statement-a".into(),
                start_bar: 0,
                end_bar_exclusive: 2,
                key_context: "home".into(),
            },
            OperaSectionPlan {
                id: "statement-b".into(),
                start_bar: 2,
                end_bar_exclusive: 4,
                key_context: "relative".into(),
            },
            OperaSectionPlan {
                id: "dialogue".into(),
                start_bar: 4,
                end_bar_exclusive: 8,
                key_context: "home".into(),
            },
            OperaSectionPlan {
                id: "interruption-cadence".into(),
                start_bar: 8,
                end_bar_exclusive: 9,
                key_context: "home".into(),
            },
        ],
        statement_a_role: VoiceRole::Melody,
        statement_b_role: VoiceRole::CounterMelody,
        dialogue_voice_order: vec![
            VoiceRole::Melody,
            VoiceRole::CounterMelody,
            VoiceRole::Melody,
            VoiceRole::CounterMelody,
        ],
        interruption_bar: 8,
        interruption_cutoff: Duration::new(2, 1),
        cadence_entry: Duration::new(2, 1),
    }
}

fn notes_in_bar(
    score: &Score,
    role: VoiceRole,
    bar_index: u8,
    meter: f64,
) -> Vec<ScoreNote> {
    let start = bar_index as f64 * meter;
    let end = start + meter;
    score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            note.role == role
                && note.onset.beats() >= start - 1e-9
                && note.onset.beats() < end - 1e-9
        })
        .collect()
}

pub(crate) fn verify_opera_plan(score: &Score, home_key: Key, plan: &OperaPlan) -> OperaVerification {
    let meter = plan.meter as f64;
    let expected_total = plan.total_bars as f64 * meter;
    let expected_sections = [
        ("statement-a", 0, 2, "home"),
        ("statement-b", 2, 4, "relative"),
        ("dialogue", 4, 8, "home"),
        ("interruption-cadence", 8, 9, "home"),
    ];
    let section_layout_matches = plan.sections.len() == expected_sections.len()
        && plan.sections.iter().zip(expected_sections).all(
            |(section, (id, start, end, key_context))| {
                section.id == id
                    && section.start_bar == start
                    && section.end_bar_exclusive == end
                    && section.key_context == key_context
            },
        );
    let timeline_matches = score.meter == plan.meter
        && (score.total_beats.beats() - expected_total).abs() <= 1e-7
        && plan.total_bars == 9
        && section_layout_matches;

    let statement_a_melody: Vec<_> = (0..2)
        .flat_map(|bar| notes_in_bar(score, VoiceRole::Melody, bar, meter))
        .collect();
    let statement_a_counter: Vec<_> = (0..2)
        .flat_map(|bar| notes_in_bar(score, VoiceRole::CounterMelody, bar, meter))
        .collect();
    let statement_a_isolated = !statement_a_melody.is_empty()
        && statement_a_counter.is_empty()
        && (0..2).all(|bar| !notes_in_bar(score, VoiceRole::Melody, bar, meter).is_empty());

    let statement_b_melody: Vec<_> = (2..4)
        .flat_map(|bar| notes_in_bar(score, VoiceRole::Melody, bar, meter))
        .collect();
    let statement_b_counter: Vec<_> = (2..4)
        .flat_map(|bar| notes_in_bar(score, VoiceRole::CounterMelody, bar, meter))
        .collect();
    let statement_b_isolated = !statement_b_counter.is_empty()
        && statement_b_melody.is_empty()
        && (2..4).all(|bar| !notes_in_bar(score, VoiceRole::CounterMelody, bar, meter).is_empty());
    let relative_key = home_key.relative();
    let expected_a_once = expected_theme_pitches(&theme_a_for_seed(plan.seed), home_key, 5);
    let expected_b_once = expected_theme_pitches(&theme_b_for_seed(plan.seed), relative_key, 4);
    let expected_a: Vec<_> = (0..2).flat_map(|_| expected_a_once.iter().copied()).collect();
    let expected_b: Vec<_> = (0..2).flat_map(|_| expected_b_once.iter().copied()).collect();
    let actual_a: Vec<_> = statement_a_melody.iter().map(|note| note.pitch.midi()).collect();
    let actual_b: Vec<_> = statement_b_counter.iter().map(|note| note.pitch.midi()).collect();
    let statement_a_matches_home_key_theme = actual_a == expected_a;
    let statement_b_uses_relative_key = actual_b == expected_b;

    let expected_dialogue = [
        VoiceRole::Melody,
        VoiceRole::CounterMelody,
        VoiceRole::Melody,
        VoiceRole::CounterMelody,
    ];
    let dialogue_alternates_one_bar_at_a_time =
        plan.dialogue_voice_order.as_slice() == expected_dialogue
        && plan.dialogue_voice_order.len() == 4
        && (0..4u8).all(|index| {
            let bar = 4 + index;
            let expected = plan.dialogue_voice_order[index as usize];
            let other = if expected == VoiceRole::Melody {
                VoiceRole::CounterMelody
            } else {
                VoiceRole::Melody
            };
            !notes_in_bar(score, expected, bar, meter).is_empty()
                && notes_in_bar(score, other, bar, meter).is_empty()
        });

    let interruption_start = plan.interruption_bar as f64 * meter;
    let cutoff = plan.interruption_cutoff.beats();
    let interruption_plan_matches =
        plan.interruption_bar == 8
            && (cutoff - 2.0).abs() <= 1e-7
            && (plan.cadence_entry.beats() - 2.0).abs() <= 1e-7;
    let counter_interruption: Vec<_> = score
        .notes
        .iter()
        .filter(|note| {
            note.role == VoiceRole::CounterMelody
                && note.onset.beats() >= interruption_start - 1e-9
                && note.onset.beats() < interruption_start + meter - 1e-9
        })
        .collect();
    let last_counter_end = counter_interruption
        .iter()
        .map(|note| (note.onset + note.duration).beats())
        .max_by(f64::total_cmp);
    let interruption_cuts_theme_b_at_exact_boundary = interruption_plan_matches
        && last_counter_end
            .is_some_and(|end| (end - (interruption_start + cutoff)).abs() <= 1e-7)
        && counter_interruption.iter().all(|note| {
            (note.onset + note.duration).beats() <= interruption_start + cutoff + 1e-7
        });

    let cadence_start = interruption_start + plan.cadence_entry.beats();
    let cadence_notes: Vec<_> = score
        .notes
        .iter()
        .filter(|note| {
            note.role == VoiceRole::Melody
                && note.onset.beats() >= interruption_start - 1e-9
                && note.onset.beats() < interruption_start + meter - 1e-9
        })
        .collect();
    let cadence_enters_at_cutoff = cadence_notes.len() == 2
        && (cadence_notes[0].onset.beats() - cadence_start).abs() <= 1e-7
        && cadence_notes[1].emphasis == Emphasis::Cadential;

    let mut a_pitches: Vec<_> = statement_a_melody.iter().map(|note| note.pitch.midi()).collect();
    let mut b_pitches: Vec<_> = statement_b_counter.iter().map(|note| note.pitch.midi()).collect();
    a_pitches.sort_unstable();
    b_pitches.sort_unstable();
    let theme_identities_are_distinct = expected_a_once != expected_b_once
        && a_pitches != b_pitches;

    OperaVerification {
        nine_bar_timeline_matches: timeline_matches,
        statement_a_isolated,
        statement_b_isolated,
        statement_a_matches_home_key_theme,
        statement_b_uses_relative_key,
        dialogue_alternates_one_bar_at_a_time,
        interruption_cuts_theme_b_at_exact_boundary,
        cadence_enters_at_cutoff,
        theme_identities_are_distinct,
    }
}

pub(crate) fn realize_opera(
    home_key: Key,
    tempo: f32,
    meter: f64,
    seed: u64,
    intent: &MusicalIntent,
) -> Score {
    realize_opera_with_plan(home_key, tempo, meter, seed, intent).score
}

/// Plan-preserving realization for the common grammar evidence boundary.
/// The canonical score-only entry point delegates here to avoid planner /
/// composer drift.
pub(crate) fn realize_opera_with_plan(
    home_key: Key,
    tempo: f32,
    meter: f64,
    seed: u64,
    intent: &MusicalIntent,
) -> OperaRealization {
    let plan = opera_plan(meter as u8, seed);
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
    let verification = verify_opera_plan(&score, home_key, &plan);
    OperaRealization { score, plan, verification }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent::default()
    }

    #[test]
    fn emitted_opera_plan_verifies_all_nine_bar_dialogue_obligations() {
        let home = Key::major(PitchClass::C);
        for seed in 0..64u64 {
            let realized = realize_opera_with_plan(home, 100.0, 4.0, seed, &intent());
            assert_eq!(realized.plan.engine_version, "two-theme-dialogue-v1");
            assert_eq!(realized.plan.total_bars, 9);
            assert_eq!(realized.plan.sections.len(), 4);
            assert!(
                realized.verification.all_verified(),
                "seed {seed}: {:?}",
                realized.verification
            );
            assert!((realized.score.total_beats.beats() - 36.0).abs() <= 1e-7);
        }
    }

    #[test]
    fn opera_plan_verifier_rejects_a_missing_dialogue_turn() {
        let home = Key::major(PitchClass::C);
        let realized = realize_opera_with_plan(home, 100.0, 4.0, 3, &intent());
        let mut corrupted = realized.score.clone();
        let dialogue_bar = 5.0 * 4.0;
        corrupted.notes.retain(|note| {
            !(note.role == VoiceRole::CounterMelody
                && note.onset.beats() >= dialogue_bar
                && note.onset.beats() < dialogue_bar + 4.0)
        });
        let verification = verify_opera_plan(&corrupted, home, &realized.plan);
        assert!(!verification.dialogue_alternates_one_bar_at_a_time);
        assert!(!verification.all_verified());
    }

    #[test]
    fn opera_plan_verifier_rejects_interruption_that_crosses_the_cutoff() {
        let home = Key::major(PitchClass::C);
        let realized = realize_opera_with_plan(home, 100.0, 4.0, 3, &intent());
        let mut corrupted = realized.score.clone();
        let last_counter = corrupted
            .notes
            .iter_mut()
            .filter(|note| {
                note.role == VoiceRole::CounterMelody
                    && note.onset.beats() >= 32.0
                    && note.onset.beats() < 36.0
            })
            .max_by(|left, right| left.onset.beats().total_cmp(&right.onset.beats()))
            .expect("interruption must have a final Theme B note");
        last_counter.duration = Duration::new(3, 2);
        let verification = verify_opera_plan(&corrupted, home, &realized.plan);
        assert!(!verification.interruption_cuts_theme_b_at_exact_boundary);
        assert!(!verification.all_verified());
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
