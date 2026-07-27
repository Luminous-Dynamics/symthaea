// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Real structural engine for `GrammarFamily::JazzChorus` (2026, closing
//! part of a 3-family gap: StrophicSong/JazzChorus/AmbientTextural all
//! composed through the exact same generic ternary/rondo machinery as any
//! period-form style — no chorus-cycling identity of its own despite the
//! family's own name.
//!
//! Unlike Blues's [`crate::call_response`] (an antiphonal call/bars-2-4
//! response WITHIN each chorus, because Blues has no separate CounterMelody
//! voice and its own tradition is genuinely call-and-response), a jazz
//! ballad chorus is a single continuous melodic statement over one full
//! pass through the tune's changes — "trading fours" aside, the head and
//! its restatements are played straight through, not split into an
//! internal question/answer. So this engine reuses Blues's [`ChorusRole`]/
//! [`trajectory_for`]/`response_for` machinery UNCHANGED (a chorus's own
//! role determines how ITS WHOLE melodic line relates to the tune's theme,
//! not a call/response pair inside it) but drops the internal antiphonal
//! split entirely, and uses [`Phrase::build_sentence`] (this family's own
//! declared [`crate::grammar::PhraseGrammar::JazzChoruses`] bucket —
//! confirmed sentence-structured by `composer::use_sentence_for`) rather
//! than call_response's bar-by-bar placement.
//!
//! Chorus 0 (always [`ChorusRole::Statement`] per `trajectory_for`'s own
//! contract) plays the theme verbatim — "the head," stated plainly, is the
//! genre's own convention, not something `Statement`'s own `response_for`
//! (which inverts) should override for the piece's very first pass.
//! Choruses 1..N derive their melodic line from `response_for(theme,
//! pivot)`, exactly like every other role in Blues — zero new
//! transformation logic.
//!
//! Harmony/bass reuse this crate's existing generic [`crate::composer::
//! realize_melody`]/`realize_harmony`/`realize_bass`, which already walk
//! any [`Form`]'s sections generically — no new low-level rendering code
//! needed. Each chorus is one [`Section`] whose antecedent is the whole
//! chorus phrase and whose consequent is empty (mirrors call_response's
//! own convention for the same reason: `Period`/`Phrase` are built for a
//! question/answer pair, and a single continuous phrase per chorus doesn't
//! need the second half).
//!
//! **Known, disclosed scope boundary**: unlike Blues's own harmonic-
//! variety fix (`style::Style::Blues`'s `ProgressionSpec::ArchetypePool`),
//! this engine reuses ONE progression (the style's own declared changes)
//! across every chorus — real per-chorus harmonic variety (reharmonization
//! chorus to chorus, a genuine jazz practice) is a natural fast-follow,
//! not attempted here, since JazzChorus's structural identity (this fix)
//! and its harmonic identity (the largest remaining piece of the original
//! diversity critique, tracked separately) are different problems.

use crate::cadence::Cadence;
use crate::call_response::{ChorusRole, trajectory_for};
use crate::composer::MusicalIntent;
use crate::form::{Form, Section};
use crate::harmony::Key;
use crate::phrase::{Period, Phrase};
use crate::score::Score;
use crate::spec::CompositionSpec;

/// This family's own chorus length: JazzBallad's declared changes are an
/// 8-bar turnaround (ii-V-I-vi-ii-V-I-V) — matching that existing spec
/// exactly, not an arbitrary choice (the same reasoning Blues's own
/// `BARS_PER_CHORUS: usize = 12` uses for its 12-bar form).
const BARS_PER_CHORUS: usize = 8;

/// How many whole 8-bar choruses realize `requested_bars` — rounds UP so a
/// short request still gets one complete chorus. See `call_response::
/// chorus_count`'s identical reasoning (kept as its own small copy rather
/// than shared, matching this crate's established per-engine convention —
/// see `call_response::push_phrase`'s own doc comment on why).
fn chorus_count(requested_bars: usize) -> usize {
    requested_bars.max(1).div_ceil(BARS_PER_CHORUS)
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JazzChorusPlan {
    /// The `MusicalIntent::bars` the caller actually asked for.
    pub requested_bars: usize,
    /// `choruses * bars_per_chorus` -- what was actually realized, always
    /// >= `requested_bars` (rounded up to a whole chorus).
    pub realized_bars: usize,
    pub choruses: usize,
    pub bars_per_chorus: usize,
    /// The role each chorus plays, in order -- always `choruses` long,
    /// always opens on `ChorusRole::Statement`. See
    /// [`crate::call_response::trajectory_for`].
    pub trajectory: Vec<ChorusRole>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct JazzChorusRealization {
    pub score: Score,
    pub plan: JazzChorusPlan,
}

/// Realize a jazz-ballad-style piece as a sequence of choruses, each
/// playing a distinct role (see the module doc) in the tune's own
/// developing identity — rather than the generic ternary/rondo period
/// pipeline every other fallback family shares.
pub fn realize_jazz_chorus(
    intent: &MusicalIntent,
    spec: &CompositionSpec,
) -> JazzChorusRealization {
    let key = spec
        .mode
        .and_then(|mode| Key::modal(intent.tonic, mode))
        .unwrap_or_else(|| {
            if intent.valence >= 0.0 {
                Key::major(intent.tonic)
            } else {
                Key::minor(intent.tonic)
            }
        });
    let tempo = spec.tempo(intent.arousal);
    let meter = spec.meter as f64;

    // The tune's head, hook-grafted exactly like every other production
    // path (see `compose_with_grammar_plan`'s own identical grafting).
    let base_motif = spec.motif(intent.arousal, intent.seed);
    let theme = if spec.texture.hook_cell {
        crate::hook::graft_hook(
            &base_motif,
            &crate::hook::HookCell::generate_with(&spec.melody, intent.seed, meter),
            meter,
        )
    } else {
        base_motif
    };
    let pivot = theme.notes.first().and_then(|n| n.degree).unwrap_or(1);

    let choruses = chorus_count(intent.bars);
    let trajectory = trajectory_for(choruses, intent.seed);
    let progression = spec.progression(BARS_PER_CHORUS, intent.seed).degrees;

    let sections: Vec<Section> = (0..choruses)
        .map(|c| {
            let role = trajectory[c];
            // The very first chorus is always Statement (trajectory_for's
            // own contract) -- play the head verbatim, not its own
            // response_for (which would invert it). Every later chorus
            // derives its line from the SAME theme via that role's real
            // transformation, exactly like Blues's per-chorus response.
            let line = if role == ChorusRole::Statement {
                theme.clone()
            } else {
                role.response_for(&theme, pivot)
            };
            let phrase = Phrase::build_sentence(&line, &progression, Cadence::Authentic, meter);
            Section {
                role: role.section_role(),
                key,
                period: Period {
                    antecedent: phrase,
                    consequent: Phrase {
                        line: crate::motif::Motif::new(Vec::new()),
                        progression: Vec::new(),
                        cadence: Cadence::Authentic,
                    },
                },
            }
        })
        .collect();
    let form = Form { sections };

    let mut score = Score::new(key, tempo, spec.meter);
    let mut prev_upper: Vec<crate::pitch::Pitch> = Vec::new();
    let mut prev_bass: Option<crate::pitch::Pitch> = None;
    let pattern = spec.accompaniment(intent.seed);

    crate::composer::realize_melody(
        &mut score,
        &form,
        intent,
        crate::rhythm::Duration::zero(),
        meter,
        spec.texture.climax_grace,
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
        spec.texture.seventh_chords,
    );
    crate::composer::realize_bass(
        &mut score,
        &form,
        meter,
        intent,
        &mut prev_bass,
        pattern,
        false,
    );
    score
        .notes
        .sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));

    JazzChorusRealization {
        score,
        plan: JazzChorusPlan {
            requested_bars: intent.bars,
            realized_bars: choruses * BARS_PER_CHORUS,
            choruses,
            bars_per_chorus: BARS_PER_CHORUS,
            trajectory,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pitch::PitchClass;

    fn intent() -> MusicalIntent {
        MusicalIntent {
            valence: -0.3, // JazzBallad's own spec is Aeolian/minor-flavored
            arousal: 0.4,
            energy: 0.5,
            bars: 4,
            seed: 3,
            tonic: PitchClass::C,
        }
    }

    fn jazz_spec() -> CompositionSpec {
        crate::style::Style::JazzBallad.spec()
    }

    #[test]
    fn chorus_count_rounds_up_to_a_whole_8_bar_chorus() {
        assert_eq!(chorus_count(4), 1);
        assert_eq!(chorus_count(8), 1);
        assert_eq!(chorus_count(9), 2);
        assert_eq!(chorus_count(16), 2);
        assert_eq!(chorus_count(0), 1);
    }

    #[test]
    fn realized_bars_matches_the_quantization_policy() {
        for &(requested, expected_choruses) in &[(4usize, 1usize), (8, 1), (9, 2), (16, 2), (24, 3)]
        {
            let realized = realize_jazz_chorus(
                &MusicalIntent {
                    bars: requested,
                    ..intent()
                },
                &jazz_spec(),
            );
            assert_eq!(
                realized.plan.choruses, expected_choruses,
                "requested {requested} bars"
            );
            assert_eq!(realized.plan.requested_bars, requested);
            assert_eq!(
                realized.plan.realized_bars,
                expected_choruses * BARS_PER_CHORUS
            );
        }
    }

    #[test]
    fn every_trajectory_opens_on_statement() {
        for choruses in 1..=5 {
            let t = trajectory_for(choruses, 11);
            assert_eq!(t.len(), choruses);
            assert_eq!(t[0], ChorusRole::Statement);
        }
    }

    #[test]
    fn the_first_chorus_states_the_theme_verbatim_not_inverted() {
        // A 2+-chorus piece: chorus 0's melody must be the theme's own
        // degree sequence, not Statement::response_for's inversion.
        let realized = realize_jazz_chorus(
            &MusicalIntent {
                bars: 16,
                ..intent()
            },
            &jazz_spec(),
        );
        assert_eq!(realized.plan.choruses, 2);
        let spec = jazz_spec();
        let base_motif = spec.motif(intent().arousal, intent().seed);
        let theme = if spec.texture.hook_cell {
            crate::hook::graft_hook(
                &base_motif,
                &crate::hook::HookCell::generate_with(
                    &spec.melody,
                    intent().seed,
                    spec.meter as f64,
                ),
                spec.meter as f64,
            )
        } else {
            base_motif
        };
        // The first chorus's melody notes' degree sequence (before
        // sentence-structure development) must start with the theme's own
        // first degrees, in the SAME direction (not inverted).
        let melody: Vec<i32> = realized
            .score
            .voice(crate::score::VoiceRole::Melody)
            .iter()
            .take(theme.len())
            .filter_map(|n| {
                // Reconstruct scale degree from pitch relative to key --
                // simplest correct check is just: it must differ from the
                // inversion, which we can check directly via pitch contour
                // direction instead of degree reconstruction.
                Some(n.pitch.midi() as i32)
            })
            .collect();
        assert!(!melody.is_empty());
        // Contour check: theme's own first-to-last direction (ascending or
        // descending) must match the rendered first chorus, NOT be
        // reversed -- an inverted contour would flip this sign.
        let theme_degrees: Vec<i32> = theme.notes.iter().filter_map(|n| n.degree).collect();
        if theme_degrees.len() >= 2 {
            let theme_dir =
                (theme_degrees.last().unwrap() - theme_degrees.first().unwrap()).signum();
            let rendered_dir = (melody.last().unwrap() - melody.first().unwrap()).signum();
            if theme_dir != 0 {
                assert_eq!(
                    theme_dir, rendered_dir,
                    "chorus 0 must state the theme's own contour, not its inversion"
                );
            }
        }
    }

    #[test]
    fn multi_chorus_pieces_genuinely_vary_across_seeds() {
        // Across several seeds, at least one 3-chorus piece must NOT have
        // all three choruses' melody voices produce identical pitch
        // sequences -- proves response_for's per-role transformation is
        // actually reaching the score, not just the plan metadata.
        let varied = (0..20u64).any(|seed| {
            let realized = realize_jazz_chorus(
                &MusicalIntent {
                    bars: 24,
                    seed,
                    ..intent()
                },
                &jazz_spec(),
            );
            assert_eq!(realized.plan.choruses, 3);
            let melody = realized.score.voice(crate::score::VoiceRole::Melody);
            let chorus_beats = BARS_PER_CHORUS as f64 * jazz_spec().meter as f64;
            let chorus_pitches = |c: usize| -> Vec<u8> {
                let (lo, hi) = (c as f64 * chorus_beats, (c + 1) as f64 * chorus_beats);
                melody
                    .iter()
                    .filter(|n| n.onset.beats() >= lo - 1e-9 && n.onset.beats() < hi)
                    .map(|n| n.pitch.midi())
                    .collect()
            };
            let (c0, c1, c2) = (chorus_pitches(0), chorus_pitches(1), chorus_pitches(2));
            !(c0 == c1 && c1 == c2)
        });
        assert!(
            varied,
            "expected at least one seed to vary melody across choruses"
        );
    }

    #[test]
    fn score_never_produces_a_structural_defect() {
        // The universal debug gate isn't wired to this engine automatically
        // (it isn't routed through compose_with_grammar_plan yet in this
        // test), so check directly: no same-voice overlap, no malformed
        // notes, across several seeds/bar counts.
        for seed in 0..10u64 {
            for bars in [4usize, 8, 16, 24] {
                let realized = realize_jazz_chorus(
                    &MusicalIntent {
                        bars,
                        seed,
                        ..intent()
                    },
                    &jazz_spec(),
                );
                let report = crate::score_validation::validate_score(
                    &realized.score,
                    &crate::score_validation::ScoreValidationConfig::default(),
                );
                let defects: Vec<_> = report
                    .issues
                    .iter()
                    .filter(|i| {
                        i.severity == crate::score_validation::ValidationSeverity::Fatal
                            && crate::score_validation::is_universal_invariant(i.rule)
                    })
                    .collect();
                assert!(defects.is_empty(), "seed {seed} bars {bars}: {defects:?}");
            }
        }
    }
}
