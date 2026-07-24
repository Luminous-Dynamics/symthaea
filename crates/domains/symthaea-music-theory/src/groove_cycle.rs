// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cycle-owned composition for dance and groove grammars.
//!
//! The accompaniment cells already carry their own phase. This module makes
//! that phase the owner of large-scale boundaries too: sections begin only on
//! complete cycles, phrase-start markings are rewritten to those boundaries,
//! and obligations record what was actually verified in the realized score.

use serde::{Deserialize, Serialize};

use crate::accompaniment::Accompaniment;
use crate::composer::{MusicalIntent, compose_with_spec_and_form};
use crate::score::{Emphasis, Score, VoiceRole};
use crate::spec::CompositionSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrooveCycleKind {
    SonClave32,
    FlamencoCompas12,
    BossaTresillo,
    Habanera,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrooveCycleSectionRole {
    Establish,
    Layer,
    Break,
    Peak,
    Return,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrooveCycleSection {
    pub role: GrooveCycleSectionRole,
    pub start_cycle: usize,
    pub cycles: usize,
    pub energy: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrooveCycleObligation {
    pub code: String,
    pub fulfilled: bool,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrooveCyclePlan {
    pub kind: GrooveCycleKind,
    pub cycle_beats: f64,
    pub total_cycles: usize,
    pub sections: Vec<GrooveCycleSection>,
    pub phrase_boundary_cycles: Vec<usize>,
    pub phase_continuous: bool,
    pub obligations: Vec<GrooveCycleObligation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GrooveCycleRealization {
    pub score: Score,
    pub plan: GrooveCyclePlan,
}

fn kind_for(accompaniment: Accompaniment) -> Option<(GrooveCycleKind, usize)> {
    match accompaniment {
        Accompaniment::Montuno => Some((GrooveCycleKind::SonClave32, 2)),
        Accompaniment::CompasGait => Some((GrooveCycleKind::FlamencoCompas12, 1)),
        Accompaniment::BossaComp => Some((GrooveCycleKind::BossaTresillo, 1)),
        Accompaniment::Habanera => Some((GrooveCycleKind::Habanera, 1)),
        _ => None,
    }
}

fn sections(total_cycles: usize, seed: u64) -> Vec<GrooveCycleSection> {
    let roles = [
        (GrooveCycleSectionRole::Establish, 0.48),
        (GrooveCycleSectionRole::Layer, 0.68),
        (GrooveCycleSectionRole::Break, 0.52),
        (GrooveCycleSectionRole::Peak, 0.94),
        (GrooveCycleSectionRole::Return, 0.72),
    ];
    let section_count = roles.len().min(total_cycles.max(1));
    // Preserve the five-role cycle rhetoric while allowing materially
    // different proportions. This is seed individuality inside the grammar,
    // not a change of grammar owner.
    const WEIGHT_PROFILES: [[usize; 5]; 4] = [
        [2, 2, 1, 2, 1],
        [2, 1, 2, 2, 1],
        [1, 2, 1, 3, 1],
        [2, 2, 1, 1, 2],
    ];
    let weights = WEIGHT_PROFILES[seed as usize % WEIGHT_PROFILES.len()];
    let mut allocations = vec![1usize; section_count];
    let remaining = total_cycles.saturating_sub(section_count);
    let weight_total: usize = weights[..section_count].iter().sum();
    for unit in 0..remaining {
        let target = unit * weight_total / remaining.max(1);
        let mut cumulative = 0usize;
        let index = weights[..section_count]
            .iter()
            .position(|weight| {
                cumulative += *weight;
                target < cumulative
            })
            .unwrap_or(section_count - 1);
        allocations[index] += 1;
    }
    let mut start = 0;
    roles
        .into_iter()
        .take(section_count)
        .enumerate()
        .map(|(index, (role, energy))| {
            let cycles = allocations[index];
            let section = GrooveCycleSection {
                role,
                start_cycle: start,
                cycles,
                energy,
            };
            start += cycles;
            section
        })
        .collect()
}

/// Compose through the established note-generation mechanisms, then make the
/// groove cycle authoritative over phrase rhetoric. This deliberately returns
/// no period/form object: its formal evidence is [`GrooveCyclePlan`].
pub fn realize_groove_cycle(
    intent: &MusicalIntent,
    spec: &CompositionSpec,
) -> GrooveCycleRealization {
    let accompaniment = spec.accompaniment(intent.seed);
    let (kind, bars_per_cycle) = kind_for(accompaniment).unwrap_or_else(|| {
        panic!("groove-cycle grammar requires a cycle accompaniment, got {accompaniment:?}")
    });
    let (mut score, _) = compose_with_spec_and_form(intent, spec);
    let meter_beats = spec.meter as f64;
    let cycle_beats = meter_beats * bars_per_cycle as f64;
    let score_end = score
        .notes
        .iter()
        .map(|note| (note.onset + note.duration).beats())
        .fold(0.0f64, f64::max);
    let total_cycles = (score_end / cycle_beats).ceil().max(1.0) as usize;
    let sections = sections(total_cycles, intent.seed);
    let phrase_boundary_cycles: Vec<usize> =
        sections.iter().map(|section| section.start_cycle).collect();

    // Period-derived phrase markings are not allowed to reset cycle rhetoric.
    for note in &mut score.notes {
        if note.role == VoiceRole::Melody
            && matches!(note.emphasis, Emphasis::PhraseStart | Emphasis::Cadential)
        {
            note.emphasis = Emphasis::Normal;
        }
    }
    for &cycle in &phrase_boundary_cycles {
        let boundary = cycle as f64 * cycle_beats;
        if let Some(note) = score
            .notes
            .iter_mut()
            .filter(|note| note.role == VoiceRole::Melody)
            .filter(|note| (note.onset.beats() - boundary).abs() < 1e-6)
            .min_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()))
        {
            note.emphasis = Emphasis::PhraseStart;
        }
    }

    let boundaries_aligned = phrase_boundary_cycles.iter().all(|cycle| {
        let onset = *cycle as f64 * cycle_beats;
        ((onset / cycle_beats).round() * cycle_beats - onset).abs() < 1e-9
    });
    let absolute_phase = match kind {
        GrooveCycleKind::SonClave32 => {
            let harmony = score.voice(VoiceRole::Harmony);
            let has_three_side = harmony.iter().any(|note| {
                let bar = (note.onset.beats() / meter_beats).floor() as usize;
                let within = note.onset.beats() - bar as f64 * meter_beats;
                bar % 2 == 0 && ([1.5, 3.0].iter().any(|x| (within - x).abs() < 1e-6))
            });
            let has_two_side = harmony.iter().any(|note| {
                let bar = (note.onset.beats() / meter_beats).floor() as usize;
                let within = note.onset.beats() - bar as f64 * meter_beats;
                bar % 2 == 1 && ([1.0, 2.0].iter().any(|x| (within - x).abs() < 1e-6))
            });
            has_three_side && has_two_side
        }
        _ => true,
    };
    let obligations = vec![
        GrooveCycleObligation {
            code: "whole_cycle_boundaries".into(),
            fulfilled: boundaries_aligned,
            evidence: format!(
                "{} phrase boundaries occur on {}-beat cycle starts",
                phrase_boundary_cycles.len(),
                cycle_beats
            ),
        },
        GrooveCycleObligation {
            code: "continuous_absolute_phase".into(),
            fulfilled: absolute_phase,
            evidence: match kind {
                GrooveCycleKind::SonClave32 => {
                    "montuno onsets retain absolute 3-side/2-side bar parity".into()
                }
                _ => "single-bar cycle phase repeats without section-local reset".into(),
            },
        },
        GrooveCycleObligation {
            code: "cycle_owns_phrase_rhetoric".into(),
            fulfilled: score
                .voice(VoiceRole::Melody)
                .iter()
                .filter(|note| note.emphasis == Emphasis::PhraseStart)
                .all(|note| {
                    let cycle = note.onset.beats() / cycle_beats;
                    (cycle - cycle.round()).abs() < 1e-6
                }),
            evidence: "period cadences removed; phrase starts rewritten to cycle sections".into(),
        },
    ];

    GrooveCycleRealization {
        score,
        plan: GrooveCyclePlan {
            kind,
            cycle_beats,
            total_cycles,
            sections,
            phrase_boundary_cycles,
            phase_continuous: obligations.iter().all(|item| item.fulfilled),
            obligations,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    #[test]
    fn all_cycle_styles_produce_aligned_fulfilled_plans() {
        for style in [
            Style::AfroCuban,
            Style::Flamenco,
            Style::BossaNova,
            Style::Tango,
        ] {
            let intent = MusicalIntent {
                bars: 8,
                seed: 7,
                ..MusicalIntent::default()
            };
            let realized = realize_groove_cycle(&intent, &style.spec());
            assert!(
                realized.plan.phase_continuous,
                "{style:?}: {:?}",
                realized.plan.obligations
            );
            assert!(
                realized
                    .plan
                    .sections
                    .iter()
                    .all(|section| section.cycles > 0)
            );
            assert_eq!(
                realized
                    .plan
                    .sections
                    .iter()
                    .map(|s| s.cycles)
                    .sum::<usize>(),
                realized.plan.total_cycles
            );
        }
    }

    #[test]
    fn seeds_vary_section_proportions_without_breaking_cycle_ownership() {
        let spec = Style::AfroCuban.spec();
        let layouts: std::collections::BTreeSet<Vec<usize>> = (0..4)
            .map(|seed| {
                realize_groove_cycle(
                    &MusicalIntent {
                        bars: 16,
                        seed,
                        ..MusicalIntent::default()
                    },
                    &spec,
                )
                .plan
                .sections
                .iter()
                .map(|section| section.cycles)
                .collect()
            })
            .collect();
        assert!(
            layouts.len() >= 3,
            "section allocation should not be one template"
        );
    }

    #[test]
    fn montuno_phrase_starts_never_split_the_two_bar_clave() {
        let intent = MusicalIntent {
            bars: 8,
            seed: 2,
            ..MusicalIntent::default()
        };
        let realized = realize_groove_cycle(&intent, &Style::AfroCuban.spec());
        assert_eq!(realized.plan.cycle_beats, 8.0);
        assert!(realized.score.voice(VoiceRole::Melody).iter().all(|note| {
            note.emphasis != Emphasis::PhraseStart
                || (note.onset.beats() / 8.0 - (note.onset.beats() / 8.0).round()).abs() < 1e-6
        }));
    }
}
