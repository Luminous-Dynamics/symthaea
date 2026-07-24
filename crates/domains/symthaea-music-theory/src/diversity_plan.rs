// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic high-level diversity plans for candidate frontiers.
//!
//! A seed must vary more than notes. This plan chooses seven independent,
//! inspectable compositional decisions and applies them through existing
//! `CompositionSpec` mechanisms. It never invents a second composition path;
//! the resolved spec remains the authoritative input recorded in the recipe.

use serde::{Deserialize, Serialize};

use crate::spec::{CompositionSpec, DevelopmentDna, FormKind, PhraseRhetoric};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FormalTopology {
    DirectStatement,
    ContrastAndReturn,
    CumulativeArc,
    InterruptionAndRestoration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MotifDevelopmentStrategy {
    LiteralRecurrence,
    Fragmentation,
    SequentialDevelopment,
    TransformedReturn,
    WithholdingAndRoleTransfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HarmonyStrategy {
    NativePacing,
    CadentialAcceleration,
    SequentialMotion,
    SustainedGravity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RhythmStrategy {
    StablePulse,
    AuthoredInterruption,
    SustainedArrival,
    RhythmicLiquidation,
    StructuralSilence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrchestrationTrajectory {
    StableEnsemble,
    SparseToFull,
    RotatingForeground,
    ProgressiveSubtraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClimaxStrategy {
    Restrained,
    ExposedPeak,
    DelayedBuild,
    DistributedPlateau,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EndingStrategy {
    RestoredReturn,
    FragmentedLiquidation,
    DecisiveClosure,
    QuietRecollection,
}

/// Provenance-clean strategy abstraction used only by matched teaching trials.
/// It encodes no pitches, rhythms, harmonies, section lengths, or orchestration
/// copied from the authored etude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompositionLessonStrategy {
    AlteredReturn,
    MotifWithholdingAndRoleTransfer,
    StructuralSilence,
    HarmonicStasisWithChangingTexture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiversityPlan {
    pub formal_topology: FormalTopology,
    pub selected_form: FormKind,
    pub motif_development: MotifDevelopmentStrategy,
    pub harmony: HarmonyStrategy,
    pub rhythm: RhythmStrategy,
    pub orchestration: OrchestrationTrajectory,
    pub climax: ClimaxStrategy,
    pub ending: EndingStrategy,
}

impl DiversityPlan {
    pub fn for_seed(spec: &CompositionSpec, seed: u64) -> Self {
        let selected_form = spec.form_pool[(seed as usize / 3) % spec.form_pool.len().max(1)];
        Self {
            formal_topology: match (seed / 5) % 4 {
                0 => FormalTopology::DirectStatement,
                1 => FormalTopology::ContrastAndReturn,
                2 => FormalTopology::CumulativeArc,
                _ => FormalTopology::InterruptionAndRestoration,
            },
            selected_form,
            motif_development: match (seed / 7) % 4 {
                0 => MotifDevelopmentStrategy::LiteralRecurrence,
                1 => MotifDevelopmentStrategy::Fragmentation,
                2 => MotifDevelopmentStrategy::SequentialDevelopment,
                _ => MotifDevelopmentStrategy::TransformedReturn,
            },
            harmony: match (seed / 11) % 4 {
                0 => HarmonyStrategy::NativePacing,
                1 => HarmonyStrategy::CadentialAcceleration,
                2 => HarmonyStrategy::SequentialMotion,
                _ => HarmonyStrategy::SustainedGravity,
            },
            rhythm: match (seed / 13) % 4 {
                0 => RhythmStrategy::StablePulse,
                1 => RhythmStrategy::AuthoredInterruption,
                2 => RhythmStrategy::SustainedArrival,
                _ => RhythmStrategy::RhythmicLiquidation,
            },
            orchestration: match (seed / 17) % 4 {
                0 => OrchestrationTrajectory::StableEnsemble,
                1 => OrchestrationTrajectory::SparseToFull,
                2 => OrchestrationTrajectory::RotatingForeground,
                _ => OrchestrationTrajectory::ProgressiveSubtraction,
            },
            climax: match (seed / 19) % 4 {
                0 => ClimaxStrategy::Restrained,
                1 => ClimaxStrategy::ExposedPeak,
                2 => ClimaxStrategy::DelayedBuild,
                _ => ClimaxStrategy::DistributedPlateau,
            },
            ending: match (seed / 23) % 4 {
                0 => EndingStrategy::RestoredReturn,
                1 => EndingStrategy::FragmentedLiquidation,
                2 => EndingStrategy::DecisiveClosure,
                _ => EndingStrategy::QuietRecollection,
            },
        }
    }

    /// Resolve the plan through mechanisms the engine already owns.
    pub fn apply(&self, spec: &mut CompositionSpec) {
        spec.form_pool = vec![self.selected_form];
        match self.formal_topology {
            FormalTopology::DirectStatement => {
                spec.texture.intro_bars = 0;
                spec.texture.staged_entrances = false;
            }
            FormalTopology::ContrastAndReturn => {
                spec.texture.staged_entrances = true;
                spec.texture.return_color = true;
            }
            FormalTopology::CumulativeArc => {
                spec.texture.intro_bars = spec.texture.intro_bars.max(1);
                spec.development = DevelopmentDna::Intensifying;
            }
            FormalTopology::InterruptionAndRestoration => {
                spec.texture.thin_departure = true;
                spec.texture.return_color = true;
                spec.rhetoric = PhraseRhetoric::Declamatory;
            }
        }
        match self.motif_development {
            MotifDevelopmentStrategy::LiteralRecurrence => {
                spec.development = DevelopmentDna::Classic;
                spec.texture.damage = spec.texture.damage.min(0.19);
            }
            MotifDevelopmentStrategy::Fragmentation => {
                spec.development = DevelopmentDna::Fragmenting;
                spec.texture.damage = spec.texture.damage.max(0.35);
            }
            MotifDevelopmentStrategy::SequentialDevelopment => {
                spec.development = DevelopmentDna::Sequential;
            }
            MotifDevelopmentStrategy::TransformedReturn => {
                spec.texture.return_color = true;
                spec.texture.damage = spec.texture.damage.max(0.5);
            }
            MotifDevelopmentStrategy::WithholdingAndRoleTransfer => {
                spec.texture.thin_departure = true;
                spec.texture.counter_melody = true;
                spec.texture.return_color = true;
                spec.texture.damage = spec.texture.damage.max(0.42);
            }
        }
        match self.harmony {
            HarmonyStrategy::NativePacing => {}
            HarmonyStrategy::CadentialAcceleration => {
                spec.texture.cadential_harmonic_rhythm = spec.meter == 4;
            }
            HarmonyStrategy::SequentialMotion => spec.texture.harmonic_sequence = true,
            HarmonyStrategy::SustainedGravity => {
                // Preserve culturally/style-specific full-drone ownership;
                // elsewhere this is a local pedal tendency, not a claim that
                // functional harmony disappeared.
                spec.texture.drone = true;
            }
        }
        match self.rhythm {
            RhythmStrategy::StablePulse => spec.texture.held_arrivals = false,
            RhythmStrategy::AuthoredInterruption => {
                spec.rhetoric = PhraseRhetoric::Declamatory;
                spec.texture.thin_departure = true;
            }
            RhythmStrategy::SustainedArrival => {
                spec.rhetoric = PhraseRhetoric::Singing;
                spec.texture.held_arrivals = true;
            }
            RhythmStrategy::RhythmicLiquidation => {
                spec.development = DevelopmentDna::Fragmenting;
                spec.texture.held_arrivals = false;
            }
            RhythmStrategy::StructuralSilence => {
                spec.rhetoric = PhraseRhetoric::Declamatory;
                spec.texture.thin_departure = true;
                spec.texture.held_arrivals = true;
            }
        }
        match self.orchestration {
            OrchestrationTrajectory::StableEnsemble => {}
            OrchestrationTrajectory::SparseToFull => {
                spec.texture.staged_entrances = true;
                spec.texture.counter_melody = true;
            }
            OrchestrationTrajectory::RotatingForeground => {
                spec.texture.counter_melody = true;
                spec.texture.return_color = true;
            }
            OrchestrationTrajectory::ProgressiveSubtraction => {
                spec.texture.thin_departure = true;
                spec.texture.counter_melody = false;
            }
        }
        match self.climax {
            ClimaxStrategy::Restrained => spec.texture.climax_grace = false,
            ClimaxStrategy::ExposedPeak => {
                spec.texture.climax_grace = true;
                spec.texture.damage = spec.texture.damage.max(0.2);
            }
            ClimaxStrategy::DelayedBuild => {
                spec.development = DevelopmentDna::Intensifying;
                spec.texture.held_arrivals = true;
            }
            ClimaxStrategy::DistributedPlateau => {
                spec.texture.climax_grace = false;
                spec.texture.counter_melody = true;
            }
        }
        match self.ending {
            EndingStrategy::RestoredReturn => {
                spec.texture.return_color = true;
                spec.texture.coda_bars = spec.texture.coda_bars.max(1);
            }
            EndingStrategy::FragmentedLiquidation => {
                spec.development = DevelopmentDna::Fragmenting;
                spec.texture.coda_bars = 0;
            }
            EndingStrategy::DecisiveClosure => {
                spec.texture.deceptive_close = false;
                spec.texture.coda_bars = spec.texture.coda_bars.max(2);
            }
            EndingStrategy::QuietRecollection => {
                spec.rhetoric = PhraseRhetoric::Singing;
                spec.texture.coda_bars = spec.texture.coda_bars.max(3);
                spec.texture.damage = spec.texture.damage.min(0.34);
            }
        }
    }

    /// Apply one lesson to plan-level decisions only. The receiving grammar
    /// still owns the actual music, which is the main source-reuse boundary.
    pub fn apply_shadow_lesson(&mut self, lesson: CompositionLessonStrategy) {
        match lesson {
            CompositionLessonStrategy::AlteredReturn => {
                self.formal_topology = FormalTopology::ContrastAndReturn;
                self.motif_development = MotifDevelopmentStrategy::TransformedReturn;
                self.ending = EndingStrategy::RestoredReturn;
            }
            CompositionLessonStrategy::MotifWithholdingAndRoleTransfer => {
                self.formal_topology = FormalTopology::InterruptionAndRestoration;
                self.motif_development = MotifDevelopmentStrategy::WithholdingAndRoleTransfer;
                self.orchestration = OrchestrationTrajectory::RotatingForeground;
            }
            CompositionLessonStrategy::StructuralSilence => {
                self.formal_topology = FormalTopology::CumulativeArc;
                self.rhythm = RhythmStrategy::StructuralSilence;
                self.climax = ClimaxStrategy::Restrained;
            }
            CompositionLessonStrategy::HarmonicStasisWithChangingTexture => {
                self.harmony = HarmonyStrategy::SustainedGravity;
                self.orchestration = OrchestrationTrajectory::SparseToFull;
                self.climax = ClimaxStrategy::DistributedPlateau;
            }
        }
    }

    pub fn difference_count(&self, other: &Self) -> usize {
        usize::from(self.formal_topology != other.formal_topology)
            + usize::from(self.selected_form != other.selected_form)
            + usize::from(self.motif_development != other.motif_development)
            + usize::from(self.harmony != other.harmony)
            + usize::from(self.rhythm != other.rhythm)
            + usize::from(self.orchestration != other.orchestration)
            + usize::from(self.climax != other.climax)
            + usize::from(self.ending != other.ending)
    }
}

/// Choose a frontier whose every pair differs on at least `minimum_axes`
/// whenever the supplied pool makes that possible.
pub fn select_plan_diverse_seeds(
    spec: &CompositionSpec,
    pool: &[u64],
    count: usize,
    minimum_axes: usize,
) -> Vec<u64> {
    let mut selected = Vec::with_capacity(count);
    let mut remaining = pool.to_vec();
    while selected.len() < count && !remaining.is_empty() {
        let best = remaining
            .iter()
            .enumerate()
            .filter_map(|(index, &seed)| {
                let plan = DiversityPlan::for_seed(spec, seed);
                if !selected.iter().all(|&prior| {
                    plan.difference_count(&DiversityPlan::for_seed(spec, prior)) >= minimum_axes
                }) {
                    return None;
                }
                let new_motif = selected.iter().all(|&prior| {
                    plan.motif_development != DiversityPlan::for_seed(spec, prior).motif_development
                });
                let new_ending = selected
                    .iter()
                    .all(|&prior| plan.ending != DiversityPlan::for_seed(spec, prior).ending);
                let minimum_difference = selected
                    .iter()
                    .map(|&prior| plan.difference_count(&DiversityPlan::for_seed(spec, prior)))
                    .min()
                    .unwrap_or(usize::MAX);
                Some((
                    index,
                    usize::from(new_ending) + usize::from(new_motif),
                    minimum_difference,
                ))
            })
            .max_by_key(|(_, priority_axes, minimum_difference)| {
                (*priority_axes, *minimum_difference)
            });
        let Some((index, _, _)) = best else {
            break;
        };
        selected.push(remaining.remove(index));
    }
    if selected.len() >= count {
        return selected;
    }
    for &seed in pool {
        if !selected.contains(&seed) {
            selected.push(seed);
            if selected.len() >= count {
                break;
            }
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::style::Style;

    #[test]
    fn selected_frontier_differs_on_three_major_axes() {
        for style in [
            Style::Classical,
            Style::Passacaglia,
            Style::Tango,
            Style::Sonata,
        ] {
            let spec = style.spec();
            let pool: Vec<u64> = (0..96).collect();
            let selected = select_plan_diverse_seeds(&spec, &pool, 3, 3);
            assert_eq!(selected.len(), 3);
            for i in 0..selected.len() {
                for j in i + 1..selected.len() {
                    let a = DiversityPlan::for_seed(&spec, selected[i]);
                    let b = DiversityPlan::for_seed(&spec, selected[j]);
                    assert!(a.difference_count(&b) >= 3, "{style:?}: {a:?} / {b:?}");
                }
            }
            assert_eq!(
                selected
                    .iter()
                    .map(|&seed| DiversityPlan::for_seed(&spec, seed).motif_development)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                3,
                "{style:?}: motif treatments collapsed"
            );
            assert_eq!(
                selected
                    .iter()
                    .map(|&seed| DiversityPlan::for_seed(&spec, seed).ending)
                    .collect::<std::collections::HashSet<_>>()
                    .len(),
                3,
                "{style:?}: endings collapsed"
            );
        }
    }

    #[test]
    fn applied_plans_remain_valid_specs() {
        for style in [
            Style::Classical,
            Style::Passacaglia,
            Style::Tango,
            Style::JazzBallad,
        ] {
            for seed in 0..48 {
                let mut spec = style.spec();
                DiversityPlan::for_seed(&spec, seed).apply(&mut spec);
                spec.validate()
                    .unwrap_or_else(|errors| panic!("{style:?}/{seed}: {errors:?}"));
            }
        }
    }

    #[test]
    fn first_teaching_wave_maps_to_typed_plan_axes() {
        let spec = Style::Classical.spec();
        let lessons = [
            CompositionLessonStrategy::AlteredReturn,
            CompositionLessonStrategy::MotifWithholdingAndRoleTransfer,
            CompositionLessonStrategy::StructuralSilence,
            CompositionLessonStrategy::HarmonicStasisWithChangingTexture,
        ];
        for lesson in lessons {
            let mut plan = DiversityPlan::for_seed(&spec, 73);
            plan.apply_shadow_lesson(lesson);
            let mut resolved = spec.clone();
            plan.apply(&mut resolved);
            resolved
                .validate()
                .unwrap_or_else(|errors| panic!("{lesson:?}: {errors:?}"));
        }
    }
}
