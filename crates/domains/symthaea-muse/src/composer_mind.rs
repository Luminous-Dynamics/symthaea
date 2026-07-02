// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composer mind: goal-directed planning and long-range form memory.
//!
//! This is Symthaea's compositional intelligence — the ability to think
//! ahead, set goals for what the music should become, remember the opening
//! theme and bring it back, and autonomously choose emotional arcs.
//!
//! Three key capabilities:
//! 1. **Compositional goals**: "build tension for 8 bars, then resolve"
//! 2. **Long-range memory**: opening theme stored for recapitulation
//! 3. **Free composition**: Symthaea chooses her own emotional trajectory

use crate::creative_agency::QualityTier;
use crate::{MusicalState, Note};

/// A compositional goal — what the music should become.
#[derive(Debug, Clone)]
pub struct CompositionGoal {
    /// What kind of musical event to aim for.
    pub target: GoalTarget,
    /// How many bars until the goal should be reached.
    pub bars_remaining: f32,
    /// How urgent is this goal (affects how strongly it shapes decisions).
    pub urgency: f32,
}

/// Types of compositional goals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalTarget {
    /// Build to a climax (increasing dynamics, density, register).
    BuildClimax,
    /// Resolve tension (V→I cadence, dynamics drop, simplify).
    Resolve,
    /// Bring back the opening theme (recapitulation).
    Recapitulate,
    /// Introduce contrasting material (new key, new rhythm).
    Contrast,
    /// Wind down to silence (fade, sparse notes, low register).
    FadeToSilence,
    /// Maintain current energy level.
    Sustain,
}

/// Long-range form memory: remembers themes across the whole composition.
pub struct FormMemory {
    /// The opening theme (first phrase rated Good or better).
    opening_theme: Option<Vec<Note>>,
    /// Best phrase of the whole composition (for callback).
    best_phrase: Option<(Vec<Note>, f32)>, // (notes, beauty_score)
    /// Current section type.
    current_section: SectionArc,
    /// Sections completed.
    sections_completed: usize,
    /// Total bars elapsed.
    bars_elapsed: f32,
}

/// Large-scale section arc for free composition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionArc {
    /// Opening: establish themes and mood.
    Exposition,
    /// Development: explore, vary, build tension.
    Development,
    /// Climax: peak intensity.
    Climax,
    /// Recapitulation: return to opening material.
    Recapitulation,
    /// Coda: wind down, final statement.
    Coda,
}

impl FormMemory {
    pub fn new() -> Self {
        Self {
            opening_theme: None,
            best_phrase: None,
            current_section: SectionArc::Exposition,
            sections_completed: 0,
            bars_elapsed: 0.0,
        }
    }

    /// Record a rated phrase. Saves opening theme and best phrase.
    pub fn record_phrase(&mut self, notes: &[Note], beauty: f32) {
        let tier = QualityTier::from_score(beauty);

        // Save first Good+ phrase as opening theme
        if self.opening_theme.is_none() && tier >= QualityTier::Good {
            self.opening_theme = Some(notes.to_vec());
        }

        // Track best phrase overall
        if let Some((_, best_score)) = &self.best_phrase {
            if beauty > *best_score {
                self.best_phrase = Some((notes.to_vec(), beauty));
            }
        } else if tier >= QualityTier::Good {
            self.best_phrase = Some((notes.to_vec(), beauty));
        }
    }

    /// Get the opening theme for recapitulation.
    pub fn opening_theme(&self) -> Option<&[Note]> {
        self.opening_theme.as_deref()
    }

    /// Get the best phrase (for climactic moments).
    pub fn best_phrase(&self) -> Option<&[Note]> {
        self.best_phrase.as_ref().map(|(n, _)| n.as_slice())
    }

    pub fn current_section(&self) -> SectionArc {
        self.current_section
    }
    pub fn bars_elapsed(&self) -> f32 {
        self.bars_elapsed
    }
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// The composer mind: plans goals and manages form.
pub struct ComposerMind {
    /// Active compositional goals (priority queue by urgency).
    goals: Vec<CompositionGoal>,
    /// Form memory (themes, best phrases, section tracking).
    pub memory: FormMemory,
    /// Bars per section (configurable).
    bars_per_section: f32,
}

impl ComposerMind {
    pub fn new() -> Self {
        Self {
            goals: Vec::new(),
            memory: FormMemory::new(),
            bars_per_section: 16.0, // 16 bars per section (32s at 120bpm)
        }
    }

    /// Advance the form by one bar. Returns any new goals triggered.
    pub fn advance_bar(&mut self, _state: &MusicalState) -> Vec<CompositionGoal> {
        self.memory.bars_elapsed += 1.0;
        let mut new_goals = Vec::new();

        // Decay existing goals
        for goal in &mut self.goals {
            goal.bars_remaining -= 1.0;
        }
        self.goals.retain(|g| g.bars_remaining > 0.0);

        // Section transitions (free composition form)
        let section_progress = self.memory.bars_elapsed / self.bars_per_section;
        let section_idx = section_progress as usize;

        if section_idx > self.memory.sections_completed {
            self.memory.sections_completed = section_idx;
            let new_section = match section_idx {
                0 => SectionArc::Exposition,
                1 => SectionArc::Development,
                2 => SectionArc::Climax,
                3 => SectionArc::Recapitulation,
                _ => SectionArc::Coda,
            };

            // Trigger goals for the new section
            match new_section {
                SectionArc::Exposition => {
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::Sustain,
                        bars_remaining: self.bars_per_section,
                        urgency: 0.5,
                    });
                }
                SectionArc::Development => {
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::Contrast,
                        bars_remaining: 4.0,
                        urgency: 0.7,
                    });
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::BuildClimax,
                        bars_remaining: self.bars_per_section,
                        urgency: 0.8,
                    });
                }
                SectionArc::Climax => {
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::BuildClimax,
                        bars_remaining: 4.0,
                        urgency: 1.0,
                    });
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::Resolve,
                        bars_remaining: self.bars_per_section,
                        urgency: 0.6,
                    });
                }
                SectionArc::Recapitulation => {
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::Recapitulate,
                        bars_remaining: 8.0,
                        urgency: 0.9,
                    });
                }
                SectionArc::Coda => {
                    new_goals.push(CompositionGoal {
                        target: GoalTarget::FadeToSilence,
                        bars_remaining: self.bars_per_section,
                        urgency: 0.8,
                    });
                }
            }

            self.memory.current_section = new_section;
        }

        self.goals.extend(new_goals.clone());
        new_goals
    }

    /// Get the most urgent active goal.
    pub fn primary_goal(&self) -> Option<&CompositionGoal> {
        self.goals
            .iter()
            .max_by(|a, b| a.urgency.total_cmp(&b.urgency))
    }

    /// Apply compositional goals to the musical state.
    /// Returns suggested modifications to consciousness parameters.
    pub fn shape_state(&self, state: &mut MusicalState) {
        if let Some(goal) = self.primary_goal() {
            let urgency = goal.urgency;
            let progress = 1.0 - (goal.bars_remaining / 16.0).clamp(0.0, 1.0);

            match goal.target {
                GoalTarget::BuildClimax => {
                    // Gradually increase arousal, consciousness, density
                    state.arousal += urgency * progress * 0.02;
                    state.consciousness_level += urgency * progress * 0.01;
                    state.dopamine += urgency * progress * 0.015;
                }
                GoalTarget::Resolve => {
                    // Move toward consonance and calm
                    state.prediction_error *= 1.0 - urgency * 0.02;
                    state.arousal *= 1.0 - urgency * 0.01;
                    state.harmony_activations[0] += urgency * 0.02; // ResonantCoherence
                }
                GoalTarget::Recapitulate => {
                    // Boost familiarity harmonies
                    state.harmony_activations[0] += urgency * 0.03;
                    state.prediction_error *= 0.95; // reduce surprise
                }
                GoalTarget::Contrast => {
                    // Increase surprise and exploration
                    state.prediction_error += urgency * 0.02;
                    state.noradrenaline += urgency * 0.01;
                }
                GoalTarget::FadeToSilence => {
                    // Gradual wind-down
                    state.arousal *= 1.0 - urgency * progress * 0.03;
                    state.consciousness_level *= 1.0 - urgency * progress * 0.01;
                    state.harmony_activations[7] += urgency * progress * 0.02; // SacredStillness
                }
                GoalTarget::Sustain => {
                    // Maintain current state (no changes)
                }
            }

            // Clamp everything
            state.arousal = state.arousal.clamp(0.0, 1.0);
            state.consciousness_level = state.consciousness_level.clamp(0.0, 1.0);
            state.dopamine = state.dopamine.clamp(0.0, 1.0);
            state.noradrenaline = state.noradrenaline.clamp(0.0, 1.0);
            state.prediction_error = state.prediction_error.clamp(0.0, 1.0);
            for h in &mut state.harmony_activations {
                *h = h.clamp(0.0, 1.0);
            }
        }
    }

    /// Should we recapitulate now? (Bring back opening theme)
    pub fn should_recapitulate(&self) -> bool {
        self.goals
            .iter()
            .any(|g| g.target == GoalTarget::Recapitulate && g.bars_remaining < 4.0)
    }

    /// Get the opening theme notes for recapitulation.
    pub fn recapitulation_theme(&self) -> Option<&[Note]> {
        if self.should_recapitulate() {
            self.memory.opening_theme()
        } else {
            None
        }
    }

    pub fn active_goal_count(&self) -> usize {
        self.goals.len()
    }
    pub fn section(&self) -> SectionArc {
        self.memory.current_section()
    }

    pub fn reset(&mut self) {
        self.goals.clear();
        self.memory.reset();
    }
}

/// Free composition: Symthaea chooses her own emotional arc.
///
/// Instead of hardcoded consciousness trajectories, the system
/// evolves its own state based on compositional goals and aesthetic
/// feedback. Call this each bar to let Symthaea guide herself.
pub fn free_compose_step(state: &mut MusicalState, composer: &mut ComposerMind, beauty_ema: f32) {
    // Advance form
    let _new_goals = composer.advance_bar(state);

    // Apply goal-directed state shaping
    composer.shape_state(state);

    // Beauty-driven self-regulation:
    // If beauty is dropping, increase exploration (prediction_error)
    // If beauty is high, increase confidence (consciousness)
    if beauty_ema < 0.4 {
        state.prediction_error = (state.prediction_error + 0.01).min(1.0);
    } else if beauty_ema > 0.7 {
        state.consciousness_level = (state.consciousness_level + 0.005).min(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn form_progresses_through_sections() {
        let mut mind = ComposerMind::new();
        mind.bars_per_section = 4.0; // short sections for testing
        let mut state = MusicalState::default();

        // Advance through all sections
        for _ in 0..20 {
            mind.advance_bar(&state);
        }

        assert_eq!(mind.section(), SectionArc::Coda);
    }

    #[test]
    fn recapitulation_triggers() {
        let mut mind = ComposerMind::new();
        mind.bars_per_section = 4.0;
        let state = MusicalState::default();

        // Record an opening theme
        let theme = vec![Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }];
        mind.memory.record_phrase(&theme, 0.8);

        // Advance to recapitulation section
        for _ in 0..14 {
            // sections: expo(4) + dev(4) + climax(4) + recap starts at 12
            mind.advance_bar(&state);
        }

        assert!(mind.memory.opening_theme().is_some());
        assert_eq!(mind.section(), SectionArc::Recapitulation);
    }

    #[test]
    fn build_climax_increases_arousal() {
        let mut mind = ComposerMind::new();
        mind.goals.push(CompositionGoal {
            target: GoalTarget::BuildClimax,
            bars_remaining: 8.0,
            urgency: 1.0,
        });

        let mut state = MusicalState {
            arousal: 0.5,
            ..Default::default()
        };
        mind.shape_state(&mut state);
        assert!(state.arousal > 0.5, "climax should increase arousal");
    }

    #[test]
    fn fade_decreases_arousal() {
        let mut mind = ComposerMind::new();
        mind.goals.push(CompositionGoal {
            target: GoalTarget::FadeToSilence,
            bars_remaining: 4.0,
            urgency: 0.8,
        });

        let mut state = MusicalState {
            arousal: 0.7,
            ..Default::default()
        };
        mind.shape_state(&mut state);
        assert!(state.arousal < 0.7, "fade should decrease arousal");
    }

    #[test]
    fn best_phrase_tracked() {
        let mut mem = FormMemory::new();
        let phrase1 = vec![Note {
            frequency: 440.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }];
        mem.record_phrase(&phrase1, 0.6);
        let phrase2 = vec![Note {
            frequency: 523.25,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }];
        mem.record_phrase(&phrase2, 0.9);

        assert!((mem.best_phrase().unwrap()[0].frequency - 523.25).abs() < 1.0);
    }

    #[test]
    fn free_compose_responds_to_beauty() {
        let mut state = MusicalState {
            consciousness_level: 0.5,
            ..Default::default()
        };
        let mut composer = ComposerMind::new();

        // Low beauty should increase exploration
        free_compose_step(&mut state, &mut composer, 0.2);
        assert!(
            state.prediction_error > 0.0,
            "should explore when beauty is low"
        );
    }
}
