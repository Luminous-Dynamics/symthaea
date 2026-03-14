//! Active emotion regulation strategies mapped to neuromodulator deltas.
//!
//! Selects context-appropriate regulation strategies based on client state,
//! alliance quality, and current affect. Each strategy maps to specific
//! neuromodulator targets following the RDoC-neuromod bridge.
//!
//! Science: Gross (2015) extended process model, Gratz & Roemer (2004) DERS,
//! Linehan (1993) DBT skills, Hayes (2006) ACT defusion.

use crate::alliance::TherapeuticAlliance;
use crate::client_model::{ClientModel, CoreAffectSnapshot};
use serde::{Deserialize, Serialize};

// ── Regulation Strategies ──────────────────────────────────────────────────

/// Evidence-based emotion regulation strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegulationStrategy {
    /// CBT: reframe thought → reduce emotional intensity.
    /// Neuromod: 5-HT↑ (mood stabilize), DA slight↑ (reappraisal reward).
    CognitiveReappraisal,
    /// DBT: tolerate distress without making it worse.
    /// Neuromod: GABA↑ (inhibit), Adenosine slight↑ (calm).
    DistressTolerance,
    /// Somatic: focus on body sensations to anchor in present.
    /// Neuromod: GABA↑ (calm), NE↓ (deactivate threat response).
    Grounding,
    /// ACT: observe thoughts without fusion.
    /// Neuromod: ACh↑ (metacognitive attention), 5-HT↑ (acceptance).
    Defusion,
    /// Humanistic: acknowledge and normalize the emotion.
    /// Neuromod: Oxytocin↑ (social bond), 5-HT↑ (safety signal).
    Validation,
    /// Psychodynamic: hold intense affect within therapeutic space.
    /// Neuromod: GABA↑ (containment), Oxytocin↑ (secure base).
    Containment,
    /// CBT: prepare for graduated exposure to feared stimuli.
    /// Neuromod: NE slight↑ (arousal for learning), ACh↑ (attention).
    ExposurePrep,
}

impl RegulationStrategy {
    /// All strategies for iteration.
    pub const ALL: [Self; 7] = [
        Self::CognitiveReappraisal,
        Self::DistressTolerance,
        Self::Grounding,
        Self::Defusion,
        Self::Validation,
        Self::Containment,
        Self::ExposurePrep,
    ];

    /// Minimum alliance required to use this strategy.
    pub fn min_alliance(&self) -> f32 {
        match self {
            Self::Grounding => 0.1,         // safe at any alliance level
            Self::Validation => 0.1,        // always appropriate
            Self::DistressTolerance => 0.2, // basic coping
            Self::Defusion => 0.3,          // requires some trust
            Self::CognitiveReappraisal => 0.4, // requires cognitive engagement
            Self::Containment => 0.5,       // requires strong therapeutic space
            Self::ExposurePrep => 0.6,      // requires strong alliance
        }
    }

    /// Whether this strategy is appropriate for crisis situations.
    pub fn crisis_safe(&self) -> bool {
        matches!(
            self,
            Self::Grounding | Self::DistressTolerance | Self::Validation | Self::Containment
        )
    }
}

// ── Neuromodulator Delta ───────────────────────────────────────────────────

/// Proposed neuromodulator changes from a regulation strategy.
///
/// Maps to the 8-transmitter bath in `symthaea-neuromodulators`.
/// Values are deltas: positive = increase, negative = decrease.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NeuromodDelta {
    pub dopamine: f32,
    pub noradrenaline: f32,
    pub serotonin: f32,
    pub acetylcholine: f32,
    pub gaba: f32,
    pub oxytocin: f32,
    pub glutamate: f32,
    pub adenosine: f32,
}

impl NeuromodDelta {
    /// Zero delta (no change).
    pub fn zero() -> Self {
        Self {
            dopamine: 0.0,
            noradrenaline: 0.0,
            serotonin: 0.0,
            acetylcholine: 0.0,
            gaba: 0.0,
            oxytocin: 0.0,
            glutamate: 0.0,
            adenosine: 0.0,
        }
    }

    /// Scale all deltas by a factor.
    pub fn scale(&self, factor: f32) -> Self {
        Self {
            dopamine: self.dopamine * factor,
            noradrenaline: self.noradrenaline * factor,
            serotonin: self.serotonin * factor,
            acetylcholine: self.acetylcholine * factor,
            gaba: self.gaba * factor,
            oxytocin: self.oxytocin * factor,
            glutamate: self.glutamate * factor,
            adenosine: self.adenosine * factor,
        }
    }
}

// ── Regulation Engine ──────────────────────────────────────────────────────

/// Context-aware regulation strategy selection and neuromod mapping.
#[derive(Debug, Clone)]
pub struct RegulationEngine {
    /// Current active strategy (if any).
    pub active_strategy: Option<RegulationStrategy>,
    /// Strategy effectiveness history (strategy → success count).
    strategy_successes: Vec<(RegulationStrategy, u32)>,
}

impl RegulationEngine {
    /// Create a new regulation engine.
    pub fn new() -> Self {
        Self {
            active_strategy: None,
            strategy_successes: Vec::new(),
        }
    }

    /// Select the most appropriate regulation strategy given context.
    ///
    /// Priority: safety first, then alliance-appropriate, then best evidence.
    pub fn select_strategy(
        &self,
        client: &ClientModel,
        alliance: &TherapeuticAlliance,
        is_crisis: bool,
    ) -> RegulationStrategy {
        let alliance_level = alliance.composite();
        let distress = client.distress();

        // Crisis: only crisis-safe strategies
        if is_crisis {
            if distress > 0.8 {
                return RegulationStrategy::Grounding;
            }
            return RegulationStrategy::Validation;
        }

        // High distress: prioritize immediate relief
        if distress > 0.7 {
            if alliance_level >= RegulationStrategy::Containment.min_alliance() {
                return RegulationStrategy::Containment;
            }
            return RegulationStrategy::DistressTolerance;
        }

        // Moderate distress: use cognitive strategies if alliance permits
        if distress > 0.4 {
            if alliance_level >= RegulationStrategy::CognitiveReappraisal.min_alliance() {
                return RegulationStrategy::CognitiveReappraisal;
            }
            if alliance_level >= RegulationStrategy::Defusion.min_alliance() {
                return RegulationStrategy::Defusion;
            }
            return RegulationStrategy::Validation;
        }

        // Low distress: can try exposure prep if alliance is strong
        if alliance_level >= RegulationStrategy::ExposurePrep.min_alliance() {
            return RegulationStrategy::ExposurePrep;
        }

        RegulationStrategy::Validation
    }

    /// Apply a regulation strategy → neuromodulator delta.
    ///
    /// Intensity scales with client distress (higher distress = stronger intervention).
    pub fn apply_strategy(
        &mut self,
        strategy: RegulationStrategy,
        distress: f32,
    ) -> NeuromodDelta {
        self.active_strategy = Some(strategy);
        let intensity = distress.clamp(0.1, 1.0) * 0.15; // max 0.15 delta per cycle

        let base = match strategy {
            RegulationStrategy::CognitiveReappraisal => NeuromodDelta {
                serotonin: 0.8,
                dopamine: 0.3,
                acetylcholine: 0.4,
                ..NeuromodDelta::zero()
            },
            RegulationStrategy::DistressTolerance => NeuromodDelta {
                gaba: 0.8,
                adenosine: 0.4,
                noradrenaline: -0.3,
                ..NeuromodDelta::zero()
            },
            RegulationStrategy::Grounding => NeuromodDelta {
                gaba: 0.7,
                noradrenaline: -0.5,
                adenosine: 0.3,
                ..NeuromodDelta::zero()
            },
            RegulationStrategy::Defusion => NeuromodDelta {
                acetylcholine: 0.7,
                serotonin: 0.5,
                ..NeuromodDelta::zero()
            },
            RegulationStrategy::Validation => NeuromodDelta {
                oxytocin: 0.8,
                serotonin: 0.5,
                ..NeuromodDelta::zero()
            },
            RegulationStrategy::Containment => NeuromodDelta {
                gaba: 0.6,
                oxytocin: 0.7,
                noradrenaline: -0.3,
                ..NeuromodDelta::zero()
            },
            RegulationStrategy::ExposurePrep => NeuromodDelta {
                noradrenaline: 0.3,
                acetylcholine: 0.6,
                dopamine: 0.2,
                ..NeuromodDelta::zero()
            },
        };

        base.scale(intensity)
    }

    /// Record that a strategy was effective (client distress decreased).
    pub fn record_success(&mut self, strategy: RegulationStrategy) {
        if let Some((_, count)) = self
            .strategy_successes
            .iter_mut()
            .find(|(s, _)| *s == strategy)
        {
            *count += 1;
        } else {
            self.strategy_successes.push((strategy, 1));
        }
    }
}

impl Default for RegulationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_distressed_client() -> ClientModel {
        let mut client = ClientModel::new();
        // distress = neg_valence*0.6 + high_arousal*0.4
        // = 0.9*0.6 + (1.0-0.5)*2.0*0.4 = 0.54 + 0.40 = 0.94 (>0.8 threshold)
        client.update_affect(CoreAffectSnapshot::new(-0.9, 1.0, 0));
        client
    }

    fn make_calm_client() -> ClientModel {
        let mut client = ClientModel::new();
        client.update_affect(CoreAffectSnapshot::new(0.3, 0.4, 0));
        client
    }

    #[test]
    fn test_crisis_returns_grounding_high_distress() {
        let engine = RegulationEngine::new();
        let client = make_distressed_client();
        let alliance = TherapeuticAlliance::new();
        let strategy = engine.select_strategy(&client, &alliance, true);
        assert_eq!(strategy, RegulationStrategy::Grounding);
    }

    #[test]
    fn test_crisis_returns_validation_moderate() {
        let engine = RegulationEngine::new();
        let mut client = ClientModel::new();
        client.update_affect(CoreAffectSnapshot::new(-0.3, 0.6, 0));
        let alliance = TherapeuticAlliance::new();
        let strategy = engine.select_strategy(&client, &alliance, true);
        assert_eq!(strategy, RegulationStrategy::Validation);
    }

    #[test]
    fn test_high_distress_low_alliance_uses_distress_tolerance() {
        let engine = RegulationEngine::new();
        let client = make_distressed_client();
        let alliance = TherapeuticAlliance::new(); // 0.3 alliance
        let strategy = engine.select_strategy(&client, &alliance, false);
        assert_eq!(strategy, RegulationStrategy::DistressTolerance);
    }

    #[test]
    fn test_low_distress_high_alliance_exposure() {
        let engine = RegulationEngine::new();
        let client = make_calm_client();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.8;
        alliance.goal_agreement = 0.8;
        alliance.task_agreement = 0.8;
        let strategy = engine.select_strategy(&client, &alliance, false);
        assert_eq!(strategy, RegulationStrategy::ExposurePrep);
    }

    #[test]
    fn test_apply_strategy_returns_nonzero_delta() {
        let mut engine = RegulationEngine::new();
        let delta = engine.apply_strategy(RegulationStrategy::Validation, 0.6);
        assert!(delta.oxytocin > 0.0);
        assert!(delta.serotonin > 0.0);
    }

    #[test]
    fn test_apply_strategy_intensity_scales_with_distress() {
        let mut engine = RegulationEngine::new();
        let low = engine.apply_strategy(RegulationStrategy::Grounding, 0.2);
        let high = engine.apply_strategy(RegulationStrategy::Grounding, 0.9);
        assert!(high.gaba.abs() > low.gaba.abs());
    }

    #[test]
    fn test_crisis_safe_strategies() {
        assert!(RegulationStrategy::Grounding.crisis_safe());
        assert!(RegulationStrategy::Validation.crisis_safe());
        assert!(!RegulationStrategy::ExposurePrep.crisis_safe());
        assert!(!RegulationStrategy::CognitiveReappraisal.crisis_safe());
    }

    #[test]
    fn test_record_success() {
        let mut engine = RegulationEngine::new();
        engine.record_success(RegulationStrategy::Grounding);
        engine.record_success(RegulationStrategy::Grounding);
        assert_eq!(engine.strategy_successes.len(), 1);
        assert_eq!(engine.strategy_successes[0].1, 2);
    }

    #[test]
    fn test_all_strategies_have_min_alliance() {
        for strategy in RegulationStrategy::ALL {
            assert!(strategy.min_alliance() >= 0.0);
            assert!(strategy.min_alliance() <= 1.0);
        }
    }
}
