//! Active emotion regulation strategies mapped to neuromodulator deltas.
//!
//! Selects context-appropriate regulation strategies based on client state,
//! alliance quality, and current affect. Each strategy maps to specific
//! neuromodulator targets following the RDoC-neuromod bridge.
//!
//! Science: Gross (2015) extended process model, Gratz & Roemer (2004) DERS,
//! Linehan (1993) DBT skills, Hayes (2006) ACT defusion.

use crate::alliance::TherapeuticAlliance;
use crate::client_model::ClientModel;
use serde::{Deserialize, Serialize};
use symthaea_clinical::rdoc::{RDocDomain, RDocProfile};

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
    /// Dream-discovered strategy preferences (strategy → cumulative Phi improvement).
    dream_strategy_bias: Vec<(RegulationStrategy, f32)>,
}

impl RegulationEngine {
    /// Create a new regulation engine.
    pub fn new() -> Self {
        Self {
            active_strategy: None,
            strategy_successes: Vec::new(),
            dream_strategy_bias: Vec::new(),
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

        // Crisis: only crisis-safe strategies (dream wisdom does NOT override safety)
        if is_crisis {
            if distress > 0.8 {
                return RegulationStrategy::Grounding;
            }
            return RegulationStrategy::Validation;
        }

        // Dream wisdom tie-breaker: if the dream engine discovered a strategy
        // that produces better consciousness quality (Phi), prefer it when
        // alliance permits and the strategy is appropriate for distress level.
        // Science: Walker (2009) — offline consolidation improves waking decisions.
        if let Some(dream_pref) = self.dream_preferred_strategy() {
            let pref_safe = !is_crisis || dream_pref.crisis_safe();
            let pref_alliance_ok = alliance_level >= dream_pref.min_alliance();
            if pref_safe && pref_alliance_ok {
                return dream_pref;
            }
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

    /// Apply a regulation strategy with RDoC-aware neuromodulator modulation.
    ///
    /// Uses the client's RDoC profile to amplify transmitters along the
    /// domain→neuromodulator mapping (Insel et al. 2010):
    /// - High NegativeValence → stronger serotonin/GABA boost
    /// - Low PositiveValence → stronger dopamine boost
    /// - High ArousalRegulatory → stronger noradrenaline reduction + adenosine
    /// - Low CognitiveSystems → stronger acetylcholine boost
    /// - Low SocialProcesses → stronger oxytocin boost
    pub fn apply_strategy_rdoc(
        &mut self,
        strategy: RegulationStrategy,
        distress: f32,
        rdoc: &RDocProfile,
    ) -> NeuromodDelta {
        let base_delta = self.apply_strategy(strategy, distress);

        // RDoC domain scores modulate the primary neuromodulator for that domain.
        // High NegativeValence (bad) → needs more serotonin/GABA.
        // Low PositiveValence (bad) → needs more dopamine.
        // For "deficit" domains, we use (1 - score) as the amplification factor.
        let neg_val = rdoc.score(RDocDomain::NegativeValence);
        let pos_val_deficit = 1.0 - rdoc.score(RDocDomain::PositiveValence);
        let cog_deficit = 1.0 - rdoc.score(RDocDomain::CognitiveSystems);
        let social_deficit = 1.0 - rdoc.score(RDocDomain::SocialProcesses);
        let arousal_dysreg = (rdoc.score(RDocDomain::ArousalRegulatory) - 0.5).abs() * 2.0;

        // Amplification factor: 1.0 (neutral) to 1.5 (maximum domain-driven boost)
        let serotonin_amp = 1.0 + neg_val * 0.5;
        let gaba_amp = 1.0 + neg_val * 0.3;
        let dopamine_amp = 1.0 + pos_val_deficit * 0.5;
        let acetylcholine_amp = 1.0 + cog_deficit * 0.4;
        let oxytocin_amp = 1.0 + social_deficit * 0.4;
        let noradrenaline_amp = 1.0 + arousal_dysreg * 0.3;
        let adenosine_amp = 1.0 + arousal_dysreg * 0.2;

        NeuromodDelta {
            dopamine: base_delta.dopamine * dopamine_amp,
            noradrenaline: base_delta.noradrenaline * noradrenaline_amp,
            serotonin: base_delta.serotonin * serotonin_amp,
            acetylcholine: base_delta.acetylcholine * acetylcholine_amp,
            gaba: base_delta.gaba * gaba_amp,
            oxytocin: base_delta.oxytocin * oxytocin_amp,
            glutamate: base_delta.glutamate,
            adenosine: base_delta.adenosine * adenosine_amp,
        }
    }

    /// Incorporate dream-discovered strategy preference.
    ///
    /// Called when the dream engine discovers that a particular strategy
    /// ordinal would have produced better Phi (consciousness quality).
    /// Strategy ordinal maps: 0=CognitiveReappraisal, 1=DistressTolerance,
    /// 2=Grounding, 3=Defusion, 4=Validation, 5=Containment, 6=ExposurePrep.
    pub fn incorporate_dream_wisdom(&mut self, strategy_ordinal: u8, phi_improvement: f32) {
        let strategy = match strategy_ordinal {
            0 => RegulationStrategy::CognitiveReappraisal,
            1 => RegulationStrategy::DistressTolerance,
            2 => RegulationStrategy::Grounding,
            3 => RegulationStrategy::Defusion,
            4 => RegulationStrategy::Validation,
            5 => RegulationStrategy::Containment,
            6 => RegulationStrategy::ExposurePrep,
            _ => return,
        };

        if let Some((_, score)) = self
            .dream_strategy_bias
            .iter_mut()
            .find(|(s, _)| *s == strategy)
        {
            // EMA: blend new insight with historical preference
            *score = *score * 0.9 + phi_improvement * 0.1;
        } else {
            self.dream_strategy_bias.push((strategy, phi_improvement));
        }
    }

    /// Get the dream-biased best strategy, if dream wisdom suggests one.
    ///
    /// Returns the strategy with the highest cumulative Phi improvement
    /// if it exceeds a minimum threshold (0.01).
    pub fn dream_preferred_strategy(&self) -> Option<RegulationStrategy> {
        self.dream_strategy_bias
            .iter()
            .filter(|(_, score)| *score > 0.01)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, _)| *s)
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
    use crate::client_model::CoreAffectSnapshot;

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

    #[test]
    fn test_apply_strategy_rdoc_amplifies_serotonin_for_high_neg_valence() {
        let mut engine = RegulationEngine::new();
        let mut rdoc_high_neg = RDocProfile::default();
        rdoc_high_neg.set_score(RDocDomain::NegativeValence, 0.9);
        let rdoc_default = RDocProfile::default();

        let delta_high = engine.apply_strategy_rdoc(
            RegulationStrategy::Validation, 0.5, &rdoc_high_neg,
        );
        let delta_default = engine.apply_strategy_rdoc(
            RegulationStrategy::Validation, 0.5, &rdoc_default,
        );
        // High NegativeValence should produce stronger serotonin delta
        assert!(
            delta_high.serotonin.abs() > delta_default.serotonin.abs(),
            "serotonin should be amplified: {} vs {}",
            delta_high.serotonin, delta_default.serotonin,
        );
    }

    #[test]
    fn test_apply_strategy_rdoc_amplifies_dopamine_for_low_pos_valence() {
        let mut engine = RegulationEngine::new();
        let mut rdoc_low_pos = RDocProfile::default();
        rdoc_low_pos.set_score(RDocDomain::PositiveValence, 0.1); // low = deficit
        let rdoc_default = RDocProfile::default();

        let delta_low = engine.apply_strategy_rdoc(
            RegulationStrategy::CognitiveReappraisal, 0.5, &rdoc_low_pos,
        );
        let delta_default = engine.apply_strategy_rdoc(
            RegulationStrategy::CognitiveReappraisal, 0.5, &rdoc_default,
        );
        assert!(
            delta_low.dopamine.abs() > delta_default.dopamine.abs(),
            "dopamine should be amplified for PositiveValence deficit",
        );
    }

    #[test]
    fn test_apply_strategy_rdoc_backward_compat() {
        // Default RDoC profile should produce approximately same results
        // as non-RDoC apply_strategy (within amplification factor).
        let mut engine = RegulationEngine::new();
        let rdoc = RDocProfile::default();
        let delta_rdoc = engine.apply_strategy_rdoc(
            RegulationStrategy::Grounding, 0.5, &rdoc,
        );
        let delta_plain = engine.apply_strategy(
            RegulationStrategy::Grounding, 0.5,
        );
        // With default RDoC (moderate scores), amplification should be modest
        // GABA should still be positive and in the same ballpark
        assert!(delta_rdoc.gaba > 0.0);
        assert!(delta_plain.gaba > 0.0);
        assert!((delta_rdoc.gaba / delta_plain.gaba - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_incorporate_dream_wisdom() {
        let mut engine = RegulationEngine::new();
        engine.incorporate_dream_wisdom(2, 0.15); // Grounding
        engine.incorporate_dream_wisdom(2, 0.25); // Grounding again
        let pref = engine.dream_preferred_strategy();
        assert_eq!(pref, Some(RegulationStrategy::Grounding));
    }

    #[test]
    fn test_dream_wisdom_ema_decay() {
        let mut engine = RegulationEngine::new();
        engine.incorporate_dream_wisdom(4, 0.5); // Validation
        engine.incorporate_dream_wisdom(4, 0.0); // Low improvement
        // EMA should decay: 0.5*0.9 + 0.0*0.1 = 0.45
        let pref = engine.dream_preferred_strategy();
        assert_eq!(pref, Some(RegulationStrategy::Validation));
    }

    #[test]
    fn test_dream_wisdom_no_preference_below_threshold() {
        let mut engine = RegulationEngine::new();
        engine.incorporate_dream_wisdom(0, 0.005); // Below 0.01 threshold
        assert_eq!(engine.dream_preferred_strategy(), None);
    }

    #[test]
    fn test_dream_wisdom_invalid_ordinal_ignored() {
        let mut engine = RegulationEngine::new();
        engine.incorporate_dream_wisdom(99, 0.5); // Invalid
        assert_eq!(engine.dream_preferred_strategy(), None);
    }

    #[test]
    fn test_dream_preference_overrides_default_selection() {
        let mut engine = RegulationEngine::new();
        // Dream discovers Defusion (ordinal 3) works well
        engine.incorporate_dream_wisdom(3, 0.3);

        let client = make_calm_client();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.5;
        alliance.goal_agreement = 0.5;
        alliance.task_agreement = 0.5;

        // Without dream: low distress + moderate alliance → ExposurePrep or Validation
        // With dream: should prefer Defusion (if alliance permits)
        let strategy = engine.select_strategy(&client, &alliance, false);
        assert_eq!(
            strategy,
            RegulationStrategy::Defusion,
            "Dream preference should override default when alliance permits",
        );
    }

    #[test]
    fn test_dream_preference_respects_crisis_safety() {
        let mut engine = RegulationEngine::new();
        // Dream discovers ExposurePrep (ordinal 6) is great — but not crisis-safe
        engine.incorporate_dream_wisdom(6, 0.5);

        let client = make_distressed_client();
        let alliance = TherapeuticAlliance::new();

        // During crisis, dream preference should NOT override safety
        let strategy = engine.select_strategy(&client, &alliance, true);
        assert!(
            strategy.crisis_safe(),
            "Crisis should always select crisis-safe strategy, got {:?}",
            strategy,
        );
    }
}
