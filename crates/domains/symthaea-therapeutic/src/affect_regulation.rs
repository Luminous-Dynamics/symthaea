// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    /// Static string for telemetry (avoids `format!("{:?}")`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CognitiveReappraisal => "CognitiveReappraisal",
            Self::DistressTolerance => "DistressTolerance",
            Self::Grounding => "Grounding",
            Self::Defusion => "Defusion",
            Self::Validation => "Validation",
            Self::Containment => "Containment",
            Self::ExposurePrep => "ExposurePrep",
        }
    }

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

    /// Broca therapeutic intent code for this strategy.
    /// 0=validate, 1=reflect, 2=reframe, 3=explore, 4=psychoeducate, 6=contain.
    pub fn intent_code(&self) -> f32 {
        match self {
            Self::Validation => 0.0,
            Self::Defusion => 1.0,
            Self::CognitiveReappraisal => 2.0,
            Self::ExposurePrep => 3.0,
            Self::DistressTolerance => 4.0,
            Self::Grounding => 4.0,
            Self::Containment => 6.0,
        }
    }

    /// Minimum alliance required to use this strategy.
    pub fn min_alliance(&self) -> f32 {
        match self {
            Self::Grounding => 0.1,            // safe at any alliance level
            Self::Validation => 0.1,           // always appropriate
            Self::DistressTolerance => 0.2,    // basic coping
            Self::Defusion => 0.3,             // requires some trust
            Self::CognitiveReappraisal => 0.4, // requires cognitive engagement
            Self::Containment => 0.5,          // requires strong therapeutic space
            Self::ExposurePrep => 0.6,         // requires strong alliance
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

/// Per-strategy effectiveness record tracking outcomes.
///
/// Records how often a strategy was applied and the mean affect-delta
/// (improvement in client distress) it produced. Negative delta = improvement.
///
/// Science: Lambert (2013) — outcome-informed treatment adjusts based on client progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEffectiveness {
    /// Number of times this strategy was applied.
    pub applications: u32,
    /// Number of times affect improved after application.
    pub successes: u32,
    /// Mean affect-delta (EMA-smoothed). Negative = improvement.
    pub mean_affect_delta: f32,
}

impl StrategyEffectiveness {
    fn new() -> Self {
        Self {
            applications: 0,
            successes: 0,
            mean_affect_delta: 0.0,
        }
    }

    /// Success rate (0-1). Returns 0.5 if no applications (neutral prior).
    pub fn success_rate(&self) -> f32 {
        if self.applications == 0 {
            0.5 // neutral prior (Bayesian: no data → uniform)
        } else {
            self.successes as f32 / self.applications as f32
        }
    }

    /// Record an application outcome.
    fn record(&mut self, affect_delta: f32) {
        self.applications += 1;
        if affect_delta < 0.0 {
            self.successes += 1; // negative delta = distress decreased = success
        }
        // EMA for mean delta (alpha=0.2 for moderate responsiveness)
        self.mean_affect_delta = self.mean_affect_delta * 0.8 + affect_delta * 0.2;
    }
}

/// Context-aware regulation strategy selection and neuromod mapping.
#[derive(Debug, Clone)]
pub struct RegulationEngine {
    /// Current active strategy (if any).
    pub active_strategy: Option<RegulationStrategy>,
    /// Strategy effectiveness history (strategy → success count).
    strategy_successes: Vec<(RegulationStrategy, u32)>,
    /// Per-strategy effectiveness tracking with affect-delta outcomes.
    /// Science: Lambert (2013) — outcome-informed treatment.
    strategy_effectiveness: std::collections::HashMap<RegulationStrategy, StrategyEffectiveness>,
    /// Distress at time of last strategy application (for delta computation).
    distress_at_application: Option<f32>,
    /// Dream-discovered strategy preferences (strategy → cumulative Phi improvement).
    dream_strategy_bias: Vec<(RegulationStrategy, f32)>,
    /// Accumulated serotonin debt from sustained negative valence.
    /// Grows when NegativeValence is high; decays toward zero otherwise.
    /// Science: sustained negative affect depletes 5-HT reserves (Jans et al., 2007).
    serotonin_debt: f32,
    /// Accumulated dopamine debt from sustained low positive valence.
    dopamine_debt: f32,
}

impl RegulationEngine {
    /// Create a new regulation engine.
    pub fn new() -> Self {
        Self {
            active_strategy: None,
            strategy_successes: Vec::new(),
            strategy_effectiveness: std::collections::HashMap::new(),
            distress_at_application: None,
            dream_strategy_bias: Vec::new(),
            serotonin_debt: 0.0,
            dopamine_debt: 0.0,
        }
    }

    /// Update temporal neuromodulator debt from current RDoC profile.
    ///
    /// Called each cycle to accumulate or decay transmitter debt.
    /// Sustained negative valence grows serotonin debt (need for more 5-HT);
    /// sustained low positive valence grows dopamine debt.
    /// Debt decays exponentially when the triggering condition resolves.
    ///
    /// Science: Jans et al. (2007) — chronic stress depletes central 5-HT.
    pub fn tick_debt(&mut self, rdoc: &RDocProfile) {
        let neg_val = rdoc.score(RDocDomain::NegativeValence);
        let pos_val = rdoc.score(RDocDomain::PositiveValence);

        // Accumulate: high negative → serotonin debt grows
        if neg_val > 0.3 {
            self.serotonin_debt = (self.serotonin_debt + (neg_val - 0.3) * 0.01).min(1.0);
        } else {
            // Decay toward zero
            self.serotonin_debt *= 0.98;
        }

        // Accumulate: low positive → dopamine debt grows
        if pos_val < 0.4 {
            self.dopamine_debt = (self.dopamine_debt + (0.4 - pos_val) * 0.01).min(1.0);
        } else {
            self.dopamine_debt *= 0.98;
        }
    }

    /// Get current serotonin debt (0-1). Higher = more 5-HT needed.
    pub fn serotonin_debt(&self) -> f32 {
        self.serotonin_debt
    }

    /// Get current dopamine debt (0-1). Higher = more DA needed.
    pub fn dopamine_debt(&self) -> f32 {
        self.dopamine_debt
    }

    /// Update RDoC profile from neuromodulator bath state (bidirectional bridge).
    ///
    /// The neuromod bath has rich temporal dynamics (PKs, circadian gating, etc.)
    /// that should be reflected back into the clinical RDoC profile.
    ///
    /// Mapping (inverse of domain→neuromod):
    /// - Low serotonin → NegativeValence ↑
    /// - Low dopamine → PositiveValence ↓
    /// - High noradrenaline → ArousalRegulatory deviation ↑
    /// - Low acetylcholine → CognitiveSystems ↓
    /// - Low oxytocin → SocialProcesses ↓
    pub fn update_rdoc_from_neuromod(rdoc: &mut RDocProfile, bath: &[f32; 8]) {
        let [da, ne, sht, ach, gaba, oxy, _glu, _aden] = *bath;

        // Per-transmitter EMA time constants (Science: Stahl 2013, Cooper et al. 2003)
        // Serotonin: slow dynamics (~weeks for tonic changes) → alpha 0.01
        // Dopamine: fast phasic dynamics (~seconds) → alpha 0.08
        // Noradrenaline: moderate (~minutes) → alpha 0.05
        // Acetylcholine: fast cholinergic (~seconds) → alpha 0.06
        // Oxytocin: slow peptide dynamics (~hours) → alpha 0.02
        // GABA: moderate inhibitory → alpha 0.04
        let alpha_sht: f32 = 0.01;
        let alpha_da: f32 = 0.08;
        let alpha_ne: f32 = 0.05;
        let alpha_ach: f32 = 0.06;
        let alpha_oxy: f32 = 0.02;
        let alpha_gaba: f32 = 0.04;

        // Low serotonin + low GABA → increase NegativeValence
        let neg_val_signal = (1.0 - sht).max(0.0) * 0.5 + (1.0 - gaba).max(0.0) * 0.3;
        let alpha_neg = alpha_sht * 0.7 + alpha_gaba * 0.3; // weighted blend
        let cur_neg = rdoc.score(RDocDomain::NegativeValence);
        rdoc.set_score(
            RDocDomain::NegativeValence,
            cur_neg * (1.0 - alpha_neg) + neg_val_signal * alpha_neg,
        );

        // Low dopamine → decrease PositiveValence
        let pos_val_signal = da.clamp(0.0, 1.0);
        let cur_pos = rdoc.score(RDocDomain::PositiveValence);
        rdoc.set_score(
            RDocDomain::PositiveValence,
            cur_pos * (1.0 - alpha_da) + pos_val_signal * alpha_da,
        );

        // High noradrenaline deviation → increase ArousalRegulatory
        let arousal_signal = (ne - 0.5).abs() * 2.0;
        let cur_arousal = rdoc.score(RDocDomain::ArousalRegulatory);
        rdoc.set_score(
            RDocDomain::ArousalRegulatory,
            cur_arousal * (1.0 - alpha_ne) + arousal_signal * alpha_ne,
        );

        // Acetylcholine → CognitiveSystems
        let cog_signal = ach.clamp(0.0, 1.0);
        let cur_cog = rdoc.score(RDocDomain::CognitiveSystems);
        rdoc.set_score(
            RDocDomain::CognitiveSystems,
            cur_cog * (1.0 - alpha_ach) + cog_signal * alpha_ach,
        );

        // Oxytocin → SocialProcesses
        let social_signal = oxy.clamp(0.0, 1.0);
        let cur_social = rdoc.score(RDocDomain::SocialProcesses);
        rdoc.set_score(
            RDocDomain::SocialProcesses,
            cur_social * (1.0 - alpha_oxy) + social_signal * alpha_oxy,
        );
    }

    /// Select the most appropriate regulation strategy given context.
    ///
    /// Priority: safety first, then effectiveness-biased, then alliance-appropriate,
    /// then dream wisdom, then default evidence-based selection.
    ///
    /// Science: Lambert (2013) outcome-informed treatment — adjust based on
    /// tracked client progress. Strategies with <30% success rate (≥3 applications)
    /// are deprioritized.
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

        // Effectiveness-biased selection: if we have enough data on a strategy
        // that works well for this client, prefer it (outcome-informed treatment).
        // Only active in non-crisis, with sufficient data (≥3 applications).
        if let Some(best) = self.most_effective_strategy() {
            if alliance_level >= best.min_alliance() {
                // Check it's not the least effective too (edge case with only 1 tracked)
                let dominated = self
                    .least_effective_strategy()
                    .map_or(false, |worst| worst == best);
                if !dominated {
                    return best;
                }
            }
        }

        // Dream wisdom tie-breaker: if the dream engine discovered a strategy
        // that produces better consciousness quality (Phi), prefer it when
        // alliance permits and the strategy is appropriate for distress level.
        // Science: Walker (2009) — offline consolidation improves waking decisions.
        if let Some(dream_pref) = self.dream_preferred_strategy() {
            let pref_safe = !is_crisis || dream_pref.crisis_safe();
            let pref_alliance_ok = alliance_level >= dream_pref.min_alliance();
            // Also check effectiveness: don't use dream-preferred if historically poor
            let not_poor = self.effectiveness(&dream_pref).map_or(true, |eff| {
                eff.applications < 3 || eff.success_rate() >= 0.3
            });
            if pref_safe && pref_alliance_ok && not_poor {
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
    pub fn apply_strategy(&mut self, strategy: RegulationStrategy, distress: f32) -> NeuromodDelta {
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

        // Temporal debt amplification: accumulated need for serotonin/dopamine
        // adds an extra boost proportional to the debt level.
        let sht_debt_amp = 1.0 + self.serotonin_debt * 0.3;
        let da_debt_amp = 1.0 + self.dopamine_debt * 0.3;

        NeuromodDelta {
            dopamine: base_delta.dopamine * dopamine_amp * da_debt_amp,
            noradrenaline: base_delta.noradrenaline * noradrenaline_amp,
            serotonin: base_delta.serotonin * serotonin_amp * sht_debt_amp,
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

    /// Clear dream strategy preference (used when prediction accuracy is low).
    pub fn clear_dream_preference(&mut self) {
        self.dream_strategy_bias.clear();
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

    /// Record the distress level at time of strategy application.
    ///
    /// Must be called before `record_outcome()` to compute affect-delta.
    pub fn record_application_distress(&mut self, distress: f32) {
        self.distress_at_application = Some(distress);
    }

    /// Record the outcome of the last applied strategy.
    ///
    /// Computes affect-delta from distress at application vs current distress.
    /// Negative delta = improvement. Also records into legacy `record_success`.
    pub fn record_outcome(&mut self, current_distress: f32) {
        if let (Some(strategy), Some(prior_distress)) =
            (self.active_strategy, self.distress_at_application)
        {
            let delta = current_distress - prior_distress; // negative = improvement
            self.strategy_effectiveness
                .entry(strategy)
                .or_insert_with(StrategyEffectiveness::new)
                .record(delta);
            if delta < 0.0 {
                self.record_success(strategy);
            }
            self.distress_at_application = None;
        }
    }

    /// Get effectiveness record for a strategy.
    pub fn effectiveness(&self, strategy: &RegulationStrategy) -> Option<&StrategyEffectiveness> {
        self.strategy_effectiveness.get(strategy)
    }

    /// Get the most effective strategy based on historical outcomes.
    ///
    /// Returns the strategy with the highest success rate (minimum 3 applications).
    pub fn most_effective_strategy(&self) -> Option<RegulationStrategy> {
        self.strategy_effectiveness
            .iter()
            .filter(|(_, eff)| eff.applications >= 3)
            .max_by(|a, b| {
                a.1.success_rate()
                    .partial_cmp(&b.1.success_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(s, _)| *s)
    }

    /// Get the least effective strategy (to avoid in selection).
    ///
    /// Returns the strategy with the lowest success rate (minimum 3 applications).
    pub fn least_effective_strategy(&self) -> Option<RegulationStrategy> {
        self.strategy_effectiveness
            .iter()
            .filter(|(_, eff)| eff.applications >= 3)
            .min_by(|a, b| {
                a.1.success_rate()
                    .partial_cmp(&b.1.success_rate())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|(_, eff)| eff.success_rate() < 0.3) // only flag truly poor strategies
            .map(|(s, _)| *s)
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

        let delta_high =
            engine.apply_strategy_rdoc(RegulationStrategy::Validation, 0.5, &rdoc_high_neg);
        let delta_default =
            engine.apply_strategy_rdoc(RegulationStrategy::Validation, 0.5, &rdoc_default);
        // High NegativeValence should produce stronger serotonin delta
        assert!(
            delta_high.serotonin.abs() > delta_default.serotonin.abs(),
            "serotonin should be amplified: {} vs {}",
            delta_high.serotonin,
            delta_default.serotonin,
        );
    }

    #[test]
    fn test_apply_strategy_rdoc_amplifies_dopamine_for_low_pos_valence() {
        let mut engine = RegulationEngine::new();
        let mut rdoc_low_pos = RDocProfile::default();
        rdoc_low_pos.set_score(RDocDomain::PositiveValence, 0.1); // low = deficit
        let rdoc_default = RDocProfile::default();

        let delta_low = engine.apply_strategy_rdoc(
            RegulationStrategy::CognitiveReappraisal,
            0.5,
            &rdoc_low_pos,
        );
        let delta_default = engine.apply_strategy_rdoc(
            RegulationStrategy::CognitiveReappraisal,
            0.5,
            &rdoc_default,
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
        let delta_rdoc = engine.apply_strategy_rdoc(RegulationStrategy::Grounding, 0.5, &rdoc);
        let delta_plain = engine.apply_strategy(RegulationStrategy::Grounding, 0.5);
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
    fn test_serotonin_debt_accumulates() {
        let mut engine = RegulationEngine::new();
        let mut rdoc = RDocProfile::default();
        rdoc.set_score(RDocDomain::NegativeValence, 0.8);
        assert_eq!(engine.serotonin_debt(), 0.0);
        for _ in 0..50 {
            engine.tick_debt(&rdoc);
        }
        assert!(
            engine.serotonin_debt() > 0.1,
            "Serotonin debt should accumulate with high NegativeValence: {}",
            engine.serotonin_debt(),
        );
    }

    #[test]
    fn test_serotonin_debt_decays() {
        let mut engine = RegulationEngine::new();
        let mut rdoc = RDocProfile::default();
        rdoc.set_score(RDocDomain::NegativeValence, 0.8);
        for _ in 0..50 {
            engine.tick_debt(&rdoc);
        }
        let peak = engine.serotonin_debt();
        // Now resolve the trigger
        rdoc.set_score(RDocDomain::NegativeValence, 0.1);
        for _ in 0..100 {
            engine.tick_debt(&rdoc);
        }
        assert!(
            engine.serotonin_debt() < peak * 0.5,
            "Serotonin debt should decay when NegativeValence resolves",
        );
    }

    #[test]
    fn test_dopamine_debt_accumulates_with_low_positive() {
        let mut engine = RegulationEngine::new();
        let mut rdoc = RDocProfile::default();
        rdoc.set_score(RDocDomain::PositiveValence, 0.1);
        for _ in 0..50 {
            engine.tick_debt(&rdoc);
        }
        assert!(
            engine.dopamine_debt() > 0.05,
            "Dopamine debt should accumulate with low PositiveValence",
        );
    }

    #[test]
    fn test_debt_amplifies_rdoc_deltas() {
        let mut engine = RegulationEngine::new();
        let rdoc = RDocProfile::default();

        // Without debt
        let delta_no_debt = engine.apply_strategy_rdoc(RegulationStrategy::Validation, 0.5, &rdoc);

        // Accumulate serotonin debt
        let mut neg_rdoc = RDocProfile::default();
        neg_rdoc.set_score(RDocDomain::NegativeValence, 0.9);
        for _ in 0..100 {
            engine.tick_debt(&neg_rdoc);
        }

        let delta_with_debt =
            engine.apply_strategy_rdoc(RegulationStrategy::Validation, 0.5, &rdoc);
        assert!(
            delta_with_debt.serotonin > delta_no_debt.serotonin,
            "Accumulated serotonin debt should amplify serotonin delta: {} vs {}",
            delta_with_debt.serotonin,
            delta_no_debt.serotonin,
        );
    }

    #[test]
    fn test_update_rdoc_from_neuromod_low_serotonin() {
        let mut rdoc = RDocProfile::default();
        let pre_neg = rdoc.score(RDocDomain::NegativeValence);
        // Low serotonin (index 2), low GABA (index 4)
        let bath = [0.5, 0.5, 0.1, 0.5, 0.1, 0.5, 0.5, 0.5];
        for _ in 0..50 {
            RegulationEngine::update_rdoc_from_neuromod(&mut rdoc, &bath);
        }
        assert!(
            rdoc.score(RDocDomain::NegativeValence) > pre_neg,
            "Low serotonin should increase NegativeValence",
        );
    }

    #[test]
    fn test_update_rdoc_from_neuromod_high_dopamine() {
        let mut rdoc = RDocProfile::default();
        // High dopamine (index 0)
        let bath = [0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        for _ in 0..50 {
            RegulationEngine::update_rdoc_from_neuromod(&mut rdoc, &bath);
        }
        assert!(
            rdoc.score(RDocDomain::PositiveValence) > 0.5,
            "High dopamine should increase PositiveValence",
        );
    }

    #[test]
    fn test_update_rdoc_from_neuromod_low_oxytocin() {
        let mut rdoc = RDocProfile::default();
        let pre_social = rdoc.score(RDocDomain::SocialProcesses);
        let bath = [0.5, 0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5];
        for _ in 0..50 {
            RegulationEngine::update_rdoc_from_neuromod(&mut rdoc, &bath);
        }
        assert!(
            rdoc.score(RDocDomain::SocialProcesses) < pre_social,
            "Low oxytocin should decrease SocialProcesses",
        );
    }

    #[test]
    fn test_strategy_effectiveness_tracking() {
        let mut engine = RegulationEngine::new();
        // Apply Validation and record improvement
        engine.apply_strategy(RegulationStrategy::Validation, 0.5);
        engine.record_application_distress(0.6);
        engine.record_outcome(0.4); // delta = -0.2 (improvement)

        let eff = engine
            .effectiveness(&RegulationStrategy::Validation)
            .unwrap();
        assert_eq!(eff.applications, 1);
        assert_eq!(eff.successes, 1);
        assert!(
            eff.mean_affect_delta < 0.0,
            "mean delta should be negative (improvement)"
        );
    }

    #[test]
    fn test_strategy_effectiveness_success_rate() {
        let mut engine = RegulationEngine::new();
        // 3 successes, 1 failure
        for delta in [-0.2, -0.1, -0.3, 0.1] {
            engine.apply_strategy(RegulationStrategy::Grounding, 0.5);
            engine.record_application_distress(0.5);
            engine.record_outcome(0.5 + delta);
        }
        let eff = engine
            .effectiveness(&RegulationStrategy::Grounding)
            .unwrap();
        assert_eq!(eff.applications, 4);
        assert_eq!(eff.successes, 3);
        assert!((eff.success_rate() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_strategy_effectiveness_neutral_prior() {
        let eff = StrategyEffectiveness::new();
        assert_eq!(eff.success_rate(), 0.5, "no data → 0.5 neutral prior");
    }

    #[test]
    fn test_most_effective_strategy() {
        let mut engine = RegulationEngine::new();
        // Validation: 4/4 success
        for _ in 0..4 {
            engine.apply_strategy(RegulationStrategy::Validation, 0.5);
            engine.record_application_distress(0.5);
            engine.record_outcome(0.3); // improvement
        }
        // Grounding: 1/4 success
        for delta in [0.1, 0.2, 0.1, -0.1] {
            engine.apply_strategy(RegulationStrategy::Grounding, 0.5);
            engine.record_application_distress(0.5);
            engine.record_outcome(0.5 + delta);
        }
        assert_eq!(
            engine.most_effective_strategy(),
            Some(RegulationStrategy::Validation)
        );
        assert_eq!(
            engine.least_effective_strategy(),
            Some(RegulationStrategy::Grounding)
        );
    }

    #[test]
    fn test_effectiveness_biased_selection() {
        let mut engine = RegulationEngine::new();
        // Build strong effectiveness record for Defusion (4/4 success)
        for _ in 0..4 {
            engine.apply_strategy(RegulationStrategy::Defusion, 0.5);
            engine.record_application_distress(0.5);
            engine.record_outcome(0.3); // improvement
        }
        let client = make_calm_client();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.5;
        alliance.goal_agreement = 0.5;
        alliance.task_agreement = 0.5;
        // Without effectiveness data: would select ExposurePrep or Validation
        // With effectiveness data: should prefer Defusion (100% success rate)
        let strategy = engine.select_strategy(&client, &alliance, false);
        assert_eq!(
            strategy,
            RegulationStrategy::Defusion,
            "should prefer historically effective strategy"
        );
    }

    #[test]
    fn test_effectiveness_bias_skips_poor_strategies() {
        let mut engine = RegulationEngine::new();
        // Defusion: 0/4 success (poor)
        for _ in 0..4 {
            engine.apply_strategy(RegulationStrategy::Defusion, 0.5);
            engine.record_application_distress(0.3);
            engine.record_outcome(0.6); // worsening
        }
        // Also add dream wisdom for Defusion
        engine.incorporate_dream_wisdom(3, 0.5); // ordinal 3 = Defusion

        let client = make_calm_client();
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.5;
        alliance.goal_agreement = 0.5;
        alliance.task_agreement = 0.5;

        let strategy = engine.select_strategy(&client, &alliance, false);
        // Should NOT select Defusion despite dream preference (poor effectiveness)
        assert_ne!(
            strategy,
            RegulationStrategy::Defusion,
            "should not prefer historically poor strategy even with dream bias"
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
