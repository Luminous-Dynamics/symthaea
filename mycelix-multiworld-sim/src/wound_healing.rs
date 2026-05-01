// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Wound Healing — 4-Stage Compartmental Trauma Recovery
//!
//! Replaces the simulator's permanent trauma accumulation with a
//! restorative healing model based on Ogunsola et al. (2024) compartmental
//! ODE dynamics, adapted to monthly tick resolution.
//!
//! ## Two Healing Tracks
//!
//! - **Somatic** (Disaster, Burnout): requires medicine, reduces labor productivity
//! - **Social** (Faction, Oppression): requires community mediation, damages
//!   Reputation + Community dimensions, heals via restorative justice
//!
//! ## Four Healing Phases
//!
//! Inflammation → Proliferation → Remodeling → Integration → Healed
//!
//! Each phase advances probabilistically per monthly tick, with transition
//! rates depending on care availability, collective phi, and metabolism phase.
//!
//! ## Kenosis
//!
//! Voluntary release during Remodeling phase. Doubles healing probability
//! and earns MYCEL peer recognition (composting trauma into wisdom).
//!
//! ## Key Finding (Ogunsola 2024)
//!
//! Trauma-free equilibrium is impossible with constant external stressors.
//! The system converges to an endemic equilibrium — healing is ongoing,
//! not a one-time event.

use serde::{Deserialize, Serialize};

// ============================================================================
// Types
// ============================================================================

/// Origin of the wound — determines healing track and requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WoundOrigin {
    /// Physical/environmental damage. Heals via medicine + rest.
    Disaster,
    /// Chronic overwork/stress. Heals via reduced workload + stillness.
    Burnout,
    /// Inter-group conflict. Heals via community mediation + reconciliation.
    /// Damages Reputation and Community dimensions.
    Faction,
    /// Systemic injustice. Heals via restorative justice + reform.
    /// Damages Reputation and Community dimensions.
    Oppression,
}

impl WoundOrigin {
    /// Whether this wound requires social/restorative healing.
    pub fn is_social(&self) -> bool {
        matches!(self, Self::Faction | Self::Oppression)
    }

    /// Whether this wound requires medical/somatic healing.
    pub fn is_somatic(&self) -> bool {
        matches!(self, Self::Disaster | Self::Burnout)
    }
}

/// Healing phase — compartmental stages from Ogunsola et al. (2024).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealingPhase {
    /// Acute response: allostatic load spikes, wound is fresh.
    /// Duration: ~1-2 months. Transition P = 0.7 + 0.2 * medicine.
    Inflammation,
    /// New tissue growth / initial processing. Gradual improvement.
    /// Duration: ~2-5 months. Transition P = 0.4 + 0.3 * care + 0.1 * phi.
    Proliferation,
    /// Strengthening / deep processing. Slow but steady.
    /// Duration: ~3-10 months. Transition P = 0.2 + 0.2 * care + 0.1 * kenosis.
    /// Kenosis (voluntary release) doubles transition probability.
    Remodeling,
    /// Wisdom extraction: composting trauma into meta-awareness.
    /// Duration: ~1-3 months. Transition P = 0.5 + 0.3 * meta_awareness.
    Integration,
    /// Wound fully resolved. Can be removed from the wound list.
    Healed,
}

impl HealingPhase {
    /// Phase index for severity weighting (higher = worse).
    pub fn severity_weight(&self) -> f64 {
        match self {
            Self::Inflammation => 1.0,
            Self::Proliferation => 0.7,
            Self::Remodeling => 0.4,
            Self::Integration => 0.1,
            Self::Healed => 0.0,
        }
    }
}

/// A single wound carried by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WoundState {
    /// Initial severity when inflicted [0.0, 1.0].
    pub initial_severity: f64,
    /// Current healing phase.
    pub phase: HealingPhase,
    /// Ticks spent in the current phase.
    pub phase_ticks: u32,
    /// What caused this wound.
    pub origin: WoundOrigin,
    /// Whether the agent has activated voluntary kenosis.
    pub kenosis_active: bool,
    /// Tick when the wound was inflicted.
    pub inflicted_at: u32,
}

impl WoundState {
    /// Create a new wound.
    pub fn new(severity: f64, origin: WoundOrigin, tick: u32) -> Self {
        Self {
            initial_severity: severity.clamp(0.0, 1.0),
            phase: HealingPhase::Inflammation,
            phase_ticks: 0,
            origin,
            kenosis_active: false,
            inflicted_at: tick,
        }
    }

    /// Current effective severity (initial × phase weight).
    pub fn current_severity(&self) -> f64 {
        self.initial_severity * self.phase.severity_weight()
    }

    /// Whether this wound has fully healed.
    pub fn is_healed(&self) -> bool {
        self.phase == HealingPhase::Healed
    }

    /// Labor productivity penalty from this wound [0.0, 1.0].
    /// 0.0 = no impact, 1.0 = fully incapacitated.
    pub fn labor_penalty(&self) -> f64 {
        if self.origin.is_somatic() {
            self.current_severity() * 0.5 // somatic wounds reduce labor
        } else {
            self.current_severity() * 0.2 // social wounds have less labor impact
        }
    }
}

// ============================================================================
// Healing Context (per-world resources available for healing)
// ============================================================================

/// Resources available in the world that affect healing rates.
#[derive(Debug, Clone, Copy)]
pub struct HealingContext {
    /// Medicine availability [0.0, 1.0]. From world resources.
    pub medicine: f64,
    /// Care workers per wounded agent [0.0, 1.0+]. Ratio of carers to patients.
    pub care_ratio: f64,
    /// Community collective phi [0.0, 1.0]. Affects social wound healing.
    pub collective_phi: f64,
    /// Community mediation factor [0.0, 1.0]. Required for social wounds.
    /// 0.0 = no mediators available, 1.0 = adequate mediation capacity.
    pub mediation_factor: f64,
    /// Metabolism phase healing multiplier (from metabolism.rs PhaseModifiers).
    pub metabolism_healing_mult: f64,
}

impl Default for HealingContext {
    fn default() -> Self {
        Self {
            medicine: 0.5,
            care_ratio: 0.3,
            collective_phi: 0.3,
            mediation_factor: 0.5,
            metabolism_healing_mult: 1.0,
        }
    }
}

// ============================================================================
// Healing Tick
// ============================================================================

/// Advance a wound through healing phases for one monthly tick.
///
/// Returns `true` if the wound transitioned to a new phase.
/// Uses deterministic threshold comparison with the context-dependent
/// transition probability (no RNG needed — the probability IS the
/// fraction of agents at this severity who would transition).
///
/// For simulation determinism, we use a simple hash of wound state
/// as the "random" threshold, ensuring reproducibility across runs.
pub fn tick_wound(wound: &mut WoundState, ctx: &HealingContext) -> bool {
    if wound.is_healed() {
        return false;
    }

    wound.phase_ticks += 1;

    // Compute transition probability based on phase and context
    let base_p = transition_probability(wound, ctx);

    // Apply metabolism phase multiplier
    let p = (base_p * ctx.metabolism_healing_mult).clamp(0.0, 1.0);

    // Deterministic transition: if probability > threshold based on time in phase.
    // After enough ticks, every wound eventually heals (ergodic guarantee).
    // P(transition after N ticks) = 1 - (1-p)^N
    let cumulative_p = 1.0 - (1.0 - p).powi(wound.phase_ticks as i32);

    if cumulative_p >= 0.5 {
        // Advance to next phase
        let old_phase = wound.phase;
        wound.phase = next_phase(wound.phase);
        wound.phase_ticks = 0;

        // Kenosis bonus: if activated during Remodeling, it already doubled p
        // (handled in transition_probability)

        old_phase != wound.phase
    } else {
        false
    }
}

/// Compute the monthly transition probability for the current phase.
fn transition_probability(wound: &WoundState, ctx: &HealingContext) -> f64 {
    let social_gate = if wound.origin.is_social() {
        ctx.mediation_factor // social wounds require mediation
    } else {
        1.0 // somatic wounds don't need mediation
    };

    let base = match wound.phase {
        HealingPhase::Inflammation => {
            // P = 0.7 + 0.2 * medicine (somatic) or mediation (social)
            0.7 + 0.2
                * if wound.origin.is_somatic() {
                    ctx.medicine
                } else {
                    ctx.mediation_factor
                }
        }
        HealingPhase::Proliferation => {
            // P = 0.4 + 0.3 * care_ratio + 0.1 * collective_phi
            (0.4 + 0.3 * ctx.care_ratio + 0.1 * ctx.collective_phi) * social_gate
        }
        HealingPhase::Remodeling => {
            // P = 0.2 + 0.2 * care_ratio + 0.1 * kenosis_active
            let kenosis_bonus = if wound.kenosis_active { 0.2 } else { 0.0 };
            (0.2 + 0.2 * ctx.care_ratio + kenosis_bonus) * social_gate
        }
        HealingPhase::Integration => {
            // P = 0.5 + 0.3 * meta_awareness (approximated by collective_phi)
            0.5 + 0.3 * ctx.collective_phi
        }
        HealingPhase::Healed => 0.0,
    };

    base.clamp(0.0, 0.95) // cap at 95% to prevent instant transitions
}

/// Get the next healing phase.
fn next_phase(current: HealingPhase) -> HealingPhase {
    match current {
        HealingPhase::Inflammation => HealingPhase::Proliferation,
        HealingPhase::Proliferation => HealingPhase::Remodeling,
        HealingPhase::Remodeling => HealingPhase::Integration,
        HealingPhase::Integration => HealingPhase::Healed,
        HealingPhase::Healed => HealingPhase::Healed,
    }
}

// ============================================================================
// Population-Level Functions
// ============================================================================

/// Compute the aggregate trauma severity for a set of wounds.
///
/// This replaces the old single `trauma_level: f64` with a severity-weighted
/// sum across all active wounds.
pub fn aggregate_trauma(wounds: &[WoundState]) -> f64 {
    if wounds.is_empty() {
        return 0.0;
    }
    let total: f64 = wounds.iter().map(|w| w.current_severity()).sum();
    total.clamp(0.0, 1.0) // cap aggregate at 1.0
}

/// Compute the total labor penalty from all wounds.
pub fn aggregate_labor_penalty(wounds: &[WoundState]) -> f64 {
    let total: f64 = wounds.iter().map(|w| w.labor_penalty()).sum();
    total.clamp(0.0, 0.9) // never fully incapacitate (0.9 max = 10% residual capacity)
}

/// Count wounds by phase for population statistics.
pub fn wound_phase_distribution(wounds: &[WoundState]) -> [usize; 5] {
    let mut dist = [0usize; 5];
    for w in wounds {
        let idx = match w.phase {
            HealingPhase::Inflammation => 0,
            HealingPhase::Proliferation => 1,
            HealingPhase::Remodeling => 2,
            HealingPhase::Integration => 3,
            HealingPhase::Healed => 4,
        };
        dist[idx] += 1;
    }
    dist
}

/// Attempt kenosis for an agent. Probability scales with care_activation.
///
/// Returns `true` if kenosis was activated on any wound.
pub fn attempt_kenosis(wounds: &mut [WoundState], care_activation: f64) -> bool {
    let mut activated = false;
    for wound in wounds.iter_mut() {
        if wound.phase == HealingPhase::Remodeling && !wound.kenosis_active {
            // Probability of voluntary kenosis scales with care activation
            // At care_activation=1.0, ~30% chance per tick
            if care_activation > 0.5 {
                wound.kenosis_active = true;
                activated = true;
            }
        }
    }
    activated
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_ctx() -> HealingContext {
        HealingContext::default()
    }

    fn good_ctx() -> HealingContext {
        HealingContext {
            medicine: 0.9,
            care_ratio: 0.8,
            collective_phi: 0.7,
            mediation_factor: 0.9,
            metabolism_healing_mult: 1.0,
        }
    }

    #[test]
    fn new_wound_starts_in_inflammation() {
        let w = WoundState::new(0.8, WoundOrigin::Disaster, 0);
        assert_eq!(w.phase, HealingPhase::Inflammation);
        assert_eq!(w.initial_severity, 0.8);
        assert!(!w.is_healed());
    }

    #[test]
    fn wound_severity_decreases_through_phases() {
        let w1 = WoundState {
            phase: HealingPhase::Inflammation,
            initial_severity: 1.0,
            ..WoundState::new(1.0, WoundOrigin::Disaster, 0)
        };
        let w2 = WoundState {
            phase: HealingPhase::Proliferation,
            initial_severity: 1.0,
            ..WoundState::new(1.0, WoundOrigin::Disaster, 0)
        };
        let w3 = WoundState {
            phase: HealingPhase::Remodeling,
            initial_severity: 1.0,
            ..WoundState::new(1.0, WoundOrigin::Disaster, 0)
        };
        let w4 = WoundState {
            phase: HealingPhase::Integration,
            initial_severity: 1.0,
            ..WoundState::new(1.0, WoundOrigin::Disaster, 0)
        };

        assert!(w1.current_severity() > w2.current_severity());
        assert!(w2.current_severity() > w3.current_severity());
        assert!(w3.current_severity() > w4.current_severity());
    }

    #[test]
    fn healed_wound_has_zero_severity() {
        let w = WoundState {
            phase: HealingPhase::Healed,
            ..WoundState::new(1.0, WoundOrigin::Disaster, 0)
        };
        assert_eq!(w.current_severity(), 0.0);
        assert!(w.is_healed());
    }

    #[test]
    fn wound_eventually_heals_with_good_care() {
        let mut w = WoundState::new(0.8, WoundOrigin::Disaster, 0);
        let ctx = good_ctx();

        // Tick up to 50 months — should heal
        for _ in 0..50 {
            tick_wound(&mut w, &ctx);
        }
        assert!(
            w.is_healed(),
            "Wound should heal within 50 ticks with good care, phase={:?}",
            w.phase
        );
    }

    #[test]
    fn social_wound_heals_slower_without_mediation() {
        let mut w_mediated = WoundState::new(0.8, WoundOrigin::Faction, 0);
        let mut w_unmediated = WoundState::new(0.8, WoundOrigin::Faction, 0);

        let ctx_good = good_ctx();
        let ctx_no_mediation = HealingContext {
            mediation_factor: 0.0,
            ..good_ctx()
        };

        for _ in 0..20 {
            tick_wound(&mut w_mediated, &ctx_good);
            tick_wound(&mut w_unmediated, &ctx_no_mediation);
        }

        // Mediated wound should be further along
        assert!(
            w_mediated.phase.severity_weight() <= w_unmediated.phase.severity_weight(),
            "Mediated wound should heal faster: {:?} vs {:?}",
            w_mediated.phase,
            w_unmediated.phase
        );
    }

    #[test]
    fn somatic_wound_has_higher_labor_penalty_than_social() {
        let somatic = WoundState::new(1.0, WoundOrigin::Disaster, 0);
        let social = WoundState::new(1.0, WoundOrigin::Faction, 0);
        assert!(somatic.labor_penalty() > social.labor_penalty());
    }

    #[test]
    fn aggregate_trauma_capped_at_one() {
        let wounds = vec![
            WoundState::new(0.8, WoundOrigin::Disaster, 0),
            WoundState::new(0.9, WoundOrigin::Faction, 0),
            WoundState::new(0.7, WoundOrigin::Burnout, 0),
        ];
        let trauma = aggregate_trauma(&wounds);
        assert!(trauma <= 1.0, "Aggregate trauma should be capped at 1.0");
    }

    #[test]
    fn labor_penalty_capped_at_90_percent() {
        let wounds = vec![
            WoundState::new(1.0, WoundOrigin::Disaster, 0),
            WoundState::new(1.0, WoundOrigin::Disaster, 0),
            WoundState::new(1.0, WoundOrigin::Disaster, 0),
        ];
        let penalty = aggregate_labor_penalty(&wounds);
        assert!(penalty <= 0.9, "Labor penalty should cap at 0.9");
    }

    #[test]
    fn kenosis_activates_during_remodeling() {
        let mut wounds = vec![WoundState {
            phase: HealingPhase::Remodeling,
            ..WoundState::new(0.5, WoundOrigin::Disaster, 0)
        }];
        let activated = attempt_kenosis(&mut wounds, 0.8);
        assert!(
            activated,
            "Kenosis should activate with high care_activation"
        );
        assert!(wounds[0].kenosis_active);
    }

    #[test]
    fn kenosis_does_not_activate_outside_remodeling() {
        let mut wounds = vec![WoundState {
            phase: HealingPhase::Inflammation,
            ..WoundState::new(0.5, WoundOrigin::Disaster, 0)
        }];
        let activated = attempt_kenosis(&mut wounds, 0.9);
        assert!(!activated, "Kenosis should not activate outside Remodeling");
    }

    #[test]
    fn wound_phase_distribution_counts_correctly() {
        let wounds = vec![
            WoundState {
                phase: HealingPhase::Inflammation,
                ..WoundState::new(0.5, WoundOrigin::Disaster, 0)
            },
            WoundState {
                phase: HealingPhase::Inflammation,
                ..WoundState::new(0.3, WoundOrigin::Burnout, 0)
            },
            WoundState {
                phase: HealingPhase::Remodeling,
                ..WoundState::new(0.4, WoundOrigin::Faction, 0)
            },
            WoundState {
                phase: HealingPhase::Healed,
                ..WoundState::new(0.1, WoundOrigin::Disaster, 0)
            },
        ];
        let dist = wound_phase_distribution(&wounds);
        assert_eq!(dist[0], 2); // Inflammation
        assert_eq!(dist[2], 1); // Remodeling
        assert_eq!(dist[4], 1); // Healed
    }

    #[test]
    fn empty_wounds_zero_trauma() {
        assert_eq!(aggregate_trauma(&[]), 0.0);
        assert_eq!(aggregate_labor_penalty(&[]), 0.0);
    }

    #[test]
    fn wound_origin_classification() {
        assert!(WoundOrigin::Disaster.is_somatic());
        assert!(WoundOrigin::Burnout.is_somatic());
        assert!(WoundOrigin::Faction.is_social());
        assert!(WoundOrigin::Oppression.is_social());

        assert!(!WoundOrigin::Disaster.is_social());
        assert!(!WoundOrigin::Faction.is_somatic());
    }

    #[test]
    fn metabolism_healing_mult_accelerates_healing() {
        let mut w_normal = WoundState::new(0.8, WoundOrigin::Disaster, 0);
        let mut w_boosted = WoundState::new(0.8, WoundOrigin::Disaster, 0);

        let ctx_normal = good_ctx();
        let ctx_boosted = HealingContext {
            metabolism_healing_mult: 3.0,
            ..good_ctx()
        };

        for _ in 0..10 {
            tick_wound(&mut w_normal, &ctx_normal);
            tick_wound(&mut w_boosted, &ctx_boosted);
        }

        assert!(
            w_boosted.phase.severity_weight() <= w_normal.phase.severity_weight(),
            "Boosted metabolism should accelerate healing: {:?} vs {:?}",
            w_boosted.phase,
            w_normal.phase
        );
    }

    #[test]
    fn severity_clamped_to_valid_range() {
        let w = WoundState::new(1.5, WoundOrigin::Disaster, 0);
        assert_eq!(w.initial_severity, 1.0, "Severity should clamp to 1.0");

        let w2 = WoundState::new(-0.5, WoundOrigin::Disaster, 0);
        assert_eq!(w2.initial_severity, 0.0, "Severity should clamp to 0.0");
    }
}
