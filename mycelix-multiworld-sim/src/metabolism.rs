// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Metabolism Cycles — Configurable 4-Phase Governance Rhythm
//!
//! Implements the Metabolism Charter's 28-day, 4-phase governance cycle as
//! a configurable overlay on the simulator's monthly tick loop.
//!
//! ## Temporal Abstraction
//!
//! Since 1 tick = 1 month ≈ 1 full metabolism cycle, each tick evaluates
//! the **aggregate effect** of all 4 phases within that month.  When
//! `cycle_ticks > 1`, the world rotates through phase emphasis:
//!
//! - `cycle_ticks=1`: balanced monthly (all 4 phases contribute equally)
//! - `cycle_ticks=4`: quarterly rotation (one dominant phase per tick)
//! - `cycle_ticks=12`: annual rhythm (3 months per phase)
//!
//! ## Four Phases
//!
//! 1. **Release**: Letting go.  Trauma processing bonus, increased resource decay.
//! 2. **Stillness**: Rest.  6× allostatic recovery, voting suppressed, reduced output.
//! 3. **Creation**: Building.  Innovation bonus, education effectiveness boost.
//! 4. **Integration**: Committing.  Governance effectiveness, voting encouraged.

use serde::{Deserialize, Serialize};

// ============================================================================
// Configuration
// ============================================================================

/// Configurable metabolism cycle parameters.
///
/// All parameters are tunable via PolicyConfig, allowing communities to
/// experiment with different rhythms (monthly, quarterly, annual, or custom).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolismConfig {
    /// Whether metabolism cycles are enabled.
    pub enabled: bool,
    /// How many ticks per full cycle.
    /// - 1 = monthly (default): all phases contribute equally per tick
    /// - 4 = quarterly: one dominant phase per tick
    /// - 12 = annual: 3 months per phase
    pub cycle_ticks: u32,
    /// Relative emphasis of each phase [Release, Stillness, Creation, Integration].
    /// Must sum to 1.0.  Default: [0.25, 0.25, 0.25, 0.25] (balanced).
    pub phase_emphasis: [f64; 4],
    /// Allostatic recovery multiplier during Stillness phase.
    pub stillness_recovery_mult: f64,
    /// Innovation rate multiplier during Creation phase.
    pub creation_innovation_mult: f64,
    /// Governance effectiveness multiplier during Integration phase.
    pub integration_governance_mult: f64,
    /// Wound healing rate multiplier during Release phase.
    pub release_healing_mult: f64,
    /// Production output multiplier during Stillness phase (reduced output).
    pub stillness_production_mult: f64,
}

impl Default for MetabolismConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cycle_ticks: 1,
            phase_emphasis: [0.25, 0.25, 0.25, 0.25],
            stillness_recovery_mult: 6.0,
            creation_innovation_mult: 1.5,
            integration_governance_mult: 1.3,
            release_healing_mult: 2.0,
            stillness_production_mult: 0.8,
        }
    }
}

impl MetabolismConfig {
    /// Emergency mode: compressed cycle with reduced Stillness.
    pub fn emergency() -> Self {
        Self {
            enabled: true,
            cycle_ticks: 1,
            phase_emphasis: [0.10, 0.10, 0.30, 0.50], // Heavy Integration + Creation
            stillness_recovery_mult: 3.0,
            creation_innovation_mult: 2.0,
            integration_governance_mult: 1.5,
            release_healing_mult: 1.5,
            stillness_production_mult: 0.9,
        }
    }

    /// Regenerative mode: extended Stillness and Release for healing communities.
    pub fn regenerative() -> Self {
        Self {
            enabled: true,
            cycle_ticks: 1,
            phase_emphasis: [0.35, 0.35, 0.15, 0.15], // Heavy Release + Stillness
            stillness_recovery_mult: 8.0,
            creation_innovation_mult: 1.2,
            integration_governance_mult: 1.1,
            release_healing_mult: 3.0,
            stillness_production_mult: 0.6,
        }
    }
}

// ============================================================================
// Phase Types
// ============================================================================

/// The four metabolism phases from the Metabolism Charter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetabolismPhase {
    /// Days 1–7: Letting go of what no longer serves.
    Release,
    /// Days 8–14: Holding uncertainty; gathering before deciding.
    Stillness,
    /// Days 15–21: Forming new patterns; experimentation.
    Creation,
    /// Days 22–28: Commitment and completion.
    Integration,
}

impl MetabolismPhase {
    /// Phase index (0-3).
    pub fn index(&self) -> usize {
        match self {
            Self::Release => 0,
            Self::Stillness => 1,
            Self::Creation => 2,
            Self::Integration => 3,
        }
    }

    /// From index.
    pub fn from_index(idx: usize) -> Self {
        match idx % 4 {
            0 => Self::Release,
            1 => Self::Stillness,
            2 => Self::Creation,
            _ => Self::Integration,
        }
    }

    /// Phase name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Release => "Release",
            Self::Stillness => "Stillness",
            Self::Creation => "Creation",
            Self::Integration => "Integration",
        }
    }
}

/// Per-tick modifiers computed from the metabolism cycle.
///
/// Each modifier is a multiplier that affects the corresponding system:
/// - `> 1.0` = enhanced during this phase
/// - `= 1.0` = no effect
/// - `< 1.0` = suppressed during this phase
#[derive(Debug, Clone, Copy)]
pub struct PhaseModifiers {
    /// Allostatic load recovery rate multiplier.
    pub recovery_mult: f64,
    /// Innovation/knowledge growth multiplier.
    pub innovation_mult: f64,
    /// Governance effectiveness multiplier (proposal quality, amendment success).
    pub governance_mult: f64,
    /// Wound healing rate multiplier.
    pub healing_mult: f64,
    /// Economic production output multiplier.
    pub production_mult: f64,
    /// Voting suppression factor [0.0, 1.0].
    /// 0.0 = no suppression, 1.0 = fully blocked.
    pub voting_suppression: f64,
}

impl Default for PhaseModifiers {
    fn default() -> Self {
        Self {
            recovery_mult: 1.0,
            innovation_mult: 1.0,
            governance_mult: 1.0,
            healing_mult: 1.0,
            production_mult: 1.0,
            voting_suppression: 0.0,
        }
    }
}

// ============================================================================
// Metabolism State (per-world)
// ============================================================================

/// Per-world metabolism cycle tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolismState {
    /// Current dominant phase (when cycle_ticks > 1).
    pub current_phase: MetabolismPhase,
    /// Ticks elapsed in the current cycle.
    pub cycle_position: u32,
    /// Total cycles completed.
    pub cycles_completed: u64,
}

impl Default for MetabolismState {
    fn default() -> Self {
        Self {
            current_phase: MetabolismPhase::Release,
            cycle_position: 0,
            cycles_completed: 0,
        }
    }
}

// ============================================================================
// Computation
// ============================================================================

/// Compute the aggregate phase modifiers for this tick.
///
/// When `cycle_ticks=1` (monthly), all 4 phases contribute equally.
/// When `cycle_ticks=4` (quarterly), the dominant phase for this tick
/// gets full weight while others contribute proportionally less.
pub fn compute_modifiers(
    config: &MetabolismConfig,
    state: &mut MetabolismState,
    _tick: u32,
) -> PhaseModifiers {
    if !config.enabled {
        return PhaseModifiers::default();
    }

    // Determine phase emphasis for this tick
    let emphasis = if config.cycle_ticks <= 1 {
        // Monthly: all phases equal
        config.phase_emphasis
    } else {
        // Multi-tick cycle: dominant phase gets boosted emphasis
        let phase_idx = (state.cycle_position / (config.cycle_ticks / 4).max(1)) as usize % 4;
        state.current_phase = MetabolismPhase::from_index(phase_idx);

        let mut e = [0.0f64; 4];
        let dominant_weight = 0.7; // dominant phase gets 70%
        let others_weight = 0.3 / 3.0; // remaining 30% split among other 3
        for i in 0..4 {
            e[i] = if i == phase_idx {
                dominant_weight
            } else {
                others_weight
            };
        }
        e
    };

    // Advance cycle position
    state.cycle_position += 1;
    if state.cycle_position >= config.cycle_ticks.max(1) {
        state.cycle_position = 0;
        state.cycles_completed += 1;
    }

    // Compute composite modifiers from phase emphasis
    // Each modifier blends the phase-specific multiplier with the neutral baseline (1.0)
    // weighted by that phase's emphasis.
    let recovery = blend(emphasis[1], config.stillness_recovery_mult);
    let innovation = blend(emphasis[2], config.creation_innovation_mult);
    let governance = blend(emphasis[3], config.integration_governance_mult);
    let healing = blend(emphasis[0], config.release_healing_mult);
    let production = blend(emphasis[1], config.stillness_production_mult);

    PhaseModifiers {
        recovery_mult: recovery,
        innovation_mult: innovation,
        governance_mult: governance,
        healing_mult: healing,
        production_mult: production,
        voting_suppression: emphasis[1] * 0.5, // Stillness partially suppresses voting
    }
}

/// Blend a phase multiplier with neutral (1.0) based on emphasis weight.
///
/// When emphasis = 0.0, returns 1.0 (no effect).
/// When emphasis = 1.0, returns the full multiplier.
fn blend(emphasis: f64, multiplier: f64) -> f64 {
    1.0 + emphasis * (multiplier - 1.0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_balanced_phases() {
        let config = MetabolismConfig::default();
        let sum: f64 = config.phase_emphasis.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Phase emphasis must sum to 1.0");
    }

    #[test]
    fn default_modifiers_are_near_neutral_for_monthly_cycle() {
        let config = MetabolismConfig::default();
        let mut state = MetabolismState::default();
        let mods = compute_modifiers(&config, &mut state, 0);

        // With equal 0.25 emphasis, recovery modifier = blend(0.25, 6.0) = 1 + 0.25*5 = 2.25
        assert!(
            mods.recovery_mult > 1.0,
            "Stillness should boost recovery: {}",
            mods.recovery_mult
        );
        assert!(
            mods.innovation_mult > 1.0,
            "Creation should boost innovation"
        );
        assert!(
            mods.governance_mult > 1.0,
            "Integration should boost governance"
        );
        assert!(mods.healing_mult > 1.0, "Release should boost healing");
    }

    #[test]
    fn disabled_metabolism_returns_neutral() {
        let mut config = MetabolismConfig::default();
        config.enabled = false;
        let mut state = MetabolismState::default();
        let mods = compute_modifiers(&config, &mut state, 0);

        assert_eq!(mods.recovery_mult, 1.0);
        assert_eq!(mods.innovation_mult, 1.0);
        assert_eq!(mods.governance_mult, 1.0);
        assert_eq!(mods.healing_mult, 1.0);
        assert_eq!(mods.production_mult, 1.0);
        assert_eq!(mods.voting_suppression, 0.0);
    }

    #[test]
    fn quarterly_cycle_rotates_phases() {
        let mut config = MetabolismConfig::default();
        config.cycle_ticks = 4;
        let mut state = MetabolismState::default();

        // Tick through 4 phases
        let phases: Vec<MetabolismPhase> = (0..4)
            .map(|t| {
                let _ = compute_modifiers(&config, &mut state, t);
                state.current_phase
            })
            .collect();

        assert_eq!(phases[0], MetabolismPhase::Release);
        assert_eq!(phases[1], MetabolismPhase::Stillness);
        assert_eq!(phases[2], MetabolismPhase::Creation);
        assert_eq!(phases[3], MetabolismPhase::Integration);
    }

    #[test]
    fn cycle_completes_and_resets() {
        let mut config = MetabolismConfig::default();
        config.cycle_ticks = 4;
        let mut state = MetabolismState::default();

        for t in 0..8 {
            compute_modifiers(&config, &mut state, t);
        }

        assert_eq!(
            state.cycles_completed, 2,
            "Should complete 2 full cycles in 8 ticks"
        );
        assert_eq!(
            state.cycle_position, 0,
            "Should reset to 0 after completing cycle"
        );
    }

    #[test]
    fn stillness_dominant_suppresses_voting() {
        let mut config = MetabolismConfig::default();
        config.cycle_ticks = 4;
        let mut state = MetabolismState::default();

        // Tick 0: Release (phase 0)
        let mods_release = compute_modifiers(&config, &mut state, 0);

        // Tick 1: Stillness (phase 1)
        let mods_stillness = compute_modifiers(&config, &mut state, 1);

        assert!(
            mods_stillness.voting_suppression > mods_release.voting_suppression,
            "Stillness should suppress voting more than Release"
        );
    }

    #[test]
    fn creation_dominant_boosts_innovation() {
        let mut config = MetabolismConfig::default();
        config.cycle_ticks = 4;
        let mut state = MetabolismState::default();

        // Skip to Creation (tick 2)
        compute_modifiers(&config, &mut state, 0);
        compute_modifiers(&config, &mut state, 1);
        let mods_creation = compute_modifiers(&config, &mut state, 2);

        // Compare with Release (tick 0, use fresh state)
        let mut state2 = MetabolismState::default();
        let mods_release = compute_modifiers(&config, &mut state2, 0);

        assert!(
            mods_creation.innovation_mult > mods_release.innovation_mult,
            "Creation should boost innovation more than Release: {} vs {}",
            mods_creation.innovation_mult,
            mods_release.innovation_mult
        );
    }

    #[test]
    fn regenerative_preset_heavy_stillness() {
        let config = MetabolismConfig::regenerative();
        assert!(
            config.phase_emphasis[1] > 0.3,
            "Regenerative should emphasize Stillness"
        );
        assert!(
            config.stillness_recovery_mult > 6.0,
            "Regenerative should have high recovery"
        );
    }

    #[test]
    fn emergency_preset_heavy_integration() {
        let config = MetabolismConfig::emergency();
        assert!(
            config.phase_emphasis[3] > 0.4,
            "Emergency should emphasize Integration"
        );
        assert!(
            config.phase_emphasis[1] < 0.15,
            "Emergency should minimize Stillness"
        );
    }

    #[test]
    fn blend_function_correctness() {
        assert_eq!(blend(0.0, 6.0), 1.0, "Zero emphasis = no effect");
        assert_eq!(blend(1.0, 6.0), 6.0, "Full emphasis = full multiplier");
        assert!(
            (blend(0.5, 6.0) - 3.5).abs() < 1e-10,
            "Half emphasis = halfway"
        );
        assert!((blend(0.25, 6.0) - 2.25).abs() < 1e-10, "Quarter emphasis");
    }

    #[test]
    fn phase_from_index_wraps() {
        assert_eq!(MetabolismPhase::from_index(0), MetabolismPhase::Release);
        assert_eq!(MetabolismPhase::from_index(4), MetabolismPhase::Release); // wraps
        assert_eq!(MetabolismPhase::from_index(7), MetabolismPhase::Integration);
    }

    #[test]
    fn all_presets_have_valid_emphasis() {
        for config in [
            MetabolismConfig::default(),
            MetabolismConfig::emergency(),
            MetabolismConfig::regenerative(),
        ] {
            let sum: f64 = config.phase_emphasis.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Phase emphasis must sum to 1.0, got {}",
                sum
            );
            for (i, &e) in config.phase_emphasis.iter().enumerate() {
                assert!(e >= 0.0, "Phase {} emphasis must be non-negative: {}", i, e);
            }
        }
    }

    #[test]
    fn annual_cycle_12_ticks() {
        let mut config = MetabolismConfig::default();
        config.cycle_ticks = 12;
        let mut state = MetabolismState::default();

        // Each phase should last 3 ticks
        for t in 0..12 {
            compute_modifiers(&config, &mut state, t);
        }

        assert_eq!(
            state.cycles_completed, 1,
            "12 ticks should complete 1 annual cycle"
        );
    }
}
