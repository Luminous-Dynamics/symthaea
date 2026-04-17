// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Graduated Sanctions — Ostrom Principle 5
//!
//! Implements graduated sanctions from Zosh et al. (2025): three levels
//! of fines that escalate with repeated violations.
//!
//! ## Key Design Choice (Zosh Finding)
//!
//! Fine revenue is **NOT redistributed** to other agents. Redistribution
//! causes draconian sanctions to emerge. Instead, fines are burned
//! (destroyed), reducing total SAP supply.
//!
//! ## Trigger
//!
//! Sanctions activate when governance oppression_index exceeds thresholds.
//! Agents with low coordination_understanding are penalized — they are
//! the ones defecting in commons dilemmas.

use crate::agent::CivAgent;
use crate::governance::WorldGovernance;
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants (from Zosh et al. 2025)
// ============================================================================

/// Fine rates as fraction of SAP balance.
const MINOR_RATE: f64 = 0.10;
const MODERATE_RATE: f64 = 0.30;
const SEVERE_RATE: f64 = 0.60;

/// Oppression thresholds that trigger sanctions.
const MINOR_THRESHOLD: f64 = 0.2;
const MODERATE_THRESHOLD: f64 = 0.4;
const SEVERE_THRESHOLD: f64 = 0.6;

/// Coordination understanding below which agents are flagged for defection.
const DEFECTION_THRESHOLD: f64 = 0.3;

/// Annual reset period in ticks.
const SANCTION_RESET_TICKS: u32 = 12;

// ============================================================================
// Types
// ============================================================================

/// Sanction severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SanctionLevel {
    None,
    Minor,
    Moderate,
    Severe,
}

impl SanctionLevel {
    /// Fine rate for this level.
    pub fn fine_rate(&self) -> f64 {
        match self {
            Self::None => 0.0,
            Self::Minor => MINOR_RATE,
            Self::Moderate => MODERATE_RATE,
            Self::Severe => SEVERE_RATE,
        }
    }
}

/// Result of applying sanctions to a population.
#[derive(Debug, Default)]
pub struct SanctionResult {
    /// Total SAP burned (destroyed, NOT redistributed).
    pub total_burned: f64,
    /// Number of agents sanctioned at each level.
    pub minor_count: u32,
    pub moderate_count: u32,
    pub severe_count: u32,
}

// ============================================================================
// Sanction Logic
// ============================================================================

/// Determine sanction level from oppression index.
pub fn sanction_level_from_oppression(oppression_index: f64) -> SanctionLevel {
    if oppression_index >= SEVERE_THRESHOLD {
        SanctionLevel::Severe
    } else if oppression_index >= MODERATE_THRESHOLD {
        SanctionLevel::Moderate
    } else if oppression_index >= MINOR_THRESHOLD {
        SanctionLevel::Minor
    } else {
        SanctionLevel::None
    }
}

/// Apply graduated sanctions to agents based on governance state.
///
/// Agents with low coordination_understanding (< 0.3) are sanctioned
/// when oppression_index exceeds thresholds. Fine is collected from
/// SAP balance and burned (per Zosh finding: no redistribution).
///
/// Returns the total SAP burned and counts per level.
pub fn apply_sanctions(
    agents: &mut [CivAgent],
    oppression_index: f64,
    current_tick: u32,
) -> SanctionResult {
    apply_sanctions_with_phase2(agents, oppression_index, current_tick, true)
}

/// `apply_sanctions` with an explicit Phase 2 toggle. When `phase2_enabled`
/// is false the violation-recording hook is skipped — this is the A2
/// counterfactual path.
pub fn apply_sanctions_with_phase2(
    agents: &mut [CivAgent],
    oppression_index: f64,
    current_tick: u32,
    phase2_enabled: bool,
) -> SanctionResult {
    let level = sanction_level_from_oppression(oppression_index);
    if level == SanctionLevel::None {
        return SanctionResult::default();
    }

    let fine_rate = level.fine_rate();
    let mut result = SanctionResult::default();

    for agent in agents.iter_mut().filter(|a| a.is_alive()) {
        // Only sanction agents who are defecting (low coordination understanding)
        if agent.coordination_understanding < DEFECTION_THRESHOLD && agent.sap_balance > 0.0 {
            let fine = agent.sap_balance * fine_rate;
            agent.sap_balance -= fine;
            result.total_burned += fine;

            // Phase 2b: restorative justice — a sanction is a moral violation.
            // Severe sanctions count as two violations (they degrade tier faster).
            if phase2_enabled {
                agent.justice.record_violation(current_tick);
                if matches!(level, SanctionLevel::Severe) {
                    agent.justice.record_violation(current_tick);
                }
            }

            match level {
                SanctionLevel::Minor => result.minor_count += 1,
                SanctionLevel::Moderate => result.moderate_count += 1,
                SanctionLevel::Severe => result.severe_count += 1,
                SanctionLevel::None => {}
            }
        }
    }

    result
}

/// Per-tick restorative-corrections pass.
///
/// Reward corrective behavior: agents accumulating care work (high
/// `tend_balance`) or strong virtue-care ethics record a correction. This is
/// probabilistic so populations with diverse ethics display diverse recovery
/// profiles. Returns the total corrections recorded.
pub fn apply_restorative_corrections(
    agents: &mut [CivAgent],
    current_tick: u32,
    rng: &mut crate::stochastic::StochasticEngine,
) -> u32 {
    let mut count = 0u32;
    for agent in agents.iter_mut().filter(|a| a.is_alive()) {
        // Base probability scales with observable care signals.
        let care_signal = (agent.tend_balance / 40.0).clamp(0.0, 1.0) * 0.5
            + agent.ethics.virtue_care * 0.25
            + agent.ethics.relational * 0.15
            + (agent.mycel_score).clamp(0.0, 1.0) * 0.10;
        let p = care_signal.clamp(0.0, 0.15); // cap at 15% per tick
        if p > 0.0 && rng.bernoulli(p) {
            agent.justice.record_correction(current_tick);
            count += 1;
        }
    }
    count
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_sanctions_below_threshold() {
        assert_eq!(sanction_level_from_oppression(0.1), SanctionLevel::None);
        assert_eq!(sanction_level_from_oppression(0.0), SanctionLevel::None);
    }

    #[test]
    fn minor_sanctions_at_threshold() {
        assert_eq!(sanction_level_from_oppression(0.2), SanctionLevel::Minor);
        assert_eq!(sanction_level_from_oppression(0.3), SanctionLevel::Minor);
    }

    #[test]
    fn moderate_sanctions_at_threshold() {
        assert_eq!(sanction_level_from_oppression(0.4), SanctionLevel::Moderate);
    }

    #[test]
    fn severe_sanctions_at_threshold() {
        assert_eq!(sanction_level_from_oppression(0.6), SanctionLevel::Severe);
        assert_eq!(sanction_level_from_oppression(1.0), SanctionLevel::Severe);
    }

    #[test]
    fn fine_rates_are_graduated() {
        assert!(SanctionLevel::Minor.fine_rate() < SanctionLevel::Moderate.fine_rate());
        assert!(SanctionLevel::Moderate.fine_rate() < SanctionLevel::Severe.fine_rate());
        assert_eq!(SanctionLevel::None.fine_rate(), 0.0);
    }

    #[test]
    fn fine_rates_match_zosh() {
        assert_eq!(SanctionLevel::Minor.fine_rate(), 0.10);
        assert_eq!(SanctionLevel::Moderate.fine_rate(), 0.30);
        assert_eq!(SanctionLevel::Severe.fine_rate(), 0.60);
    }
}
