// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # KREDIT System
//!
//! Sponsor-collateralized credit for AI agents.
//!
//! ## Trust-Derived KREDIT
//!
//! KREDIT caps are now derived from K-Vector trust scores rather than
//! being arbitrarily set by sponsors. This creates accountability:
//! - Higher trust = higher KREDIT cap
//! - Poor behavior degrades trust = lower future KREDIT
//! - Gaming is prevented because trust requires consistent positive outcomes

use super::{AgentClass, AgentStatus, InstrumentalActor};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};

/// KREDIT allocation for an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KreditAllocation {
    /// Maximum KREDIT per epoch (30 days)
    pub epoch_cap: u64,
    /// Current epoch balance
    pub current_balance: i64,
    /// Sponsor collateral
    pub sponsor_collateral: SponsorCollateral,
    /// Epoch start timestamp
    pub epoch_start: u64,
}

/// Sponsor's collateral commitment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SponsorCollateral {
    /// Sponsor's CIV at allocation time
    pub civ_at_allocation: f64,
    /// SAP locked as collateral
    pub sap_locked: u64,
    /// Maximum liability sponsor accepts
    pub max_liability: u64,
}

/// KREDIT cost table for actions
#[derive(Debug, Clone, Copy)]
pub struct KreditCosts;

impl KreditCosts {
    /// Transaction cost per SAP
    pub const TRANSACTION_PER_SAP: f64 = 0.1;
    /// Minimum transaction cost
    pub const TRANSACTION_MIN: u64 = 1;
    /// API call cost
    pub const API_CALL: u64 = 1;
    /// Data query cost
    pub const DATA_QUERY: u64 = 1;
    /// Validation participation cost
    pub const VALIDATION: u64 = 10;
    /// Proposal submission cost
    pub const PROPOSAL: u64 = 100;
    /// HEARTH interaction cost
    pub const HEARTH_INTERACTION: u64 = 5;
}

/// Calculate KREDIT cap for an agent (legacy sponsor-based method)
///
/// Note: For trust-derived KREDIT, use `calculate_kredit_cap_from_trust` instead.
pub fn calculate_kredit_cap(
    sponsor_civ: f64,
    sap_locked: u64,
    agent_class: AgentClass,
    active_agent_count: u32,
) -> Result<u64, KreditError> {
    // Minimum sponsor CIV
    if sponsor_civ < 0.5 {
        return Err(KreditError::InsufficientSponsorCiv);
    }

    // Base KREDIT from sponsor CIV (quadratic scaling)
    let civ_factor = sponsor_civ.powi(2);

    // Class multiplier
    let class_multiplier = match agent_class {
        AgentClass::Autonomous => 0.5,
        AgentClass::Supervised => 1.0,
        AgentClass::Assistive => 1.5,
        AgentClass::Observer => 0.1,
    };

    // Collateral factor (SAP locked / 1000)
    let collateral_factor = (sap_locked as f64 / 1000.0).min(10.0);

    // Total active agents penalty
    let agent_penalty = 1.0 / (active_agent_count as f64 + 1.0);

    let kredit_cap =
        (10_000.0 * civ_factor * class_multiplier * collateral_factor * agent_penalty) as u64;

    Ok(kredit_cap.max(100)) // Minimum 100 KREDIT
}

/// Calculate KREDIT cap from agent's K-Vector trust score
///
/// This is the new trust-derived method where KREDIT allocation
/// is determined by the agent's proven trustworthiness rather than
/// arbitrary sponsor decisions.
///
/// ## Formula
///
/// kredit_cap = base_kredit * trust_factor * class_multiplier * sponsor_factor
///
/// Where:
/// - base_kredit = 20,000
/// - trust_factor = trust_score^1.5 (superlinear reward for high trust)
/// - class_multiplier varies by AgentClass
/// - sponsor_factor = sqrt(sponsor_civ) (softer CIV dependency)
pub fn calculate_kredit_cap_from_trust(
    k_vector: &KVector,
    agent_class: AgentClass,
    sponsor_civ: f64,
) -> Result<u64, KreditError> {
    // Still require minimum sponsor CIV
    if sponsor_civ < 0.5 {
        return Err(KreditError::InsufficientSponsorCiv);
    }

    let trust_score = k_vector.trust_score();

    // Superlinear trust factor (rewards high trust agents)
    let trust_factor = (trust_score as f64).powf(1.5);

    // Class multiplier (same as legacy)
    let class_multiplier = match agent_class {
        AgentClass::Autonomous => 0.5,
        AgentClass::Supervised => 1.0,
        AgentClass::Assistive => 1.5,
        AgentClass::Observer => 0.1,
    };

    // Softer sponsor CIV factor (sqrt instead of square)
    // This ensures trust matters more than sponsor backing
    let sponsor_factor = sponsor_civ.sqrt();

    let base_kredit = 20_000.0;
    let kredit_cap = (base_kredit * trust_factor * class_multiplier * sponsor_factor) as u64;

    Ok(kredit_cap.max(100)) // Minimum 100 KREDIT
}

/// Recalculate and update agent's KREDIT cap based on current trust score
///
/// Call this after updating the agent's K-Vector to adjust their
/// KREDIT allocation based on evolved trust.
pub fn recalculate_agent_kredit_cap(
    agent: &mut InstrumentalActor,
    sponsor_civ: f64,
) -> Result<u64, KreditError> {
    let new_cap = calculate_kredit_cap_from_trust(&agent.k_vector, agent.agent_class, sponsor_civ)?;

    agent.kredit_cap = new_cap;

    // If balance exceeds new cap (trust decreased), truncate
    if agent.kredit_balance > new_cap as i64 {
        agent.kredit_balance = new_cap as i64;
    }

    Ok(new_cap)
}

/// Consume KREDIT for an action
pub fn consume_kredit(agent: &mut InstrumentalActor, cost: u64) -> Result<(), KreditError> {
    // Check if action would exceed floor (allow negative up to -10%)
    let floor = -(agent.kredit_cap as i64 / 10);
    if (agent.kredit_balance - cost as i64) < floor {
        return Err(KreditError::InsufficientKredit);
    }

    agent.kredit_balance -= cost as i64;

    // Check throttle threshold
    if agent.kredit_balance < 0 {
        agent.status = AgentStatus::Throttled;
    }

    Ok(())
}

/// Process negative KREDIT handling
pub fn process_negative_kredit(
    agent: &mut InstrumentalActor,
    sponsor_sap_locked: &mut u64,
    sponsor_sap_balance: &mut u64,
    sponsor_civ: &mut f64,
) -> LiabilityResult {
    if agent.kredit_balance >= 0 {
        return LiabilityResult::NoAction;
    }

    let deficit_ratio = -agent.kredit_balance as f64 / agent.kredit_cap as f64;

    match deficit_ratio {
        r if r <= 0.1 => {
            agent.status = AgentStatus::Throttled;
            LiabilityResult::Throttled
        }
        r if r <= 0.2 => {
            agent.status = AgentStatus::Suspended;
            LiabilityResult::Suspended
        }
        _ => {
            // Deduct from sponsor collateral
            let liability = (-agent.kredit_balance as u64).min(*sponsor_sap_locked);
            *sponsor_sap_locked -= liability;
            *sponsor_sap_balance -= liability;

            // CIV penalty
            *sponsor_civ = (*sponsor_civ - 0.05).max(0.0);

            // Revoke agent
            agent.status = AgentStatus::Revoked;

            LiabilityResult::LiabilityTriggered { amount: liability }
        }
    }
}

/// Result of liability processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiabilityResult {
    /// No action needed
    NoAction,
    /// Agent throttled
    Throttled,
    /// Agent suspended
    Suspended,
    /// Liability triggered, SAP deducted
    LiabilityTriggered {
        /// Amount of SAP deducted.
        amount: u64,
    },
}

/// KREDIT-related errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KreditError {
    /// Sponsor CIV below minimum
    InsufficientSponsorCiv,
    /// Not enough KREDIT for action
    InsufficientKredit,
    /// Agent not in operational status
    AgentNotOperational,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;

    #[test]
    fn test_kredit_cap_calculation() {
        let cap = calculate_kredit_cap(0.7, 5000, AgentClass::Supervised, 0).unwrap();

        // 10000 * 0.49 * 1.0 * 5.0 * 1.0 = 24500
        assert!(cap > 20000 && cap < 30000);
    }

    #[test]
    fn test_insufficient_civ() {
        let result = calculate_kredit_cap(0.3, 5000, AgentClass::Supervised, 0);
        assert_eq!(result, Err(KreditError::InsufficientSponsorCiv));
    }

    #[test]
    fn test_consume_kredit() {
        let mut agent = InstrumentalActor {
            agent_id: super::super::AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 1000,
            kredit_cap: 1000,
            constraints: super::super::AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: super::super::EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        // Normal consumption
        assert!(consume_kredit(&mut agent, 500).is_ok());
        assert_eq!(agent.kredit_balance, 500);

        // Consume to negative
        assert!(consume_kredit(&mut agent, 600).is_ok());
        assert!(agent.kredit_balance < 0);
        assert_eq!(agent.status, AgentStatus::Throttled);
    }

    #[test]
    fn test_liability_trigger() {
        let mut agent = InstrumentalActor {
            agent_id: super::super::AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: -300, // 30% deficit
            kredit_cap: 1000,
            constraints: super::super::AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: super::super::EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        let mut sap_locked = 500;
        let mut sap_balance = 1000;
        let mut civ = 0.7;

        let result =
            process_negative_kredit(&mut agent, &mut sap_locked, &mut sap_balance, &mut civ);

        assert!(matches!(result, LiabilityResult::LiabilityTriggered { .. }));
        assert_eq!(agent.status, AgentStatus::Revoked);
        assert!(civ < 0.7);
    }

    #[test]
    fn test_trust_derived_kredit_cap() {
        // High trust K-Vector (verified agent)
        let high_trust = KVector::new(0.9, 0.8, 1.0, 0.9, 0.5, 0.7, 0.8, 0.6, 0.9, 0.85);
        let high_cap =
            calculate_kredit_cap_from_trust(&high_trust, AgentClass::Supervised, 0.8).unwrap();

        // Low trust K-Vector (unverified agent)
        let low_trust = KVector::new(0.3, 0.2, 0.5, 0.3, 0.1, 0.1, 0.2, 0.1, 0.0, 0.2);
        let low_cap =
            calculate_kredit_cap_from_trust(&low_trust, AgentClass::Supervised, 0.8).unwrap();

        // High trust should get much more KREDIT
        assert!(high_cap > low_cap * 3);
    }

    #[test]
    fn test_recalculate_kredit_cap() {
        let mut agent = InstrumentalActor {
            agent_id: super::super::AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 5000,
            constraints: super::super::AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new(0.9, 0.8, 1.0, 0.9, 0.5, 0.7, 0.8, 0.6, 0.85, 0.85),
            epistemic_stats: super::super::EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        };

        let new_cap = recalculate_agent_kredit_cap(&mut agent, 0.8).unwrap();

        // High trust should result in cap > 5000
        assert!(new_cap > 5000);
        assert_eq!(agent.kredit_cap, new_cap);
    }
}
