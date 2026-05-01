// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Three-Currency Economic System — MYCEL / SAP / TEND
//!
//! Full per-agent accounting for the canonical Mycelix three-currency economy
//! as defined in the Economic Charter (v1.1).
//!
//! ## MYCEL (Soulbound Reputation)
//!
//! Non-transferable governance weight. Formula:
//! ```text
//! MYCEL = Participation(40%) + PeerRecognition(20%) + Quality(20%) + Longevity(20%)
//! ```
//! Decay: 5%/year inactive. Floor: 0.05. Jubilee: 0.8× every 4 years.
//! Genesis cohort (first 1000 agents): start at 0.5 instead of 0.1.
//!
//! ## SAP (Demurrage Circulation)
//!
//! Transferable medium of exchange with 2%/year demurrage.
//! Demurrage flows: 70% local commons, 20% planetary, 10% system.
//! Constitutional reserve: 25% of total SAP in commons pools.
//!
//! ## TEND (Mutual Credit)
//!
//! Bounded zero-sum care economy. 1 TEND = 1 hour of service.
//! Bounds: ±40 per agent. Earned via teaching/care, spent on services.

use serde::{Deserialize, Serialize};

// ============================================================================
// Constants (from Economic Charter v1.1)
// ============================================================================

/// MYCEL passive decay: 5% per year (applied monthly: 1 - (1-0.05)^(1/12)).
pub const MYCEL_ANNUAL_DECAY: f64 = 0.05;

/// MYCEL monthly decay factor: (1 - 0.05)^(1/12) ≈ 0.99573.
pub const MYCEL_MONTHLY_FACTOR: f64 = 0.99573;

/// MYCEL decay floor: cannot fall below this (baseline humanity recognition).
pub const MYCEL_FLOOR: f64 = 0.05;

/// MYCEL growth cap per month.
pub const MYCEL_MONTHLY_CAP: f64 = 0.1;

/// MYCEL Jubilee compression factor: 0.8× every 4 years.
pub const MYCEL_JUBILEE_FACTOR: f64 = 0.8;

/// MYCEL Jubilee period in ticks (4 years × 12 months).
pub const MYCEL_JUBILEE_TICKS: u32 = 48;

/// MYCEL genesis cohort starting score (first 1000 agents).
pub const MYCEL_GENESIS_START: f64 = 0.5;

/// MYCEL default starting score (after genesis).
pub const MYCEL_DEFAULT_START: f64 = 0.1;

/// SAP annual demurrage rate: 2%.
pub const SAP_ANNUAL_DEMURRAGE: f64 = 0.02;

/// SAP monthly demurrage factor: (1 - 0.02)^(1/12) ≈ 0.99832.
pub const SAP_MONTHLY_FACTOR: f64 = 0.99832;

/// SAP demurrage flow: 70% local, 20% planetary, 10% system.
pub const SAP_FLOW_LOCAL: f64 = 0.70;
pub const SAP_FLOW_PLANETARY: f64 = 0.20;
pub const SAP_FLOW_SYSTEM: f64 = 0.10;

/// SAP constitutional reserve: 25% of total must remain in commons.
pub const SAP_COMMONS_RESERVE_FRACTION: f64 = 0.25;

/// SAP exempt floor: agents below this balance don't pay demurrage.
pub const SAP_EXEMPT_FLOOR: f64 = 200.0;

/// TEND balance bounds: ±40 per agent.
pub const TEND_LIMIT: f64 = 40.0;

/// TEND reward per teaching interaction.
pub const TEND_TEACHING_REWARD: f64 = 5.0;

/// TEND reward per care work interaction.
pub const TEND_CARE_REWARD: f64 = 3.0;

// ============================================================================
// MYCEL Component Weights
// ============================================================================

/// MYCEL composite weights (from Economic Charter).
pub const MYCEL_W_PARTICIPATION: f64 = 0.40;
pub const MYCEL_W_RECOGNITION: f64 = 0.20;
pub const MYCEL_W_QUALITY: f64 = 0.20;
pub const MYCEL_W_LONGEVITY: f64 = 0.20;

// ============================================================================
// Per-World Currency State
// ============================================================================

/// Aggregate currency state for a world.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldCurrencyState {
    /// SAP held in the world's local commons pool (exempt from demurrage).
    pub sap_commons_local: f64,
    /// SAP held in the planetary commons pool (shared with same-body worlds).
    pub sap_commons_planetary: f64,
    /// SAP in the system-wide interplanetary treasury.
    pub sap_commons_system: f64,
    /// Total SAP in circulation (agent balances + commons).
    pub sap_total_supply: f64,
    /// Total demurrage collected this tick.
    pub sap_demurrage_collected: f64,
    /// Mean MYCEL across all living agents.
    pub mycel_mean: f64,
    /// Total TEND in positive balances (care economy size).
    pub tend_total_positive: f64,
    /// TEND velocity: transactions / mean_absolute_balance (target: 4-6/year).
    pub tend_velocity: f64,
    /// TEND transactions this tick.
    pub tend_transactions_this_tick: u32,
    /// Next Jubilee tick.
    pub next_jubilee_tick: u32,
    /// Total agents ever created (for genesis cohort tracking).
    pub total_agents_created: u64,
}

impl Default for WorldCurrencyState {
    fn default() -> Self {
        Self {
            sap_commons_local: 0.0,
            sap_commons_planetary: 0.0,
            sap_commons_system: 0.0,
            sap_total_supply: 10_000.0, // initial SAP pool
            sap_demurrage_collected: 0.0,
            mycel_mean: 0.1,
            tend_total_positive: 0.0,
            tend_velocity: 0.0,
            tend_transactions_this_tick: 0,
            next_jubilee_tick: MYCEL_JUBILEE_TICKS,
            total_agents_created: 0,
        }
    }
}

// ============================================================================
// MYCEL Computation
// ============================================================================

/// MYCEL component inputs for a single agent.
#[derive(Debug, Clone, Copy)]
pub struct MycelInputs {
    /// Did the agent participate this tick? (voted, proposed, taught)
    pub participated: bool,
    /// Peer recognition received this tick [0.0, 1.0].
    pub recognition: f64,
    /// Quality score: skill accuracy × consciousness coherence [0.0, 1.0].
    pub quality: f64,
    /// Years active in the community.
    pub years_active: f64,
}

/// Compute the new MYCEL score for an agent.
///
/// Applies growth (capped at MYCEL_MONTHLY_CAP), passive decay for
/// inactive agents, and floor enforcement.
pub fn compute_mycel(current: f64, inputs: &MycelInputs) -> f64 {
    // Component scores
    let participation = if inputs.participated { 1.0 } else { 0.0 };
    let recognition = inputs.recognition.clamp(0.0, 1.0);
    let quality = inputs.quality.clamp(0.0, 1.0);
    let longevity = (inputs.years_active / 50.0).clamp(0.0, 1.0);

    // Target score from components
    let target = MYCEL_W_PARTICIPATION * participation
        + MYCEL_W_RECOGNITION * recognition
        + MYCEL_W_QUALITY * quality
        + MYCEL_W_LONGEVITY * longevity;

    // Growth: move toward target, capped at MYCEL_MONTHLY_CAP
    let delta = target - current;
    let growth = delta.clamp(-MYCEL_MONTHLY_CAP, MYCEL_MONTHLY_CAP);

    // Apply passive decay for inactive agents
    let decayed = if !inputs.participated {
        current * MYCEL_MONTHLY_FACTOR
    } else {
        current
    };

    // Apply growth
    let new_score = (decayed + growth * 0.1).clamp(MYCEL_FLOOR, 1.0);

    new_score
}

/// Apply Jubilee compression: 0.8× all MYCEL scores.
/// Called every MYCEL_JUBILEE_TICKS ticks.
pub fn apply_jubilee(scores: &mut [f64]) {
    for score in scores.iter_mut() {
        *score = (*score * MYCEL_JUBILEE_FACTOR).max(MYCEL_FLOOR);
    }
}

// ============================================================================
// SAP Demurrage
// ============================================================================

/// Apply monthly SAP demurrage to agent balances.
///
/// Returns the total demurrage collected for distribution to commons.
/// Agents below SAP_EXEMPT_FLOOR are exempt.
pub fn apply_sap_demurrage(balances: &mut [f64]) -> f64 {
    let mut collected = 0.0f64;
    for balance in balances.iter_mut() {
        if *balance > SAP_EXEMPT_FLOOR {
            let taxable = *balance - SAP_EXEMPT_FLOOR;
            let demurrage = taxable * (1.0 - SAP_MONTHLY_FACTOR);
            *balance -= demurrage;
            collected += demurrage;
        }
    }
    collected
}

/// Distribute collected demurrage to commons pools.
///
/// Returns (local, planetary, system) amounts.
pub fn distribute_demurrage(collected: f64) -> (f64, f64, f64) {
    (
        collected * SAP_FLOW_LOCAL,
        collected * SAP_FLOW_PLANETARY,
        collected * SAP_FLOW_SYSTEM,
    )
}

/// Commons spending allocation for public goods.
#[derive(Debug, Clone, Copy)]
pub struct CommonsSpending {
    /// Infrastructure investment (40% of spend).
    pub infrastructure: f64,
    /// Education funding (30% of spend).
    pub education: f64,
    /// Medicine/health funding (30% of spend).
    pub medicine: f64,
}

/// Spend excess commons pool on public goods, respecting 25% reserve.
///
/// `rate` controls how much of excess is spent per tick (0.1 = 10%).
/// Returns the spending allocation. Deducts from commons_local.
pub fn spend_commons(state: &mut WorldCurrencyState, rate: f64) -> CommonsSpending {
    let reserve_min = state.sap_total_supply * SAP_COMMONS_RESERVE_FRACTION;
    let total_commons =
        state.sap_commons_local + state.sap_commons_planetary + state.sap_commons_system;
    let excess = (total_commons - reserve_min).max(0.0);
    let spend = excess * rate.clamp(0.0, 0.5);

    // Spend from local pool first
    let actual_spend = spend.min(state.sap_commons_local);
    state.sap_commons_local -= actual_spend;

    CommonsSpending {
        infrastructure: actual_spend * 0.4,
        education: actual_spend * 0.3,
        medicine: actual_spend * 0.3,
    }
}

/// Check if the constitutional SAP commons reserve is maintained.
///
/// Returns true if commons pools hold ≥ 25% of total supply.
pub fn check_commons_reserve(state: &WorldCurrencyState) -> bool {
    if state.sap_total_supply <= 0.0 {
        return true; // no supply = trivially satisfied
    }
    let commons_total =
        state.sap_commons_local + state.sap_commons_planetary + state.sap_commons_system;
    commons_total / state.sap_total_supply >= SAP_COMMONS_RESERVE_FRACTION
}

// ============================================================================
// TEND Mutual Credit
// ============================================================================

/// Execute a TEND transaction between two agents.
///
/// Returns true if the transaction succeeded (within bounds).
/// TEND is zero-sum: sender loses, receiver gains.
pub fn tend_transfer(sender_balance: &mut f64, receiver_balance: &mut f64, amount: f64) -> bool {
    if amount <= 0.0 {
        return false;
    }

    let new_sender = *sender_balance - amount;
    let new_receiver = *receiver_balance + amount;

    // Check bounds
    if new_sender < -TEND_LIMIT || new_receiver > TEND_LIMIT {
        return false;
    }

    *sender_balance = new_sender;
    *receiver_balance = new_receiver;
    true
}

/// Compute TEND velocity: transactions per mean absolute balance.
/// Target: 4-6 turns/year (from WIR/Sardex empirics).
pub fn compute_tend_velocity(transactions: u32, balances: &[f64]) -> f64 {
    if balances.is_empty() {
        return 0.0;
    }
    let mean_abs: f64 = balances.iter().map(|b| b.abs()).sum::<f64>() / balances.len() as f64;
    if mean_abs < 0.01 {
        return 0.0;
    }
    transactions as f64 / mean_abs
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MYCEL ----

    #[test]
    fn mycel_starts_at_default() {
        assert_eq!(MYCEL_DEFAULT_START, 0.1);
        assert_eq!(MYCEL_GENESIS_START, 0.5);
    }

    #[test]
    fn mycel_grows_with_participation() {
        let score = compute_mycel(
            0.1,
            &MycelInputs {
                participated: true,
                recognition: 0.5,
                quality: 0.7,
                years_active: 5.0,
            },
        );
        assert!(
            score > 0.1,
            "Active participation should grow MYCEL: {}",
            score
        );
    }

    #[test]
    fn mycel_decays_without_participation() {
        let score = compute_mycel(
            0.5,
            &MycelInputs {
                participated: false,
                recognition: 0.0,
                quality: 0.0,
                years_active: 10.0,
            },
        );
        assert!(score < 0.5, "Inactive agent should decay: {}", score);
    }

    #[test]
    fn mycel_never_below_floor() {
        let score = compute_mycel(
            0.06,
            &MycelInputs {
                participated: false,
                recognition: 0.0,
                quality: 0.0,
                years_active: 0.0,
            },
        );
        assert!(
            score >= MYCEL_FLOOR,
            "MYCEL should never go below floor: {}",
            score
        );
    }

    #[test]
    fn mycel_never_above_one() {
        let score = compute_mycel(
            0.95,
            &MycelInputs {
                participated: true,
                recognition: 1.0,
                quality: 1.0,
                years_active: 50.0,
            },
        );
        assert!(score <= 1.0, "MYCEL should never exceed 1.0: {}", score);
    }

    #[test]
    fn jubilee_compresses_scores() {
        let mut scores = vec![0.8, 0.6, 0.4, 0.1, 0.05];
        apply_jubilee(&mut scores);
        assert!((scores[0] - 0.64).abs() < 1e-6, "0.8 × 0.8 = 0.64");
        assert!((scores[1] - 0.48).abs() < 1e-6, "0.6 × 0.8 = 0.48");
        assert!(scores[4] >= MYCEL_FLOOR, "Jubilee respects floor");
    }

    #[test]
    fn mycel_longevity_component_saturates() {
        // Test the longevity component directly
        let longevity_5yr = (5.0f64 / 50.0).clamp(0.0, 1.0);
        let longevity_50yr = (50.0f64 / 50.0).clamp(0.0, 1.0);
        let longevity_100yr = (100.0f64 / 50.0).clamp(0.0, 1.0);
        assert!(
            longevity_50yr > longevity_5yr,
            "50yr should score higher than 5yr"
        );
        assert_eq!(
            longevity_50yr, longevity_100yr,
            "Longevity saturates at 50yr"
        );
        assert_eq!(longevity_50yr, 1.0, "50yr gives max longevity");
    }

    // ---- SAP ----

    #[test]
    fn sap_demurrage_exempts_floor() {
        let mut balances = vec![100.0, 300.0, 500.0];
        let collected = apply_sap_demurrage(&mut balances);
        assert_eq!(balances[0], 100.0, "Below floor should be exempt");
        assert!(balances[1] < 300.0, "Above floor should pay demurrage");
        assert!(collected > 0.0, "Should collect some demurrage");
    }

    #[test]
    fn sap_demurrage_distribution_sums_to_collected() {
        let collected = 100.0;
        let (local, planetary, system) = distribute_demurrage(collected);
        assert!((local + planetary + system - collected).abs() < 1e-6);
        assert!((local - 70.0).abs() < 1e-6, "70% local");
        assert!((planetary - 20.0).abs() < 1e-6, "20% planetary");
        assert!((system - 10.0).abs() < 1e-6, "10% system");
    }

    #[test]
    fn sap_commons_reserve_check() {
        let mut state = WorldCurrencyState::default();
        state.sap_total_supply = 1000.0;
        state.sap_commons_local = 200.0;
        state.sap_commons_planetary = 50.0;
        state.sap_commons_system = 10.0;
        // 260/1000 = 26% >= 25% → passes
        assert!(check_commons_reserve(&state));

        state.sap_commons_local = 100.0;
        // 160/1000 = 16% < 25% → fails
        assert!(!check_commons_reserve(&state));
    }

    // ---- TEND ----

    #[test]
    fn tend_transfer_succeeds_within_bounds() {
        let mut sender = 10.0f64;
        let mut receiver = 5.0f64;
        assert!(tend_transfer(&mut sender, &mut receiver, 5.0));
        assert_eq!(sender, 5.0);
        assert_eq!(receiver, 10.0);
    }

    #[test]
    fn tend_transfer_rejected_exceeds_negative_bound() {
        let mut sender = -35.0f64;
        let mut receiver = 0.0f64;
        // Would push sender to -45 which exceeds -40 limit
        assert!(!tend_transfer(&mut sender, &mut receiver, 10.0));
        assert_eq!(sender, -35.0, "Balance unchanged on rejection");
    }

    #[test]
    fn tend_transfer_rejected_exceeds_positive_bound() {
        let mut sender = 0.0f64;
        let mut receiver = 35.0f64;
        // Would push receiver to 45 which exceeds +40 limit
        assert!(!tend_transfer(&mut sender, &mut receiver, 10.0));
        assert_eq!(receiver, 35.0, "Balance unchanged on rejection");
    }

    #[test]
    fn tend_zero_sum_property() {
        let mut a = 20.0f64;
        let mut b = -10.0f64;
        let sum_before = a + b;
        tend_transfer(&mut a, &mut b, 15.0);
        let sum_after = a + b;
        assert!(
            (sum_before - sum_after).abs() < 1e-10,
            "TEND must be zero-sum"
        );
    }

    #[test]
    fn tend_velocity_computation() {
        let balances = vec![10.0, -5.0, 20.0, -15.0, 0.0];
        let velocity = compute_tend_velocity(50, &balances);
        // mean_abs = (10+5+20+15+0)/5 = 10
        // velocity = 50/10 = 5.0 (within healthy 4-6 range)
        assert!((velocity - 5.0).abs() < 1e-6);
    }

    #[test]
    fn tend_velocity_zero_when_no_balances() {
        assert_eq!(compute_tend_velocity(10, &[]), 0.0);
        assert_eq!(compute_tend_velocity(10, &[0.0, 0.0, 0.0]), 0.0);
    }

    // ---- Constants consistency ----

    #[test]
    fn mycel_weights_sum_to_one() {
        let sum = MYCEL_W_PARTICIPATION + MYCEL_W_RECOGNITION + MYCEL_W_QUALITY + MYCEL_W_LONGEVITY;
        assert!((sum - 1.0).abs() < 1e-10, "MYCEL weights must sum to 1.0");
    }

    #[test]
    fn sap_flow_fractions_sum_to_one() {
        let sum = SAP_FLOW_LOCAL + SAP_FLOW_PLANETARY + SAP_FLOW_SYSTEM;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "SAP flow fractions must sum to 1.0"
        );
    }

    #[test]
    fn monthly_factors_are_correct() {
        // MYCEL: (1 - 0.05)^(1/12) ≈ 0.99573
        let expected_mycel = (1.0 - MYCEL_ANNUAL_DECAY).powf(1.0 / 12.0);
        assert!((MYCEL_MONTHLY_FACTOR - expected_mycel).abs() < 0.001);

        // SAP: (1 - 0.02)^(1/12) ≈ 0.99832
        let expected_sap = (1.0 - SAP_ANNUAL_DEMURRAGE).powf(1.0 / 12.0);
        assert!((SAP_MONTHLY_FACTOR - expected_sap).abs() < 0.001);
    }
}
