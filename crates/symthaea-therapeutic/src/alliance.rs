// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Therapeutic alliance tracking (Bordin 1979).
//!
//! Implements Bordin's three-component working alliance model:
//! - **Bond**: Emotional connection and trust
//! - **Goal Agreement**: Shared understanding of therapeutic goals
//! - **Task Agreement**: Agreement on methods/techniques to use
//!
//! Also tracks alliance ruptures (Safran & Muran 2000) — microprocess
//! markers of withdrawal and confrontation — with repair mechanisms.
//!
//! Science: Bordin (1979), Horvath & Symonds (1991) meta-analysis (r=0.26),
//! Safran & Muran (2000), Eubanks et al. (2018) rupture-repair markers.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Maximum rupture/repair history entries.
const MAX_HISTORY: usize = 64;

// ── Rupture Types ──────────────────────────────────────────────────────────

/// Types of alliance ruptures (Safran & Muran 2000).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuptureType {
    /// Client disengages, avoids, or becomes compliant (low agency).
    /// Markers: minimal responses, topic avoidance, compliance without engagement.
    Withdrawal,
    /// Client expresses dissatisfaction or hostility toward the process.
    /// Markers: criticism of methods, impatience, direct complaints.
    Confrontation,
}

impl RuptureType {
    /// Static string for telemetry (avoids `format!("{:?}")`).
    pub fn as_str(&self) -> &'static str {
        match self {
            RuptureType::Withdrawal => "Withdrawal",
            RuptureType::Confrontation => "Confrontation",
        }
    }
}

/// Repair strategies for alliance ruptures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RepairStrategy {
    /// Acknowledge the rupture explicitly
    Acknowledge,
    /// Explore the client's experience of the rupture
    Explore,
    /// Adjust therapeutic approach based on feedback
    Adjust,
    /// Validate the client's feelings about the process
    Validate,
    /// Renegotiate goals or tasks
    Renegotiate,
}

/// Record of a rupture event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuptureRecord {
    pub rupture_type: RuptureType,
    pub cycle: u64,
    pub alliance_at_rupture: f32,
    pub repaired: bool,
    pub repair_strategy: Option<RepairStrategy>,
}

// ── Therapeutic Alliance ───────────────────────────────────────────────────

/// Bordin's (1979) three-component working alliance model.
///
/// Each component ranges from 0.0 (absent) to 1.0 (strong).
/// The composite score is the harmonic mean of all three (weakest link matters).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TherapeuticAlliance {
    /// Emotional bond between client and therapist (trust, warmth, liking).
    pub bond: f32,
    /// Agreement on therapeutic goals.
    pub goal_agreement: f32,
    /// Agreement on therapeutic tasks/methods.
    pub task_agreement: f32,

    /// Number of detected ruptures.
    pub rupture_count: u32,
    /// Number of successful repairs.
    pub repair_count: u32,

    /// Rupture/repair history.
    pub history: VecDeque<RuptureRecord>,

    /// Exponential moving average of composite alliance.
    alliance_ema: f32,
}

impl TherapeuticAlliance {
    /// Create a new alliance with initial moderate levels.
    ///
    /// Starting at 0.3 reflects early therapeutic relationship before trust is built.
    pub fn new() -> Self {
        Self {
            bond: 0.3,
            goal_agreement: 0.3,
            task_agreement: 0.3,
            rupture_count: 0,
            repair_count: 0,
            history: VecDeque::new(),
            alliance_ema: 0.3,
        }
    }

    /// Composite alliance score (harmonic mean of three components).
    ///
    /// Harmonic mean ensures the weakest component dominates — a strong
    /// bond can't compensate for goal disagreement.
    pub fn composite(&self) -> f32 {
        let a = self.bond;
        let b = self.goal_agreement;
        let c = self.task_agreement;
        if a <= 0.0 || b <= 0.0 || c <= 0.0 {
            return 0.0;
        }
        3.0 / (1.0 / a + 1.0 / b + 1.0 / c)
    }

    /// EMA-smoothed composite (less volatile than raw composite).
    pub fn smoothed_composite(&self) -> f32 {
        self.alliance_ema
    }

    /// Detect a rupture from client affect indicators.
    ///
    /// Withdrawal: low arousal + negative valence (disengagement).
    /// Confrontation: high arousal + negative valence (frustration).
    pub fn detect_rupture(&self, valence: f32, arousal: f32) -> Option<RuptureType> {
        if valence > -0.3 {
            return None; // positive or neutral affect — no rupture
        }
        if arousal < 0.3 {
            Some(RuptureType::Withdrawal)
        } else if arousal > 0.7 {
            Some(RuptureType::Confrontation)
        } else {
            None
        }
    }

    /// Register a detected rupture.
    pub fn register_rupture(&mut self, rupture_type: RuptureType, cycle: u64) {
        self.rupture_count += 1;
        let record = RuptureRecord {
            rupture_type,
            cycle,
            alliance_at_rupture: self.composite(),
            repaired: false,
            repair_strategy: None,
        };
        self.history.push_back(record);
        if self.history.len() > MAX_HISTORY {
            self.history.pop_front();
        }

        // Rupture damages the alliance
        match rupture_type {
            RuptureType::Withdrawal => {
                self.bond -= 0.05;
                self.task_agreement -= 0.08;
            }
            RuptureType::Confrontation => {
                self.bond -= 0.08;
                self.goal_agreement -= 0.05;
            }
        }
        self.clamp_all();
        self.update_ema();
    }

    /// Attempt to repair the most recent rupture.
    ///
    /// Returns the new composite alliance score. Successful repair can
    /// *strengthen* the alliance beyond pre-rupture levels (Safran & Muran 2000).
    pub fn attempt_repair(&mut self, strategy: RepairStrategy) -> f32 {
        // Find most recent unrepaired rupture
        if let Some(record) = self.history.iter_mut().rev().find(|r| !r.repaired) {
            record.repaired = true;
            record.repair_strategy = Some(strategy);
            self.repair_count += 1;

            // Repair boost (can exceed pre-rupture level)
            let boost = match strategy {
                RepairStrategy::Acknowledge => 0.05,
                RepairStrategy::Explore => 0.08,
                RepairStrategy::Adjust => 0.06,
                RepairStrategy::Validate => 0.07,
                RepairStrategy::Renegotiate => 0.09,
            };

            match record.rupture_type {
                RuptureType::Withdrawal => {
                    self.bond += boost;
                    self.task_agreement += boost * 1.2;
                }
                RuptureType::Confrontation => {
                    self.bond += boost * 1.2;
                    self.goal_agreement += boost;
                }
            }
        }
        self.clamp_all();
        self.update_ema();
        self.composite()
    }

    /// Naturally grow the alliance (called each therapeutic interaction).
    ///
    /// Small increments reflect trust building over time.
    pub fn grow(&mut self, bond_delta: f32, goal_delta: f32, task_delta: f32) {
        self.bond = (self.bond + bond_delta).clamp(0.0, 1.0);
        self.goal_agreement = (self.goal_agreement + goal_delta).clamp(0.0, 1.0);
        self.task_agreement = (self.task_agreement + task_delta).clamp(0.0, 1.0);
        self.update_ema();
    }

    /// Repair success rate.
    pub fn repair_rate(&self) -> f32 {
        if self.rupture_count == 0 {
            return 1.0;
        }
        self.repair_count as f32 / self.rupture_count as f32
    }

    /// Count of withdrawal ruptures.
    pub fn withdrawal_count(&self) -> u32 {
        self.history
            .iter()
            .filter(|r| r.rupture_type == RuptureType::Withdrawal)
            .count() as u32
    }

    /// Count of confrontation ruptures.
    pub fn confrontation_count(&self) -> u32 {
        self.history
            .iter()
            .filter(|r| r.rupture_type == RuptureType::Confrontation)
            .count() as u32
    }

    /// Last rupture type (if any).
    pub fn last_rupture_type(&self) -> Option<RuptureType> {
        self.history.back().map(|r| r.rupture_type)
    }

    /// Whether the most recent rupture was repaired.
    pub fn last_rupture_repaired(&self) -> bool {
        self.history.back().map_or(true, |r| r.repaired)
    }

    /// Alliance trajectory: recent composite values for trend analysis.
    /// Returns up to `n` most recent EMA-smoothed values (1 per grow/rupture/repair call).
    pub fn alliance_trend_direction(&self) -> f32 {
        // Compare current EMA to initial (0.3).
        // Positive = growing, negative = declining.
        self.alliance_ema - 0.3
    }

    fn clamp_all(&mut self) {
        self.bond = self.bond.clamp(0.0, 1.0);
        self.goal_agreement = self.goal_agreement.clamp(0.0, 1.0);
        self.task_agreement = self.task_agreement.clamp(0.0, 1.0);
    }

    fn update_ema(&mut self) {
        let alpha = 0.15;
        self.alliance_ema = alpha * self.composite() + (1.0 - alpha) * self.alliance_ema;
    }
}

impl Default for TherapeuticAlliance {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_alliance_moderate() {
        let alliance = TherapeuticAlliance::new();
        assert!((alliance.composite() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_composite_harmonic_mean() {
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 1.0;
        alliance.goal_agreement = 1.0;
        alliance.task_agreement = 1.0;
        assert!((alliance.composite() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_composite_weakest_link() {
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.9;
        alliance.goal_agreement = 0.9;
        alliance.task_agreement = 0.1; // weak task agreement
        let composite = alliance.composite();
        // Harmonic mean should be much lower than arithmetic mean
        assert!(
            composite < 0.5,
            "composite {:.3} should be < 0.5",
            composite
        );
    }

    #[test]
    fn test_composite_zero_component() {
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.0;
        assert_eq!(alliance.composite(), 0.0);
    }

    #[test]
    fn test_detect_rupture_withdrawal() {
        let alliance = TherapeuticAlliance::new();
        let rupture = alliance.detect_rupture(-0.5, 0.2);
        assert_eq!(rupture, Some(RuptureType::Withdrawal));
    }

    #[test]
    fn test_detect_rupture_confrontation() {
        let alliance = TherapeuticAlliance::new();
        let rupture = alliance.detect_rupture(-0.6, 0.8);
        assert_eq!(rupture, Some(RuptureType::Confrontation));
    }

    #[test]
    fn test_no_rupture_positive_affect() {
        let alliance = TherapeuticAlliance::new();
        assert_eq!(alliance.detect_rupture(0.5, 0.8), None);
    }

    #[test]
    fn test_rupture_damages_alliance() {
        let mut alliance = TherapeuticAlliance::new();
        let pre = alliance.composite();
        alliance.register_rupture(RuptureType::Withdrawal, 100);
        assert!(alliance.composite() < pre);
        assert_eq!(alliance.rupture_count, 1);
    }

    #[test]
    fn test_repair_restores_alliance() {
        let mut alliance = TherapeuticAlliance::new();
        alliance.bond = 0.6;
        alliance.goal_agreement = 0.6;
        alliance.task_agreement = 0.6;
        alliance.register_rupture(RuptureType::Confrontation, 100);
        let post_rupture = alliance.composite();
        alliance.attempt_repair(RepairStrategy::Explore);
        assert!(alliance.composite() > post_rupture);
        assert_eq!(alliance.repair_count, 1);
    }

    #[test]
    fn test_repair_rate() {
        let mut alliance = TherapeuticAlliance::new();
        alliance.register_rupture(RuptureType::Withdrawal, 1);
        alliance.register_rupture(RuptureType::Confrontation, 2);
        alliance.attempt_repair(RepairStrategy::Validate);
        assert!((alliance.repair_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_repair_rate_no_ruptures() {
        let alliance = TherapeuticAlliance::new();
        assert_eq!(alliance.repair_rate(), 1.0);
    }

    #[test]
    fn test_alliance_grow() {
        let mut alliance = TherapeuticAlliance::new();
        let pre = alliance.composite();
        alliance.grow(0.05, 0.05, 0.05);
        assert!(alliance.composite() > pre);
    }

    #[test]
    fn test_alliance_history_bounded() {
        let mut alliance = TherapeuticAlliance::new();
        for i in 0..100 {
            alliance.register_rupture(RuptureType::Withdrawal, i);
        }
        assert!(alliance.history.len() <= MAX_HISTORY);
    }

    #[test]
    fn test_smoothed_composite() {
        let mut alliance = TherapeuticAlliance::new();
        alliance.grow(0.3, 0.3, 0.3);
        // EMA should lag behind raw composite
        assert!(alliance.smoothed_composite() < alliance.composite());
    }
}
