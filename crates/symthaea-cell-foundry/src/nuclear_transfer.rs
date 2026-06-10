// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Somatic Cell Nuclear Transfer (SCNT) protocol state machine.
//!
//! Models the multi-stage SCNT process: enucleation, nuclear injection,
//! activation, and early embryonic development to blastocyst stage.

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::types::{CellState, CellType, ScntStage};

/// SCNT protocol state machine.
///
/// Each stage has an associated success probability. The protocol advances
/// stochastically: each `advance()` call rolls the dice for the current stage
/// and either progresses or fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScntProtocol {
    /// Current SCNT stage (enucleation → blastocyst).
    pub stage: ScntStage,
    /// Somatic donor cell whose nucleus will be transferred.
    pub donor_cell: CellState,
    /// Enucleated oocyte that will receive the donor nucleus.
    pub recipient_oocyte: CellState,
    /// Base probability of success at each stage (derived from donor/oocyte viability).
    pub success_probability: f64,
    /// Number of advance() calls made so far.
    pub attempts: u32,
}

impl ScntProtocol {
    /// Create a new SCNT protocol with donor somatic cell and recipient oocyte.
    pub fn new(donor: CellState, oocyte: CellState) -> Self {
        // Base success probability depends on donor viability and oocyte quality
        let base_prob = donor.viability * oocyte.viability * 0.5;
        Self {
            stage: ScntStage::Enucleation,
            donor_cell: donor,
            recipient_oocyte: oocyte,
            success_probability: base_prob.clamp(0.0, 1.0),
            attempts: 0,
        }
    }

    /// Advance the protocol by one stage.
    ///
    /// Uses the RNG to determine success/failure at each stage. The probability
    /// of success varies by stage: enucleation is relatively easy, while
    /// blastocyst formation is the hardest step.
    pub fn advance(&mut self, rng: &mut impl Rng) -> ScntStage {
        if self.stage == ScntStage::Complete || self.stage == ScntStage::Failed {
            return self.stage;
        }

        self.attempts += 1;

        // Stage-specific success multipliers (calibrated to real SCNT success rates)
        let stage_multiplier = match self.stage {
            ScntStage::Enucleation => 0.95,      // Technical step, high success
            ScntStage::NuclearInjection => 0.85, // Moderate difficulty
            ScntStage::Activation => 0.70,       // Needs proper reprogramming signals
            ScntStage::FirstDivision => 0.60,    // Nuclear-cytoplasmic compatibility
            ScntStage::BlastocystFormation => 0.40, // Most critical step
            ScntStage::Complete | ScntStage::Failed => 1.0,
        };

        let roll: f64 = rng.r#gen();
        let threshold = self.success_probability * stage_multiplier;

        if roll < threshold {
            // Success: advance to next stage
            self.stage = match self.stage {
                ScntStage::Enucleation => ScntStage::NuclearInjection,
                ScntStage::NuclearInjection => ScntStage::Activation,
                ScntStage::Activation => ScntStage::FirstDivision,
                ScntStage::FirstDivision => ScntStage::BlastocystFormation,
                ScntStage::BlastocystFormation => {
                    // Update cell state on completion
                    self.donor_cell.cell_type = CellType::Embryonic;
                    self.donor_cell.pluripotency_score = 0.9;
                    ScntStage::Complete
                }
                ScntStage::Complete | ScntStage::Failed => self.stage,
            };
        } else {
            self.stage = ScntStage::Failed;
        }

        self.stage
    }

    /// Whether the protocol has completed successfully.
    pub fn is_complete(&self) -> bool {
        self.stage == ScntStage::Complete
    }

    /// Whether the protocol has failed.
    pub fn is_failed(&self) -> bool {
        self.stage == ScntStage::Failed
    }

    /// Run the full protocol to completion (or failure).
    pub fn run_to_completion(&mut self, rng: &mut impl Rng) -> ScntStage {
        while self.stage != ScntStage::Complete && self.stage != ScntStage::Failed {
            self.advance(rng);
        }
        self.stage
    }

    /// Compute the cumulative probability of completing all remaining stages.
    pub fn remaining_success_probability(&self) -> f64 {
        if self.stage == ScntStage::Complete {
            return 1.0;
        }
        if self.stage == ScntStage::Failed {
            return 0.0;
        }

        let stages_remaining: Vec<f64> = match self.stage {
            ScntStage::Enucleation => vec![0.95, 0.85, 0.70, 0.60, 0.40],
            ScntStage::NuclearInjection => vec![0.85, 0.70, 0.60, 0.40],
            ScntStage::Activation => vec![0.70, 0.60, 0.40],
            ScntStage::FirstDivision => vec![0.60, 0.40],
            ScntStage::BlastocystFormation => vec![0.40],
            ScntStage::Complete | ScntStage::Failed => vec![],
        };

        let base = self.success_probability;
        stages_remaining
            .iter()
            .fold(1.0, |acc, &mult| acc * base * mult)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    type TestRng = rand::rngs::StdRng;

    fn make_rng(seed: u64) -> TestRng {
        TestRng::seed_from_u64(seed)
    }

    #[test]
    fn test_initial_stage() {
        let donor = CellState::new_somatic();
        let oocyte = CellState::new_oocyte();
        let protocol = ScntProtocol::new(donor, oocyte);
        assert_eq!(protocol.stage, ScntStage::Enucleation);
    }

    #[test]
    fn test_success_probability_bounded() {
        let donor = CellState::new_somatic();
        let oocyte = CellState::new_oocyte();
        let protocol = ScntProtocol::new(donor, oocyte);
        assert!(protocol.success_probability >= 0.0 && protocol.success_probability <= 1.0);
    }

    #[test]
    fn test_run_to_completion_terminates() {
        let donor = CellState::new_somatic();
        let oocyte = CellState::new_oocyte();
        let mut protocol = ScntProtocol::new(donor, oocyte);
        let mut rng = make_rng(42);

        let final_stage = protocol.run_to_completion(&mut rng);
        assert!(
            final_stage == ScntStage::Complete || final_stage == ScntStage::Failed,
            "Protocol should terminate in Complete or Failed"
        );
    }

    #[test]
    fn test_advance_increments_attempts() {
        let donor = CellState::new_somatic();
        let oocyte = CellState::new_oocyte();
        let mut protocol = ScntProtocol::new(donor, oocyte);
        let mut rng = make_rng(42);

        assert_eq!(protocol.attempts, 0);
        protocol.advance(&mut rng);
        assert_eq!(protocol.attempts, 1);
    }

    #[test]
    fn test_completed_does_not_advance() {
        let donor = CellState::new_somatic();
        let oocyte = CellState::new_oocyte();
        let mut protocol = ScntProtocol::new(donor, oocyte);
        let mut rng = make_rng(42);

        protocol.run_to_completion(&mut rng);
        let final_stage = protocol.stage;
        let attempts_before = protocol.attempts;

        protocol.advance(&mut rng);
        assert_eq!(protocol.stage, final_stage);
        assert_eq!(protocol.attempts, attempts_before);
    }

    #[test]
    fn test_remaining_probability_decreases_with_stages() {
        let donor = CellState::new_somatic();
        let oocyte = CellState::new_oocyte();
        let protocol = ScntProtocol::new(donor, oocyte);
        let remaining = protocol.remaining_success_probability();
        assert!(remaining >= 0.0 && remaining <= 1.0);
        // Cumulative probability through 5 stages should be quite low
        assert!(
            remaining < 0.5,
            "5-stage cumulative should be low: {}",
            remaining
        );
    }

    #[test]
    fn test_statistical_success_rate() {
        // Run many trials and check success rate is reasonable
        let mut successes = 0;
        let n = 200;
        for i in 0..n {
            let donor = CellState::new_somatic();
            let oocyte = CellState::new_oocyte();
            let mut protocol = ScntProtocol::new(donor, oocyte);
            let mut rng = make_rng(i);
            protocol.run_to_completion(&mut rng);
            if protocol.is_complete() {
                successes += 1;
            }
        }
        // Should have some successes and some failures
        let rate = successes as f64 / n as f64;
        assert!(
            rate > 0.0 && rate < 1.0,
            "Success rate should be between 0 and 1, got {}",
            rate
        );
    }
}
