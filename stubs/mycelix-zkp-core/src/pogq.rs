// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proof-of-Gradient-Quality (PoGQ) v4.1 -- shared simulation logic.
//!
//! This module provides the PoGQ state machine used by both RISC0 and
//! Winterfell backends to compute expected quarantine decisions for
//! gradient quality in federated learning.
//!
//! ## PoGQ State Machine
//!
//! 8 registers tracked across execution (Q16.16 fixed-point):
//! - `ema`: Exponential moving average of quality scores
//! - `consec_viol`: Consecutive violation counter
//! - `consec_clear`: Consecutive clear counter
//! - `quarantined`: Quarantine status (0/1)
//! - `x_t`: Current hybrid quality score (witness)
//! - `threshold`: Conformal threshold
//! - `beta`: EMA smoothing factor (default 0.85)
//! - `round`: Current round number

use serde::{Deserialize, Serialize};

use crate::fixed_point::FixedPoint;

/// PoGQ public parameters -- shared between both backends.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoGQPublicInputs {
    /// EMA smoothing factor (default 0.85 = 55705 in Q16.16).
    pub beta: FixedPoint,
    /// Conformal quality threshold.
    pub threshold: FixedPoint,
    /// Warm-up rounds (no quarantine during warm-up).
    pub warmup_rounds: u64,
    /// Consecutive violations to enter quarantine.
    pub k_violations: u64,
    /// Consecutive clears to release from quarantine.
    pub m_clears: u64,
    /// Initial EMA value.
    pub ema_init: FixedPoint,
    /// Initial violation counter.
    pub viol_init: u64,
    /// Initial clear counter.
    pub clear_init: u64,
    /// Initial quarantine status (0 = not quarantined).
    pub quar_init: u64,
    /// Initial round number.
    pub round_init: u64,
}

impl Default for PoGQPublicInputs {
    fn default() -> Self {
        Self {
            beta: FixedPoint::from_f32(0.85),
            threshold: FixedPoint::ONE, // 1.0 = always pass
            warmup_rounds: 0,
            k_violations: 2,
            m_clears: 1,
            ema_init: FixedPoint::from_f32(0.75),
            viol_init: 0,
            clear_init: 0,
            quar_init: 0,
            round_init: 0,
        }
    }
}

/// PoGQ witness -- private quality scores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoGQWitness {
    /// Hybrid quality scores (one per round).
    pub scores: Vec<FixedPoint>,
}

/// PoGQ state at a single timestep.
#[derive(Clone, Debug, Default)]
pub struct PoGQState {
    pub ema: FixedPoint,
    pub consec_viol: u64,
    pub consec_clear: u64,
    pub quarantined: bool,
    pub round: u64,
}

/// Result of PoGQ simulation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoGQResult {
    /// Final quarantine decision (true = quarantined).
    pub quarantined: bool,
    /// Final EMA value.
    pub final_ema: FixedPoint,
    /// Final consecutive violation count.
    pub final_viol: u64,
    /// Final consecutive clear count.
    pub final_clear: u64,
    /// Number of rounds processed.
    pub rounds_processed: u64,
    /// Per-round trace (for debugging / proof generation).
    pub trace: Vec<PoGQTraceRow>,
}

/// One row of the PoGQ execution trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoGQTraceRow {
    pub round: u64,
    pub score: FixedPoint,
    pub ema: FixedPoint,
    pub violation: bool,
    pub in_warmup: bool,
    pub consec_viol: u64,
    pub consec_clear: u64,
    pub quarantined: bool,
}

/// Simulate the PoGQ state machine.
///
/// This is the reference implementation used to compute expected outputs
/// for both RISC0 guest programs and Winterfell AIR circuits. Both
/// backends must produce identical results for the same inputs.
pub fn simulate_pogq(inputs: &PoGQPublicInputs, witness: &PoGQWitness) -> PoGQResult {
    let one_minus_beta = FixedPoint::ONE - inputs.beta;

    let mut state = PoGQState {
        ema: inputs.ema_init,
        consec_viol: inputs.viol_init,
        consec_clear: inputs.clear_init,
        quarantined: inputs.quar_init != 0,
        round: inputs.round_init,
    };

    let mut trace = Vec::with_capacity(witness.scores.len());

    for score in &witness.scores {
        state.round += 1;
        let in_warmup = state.round <= inputs.warmup_rounds;

        // 1. EMA update: ema = beta * ema + (1-beta) * x
        state.ema = inputs.beta.mul(state.ema) + one_minus_beta.mul(*score);

        // 2. Violation detection (strict less-than)
        let violation = score.raw() < inputs.threshold.raw();

        // 3-4. Counter updates
        if violation {
            state.consec_viol += 1;
            state.consec_clear = 0;
        } else {
            state.consec_clear += 1;
            state.consec_viol = 0;
        }

        // 5. Hysteresis state transitions (only outside warm-up)
        if !in_warmup {
            if !state.quarantined {
                if state.consec_viol >= inputs.k_violations {
                    state.quarantined = true;
                }
            } else if state.consec_clear >= inputs.m_clears {
                state.quarantined = false;
            }
        }

        trace.push(PoGQTraceRow {
            round: state.round,
            score: *score,
            ema: state.ema,
            violation,
            in_warmup,
            consec_viol: state.consec_viol,
            consec_clear: state.consec_clear,
            quarantined: state.quarantined,
        });
    }

    PoGQResult {
        quarantined: state.quarantined,
        final_ema: state.ema,
        final_viol: state.consec_viol,
        final_clear: state.consec_clear,
        rounds_processed: witness.scores.len() as u64,
        trace,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_pogq_all_pass() {
        let inputs = PoGQPublicInputs {
            threshold: FixedPoint::from_f32(0.5),
            ..Default::default()
        };
        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.9),
                FixedPoint::from_f32(0.8),
                FixedPoint::from_f32(0.95),
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(!result.quarantined);
        assert_eq!(result.rounds_processed, 3);
    }

    #[test]
    fn test_simulate_pogq_quarantines_on_violations() {
        let inputs = PoGQPublicInputs {
            threshold: FixedPoint::from_f32(0.9),
            k_violations: 2,
            ..Default::default()
        };
        let witness = PoGQWitness {
            scores: vec![FixedPoint::from_f32(0.1), FixedPoint::from_f32(0.1)],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(result.quarantined);
    }
}
