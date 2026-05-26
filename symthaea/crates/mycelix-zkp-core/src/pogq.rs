// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proof-of-Gradient-Quality (PoGQ) v4.1 — shared simulation logic.
//!
//! Ported from `mycelix-core/0TML/tests/integration/zkbackend_abstraction.py` (895 lines).
//! This module provides the PoGQ state machine used by both RISC0 and Winterfell backends
//! to compute expected quarantine decisions for gradient quality in federated learning.
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
//!
//! ## AIR Constraints (5 polynomial equations, identical in both backends)
//!
//! 1. EMA Update: `ema[t+1] = (beta * ema[t] + (1-beta) * x[t])`
//! 2. Violation Flag: `is_viol = (x[t] < threshold) ? 1 : 0`
//! 3. Violation Counter: `viol[t+1] = is_viol ? viol[t] + 1 : 0`
//! 4. Clear Counter: `clear[t+1] = !is_viol ? clear[t] + 1 : 0`
//! 5. Hysteresis: Multi-condition quarantine entry/release

use serde::{Deserialize, Serialize};

use crate::fixed_point::FixedPoint;

/// PoGQ public parameters — shared between both backends.
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

/// PoGQ witness — private quality scores.
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
/// for both RISC0 guest programs and Winterfell AIR circuits.
/// Both backends must produce identical results for the same inputs.
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
                // Not quarantined — enter if k consecutive violations
                if state.consec_viol >= inputs.k_violations {
                    state.quarantined = true;
                }
            } else {
                // Quarantined — release if m consecutive clears
                if state.consec_clear >= inputs.m_clears {
                    state.quarantined = false;
                }
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

/// Compare results from two backends (Rust equivalent of DualBackendTester.compare_backends).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualBackendComparison {
    pub risc0_time_ms: Option<f64>,
    pub winterfell_time_ms: Option<f64>,
    pub speedup: Option<f64>,
    pub decisions_match: Option<bool>,
    pub risc0_proof_size: Option<usize>,
    pub winterfell_proof_size: Option<usize>,
}

impl DualBackendComparison {
    /// Create from two proof results.
    pub fn from_results(
        risc0: Option<&crate::types::ProofResult>,
        winterfell: Option<&crate::types::ProofResult>,
        risc0_decision: Option<bool>,
        winterfell_decision: Option<bool>,
    ) -> Self {
        let speedup = match (risc0, winterfell) {
            (Some(r), Some(w)) if w.proving_time_ms > 0 => {
                Some(r.proving_time_ms as f64 / w.proving_time_ms as f64)
            }
            _ => None,
        };

        Self {
            risc0_time_ms: risc0.map(|r| r.proving_time_ms as f64),
            winterfell_time_ms: winterfell.map(|w| w.proving_time_ms as f64),
            speedup,
            decisions_match: match (risc0_decision, winterfell_decision) {
                (Some(r), Some(w)) => Some(r == w),
                _ => None,
            },
            risc0_proof_size: risc0.map(|r| r.proof_size),
            winterfell_proof_size: winterfell.map(|w| w.proof_size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_inputs(threshold: f32) -> PoGQPublicInputs {
        PoGQPublicInputs {
            threshold: FixedPoint::from_f32(threshold),
            ..Default::default()
        }
    }

    #[test]
    fn test_all_pass_no_quarantine() {
        let inputs = default_inputs(0.5);
        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.8),
                FixedPoint::from_f32(0.9),
                FixedPoint::from_f32(0.7),
                FixedPoint::from_f32(0.85),
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(!result.quarantined, "all scores above threshold, no quarantine");
        assert_eq!(result.final_viol, 0);
        assert_eq!(result.rounds_processed, 4);
    }

    #[test]
    fn test_violations_trigger_quarantine() {
        let inputs = default_inputs(0.9); // High threshold
        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.5), // violation 1
                FixedPoint::from_f32(0.4), // violation 2 → quarantine (k=2)
                FixedPoint::from_f32(0.3), // still quarantined
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(result.quarantined, "2 consecutive violations should trigger quarantine (k=2)");
    }

    #[test]
    fn test_quarantine_release() {
        let mut inputs = default_inputs(0.9);
        inputs.quar_init = 1; // Start quarantined
        inputs.m_clears = 1;  // Release after 1 clear

        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.95), // clear → release (m=1)
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(!result.quarantined, "1 clear should release from quarantine (m=1)");
    }

    #[test]
    fn test_warmup_prevents_quarantine() {
        let mut inputs = default_inputs(0.9);
        inputs.warmup_rounds = 5;

        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.1), // violation but in warmup
                FixedPoint::from_f32(0.1), // violation but in warmup
                FixedPoint::from_f32(0.1), // violation but in warmup
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(!result.quarantined, "warmup prevents quarantine even with violations");
        assert!(result.trace[0].in_warmup);
    }

    #[test]
    fn test_ema_update() {
        let inputs = default_inputs(0.5);
        let witness = PoGQWitness {
            scores: vec![FixedPoint::from_f32(1.0)],
        };
        let result = simulate_pogq(&inputs, &witness);
        // ema = 0.85 * 0.75 + 0.15 * 1.0 = 0.6375 + 0.15 = 0.7875
        let expected = 0.7875f32;
        let actual = result.final_ema.to_f32();
        assert!(
            (expected - actual).abs() < 0.01,
            "EMA: expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_trace_length_matches_scores() {
        let inputs = default_inputs(0.5);
        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.8),
                FixedPoint::from_f32(0.7),
                FixedPoint::from_f32(0.6),
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert_eq!(result.trace.len(), 3);
        assert_eq!(result.rounds_processed, 3);
    }

    #[test]
    fn test_counter_reset_on_transition() {
        let inputs = default_inputs(0.5);
        let witness = PoGQWitness {
            scores: vec![
                FixedPoint::from_f32(0.3), // viol 1
                FixedPoint::from_f32(0.8), // clear (resets viol)
                FixedPoint::from_f32(0.3), // viol 1 (fresh)
            ],
        };
        let result = simulate_pogq(&inputs, &witness);
        assert!(!result.quarantined, "violation counter resets on clear");
        assert_eq!(result.final_viol, 1); // Only 1 consecutive
    }

    #[test]
    fn test_dual_backend_comparison() {
        use crate::types::{BackendId, ProofResult};

        let r0 = ProofResult {
            proof: vec![0; 221_000],
            backend: BackendId::Risc0,
            proving_time_ms: 46600,
            proof_size: 221_000,
        };
        let wf = ProofResult {
            proof: vec![0; 200_000],
            backend: BackendId::Winterfell,
            proving_time_ms: 8000,
            proof_size: 200_000,
        };

        let cmp = DualBackendComparison::from_results(
            Some(&r0),
            Some(&wf),
            Some(true),
            Some(true),
        );

        assert!(cmp.decisions_match == Some(true));
        let speedup = cmp.speedup.unwrap();
        assert!(speedup > 5.0, "RISC0 should be ~5.8x slower: {}", speedup);
    }
}
