// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Public inputs and private witness structures

use serde::{Deserialize, Serialize};
use winterfell::math::fields::f64::BaseElement;

/// Public inputs (commitments visible to verifier)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicInputs {
    // Hash commitments (SHA-256 hex strings)
    pub h_calib: String,
    pub h_model: String,
    pub h_grad: String,

    // PoGQ parameters (fixed-point)
    pub beta_fp: u64,           // EMA smoothing (0.85 * 65536)
    pub w: u64,                 // Warm-up rounds
    pub k: u64,                 // Hysteresis: violations to quarantine
    pub m: u64,                 // Hysteresis: clears to release
    pub egregious_cap_fp: u64,  // Cap for hybrid score

    // Conformal threshold (fixed-point)
    pub threshold_fp: u64,

    // Previous state (fixed-point for ema, integers for counters)
    pub ema_prev_fp: u64,
    pub consec_viol_prev: u64,
    pub consec_clear_prev: u64,
    pub quarantined_prev: u64,  // 0 or 1

    // Current round
    pub current_round: u64,

    // Expected output
    pub quarantine_out: u64,    // 0 or 1
}

impl PublicInputs {
    /// Validate public inputs
    pub fn validate(&self) -> Result<(), String> {
        // Check hash lengths
        if self.h_calib.len() != 64 {
            return Err(format!("h_calib must be 64 hex chars, got {}", self.h_calib.len()));
        }
        if self.h_model.len() != 64 {
            return Err(format!("h_model must be 64 hex chars, got {}", self.h_model.len()));
        }
        if self.h_grad.len() != 64 {
            return Err(format!("h_grad must be 64 hex chars, got {}", self.h_grad.len()));
        }

        // Check fixed-point bounds
        if self.beta_fp > 65536 {
            return Err(format!("beta_fp out of range: {}", self.beta_fp));
        }
        if self.threshold_fp > 65536 {
            return Err(format!("threshold_fp out of range: {}", self.threshold_fp));
        }
        if self.ema_prev_fp > 65536 {
            return Err(format!("ema_prev_fp out of range: {}", self.ema_prev_fp));
        }

        // Check boolean flags
        if self.quarantined_prev > 1 {
            return Err(format!("quarantined_prev must be 0 or 1, got {}", self.quarantined_prev));
        }
        if self.quarantine_out > 1 {
            return Err(format!("quarantine_out must be 0 or 1, got {}", self.quarantine_out));
        }

        Ok(())
    }

    /// Convert to field elements for STARK
    pub fn to_field_elements(&self) -> Vec<BaseElement> {
        vec![
            BaseElement::new(self.current_round),
            BaseElement::new(self.w),
            BaseElement::new(self.beta_fp),
            BaseElement::new(self.threshold_fp),
            BaseElement::new(self.ema_prev_fp),
            BaseElement::new(self.consec_viol_prev),
            BaseElement::new(self.consec_clear_prev),
            BaseElement::new(self.quarantined_prev),
            BaseElement::new(self.quarantine_out),
            BaseElement::new(self.k),
            BaseElement::new(self.m),
        ]
    }

    /// Parse from JSON
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse public inputs: {}", e))
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize public inputs: {}", e))
    }
}

/// Private witness (only known to prover)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateWitness {
    pub x_t_fp: u64,        // Hybrid score this round (fixed-point)
    pub in_warmup: u64,     // Boolean: 1 if t ≤ W, else 0
    pub violation_t: u64,   // Boolean: 1 if x_t < threshold, else 0
    pub release_t: u64,     // Boolean: 1 if releasing from quarantine, else 0
}

impl PrivateWitness {
    /// Validate private witness
    pub fn validate(&self) -> Result<(), String> {
        // Check fixed-point bounds
        if self.x_t_fp > 65536 {
            return Err(format!("x_t_fp out of range: {}", self.x_t_fp));
        }

        // Check boolean flags
        if self.in_warmup > 1 {
            return Err(format!("in_warmup must be 0 or 1, got {}", self.in_warmup));
        }
        if self.violation_t > 1 {
            return Err(format!("violation_t must be 0 or 1, got {}", self.violation_t));
        }
        if self.release_t > 1 {
            return Err(format!("release_t must be 0 or 1, got {}", self.release_t));
        }

        Ok(())
    }

    /// Parse from JSON
    pub fn from_json(json_str: &str) -> Result<Self, String> {
        serde_json::from_str(json_str)
            .map_err(|e| format!("Failed to parse witness: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_inputs_json_roundtrip() {
        let public = PublicInputs {
            h_calib: "a".repeat(64),
            h_model: "b".repeat(64),
            h_grad: "c".repeat(64),
            beta_fp: 55705,
            w: 3,
            k: 2,
            m: 3,
            egregious_cap_fp: 65535,
            threshold_fp: 58982,
            ema_prev_fp: 49152,
            consec_viol_prev: 1,
            consec_clear_prev: 0,
            quarantined_prev: 0,
            current_round: 5,
            quarantine_out: 0,
        };

        let json = public.to_json().unwrap();
        let parsed = PublicInputs::from_json(&json).unwrap();

        assert_eq!(public.beta_fp, parsed.beta_fp);
        assert_eq!(public.current_round, parsed.current_round);
    }

    #[test]
    fn test_witness_json_roundtrip() {
        let witness = PrivateWitness {
            x_t_fp: 52428,
            in_warmup: 0,
            violation_t: 1,
            release_t: 0,
        };

        let json = serde_json::to_string(&witness).unwrap();
        let parsed: PrivateWitness = serde_json::from_str(&json).unwrap();

        assert_eq!(witness.x_t_fp, parsed.x_t_fp);
        assert_eq!(witness.violation_t, parsed.violation_t);
    }
}
