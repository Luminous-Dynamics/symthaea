// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SSM-based temporal backend for benchmark trial-by-trial dynamics.
//!
//! Wraps `symthaea_ssm::SsmState` with `d_model=1` (scalar trial activation)
//! and configurable `d_state` and decay rate. Replaces ad-hoc temporal decay
//! models (linear, piecewise, exponential) with principled state-space recurrence.

use symthaea_ssm::{SelectiveParams, SsmState};

/// SSM temporal backend for trial-level dynamics.
///
/// Uses a diagonal SSM (`d_model=1`, `d_state=4` default) where the `A` parameter
/// controls decay rate: `A=-0.5` → fast decay, `A=-0.05` → slow decay.
/// Each `step()` call advances one trial.
pub struct SsmTemporalBackend {
    state: SsmState,
    params: SelectiveParams,
}

impl SsmTemporalBackend {
    /// Create a new SSM temporal backend.
    ///
    /// - `decay_rate`: Negative value controlling decay speed (e.g., -0.03 for slow, -0.5 for fast).
    /// - `d_state`: Number of hidden dimensions (default recommendation: 4).
    pub fn new(decay_rate: f32, d_state: usize) -> Self {
        let d_model = 1;
        let mut params = SelectiveParams::new(d_model, d_state);
        // Set A to the desired decay rate across all hidden dims
        for a in &mut params.a {
            *a = decay_rate;
        }
        // B controls input coupling (uniform)
        for b in &mut params.b {
            *b = 1.0;
        }
        // C controls output readout (uniform, normalized by d_state)
        let c_val = 1.0 / d_state as f32;
        for c in &mut params.c {
            *c = c_val;
        }
        // Delta = 1.0 (one step per trial)
        params.set_delta(1.0);

        Self {
            state: SsmState::new(d_model, d_state),
            params,
        }
    }

    /// Step one trial: input activation → output (decayed) activation.
    ///
    /// Input is a scalar activation (e.g., 1.0 for "stimulus present", 0.0 for "no input").
    /// Returns the output activation after SSM processing.
    pub fn step(&mut self, input: f32) -> f32 {
        let input_slice = [input];
        let mut output = [0.0_f32];
        self.state.step(&input_slice, &self.params, &mut output);
        output[0]
    }

    /// Read current hidden state magnitude (L2 norm / sqrt(d_state)).
    ///
    /// Higher values indicate stronger memory trace.
    pub fn memory_strength(&self) -> f32 {
        let sum_sq: f32 = self.state.hidden_state.iter().map(|h| h * h).sum();
        (sum_sq / self.state.d_state as f32).sqrt()
    }

    /// Reset hidden state to zero (e.g., between conditions).
    pub fn reset(&mut self) {
        for h in &mut self.state.hidden_state {
            *h = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssm_temporal_step_decays() {
        let mut ssm = SsmTemporalBackend::new(-0.5, 4);
        // Input a pulse then let it decay
        let initial = ssm.step(1.0);
        let after1 = ssm.step(0.0);
        let after2 = ssm.step(0.0);
        assert!(initial > 0.0, "initial={}", initial);
        assert!(after1 < initial, "should decay: {} vs {}", after1, initial);
        assert!(
            after2 < after1,
            "should decay further: {} vs {}",
            after2,
            after1
        );
    }

    #[test]
    fn test_ssm_temporal_step_refreshes() {
        let mut ssm = SsmTemporalBackend::new(-0.3, 4);
        ssm.step(1.0);
        let before_refresh = ssm.step(0.0);
        let after_refresh = ssm.step(1.0);
        assert!(
            after_refresh > before_refresh,
            "refresh should increase output: {} vs {}",
            after_refresh,
            before_refresh
        );
    }

    #[test]
    fn test_ssm_temporal_memory_strength() {
        let mut ssm = SsmTemporalBackend::new(-0.2, 4);
        assert!(
            (ssm.memory_strength() - 0.0).abs() < 1e-6,
            "initial strength should be ~0"
        );
        ssm.step(1.0);
        let s1 = ssm.memory_strength();
        assert!(s1 > 0.0, "strength after input should be >0: {}", s1);
        ssm.step(0.0);
        let s2 = ssm.memory_strength();
        assert!(s2 < s1, "strength should decay: {} vs {}", s2, s1);
    }

    #[test]
    fn test_ssm_temporal_reset() {
        let mut ssm = SsmTemporalBackend::new(-0.2, 4);
        ssm.step(1.0);
        ssm.step(1.0);
        assert!(ssm.memory_strength() > 0.0);
        ssm.reset();
        assert!(
            (ssm.memory_strength() - 0.0).abs() < 1e-6,
            "reset should zero state"
        );
    }

    #[test]
    fn test_ssm_temporal_decay_rate_effect() {
        // Faster decay (larger |A|) should produce weaker output after same delay
        let mut fast = SsmTemporalBackend::new(-1.0, 4);
        let mut slow = SsmTemporalBackend::new(-0.1, 4);
        fast.step(1.0);
        slow.step(1.0);
        // After 5 decay steps
        for _ in 0..5 {
            fast.step(0.0);
            slow.step(0.0);
        }
        assert!(
            fast.memory_strength() < slow.memory_strength(),
            "fast decay {} should be < slow decay {}",
            fast.memory_strength(),
            slow.memory_strength()
        );
    }

    #[test]
    fn test_ssm_temporal_deterministic() {
        let mut ssm1 = SsmTemporalBackend::new(-0.3, 4);
        let mut ssm2 = SsmTemporalBackend::new(-0.3, 4);
        for input in &[1.0, 0.0, 0.5, 0.0, 1.0] {
            let o1 = ssm1.step(*input);
            let o2 = ssm2.step(*input);
            assert!(
                (o1 - o2).abs() < 1e-10,
                "outputs should match: {} vs {}",
                o1,
                o2
            );
        }
    }
}
