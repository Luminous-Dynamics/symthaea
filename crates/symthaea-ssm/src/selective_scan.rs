// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagonal SSM selective scan core (zero-allocation update loop).

#[derive(Debug, Clone)]
pub struct SelectiveParams {
    pub delta: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub c: Vec<f32>,
    pub d_model: usize,
    pub d_state: usize,
}

impl SelectiveParams {
    pub fn new(d_model: usize, d_state: usize) -> Self {
        let len = d_model.saturating_mul(d_state);
        Self {
            delta: vec![1.0; d_model],
            a: vec![-0.5; len],
            b: vec![0.1; len],
            c: vec![0.1; len],
            d_model,
            d_state,
        }
    }

    pub fn set_delta(&mut self, delta: f32) {
        for d in &mut self.delta {
            *d = delta;
        }
    }

    pub fn len_ok(&self) -> bool {
        self.delta.len() == self.d_model
            && self.a.len() == self.d_model * self.d_state
            && self.b.len() == self.d_model * self.d_state
            && self.c.len() == self.d_model * self.d_state
    }
}

#[derive(Debug, Clone)]
pub struct SsmState {
    pub hidden_state: Vec<f32>,
    pub d_model: usize,
    pub d_state: usize,
}

impl SsmState {
    pub fn new(d_model: usize, d_state: usize) -> Self {
        Self {
            hidden_state: vec![0.0; d_model.saturating_mul(d_state)],
            d_model,
            d_state,
        }
    }

    /// Update the state in-place for one timestep.
    ///
    /// - `input` length must be `d_model`.
    /// - `output` length must be `d_model`.
    pub fn step(&mut self, input: &[f32], params: &SelectiveParams, output: &mut [f32]) {
        assert!(params.len_ok());
        assert_eq!(input.len(), self.d_model);
        assert_eq!(output.len(), self.d_model);

        let d_state = self.d_state;
        for i in 0..self.d_model {
            let offset = i * d_state;
            let dt = params.delta[i];
            let x_i = input[i];
            let mut y_i = 0.0_f32;

            for j in 0..d_state {
                let idx = offset + j;
                let a_val = params.a[idx];
                let b_val = params.b[idx];
                let c_val = params.c[idx];

                let dt_a = dt * a_val;
                let bar_a = dt_a.exp();
                let bar_b = if a_val.abs() > 1e-6 {
                    ((bar_a - 1.0) / a_val) * b_val
                } else {
                    dt * b_val
                };

                let h = bar_a * self.hidden_state[idx] + bar_b * x_i;
                self.hidden_state[idx] = h;
                y_i += c_val * h;
            }

            output[i] = y_i;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selective_params_new_dimensions() {
        let params = SelectiveParams::new(4, 8);
        assert_eq!(params.d_model, 4);
        assert_eq!(params.d_state, 8);
        assert_eq!(params.delta.len(), 4);
        assert_eq!(params.a.len(), 32);
        assert_eq!(params.b.len(), 32);
        assert_eq!(params.c.len(), 32);
    }

    #[test]
    fn selective_params_defaults() {
        let params = SelectiveParams::new(2, 3);
        assert!(params.delta.iter().all(|&d| (d - 1.0).abs() < 1e-9));
        assert!(params.a.iter().all(|&a| (a - (-0.5)).abs() < 1e-9));
        assert!(params.b.iter().all(|&b| (b - 0.1).abs() < 1e-9));
        assert!(params.c.iter().all(|&c| (c - 0.1).abs() < 1e-9));
    }

    #[test]
    fn selective_params_len_ok_valid() {
        let params = SelectiveParams::new(4, 8);
        assert!(params.len_ok());
    }

    #[test]
    fn selective_params_len_ok_invalid() {
        let mut params = SelectiveParams::new(4, 8);
        params.delta.push(0.0);
        assert!(!params.len_ok());
    }

    #[test]
    fn selective_params_set_delta() {
        let mut params = SelectiveParams::new(4, 8);
        params.set_delta(0.01);
        assert!(params.delta.iter().all(|&d| (d - 0.01).abs() < 1e-9));
        assert!(params.len_ok());
    }

    #[test]
    fn selective_params_zero_dimensions() {
        let params = SelectiveParams::new(0, 0);
        assert_eq!(params.delta.len(), 0);
        assert_eq!(params.a.len(), 0);
        assert!(params.len_ok());
    }

    #[test]
    fn ssm_state_new_zeroed() {
        let state = SsmState::new(4, 8);
        assert_eq!(state.d_model, 4);
        assert_eq!(state.d_state, 8);
        assert_eq!(state.hidden_state.len(), 32);
        assert!(state.hidden_state.iter().all(|&h| h == 0.0));
    }

    #[test]
    fn step_zero_input_no_accumulation() {
        let params = SelectiveParams::new(2, 2);
        let mut state = SsmState::new(2, 2);
        let input = [0.0_f32; 2];
        let mut output = [0.0_f32; 2];
        state.step(&input, &params, &mut output);
        assert!(
            output.iter().all(|&y| y.abs() < 1e-9),
            "zero input from zero state must produce zero output, got {:?}",
            output
        );
        assert!(state.hidden_state.iter().all(|&h| h.abs() < 1e-9));
    }

    #[test]
    fn step_single_input_produces_nonzero_output() {
        let params = SelectiveParams::new(1, 1);
        let mut state = SsmState::new(1, 1);
        let input = [1.0_f32];
        let mut output = [0.0_f32];
        state.step(&input, &params, &mut output);
        assert!(
            output[0].abs() > 1e-6,
            "non-zero input should produce non-zero output"
        );
        assert!(output[0].is_finite(), "output must be finite");
    }

    #[test]
    fn zoh_discretization_decay_with_negative_a() {
        let mut params = SelectiveParams::new(1, 1);
        params.a[0] = -1.0;
        params.delta[0] = 1.0;
        let mut state = SsmState::new(1, 1);
        let mut output = [0.0_f32];
        state.step(&[1.0], &params, &mut output);
        let h_after_pulse = state.hidden_state[0];
        assert!(h_after_pulse.abs() > 0.0);
        let expected_decay = (-1.0_f32).exp();
        state.step(&[0.0], &params, &mut output);
        let h_after_decay = state.hidden_state[0];
        let ratio = h_after_decay / h_after_pulse;
        assert!(
            (ratio - expected_decay).abs() < 1e-5,
            "hidden state should decay by exp(delta*A)={}, got ratio={}",
            expected_decay,
            ratio
        );
    }

    #[test]
    fn state_accumulates_with_repeated_input() {
        let params = SelectiveParams::new(1, 1);
        let mut state = SsmState::new(1, 1);
        let mut output = [0.0_f32];
        let mut outputs = Vec::new();
        for _ in 0..10 {
            state.step(&[1.0], &params, &mut output);
            outputs.push(output[0]);
        }
        let last = *outputs.last().unwrap();
        assert!(
            last.abs() > outputs[0].abs(),
            "repeated input should accumulate: first={}, last={}",
            outputs[0],
            last
        );
    }

    #[test]
    fn state_decays_with_zero_input_after_charging() {
        let params = SelectiveParams::new(1, 1);
        let mut state = SsmState::new(1, 1);
        let mut output = [0.0_f32];
        for _ in 0..5 {
            state.step(&[1.0], &params, &mut output);
        }
        let charged_output = output[0];
        assert!(charged_output.abs() > 1e-6);
        for _ in 0..20 {
            state.step(&[0.0], &params, &mut output);
        }
        assert!(
            output[0].abs() < charged_output.abs(),
            "output should decay with zero input: charged={}, now={}",
            charged_output,
            output[0]
        );
    }

    #[test]
    fn determinism_same_inputs_same_outputs() {
        let params = SelectiveParams::new(3, 4);
        let inputs: Vec<[f32; 3]> = (0..5)
            .map(|i| {
                let v = i as f32 * 0.3;
                [v, v + 0.1, v - 0.2]
            })
            .collect();
        let mut state1 = SsmState::new(3, 4);
        let mut outputs1 = Vec::new();
        for inp in &inputs {
            let mut out = [0.0_f32; 3];
            state1.step(inp, &params, &mut out);
            outputs1.push(out);
        }
        let mut state2 = SsmState::new(3, 4);
        let mut outputs2 = Vec::new();
        for inp in &inputs {
            let mut out = [0.0_f32; 3];
            state2.step(inp, &params, &mut out);
            outputs2.push(out);
        }
        for (o1, o2) in outputs1.iter().zip(outputs2.iter()) {
            for (a, b) in o1.iter().zip(o2.iter()) {
                assert!((a - b).abs() < 1e-9, "determinism violated: {} != {}", a, b);
            }
        }
    }

    #[test]
    fn large_negative_a_fast_decay() {
        let mut params = SelectiveParams::new(1, 1);
        params.a[0] = -10.0;
        params.delta[0] = 1.0;
        let mut state = SsmState::new(1, 1);
        let mut output = [0.0_f32];
        state.step(&[1.0], &params, &mut output);
        let h0 = state.hidden_state[0];
        state.step(&[0.0], &params, &mut output);
        let h1 = state.hidden_state[0];
        assert!(
            h1.abs() < h0.abs() * 0.001,
            "large negative A should cause near-total decay: h0={}, h1={}",
            h0,
            h1
        );
    }

    #[test]
    fn very_small_delta_minimal_state_change() {
        let mut params = SelectiveParams::new(1, 1);
        params.set_delta(1e-6);
        let mut state = SsmState::new(1, 1);
        let mut output = [0.0_f32];
        state.step(&[1.0], &params, &mut output);
        assert!(
            output[0].abs() < 0.01,
            "very small delta should produce very small output, got {}",
            output[0]
        );
    }

    #[test]
    fn multi_dimension_independence() {
        let mut params = SelectiveParams::new(2, 1);
        params.a[0] = -0.1;
        params.a[1] = -5.0;
        let mut state = SsmState::new(2, 1);
        let mut output = [0.0_f32; 2];
        state.step(&[1.0, 1.0], &params, &mut output);
        state.step(&[0.0, 0.0], &params, &mut output);
        let ratio = state.hidden_state[1].abs() / state.hidden_state[0].abs().max(1e-30);
        assert!(
            ratio < 0.1,
            "fast-decay dimension should decay more: h0={}, h1={}",
            state.hidden_state[0],
            state.hidden_state[1]
        );
    }

    #[test]
    fn a_near_zero_uses_linear_approximation() {
        let mut params = SelectiveParams::new(1, 1);
        params.a[0] = 1e-8;
        let mut state = SsmState::new(1, 1);
        let mut output = [0.0_f32];
        state.step(&[1.0], &params, &mut output);
        assert!(
            output[0].is_finite(),
            "near-zero A should not cause NaN or Inf"
        );
        assert!(
            output[0].abs() > 0.0,
            "near-zero A with input should produce output"
        );
        let mut params2 = SelectiveParams::new(1, 1);
        params2.a[0] = 1e-5;
        let mut state2 = SsmState::new(1, 1);
        let mut output2 = [0.0_f32];
        state2.step(&[1.0], &params2, &mut output2);
        assert!(
            (output[0] - output2[0]).abs() < 0.01,
            "branch transition should be smooth: linear={}, exact={}",
            output[0],
            output2[0]
        );
    }
}
