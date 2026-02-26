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
        debug_assert!(params.len_ok());
        debug_assert_eq!(input.len(), self.d_model);
        debug_assert_eq!(output.len(), self.d_model);

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
