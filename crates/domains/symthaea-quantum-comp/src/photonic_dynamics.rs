//! Classical numerical reference for XY-to-Ising phase-quadrature dynamics.
//!
//! This is a source-bounded software model of the classical difference equations
//! described by Kim & Yamamoto (Scientific Reports 16, 10396, 2026,
//! doi:10.1038/s41598-026-41315-7). It is not an optical-device simulator,
//! quantum-noise model, calibrated photonic backend, or optimization solver.
//! Coefficients, state, and schedule values must be supplied explicitly.

use core::fmt;

/// Validated numerical parameters for the phase-quadrature update.
///
/// Couplings use a symmetric row-major matrix with an exactly zero diagonal.
/// The model implements the classical equations only and neglects quantum
/// fluctuations, as does the source model it transcribes.
#[derive(Debug, Clone, PartialEq)]
pub struct PhotonicDynamicsConfig {
    vertex_count: usize,
    couplings: Vec<f64>,
    delta_t: f64,
    momentum: f64,
    small_signal_gain_power: f64,
    saturation_pulse_energy: f64,
}

/// Current and previous in-phase/out-of-phase amplitudes.
///
/// The previous vectors represent the state at the prior round trip. They must
/// not alias or be replaced by the current vectors: the momentum term explicitly
/// depends on the difference between consecutive states.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureState {
    /// Current in-phase quadrature (x_i(t)).
    pub x: Vec<f64>,
    /// Current out-of-phase quadrature (p_i(t)).
    pub p: Vec<f64>,
    /// Previous in-phase quadrature (x_i(t-\Delta t)).
    pub previous_x: Vec<f64>,
    /// Previous out-of-phase quadrature (p_i(t-\Delta t)).
    pub previous_p: Vec<f64>,
}

/// One numerical update and diagnostic values from the exact step just evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadratureStep {
    /// New state; old current state is copied into the previous-state fields.
    pub next_state: QuadratureState,
    /// Per-oscillator phase-sensitive gain coefficient (g_x(t)).
    pub gain_x: Vec<f64>,
    /// Per-oscillator transition-dependent gain coefficient (g_p(t)).
    pub gain_p: Vec<f64>,
    /// Squared magnitude of each oscillator's complex feedback field.
    pub feedback_norm_squared: Vec<f64>,
}

/// Validation or numerical errors for the classical quadrature model.
#[derive(Debug, Clone, PartialEq)]
pub enum PhotonicDynamicsError {
    /// Coupling matrix rows or columns are not square and equal to the declared size.
    InvalidMatrixShape {
        /// Number of rows supplied.
        rows: usize,
        /// Expected number of columns per row.
        expected_columns: usize,
        /// First row with a different column count, if available.
        invalid_row: Option<usize>,
    },
    /// A coupling coefficient is non-finite.
    NonFiniteCoupling {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },
    /// The Ising/XY source model requires zero self-coupling.
    NonZeroDiagonal {
        /// Vertex with a nonzero self-coupling.
        vertex: usize,
    },
    /// Coupling entries are not exactly symmetric.
    AsymmetricCouplings {
        /// First vertex.
        row: usize,
        /// Second vertex.
        column: usize,
    },
    /// A time step or other parameter is outside its valid domain.
    InvalidParameter {
        /// Parameter identifier.
        name: &'static str,
        /// Supplied value.
        value: f64,
    },
    /// Transition coefficient must be finite and between zero and one.
    InvalidTransition {
        /// Supplied transition coefficient.
        value: f64,
    },
    /// State vector lengths disagree with the configured oscillator count.
    StateLengthMismatch {
        /// Vector identifier.
        name: &'static str,
        /// Expected vector length.
        expected: usize,
        /// Actual vector length.
        actual: usize,
    },
    /// State input contains a non-finite value.
    NonFiniteState {
        /// Vector identifier.
        name: &'static str,
        /// Element index.
        index: usize,
    },
    /// An intermediate value or output is non-finite.
    NumericOverflow {
        /// Stage that produced the non-finite value.
        stage: &'static str,
        /// Oscillator index.
        vertex: usize,
    },
}

impl fmt::Display for PhotonicDynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMatrixShape {
                rows,
                expected_columns,
                invalid_row,
            } => write!(
                f,
                "coupling matrix has {rows} rows; expected {expected_columns} columns per row (first invalid row: {invalid_row:?})"
            ),
            Self::NonFiniteCoupling { row, column } => {
                write!(f, "coupling ({row}, {column}) is non-finite")
            }
            Self::NonZeroDiagonal { vertex } => {
                write!(f, "self-coupling at vertex {vertex} must be zero")
            }
            Self::AsymmetricCouplings { row, column } => {
                write!(f, "couplings ({row}, {column}) and ({column}, {row}) differ")
            }
            Self::InvalidParameter { name, value } => {
                write!(f, "parameter {name} has invalid value {value}")
            }
            Self::InvalidTransition { value } => {
                write!(f, "transition coefficient must lie in [0, 1], got {value}")
            }
            Self::StateLengthMismatch {
                name,
                expected,
                actual,
            } => write!(f, "{name} has length {actual}; expected {expected}"),
            Self::NonFiniteState { name, index } => {
                write!(f, "{name}[{index}] is non-finite")
            }
            Self::NumericOverflow { stage, vertex } => {
                write!(f, "non-finite value during {stage} at vertex {vertex}")
            }
        }
    }
}

impl std::error::Error for PhotonicDynamicsError {}

impl PhotonicDynamicsConfig {
    /// Construct a validated model from a symmetric coupling matrix.
    ///
    /// `small_signal_gain_power` is the paper's power-gain parameter G; it must
    /// be positive and finite. `delta_t` and `saturation_pulse_energy` must be
    /// positive and finite. `momentum` is the non-negative coefficient kappa.
    /// The coupling matrix must be finite, symmetric and have zero diagonal;
    /// this constructor does not symmetrize or otherwise repair supplied data.
    pub fn new(
        couplings: Vec<Vec<f64>>,
        delta_t: f64,
        momentum: f64,
        small_signal_gain_power: f64,
        saturation_pulse_energy: f64,
    ) -> Result<Self, PhotonicDynamicsError> {
        let vertex_count = couplings.len();
        for (row, values) in couplings.iter().enumerate() {
            if values.len() != vertex_count {
                return Err(PhotonicDynamicsError::InvalidMatrixShape {
                    rows: vertex_count,
                    expected_columns: vertex_count,
                    invalid_row: Some(row),
                });
            }
        }

        for (row, values) in couplings.iter().enumerate() {
            for (column, &value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PhotonicDynamicsError::NonFiniteCoupling { row, column });
                }
                if row == column && value != 0.0 {
                    return Err(PhotonicDynamicsError::NonZeroDiagonal { vertex: row });
                }
                if values[column] != couplings[column][row] {
                    return Err(PhotonicDynamicsError::AsymmetricCouplings { row, column });
                }
            }
        }

        for (name, value) in [
            ("delta_t", delta_t),
            ("small_signal_gain_power", small_signal_gain_power),
            ("saturation_pulse_energy", saturation_pulse_energy),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(PhotonicDynamicsError::InvalidParameter { name, value });
            }
        }
        if !momentum.is_finite() || momentum < 0.0 {
            return Err(PhotonicDynamicsError::InvalidParameter {
                name: "momentum",
                value: momentum,
            });
        }

        Ok(Self {
            vertex_count,
            couplings: couplings.into_iter().flatten().collect(),
            delta_t,
            momentum,
            small_signal_gain_power,
            saturation_pulse_energy,
        })
    }

    /// Number of oscillator variables.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// The declared round-trip integration step.
    pub fn delta_t(&self) -> f64 {
        self.delta_t
    }

    /// One explicit classical update with transition coefficient (c(t)).
    ///
    /// `transition=0` is phase-insensitive XY-like gain; `transition=1`
    /// is phase-sensitive Ising-like gain. For every oscillator, the saturation
    /// denominator uses its own complex feedback field:
    ///
    /// `F_i = sum_j J_ij * (x_j + i*p_j)`
    ///
    /// `g_x = sqrt(G) / (1 + |F_i|^2/E_s)`
    ///
    /// `g_p = ((1-c)*sqrt(G) + c/sqrt(G)) / (1 + |F_i|^2/E_s)`
    ///
    /// This is the source paper's classical update. It does not simulate photons,
    /// shot noise, detector calibration, or a physical device.
    pub fn step(
        &self,
        state: &QuadratureState,
        transition: f64,
    ) -> Result<QuadratureStep, PhotonicDynamicsError> {
        if !transition.is_finite() || !(0.0..=1.0).contains(&transition) {
            return Err(PhotonicDynamicsError::InvalidTransition { value: transition });
        }
        self.validate_state(state)?;

        let root_gain = self.small_signal_gain_power.sqrt();
        let inverse_root_gain = 1.0 / root_gain;
        let mut next_x = Vec::with_capacity(self.vertex_count);
        let mut next_p = Vec::with_capacity(self.vertex_count);
        let mut gain_x = Vec::with_capacity(self.vertex_count);
        let mut gain_p = Vec::with_capacity(self.vertex_count);
        let mut feedback_norm_squared = Vec::with_capacity(self.vertex_count);

        for vertex in 0..self.vertex_count {
            let mut feedback_x = 0.0;
            let mut feedback_p = 0.0;
            for other in 0..self.vertex_count {
                let coupling = self.couplings[vertex * self.vertex_count + other];
                feedback_x += coupling * state.x[other];
                feedback_p += coupling * state.p[other];
            }
            if !feedback_x.is_finite() || !feedback_p.is_finite() {
                return Err(PhotonicDynamicsError::NumericOverflow {
                    stage: "feedback field",
                    vertex,
                });
            }

            let norm_squared = feedback_x * feedback_x + feedback_p * feedback_p;
            let denominator = 1.0 + norm_squared / self.saturation_pulse_energy;
            if !norm_squared.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
                return Err(PhotonicDynamicsError::NumericOverflow {
                    stage: "feedback saturation",
                    vertex,
                });
            }

            let gx = root_gain / denominator;
            let gp = ((1.0 - transition) * root_gain
                + transition * inverse_root_gain)
                / denominator;
            let dx = -state.x[vertex]
                + gx * feedback_x
                + self.momentum * (state.x[vertex] - state.previous_x[vertex]);
            let dp = -state.p[vertex]
                + gp * feedback_p
                + self.momentum * (state.p[vertex] - state.previous_p[vertex]);
            let updated_x = state.x[vertex] + self.delta_t * dx;
            let updated_p = state.p[vertex] + self.delta_t * dp;

            if !gx.is_finite()
                || !gp.is_finite()
                || !dx.is_finite()
                || !dp.is_finite()
                || !updated_x.is_finite()
                || !updated_p.is_finite()
            {
                return Err(PhotonicDynamicsError::NumericOverflow {
                    stage: "quadrature update",
                    vertex,
                });
            }

            gain_x.push(gx);
            gain_p.push(gp);
            feedback_norm_squared.push(norm_squared);
            next_x.push(updated_x);
            next_p.push(updated_p);
        }

        Ok(QuadratureStep {
            next_state: QuadratureState {
                previous_x: state.x.clone(),
                previous_p: state.p.clone(),
                x: next_x,
                p: next_p,
            },
            gain_x,
            gain_p,
            feedback_norm_squared,
        })
    }

    fn validate_state(
        &self,
        state: &QuadratureState,
    ) -> Result<(), PhotonicDynamicsError> {
        for (name, values) in [
            ("x", &state.x),
            ("p", &state.p),
            ("previous_x", &state.previous_x),
            ("previous_p", &state.previous_p),
        ] {
            if values.len() != self.vertex_count {
                return Err(PhotonicDynamicsError::StateLengthMismatch {
                    name,
                    expected: self.vertex_count,
                    actual: values.len(),
                });
            }
            for (index, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(PhotonicDynamicsError::NonFiniteState { name, index });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(x: &[f64], p: &[f64], previous_x: &[f64], previous_p: &[f64]) -> QuadratureState {
        QuadratureState {
            x: x.to_vec(),
            p: p.to_vec(),
            previous_x: previous_x.to_vec(),
            previous_p: previous_p.to_vec(),
        }
    }

    fn two_spin_config() -> PhotonicDynamicsConfig {
        PhotonicDynamicsConfig::new(
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            0.1,
            0.25,
            4.0,
            100.0,
        )
        .expect("valid two-spin configuration")
    }

    #[test]
    fn xy_limit_applies_equal_quadrature_gain() {
        let model = two_spin_config();
        let s = state(&[0.0, 1.0], &[0.0, 2.0], &[0.0, 0.0], &[0.0, 0.0]);
        let report = model.step(&s, 0.0).expect("valid XY-like step");
        assert_eq!(report.gain_x[0], report.gain_p[0]);
        assert_eq!(report.gain_x[1], report.gain_p[1]);
    }

    #[test]
    fn ising_limit_suppresses_out_of_phase_gain_by_inverse_power_gain_root() {
        let model = two_spin_config();
        let s = state(&[0.0, 1.0], &[0.0, 2.0], &[0.0, 0.0], &[0.0, 0.0]);
        let report = model.step(&s, 1.0).expect("valid Ising-like step");
        assert!((report.gain_x[1] - 2.0).abs() < 1e-12);
        assert!((report.gain_p[1] - 0.5).abs() < 1e-12);
        assert!(report.gain_p[1] < report.gain_x[1]);
    }

    #[test]
    fn transition_interpolates_between_the_two_gain_limits() {
        let model = two_spin_config();
        let s = state(&[0.0, 1.0], &[0.0, 2.0], &[0.0, 0.0], &[0.0, 0.0]);
        let xy = model.step(&s, 0.0).expect("XY step");
        let middle = model.step(&s, 0.5).expect("mid-transition step");
        let ising = model.step(&s, 1.0).expect("Ising step");
        for i in 0..2 {
            assert_eq!(xy.gain_x[i], middle.gain_x[i]);
            assert!(middle.gain_p[i] >= ising.gain_p[i]);
            assert!(middle.gain_p[i] <= xy.gain_p[i]);
        }
    }

    #[test]
    fn zero_coupling_update_matches_the_loss_and_momentum_equations() {
        let model =
            PhotonicDynamicsConfig::new(vec![vec![0.0]], 0.1, 0.25, 4.0, 1.0)
                .expect("valid one-spin uncoupled model");
        let s = state(&[1.0], &[2.0], &[0.5], &[1.0]);
        let out = model.step(&s, 0.3).expect("valid uncoupled step");
        assert!((out.next_state.x[0] - 0.9125).abs() < 1e-12);
        assert!((out.next_state.p[0] - 1.825).abs() < 1e-12);
        assert_eq!(out.next_state.previous_x, s.x);
        assert_eq!(out.next_state.previous_p, s.p);
    }

    #[test]
    fn transition_and_state_validation_fail_closed() {
        let model = two_spin_config();
        let s = state(&[0.0, 1.0], &[0.0, 2.0], &[0.0, 0.0], &[0.0, 0.0]);
        assert!(matches!(
            model.step(&s, f64::NAN),
            Err(PhotonicDynamicsError::InvalidTransition { .. })
        ));
        assert!(matches!(
            model.step(&s, 1.01),
            Err(PhotonicDynamicsError::InvalidTransition { .. })
        ));
        let wrong = state(&[0.0], &[0.0, 2.0], &[0.0, 0.0], &[0.0, 0.0]);
        assert!(matches!(
            model.step(&wrong, 0.0),
            Err(PhotonicDynamicsError::StateLengthMismatch { .. })
        ));
        let non_finite = state(&[0.0, f64::INFINITY], &[0.0, 2.0], &[0.0, 0.0], &[0.0, 0.0]);
        assert!(matches!(
            model.step(&non_finite, 0.0),
            Err(PhotonicDynamicsError::NonFiniteState { .. })
        ));
    }

    #[test]
    fn configuration_rejects_invalid_physical_and_numerical_inputs() {
        assert!(matches!(
            PhotonicDynamicsConfig::new(vec![vec![1.0]], 0.1, 0.0, 4.0, 1.0),
            Err(PhotonicDynamicsError::NonZeroDiagonal { .. })
        ));
        assert!(matches!(
            PhotonicDynamicsConfig::new(vec![vec![0.0, 2.0], vec![1.0, 0.0]], 0.1, 0.0, 4.0, 1.0),
            Err(PhotonicDynamicsError::AsymmetricCouplings { .. })
        ));
        assert!(matches!(
            PhotonicDynamicsConfig::new(vec![vec![0.0]], 0.1, 0.0, 0.0, 1.0),
            Err(PhotonicDynamicsError::InvalidParameter { .. })
        ));
        assert!(matches!(
            PhotonicDynamicsConfig::new(vec![vec![0.0]], 0.1, -0.1, 4.0, 1.0),
            Err(PhotonicDynamicsError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn non_finite_feedback_is_reported_instead_of_emitting_a_candidate() {
        let model = PhotonicDynamicsConfig::new(
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            0.1,
            0.0,
            4.0,
            1.0,
        )
        .expect("valid config");
        let s = state(
            &[1.0e308, 1.0e308],
            &[0.0, 0.0],
            &[0.0, 0.0],
            &[0.0, 0.0],
        );
        assert!(matches!(
            model.step(&s, 0.0),
            Err(PhotonicDynamicsError::NumericOverflow { .. })
        ));
    }
}
