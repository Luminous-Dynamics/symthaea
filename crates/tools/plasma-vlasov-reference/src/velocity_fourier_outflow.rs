//! Linearized Maxwellian Vlasov–Poisson in velocity-Fourier coordinates.
//!
//! With g_hat_k(eta) = integral g_k(v) exp(-i eta v) dv and a normalized
//! Maxwellian equilibrium, the linearized nonzero-k mode equation is
//!
//!   partial_t g_hat_k - k partial_eta g_hat_k
//!     + (omega_p^2 / k) eta Phi(eta) g_hat_k(0,t) = 0,
//!
//! where Phi(eta) = exp(-thermal_speed^2 * eta^2 / 2). The advection speed in
//! eta is -k. The incoming perturbation is zero at the upstream boundary; the
//! outgoing characteristic leaves the opposite side and is never wrapped.
//!
//! This is a characteristic outflow prototype, not the exponential spectral
//! sponge. Scope is one nonzero spatial Fourier mode, linearized around a
//! fixed Maxwellian. Nonlinear spatial-mode convolution is not implemented.

use std::error::Error;
use std::fmt;

/// Configuration for one linearized spatial mode in transformed velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FourierOutflowConfig {
    /// Signed physical spatial wavenumber. It must be finite and nonzero.
    pub spatial_wave_number: f64,
    /// Symmetric transformed-velocity interval [-eta_max_abs, eta_max_abs].
    pub eta_max_abs: f64,
    /// Odd number of eta nodes so eta=0 is represented exactly.
    pub eta_nodes: usize,
    /// Maxwellian thermal speed.
    pub thermal_speed: f64,
    /// Squared plasma frequency in the chosen normalization.
    pub plasma_frequency_squared: f64,
}

impl FourierOutflowConfig {
    pub fn validate(self) -> Result<(), FourierOutflowError> {
        if !self.spatial_wave_number.is_finite() || self.spatial_wave_number == 0.0 {
            return Err(FourierOutflowError::InvalidSpatialWaveNumber);
        }
        if !self.eta_max_abs.is_finite() || self.eta_max_abs <= 0.0 {
            return Err(FourierOutflowError::InvalidEtaExtent);
        }
        if self.eta_nodes < 9 || self.eta_nodes % 2 == 0 {
            return Err(FourierOutflowError::InvalidEtaNodeCount);
        }
        if !self.thermal_speed.is_finite() || self.thermal_speed <= 0.0 {
            return Err(FourierOutflowError::InvalidThermalSpeed);
        }
        if !self.plasma_frequency_squared.is_finite() || self.plasma_frequency_squared <= 0.0 {
            return Err(FourierOutflowError::InvalidPlasmaFrequency);
        }
        Ok(())
    }

    pub fn eta_min(self) -> f64 {
        -self.eta_max_abs
    }

    pub fn eta_max(self) -> f64 {
        self.eta_max_abs
    }

    pub fn eta_spacing(self) -> f64 {
        2.0 * self.eta_max_abs / (self.eta_nodes - 1) as f64
    }

    pub fn eta(self, index: usize) -> Option<f64> {
        (index < self.eta_nodes).then(|| self.eta_min() + index as f64 * self.eta_spacing())
    }

    pub fn equilibrium_transform(self, eta: f64) -> f64 {
        (-0.5 * (self.thermal_speed * eta).powi(2)).exp()
    }

    fn zero_index(self) -> usize {
        self.eta_nodes / 2
    }
}

/// Complex Fourier coefficient used by the standalone transformed solver.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FourierOutflowValue {
    pub real: f64,
    pub imaginary: f64,
}

impl FourierOutflowValue {
    pub const ZERO: Self = Self {
        real: 0.0,
        imaginary: 0.0,
    };

    pub const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }

    pub fn scale(self, factor: f64) -> Self {
        Self {
            real: self.real * factor,
            imaginary: self.imaginary * factor,
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imaginary: self.imaginary + other.imaginary,
        }
    }

    pub fn magnitude_squared(self) -> f64 {
        self.real
            .mul_add(self.real, self.imaginary * self.imaginary)
    }

    pub fn amplitude(self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.imaginary.is_finite()
    }
}

/// Fail-closed errors for the linearized transformed outflow prototype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FourierOutflowError {
    InvalidSpatialWaveNumber,
    InvalidEtaExtent,
    InvalidEtaNodeCount,
    InvalidThermalSpeed,
    InvalidPlasmaFrequency,
    InvalidPerturbationAmplitude,
    InvalidSampleCount,
    InvalidTimeStep,
    NonFiniteInitialData,
    NonFiniteState,
    AllocationFailed,
}

impl fmt::Display for FourierOutflowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpatialWaveNumber => {
                write!(formatter, "spatial wavenumber must be finite and nonzero")
            }
            Self::InvalidEtaExtent => write!(formatter, "eta extent must be finite and positive"),
            Self::InvalidEtaNodeCount => {
                write!(formatter, "eta node count must be odd and at least nine")
            }
            Self::InvalidThermalSpeed => {
                write!(formatter, "thermal speed must be finite and positive")
            }
            Self::InvalidPlasmaFrequency => {
                write!(
                    formatter,
                    "squared plasma frequency must be finite and positive"
                )
            }
            Self::InvalidPerturbationAmplitude => {
                write!(formatter, "perturbation amplitude must be finite")
            }
            Self::InvalidSampleCount => {
                write!(formatter, "initial sample count does not match eta grid")
            }
            Self::InvalidTimeStep => {
                write!(formatter, "time step must be finite and positive")
            }
            Self::NonFiniteInitialData => {
                write!(formatter, "initial transformed distribution is non-finite")
            }
            Self::NonFiniteState => {
                write!(
                    formatter,
                    "Fourier-outflow update produced a non-finite state"
                )
            }
            Self::AllocationFailed => {
                write!(
                    formatter,
                    "unable to allocate Fourier-outflow candidate state"
                )
            }
        }
    }
}

impl Error for FourierOutflowError {}

/// Streaming-only diagnostic. The outflow estimate integrates the old state
/// in the characteristic slab leaving the transformed-coordinate domain.
/// The net L2 drop also includes interpolation error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FourierOutflowStreamingReport {
    pub duration: f64,
    pub l2_before: f64,
    pub l2_after: f64,
    pub estimated_outflow_l2: f64,
    pub net_l2_drop: f64,
    pub outflow_accounting_residual: f64,
}

/// Split-step diagnostic for the linearized transformed equation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FourierOutflowStepReport {
    pub duration: f64,
    pub time_before: f64,
    pub time_after: f64,
    pub l2_before: f64,
    pub l2_after: f64,
    pub streaming_net_l2_drop: f64,
    pub estimated_outflow_l2: f64,
    pub streaming_outflow_accounting_residual: f64,
    /// Signed change in transformed L2 caused by the linearized field source.
    pub source_l2_change: f64,
    pub electric_mode_before: FourierOutflowValue,
    pub electric_mode_after: FourierOutflowValue,
}

/// One spatial Fourier mode of the linearized Vlasov–Poisson system.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearizedMaxwellianFourierOutflow {
    config: FourierOutflowConfig,
    perturbation: Vec<FourierOutflowValue>,
    time: f64,
}

impl LinearizedMaxwellianFourierOutflow {
    /// Initialize g_hat_k(eta,0) = cosine_amplitude * Phi(eta).
    ///
    /// For a physical perturbation alpha*cos(k*x), the positive-k Fourier
    /// coefficient is alpha/2, so pass alpha/2 here.
    pub fn from_maxwellian_mode(
        config: FourierOutflowConfig,
        cosine_amplitude: f64,
    ) -> Result<Self, FourierOutflowError> {
        config.validate()?;
        if !cosine_amplitude.is_finite() {
            return Err(FourierOutflowError::InvalidPerturbationAmplitude);
        }
        let mut perturbation = Vec::new();
        perturbation
            .try_reserve_exact(config.eta_nodes)
            .map_err(|_| FourierOutflowError::AllocationFailed)?;
        for index in 0..config.eta_nodes {
            let eta = config.eta(index).expect("validated eta index");
            perturbation.push(FourierOutflowValue::new(
                cosine_amplitude * config.equilibrium_transform(eta),
                0.0,
            ));
        }
        Self::from_samples(config, perturbation)
    }

    /// Construct from explicit transformed samples for manufactured tests.
    pub fn from_samples(
        config: FourierOutflowConfig,
        perturbation: Vec<FourierOutflowValue>,
    ) -> Result<Self, FourierOutflowError> {
        config.validate()?;
        if perturbation.len() != config.eta_nodes {
            return Err(FourierOutflowError::InvalidSampleCount);
        }
        if perturbation.iter().any(|value| !value.is_finite()) {
            return Err(FourierOutflowError::NonFiniteInitialData);
        }
        Ok(Self {
            config,
            perturbation,
            time: 0.0,
        })
    }

    pub fn config(&self) -> FourierOutflowConfig {
        self.config
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn eta(&self, index: usize) -> Option<f64> {
        self.config.eta(index)
    }

    pub fn transformed_value(&self, index: usize) -> Option<FourierOutflowValue> {
        self.perturbation.get(index).copied()
    }

    /// Density Fourier coefficient is the eta=0 trace.
    pub fn density_mode(&self) -> FourierOutflowValue {
        self.perturbation[self.config.zero_index()]
    }

    /// Poisson convention: i*k*E_k = -g_hat_k(0), hence E_k = i*g_hat_k(0)/k.
    pub fn electric_mode_coefficient(&self) -> FourierOutflowValue {
        let density = self.density_mode();
        FourierOutflowValue::new(
            -density.imaginary / self.config.spatial_wave_number,
            density.real / self.config.spatial_wave_number,
        )
    }

    /// Transformed perturbation L2 integral using trapezoidal eta quadrature.
    pub fn transformed_l2_norm(&self) -> f64 {
        l2_integral(&self.perturbation, self.config.eta_spacing())
    }

    /// Advance only the characteristic advection with zero inflow and no wrap.
    pub fn advance_free_streaming(
        &mut self,
        duration: f64,
    ) -> Result<FourierOutflowStreamingReport, FourierOutflowError> {
        validate_duration(duration)?;
        let mut candidate = self.clone();
        let report = candidate.streaming_substep(duration)?;
        candidate.time += duration;
        if !candidate.time.is_finite() {
            return Err(FourierOutflowError::NonFiniteState);
        }
        *self = candidate;
        Ok(report)
    }

    /// Advance the linearized transformed equation with a symmetric
    /// characteristic / field-source / characteristic split.
    ///
    /// The source substep is exact for frozen g_hat_k(0): its eta=0 derivative
    /// is zero, so the density trace remains fixed during that substep.
    pub fn step(&mut self, duration: f64) -> Result<FourierOutflowStepReport, FourierOutflowError> {
        validate_duration(duration)?;
        let mut candidate = self.clone();
        let time_before = candidate.time;
        let l2_before = candidate.transformed_l2_norm();
        let electric_mode_before = candidate.electric_mode_coefficient();

        let first_stream = candidate.streaming_substep(0.5 * duration)?;
        let l2_before_source = candidate.transformed_l2_norm();
        candidate.source_substep(duration)?;
        let l2_after_source = candidate.transformed_l2_norm();
        let source_l2_change = l2_after_source - l2_before_source;
        let second_stream = candidate.streaming_substep(0.5 * duration)?;

        candidate.time += duration;
        if !candidate.time.is_finite() {
            return Err(FourierOutflowError::NonFiniteState);
        }
        let l2_after = candidate.transformed_l2_norm();
        let electric_mode_after = candidate.electric_mode_coefficient();
        if !l2_after.is_finite()
            || !source_l2_change.is_finite()
            || !electric_mode_after.is_finite()
        {
            return Err(FourierOutflowError::NonFiniteState);
        }

        let report = FourierOutflowStepReport {
            duration,
            time_before,
            time_after: candidate.time,
            l2_before,
            l2_after,
            streaming_net_l2_drop: first_stream.net_l2_drop + second_stream.net_l2_drop,
            estimated_outflow_l2: first_stream.estimated_outflow_l2
                + second_stream.estimated_outflow_l2,
            streaming_outflow_accounting_residual: first_stream.outflow_accounting_residual
                + second_stream.outflow_accounting_residual,
            source_l2_change,
            electric_mode_before,
            electric_mode_after,
        };
        *self = candidate;
        Ok(report)
    }

    fn source_substep(&mut self, duration: f64) -> Result<(), FourierOutflowError> {
        let density = self.density_mode();
        let k = self.config.spatial_wave_number;
        let omega_p_squared = self.config.plasma_frequency_squared;
        for index in 0..self.config.eta_nodes {
            let eta = self.config.eta(index).expect("validated eta index");
            let phi = self.config.equilibrium_transform(eta);
            let source_scale = -duration * omega_p_squared * eta * phi / k;
            let updated = self.perturbation[index].add(density.scale(source_scale));
            if !updated.is_finite() {
                return Err(FourierOutflowError::NonFiniteState);
            }
            self.perturbation[index] = updated;
        }
        Ok(())
    }

    fn streaming_substep(
        &mut self,
        duration: f64,
    ) -> Result<FourierOutflowStreamingReport, FourierOutflowError> {
        let config = self.config;
        let l2_before = self.transformed_l2_norm();
        let old = self.perturbation.clone();
        let estimated_outflow_l2 = estimate_outflow_l2(&old, config, duration);
        let mut next = Vec::new();
        next.try_reserve_exact(config.eta_nodes)
            .map_err(|_| FourierOutflowError::AllocationFailed)?;

        for index in 0..config.eta_nodes {
            let eta = config.eta(index).expect("validated eta index");
            // Characteristics satisfy d(eta)/dt = -k; the new value at eta
            // samples the old solution at eta + k*duration.
            let source_eta = eta + config.spatial_wave_number * duration;
            let value = if source_eta < config.eta_min() || source_eta > config.eta_max() {
                FourierOutflowValue::ZERO
            } else {
                interpolate_cubic(&old, source_eta, config)
            };
            if !value.is_finite() {
                return Err(FourierOutflowError::NonFiniteState);
            }
            next.push(value);
        }

        self.perturbation = next;
        let l2_after = self.transformed_l2_norm();
        if !l2_after.is_finite() {
            return Err(FourierOutflowError::NonFiniteState);
        }
        let net_l2_drop = l2_before - l2_after;
        Ok(FourierOutflowStreamingReport {
            duration,
            l2_before,
            l2_after,
            estimated_outflow_l2,
            net_l2_drop,
            outflow_accounting_residual: net_l2_drop - estimated_outflow_l2,
        })
    }
}

fn validate_duration(duration: f64) -> Result<(), FourierOutflowError> {
    if !duration.is_finite() || duration <= 0.0 {
        return Err(FourierOutflowError::InvalidTimeStep);
    }
    Ok(())
}

fn l2_integral(values: &[FourierOutflowValue], spacing: f64) -> f64 {
    let mut sum = 0.0;
    for (index, value) in values.iter().enumerate() {
        let weight = if index == 0 || index + 1 == values.len() {
            0.5
        } else {
            1.0
        };
        sum = value.magnitude_squared().mul_add(weight * spacing, sum);
    }
    sum
}

fn estimate_outflow_l2(
    values: &[FourierOutflowValue],
    config: FourierOutflowConfig,
    duration: f64,
) -> f64 {
    let k = config.spatial_wave_number;
    let threshold = if k > 0.0 {
        config.eta_min() + k * duration
    } else {
        config.eta_max() + k * duration
    };
    let spacing = config.eta_spacing();
    let mut sum = 0.0;
    for (index, value) in values.iter().enumerate() {
        let eta = config.eta(index).expect("validated eta index");
        let exits = if k > 0.0 {
            eta <= threshold
        } else {
            eta >= threshold
        };
        if exits {
            let weight = if index == 0 || index + 1 == values.len() {
                0.5
            } else {
                1.0
            };
            sum = value.magnitude_squared().mul_add(weight * spacing, sum);
        }
    }
    sum
}

fn interpolate_cubic(
    values: &[FourierOutflowValue],
    eta: f64,
    config: FourierOutflowConfig,
) -> FourierOutflowValue {
    let position = (eta - config.eta_min()) / config.eta_spacing();
    let base = position.floor() as isize;
    let maximum_start = (values.len() - 4) as isize;
    let start = (base - 1).clamp(0, maximum_start) as usize;
    let mut result = FourierOutflowValue::ZERO;

    for offset in 0..4 {
        let index = start + offset;
        let mut weight = 1.0;
        for other in 0..4 {
            if other == offset {
                continue;
            }
            let other_index = start + other;
            weight *= (position - other_index as f64) / (index as f64 - other_index as f64);
        }
        result = result.add(values[index].scale(weight));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(spatial_wave_number: f64) -> FourierOutflowConfig {
        FourierOutflowConfig {
            spatial_wave_number,
            eta_max_abs: 16.0,
            eta_nodes: 2049,
            thermal_speed: 1.0,
            plasma_frequency_squared: 1.0,
        }
    }

    #[test]
    fn free_streaming_matches_the_continuous_characteristic_oracle() {
        let amplitude = 5.0e-3;
        let duration = 0.3713;
        let mut state =
            LinearizedMaxwellianFourierOutflow::from_maxwellian_mode(config(1.0), amplitude)
                .unwrap();
        state.advance_free_streaming(duration).unwrap();

        let mut maximum_error = 0.0_f64;
        for index in 0..state.config.eta_nodes {
            let eta = state.eta(index).unwrap();
            let source_eta = eta + duration;
            if source_eta < state.config.eta_min() || source_eta > state.config.eta_max() {
                continue;
            }
            let expected =
                amplitude * (-0.5 * (state.config.thermal_speed * source_eta).powi(2)).exp();
            let observed = state.transformed_value(index).unwrap().real;
            maximum_error = maximum_error.max((observed - expected).abs());
        }
        assert!(
            maximum_error / amplitude < 1.0e-7,
            "characteristic relative error = {:e}",
            maximum_error / amplitude
        );
    }

    #[test]
    fn positive_k_outflow_removes_negative_eta_packet_without_periodic_wrap() {
        let config = FourierOutflowConfig {
            eta_max_abs: 4.0,
            eta_nodes: 1025,
            ..config(1.0)
        };
        let mut values = vec![FourierOutflowValue::ZERO; config.eta_nodes];
        for (index, value) in values.iter_mut().enumerate() {
            let eta = config.eta(index).unwrap();
            let envelope = (-0.5 * ((eta + 3.8) / 0.08).powi(2)).exp();
            *value = FourierOutflowValue::new(envelope, 0.0);
        }
        let mut state = LinearizedMaxwellianFourierOutflow::from_samples(config, values).unwrap();
        let before = state.transformed_l2_norm();
        let report = state.advance_free_streaming(0.5).unwrap();
        let after = state.transformed_l2_norm();

        assert!(report.estimated_outflow_l2 > 0.9 * before);
        assert!(
            after < 1.0e-3 * before,
            "remaining norm fraction = {:e}",
            after / before
        );
        assert_eq!(
            state.transformed_value(config.eta_nodes - 1),
            Some(FourierOutflowValue::ZERO)
        );
        assert_eq!(state.time(), 0.5);
    }

    #[test]
    fn negative_k_outflow_removes_positive_eta_packet_without_periodic_wrap() {
        let config = FourierOutflowConfig {
            eta_max_abs: 4.0,
            eta_nodes: 1025,
            ..config(-1.0)
        };
        let mut values = vec![FourierOutflowValue::ZERO; config.eta_nodes];
        for (index, value) in values.iter_mut().enumerate() {
            let eta = config.eta(index).unwrap();
            let envelope = (-0.5 * ((eta - 3.8) / 0.08).powi(2)).exp();
            *value = FourierOutflowValue::new(envelope, 0.0);
        }
        let mut state = LinearizedMaxwellianFourierOutflow::from_samples(config, values).unwrap();
        let before = state.transformed_l2_norm();
        let report = state.advance_free_streaming(0.5).unwrap();
        let after = state.transformed_l2_norm();

        assert!(report.estimated_outflow_l2 > 0.9 * before);
        assert!(after < 1.0e-3 * before, "remaining norm fraction = {:e}", after / before);
        assert_eq!(state.transformed_value(0), Some(FourierOutflowValue::ZERO));
    }

    #[test]
    fn poisson_field_uses_the_zero_eta_density_trace() {
        let amplitude = 1.0e-2;
        let state =
            LinearizedMaxwellianFourierOutflow::from_maxwellian_mode(config(2.0), amplitude)
                .unwrap();
        let density = state.density_mode();
        let electric = state.electric_mode_coefficient();
        assert!((density.real - amplitude).abs() < 1.0e-14);
        assert!(electric.real.abs() < 1.0e-14);
        assert!((electric.imaginary - amplitude / 2.0).abs() < 1.0e-14);
    }

    #[test]
    fn field_source_leaves_zero_eta_density_trace_unchanged() {
        let mut state =
            LinearizedMaxwellianFourierOutflow::from_maxwellian_mode(config(1.0), 1.0e-2).unwrap();
        let density_before = state.density_mode();
        let index = config(1.0).zero_index() + 5;
        let off_axis_before = state.transformed_value(index).unwrap();
        state.source_substep(0.05).unwrap();
        let density_after = state.density_mode();
        let off_axis_after = state.transformed_value(index).unwrap();

        assert_eq!(density_after, density_before);
        assert_ne!(off_axis_after, off_axis_before);
    }

    #[test]
    fn split_step_advances_time_and_reports_outflow_and_source_exchange() {
        let mut state =
            LinearizedMaxwellianFourierOutflow::from_maxwellian_mode(config(1.0), 1.0e-2)
                .unwrap();
        let before = state.electric_mode_coefficient();
        let report = state.step(0.02).unwrap();
        assert_eq!(report.time_before, 0.0);
        assert!((report.time_after - 0.02).abs() < 1.0e-15);
        assert_eq!(state.time(), 0.02);
        assert_eq!(report.electric_mode_before, before);
        assert!(report.l2_before.is_finite() && report.l2_after.is_finite());
    }

    #[test]
    fn invalid_steps_are_atomic() {
        let mut state =
            LinearizedMaxwellianFourierOutflow::from_maxwellian_mode(config(1.0), 1.0e-2)
                .unwrap();
        let before = state.clone();
        assert_eq!(state.step(0.0), Err(FourierOutflowError::InvalidTimeStep));
        assert_eq!(
            state.step(f64::NAN),
            Err(FourierOutflowError::InvalidTimeStep)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn invalid_configs_fail_closed() {
        assert_eq!(
            FourierOutflowConfig {
                spatial_wave_number: 0.0,
                ..config(1.0)
            }
            .validate(),
            Err(FourierOutflowError::InvalidSpatialWaveNumber)
        );
        assert_eq!(
            FourierOutflowConfig {
                eta_nodes: 10,
                ..config(1.0)
            }
            .validate(),
            Err(FourierOutflowError::InvalidEtaNodeCount)
        );
    }
}
