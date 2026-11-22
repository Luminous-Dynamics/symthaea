//! Reversible storage for phase-mixed velocity-Fourier structure.
//!
//! A packet stores signed complex velocity-Fourier coefficients. Its eta
//! support is non-periodic: free streaming translates each spatial Fourier
//! mode along eta and expands the stored support rather than wrapping it.
//!
//! This module is a discrete transfer and free-streaming primitive. It does
//! not yet close self-consistent field feedback while a packet is vaulted.

use std::f64::consts::TAU;
use std::fmt;

use super::{Complex, SolverError, VlasovPoisson1d1v, fft, signed_mode};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ComplexCoefficient {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexCoefficient {
    fn from_internal(value: Complex) -> Self {
        Self { real: value.re, imaginary: value.im }
    }

    fn into_internal(self) -> Complex {
        Complex::new(self.real, self.imaginary)
    }

    pub fn amplitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryReconstruction {
    pub values: Vec<f64>,
    pub maximum_imaginary_residual: f64,
    pub maximum_absolute_real_value: f64,
    pub relative_imaginary_residual: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryRestoreReceipt {
    pub packet_id: u64,
    pub restored_casimir: f64,
    pub active_casimir_before: f64,
    pub active_casimir_after: f64,
    pub casimir_closure_residual: f64,
    pub capture_reconstruction_error: f64,
    pub relative_imaginary_residual: f64,
    pub mass_delta: f64,
    pub minimum_distribution_after: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum KineticMemoryError {
    Solver(SolverError),
    InvalidSpatialGrid,
    InvalidDomainLength,
    InvalidVelocityGrid,
    InvalidEtaGrid,
    InvalidTolerance,
    InvalidPacketId,
    ShapeMismatch { expected: usize, actual: usize },
    NonFiniteImpulse { index: usize },
    NonFinitePhase { index: usize },
    NonGridAlignedStreaming { spatial_mode: isize, requested_bins: f64 },
    StreamingShiftOverflow,
    GridSizeOverflow,
    AllocationFailed { elements: usize },
    NonFiniteTransform,
    CaptureReconstructionExceeded { relative_error: f64, tolerance: f64 },
    CaptureCasimirClosureExceeded { relative_error: f64, tolerance: f64 },
    CaptureFieldChanged { relative_error: f64, tolerance: f64 },
    IncompatibleSolverGrid,
    OutsideRestorationBand { relative_amplitude: f64, tolerance: f64 },
    ReconstructionImaginaryResidual { relative_residual: f64, tolerance: f64 },
    AlreadyConsumed,
}

impl fmt::Display for KineticMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Solver(error) => write!(f, "solver failure: {error}"),
            Self::InvalidSpatialGrid => write!(f, "spatial grid must be a power of two >= 4"),
            Self::InvalidDomainLength => write!(f, "spatial domain length is invalid"),
            Self::InvalidVelocityGrid => write!(f, "velocity grid metadata is invalid"),
            Self::InvalidEtaGrid => write!(f, "velocity-Fourier grid metadata is invalid"),
            Self::InvalidTolerance => write!(f, "tolerance must be finite and positive"),
            Self::InvalidPacketId => write!(f, "packet ID zero is reserved"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "expected {expected} values, received {actual}")
            }
            Self::NonFiniteImpulse { index } => write!(f, "impulse {index} is non-finite"),
            Self::NonFinitePhase { index } => write!(f, "phase {index} is non-finite"),
            Self::NonGridAlignedStreaming { spatial_mode, requested_bins } => {
                write!(f, "mode {spatial_mode} requests non-grid eta shift {requested_bins}")
            }
            Self::StreamingShiftOverflow => write!(f, "streaming shift exceeds integer range"),
            Self::GridSizeOverflow => write!(f, "memory grid size overflows usize"),
            Self::AllocationFailed { elements } => {
                write!(f, "unable to allocate {elements} memory values")
            }
            Self::NonFiniteTransform => write!(f, "memory transform produced non-finite values"),
            Self::CaptureReconstructionExceeded { relative_error, tolerance } => {
                write!(f, "capture reconstruction error {relative_error:e} exceeds {tolerance:e}")
            }
            Self::CaptureCasimirClosureExceeded { relative_error, tolerance } => {
                write!(f, "capture C2 closure error {relative_error:e} exceeds {tolerance:e}")
            }
            Self::CaptureFieldChanged { relative_error, tolerance } => {
                write!(f, "capture changed field by {relative_error:e}, tolerance {tolerance:e}")
            }
            Self::IncompatibleSolverGrid => write!(f, "packet and active solver grids differ"),
            Self::OutsideRestorationBand { relative_amplitude, tolerance } => {
                write!(f, "out-of-band amplitude {relative_amplitude:e} exceeds {tolerance:e}")
            }
            Self::ReconstructionImaginaryResidual { relative_residual, tolerance } => {
                write!(f, "imaginary residual {relative_residual:e} exceeds {tolerance:e}")
            }
            Self::AlreadyConsumed => write!(f, "packet has already been restored"),
        }
    }
}

impl std::error::Error for KineticMemoryError {}

/// Row-major x/eta packet. Transform convention:
/// F(x,eta) = integral f(x,v) exp(-i eta (v-v_origin)) dv.
#[derive(Debug, PartialEq)]
pub struct KineticMemoryPacket {
    packet_id: u64,
    nx: usize,
    domain_length: f64,
    velocity_origin: f64,
    reference_nv: usize,
    reference_dv: f64,
    maximum_restore_eta: f64,
    eta_min: f64,
    eta_spacing: f64,
    eta_points: usize,
    time: f64,
    coefficients: Vec<ComplexCoefficient>,
    captured_casimir: f64,
    capture_reconstruction_error: f64,
    consumed: bool,
}

impl KineticMemoryPacket {
    /// Escrow all velocity-Fourier modes |m| >= minimum_absolute_mode.
    /// The zero mode stays active, so capture leaves charge density and the
    /// Poisson source unchanged. The transfer commits only after the original
    /// sampled distribution, C2 and electric field are independently checked.
    pub fn capture_high_velocity_modes(
        solver: &mut VlasovPoisson1d1v,
        packet_id: u64,
        minimum_absolute_mode: usize,
        relative_tolerance: f64,
    ) -> Result<Self, KineticMemoryError> {
        if packet_id == 0 {
            return Err(KineticMemoryError::InvalidPacketId);
        }
        validate_tolerance(relative_tolerance)?;
        let config = solver.config;
        config.validate().map_err(KineticMemoryError::Solver)?;
        if minimum_absolute_mode == 0 || minimum_absolute_mode > config.nv / 2 {
            return Err(KineticMemoryError::InvalidVelocityGrid);
        }

        let before_budget = solver.quadratic_casimir_budget().map_err(KineticMemoryError::Solver)?;
        let original_distribution = solver.distribution.clone();
        let original_field = solver.electric_field.clone();
        let expected = config.nx.checked_mul(config.nv).ok_or(KineticMemoryError::GridSizeOverflow)?;

        let mut active_candidate = Vec::new();
        active_candidate.try_reserve_exact(expected)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: expected })?;
        let mut packet_coefficients = Vec::new();
        packet_coefficients.try_reserve_exact(expected)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: expected })?;
        let mut line = Vec::new();
        line.try_reserve_exact(config.nv)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: config.nv })?;
        line.resize(config.nv, Complex::ZERO);

        let mut maximum_imaginary = 0.0_f64;
        let mut maximum_real = 0.0_f64;
        for ix in 0..config.nx {
            let row_start = ix * config.nv;
            for (iv, value) in line.iter_mut().enumerate() {
                *value = Complex::new(original_distribution[row_start + iv], 0.0);
            }
            fft(&mut line, false);

            // Store in fftshift order: signed eta modes -Nv/2 .. Nv/2-1.
            for eta_index in 0..config.nv {
                let signed_eta_mode = eta_index as isize - (config.nv / 2) as isize;
                let raw_mode = if signed_eta_mode < 0 {
                    (config.nv as isize + signed_eta_mode) as usize
                } else {
                    signed_eta_mode as usize
                };
                if signed_eta_mode.unsigned_abs() >= minimum_absolute_mode {
                    packet_coefficients.push(ComplexCoefficient::from_internal(
                        line[raw_mode].scale(config.dv()),
                    ));
                    line[raw_mode] = Complex::ZERO;
                } else {
                    packet_coefficients.push(ComplexCoefficient::default());
                }
            }

            fft(&mut line, true);
            for value in &line {
                maximum_imaginary = maximum_imaginary.max(value.im.abs());
                maximum_real = maximum_real.max(value.re.abs());
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(KineticMemoryError::NonFiniteTransform);
                }
                active_candidate.push(value.re);
            }
        }

        let active_imaginary_relative = maximum_imaginary / maximum_real.max(f64::MIN_POSITIVE);
        if active_imaginary_relative > relative_tolerance {
            return Err(KineticMemoryError::CaptureReconstructionExceeded {
                relative_error: active_imaginary_relative,
                tolerance: relative_tolerance,
            });
        }

        let eta_spacing = TAU / (config.nv as f64 * config.dv());
        let eta_min = -(config.nv / 2) as f64 * eta_spacing;
        let maximum_restore_eta = std::f64::consts::PI / config.dv();
        let captured_casimir = packet_casimir(
            config.nx,
            config.domain_length,
            eta_spacing,
            &packet_coefficients,
        );

        let mut packet = Self {
            packet_id,
            nx: config.nx,
            domain_length: config.domain_length,
            velocity_origin: config.velocity_min,
            reference_nv: config.nv,
            reference_dv: config.dv(),
            maximum_restore_eta,
            eta_min,
            eta_spacing,
            eta_points: config.nv,
            time: solver.time,
            coefficients: packet_coefficients,
            captured_casimir,
            capture_reconstruction_error: 0.0,
            consumed: false,
        };

        let velocities: Vec<f64> = (0..config.nv).map(|iv| config.velocity(iv)).collect();
        let reconstruction = packet.reconstruct_at_velocities(&velocities)?;
        let mut maximum_error = 0.0_f64;
        let mut maximum_reference = 0.0_f64;
        for index in 0..expected {
            let reconstructed = active_candidate[index] + reconstruction.values[index];
            maximum_error = maximum_error.max((reconstructed - original_distribution[index]).abs());
            maximum_reference = maximum_reference.max(original_distribution[index].abs());
        }
        let relative_error = maximum_error / maximum_reference.max(f64::MIN_POSITIVE);
        packet.capture_reconstruction_error = relative_error;
        if relative_error > relative_tolerance {
            return Err(KineticMemoryError::CaptureReconstructionExceeded {
                relative_error,
                tolerance: relative_tolerance,
            });
        }

        let active_casimir = distribution_casimir(config, &active_candidate);
        let c2_error = (active_casimir + captured_casimir - before_budget.total_casimir).abs()
            / before_budget.total_casimir.abs().max(f64::MIN_POSITIVE);
        if c2_error > relative_tolerance {
            return Err(KineticMemoryError::CaptureCasimirClosureExceeded {
                relative_error: c2_error,
                tolerance: relative_tolerance,
            });
        }

        solver.distribution = active_candidate;
        if let Err(error) = solver.refresh_electric_field().and_then(|()| solver.ensure_finite_state()) {
            solver.distribution = original_distribution;
            solver.electric_field = original_field;
            return Err(KineticMemoryError::Solver(error));
        }

        let field_scale = original_field.iter().map(|value| value.abs()).fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let field_delta = solver.electric_field.iter().zip(original_field.iter())
            .map(|(after, before)| (after - before).abs())
            .fold(0.0_f64, f64::max);
        let relative_field_delta = field_delta / field_scale;
        if relative_field_delta > relative_tolerance {
            solver.distribution = original_distribution;
            solver.electric_field = original_field;
            return Err(KineticMemoryError::CaptureFieldChanged {
                relative_error: relative_field_delta,
                tolerance: relative_tolerance,
            });
        }

        let active_after = match solver.quadratic_casimir_budget() {
            Ok(budget) => budget.total_casimir,
            Err(error) => {
                solver.distribution = original_distribution;
                solver.electric_field = original_field;
                return Err(KineticMemoryError::Solver(error));
            }
        };
        let actual_c2_error = (active_after + packet.captured_casimir - before_budget.total_casimir).abs()
            / before_budget.total_casimir.abs().max(f64::MIN_POSITIVE);
        if actual_c2_error > relative_tolerance {
            solver.distribution = original_distribution;
            solver.electric_field = original_field;
            return Err(KineticMemoryError::CaptureCasimirClosureExceeded {
                relative_error: actual_c2_error,
                tolerance: relative_tolerance,
            });
        }

        Ok(packet)
    }

    pub fn packet_id(&self) -> u64 { self.packet_id }
    pub fn time(&self) -> f64 { self.time }
    pub fn spatial_points(&self) -> usize { self.nx }
    pub fn eta_points(&self) -> usize { self.eta_points }
    pub fn eta_min(&self) -> f64 { self.eta_min }
    pub fn eta_spacing(&self) -> f64 { self.eta_spacing }
    pub fn eta_max(&self) -> f64 {
        self.eta_min + self.eta_points.saturating_sub(1) as f64 * self.eta_spacing
    }
    pub fn is_consumed(&self) -> bool { self.consumed }
    pub fn capture_reconstruction_error(&self) -> f64 { self.capture_reconstruction_error }

    pub fn coefficient(&self, x_index: usize, eta_index: usize) -> Option<ComplexCoefficient> {
        if x_index >= self.nx || eta_index >= self.eta_points {
            return None;
        }
        self.coefficients.get(x_index * self.eta_points + eta_index).copied()
    }

    /// Discrete Parseval C2: dx*d_eta/(2*pi) times the squared coefficient sum.
    pub fn quadratic_casimir(&self) -> f64 {
        packet_casimir(self.nx, self.domain_length, self.eta_spacing, &self.coefficients)
    }

    /// Free-stream each spatial Fourier mode as eta -> eta-k*dt.
    /// Non-grid-aligned shifts return an error without changing the packet.
    pub fn advance_free_streaming(&mut self, dt: f64) -> Result<(), KineticMemoryError> {
        self.ensure_unconsumed()?;
        if !dt.is_finite() || !(self.time + dt).is_finite() {
            return Err(KineticMemoryError::NonFiniteTransform);
        }
        self.translate_free_streaming(dt)?;
        self.time += dt;
        Ok(())
    }

    /// Instantaneous velocity translation: F(x,eta) <- exp(-i eta delta_v(x)) F.
    pub fn apply_velocity_impulse(&mut self, impulses: &[f64]) -> Result<(), KineticMemoryError> {
        self.ensure_unconsumed()?;
        if impulses.len() != self.nx {
            return Err(KineticMemoryError::ShapeMismatch {
                expected: self.nx,
                actual: impulses.len(),
            });
        }
        for (index, impulse) in impulses.iter().copied().enumerate() {
            if !impulse.is_finite() {
                return Err(KineticMemoryError::NonFiniteImpulse { index });
            }
        }

        let mut candidate = Vec::new();
        candidate.try_reserve_exact(self.coefficients.len())
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: self.coefficients.len() })?;
        for (x_index, impulse) in impulses.iter().copied().enumerate() {
            for eta_index in 0..self.eta_points {
                let index = x_index * self.eta_points + eta_index;
                let eta = self.eta_min + eta_index as f64 * self.eta_spacing;
                let phase = -eta * impulse;
                if !phase.is_finite() {
                    return Err(KineticMemoryError::NonFinitePhase { index });
                }
                let value = self.coefficients[index].into_internal().mul(Complex::cis(phase));
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(KineticMemoryError::NonFiniteTransform);
                }
                candidate.push(ComplexCoefficient::from_internal(value));
            }
        }
        self.coefficients = candidate;
        Ok(())
    }

    pub fn reconstruct_at_velocities(
        &self,
        velocities: &[f64],
    ) -> Result<MemoryReconstruction, KineticMemoryError> {
        self.ensure_unconsumed()?;
        if velocities.iter().any(|velocity| !velocity.is_finite()) {
            return Err(KineticMemoryError::InvalidVelocityGrid);
        }
        let count = self.nx.checked_mul(velocities.len()).ok_or(KineticMemoryError::GridSizeOverflow)?;
        let mut values = Vec::new();
        values.try_reserve_exact(count)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: count })?;
        let weight = self.eta_spacing / TAU;
        let mut maximum_imaginary = 0.0_f64;
        let mut maximum_real = 0.0_f64;
        for x_index in 0..self.nx {
            for velocity in velocities.iter().copied() {
                let mut sum = Complex::ZERO;
                for eta_index in 0..self.eta_points {
                    let eta = self.eta_min + eta_index as f64 * self.eta_spacing;
                    let phase = eta * (velocity - self.velocity_origin);
                    if !phase.is_finite() {
                        return Err(KineticMemoryError::NonFiniteTransform);
                    }
                    let coefficient = self.coefficients[x_index * self.eta_points + eta_index]
                        .into_internal();
                    sum = sum.add(coefficient.mul(Complex::cis(phase)));
                }
                let result = sum.scale(weight);
                if !result.re.is_finite() || !result.im.is_finite() {
                    return Err(KineticMemoryError::NonFiniteTransform);
                }
                maximum_imaginary = maximum_imaginary.max(result.im.abs());
                maximum_real = maximum_real.max(result.re.abs());
                values.push(result.re);
            }
        }
        Ok(MemoryReconstruction {
            values,
            maximum_imaginary_residual: maximum_imaginary,
            maximum_absolute_real_value: maximum_real,
            relative_imaginary_residual: maximum_imaginary / maximum_real.max(f64::MIN_POSITIVE),
        })
    }

    /// Restore a packet once. Material content outside the active solver's
    /// velocity Nyquist band blocks restoration rather than silently aliasing.
    pub fn restore_into_solver(
        &mut self,
        solver: &mut VlasovPoisson1d1v,
        relative_tolerance: f64,
    ) -> Result<MemoryRestoreReceipt, KineticMemoryError> {
        self.ensure_unconsumed()?;
        validate_tolerance(relative_tolerance)?;
        let config = solver.config;
        if config.nx != self.nx
            || config.nv != self.reference_nv
            || (config.domain_length - self.domain_length).abs() > relative_tolerance * self.domain_length.abs().max(1.0)
            || (config.velocity_min - self.velocity_origin).abs() > relative_tolerance * self.velocity_origin.abs().max(1.0)
            || (config.dv() - self.reference_dv).abs() > relative_tolerance * self.reference_dv.abs().max(1.0)
        {
            return Err(KineticMemoryError::IncompatibleSolverGrid);
        }

        let amplitude_scale = self.coefficients.iter()
            .map(|value| value.amplitude())
            .fold(0.0_f64, f64::max)
            .max(f64::MIN_POSITIVE);
        let mut maximum_out_of_band = 0.0_f64;
        for eta_index in 0..self.eta_points {
            let eta = self.eta_min + eta_index as f64 * self.eta_spacing;
            if eta.abs() > self.maximum_restore_eta * (1.0 + relative_tolerance) {
                for x_index in 0..self.nx {
                    maximum_out_of_band = maximum_out_of_band.max(
                        self.coefficients[x_index * self.eta_points + eta_index].amplitude()
                    );
                }
            }
        }
        let relative_out_of_band = maximum_out_of_band / amplitude_scale;
        if relative_out_of_band > relative_tolerance {
            return Err(KineticMemoryError::OutsideRestorationBand {
                relative_amplitude: relative_out_of_band,
                tolerance: relative_tolerance,
            });
        }

        let velocities: Vec<f64> = (0..config.nv).map(|iv| config.velocity(iv)).collect();
        let reconstruction = self.reconstruct_at_velocities(&velocities)?;
        if reconstruction.relative_imaginary_residual > relative_tolerance {
            return Err(KineticMemoryError::ReconstructionImaginaryResidual {
                relative_residual: reconstruction.relative_imaginary_residual,
                tolerance: relative_tolerance,
            });
        }

        let before_budget = solver.quadratic_casimir_budget().map_err(KineticMemoryError::Solver)?;
        let before_diagnostics = solver.diagnostics().map_err(KineticMemoryError::Solver)?;
        let previous_distribution = solver.distribution.clone();
        let previous_field = solver.electric_field.clone();
        let mut candidate = Vec::new();
        candidate.try_reserve_exact(previous_distribution.len())
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: previous_distribution.len() })?;
        for (active, contribution) in previous_distribution.iter().copied()
            .zip(reconstruction.values.iter().copied())
        {
            let value = active + contribution;
            if !value.is_finite() {
                return Err(KineticMemoryError::NonFiniteTransform);
            }
            candidate.push(value);
        }

        solver.distribution = candidate;
        if let Err(error) = solver.refresh_electric_field().and_then(|()| solver.ensure_finite_state()) {
            solver.distribution = previous_distribution;
            solver.electric_field = previous_field;
            return Err(KineticMemoryError::Solver(error));
        }
        let after_budget = match solver.quadratic_casimir_budget() {
            Ok(budget) => budget,
            Err(error) => {
                solver.distribution = previous_distribution;
                solver.electric_field = previous_field;
                return Err(KineticMemoryError::Solver(error));
            }
        };
        let after_diagnostics = match solver.diagnostics() {
            Ok(diagnostics) => diagnostics,
            Err(error) => {
                solver.distribution = previous_distribution;
                solver.electric_field = previous_field;
                return Err(KineticMemoryError::Solver(error));
            }
        };

        let packet_casimir = self.quadratic_casimir();
        let receipt = MemoryRestoreReceipt {
            packet_id: self.packet_id,
            restored_casimir: packet_casimir,
            active_casimir_before: before_budget.total_casimir,
            active_casimir_after: after_budget.total_casimir,
            casimir_closure_residual: after_budget.total_casimir - before_budget.total_casimir - packet_casimir,
            capture_reconstruction_error: self.capture_reconstruction_error,
            relative_imaginary_residual: reconstruction.relative_imaginary_residual,
            mass_delta: after_diagnostics.total_mass - before_diagnostics.total_mass,
            minimum_distribution_after: after_diagnostics.minimum_distribution,
        };
        self.consumed = true;
        Ok(receipt)
    }

    fn ensure_unconsumed(&self) -> Result<(), KineticMemoryError> {
        if self.consumed { Err(KineticMemoryError::AlreadyConsumed) } else { Ok(()) }
    }

    fn translate_free_streaming(&mut self, dt: f64) -> Result<(), KineticMemoryError> {
        if dt == 0.0 {
            return Ok(());
        }
        let spectral_count = self.nx.checked_mul(self.eta_points).ok_or(KineticMemoryError::GridSizeOverflow)?;
        let mut spectral = Vec::new();
        spectral.try_reserve_exact(spectral_count)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: spectral_count })?;
        spectral.resize(spectral_count, Complex::ZERO);
        let mut line = Vec::new();
        line.try_reserve_exact(self.nx)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: self.nx })?;
        line.resize(self.nx, Complex::ZERO);

        for eta_index in 0..self.eta_points {
            for (x_index, value) in line.iter_mut().enumerate() {
                *value = self.coefficients[x_index * self.eta_points + eta_index].into_internal();
            }
            fft(&mut line, false);
            for (mode, value) in line.iter().copied().enumerate() {
                spectral[mode * self.eta_points + eta_index] = value;
            }
        }

        let mut shifts = Vec::new();
        shifts.try_reserve_exact(self.nx)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: self.nx })?;
        let mut min_shift = i64::MAX;
        let mut max_shift = i64::MIN;
        for mode in 0..self.nx {
            let signed = signed_mode(mode, self.nx);
            let k = TAU * signed as f64 / self.domain_length;
            let requested = k * dt / self.eta_spacing;
            if !requested.is_finite() || requested.abs() > (i64::MAX / 4) as f64 {
                return Err(KineticMemoryError::StreamingShiftOverflow);
            }
            let rounded = requested.round();
            if (requested - rounded).abs() > 1.0e-10 * requested.abs().max(1.0) {
                return Err(KineticMemoryError::NonGridAlignedStreaming {
                    spatial_mode: signed,
                    requested_bins: requested,
                });
            }
            let shift = rounded as i64;
            shifts.push(shift);
            min_shift = min_shift.min(shift);
            max_shift = max_shift.max(shift);
        }

        let span = max_shift.checked_sub(min_shift).ok_or(KineticMemoryError::StreamingShiftOverflow)?;
        let span = usize::try_from(span).map_err(|_| KineticMemoryError::StreamingShiftOverflow)?;
        let new_eta_points = self.eta_points.checked_add(span).ok_or(KineticMemoryError::GridSizeOverflow)?;
        let new_count = self.nx.checked_mul(new_eta_points).ok_or(KineticMemoryError::GridSizeOverflow)?;
        let mut shifted = Vec::new();
        shifted.try_reserve_exact(new_count)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: new_count })?;
        shifted.resize(new_count, Complex::ZERO);

        for mode in 0..self.nx {
            let signed = signed_mode(mode, self.nx);
            let k = TAU * signed as f64 / self.domain_length;
            let origin_phase = -k * self.velocity_origin * dt;
            if !origin_phase.is_finite() {
                return Err(KineticMemoryError::NonFiniteTransform);
            }
            let origin_factor = Complex::cis(origin_phase);
            for eta_index in 0..self.eta_points {
                let new_index = (eta_index as i64).checked_sub(shifts[mode])
                    .and_then(|index| index.checked_add(max_shift))
                    .ok_or(KineticMemoryError::StreamingShiftOverflow)?;
                let new_index = usize::try_from(new_index)
                    .map_err(|_| KineticMemoryError::StreamingShiftOverflow)?;
                shifted[mode * new_eta_points + new_index] =
                    spectral[mode * self.eta_points + eta_index].mul(origin_factor);
            }
        }

        let new_eta_min = self.eta_min - max_shift as f64 * self.eta_spacing;
        if !new_eta_min.is_finite() {
            return Err(KineticMemoryError::NonFiniteTransform);
        }
        let mut candidate = Vec::new();
        candidate.try_reserve_exact(new_count)
            .map_err(|_| KineticMemoryError::AllocationFailed { elements: new_count })?;
        candidate.resize(new_count, ComplexCoefficient::default());
        for eta_index in 0..new_eta_points {
            for (mode, value) in line.iter_mut().enumerate() {
                *value = shifted[mode * new_eta_points + eta_index];
            }
            fft(&mut line, true);
            for (x_index, value) in line.iter().copied().enumerate() {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(KineticMemoryError::NonFiniteTransform);
                }
                candidate[x_index * new_eta_points + eta_index] = ComplexCoefficient::from_internal(value);
            }
        }

        self.coefficients = candidate;
        self.eta_min = new_eta_min;
        self.eta_points = new_eta_points;
        Ok(())
    }
}

fn validate_tolerance(tolerance: f64) -> Result<(), KineticMemoryError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(KineticMemoryError::InvalidTolerance);
    }
    Ok(())
}

fn distribution_casimir(config: super::VlasovPoissonConfig, distribution: &[f64]) -> f64 {
    let weight = config.dx() * config.dv();
    distribution.iter().map(|value| value * value * weight).sum()
}

fn packet_casimir(nx: usize, domain_length: f64, eta_spacing: f64, values: &[ComplexCoefficient]) -> f64 {
    let weight = domain_length / nx as f64 * eta_spacing / TAU;
    values.iter().map(|value| {
        value.real.mul_add(value.real, value.imaginary * value.imaginary) * weight
    }).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VlasovPoissonConfig, maxwellian_perturbation};

    fn config() -> VlasovPoissonConfig {
        VlasovPoissonConfig {
            nx: 16,
            nv: 64,
            domain_length: 2.0 * TAU,
            velocity_min: -6.0,
            velocity_max: 6.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        }
    }

    fn solver() -> VlasovPoisson1d1v {
        let config = config();
        let initial = maxwellian_perturbation(config, 0.03, 1, 1.0).unwrap();
        VlasovPoisson1d1v::new(config, initial).unwrap()
    }

    #[test]
    fn capture_and_restore_preserves_distribution_c2_and_field() {
        let mut solver = solver();
        let original = solver.distribution().to_vec();
        let field_energy_before = solver.diagnostics().unwrap().field_energy;
        let c2_before = solver.quadratic_casimir_budget().unwrap().total_casimir;
        let mut packet = KineticMemoryPacket::capture_high_velocity_modes(&mut solver, 11, 8, 1.0e-10).unwrap();
        assert!(packet.quadratic_casimir() > 0.0);
        let active_c2 = solver.quadratic_casimir_budget().unwrap().total_casimir;
        assert!((active_c2 + packet.quadratic_casimir() - c2_before).abs() < 1.0e-10 * c2_before);
        assert!((solver.diagnostics().unwrap().field_energy - field_energy_before).abs() < 1.0e-10);

        let receipt = packet.restore_into_solver(&mut solver, 1.0e-10).unwrap();
        assert_eq!(receipt.packet_id, 11);
        assert!(packet.is_consumed());
        assert!(receipt.capture_reconstruction_error < 1.0e-10);
        assert!(receipt.casimir_closure_residual.abs() < 1.0e-10 * c2_before);
        for (actual, expected) in solver.distribution().iter().zip(original.iter()) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        assert!(matches!(packet.restore_into_solver(&mut solver, 1.0e-10), Err(KineticMemoryError::AlreadyConsumed)));
    }

    #[test]
    fn free_streaming_expands_eta_support_without_periodic_wraparound() {
        let mut solver = solver();
        let mut packet = KineticMemoryPacket::capture_high_velocity_modes(&mut solver, 12, 8, 1.0e-10).unwrap();
        let old_points = packet.eta_points();
        let old_c2 = packet.quadratic_casimir();
        let old_min = packet.eta_min();
        // For Lx=2pi and deta=2pi/12=pi/6, dt=pi/6 shifts mode k=1 by one bin.
        packet.advance_free_streaming(std::f64::consts::PI / 6.0).unwrap();
        assert!(packet.eta_points() > old_points);
        assert!(packet.eta_min() < old_min);
        assert!((packet.quadratic_casimir() - old_c2).abs() < 1.0e-10);
        assert!((packet.time() - std::f64::consts::PI / 6.0).abs() < 1.0e-14);
    }

    #[test]
    fn velocity_impulse_preserves_packet_c2_and_is_reversible() {
        let mut solver = solver();
        let mut packet = KineticMemoryPacket::capture_high_velocity_modes(&mut solver, 13, 6, 1.0e-10).unwrap();
        let before = packet.coefficients.clone();
        let c2_before = packet.quadratic_casimir();
        let config = config();
        let impulse: Vec<f64> = (0..config.nx)
            .map(|ix| 0.03 * (TAU * config.x(ix) / config.domain_length).sin())
            .collect();
        packet.apply_velocity_impulse(&impulse).unwrap();
        assert!((packet.quadratic_casimir() - c2_before).abs() < 1.0e-10);
        let inverse: Vec<f64> = impulse.iter().map(|value| -*value).collect();
        packet.apply_velocity_impulse(&inverse).unwrap();
        for (actual, expected) in packet.coefficients.iter().zip(before.iter()) {
            assert!((actual.real - expected.real).abs() < 1.0e-10);
            assert!((actual.imaginary - expected.imaginary).abs() < 1.0e-10);
        }
    }

    #[test]
    fn non_grid_aligned_streaming_fails_without_mutation() {
        let mut solver = solver();
        let mut packet = KineticMemoryPacket::capture_high_velocity_modes(&mut solver, 14, 8, 1.0e-10).unwrap();
        let before = packet.coefficients.clone();
        let eta_min = packet.eta_min();
        assert!(matches!(
            packet.advance_free_streaming(0.013),
            Err(KineticMemoryError::NonGridAlignedStreaming { .. })
        ));
        assert_eq!(packet.coefficients, before);
        assert_eq!(packet.eta_min(), eta_min);
    }

    #[test]
    fn invalid_capture_inputs_leave_solver_unchanged() {
        let mut solver = solver();
        let before = solver.distribution().to_vec();
        assert!(KineticMemoryPacket::capture_high_velocity_modes(&mut solver, 15, 0, 1.0e-10).is_err());
        assert!(KineticMemoryPacket::capture_high_velocity_modes(&mut solver, 15, 8, f64::NAN).is_err());
        assert_eq!(solver.distribution(), before.as_slice());
    }
}
