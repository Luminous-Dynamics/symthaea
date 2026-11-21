//! Independent periodic 1D1V collisionless Vlasov-Poisson reference solver.
//!
//! Distribution-space Strang splitting with Fourier translations in periodic
//! x and a finite periodic numerical velocity interval. This is a verification
//! prototype, not a production PIC solver or a certified continuum solution.
//!
//! Normalized convention:
//!   rho(x) = n_background - integral f(x,v) dv
//!   dE/dx = rho / epsilon_0
//!   dv/dt = (q/m) E
//! The common electron convention uses q/m = -1 and n_background = 1.

#![forbid(unsafe_code)]

pub mod echo_analysis;

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// Frozen grid and normalized electrostatic parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VlasovPoissonConfig {
    pub nx: usize,
    pub nv: usize,
    pub domain_length: f64,
    pub velocity_min: f64,
    pub velocity_max: f64,
    pub charge_to_mass: f64,
    pub permittivity: f64,
    pub background_density: f64,
}

impl VlasovPoissonConfig {
    pub fn validate(self) -> Result<(), SolverError> {
        if self.nx < 4 || !self.nx.is_power_of_two() {
            return Err(SolverError::InvalidGrid {
                axis: "x",
                value: self.nx,
            });
        }
        if self.nv < 4 || !self.nv.is_power_of_two() {
            return Err(SolverError::InvalidGrid {
                axis: "v",
                value: self.nv,
            });
        }
        if self.nx.checked_mul(self.nv).is_none() {
            return Err(SolverError::GridSizeOverflow);
        }
        if !self.domain_length.is_finite() || self.domain_length <= 0.0 {
            return Err(SolverError::InvalidDomainLength);
        }
        if !self.velocity_min.is_finite()
            || !self.velocity_max.is_finite()
            || self.velocity_max <= self.velocity_min
        {
            return Err(SolverError::InvalidVelocityInterval);
        }
        if !self.charge_to_mass.is_finite() || self.charge_to_mass >= 0.0 {
            return Err(SolverError::InvalidChargeToMass);
        }
        if !self.permittivity.is_finite() || self.permittivity <= 0.0 {
            return Err(SolverError::InvalidPermittivity);
        }
        if !self.background_density.is_finite() || self.background_density < 0.0 {
            return Err(SolverError::InvalidBackgroundDensity);
        }
        Ok(())
    }

    pub fn dx(self) -> f64 {
        self.domain_length / self.nx as f64
    }
    pub fn dv(self) -> f64 {
        (self.velocity_max - self.velocity_min) / self.nv as f64
    }
    pub fn x(self, ix: usize) -> f64 {
        ix as f64 * self.dx()
    }
    pub fn velocity(self, iv: usize) -> f64 {
        self.velocity_min + iv as f64 * self.dv()
    }
}

/// Raw state observations. These are not universal acceptance certificates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VlasovPoissonDiagnostics {
    pub time: f64,
    pub total_mass: f64,
    pub momentum: f64,
    pub kinetic_energy: f64,
    pub field_energy: f64,
    pub total_energy: f64,
    pub net_charge: f64,
    pub max_poisson_residual: f64,
    pub minimum_distribution: f64,
    /// Maximum absolute value at the two velocity endpoints divided by the
    /// global maximum absolute value. This is a warning, not a tail bound.
    pub velocity_boundary_ratio: f64,
}

/// Complex coefficient of a positive spatial Fourier mode of the electric field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FourierModeCoefficient {
    /// Real part of the normalised positive-mode coefficient.
    pub real: f64,
    /// Imaginary part of the normalised positive-mode coefficient.
    pub imaginary: f64,
}

impl FourierModeCoefficient {
    /// Magnitude of the complex mode coefficient.
    pub fn amplitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }

    /// Principal phase in radians, in the interval [-pi, pi].
    pub fn phase(self) -> f64 {
        self.imaginary.atan2(self.real)
    }
}

/// Spectral-tail energy observations for one phase-space axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralAxisTail {
    /// Sum of squared Fourier coefficients over all modes.
    pub total_spectral_energy: f64,
    /// Sum of squared coefficients in the configured high-frequency band.
    pub high_frequency_energy: f64,
    /// High-frequency energy divided by total spectral energy, or zero for a
    /// zero-energy axis.
    pub high_frequency_fraction: f64,
    /// Nyquist-mode energy divided by total spectral energy, or zero for a
    /// zero-energy axis.
    pub nyquist_energy_fraction: f64,
}

/// Independent spectral-tail measurements along x and v.
///
/// This is a resolution-warning diagnostic, not a rigorous truncation-error
/// bound and not an automatic grid-expansion decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSpaceSpectralTail {
    /// Modes with |k| >= ceil(tail_start_fraction * Nyquist) count as high.
    pub tail_start_fraction: f64,
    pub x: SpectralAxisTail,
    pub velocity: SpectralAxisTail,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    InvalidGrid {
        axis: &'static str,
        value: usize,
    },
    InvalidDomainLength,
    InvalidVelocityInterval,
    InvalidChargeToMass,
    InvalidPermittivity,
    InvalidBackgroundDensity,
    GridSizeOverflow,
    DistributionLengthMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteDistribution {
        index: usize,
    },
    InvalidTimeStep,
    VelocityImpulseLengthMismatch {
        expected: usize,
        actual: usize,
    },
    NonFiniteVelocityImpulse {
        index: usize,
    },
    NonFiniteVelocityImpulsePhase {
        index: usize,
    },
    InvalidThermalSpeed,
    InvalidPerturbationAmplitude,
    InvalidSpatialMode {
        mode: usize,
        maximum_exclusive: usize,
    },
    InvalidElectricMode {
        mode: usize,
        maximum_exclusive: usize,
    },
    InvalidSpectralTailFraction,
    InvalidInterpolationGrid {
        source_len: usize,
        target_len: usize,
    },
    AllocationFailed {
        elements: usize,
    },
    NonFiniteInterpolationInput {
        index: usize,
    },
    NonFiniteState,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid { axis, value } => {
                write!(f, "{axis}-grid size {value} must be a power of two >= 4")
            }
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::InvalidVelocityInterval => write!(
                f,
                "velocity interval endpoints must be finite and increasing"
            ),
            Self::InvalidChargeToMass => write!(
                f,
                "electron charge-to-mass ratio must be finite and negative"
            ),
            Self::InvalidPermittivity => write!(f, "permittivity must be finite and positive"),
            Self::InvalidBackgroundDensity => {
                write!(f, "background density must be finite and non-negative")
            }
            Self::GridSizeOverflow => write!(f, "phase-space grid size nx * nv overflows usize"),
            Self::DistributionLengthMismatch { expected, actual } => write!(
                f,
                "distribution has {actual} values; expected {expected} (nx * nv)"
            ),
            Self::NonFiniteDistribution { index } => {
                write!(f, "distribution value {index} is non-finite")
            }
            Self::InvalidTimeStep => write!(f, "time step must be finite and positive"),
            Self::VelocityImpulseLengthMismatch { expected, actual } => write!(
                f,
                "velocity impulse has {actual} values; expected {expected} (nx)"
            ),
            Self::NonFiniteVelocityImpulse { index } => {
                write!(f, "velocity impulse at spatial index {index} is non-finite")
            }
            Self::NonFiniteVelocityImpulsePhase { index } => write!(
                f,
                "velocity impulse at spatial index {index} produces a non-finite Fourier phase"
            ),
            Self::InvalidThermalSpeed => write!(f, "thermal speed must be finite and positive"),
            Self::InvalidPerturbationAmplitude => write!(
                f,
                "Maxwellian perturbation amplitude must satisfy |alpha| < 1"
            ),
            Self::InvalidSpatialMode {
                mode,
                maximum_exclusive,
            } => write!(
                f,
                "spatial mode {mode} must be positive and less than {maximum_exclusive}"
            ),
            Self::InvalidElectricMode {
                mode,
                maximum_exclusive,
            } => write!(
                f,
                "electric mode {mode} must be positive and less than {maximum_exclusive}"
            ),
            Self::InvalidSpectralTailFraction => write!(
                f,
                "spectral tail start fraction must be finite and in (0, 1]"
            ),
            Self::InvalidInterpolationGrid {
                source_len,
                target_len,
            } => write!(
                f,
                "zero-padding requires power-of-two source length >= 4 and power-of-two target length >= source length (got {source_len} -> {target_len})"
            ),
            Self::AllocationFailed { elements } => {
                write!(f, "could not allocate {elements} interpolation elements")
            }
            Self::NonFiniteInterpolationInput { index } => {
                write!(f, "interpolation input {index} is non-finite")
            }
            Self::NonFiniteState => write!(f, "solver produced a non-finite state"),
        }
    }
}

impl Error for SolverError {}

/// Construct a discretely normalized Maxwellian with density
/// 1 + alpha*cos(2*pi*mode*x/L). Thermal speed is the Gaussian standard
/// deviation. The sampled velocity integral is normalized to one.
pub fn maxwellian_perturbation(
    config: VlasovPoissonConfig,
    alpha: f64,
    mode: usize,
    thermal_speed: f64,
) -> Result<Vec<f64>, SolverError> {
    config.validate()?;
    if !thermal_speed.is_finite() || thermal_speed <= 0.0 {
        return Err(SolverError::InvalidThermalSpeed);
    }
    if !alpha.is_finite() || alpha.abs() >= 1.0 {
        return Err(SolverError::InvalidPerturbationAmplitude);
    }
    if mode == 0 || mode >= config.nx / 2 {
        return Err(SolverError::InvalidSpatialMode {
            mode,
            maximum_exclusive: config.nx / 2,
        });
    }

    let mut maxwellian = Vec::with_capacity(config.nv);
    for iv in 0..config.nv {
        let velocity = config.velocity(iv);
        maxwellian
            .push((-0.5 * (velocity / thermal_speed).powi(2)).exp() / (TAU.sqrt() * thermal_speed));
    }
    let integral = maxwellian.iter().sum::<f64>() * config.dv();
    if !integral.is_finite() || integral <= 0.0 {
        return Err(SolverError::NonFiniteState);
    }
    for value in &mut maxwellian {
        *value /= integral;
    }

    let wave_number = TAU * mode as f64 / config.domain_length;
    let mut distribution = vec![0.0; config.nx * config.nv];
    for ix in 0..config.nx {
        let density = 1.0 + alpha * (wave_number * config.x(ix)).cos();
        for iv in 0..config.nv {
            distribution[ix * config.nv + iv] = density * maxwellian[iv];
        }
    }
    Ok(distribution)
}

/// Independent distribution-space solver for periodic 1D1V collisionless
/// Vlasov-Poisson. Storage is x-major: f[ix * nv + iv].
///
/// One Strang step is x-advection(dt/2), self-consistent velocity-advection(dt),
/// x-advection(dt/2). The finite velocity interval is numerically periodic;
/// physical adequacy requires velocity-grid and velocity-domain convergence.
#[derive(Debug, Clone)]
pub struct VlasovPoisson1d1v {
    config: VlasovPoissonConfig,
    distribution: Vec<f64>,
    electric_field: Vec<f64>,
    time: f64,
}

impl VlasovPoisson1d1v {
    pub fn new(config: VlasovPoissonConfig, distribution: Vec<f64>) -> Result<Self, SolverError> {
        config.validate()?;
        let expected = config.nx * config.nv;
        if distribution.len() != expected {
            return Err(SolverError::DistributionLengthMismatch {
                expected,
                actual: distribution.len(),
            });
        }
        for (index, value) in distribution.iter().enumerate() {
            if !value.is_finite() {
                return Err(SolverError::NonFiniteDistribution { index });
            }
        }
        let mut solver = Self {
            config,
            distribution,
            electric_field: vec![0.0; config.nx],
            time: 0.0,
        };
        solver.refresh_electric_field()?;
        Ok(solver)
    }

    pub fn config(&self) -> VlasovPoissonConfig {
        self.config
    }
    pub fn distribution(&self) -> &[f64] {
        &self.distribution
    }
    pub fn electric_field(&self) -> &[f64] {
        &self.electric_field
    }
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Advance only the free-streaming subflow. Exposed for manufactured
    /// characteristic-solution tests; this deliberately omits self-fields.
    pub fn advance_free_streaming(&mut self, dt: f64) -> Result<(), SolverError> {
        validate_dt(dt)?;
        self.advect_x(dt);
        self.time += dt;
        if !self.time.is_finite() {
            return Err(SolverError::NonFiniteState);
        }
        self.refresh_electric_field()?;
        self.ensure_finite_state()
    }

    /// Apply an instantaneous, spatially varying velocity translation.
    ///
    /// For each spatial cell this applies the exact periodic Fourier
    /// characteristic map f_plus(x, v) = f_minus(x, v - delta_v(x)).
    /// This is an externally prescribed impulse, not a time-integrated
    /// self-consistent field step, and the simulation clock is unchanged.
    ///
    /// The finite periodic velocity interval and its Nyquist-mode projection
    /// are the same as the existing velocity-advection subflow. Inputs and
    /// transformed candidate rows are validated before the state is committed.
    pub fn apply_velocity_impulse(&mut self, velocity_impulses: &[f64]) -> Result<(), SolverError> {
        let config = self.config;
        if velocity_impulses.len() != config.nx {
            return Err(SolverError::VelocityImpulseLengthMismatch {
                expected: config.nx,
                actual: velocity_impulses.len(),
            });
        }
        for (index, impulse) in velocity_impulses.iter().copied().enumerate() {
            if !impulse.is_finite() {
                return Err(SolverError::NonFiniteVelocityImpulse { index });
            }
        }

        let period = config.velocity_max - config.velocity_min;
        let expected = config
            .nx
            .checked_mul(config.nv)
            .ok_or(SolverError::GridSizeOverflow)?;

        let mut candidate = Vec::new();
        candidate
            .try_reserve_exact(expected)
            .map_err(|_| SolverError::AllocationFailed { elements: expected })?;

        let mut line = Vec::new();
        line.try_reserve_exact(config.nv)
            .map_err(|_| SolverError::AllocationFailed {
                elements: config.nv,
            })?;

        for (ix, impulse) in velocity_impulses.iter().copied().enumerate() {
            line.clear();
            let row_start = ix * config.nv;
            let row_end = row_start + config.nv;
            for density in &self.distribution[row_start..row_end] {
                line.push(Complex::new(*density, 0.0));
            }

            fft(&mut line, false);
            for (mode, coefficient) in line.iter_mut().enumerate() {
                let eta = signed_mode(mode, config.nv) as f64 * TAU / period;
                let phase = -eta * impulse;
                if !phase.is_finite() {
                    return Err(SolverError::NonFiniteVelocityImpulsePhase { index: ix });
                }

                // Match the existing finite-periodic-velocity pusher's real
                // Nyquist projection; all other modes retain their full phase.
                let multiplier = if mode == config.nv / 2 {
                    Complex::new(phase.cos(), 0.0)
                } else {
                    Complex::cis(phase)
                };
                *coefficient = coefficient.mul(multiplier);
            }

            fft(&mut line, true);
            for value in &line {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(SolverError::NonFiniteState);
                }
                candidate.push(value.re);
            }
        }

        // Prepare a rollback copy before mutating the solver. If field refresh
        // or the final state check fails, restore both state and field exactly.
        let mut previous_electric_field = Vec::new();
        previous_electric_field
            .try_reserve_exact(self.electric_field.len())
            .map_err(|_| SolverError::AllocationFailed {
                elements: self.electric_field.len(),
            })?;
        previous_electric_field.extend_from_slice(&self.electric_field);

        let previous_distribution = std::mem::replace(&mut self.distribution, candidate);
        if let Err(error) = self
            .refresh_electric_field()
            .and_then(|_| self.ensure_finite_state())
        {
            self.distribution = previous_distribution;
            self.electric_field = previous_electric_field;
            return Err(error);
        }

        Ok(())
    }

    pub fn step(&mut self, dt: f64) -> Result<(), SolverError> {
        validate_dt(dt)?;
        self.advect_x(0.5 * dt);
        self.refresh_electric_field()?;
        self.advect_velocity(dt);
        self.advect_x(0.5 * dt);
        self.time += dt;
        if !self.time.is_finite() {
            return Err(SolverError::NonFiniteState);
        }
        self.refresh_electric_field()?;
        self.ensure_finite_state()
    }

    /// Complex positive spatial Fourier coefficient for the requested mode.
    ///
    /// Returning real and imaginary parts retains temporal phase information.
    /// Its magnitude is the Fourier coefficient magnitude, not the real-space
    /// cosine peak amplitude.
    pub fn electric_mode_coefficient(
        &self,
        mode: usize,
    ) -> Result<FourierModeCoefficient, SolverError> {
        if mode == 0 || mode >= self.config.nx / 2 {
            return Err(SolverError::InvalidElectricMode {
                mode,
                maximum_exclusive: self.config.nx / 2,
            });
        }
        let mut spectrum: Vec<Complex> = self
            .electric_field
            .iter()
            .map(|value| Complex::new(*value, 0.0))
            .collect();
        fft(&mut spectrum, false);
        let coefficient = spectrum[mode].scale(1.0 / self.config.nx as f64);
        Ok(FourierModeCoefficient {
            real: coefficient.re,
            imaginary: coefficient.im,
        })
    }

    /// Magnitude of the positive spatial Fourier coefficient (not cosine
    /// peak amplitude) for the requested electric-field mode.
    pub fn electric_mode_amplitude(&self, mode: usize) -> Result<f64, SolverError> {
        Ok(self.electric_mode_coefficient(mode)?.amplitude())
    }

    /// Measure phase-space spectral-tail energy independently along x and v.
    ///
    /// tail_start_fraction is a fixed diagnostic band threshold in (0, 1].
    /// For example, 0.75 counts modes whose absolute index is at least 75% of
    /// the Nyquist index. It must be frozen in the run profile. The returned
    /// fractions are observed spectral occupancy, not a physical error bound.
    pub fn spectral_tail_diagnostics(
        &self,
        tail_start_fraction: f64,
    ) -> Result<PhaseSpaceSpectralTail, SolverError> {
        if !tail_start_fraction.is_finite()
            || tail_start_fraction <= 0.0
            || tail_start_fraction > 1.0
        {
            return Err(SolverError::InvalidSpectralTailFraction);
        }

        let x_start = ((tail_start_fraction * (self.config.nx / 2) as f64).ceil() as usize).max(1);
        let v_start = ((tail_start_fraction * (self.config.nv / 2) as f64).ceil() as usize).max(1);
        let x = spectral_tail_axis_x(&self.distribution, self.config, x_start)?;
        let velocity = spectral_tail_axis_v(&self.distribution, self.config, v_start)?;

        Ok(PhaseSpaceSpectralTail {
            tail_start_fraction,
            x,
            velocity,
        })
    }

    pub fn diagnostics(&self) -> Result<VlasovPoissonDiagnostics, SolverError> {
        let config = self.config;
        let dx = config.dx();
        let dv = config.dv();
        let particle_mass = -1.0 / config.charge_to_mass;
        let mut mass = 0.0;
        let mut momentum = 0.0;
        let mut kinetic_energy = 0.0;
        let mut minimum_distribution = f64::INFINITY;
        let mut maximum_abs_distribution: f64 = 0.0;
        let mut boundary_abs: f64 = 0.0;

        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let value = self.distribution[ix * config.nv + iv];
                let velocity = config.velocity(iv);
                mass += particle_mass * value * dx * dv;
                momentum += particle_mass * value * velocity * dx * dv;
                kinetic_energy += 0.5 * particle_mass * value * velocity * velocity * dx * dv;
                minimum_distribution = minimum_distribution.min(value);
                maximum_abs_distribution = maximum_abs_distribution.max(value.abs());
                if iv == 0 || iv + 1 == config.nv {
                    boundary_abs = boundary_abs.max(value.abs());
                }
            }
        }
        let field_energy = 0.5
            * config.permittivity
            * self
                .electric_field
                .iter()
                .map(|value| value * value * dx)
                .sum::<f64>();
        let rho = self.charge_density()?;
        let net_charge = rho.iter().sum::<f64>() * dx;
        let max_poisson_residual = self.max_poisson_residual(&rho);
        let velocity_boundary_ratio = if maximum_abs_distribution > 0.0 {
            boundary_abs / maximum_abs_distribution
        } else {
            0.0
        };
        let result = VlasovPoissonDiagnostics {
            time: self.time,
            total_mass: mass,
            momentum,
            kinetic_energy,
            field_energy,
            total_energy: kinetic_energy + field_energy,
            net_charge,
            max_poisson_residual,
            minimum_distribution,
            velocity_boundary_ratio,
        };
        if [
            result.time,
            result.total_mass,
            result.momentum,
            result.kinetic_energy,
            result.field_energy,
            result.total_energy,
            result.net_charge,
            result.max_poisson_residual,
            result.minimum_distribution,
            result.velocity_boundary_ratio,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(SolverError::NonFiniteState);
        }
        Ok(result)
    }

    fn ensure_finite_state(&self) -> Result<(), SolverError> {
        if self.distribution.iter().any(|value| !value.is_finite())
            || self.electric_field.iter().any(|value| !value.is_finite())
        {
            return Err(SolverError::NonFiniteState);
        }
        Ok(())
    }

    fn advect_x(&mut self, dt: f64) {
        let config = self.config;
        let mut line = vec![Complex::ZERO; config.nx];
        for iv in 0..config.nv {
            for (ix, value) in line.iter_mut().enumerate() {
                *value = Complex::new(self.distribution[ix * config.nv + iv], 0.0);
            }
            fft(&mut line, false);
            for (mode, coefficient) in line.iter_mut().enumerate() {
                let k = signed_mode(mode, config.nx) as f64 * TAU / config.domain_length;
                let phase = -k * config.velocity(iv) * dt;
                let multiplier = if mode == config.nx / 2 {
                    Complex::new(phase.cos(), 0.0)
                } else {
                    Complex::cis(phase)
                };
                *coefficient = coefficient.mul(multiplier);
            }
            fft(&mut line, true);
            for (ix, value) in line.iter().enumerate() {
                self.distribution[ix * config.nv + iv] = value.re;
            }
        }
    }

    fn advect_velocity(&mut self, dt: f64) {
        let config = self.config;
        let period = config.velocity_max - config.velocity_min;
        let mut line = vec![Complex::ZERO; config.nv];
        for (ix, electric_value) in self.electric_field.iter().enumerate() {
            let acceleration = config.charge_to_mass * *electric_value;
            let row = &self.distribution[ix * config.nv..(ix + 1) * config.nv];
            for (value, density) in line.iter_mut().zip(row) {
                *value = Complex::new(*density, 0.0);
            }
            fft(&mut line, false);
            for (mode, coefficient) in line.iter_mut().enumerate() {
                let eta = signed_mode(mode, config.nv) as f64 * TAU / period;
                let phase = -eta * acceleration * dt;
                let multiplier = if mode == config.nv / 2 {
                    Complex::new(phase.cos(), 0.0)
                } else {
                    Complex::cis(phase)
                };
                *coefficient = coefficient.mul(multiplier);
            }
            fft(&mut line, true);
            for (iv, value) in line.iter().enumerate() {
                self.distribution[ix * config.nv + iv] = value.re;
            }
        }
    }

    fn charge_density(&self) -> Result<Vec<f64>, SolverError> {
        let config = self.config;
        let mut rho = vec![0.0; config.nx];
        for (ix, density) in rho.iter_mut().enumerate() {
            let number_density = self.distribution[ix * config.nv..(ix + 1) * config.nv]
                .iter()
                .sum::<f64>()
                * config.dv();
            *density = config.background_density - number_density;
        }
        if rho.iter().any(|value| !value.is_finite()) {
            return Err(SolverError::NonFiniteState);
        }
        Ok(rho)
    }

    fn refresh_electric_field(&mut self) -> Result<(), SolverError> {
        let rho = self.charge_density()?;
        self.electric_field = solve_periodic_poisson_spectral(&rho, self.config)?;
        Ok(())
    }

    fn max_poisson_residual(&self, rho: &[f64]) -> f64 {
        let mut spectrum: Vec<Complex> = self
            .electric_field
            .iter()
            .map(|value| Complex::new(*value, 0.0))
            .collect();
        fft(&mut spectrum, false);
        for (mode, coefficient) in spectrum.iter_mut().enumerate() {
            let k = signed_mode(mode, self.config.nx) as f64 * TAU / self.config.domain_length;
            if mode == self.config.nx / 2 {
                *coefficient = Complex::ZERO;
            } else {
                *coefficient = coefficient.mul_i(k);
            }
        }
        fft(&mut spectrum, true);
        let mut maximum: f64 = 0.0;
        for ix in 0..self.config.nx {
            maximum = maximum.max((spectrum[ix].re - rho[ix] / self.config.permittivity).abs());
        }
        maximum
    }
}

/// Solve dE/dx = rho/epsilon_0 with a zero-mean periodic spectral gauge.
/// The zero spatial mode and collocation Nyquist mode are set to zero. Their
/// residual is retained in the solver diagnostics; non-neutral data is not
/// silently promoted to a valid periodic Poisson solution.
pub fn solve_periodic_poisson_spectral(
    charge_density: &[f64],
    config: VlasovPoissonConfig,
) -> Result<Vec<f64>, SolverError> {
    config.validate()?;
    if charge_density.len() != config.nx {
        return Err(SolverError::DistributionLengthMismatch {
            expected: config.nx,
            actual: charge_density.len(),
        });
    }
    if charge_density.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::NonFiniteState);
    }

    let mut spectrum: Vec<Complex> = charge_density
        .iter()
        .map(|value| Complex::new(*value, 0.0))
        .collect();
    fft(&mut spectrum, false);
    for (mode, coefficient) in spectrum.iter_mut().enumerate() {
        if mode == 0 || mode == config.nx / 2 {
            *coefficient = Complex::ZERO;
        } else {
            let k = signed_mode(mode, config.nx) as f64 * TAU / config.domain_length;
            *coefficient = coefficient.mul(Complex::new(0.0, -1.0 / (k * config.permittivity)));
        }
    }
    fft(&mut spectrum, true);
    let field: Vec<f64> = spectrum.iter().map(|value| value.re).collect();
    if field.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::NonFiniteState);
    }
    Ok(field)
}

fn spectral_tail_axis_x(
    distribution: &[f64],
    config: VlasovPoissonConfig,
    tail_start_mode: usize,
) -> Result<SpectralAxisTail, SolverError> {
    let mut total = 0.0;
    let mut high = 0.0;
    let mut nyquist = 0.0;
    let mut line = vec![Complex::ZERO; config.nx];

    for iv in 0..config.nv {
        for (ix, value) in line.iter_mut().enumerate() {
            *value = Complex::new(distribution[ix * config.nv + iv], 0.0);
        }
        fft(&mut line, false);
        for (mode, coefficient) in line.iter().enumerate() {
            let power = coefficient
                .re
                .mul_add(coefficient.re, coefficient.im * coefficient.im);
            if !power.is_finite() {
                return Err(SolverError::NonFiniteState);
            }
            total += power;
            if signed_mode(mode, config.nx).unsigned_abs() >= tail_start_mode {
                high += power;
            }
            if mode == config.nx / 2 {
                nyquist += power;
            }
        }
    }

    spectral_axis_result(total, high, nyquist)
}

fn spectral_tail_axis_v(
    distribution: &[f64],
    config: VlasovPoissonConfig,
    tail_start_mode: usize,
) -> Result<SpectralAxisTail, SolverError> {
    let mut total = 0.0;
    let mut high = 0.0;
    let mut nyquist = 0.0;
    let mut line = vec![Complex::ZERO; config.nv];

    for ix in 0..config.nx {
        let row = &distribution[ix * config.nv..(ix + 1) * config.nv];
        for (value, sample) in line.iter_mut().zip(row) {
            *value = Complex::new(*sample, 0.0);
        }
        fft(&mut line, false);
        for (mode, coefficient) in line.iter().enumerate() {
            let power = coefficient
                .re
                .mul_add(coefficient.re, coefficient.im * coefficient.im);
            if !power.is_finite() {
                return Err(SolverError::NonFiniteState);
            }
            total += power;
            if signed_mode(mode, config.nv).unsigned_abs() >= tail_start_mode {
                high += power;
            }
            if mode == config.nv / 2 {
                nyquist += power;
            }
        }
    }

    spectral_axis_result(total, high, nyquist)
}

fn spectral_axis_result(
    total: f64,
    high: f64,
    nyquist: f64,
) -> Result<SpectralAxisTail, SolverError> {
    if !total.is_finite() || !high.is_finite() || !nyquist.is_finite() {
        return Err(SolverError::NonFiniteState);
    }
    let (high_fraction, nyquist_fraction) = if total > 0.0 {
        (high / total, nyquist / total)
    } else {
        (0.0, 0.0)
    };
    if !high_fraction.is_finite() || !nyquist_fraction.is_finite() {
        return Err(SolverError::NonFiniteState);
    }
    Ok(SpectralAxisTail {
        total_spectral_energy: total,
        high_frequency_energy: high,
        high_frequency_fraction: high_fraction,
        nyquist_energy_fraction: nyquist_fraction,
    })
}

/// Interpolate a real periodic line onto a finer power-of-two grid by exact
/// trigonometric zero-padding of its discrete Fourier coefficients.
///
/// The source samples are assumed to cover one period uniformly. This function
/// does not change the physical period or apply a filter. The old Nyquist
/// coefficient is split equally between the positive and negative Nyquist
/// harmonics so the refined interpolant remains real-valued.
///
/// This is a transfer oracle only: callers must separately qualify the
/// physical effect and cost of changing a simulation grid.
pub fn zero_pad_periodic_line(samples: &[f64], target_len: usize) -> Result<Vec<f64>, SolverError> {
    let source_len = samples.len();
    if source_len < 4
        || !source_len.is_power_of_two()
        || target_len < source_len
        || !target_len.is_power_of_two()
    {
        return Err(SolverError::InvalidInterpolationGrid {
            source_len,
            target_len,
        });
    }
    for (index, sample) in samples.iter().enumerate() {
        if !sample.is_finite() {
            return Err(SolverError::NonFiniteInterpolationInput { index });
        }
    }
    if target_len == source_len {
        return Ok(samples.to_vec());
    }

    let mut source_spectrum = Vec::new();
    source_spectrum
        .try_reserve_exact(source_len)
        .map_err(|_| SolverError::AllocationFailed {
            elements: source_len,
        })?;
    source_spectrum.extend(samples.iter().map(|sample| Complex::new(*sample, 0.0)));
    fft(&mut source_spectrum, false);

    let scale = target_len as f64 / source_len as f64;
    if !scale.is_finite() {
        return Err(SolverError::NonFiniteState);
    }

    let mut target_spectrum = Vec::new();
    target_spectrum
        .try_reserve_exact(target_len)
        .map_err(|_| SolverError::AllocationFailed {
            elements: target_len,
        })?;
    target_spectrum.resize(target_len, Complex::ZERO);

    // DC and all strictly positive modes below the source Nyquist.
    for mode in 0..(source_len / 2) {
        target_spectrum[mode] = source_spectrum[mode].scale(scale);
    }

    // Negative modes retain their signed frequency index on the larger grid.
    for (source_index, coefficient) in source_spectrum.iter().enumerate().skip(source_len / 2 + 1) {
        let negative_mode = source_len - source_index;
        let target_index = target_len - negative_mode;
        target_spectrum[target_index] = coefficient.scale(scale);
    }

    // The source Nyquist mode aliases +N/2 and -N/2 on the coarse grid.
    // Split its real coefficient between both target frequencies.
    let nyquist = source_spectrum[source_len / 2].re * (0.5 * scale);
    target_spectrum[source_len / 2] = Complex::new(nyquist, 0.0);
    target_spectrum[target_len - source_len / 2] = Complex::new(nyquist, 0.0);

    fft(&mut target_spectrum, true);
    let mut refined = Vec::new();
    refined
        .try_reserve_exact(target_len)
        .map_err(|_| SolverError::AllocationFailed {
            elements: target_len,
        })?;
    for value in &target_spectrum {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(SolverError::NonFiniteState);
        }
        refined.push(value.re);
    }
    Ok(refined)
}

fn validate_dt(dt: f64) -> Result<(), SolverError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SolverError::InvalidTimeStep);
    }
    Ok(())
}

fn signed_mode(index: usize, length: usize) -> isize {
    if index <= length / 2 {
        index as isize
    } else {
        index as isize - length as isize
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn cis(angle: f64) -> Self {
        Self {
            re: angle.cos(),
            im: angle.sin(),
        }
    }
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
    fn mul_i(self, scalar: f64) -> Self {
        Self::new(-self.im * scalar, self.re * scalar)
    }
    fn scale(self, scalar: f64) -> Self {
        Self::new(self.re * scalar, self.im * scalar)
    }
}

/// In-place radix-2 Cooley-Tukey FFT. Forward is unnormalised; inverse divides by N.
fn fft(values: &mut [Complex], inverse: bool) {
    let length = values.len();
    debug_assert!(length >= 2 && length.is_power_of_two());

    let mut reversed = 0usize;
    for index in 1..length {
        let mut bit = length >> 1;
        while reversed & bit != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let mut span = 2usize;
    loop {
        let half = span / 2;
        let direction = if inverse { 1.0 } else { -1.0 };
        let root = Complex::cis(direction * TAU / span as f64);
        for block in (0..length).step_by(span) {
            let mut twiddle = Complex::new(1.0, 0.0);
            for offset in 0..half {
                let even_index = block + offset;
                let odd_index = even_index + half;
                let even = values[even_index];
                let odd = values[odd_index].mul(twiddle);
                values[even_index] = even.add(odd);
                values[odd_index] = even.sub(odd);
                twiddle = twiddle.mul(root);
            }
        }
        if span == length {
            break;
        }
        span *= 2;
    }
    if inverse {
        let scale = 1.0 / length as f64;
        for value in values {
            *value = value.scale(scale);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(nx: usize, nv: usize) -> VlasovPoissonConfig {
        VlasovPoissonConfig {
            nx,
            nv,
            domain_length: 4.0 * std::f64::consts::PI,
            velocity_min: -6.0,
            velocity_max: 6.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        }
    }

    #[test]
    fn fft_forward_inverse_round_trip() {
        let original: Vec<Complex> = (0..16)
            .map(|i| Complex::new(i as f64 * 0.25, (i as f64 * 0.3).sin()))
            .collect();
        let mut actual = original.clone();
        fft(&mut actual, false);
        fft(&mut actual, true);
        for (a, b) in actual.iter().zip(original.iter()) {
            assert!((a.re - b.re).abs() < 1.0e-12);
            assert!((a.im - b.im).abs() < 1.0e-12);
        }
    }

    #[test]
    fn homogeneous_maxwellian_is_a_discrete_equilibrium() {
        let config = test_config(32, 64);
        let initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let before = solver.diagnostics().unwrap();
        assert!(before.max_poisson_residual < 1.0e-12);
        assert!(
            solver
                .electric_field()
                .iter()
                .all(|value| value.abs() < 1.0e-12)
        );
        solver.step(0.05).unwrap();
        let after = solver.diagnostics().unwrap();
        assert!(after.max_poisson_residual < 1.0e-12);
        assert!((after.total_mass - before.total_mass).abs() < 1.0e-11);
        for (a, b) in solver.distribution().iter().zip(initial.iter()) {
            assert!((a - b).abs() < 1.0e-11);
        }
    }

    #[test]
    fn manufactured_free_streaming_matches_characteristics() {
        let config = test_config(32, 64);
        let alpha = 0.05;
        let mode = 2usize;
        let thermal_speed = 1.1;
        let dt = 0.37;
        let initial = maxwellian_perturbation(config, alpha, mode, thermal_speed).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        solver.advance_free_streaming(dt).unwrap();

        let k = TAU * mode as f64 / config.domain_length;
        let mut maximum_error: f64 = 0.0;
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let original = initial[ix * config.nv + iv];
                let maxwellian = original / (1.0 + alpha * (k * config.x(ix)).cos());
                let characteristic =
                    1.0 + alpha * (k * (config.x(ix) - config.velocity(iv) * dt)).cos();
                let expected = maxwellian * characteristic;
                let actual = solver.distribution()[ix * config.nv + iv];
                maximum_error = maximum_error.max((actual - expected).abs());
            }
        }
        assert!(
            maximum_error < 2.0e-11,
            "free-streaming max error {maximum_error:e}"
        );
    }

    fn periodic_velocity_mode_initial_state(config: VlasovPoissonConfig, mode: usize) -> Vec<f64> {
        let period = config.velocity_max - config.velocity_min;
        let eta = TAU * mode as f64 / period;
        let base = 1.0 / period;
        let amplitude = 0.1 / period;
        (0..config.nx)
            .flat_map(|_| {
                (0..config.nv).map(move |iv| {
                    base + amplitude * (eta * (config.velocity(iv) - config.velocity_min)).cos()
                })
            })
            .collect()
    }

    fn discrete_mass(config: VlasovPoissonConfig, distribution: &[f64]) -> f64 {
        distribution.iter().sum::<f64>() * config.dx() * config.dv()
    }

    #[test]
    fn constant_velocity_impulse_matches_manufactured_periodic_characteristic() {
        let config = test_config(16, 64);
        let velocity_mode = 4usize;
        let initial = periodic_velocity_mode_initial_state(config, velocity_mode);
        let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let period = config.velocity_max - config.velocity_min;
        let eta = TAU * velocity_mode as f64 / period;
        let impulse = 0.23;
        let impulses = vec![impulse; config.nx];

        solver.apply_velocity_impulse(&impulses).unwrap();

        let base = 1.0 / period;
        let amplitude = 0.1 / period;
        let mut maximum_error: f64 = 0.0;
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let shifted_velocity = config.velocity(iv) - config.velocity_min - impulse;
                let expected = base + amplitude * (eta * shifted_velocity).cos();
                let actual = solver.distribution()[ix * config.nv + iv];
                maximum_error = maximum_error.max((actual - expected).abs());
            }
        }
        assert!(
            maximum_error < 2.0e-12,
            "constant impulse max error {maximum_error:e}"
        );
    }

    #[test]
    fn spatially_varying_velocity_impulse_matches_characteristics_and_preserves_mass() {
        let config = test_config(32, 64);
        let velocity_mode = 3usize;
        let initial = periodic_velocity_mode_initial_state(config, velocity_mode);
        let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let before_mass = discrete_mass(config, solver.distribution());
        let before_time = solver.time();
        let period = config.velocity_max - config.velocity_min;
        let eta = TAU * velocity_mode as f64 / period;
        let impulses: Vec<f64> = (0..config.nx)
            .map(|ix| 0.14 * (TAU * ix as f64 / config.nx as f64 + 0.3).cos())
            .collect();

        solver.apply_velocity_impulse(&impulses).unwrap();

        let base = 1.0 / period;
        let amplitude = 0.1 / period;
        let mut maximum_error: f64 = 0.0;
        for (ix, impulse) in impulses.iter().copied().enumerate() {
            for iv in 0..config.nv {
                let shifted_velocity = config.velocity(iv) - config.velocity_min - impulse;
                let expected = base + amplitude * (eta * shifted_velocity).cos();
                let actual = solver.distribution()[ix * config.nv + iv];
                maximum_error = maximum_error.max((actual - expected).abs());
            }
        }

        assert!(
            maximum_error < 2.0e-12,
            "instantaneous impulse max error {maximum_error:e}"
        );
        assert!((discrete_mass(config, solver.distribution()) - before_mass).abs() < 2.0e-13);
        assert_eq!(solver.time(), before_time);
    }

    #[test]
    fn zero_velocity_impulse_is_identity_to_roundoff_and_does_not_advance_time() {
        let config = test_config(16, 32);
        let initial = periodic_velocity_mode_initial_state(config, 2);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let before_time = solver.time();
        let zero_impulses = vec![0.0; config.nx];
        solver.apply_velocity_impulse(&zero_impulses).unwrap();

        let maximum_error = solver
            .distribution()
            .iter()
            .zip(initial.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f64, f64::max);
        assert!(maximum_error < 2.0e-13, "identity error {maximum_error:e}");
        assert_eq!(solver.time(), before_time);
    }

    #[test]
    fn invalid_velocity_impulses_fail_before_state_mutation() {
        let config = test_config(16, 32);
        let initial = periodic_velocity_mode_initial_state(config, 2);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let before_field = solver.electric_field().to_vec();
        let before_time = solver.time();

        assert_eq!(
            solver.apply_velocity_impulse(&[0.0; 3]),
            Err(SolverError::VelocityImpulseLengthMismatch {
                expected: config.nx,
                actual: 3,
            })
        );
        let mut non_finite = vec![0.0; config.nx];
        non_finite[5] = f64::NAN;
        assert_eq!(
            solver.apply_velocity_impulse(&non_finite),
            Err(SolverError::NonFiniteVelocityImpulse { index: 5 })
        );
        let mut overflowing_phase = vec![0.0; config.nx];
        overflowing_phase[7] = f64::MAX;
        assert_eq!(
            solver.apply_velocity_impulse(&overflowing_phase),
            Err(SolverError::NonFiniteVelocityImpulsePhase { index: 7 })
        );

        assert_eq!(solver.distribution(), initial.as_slice());
        assert_eq!(solver.electric_field(), before_field.as_slice());
        assert_eq!(solver.time(), before_time);
    }

    #[test]
    fn spectral_poisson_solves_a_single_cosine_mode() {
        let config = test_config(32, 32);
        let mode = 3usize;
        let amplitude = 0.2;
        let k = TAU * mode as f64 / config.domain_length;
        let rho: Vec<f64> = (0..config.nx)
            .map(|ix| amplitude * (k * config.x(ix)).cos())
            .collect();
        let electric = solve_periodic_poisson_spectral(&rho, config).unwrap();
        let mut maximum_error: f64 = 0.0;
        for (ix, actual) in electric.iter().enumerate() {
            let expected = amplitude / k * (k * config.x(ix)).sin();
            maximum_error = maximum_error.max((*actual - expected).abs());
        }
        assert!(maximum_error < 2.0e-12);
    }

    #[test]
    fn non_neutral_periodic_data_remains_visible_in_residual() {
        let mut config = test_config(16, 32);
        config.background_density = 1.2;
        let initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        let solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let diagnostics = solver.diagnostics().unwrap();
        assert!((diagnostics.net_charge - 0.2 * config.domain_length).abs() < 1.0e-11);
        assert!(diagnostics.max_poisson_residual > 0.19);
    }

    #[test]
    fn coupled_step_preserves_mass_and_reports_poisson_residual() {
        let config = test_config(32, 64);
        let initial = maxwellian_perturbation(config, 1.0e-3, 1, 1.0).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let before = solver.diagnostics().unwrap();
        for _ in 0..20 {
            solver.step(0.05).unwrap();
        }
        let after = solver.diagnostics().unwrap();
        assert!((after.total_mass - before.total_mass).abs() < 5.0e-11);
        assert!((after.momentum - before.momentum).abs() < 1.0e-9);
        assert!((after.total_energy - before.total_energy).abs() < 1.0e-7);
        assert!(after.max_poisson_residual < 2.0e-11);
        assert!(after.total_energy.is_finite());
        assert!(after.velocity_boundary_ratio < 1.0e-5);
    }

    #[test]
    fn coupled_strang_step_has_second_order_timestep_convergence() {
        fn final_mode_amplitude(dt: f64) -> f64 {
            let config = test_config(32, 64);
            let initial = maxwellian_perturbation(config, 1.0e-3, 1, 1.0).unwrap();
            let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
            let steps = (1.0 / dt).round() as usize;
            for _ in 0..steps {
                solver.step(dt).unwrap();
            }
            solver.electric_mode_amplitude(1).unwrap()
        }

        let coarse = final_mode_amplitude(0.04);
        let medium = final_mode_amplitude(0.02);
        let fine = final_mode_amplitude(0.01);
        let coarse_difference = (coarse - medium).abs();
        let fine_difference = (medium - fine).abs();
        let observed_ratio = coarse_difference / fine_difference;

        assert!(fine_difference > 0.0);
        assert!(
            (3.0..=5.0).contains(&observed_ratio),
            "expected second-order refinement ratio near 4, got {observed_ratio}"
        );
    }

    #[test]
    fn phase_space_grid_size_overflow_is_rejected() {
        let side = 1usize << (usize::BITS / 2);
        let mut config = test_config(side, side);
        config.nx = side;
        config.nv = side;
        assert!(matches!(
            config.validate(),
            Err(SolverError::GridSizeOverflow)
        ));
    }

    #[test]
    fn spectral_tail_diagnostic_separates_x_and_velocity_high_modes() {
        let config = test_config(32, 32);
        let x_mode = 14usize;
        let x_high: Vec<f64> = (0..config.nx)
            .flat_map(|ix| {
                let value = (TAU * x_mode as f64 * ix as f64 / config.nx as f64).cos();
                std::iter::repeat_n(value, config.nv)
            })
            .collect();
        let x_solver = VlasovPoisson1d1v::new(config, x_high).unwrap();
        let x_tail = x_solver.spectral_tail_diagnostics(0.75).unwrap();
        assert!(x_tail.x.high_frequency_fraction > 0.999_999);
        assert!(x_tail.velocity.high_frequency_fraction < 1.0e-12);

        let v_mode = 14usize;
        let mut v_high = vec![0.0; config.nx * config.nv];
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                v_high[ix * config.nv + iv] =
                    (TAU * v_mode as f64 * iv as f64 / config.nv as f64).cos();
            }
        }
        let v_solver = VlasovPoisson1d1v::new(config, v_high).unwrap();
        let v_tail = v_solver.spectral_tail_diagnostics(0.75).unwrap();
        assert!(v_tail.velocity.high_frequency_fraction > 0.999_999);
        assert!(v_tail.x.high_frequency_fraction < 1.0e-12);
    }

    #[test]
    fn spectral_tail_fraction_is_validated() {
        let config = test_config(16, 32);
        let initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        let solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        for invalid in [0.0, -0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                solver.spectral_tail_diagnostics(invalid),
                Err(SolverError::InvalidSpectralTailFraction)
            ));
        }
    }

    #[test]
    fn zero_padding_preserves_band_limited_trigonometric_interpolant() {
        let source_len = 16;
        let target_len = 64;
        let samples: Vec<f64> = (0..source_len)
            .map(|index| {
                let x = index as f64 / source_len as f64;
                0.7 + 0.4 * (TAU * 3.0 * x).cos() - 0.2 * (TAU * 5.0 * x).sin()
            })
            .collect();
        let refined = zero_pad_periodic_line(&samples, target_len).unwrap();

        for (index, expected) in samples.iter().enumerate() {
            let actual = refined[index * (target_len / source_len)];
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "old node {index}: {actual} != {expected}"
            );
        }
        for (index, actual) in refined.iter().enumerate() {
            let x = index as f64 / target_len as f64;
            let expected = 0.7 + 0.4 * (TAU * 3.0 * x).cos() - 0.2 * (TAU * 5.0 * x).sin();
            assert!((actual - expected).abs() < 2.0e-12);
        }
    }

    #[test]
    fn zero_padding_splits_source_nyquist_mode_symmetrically() {
        let source_len = 16;
        let target_len = 64;
        let samples: Vec<f64> = (0..source_len)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let refined = zero_pad_periodic_line(&samples, target_len).unwrap();

        for (index, actual) in refined.iter().enumerate() {
            let x = index as f64 / target_len as f64;
            let expected = (TAU * (source_len / 2) as f64 * x).cos();
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "refined node {index}: {actual} != {expected}"
            );
        }
        for index in 0..source_len {
            assert!((refined[index * 4] - samples[index]).abs() < 2.0e-12);
        }
    }

    #[test]
    fn zero_padding_preserves_constant_lines_and_equal_grid_is_identity() {
        let samples = vec![2.5; 8];
        let same = zero_pad_periodic_line(&samples, 8).unwrap();
        let refined = zero_pad_periodic_line(&samples, 32).unwrap();
        assert_eq!(same, samples);
        for value in refined {
            assert!((value - 2.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn zero_padding_rejects_invalid_grids_and_non_finite_samples() {
        for (source_len, target_len) in [(3, 8), (12, 32), (16, 8), (16, 48)] {
            let samples = vec![0.0; source_len];
            assert!(matches!(
                zero_pad_periodic_line(&samples, target_len),
                Err(SolverError::InvalidInterpolationGrid { .. })
            ));
        }
        let mut samples = vec![0.0; 8];
        samples[3] = f64::NAN;
        assert!(matches!(
            zero_pad_periodic_line(&samples, 32),
            Err(SolverError::NonFiniteInterpolationInput { index: 3 })
        ));
    }

    #[test]
    fn low_spectral_tail_does_not_hide_truncated_velocity_domain() {
        let mut config = test_config(64, 128);
        config.velocity_min = -4.0;
        config.velocity_max = 4.0;
        let initial = maxwellian_perturbation(config, 1.0e-3, 1, 1.0).unwrap();
        let solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let physical = solver.diagnostics().unwrap();
        let tail = solver.spectral_tail_diagnostics(0.75).unwrap();

        assert!(physical.velocity_boundary_ratio > 1.0e-4);
        assert!(tail.velocity.high_frequency_fraction < 1.0e-8);
    }

    #[test]
    fn invalid_time_steps_are_rejected() {
        let config = test_config(16, 32);
        let initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        assert!(matches!(
            solver.step(0.0),
            Err(SolverError::InvalidTimeStep)
        ));
        assert!(matches!(
            solver.advance_free_streaming(f64::NAN),
            Err(SolverError::InvalidTimeStep)
        ));
    }

    #[test]
    fn invalid_grids_and_non_finite_inputs_fail_closed() {
        let mut config = test_config(16, 32);
        config.nx = 12;
        assert!(matches!(
            config.validate(),
            Err(SolverError::InvalidGrid { .. })
        ));
        let config = test_config(16, 32);
        let mut initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        initial[3] = f64::INFINITY;
        assert!(matches!(
            VlasovPoisson1d1v::new(config, initial),
            Err(SolverError::NonFiniteDistribution { index: 3 })
        ));
    }

    #[test]
    fn electric_mode_coefficient_retains_phase_and_matches_amplitude() {
        let config = test_config(32, 64);
        let initial = maxwellian_perturbation(config, 1.0e-3, 1, 1.0).unwrap();
        let solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        let coefficient = solver.electric_mode_coefficient(1).unwrap();

        assert!(coefficient.real.abs() < 1.0e-14);
        assert!(coefficient.imaginary > 0.0);
        assert!(
            (coefficient.amplitude() - solver.electric_mode_amplitude(1).unwrap()).abs() < 1.0e-15
        );
        assert!(coefficient.phase().is_finite());
    }

    #[test]
    fn electric_mode_amplitude_is_finite_and_mode_checked() {
        let config = test_config(32, 64);
        let initial = maxwellian_perturbation(config, 1.0e-3, 1, 1.0).unwrap();
        let solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        assert!(solver.electric_mode_amplitude(1).unwrap() > 0.0);
        assert!(matches!(
            solver.electric_mode_amplitude(16),
            Err(SolverError::InvalidElectricMode { .. })
        ));
    }
}
