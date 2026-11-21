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
            return Err(SolverError::InvalidGrid { axis: "x", value: self.nx });
        }
        if self.nv < 4 || !self.nv.is_power_of_two() {
            return Err(SolverError::InvalidGrid { axis: "v", value: self.nv });
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
        if !self.charge_to_mass.is_finite() {
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

    pub fn dx(self) -> f64 { self.domain_length / self.nx as f64 }
    pub fn dv(self) -> f64 { (self.velocity_max - self.velocity_min) / self.nv as f64 }
    pub fn x(self, ix: usize) -> f64 { ix as f64 * self.dx() }
    pub fn velocity(self, iv: usize) -> f64 { self.velocity_min + iv as f64 * self.dv() }
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

#[derive(Debug, Clone, PartialEq)]
pub enum SolverError {
    InvalidGrid { axis: &'static str, value: usize },
    InvalidDomainLength,
    InvalidVelocityInterval,
    InvalidChargeToMass,
    InvalidPermittivity,
    InvalidBackgroundDensity,
    DistributionLengthMismatch { expected: usize, actual: usize },
    NonFiniteDistribution { index: usize },
    InvalidTimeStep,
    InvalidThermalSpeed,
    InvalidPerturbationAmplitude,
    InvalidSpatialMode { mode: usize, maximum_exclusive: usize },
    InvalidElectricMode { mode: usize, maximum_exclusive: usize },
    NonFiniteState,
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid { axis, value } => write!(f, "{axis}-grid size {value} must be a power of two >= 4"),
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::InvalidVelocityInterval => write!(f, "velocity interval endpoints must be finite and increasing"),
            Self::InvalidChargeToMass => write!(f, "charge-to-mass ratio must be finite"),
            Self::InvalidPermittivity => write!(f, "permittivity must be finite and positive"),
            Self::InvalidBackgroundDensity => write!(f, "background density must be finite and non-negative"),
            Self::DistributionLengthMismatch { expected, actual } => write!(f, "distribution has {actual} values; expected {expected} (nx * nv)"),
            Self::NonFiniteDistribution { index } => write!(f, "distribution value {index} is non-finite"),
            Self::InvalidTimeStep => write!(f, "time step must be finite and positive"),
            Self::InvalidThermalSpeed => write!(f, "thermal speed must be finite and positive"),
            Self::InvalidPerturbationAmplitude => write!(f, "Maxwellian perturbation amplitude must satisfy |alpha| < 1"),
            Self::InvalidSpatialMode { mode, maximum_exclusive } => write!(f, "spatial mode {mode} must be positive and less than {maximum_exclusive}"),
            Self::InvalidElectricMode { mode, maximum_exclusive } => write!(f, "electric mode {mode} must be positive and less than {maximum_exclusive}"),
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
        return Err(SolverError::InvalidSpatialMode { mode, maximum_exclusive: config.nx / 2 });
    }

    let mut maxwellian = Vec::with_capacity(config.nv);
    for iv in 0..config.nv {
        let velocity = config.velocity(iv);
        maxwellian.push((-0.5 * (velocity / thermal_speed).powi(2)).exp()
            / (TAU.sqrt() * thermal_speed));
    }
    let integral = maxwellian.iter().sum::<f64>() * config.dv();
    if !integral.is_finite() || integral <= 0.0 {
        return Err(SolverError::NonFiniteState);
    }
    for value in &mut maxwellian { *value /= integral; }

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
            return Err(SolverError::DistributionLengthMismatch { expected, actual: distribution.len() });
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

    pub fn config(&self) -> VlasovPoissonConfig { self.config }
    pub fn distribution(&self) -> &[f64] { &self.distribution }
    pub fn electric_field(&self) -> &[f64] { &self.electric_field }
    pub fn time(&self) -> f64 { self.time }

    /// Advance only the free-streaming subflow. Exposed for manufactured
    /// characteristic-solution tests; this deliberately omits self-fields.
    pub fn advance_free_streaming(&mut self, dt: f64) -> Result<(), SolverError> {
        validate_dt(dt)?;
        self.advect_x(dt)?;
        self.time += dt;
        if !self.time.is_finite() { return Err(SolverError::NonFiniteState); }
        self.refresh_electric_field()?;
        self.ensure_finite_state()
    }

    pub fn step(&mut self, dt: f64) -> Result<(), SolverError> {
        validate_dt(dt)?;
        self.advect_x(0.5 * dt)?;
        self.refresh_electric_field()?;
        self.advect_velocity(dt)?;
        self.advect_x(0.5 * dt)?;
        self.time += dt;
        if !self.time.is_finite() { return Err(SolverError::NonFiniteState); }
        self.refresh_electric_field()?;
        self.ensure_finite_state()
    }

    /// Magnitude of the positive spatial Fourier coefficient (not cosine
    /// peak amplitude) for the requested electric-field mode.
    pub fn electric_mode_amplitude(&self, mode: usize) -> Result<f64, SolverError> {
        if mode == 0 || mode >= self.config.nx / 2 {
            return Err(SolverError::InvalidElectricMode { mode, maximum_exclusive: self.config.nx / 2 });
        }
        let mut spectrum: Vec<Complex> = self.electric_field.iter()
            .map(|value| Complex::new(*value, 0.0)).collect();
        fft(&mut spectrum, false);
        Ok(spectrum[mode].norm() / self.config.nx as f64)
    }

    pub fn diagnostics(&self) -> Result<VlasovPoissonDiagnostics, SolverError> {
        let config = self.config;
        let dx = config.dx();
        let dv = config.dv();
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
                mass += value * dx * dv;
                momentum += value * velocity * dx * dv;
                kinetic_energy += 0.5 * value * velocity * velocity * dx * dv;
                minimum_distribution = minimum_distribution.min(value);
                maximum_abs_distribution = maximum_abs_distribution.max(value.abs());
                if iv == 0 || iv + 1 == config.nv {
                    boundary_abs = boundary_abs.max(value.abs());
                }
            }
        }
        let field_energy = 0.5 * self.electric_field.iter()
            .map(|value| value * value * dx).sum::<f64>();
        let rho = self.charge_density()?;
        let net_charge = rho.iter().sum::<f64>() * dx;
        let max_poisson_residual = self.max_poisson_residual(&rho)?;
        let velocity_boundary_ratio = if maximum_abs_distribution > 0.0 {
            boundary_abs / maximum_abs_distribution
        } else { 0.0 };
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
            result.time, result.total_mass, result.momentum, result.kinetic_energy,
            result.field_energy, result.total_energy, result.net_charge,
            result.max_poisson_residual, result.minimum_distribution,
            result.velocity_boundary_ratio,
        ].iter().any(|value| !value.is_finite()) {
            return Err(SolverError::NonFiniteState);
        }
        Ok(result)
    }

    fn ensure_finite_state(&self) -> Result<(), SolverError> {
        if self.distribution.iter().any(|value| !value.is_finite())
            || self.electric_field.iter().any(|value| !value.is_finite()) {
            return Err(SolverError::NonFiniteState);
        }
        Ok(())
    }

    fn advect_x(&mut self, dt: f64) -> Result<(), SolverError> {
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
                } else { Complex::cis(phase) };
                *coefficient = coefficient.mul(multiplier);
            }
            fft(&mut line, true);
            for (ix, value) in line.iter().enumerate() {
                self.distribution[ix * config.nv + iv] = value.re;
            }
        }
        Ok(())
    }

    fn advect_velocity(&mut self, dt: f64) -> Result<(), SolverError> {
        let config = self.config;
        let period = config.velocity_max - config.velocity_min;
        let mut line = vec![Complex::ZERO; config.nv];
        for ix in 0..config.nx {
            let acceleration = config.charge_to_mass * self.electric_field[ix];
            for iv in 0..config.nv {
                line[iv] = Complex::new(self.distribution[ix * config.nv + iv], 0.0);
            }
            fft(&mut line, false);
            for (mode, coefficient) in line.iter_mut().enumerate() {
                let eta = signed_mode(mode, config.nv) as f64 * TAU / period;
                let phase = -eta * acceleration * dt;
                let multiplier = if mode == config.nv / 2 {
                    Complex::new(phase.cos(), 0.0)
                } else { Complex::cis(phase) };
                *coefficient = coefficient.mul(multiplier);
            }
            fft(&mut line, true);
            for (iv, value) in line.iter().enumerate() {
                self.distribution[ix * config.nv + iv] = value.re;
            }
        }
        Ok(())
    }

    fn charge_density(&self) -> Result<Vec<f64>, SolverError> {
        let config = self.config;
        let mut rho = vec![0.0; config.nx];
        for (ix, density) in rho.iter_mut().enumerate() {
            let number_density = self.distribution[ix * config.nv..(ix + 1) * config.nv]
                .iter().sum::<f64>() * config.dv();
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

    fn max_poisson_residual(&self, rho: &[f64]) -> Result<f64, SolverError> {
        let mut spectrum: Vec<Complex> = self.electric_field.iter()
            .map(|value| Complex::new(*value, 0.0)).collect();
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
        Ok(maximum)
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

    let mut spectrum: Vec<Complex> = charge_density.iter()
        .map(|value| Complex::new(*value, 0.0)).collect();
    fft(&mut spectrum, false);
    for mode in 0..config.nx {
        if mode == 0 || mode == config.nx / 2 {
            spectrum[mode] = Complex::ZERO;
        } else {
            let k = signed_mode(mode, config.nx) as f64 * TAU / config.domain_length;
            spectrum[mode] = spectrum[mode].mul(Complex::new(0.0, -1.0 / (k * config.permittivity)));
        }
    }
    fft(&mut spectrum, true);
    let field: Vec<f64> = spectrum.iter().map(|value| value.re).collect();
    if field.iter().any(|value| !value.is_finite()) {
        return Err(SolverError::NonFiniteState);
    }
    Ok(field)
}

fn validate_dt(dt: f64) -> Result<(), SolverError> {
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SolverError::InvalidTimeStep);
    }
    Ok(())
}

fn signed_mode(index: usize, length: usize) -> isize {
    if index <= length / 2 { index as isize } else { index as isize - length as isize }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex { re: f64, im: f64 }

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    fn new(re: f64, im: f64) -> Self { Self { re, im } }
    fn cis(angle: f64) -> Self { Self { re: angle.cos(), im: angle.sin() } }
    fn add(self, rhs: Self) -> Self { Self::new(self.re + rhs.re, self.im + rhs.im) }
    fn sub(self, rhs: Self) -> Self { Self::new(self.re - rhs.re, self.im - rhs.im) }
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.re * rhs.re - self.im * rhs.im, self.re * rhs.im + self.im * rhs.re)
    }
    fn mul_i(self, scalar: f64) -> Self { Self::new(-self.im * scalar, self.re * scalar) }
    fn scale(self, scalar: f64) -> Self { Self::new(self.re * scalar, self.im * scalar) }
    fn norm(self) -> f64 { self.re.hypot(self.im) }
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
        if index < reversed { values.swap(index, reversed); }
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
        if span == length { break; }
        span *= 2;
    }
    if inverse {
        let scale = 1.0 / length as f64;
        for value in values { *value = value.scale(scale); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(nx: usize, nv: usize) -> VlasovPoissonConfig {
        VlasovPoissonConfig {
            nx, nv,
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
        assert!(solver.electric_field().iter().all(|value| value.abs() < 1.0e-12));
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
                let characteristic = 1.0 + alpha * (k * (config.x(ix) - config.velocity(iv) * dt)).cos();
                let expected = maxwellian * characteristic;
                let actual = solver.distribution()[ix * config.nv + iv];
                maximum_error = maximum_error.max((actual - expected).abs());
            }
        }
        assert!(maximum_error < 2.0e-11, "free-streaming max error {maximum_error:e}");
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
        for ix in 0..config.nx {
            let expected = -amplitude / k * (k * config.x(ix)).sin();
            maximum_error = maximum_error.max((electric[ix] - expected).abs());
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
        for _ in 0..20 { solver.step(0.05).unwrap(); }
        let after = solver.diagnostics().unwrap();
        assert!((after.total_mass - before.total_mass).abs() < 5.0e-11);
        assert!(after.max_poisson_residual < 2.0e-11);
        assert!(after.total_energy.is_finite());
        assert!(after.velocity_boundary_ratio < 1.0e-5);
    }

    #[test]
    fn invalid_time_steps_are_rejected() {
        let config = test_config(16, 32);
        let initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        let mut solver = VlasovPoisson1d1v::new(config, initial).unwrap();
        assert!(matches!(solver.step(0.0), Err(SolverError::InvalidTimeStep)));
        assert!(matches!(solver.advance_free_streaming(f64::NAN), Err(SolverError::InvalidTimeStep)));
    }

    #[test]
    fn invalid_grids_and_non_finite_inputs_fail_closed() {
        let mut config = test_config(16, 32);
        config.nx = 12;
        assert!(matches!(config.validate(), Err(SolverError::InvalidGrid { .. })));
        let config = test_config(16, 32);
        let mut initial = maxwellian_perturbation(config, 0.0, 1, 1.0).unwrap();
        initial[3] = f64::INFINITY;
        assert!(matches!(
            VlasovPoisson1d1v::new(config, initial),
            Err(SolverError::NonFiniteDistribution { index: 3 })
        ));
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
