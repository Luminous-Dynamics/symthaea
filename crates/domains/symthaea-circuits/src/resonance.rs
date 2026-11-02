// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Validated analytical resonance references for passive series-RLC circuits.
//!
//! This module complements the legacy scalar helpers in crate::ac with
//! parameter validation and explicit derived quantities. It is intended for
//! analytical reference cases and benchmark oracles, not as a replacement for
//! a circuit simulator or a physical measurement.
//!
//! All quantities use SI units. The model assumes linear, lumped, passive
//! components and sinusoidal steady state for impedance calculations.
//! Distributed effects, radiation, nonlinear components, parasitics not
//! represented by the supplied R/L/C values, and transformer/core losses are
//! outside this model.

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// Failure to construct or evaluate a physically admissible reference model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelError {
    /// A supplied parameter is NaN or infinite.
    NonFinite(&'static str),
    /// A parameter that must be strictly positive is zero or negative.
    NonPositive(&'static str),
    /// A derived quantity is outside the representable finite f64 range.
    NonRepresentable(&'static str),
    /// The magnitude of the coupling coefficient must be strictly below one.
    CouplingCoefficientOutOfRange,
    /// The coupled-mode discriminant is invalid beyond round-off tolerance.
    InvalidDiscriminant,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::NonPositive(name) => write!(f, "{name} must be greater than zero"),
            Self::NonRepresentable(name) => {
                write!(f, "derived quantity {name} is not finitely representable")
            }
            Self::CouplingCoefficientOutOfRange => {
                write!(f, "absolute coupling coefficient must be less than one")
            }
            Self::InvalidDiscriminant => {
                write!(f, "coupled-mode discriminant is invalid")
            }
        }
    }
}

impl Error for ModelError {}

fn require_positive_finite(value: f64, name: &'static str) -> Result<(), ModelError> {
    if !value.is_finite() {
        return Err(ModelError::NonFinite(name));
    }
    if value <= 0.0 {
        return Err(ModelError::NonPositive(name));
    }
    Ok(())
}

fn require_finite(value: f64, name: &'static str) -> Result<(), ModelError> {
    if !value.is_finite() {
        return Err(ModelError::NonFinite(name));
    }
    Ok(())
}

/// A validated passive, lumped, series-RLC resonator.
///
/// This ideal model has a positive resistance, inductance and capacitance.
/// An ideal lossless resonator (R = 0) is deliberately not admitted because
/// its quality factor is infinite and cannot be represented as a finite
/// benchmark result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesRlc {
    resistance_ohm: f64,
    inductance_henry: f64,
    capacitance_farads: f64,
}

/// Resonance metrics for a validated series-RLC model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonanceMetrics {
    /// Undamped natural resonance frequency, in hertz.
    pub resonant_frequency_hz: f64,
    /// Series quality factor Q = omega_0 L / R.
    pub quality_factor: f64,
    /// Ideal series-RLC half-power bandwidth R / (2 pi L), in hertz.
    pub bandwidth_hz: f64,
}

/// Complex impedance represented by real and imaginary components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexImpedance {
    /// Resistive component, in ohms.
    pub real_ohm: f64,
    /// Reactive component, in ohms.
    pub imaginary_ohm: f64,
}

impl ComplexImpedance {
    /// Magnitude of the impedance, in ohms.
    pub fn magnitude_ohm(self) -> f64 {
        self.real_ohm.hypot(self.imaginary_ohm)
    }

    /// Principal impedance phase, in radians.
    pub fn phase_rad(self) -> f64 {
        self.imaginary_ohm.atan2(self.real_ohm)
    }
}

impl SeriesRlc {
    /// Construct a validated passive series-RLC resonator.
    pub fn new(
        resistance_ohm: f64,
        inductance_henry: f64,
        capacitance_farads: f64,
    ) -> Result<Self, ModelError> {
        require_positive_finite(resistance_ohm, "resistance_ohm")?;
        require_positive_finite(inductance_henry, "inductance_henry")?;
        require_positive_finite(capacitance_farads, "capacitance_farads")?;

        let model = Self {
            resistance_ohm,
            inductance_henry,
            capacitance_farads,
        };

        // Reject extreme inputs whose key reference metrics underflow or
        // overflow instead of allowing infinities/zeros into benchmark data.
        let metrics = model.metrics_unchecked();
        require_positive_finite(
            metrics.resonant_frequency_hz,
            "resonant_frequency_hz",
        )
        .map_err(|_| ModelError::NonRepresentable("resonant_frequency_hz"))?;
        require_positive_finite(metrics.quality_factor, "quality_factor")
            .map_err(|_| ModelError::NonRepresentable("quality_factor"))?;
        require_positive_finite(metrics.bandwidth_hz, "bandwidth_hz")
            .map_err(|_| ModelError::NonRepresentable("bandwidth_hz"))?;

        Ok(model)
    }

    /// Resistance in ohms.
    pub fn resistance_ohm(self) -> f64 {
        self.resistance_ohm
    }

    /// Inductance in henries.
    pub fn inductance_henry(self) -> f64 {
        self.inductance_henry
    }

    /// Capacitance in farads.
    pub fn capacitance_farads(self) -> f64 {
        self.capacitance_farads
    }

    /// Calculate natural resonance, quality factor and ideal half-power
    /// bandwidth for this model.
    pub fn metrics(self) -> ResonanceMetrics {
        self.metrics_unchecked()
    }

    fn metrics_unchecked(self) -> ResonanceMetrics {
        // Divide sequentially to avoid forming L*C, which may overflow even
        // when the final frequency is representable.
        let omega_0 =
            (1.0 / self.inductance_henry.sqrt()) / self.capacitance_farads.sqrt();
        let resonant_frequency_hz = omega_0 / TAU;
        let quality_factor = (omega_0 * self.inductance_henry) / self.resistance_ohm;
        let bandwidth_hz = (self.resistance_ohm / self.inductance_henry) / TAU;

        ResonanceMetrics {
            resonant_frequency_hz,
            quality_factor,
            bandwidth_hz,
        }
    }

    /// Calculate steady-state input impedance at a positive frequency.
    ///
    /// Z(f) = R + j(2 pi f L - 1/(2 pi f C)).
    pub fn impedance_at_hz(self, frequency_hz: f64) -> Result<ComplexImpedance, ModelError> {
        require_positive_finite(frequency_hz, "frequency_hz")?;

        let omega = TAU * frequency_hz;
        require_positive_finite(omega, "angular_frequency_rad_s")
            .map_err(|_| ModelError::NonRepresentable("angular_frequency_rad_s"))?;

        let inductive_ohm = omega * self.inductance_henry;
        let capacitive_ohm = 1.0 / (omega * self.capacitance_farads);
        let imaginary_ohm = inductive_ohm - capacitive_ohm;

        if !inductive_ohm.is_finite()
            || !capacitive_ohm.is_finite()
            || !imaginary_ohm.is_finite()
        {
            return Err(ModelError::NonRepresentable("impedance"));
        }

        Ok(ComplexImpedance {
            real_ohm: self.resistance_ohm,
            imaginary_ohm,
        })
    }
}

/// Two series-RLC resonators coupled through mutual inductance.
///
/// The coupled-mode calculation is lossless (R is not used to determine the
/// natural-mode frequencies). It models two linear inductors with mutual
/// inductance M = k * sqrt(L1 * L2) and independent series capacitances.
/// The sign of k changes the mode shape but not the two eigenfrequencies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledSeriesRlc {
    first: SeriesRlc,
    second: SeriesRlc,
    coupling_coefficient: f64,
    mutual_inductance_henry: f64,
}

/// The lower and upper natural-mode frequencies of coupled resonators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledModeFrequencies {
    /// Lower natural-mode frequency, in hertz.
    pub lower_hz: f64,
    /// Upper natural-mode frequency, in hertz.
    pub upper_hz: f64,
}

impl CoupledSeriesRlc {
    /// Construct two validated resonators with a dimensionless signed
    /// coupling coefficient k, where M = k * sqrt(L1 * L2).
    ///
    /// A passive, nonsingular inductance matrix requires |k| < 1.
    pub fn new(
        first: SeriesRlc,
        second: SeriesRlc,
        coupling_coefficient: f64,
    ) -> Result<Self, ModelError> {
        require_finite(coupling_coefficient, "coupling_coefficient")?;
        if coupling_coefficient.abs() >= 1.0 {
            return Err(ModelError::CouplingCoefficientOutOfRange);
        }

        let geometric_mean =
            first.inductance_henry.sqrt() * second.inductance_henry.sqrt();
        let mutual_inductance_henry = coupling_coefficient * geometric_mean;
        if !mutual_inductance_henry.is_finite()
            || (coupling_coefficient != 0.0 && mutual_inductance_henry == 0.0)
        {
            return Err(ModelError::NonRepresentable("mutual_inductance_henry"));
        }

        Ok(Self {
            first,
            second,
            coupling_coefficient,
            mutual_inductance_henry,
        })
    }

    /// Signed dimensionless coupling coefficient.
    pub fn coupling_coefficient(self) -> f64 {
        self.coupling_coefficient
    }

    /// Mutual inductance in henries.
    pub fn mutual_inductance_henry(self) -> f64 {
        self.mutual_inductance_henry
    }

    /// References to the two resonator models.
    pub fn resonators(self) -> (SeriesRlc, SeriesRlc) {
        (self.first, self.second)
    }

    /// Compute the two undamped natural-mode frequencies, sorted ascending.
    ///
    /// For w1^2 = 1/(L1 C1), w2^2 = 1/(L2 C2), and
    /// k = M/sqrt(L1 L2), the squared modes are the positive roots of:
    ///
    /// (1-k^2) x^2 - (w1^2+w2^2) x + w1^2 w2^2 = 0.
    ///
    /// The calculation is scaled to avoid forming L1*L2, C1*C2 or
    /// unscaled discriminants that can overflow for otherwise valid inputs.
    pub fn mode_frequencies_hz(self) -> Result<CoupledModeFrequencies, ModelError> {
        let first_omega = natural_angular_frequency(self.first)?;
        let second_omega = natural_angular_frequency(self.second)?;
        let first_sq = first_omega * first_omega;
        let second_sq = second_omega * second_omega;

        if !first_sq.is_finite() || first_sq <= 0.0 {
            return Err(ModelError::NonRepresentable("first_omega_squared"));
        }
        if !second_sq.is_finite() || second_sq <= 0.0 {
            return Err(ModelError::NonRepresentable("second_omega_squared"));
        }

        let scale = first_sq.max(second_sq);
        let u = first_sq / scale;
        let v = second_sq / scale;
        let sum = u + v;
        let a = 1.0 - self.coupling_coefficient * self.coupling_coefficient;

        let mut discriminant = 1.0 - (4.0 * a * u * v) / (sum * sum);
        if discriminant < 0.0 {
            // Allow only a small negative caused by floating-point round-off.
            if discriminant >= -32.0 * f64::EPSILON {
                discriminant = 0.0;
            } else {
                return Err(ModelError::InvalidDiscriminant);
            }
        }

        let q = 0.5 * sum * (1.0 + discriminant.sqrt());
        let high_sq = scale * (q / a);
        let low_sq = scale * ((u / q) * v);

        if !low_sq.is_finite()
            || !high_sq.is_finite()
            || low_sq <= 0.0
            || high_sq <= 0.0
        {
            return Err(ModelError::NonRepresentable("coupled_mode_frequency"));
        }

        let lower_hz = low_sq.sqrt() / TAU;
        let upper_hz = high_sq.sqrt() / TAU;
        if !lower_hz.is_finite()
            || !upper_hz.is_finite()
            || lower_hz <= 0.0
            || upper_hz <= 0.0
        {
            return Err(ModelError::NonRepresentable("coupled_mode_frequency_hz"));
        }

        Ok(CoupledModeFrequencies {
            lower_hz,
            upper_hz,
        })
    }
}

fn natural_angular_frequency(model: SeriesRlc) -> Result<f64, ModelError> {
    let omega =
        (1.0 / model.inductance_henry.sqrt()) / model.capacitance_farads.sqrt();
    require_positive_finite(omega, "natural_angular_frequency")
        .map_err(|_| ModelError::NonRepresentable("natural_angular_frequency"))?;
    Ok(omega)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SeriesRlc {
        SeriesRlc::new(50.0, 1e-3, 1e-6).expect("valid series-RLC fixture")
    }

    #[test]
    fn series_rlc_known_answer_metrics() {
        let metrics = fixture().metrics();
        assert!(
            (metrics.resonant_frequency_hz - 5032.9212104487).abs() < 1e-8,
            "f0={}",
            metrics.resonant_frequency_hz
        );
        assert!(
            (metrics.quality_factor - 0.6324555320336759).abs() < 1e-12,
            "Q={}",
            metrics.quality_factor
        );
        assert!(
            (metrics.bandwidth_hz - 7957.7471545948).abs() < 1e-8,
            "BW={}",
            metrics.bandwidth_hz
        );
    }

    #[test]
    fn impedance_is_resistive_at_resonance() {
        let resonator = fixture();
        let metrics = resonator.metrics();
        let impedance = resonator
            .impedance_at_hz(metrics.resonant_frequency_hz)
            .expect("finite resonance impedance");

        assert!(impedance.imaginary_ohm.abs() < 1e-9);
        assert!((impedance.magnitude_ohm() - 50.0).abs() < 1e-9);
        assert!(impedance.phase_rad().abs() < 1e-12);
    }

    #[test]
    fn half_power_bandwidth_matches_r_over_two_pi_l() {
        let resonator = fixture();
        let r = resonator.resistance_ohm();
        let l = resonator.inductance_henry();
        let c = resonator.capacitance_farads();

        // Solve |X_L - X_C| = R at the two half-power frequencies.
        let root = (r * r + 4.0 * l / c).sqrt();
        let omega_low = (root - r) / (2.0 * l);
        let omega_high = (root + r) / (2.0 * l);
        let f_low = omega_low / TAU;
        let f_high = omega_high / TAU;

        let lower = resonator.impedance_at_hz(f_low).expect("lower half-power");
        let upper = resonator.impedance_at_hz(f_high).expect("upper half-power");
        let target = 2.0_f64.sqrt() * r;

        assert!((lower.magnitude_ohm() - target).abs() < 1e-8);
        assert!((upper.magnitude_ohm() - target).abs() < 1e-8);
        assert!(
            ((f_high - f_low) - resonator.metrics().bandwidth_hz).abs() < 1e-8
        );
    }

    #[test]
    fn rejects_invalid_passive_parameters() {
        assert!(matches!(
            SeriesRlc::new(0.0, 1e-3, 1e-6),
            Err(ModelError::NonPositive("resistance_ohm"))
        ));
        assert!(matches!(
            SeriesRlc::new(50.0, f64::NAN, 1e-6),
            Err(ModelError::NonFinite("inductance_henry"))
        ));
        assert!(matches!(
            SeriesRlc::new(50.0, 1e-3, -1.0),
            Err(ModelError::NonPositive("capacitance_farads"))
        ));
    }

    #[test]
    fn rejects_invalid_frequency_and_coupling() {
        let resonator = fixture();
        assert!(matches!(
            resonator.impedance_at_hz(0.0),
            Err(ModelError::NonPositive("frequency_hz"))
        ));

        assert!(matches!(
            CoupledSeriesRlc::new(resonator, resonator, 1.0),
            Err(ModelError::CouplingCoefficientOutOfRange)
        ));
        assert!(matches!(
            CoupledSeriesRlc::new(resonator, resonator, f64::NAN),
            Err(ModelError::NonFinite("coupling_coefficient"))
        ));
    }

    #[test]
    fn identical_coupled_resonators_have_expected_mode_splitting() {
        let resonator = fixture();
        let k = 0.25;
        let coupled =
            CoupledSeriesRlc::new(resonator, resonator, k).expect("valid coupled fixture");
        let modes = coupled.mode_frequencies_hz().expect("finite modes");
        let f0 = resonator.metrics().resonant_frequency_hz;

        let expected_lower = f0 / (1.0 + k).sqrt();
        let expected_upper = f0 / (1.0 - k).sqrt();

        assert!((modes.lower_hz - expected_lower).abs() < 1e-9);
        assert!((modes.upper_hz - expected_upper).abs() < 1e-9);
        assert!(modes.lower_hz < modes.upper_hz);
    }

    #[test]
    fn detuned_coupled_resonators_match_known_mode_frequencies() {
        let first = SeriesRlc::new(10.0, 1e-3, 1e-6).expect("first resonator");
        let second = SeriesRlc::new(20.0, 2e-3, 1e-6).expect("second resonator");
        let k = 0.3;
        let coupled =
            CoupledSeriesRlc::new(first, second, k).expect("valid detuned pair");
        let modes = coupled.mode_frequencies_hz().expect("finite coupled modes");

        assert!((modes.lower_hz - 3427.8479810908657).abs() < 1e-8);
        assert!((modes.upper_hz - 5477.507940293483).abs() < 1e-8);
        assert!(
            (coupled.mutual_inductance_henry() - k * (2e-6_f64).sqrt()).abs() < 1e-15
        );
    }

    #[test]
    fn zero_coupling_recovers_independent_resonances() {
        let first = SeriesRlc::new(10.0, 1e-3, 1e-6).expect("first resonator");
        let second = SeriesRlc::new(20.0, 2e-3, 1e-6).expect("second resonator");
        let coupled =
            CoupledSeriesRlc::new(first, second, 0.0).expect("uncoupled pair");
        let modes = coupled.mode_frequencies_hz().expect("finite modes");

        let expected = [
            first.metrics().resonant_frequency_hz,
            second.metrics().resonant_frequency_hz,
        ];
        let expected_low = expected[0].min(expected[1]);
        let expected_high = expected[0].max(expected[1]);

        assert!((modes.lower_hz - expected_low).abs() < 1e-9);
        assert!((modes.upper_hz - expected_high).abs() < 1e-9);
    }

    #[test]
    fn coupling_sign_changes_mutual_inductance_sign_not_mode_frequencies() {
        let resonator = fixture();
        let positive =
            CoupledSeriesRlc::new(resonator, resonator, 0.2).expect("positive coupling");
        let negative =
            CoupledSeriesRlc::new(resonator, resonator, -0.2).expect("negative coupling");
        let positive_modes = positive.mode_frequencies_hz().expect("positive modes");
        let negative_modes = negative.mode_frequencies_hz().expect("negative modes");

        assert!(positive.mutual_inductance_henry() > 0.0);
        assert!(negative.mutual_inductance_henry() < 0.0);
        assert!((positive_modes.lower_hz - negative_modes.lower_hz).abs() < 1e-9);
        assert!((positive_modes.upper_hz - negative_modes.upper_hz).abs() < 1e-9);
    }
}
