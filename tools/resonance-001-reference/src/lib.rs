#![forbid(unsafe_code)]

//! RESONANCE-001 analytical reference models for passive lumped RLC networks.
//! This is a closed-form oracle, not a circuit/field solver or physical measurement.
//! All quantities use SI units; frequency arguments are in hertz.

use std::f64::consts::TAU;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeriesRlcV1 {
    pub resistance_ohm: f64,
    pub inductance_h: f64,
    pub capacitance_f: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParallelRlcV1 {
    pub resistance_ohm: f64,
    pub inductance_h: f64,
    pub capacitance_f: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexValueV1 {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexValueV1 {
    pub fn magnitude(self) -> f64 { self.real.hypot(self.imaginary) }
    pub fn phase_rad(self) -> f64 { self.imaginary.atan2(self.real) }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResonantSummaryV1 {
    pub angular_frequency_rad_s: f64,
    pub frequency_hz: f64,
    pub quality_factor: f64,
    pub half_power_bandwidth_hz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledResonatorV1 {
    pub inductance_1_h: f64,
    pub capacitance_1_f: f64,
    pub inductance_2_h: f64,
    pub capacitance_2_f: f64,
    pub coupling_coefficient: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledModeSummaryV1 {
    pub isolated_frequency_1_hz: f64,
    pub isolated_frequency_2_hz: f64,
    pub mutual_inductance_h: f64,
    pub lower_mode_frequency_hz: f64,
    pub upper_mode_frequency_hz: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelErrorV1 {
    NonFiniteParameter,
    NonPositiveResistance,
    NonPositiveInductance,
    NonPositiveCapacitance,
    NonPositiveFrequency,
    InvalidCouplingCoefficient,
}

fn validate_rlc(r: f64, l: f64, c: f64) -> Result<(), ModelErrorV1> {
    if !r.is_finite() || !l.is_finite() || !c.is_finite() {
        return Err(ModelErrorV1::NonFiniteParameter);
    }
    if r <= 0.0 { return Err(ModelErrorV1::NonPositiveResistance); }
    if l <= 0.0 { return Err(ModelErrorV1::NonPositiveInductance); }
    if c <= 0.0 { return Err(ModelErrorV1::NonPositiveCapacitance); }
    Ok(())
}

fn validate_frequency(f: f64) -> Result<(), ModelErrorV1> {
    if !f.is_finite() { return Err(ModelErrorV1::NonFiniteParameter); }
    if f <= 0.0 { return Err(ModelErrorV1::NonPositiveFrequency); }
    Ok(())
}

impl SeriesRlcV1 {
    pub fn summary(self) -> Result<ResonantSummaryV1, ModelErrorV1> {
        validate_rlc(self.resistance_ohm, self.inductance_h, self.capacitance_f)?;
        let omega = 1.0 / (self.inductance_h * self.capacitance_f).sqrt();
        Ok(ResonantSummaryV1 {
            angular_frequency_rad_s: omega,
            frequency_hz: omega / TAU,
            quality_factor: omega * self.inductance_h / self.resistance_ohm,
            half_power_bandwidth_hz: self.resistance_ohm / (TAU * self.inductance_h),
        })
    }

    /// Input impedance: R + j(omega L - 1/(omega C)).
    pub fn impedance_at(self, frequency_hz: f64) -> Result<ComplexValueV1, ModelErrorV1> {
        validate_rlc(self.resistance_ohm, self.inductance_h, self.capacitance_f)?;
        validate_frequency(frequency_hz)?;
        let omega = TAU * frequency_hz;
        Ok(ComplexValueV1 {
            real: self.resistance_ohm,
            imaginary: omega * self.inductance_h - 1.0 / (omega * self.capacitance_f),
        })
    }
}

impl ParallelRlcV1 {
    pub fn summary(self) -> Result<ResonantSummaryV1, ModelErrorV1> {
        validate_rlc(self.resistance_ohm, self.inductance_h, self.capacitance_f)?;
        let omega = 1.0 / (self.inductance_h * self.capacitance_f).sqrt();
        Ok(ResonantSummaryV1 {
            angular_frequency_rad_s: omega,
            frequency_hz: omega / TAU,
            quality_factor: self.resistance_ohm * (self.capacitance_f / self.inductance_h).sqrt(),
            half_power_bandwidth_hz: 1.0 / (TAU * self.resistance_ohm * self.capacitance_f),
        })
    }

    /// Input impedance is the reciprocal of Y = 1/R + j(omega C - 1/(omega L)).
    pub fn impedance_at(self, frequency_hz: f64) -> Result<ComplexValueV1, ModelErrorV1> {
        validate_rlc(self.resistance_ohm, self.inductance_h, self.capacitance_f)?;
        validate_frequency(frequency_hz)?;
        let omega = TAU * frequency_hz;
        let g = 1.0 / self.resistance_ohm;
        let b = omega * self.capacitance_f - 1.0 / (omega * self.inductance_h);
        let denominator = g.mul_add(g, b * b);
        Ok(ComplexValueV1 { real: g / denominator, imaginary: -b / denominator })
    }
}

impl CoupledResonatorV1 {
    pub fn summary(self) -> Result<CoupledModeSummaryV1, ModelErrorV1> {
        for value in [self.inductance_1_h, self.capacitance_1_f,
                      self.inductance_2_h, self.capacitance_2_f,
                      self.coupling_coefficient] {
            if !value.is_finite() { return Err(ModelErrorV1::NonFiniteParameter); }
        }
        if self.inductance_1_h <= 0.0 || self.inductance_2_h <= 0.0 {
            return Err(ModelErrorV1::NonPositiveInductance);
        }
        if self.capacitance_1_f <= 0.0 || self.capacitance_2_f <= 0.0 {
            return Err(ModelErrorV1::NonPositiveCapacitance);
        }
        let k = self.coupling_coefficient;
        if !(0.0..1.0).contains(&k) {
            return Err(ModelErrorV1::InvalidCouplingCoefficient);
        }

        let w1 = 1.0 / (self.inductance_1_h * self.capacitance_1_f).sqrt();
        let w2 = 1.0 / (self.inductance_2_h * self.capacitance_2_f).sqrt();
        let a = w1 * w1;
        let b = w2 * w2;
        let discriminant = ((a - b).powi(2) + 4.0 * k * k * a * b).sqrt();
        let denominator = 2.0 * (1.0 - k * k);
        let lower_w2 = (a + b - discriminant) / denominator;
        let upper_w2 = (a + b + discriminant) / denominator;
        if lower_w2 <= 0.0 || upper_w2 <= 0.0 {
            return Err(ModelErrorV1::InvalidCouplingCoefficient);
        }

        Ok(CoupledModeSummaryV1 {
            isolated_frequency_1_hz: w1 / TAU,
            isolated_frequency_2_hz: w2 / TAU,
            mutual_inductance_h: k * (self.inductance_1_h * self.inductance_2_h).sqrt(),
            lower_mode_frequency_hz: lower_w2.sqrt() / TAU,
            upper_mode_frequency_hz: upper_w2.sqrt() / TAU,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!((actual - expected).abs() <= 1.0e-12 * scale,
            "actual={actual:.15e}, expected={expected:.15e}");
    }

    #[test]
    fn series_rlc_known_answer() {
        let m = SeriesRlcV1 { resistance_ohm: 10.0, inductance_h: 0.01, capacitance_f: 1.0e-6 };
        let s = m.summary().unwrap();
        close(s.angular_frequency_rad_s, 10_000.0);
        close(s.frequency_hz, 1_591.5494309189535);
        close(s.quality_factor, 10.0);
        close(s.half_power_bandwidth_hz, 159.15494309189535);
        let z = m.impedance_at(s.frequency_hz).unwrap();
        close(z.real, 10.0);
        close(z.imaginary, 0.0);
    }

    #[test]
    fn parallel_rlc_known_answer() {
        let m = ParallelRlcV1 { resistance_ohm: 1_000.0, inductance_h: 0.01, capacitance_f: 1.0e-6 };
        let s = m.summary().unwrap();
        close(s.angular_frequency_rad_s, 10_000.0);
        close(s.frequency_hz, 1_591.5494309189535);
        close(s.quality_factor, 10.0);
        close(s.half_power_bandwidth_hz, 159.15494309189535);
        let z = m.impedance_at(s.frequency_hz).unwrap();
        close(z.real, 1_000.0);
        close(z.imaginary, 0.0);
    }

    #[test]
    fn uncoupled_modes_recover_both_isolated_frequencies() {
        let m = CoupledResonatorV1 {
            inductance_1_h: 0.01, capacitance_1_f: 1.0e-6,
            inductance_2_h: 0.0225, capacitance_2_f: 2.0e-6,
            coupling_coefficient: 0.0,
        };
        let s = m.summary().unwrap();
        let mut frequencies = [s.isolated_frequency_1_hz, s.isolated_frequency_2_hz];
        frequencies.sort_by(f64::total_cmp);
        close(s.lower_mode_frequency_hz, frequencies[0]);
        close(s.upper_mode_frequency_hz, frequencies[1]);
        close(s.mutual_inductance_h, 0.0);
    }

    #[test]
    fn identical_resonators_split_by_expected_coupling_law() {
        let m = CoupledResonatorV1 {
            inductance_1_h: 0.01, capacitance_1_f: 1.0e-6,
            inductance_2_h: 0.01, capacitance_2_f: 1.0e-6,
            coupling_coefficient: 0.2,
        };
        let s = m.summary().unwrap();
        close(s.lower_mode_frequency_hz, s.isolated_frequency_1_hz / 1.2_f64.sqrt());
        close(s.upper_mode_frequency_hz, s.isolated_frequency_1_hz / 0.8_f64.sqrt());
        close(s.mutual_inductance_h, 0.002);
    }

    #[test]
    fn invalid_physical_parameters_are_rejected() {
        let bad = SeriesRlcV1 { resistance_ohm: 0.0, inductance_h: 0.01, capacitance_f: 1.0e-6 };
        assert_eq!(bad.summary(), Err(ModelErrorV1::NonPositiveResistance));
        let bad_k = CoupledResonatorV1 {
            inductance_1_h: 0.01, capacitance_1_f: 1.0e-6,
            inductance_2_h: 0.01, capacitance_2_f: 1.0e-6,
            coupling_coefficient: 1.0,
        };
        assert_eq!(bad_k.summary(), Err(ModelErrorV1::InvalidCouplingCoefficient));
    }

    #[test]
    fn zero_and_nonfinite_frequency_are_rejected() {
        let m = SeriesRlcV1 { resistance_ohm: 10.0, inductance_h: 0.01, capacitance_f: 1.0e-6 };
        assert_eq!(m.impedance_at(0.0), Err(ModelErrorV1::NonPositiveFrequency));
        assert_eq!(m.impedance_at(f64::NAN), Err(ModelErrorV1::NonFiniteParameter));
    }
}
