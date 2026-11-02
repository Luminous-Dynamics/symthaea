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

    pub fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imaginary: self.imaginary + other.imaginary,
        }
    }

    pub fn multiply(self, other: Self) -> Self {
        Self {
            real: self.real * other.real - self.imaginary * other.imaginary,
            imaginary: self.real * other.imaginary + self.imaginary * other.real,
        }
    }

    pub fn scale(self, factor: f64) -> Self {
        Self {
            real: self.real * factor,
            imaginary: self.imaginary * factor,
        }
    }

    pub fn divide(self, other: Self) -> Result<Self, ModelErrorV1> {
        let denominator = other.real.mul_add(other.real, other.imaginary * other.imaginary);
        if denominator == 0.0 {
            return Err(ModelErrorV1::ZeroComplexDenominator);
        }
        if !denominator.is_finite() {
            return Err(ModelErrorV1::NonFiniteParameter);
        }
        let result = Self {
            real: (self.real * other.real + self.imaginary * other.imaginary) / denominator,
            imaginary: (self.imaginary * other.real - self.real * other.imaginary) / denominator,
        };
        if !result.real.is_finite() || !result.imaginary.is_finite() {
            return Err(ModelErrorV1::NonFiniteParameter);
        }
        Ok(result)
    }
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrivenCoupledSeriesRlcV1 {
    pub primary_resistance_ohm: f64,
    pub primary_inductance_h: f64,
    pub primary_capacitance_f: f64,
    pub secondary_resistance_ohm: f64,
    pub secondary_inductance_h: f64,
    pub secondary_capacitance_f: f64,
    /// Coupling magnitude in [0, 1).
    pub coupling_coefficient: f64,
    /// Dot-convention sign: +1 or -1. Magnitude is kept separate.
    pub secondary_winding_polarity: i8,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrivenCoupledResponseV1 {
    pub frequency_hz: f64,
    pub mutual_inductance_h: f64,
    pub primary_input_impedance_ohm: ComplexValueV1,
    /// Primary current divided by source voltage, in A/V.
    pub primary_current_per_volt_a_per_v: ComplexValueV1,
    /// Source active power divided by squared RMS source voltage.
    pub primary_input_power_w_per_v2: f64,
    /// Primary series-resistor dissipation divided by squared RMS source voltage.
    pub primary_resistor_power_w_per_v2: f64,
    /// Source power minus primary and secondary resistor dissipation.
    pub power_balance_residual_w_per_v2: f64,
    /// Secondary loop current divided by primary source voltage, in A/V.
    pub secondary_current_per_volt_a_per_v: ComplexValueV1,
    /// Voltage across the secondary series resistance divided by source voltage.
    pub secondary_resistor_voltage_per_volt: ComplexValueV1,
    /// Secondary resistor dissipation divided by primary source voltage squared.
    pub secondary_resistor_power_w_per_v2: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelErrorV1 {
    NonFiniteParameter,
    NonPositiveResistance,
    NonPositiveInductance,
    NonPositiveCapacitance,
    NonPositiveFrequency,
    InvalidCouplingCoefficient,
    InvalidCouplingPolarity,
    ZeroComplexDenominator,
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

impl DrivenCoupledSeriesRlcV1 {
    fn validate(self) -> Result<(), ModelErrorV1> {
        validate_rlc(
            self.primary_resistance_ohm,
            self.primary_inductance_h,
            self.primary_capacitance_f,
        )?;
        validate_rlc(
            self.secondary_resistance_ohm,
            self.secondary_inductance_h,
            self.secondary_capacitance_f,
        )?;
        if !self.coupling_coefficient.is_finite() {
            return Err(ModelErrorV1::NonFiniteParameter);
        }
        if !(0.0..1.0).contains(&self.coupling_coefficient) {
            return Err(ModelErrorV1::InvalidCouplingCoefficient);
        }
        if self.secondary_winding_polarity != 1 && self.secondary_winding_polarity != -1 {
            return Err(ModelErrorV1::InvalidCouplingPolarity);
        }
        Ok(())
    }

    /// Solve the sinusoidal steady-state response of two series RLC loops
    /// coupled by ideal mutual inductance. The primary loop is driven by a
    /// voltage source; the secondary loop is closed and has no independent
    /// source. Both loop resistances include all declared series damping/load.
    ///
    /// Z1 = R1 + j(omega L1 - 1/(omega C1))
    /// Z2 = R2 + j(omega L2 - 1/(omega C2))
    /// M = polarity * k * sqrt(L1 L2)
    /// Zin = Z1 + (omega M)^2 / Z2
    /// I2/V1 = -j omega M / (Z1 Z2 + (omega M)^2)
    ///
    /// Assumptions: linear lumped components, sinusoidal steady state, constant
    /// parameters, ideal mutual inductance and no distributed or radiative effects.
    pub fn response_at(
        self,
        frequency_hz: f64,
    ) -> Result<DrivenCoupledResponseV1, ModelErrorV1> {
        self.validate()?;
        validate_frequency(frequency_hz)?;
        let omega = TAU * frequency_hz;
        if !omega.is_finite() {
            return Err(ModelErrorV1::NonFiniteParameter);
        }

        let mutual_inductance = self.secondary_winding_polarity as f64
            * self.coupling_coefficient
            * (self.primary_inductance_h * self.secondary_inductance_h).sqrt();
        let omega_m = omega * mutual_inductance;

        let z1 = ComplexValueV1 {
            real: self.primary_resistance_ohm,
            imaginary: omega * self.primary_inductance_h
                - 1.0 / (omega * self.primary_capacitance_f),
        };
        let z2 = ComplexValueV1 {
            real: self.secondary_resistance_ohm,
            imaginary: omega * self.secondary_inductance_h
                - 1.0 / (omega * self.secondary_capacitance_f),
        };

        let reflected_impedance = ComplexValueV1 {
            real: omega_m * omega_m,
            imaginary: 0.0,
        }.divide(z2)?;
        let input_impedance = z1.add(reflected_impedance);

        let transfer_denominator = z1.multiply(z2).add(ComplexValueV1 {
            real: omega_m * omega_m,
            imaginary: 0.0,
        });
        let secondary_current_per_volt = ComplexValueV1 {
            real: 0.0,
            imaginary: -omega_m,
        }.divide(transfer_denominator)?;
        let primary_current_per_volt = ComplexValueV1 {
            real: 1.0,
            imaginary: 0.0,
        }.divide(input_impedance)?;
        let secondary_resistor_voltage_per_volt =
            secondary_current_per_volt.scale(self.secondary_resistance_ohm);

        // RMS phasors: P_in / |V_source|^2 = Re(I_primary / V_source).
        let primary_input_power_per_v2 = primary_current_per_volt.real;
        let primary_resistor_power_per_v2 =
            primary_current_per_volt.magnitude().powi(2) * self.primary_resistance_ohm;
        let secondary_resistor_power_per_v2 =
            secondary_current_per_volt.magnitude().powi(2) * self.secondary_resistance_ohm;
        let power_balance_residual_per_v2 = primary_input_power_per_v2
            - primary_resistor_power_per_v2
            - secondary_resistor_power_per_v2;

        if !input_impedance.real.is_finite()
            || !input_impedance.imaginary.is_finite()
            || !primary_current_per_volt.real.is_finite()
            || !primary_current_per_volt.imaginary.is_finite()
            || !primary_input_power_per_v2.is_finite()
            || !primary_resistor_power_per_v2.is_finite()
            || !secondary_resistor_power_per_v2.is_finite()
            || !power_balance_residual_per_v2.is_finite()
            || primary_input_power_per_v2 < 0.0
            || primary_resistor_power_per_v2 < 0.0
            || secondary_resistor_power_per_v2 < 0.0
        {
            return Err(ModelErrorV1::NonFiniteParameter);
        }

        Ok(DrivenCoupledResponseV1 {
            frequency_hz,
            mutual_inductance_h: mutual_inductance,
            primary_input_impedance_ohm: input_impedance,
            primary_current_per_volt_a_per_v: primary_current_per_volt,
            primary_input_power_w_per_v2: primary_input_power_per_v2,
            primary_resistor_power_w_per_v2: primary_resistor_power_per_v2,
            power_balance_residual_w_per_v2: power_balance_residual_per_v2,
            secondary_current_per_volt_a_per_v: secondary_current_per_volt,
            secondary_resistor_voltage_per_volt,
            secondary_resistor_power_w_per_v2: secondary_resistor_power_per_v2,
        })
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
        // Use the product-of-roots identity for the lower mode to avoid
        // catastrophic cancellation when the isolated resonances are far apart.
        let upper_w2 = (a + b + discriminant) / denominator;
        let lower_w2 = (2.0 * a * b) / (a + b + discriminant);
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
    fn driven_lossy_coupled_loops_match_resonant_known_answer() {
        let model = DrivenCoupledSeriesRlcV1 {
            primary_resistance_ohm: 10.0,
            primary_inductance_h: 0.01,
            primary_capacitance_f: 1.0e-6,
            secondary_resistance_ohm: 20.0,
            secondary_inductance_h: 0.01,
            secondary_capacitance_f: 1.0e-6,
            coupling_coefficient: 0.1,
            secondary_winding_polarity: 1,
        };
        let response = model.response_at(1_591.5494309189535).unwrap();
        close(response.mutual_inductance_h, 0.001);
        close(response.primary_input_impedance_ohm.real, 15.0);
        close(response.primary_input_impedance_ohm.imaginary, 0.0);
        close(response.secondary_current_per_volt_a_per_v.real, 0.0);
        close(response.secondary_current_per_volt_a_per_v.imaginary, -1.0 / 30.0);
        close(response.secondary_resistor_voltage_per_volt.real, 0.0);
        close(response.secondary_resistor_voltage_per_volt.imaginary, -2.0 / 3.0);
        close(response.secondary_resistor_power_w_per_v2, 1.0 / 45.0);
        close(response.primary_current_per_volt_a_per_v.real, 1.0 / 15.0);
        close(response.primary_current_per_volt_a_per_v.imaginary, 0.0);
        close(response.primary_input_power_w_per_v2, 1.0 / 15.0);
        close(response.primary_resistor_power_w_per_v2, 2.0 / 45.0);
        close(response.power_balance_residual_w_per_v2, 0.0);
    }

    #[test]
    fn passive_power_balance_holds_across_a_frequency_sweep() {
        let model = DrivenCoupledSeriesRlcV1 {
            primary_resistance_ohm: 10.0,
            primary_inductance_h: 0.01,
            primary_capacitance_f: 1.0e-6,
            secondary_resistance_ohm: 20.0,
            secondary_inductance_h: 0.01,
            secondary_capacitance_f: 1.0e-6,
            coupling_coefficient: 0.1,
            secondary_winding_polarity: 1,
        };
        for frequency in [500.0, 1_000.0, 1_591.5494309189535, 2_500.0, 5_000.0] {
            let response = model.response_at(frequency).unwrap();
            let scale = response.primary_input_power_w_per_v2.abs().max(1.0);
            assert!(
                response.power_balance_residual_w_per_v2.abs() <= 1.0e-12 * scale,
                "frequency={frequency}, residual={}",
                response.power_balance_residual_w_per_v2
            );
        }
    }

    #[test]
    fn reversing_secondary_dot_polarity_reverses_transfer_phase_not_input_impedance() {
        let mut model = DrivenCoupledSeriesRlcV1 {
            primary_resistance_ohm: 10.0,
            primary_inductance_h: 0.01,
            primary_capacitance_f: 1.0e-6,
            secondary_resistance_ohm: 20.0,
            secondary_inductance_h: 0.01,
            secondary_capacitance_f: 1.0e-6,
            coupling_coefficient: 0.1,
            secondary_winding_polarity: 1,
        };
        let positive = model.response_at(1_591.5494309189535).unwrap();
        model.secondary_winding_polarity = -1;
        let negative = model.response_at(1_591.5494309189535).unwrap();
        close(positive.primary_input_impedance_ohm.real, negative.primary_input_impedance_ohm.real);
        close(positive.primary_input_impedance_ohm.imaginary, negative.primary_input_impedance_ohm.imaginary);
        close(positive.secondary_current_per_volt_a_per_v.imaginary, -negative.secondary_current_per_volt_a_per_v.imaginary);
        close(positive.secondary_resistor_power_w_per_v2, negative.secondary_resistor_power_w_per_v2);
    }

    #[test]
    fn driven_coupled_loops_reject_invalid_polarity_and_zero_frequency() {
        let model = DrivenCoupledSeriesRlcV1 {
            primary_resistance_ohm: 10.0,
            primary_inductance_h: 0.01,
            primary_capacitance_f: 1.0e-6,
            secondary_resistance_ohm: 20.0,
            secondary_inductance_h: 0.01,
            secondary_capacitance_f: 1.0e-6,
            coupling_coefficient: 0.1,
            secondary_winding_polarity: 0,
        };
        assert_eq!(model.response_at(1_000.0), Err(ModelErrorV1::InvalidCouplingPolarity));
        let valid = DrivenCoupledSeriesRlcV1 { secondary_winding_polarity: 1, ..model };
        assert_eq!(valid.response_at(0.0), Err(ModelErrorV1::NonPositiveFrequency));
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
