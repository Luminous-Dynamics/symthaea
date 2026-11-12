//! Small-signal interaction reference for two constant-power converter
//! inputs connected through resistive feeders to a common Thevenin source.
//!
//! This is a deliberately narrow RC/CPL model. It is not a converter control
//! model, switching model, aircraft protection model, or validated physical
//! impedance model. The frequency response is an exact linearization of the
//! declared reduced-order equations around a supplied equilibrium witness.

#![forbid(unsafe_code)]

/// Parameters for a common source and two feeder-connected CPL input nodes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkParameters {
    pub source_voltage_v: f64,
    pub source_resistance_ohm: f64,
    pub bus_capacitance_f: f64,
    pub feeder_1_resistance_ohm: f64,
    pub feeder_2_resistance_ohm: f64,
    pub input_1_capacitance_f: f64,
    pub input_2_capacitance_f: f64,
    pub load_1_power_w: f64,
    pub load_2_power_w: f64,
}

/// Supplied steady-state voltage witness. It is checked against KCL before
/// linearization or frequency-response evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OperatingPoint {
    pub bus_voltage_v: f64,
    pub input_1_voltage_v: f64,
    pub input_2_voltage_v: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractionError {
    NonFiniteInput(&'static str),
    NonPositiveInput(&'static str),
    NegativeTolerance,
    InvalidOperatingPointResidual,
    NonFiniteDerivedValue(&'static str),
    InvalidAngularFrequency,
    SingularFrequencyResponse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StabilityDisposition {
    /// Routh-Hurwitz conditions for a real cubic are strictly satisfied.
    HurwitzStable,
    /// The characteristic polynomial is negative at zero and positive for
    /// sufficiently large positive real s; at least one positive real pole
    /// therefore exists.
    PositiveRealPoleGuaranteed,
    /// A strict Hurwitz result was not established. This includes numerical
    /// boundary cases and other non-Hurwitz classifications.
    NotCertifiedStable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionAnalysis {
    /// Row-major Jacobian for states [V_bus, V_input_1, V_input_2].
    pub jacobian: [[f64; 3]; 3],
    /// Coefficients [1, a1, a2, a3] of det(sI - J).
    pub characteristic_coefficients: [f64; 4],
    /// Isolated branch poles when the bus voltage is held stiff.
    pub isolated_branch_poles_per_s: [f64; 2],
    /// Dimensionless Routh-Hurwitz quantity after normalizing the pole-rate
    /// scale. This avoids comparing coefficients with different SI dimensions.
    pub normalized_routh_hurwitz_delta: f64,
    /// Conservative numerical disposition; not a physical safety decision.
    pub stability: StabilityDisposition,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexValue {
    pub real: f64,
    pub imag: f64,
}

impl ComplexValue {
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        }
    }

    fn multiply(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }

    fn divide(self, rhs: Self) -> Result<Self, InteractionError> {
        let denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(InteractionError::SingularFrequencyResponse);
        }
        let result = Self {
            real: (self.real * rhs.real + self.imag * rhs.imag) / denominator,
            imag: (self.imag * rhs.real - self.real * rhs.imag) / denominator,
        };
        validate_complex(result)?;
        Ok(result)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReturnRatioResponse {
    pub angular_frequency_rad_s: f64,
    pub branch_1_admittance_s: ComplexValue,
    pub branch_2_admittance_s: ComplexValue,
    pub source_impedance_ohm: ComplexValue,
    /// L(jw) = Z_source(jw) * (Y_1(jw) + Y_2(jw)).
    pub return_ratio: ComplexValue,
}

/// Analyze the coupled three-state RC/CPL model around a supplied equilibrium.
///
/// The caller supplies an absolute KCL residual tolerance in amperes. This is
/// an explicit numerical input, not an inferred metrology or confidence bound.
pub fn analyze(
    p: NetworkParameters,
    op: OperatingPoint,
    equilibrium_residual_tolerance_a: f64,
) -> Result<InteractionAnalysis, InteractionError> {
    validate_parameters(p)?;
    validate_operating_point(op)?;
    validate_tolerance(equilibrium_residual_tolerance_a)?;
    check_equilibrium(p, op, equilibrium_residual_tolerance_a)?;

    let gs = 1.0 / p.source_resistance_ohm;
    let g1 = 1.0 / p.feeder_1_resistance_ohm;
    let g2 = 1.0 / p.feeder_2_resistance_ohm;
    let h1 = p.load_1_power_w / (op.input_1_voltage_v * op.input_1_voltage_v);
    let h2 = p.load_2_power_w / (op.input_2_voltage_v * op.input_2_voltage_v);

    let j00 = -(gs + g1 + g2) / p.bus_capacitance_f;
    let j01 = g1 / p.bus_capacitance_f;
    let j02 = g2 / p.bus_capacitance_f;
    let j10 = g1 / p.input_1_capacitance_f;
    let j11 = (-g1 + h1) / p.input_1_capacitance_f;
    let j20 = g2 / p.input_2_capacitance_f;
    let j22 = (-g2 + h2) / p.input_2_capacitance_f;

    let jacobian = [[j00, j01, j02], [j10, j11, 0.0], [j20, 0.0, j22]];

    let trace = j00 + j11 + j22;
    let principal_minors = (j00 * j11 - j01 * j10) + (j00 * j22 - j02 * j20) + (j11 * j22);
    let determinant = j00 * j11 * j22 - j01 * j10 * j22 - j02 * j11 * j20;

    // det(sI - J) = s^3 - tr(J)s^2 + sum(principal minors)s - det(J).
    let a1 = -trace;
    let a2 = principal_minors;
    let a3 = -determinant;
    let derived = [a1, a2, a3, j00, j01, j02, j10, j11, j20, j22];
    if derived.iter().any(|value| !value.is_finite()) {
        return Err(InteractionError::NonFiniteDerivedValue("linearization"));
    }

    // Normalize the three dimensionful coefficients by one characteristic
    // pole-rate. alpha_1, alpha_2 and alpha_3 are dimensionless, and the
    // normalized Routh-Hurwitz determinant is invariant to uniform time-unit
    // scaling of the same dynamical system.
    let rate_scale = a1.abs().max(a2.abs().sqrt()).max(a3.abs().cbrt());
    let (alpha_1, alpha_2, alpha_3) = if rate_scale > 0.0 && rate_scale.is_finite() {
        (
            a1 / rate_scale,
            a2.signum() * (a2.abs().sqrt() / rate_scale).powi(2),
            a3.signum() * (a3.abs().cbrt() / rate_scale).powi(3),
        )
    } else {
        (0.0, 0.0, 0.0)
    };
    let normalized_routh_delta = alpha_1 * alpha_2 - alpha_3;
    if !alpha_1.is_finite()
        || !alpha_2.is_finite()
        || !alpha_3.is_finite()
        || !normalized_routh_delta.is_finite()
    {
        return Err(InteractionError::NonFiniteDerivedValue(
            "normalized_routh_hurwitz",
        ));
    }
    let guard = 64.0 * f64::EPSILON;

    let stability = if alpha_3 < -guard {
        StabilityDisposition::PositiveRealPoleGuaranteed
    } else if alpha_1 > guard
        && alpha_2 > guard
        && alpha_3 > guard
        && normalized_routh_delta > guard
    {
        StabilityDisposition::HurwitzStable
    } else {
        StabilityDisposition::NotCertifiedStable
    };

    let isolated_1 = (-g1 + h1) / p.input_1_capacitance_f;
    let isolated_2 = (-g2 + h2) / p.input_2_capacitance_f;
    if !isolated_1.is_finite() || !isolated_2.is_finite() {
        return Err(InteractionError::NonFiniteDerivedValue(
            "isolated_branch_poles",
        ));
    }

    Ok(InteractionAnalysis {
        jacobian,
        characteristic_coefficients: [1.0, a1, a2, a3],
        isolated_branch_poles_per_s: [isolated_1, isolated_2],
        normalized_routh_hurwitz_delta: normalized_routh_delta,
        stability,
    })
}

/// Evaluate the linearized branch input admittances, source impedance and
/// common-mode return ratio at one angular frequency.
///
/// The sign convention is positive current drawn from the common bus:
/// L(jw) = Z_source(jw) * (Y_1(jw) + Y_2(jw)).
pub fn evaluate_return_ratio(
    p: NetworkParameters,
    op: OperatingPoint,
    equilibrium_residual_tolerance_a: f64,
    angular_frequency_rad_s: f64,
) -> Result<ReturnRatioResponse, InteractionError> {
    validate_parameters(p)?;
    validate_operating_point(op)?;
    validate_tolerance(equilibrium_residual_tolerance_a)?;
    check_equilibrium(p, op, equilibrium_residual_tolerance_a)?;

    if !angular_frequency_rad_s.is_finite() || angular_frequency_rad_s < 0.0 {
        return Err(InteractionError::InvalidAngularFrequency);
    }

    let g1 = 1.0 / p.feeder_1_resistance_ohm;
    let g2 = 1.0 / p.feeder_2_resistance_ohm;
    let h1 = p.load_1_power_w / (op.input_1_voltage_v * op.input_1_voltage_v);
    let h2 = p.load_2_power_w / (op.input_2_voltage_v * op.input_2_voltage_v);
    let omega = angular_frequency_rad_s;

    // Yi(s) = Gi * (Ci*s - hi)/(Ci*s + Gi - hi), evaluated at s = jw.
    let y1 = ComplexValue {
        real: -g1 * h1,
        imag: g1 * p.input_1_capacitance_f * omega,
    }
    .divide(ComplexValue {
        real: g1 - h1,
        imag: p.input_1_capacitance_f * omega,
    })?;

    let y2 = ComplexValue {
        real: -g2 * h2,
        imag: g2 * p.input_2_capacitance_f * omega,
    }
    .divide(ComplexValue {
        real: g2 - h2,
        imag: p.input_2_capacitance_f * omega,
    })?;

    // Zs(s) = 1/(Gs + C0*s), evaluated at s = jw.
    let source_impedance = ComplexValue {
        real: 1.0,
        imag: 0.0,
    }
    .divide(ComplexValue {
        real: 1.0 / p.source_resistance_ohm,
        imag: p.bus_capacitance_f * omega,
    })?;

    let return_ratio = source_impedance.multiply(y1.add(y2));
    validate_complex(return_ratio)?;

    Ok(ReturnRatioResponse {
        angular_frequency_rad_s: omega,
        branch_1_admittance_s: y1,
        branch_2_admittance_s: y2,
        source_impedance_ohm: source_impedance,
        return_ratio,
    })
}

fn validate_parameters(p: NetworkParameters) -> Result<(), InteractionError> {
    for (name, value) in [
        ("source_voltage_v", p.source_voltage_v),
        ("source_resistance_ohm", p.source_resistance_ohm),
        ("bus_capacitance_f", p.bus_capacitance_f),
        ("feeder_1_resistance_ohm", p.feeder_1_resistance_ohm),
        ("feeder_2_resistance_ohm", p.feeder_2_resistance_ohm),
        ("input_1_capacitance_f", p.input_1_capacitance_f),
        ("input_2_capacitance_f", p.input_2_capacitance_f),
        ("load_1_power_w", p.load_1_power_w),
        ("load_2_power_w", p.load_2_power_w),
    ] {
        validate_positive(name, value)?;
    }
    Ok(())
}

fn validate_operating_point(op: OperatingPoint) -> Result<(), InteractionError> {
    for (name, value) in [
        ("bus_voltage_v", op.bus_voltage_v),
        ("input_1_voltage_v", op.input_1_voltage_v),
        ("input_2_voltage_v", op.input_2_voltage_v),
    ] {
        validate_positive(name, value)?;
    }
    Ok(())
}

fn validate_tolerance(value: f64) -> Result<(), InteractionError> {
    if !value.is_finite() {
        return Err(InteractionError::NonFiniteInput(
            "equilibrium_residual_tolerance_a",
        ));
    }
    if value < 0.0 {
        return Err(InteractionError::NegativeTolerance);
    }
    Ok(())
}

fn check_equilibrium(
    p: NetworkParameters,
    op: OperatingPoint,
    tolerance_a: f64,
) -> Result<(), InteractionError> {
    let source_current = (p.source_voltage_v - op.bus_voltage_v) / p.source_resistance_ohm;
    let branch_1_current = (op.bus_voltage_v - op.input_1_voltage_v) / p.feeder_1_resistance_ohm;
    let branch_2_current = (op.bus_voltage_v - op.input_2_voltage_v) / p.feeder_2_resistance_ohm;
    let load_1_current = p.load_1_power_w / op.input_1_voltage_v;
    let load_2_current = p.load_2_power_w / op.input_2_voltage_v;

    let residuals = [
        source_current - branch_1_current - branch_2_current,
        branch_1_current - load_1_current,
        branch_2_current - load_2_current,
    ];
    if residuals.iter().any(|value| !value.is_finite()) {
        return Err(InteractionError::NonFiniteDerivedValue(
            "equilibrium_residual",
        ));
    }
    if residuals.iter().any(|value| value.abs() > tolerance_a) {
        return Err(InteractionError::InvalidOperatingPointResidual);
    }
    Ok(())
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), InteractionError> {
    if !value.is_finite() {
        return Err(InteractionError::NonFiniteInput(name));
    }
    if value <= 0.0 {
        return Err(InteractionError::NonPositiveInput(name));
    }
    Ok(())
}

fn validate_complex(value: ComplexValue) -> Result<(), InteractionError> {
    if !value.real.is_finite() || !value.imag.is_finite() {
        return Err(InteractionError::NonFiniteDerivedValue("complex_response"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (NetworkParameters, OperatingPoint) {
        (
            NetworkParameters {
                source_voltage_v: 2.5,
                source_resistance_ohm: 1.0,
                bus_capacitance_f: 1.0,
                feeder_1_resistance_ohm: 1.0,
                feeder_2_resistance_ohm: 1.0,
                input_1_capacitance_f: 1.0,
                input_2_capacitance_f: 1.0,
                load_1_power_w: 0.5,
                load_2_power_w: 0.5,
            },
            OperatingPoint {
                bus_voltage_v: 1.5,
                input_1_voltage_v: 1.0,
                input_2_voltage_v: 1.0,
            },
        )
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn individually_stable_branches_can_be_unstable_when_coupled() {
        let (p, op) = fixture();
        let result = analyze(p, op, 1e-12).expect("valid exact operating point");

        close(result.isolated_branch_poles_per_s[0], -0.5, 1e-12);
        close(result.isolated_branch_poles_per_s[1], -0.5, 1e-12);
        assert_eq!(
            result.stability,
            StabilityDisposition::PositiveRealPoleGuaranteed
        );

        let coefficients = result.characteristic_coefficients;
        close(coefficients[1], 4.0, 1e-12);
        close(coefficients[2], 1.25, 1e-12);
        close(coefficients[3], -0.25, 1e-12);
    }

    #[test]
    fn exact_frequency_response_matches_frozen_return_ratio_points() {
        let (p, op) = fixture();

        let dc = evaluate_return_ratio(p, op, 1e-12, 0.0).unwrap();
        close(dc.branch_1_admittance_s.real, -1.0, 1e-12);
        close(dc.branch_1_admittance_s.imag, 0.0, 1e-12);
        close(dc.branch_2_admittance_s.real, -1.0, 1e-12);
        close(dc.source_impedance_ohm.real, 1.0, 1e-12);
        close(dc.return_ratio.real, -2.0, 1e-12);

        let omega_one = evaluate_return_ratio(p, op, 1e-12, 1.0).unwrap();
        close(omega_one.branch_1_admittance_s.real, 0.6, 1e-12);
        close(omega_one.branch_1_admittance_s.imag, 0.8, 1e-12);
        close(omega_one.source_impedance_ohm.real, 0.5, 1e-12);
        close(omega_one.source_impedance_ohm.imag, -0.5, 1e-12);
        close(omega_one.return_ratio.real, 1.4, 1e-12);
        close(omega_one.return_ratio.imag, 0.2, 1e-12);
    }

    #[test]
    fn low_power_symmetric_network_passes_routh_hurwitz() {
        let p = NetworkParameters {
            source_voltage_v: 1.15,
            source_resistance_ohm: 1.0,
            bus_capacitance_f: 1.0,
            feeder_1_resistance_ohm: 1.0,
            feeder_2_resistance_ohm: 1.0,
            input_1_capacitance_f: 1.0,
            input_2_capacitance_f: 1.0,
            load_1_power_w: 0.05,
            load_2_power_w: 0.05,
        };
        let op = OperatingPoint {
            bus_voltage_v: 1.05,
            input_1_voltage_v: 1.0,
            input_2_voltage_v: 1.0,
        };
        let result = analyze(p, op, 1e-12).expect("stable operating point");
        assert_eq!(result.stability, StabilityDisposition::HurwitzStable);
        assert!(result.characteristic_coefficients[1] > 0.0);
        assert!(result.characteristic_coefficients[2] > 0.0);
        assert!(result.characteristic_coefficients[3] > 0.0);
        assert!(result.normalized_routh_hurwitz_delta > 0.0);
    }

    #[test]
    fn normalized_routh_disposition_is_invariant_to_uniform_time_scaling() {
        let (mut p, op) = fixture();
        let baseline = analyze(p, op, 1e-12).expect("unstable baseline");

        // Multiplying all capacitances by ten slows all poles by ten without
        // changing their stability. The normalized Routh quantity is invariant.
        p.bus_capacitance_f *= 10.0;
        p.input_1_capacitance_f *= 10.0;
        p.input_2_capacitance_f *= 10.0;
        let scaled = analyze(p, op, 1e-12).expect("scaled unstable system");

        assert_eq!(baseline.stability, scaled.stability);
        close(
            baseline.normalized_routh_hurwitz_delta,
            scaled.normalized_routh_hurwitz_delta,
            1e-12,
        );
        close(
            scaled.characteristic_coefficients[1],
            baseline.characteristic_coefficients[1] / 10.0,
            1e-12,
        );
        close(
            scaled.characteristic_coefficients[2],
            baseline.characteristic_coefficients[2] / 100.0,
            1e-12,
        );
        close(
            scaled.characteristic_coefficients[3],
            baseline.characteristic_coefficients[3] / 1000.0,
            1e-12,
        );
    }

    #[test]
    fn asymmetric_branches_can_be_stable_individually_but_unstable_when_coupled() {
        let p = NetworkParameters {
            source_voltage_v: 17.0 / 8.0,
            source_resistance_ohm: 1.0,
            bus_capacitance_f: 1.0,
            feeder_1_resistance_ohm: 1.0,
            feeder_2_resistance_ohm: 2.0,
            input_1_capacitance_f: 1.0,
            input_2_capacitance_f: 2.0,
            load_1_power_w: 0.5,
            load_2_power_w: 5.0 / 32.0,
        };
        let op = OperatingPoint {
            bus_voltage_v: 1.5,
            input_1_voltage_v: 1.0,
            input_2_voltage_v: 1.25,
        };
        let result = analyze(p, op, 1e-12).expect("asymmetric exact operating point");
        close(result.isolated_branch_poles_per_s[0], -0.5, 1e-12);
        close(result.isolated_branch_poles_per_s[1], -0.2, 1e-12);
        assert_eq!(
            result.stability,
            StabilityDisposition::PositiveRealPoleGuaranteed
        );
        close(result.characteristic_coefficients[1], 16.0 / 5.0, 1e-12);
        close(result.characteristic_coefficients[2], 29.0 / 40.0, 1e-12);
        close(result.characteristic_coefficients[3], -1.0 / 80.0, 1e-12);

        let dc = evaluate_return_ratio(p, op, 1e-12, 0.0).unwrap();
        close(dc.branch_1_admittance_s.real, -1.0, 1e-12);
        close(dc.branch_2_admittance_s.real, -1.0 / 8.0, 1e-12);
        close(dc.source_impedance_ohm.real, 1.0, 1e-12);
        close(dc.return_ratio.real, -9.0 / 8.0, 1e-12);

        let omega_one = evaluate_return_ratio(p, op, 1e-12, 1.0).unwrap();
        close(omega_one.branch_1_admittance_s.real, 3.0 / 5.0, 1e-12);
        close(omega_one.branch_1_admittance_s.imag, 4.0 / 5.0, 1e-12);
        close(omega_one.branch_2_admittance_s.real, 99.0 / 208.0, 1e-12);
        close(omega_one.branch_2_admittance_s.imag, 25.0 / 208.0, 1e-12);
        close(omega_one.source_impedance_ohm.real, 0.5, 1e-12);
        close(omega_one.source_impedance_ohm.imag, -0.5, 1e-12);
        close(omega_one.return_ratio.real, 519.0 / 520.0, 1e-12);
        close(omega_one.return_ratio.imag, -81.0 / 1040.0, 1e-12);
    }

    #[test]
    fn invalid_operating_point_and_frequency_fail_closed() {
        let (p, mut op) = fixture();
        op.input_1_voltage_v = 1.2;
        assert_eq!(
            analyze(p, op, 1e-12),
            Err(InteractionError::InvalidOperatingPointResidual)
        );

        let (_, op) = fixture();
        assert_eq!(
            evaluate_return_ratio(fixture().0, op, 1e-12, f64::NAN),
            Err(InteractionError::InvalidAngularFrequency)
        );
    }
}
