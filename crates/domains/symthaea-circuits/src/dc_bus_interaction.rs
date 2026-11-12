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
    InvalidConductanceInterval,
    UnrepresentableCriticalConductance,
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

/// Closed positive interval for a conductance-like parameter, in siemens.
///
/// Bounds must be deterministic hard limits. Statistical confidence intervals
/// and datasheet typical values are not automatically admissible hard bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductanceIntervalSiemens {
    /// Closed lower bound, siemens.
    pub lower_siemens: f64,
    /// Closed upper bound, siemens.
    pub upper_siemens: f64,
}

impl ConductanceIntervalSiemens {
    /// Construct a finite, strictly positive, ordered interval.
    pub fn new(lower_siemens: f64, upper_siemens: f64) -> Result<Self, InteractionError> {
        if !lower_siemens.is_finite()
            || !upper_siemens.is_finite()
            || !lower_siemens.is_normal()
            || !upper_siemens.is_normal()
            || lower_siemens <= 0.0
            || upper_siemens <= 0.0
            || lower_siemens > upper_siemens
        {
            return Err(InteractionError::InvalidConductanceInterval);
        }
        Ok(Self { lower_siemens, upper_siemens })
    }
}

/// Distribution-free disposition for the symmetric two-feeder RC/CPL model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustStabilityDisposition {
    /// Every admitted parameter combination is strictly Hurwitz stable.
    RobustlyHurwitzStable,
    /// Every admitted parameter combination has a positive real common-mode pole.
    RobustlyUnstablePositiveRealPole,
    /// The uncertainty set touches or overlaps the critical threshold range.
    NotEstablishedBoundaryOrOverlap,
}

/// Robust endpoint classification for the symmetric two-converter model.
///
/// The threshold bounds include a conservative binary64 guard. This is not a
/// formally outward-rounded interval proof; exact-rational fixtures are the
/// independent reference for the frozen known-answer cases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustStabilityReport {
    /// Conservative lower bound on the critical conductance, siemens.
    pub critical_threshold_lower_siemens: f64,
    /// Conservative upper bound on the critical conductance, siemens.
    pub critical_threshold_upper_siemens: f64,
    /// Absolute binary64 arithmetic guard applied to each threshold, siemens.
    pub numerical_guard_siemens: f64,
    /// Dimensionless strict separation margin for a robust disposition.
    /// Zero for a boundary/mixed/not-established result.
    pub normalized_margin: f64,
    pub disposition: RobustStabilityDisposition,
}

/// Classify a symmetric two-converter RC/CPL network for rectangular hard bounds.
///
/// The model requires one common source conductance Gs, two identical feeder
/// conductances G, and two identical CPL negative incremental conductances h,
/// all in siemens. Its exact common-mode threshold is:
///
/// h_crit(Gs, G) = Gs*G / (Gs + 2*G).
///
/// The threshold is strictly increasing in both Gs and G, so its exact extrema
/// over independent positive intervals occur at the corresponding endpoints.
/// No parameter-grid sampling or probability model is used.
pub fn classify_symmetric_hard_bounds(
    source_conductance: ConductanceIntervalSiemens,
    feeder_conductance: ConductanceIntervalSiemens,
    negative_incremental_conductance: ConductanceIntervalSiemens,
) -> Result<RobustStabilityReport, InteractionError> {
    let threshold_lower = common_mode_critical_conductance(
        source_conductance.lower_siemens,
        feeder_conductance.lower_siemens,
    )?;
    let threshold_upper = common_mode_critical_conductance(
        source_conductance.upper_siemens,
        feeder_conductance.upper_siemens,
    )?;
    if threshold_lower > threshold_upper {
        return Err(InteractionError::UnrepresentableCriticalConductance);
    }

    let scale = threshold_lower.max(threshold_upper);
    let guard = 64.0 * f64::EPSILON * scale;
    if !guard.is_finite() || guard <= 0.0 {
        return Err(InteractionError::UnrepresentableCriticalConductance);
    }

    // Widen the threshold range instead of allowing nearest-rounded values to
    // create a false strict separation.
    let conservative_lower = threshold_lower - guard;
    let conservative_upper = threshold_upper + guard;
    if !conservative_upper.is_finite() {
        return Err(InteractionError::UnrepresentableCriticalConductance);
    }

    let (disposition, normalized_margin) =
        if negative_incremental_conductance.upper_siemens < conservative_lower
            && conservative_lower > 0.0
        {
            (
                RobustStabilityDisposition::RobustlyHurwitzStable,
                (conservative_lower - negative_incremental_conductance.upper_siemens)
                    / conservative_lower,
            )
        } else if negative_incremental_conductance.lower_siemens > conservative_upper
            && conservative_upper > 0.0
        {
            (
                RobustStabilityDisposition::RobustlyUnstablePositiveRealPole,
                (negative_incremental_conductance.lower_siemens - conservative_upper)
                    / conservative_upper,
            )
        } else {
            (RobustStabilityDisposition::NotEstablishedBoundaryOrOverlap, 0.0)
        };

    if !normalized_margin.is_finite() || normalized_margin < 0.0 {
        return Err(InteractionError::UnrepresentableCriticalConductance);
    }

    Ok(RobustStabilityReport {
        critical_threshold_lower_siemens: conservative_lower,
        critical_threshold_upper_siemens: conservative_upper,
        numerical_guard_siemens: guard,
        normalized_margin,
        disposition,
    })
}

/// Evaluate Gs*G/(Gs+2G) without forming the potentially overflowing product.
fn common_mode_critical_conductance(
    source_conductance_siemens: f64,
    feeder_conductance_siemens: f64,
) -> Result<f64, InteractionError> {
    let value = if source_conductance_siemens >= feeder_conductance_siemens {
        feeder_conductance_siemens
            / (1.0 + 2.0 * (feeder_conductance_siemens / source_conductance_siemens))
    } else {
        source_conductance_siemens
            / ((source_conductance_siemens / feeder_conductance_siemens) + 2.0)
    };

    if !value.is_finite() || !value.is_normal() || value <= 0.0 {
        return Err(InteractionError::UnrepresentableCriticalConductance);
    }
    Ok(value)
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
    fn robust_hard_bounds_classify_a_stable_set() {
        let result = classify_symmetric_hard_bounds(
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.05, 0.1).unwrap(),
        ).unwrap();
        assert_eq!(result.disposition, RobustStabilityDisposition::RobustlyHurwitzStable);
        close(result.normalized_margin, 2.0 / 3.0, 1e-12);
    }

    #[test]
    fn robust_hard_bounds_classify_an_unstable_set() {
        let result = classify_symmetric_hard_bounds(
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.5, 0.6).unwrap(),
        ).unwrap();
        assert_eq!(
            result.disposition,
            RobustStabilityDisposition::RobustlyUnstablePositiveRealPole
        );
        close(result.normalized_margin, 4.0 / 11.0, 1e-12);
    }

    #[test]
    fn robust_hard_bounds_with_overlap_do_not_claim_stability() {
        let result = classify_symmetric_hard_bounds(
            ConductanceIntervalSiemens::new(0.8, 1.2).unwrap(),
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.3, 0.35).unwrap(),
        ).unwrap();
        assert_eq!(
            result.disposition,
            RobustStabilityDisposition::NotEstablishedBoundaryOrOverlap
        );
        assert_eq!(result.normalized_margin, 0.0);
    }

    #[test]
    fn exact_threshold_touch_is_not_a_robust_pass() {
        let result = classify_symmetric_hard_bounds(
            ConductanceIntervalSiemens::new(1.0, 1.0).unwrap(),
            ConductanceIntervalSiemens::new(1.0, 1.0).unwrap(),
            ConductanceIntervalSiemens::new(1.0 / 3.0, 1.0 / 3.0).unwrap(),
        ).unwrap();
        assert_eq!(
            result.disposition,
            RobustStabilityDisposition::NotEstablishedBoundaryOrOverlap
        );
    }

    #[test]
    fn robust_normalized_margin_is_invariant_to_conductance_scaling() {
        let scale = 1_000_000.0;
        let base = classify_symmetric_hard_bounds(
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.9, 1.1).unwrap(),
            ConductanceIntervalSiemens::new(0.05, 0.1).unwrap(),
        ).unwrap();
        let scaled = classify_symmetric_hard_bounds(
            ConductanceIntervalSiemens::new(0.9 * scale, 1.1 * scale).unwrap(),
            ConductanceIntervalSiemens::new(0.9 * scale, 1.1 * scale).unwrap(),
            ConductanceIntervalSiemens::new(0.05 * scale, 0.1 * scale).unwrap(),
        ).unwrap();
        assert_eq!(base.disposition, scaled.disposition);
        close(base.normalized_margin, scaled.normalized_margin, 1e-12);
    }

    #[test]
    fn widening_a_stable_interval_can_only_weaken_the_disposition() {
        let source = ConductanceIntervalSiemens::new(0.9, 1.1).unwrap();
        let feeder = ConductanceIntervalSiemens::new(0.9, 1.1).unwrap();
        let narrow = classify_symmetric_hard_bounds(
            source, feeder, ConductanceIntervalSiemens::new(0.05, 0.1).unwrap()
        ).unwrap();
        let widened = classify_symmetric_hard_bounds(
            source, feeder, ConductanceIntervalSiemens::new(0.05, 0.31).unwrap()
        ).unwrap();
        assert_eq!(narrow.disposition, RobustStabilityDisposition::RobustlyHurwitzStable);
        assert_eq!(
            widened.disposition,
            RobustStabilityDisposition::NotEstablishedBoundaryOrOverlap
        );
    }

    #[test]
    fn invalid_hard_intervals_fail_closed() {
        assert_eq!(
            ConductanceIntervalSiemens::new(0.0, 1.0),
            Err(InteractionError::InvalidConductanceInterval)
        );
        assert_eq!(
            ConductanceIntervalSiemens::new(1.0, 0.5),
            Err(InteractionError::InvalidConductanceInterval)
        );
        assert_eq!(
            ConductanceIntervalSiemens::new(0.1, f64::INFINITY),
            Err(InteractionError::InvalidConductanceInterval)
        );
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
