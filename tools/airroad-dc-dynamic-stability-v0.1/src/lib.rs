//! Analytical V0.1 reference for a Thevenin source feeding a constant-power
//! load through a lumped DC-link capacitor.
//!
//! This model is intentionally narrow. It is not a converter, battery,
//! protection, aircraft-network, or flight-safety model. Floating-point
//! classifications near the fold are reported as ambiguous rather than
//! silently promoted to a feasibility decision.

#![forbid(unsafe_code)]

use std::fmt;

const NEAR_FOLD_GUARD: f64 = 64.0 * f64::EPSILON;

/// Positive, finite model parameters with explicit SI-unit field names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BusParameters {
    source_voltage_v: f64,
    source_resistance_ohm: f64,
    bus_capacitance_f: f64,
    load_power_w: f64,
}

impl BusParameters {
    /// Construct validated parameters. Every value must be finite and > 0.
    pub fn new(
        source_voltage_v: f64,
        source_resistance_ohm: f64,
        bus_capacitance_f: f64,
        load_power_w: f64,
    ) -> Result<Self, ModelError> {
        validate_positive("source_voltage_v", source_voltage_v)?;
        validate_positive("source_resistance_ohm", source_resistance_ohm)?;
        validate_positive("bus_capacitance_f", bus_capacitance_f)?;
        validate_positive("load_power_w", load_power_w)?;

        Ok(Self {
            source_voltage_v,
            source_resistance_ohm,
            bus_capacitance_f,
            load_power_w,
        })
    }

    pub fn source_voltage_v(self) -> f64 { self.source_voltage_v }
    pub fn source_resistance_ohm(self) -> f64 { self.source_resistance_ohm }
    pub fn bus_capacitance_f(self) -> f64 { self.bus_capacitance_f }
    pub fn load_power_w(self) -> f64 { self.load_power_w }
}

/// Input or numerical-domain failure. No failure is converted into a
/// fabricated finite operating point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelError {
    NonFiniteParameter(&'static str),
    NonPositiveParameter(&'static str),
    NonFiniteDerivedValue(&'static str),
    NonPositiveInitialVoltage,
    NonFiniteInitialVoltage,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteParameter(name) => write!(f, "{name} must be finite"),
            Self::NonPositiveParameter(name) => write!(f, "{name} must be strictly positive"),
            Self::NonFiniteDerivedValue(name) => {
                write!(f, "derived value {name} is non-finite or unrepresentable")
            }
            Self::NonPositiveInitialVoltage => {
                write!(f, "initial_voltage_v must be strictly positive")
            }
            Self::NonFiniteInitialVoltage => write!(f, "initial_voltage_v must be finite"),
        }
    }
}

impl std::error::Error for ModelError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalStability {
    LocallyAsymptoticallyStable,
    Unstable,
    NonHyperbolicFold,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquilibriumPoint {
    pub voltage_v: f64,
    pub current_a: f64,
    pub eigenvalue_per_s: f64,
    pub stability: LocalStability,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EquilibriumSet {
    Two {
        high_voltage: EquilibriumPoint,
        low_voltage: EquilibriumPoint,
    },
    CriticalFold { equilibrium: EquilibriumPoint },
    NoPositiveEquilibrium,
    NearFoldAmbiguous { normalized_fold_margin: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquilibriumReport {
    /// mu = 1 - P/Pcritical = 1 - 4 Rs P / Vs^2.
    pub normalized_fold_margin: f64,
    /// 4 Rs P / Vs^2.
    pub power_loading_fraction: f64,
    /// Vs^2 / (4 Rs), if representable as binary64.
    pub critical_load_power_w: Option<f64>,
    pub equilibria: EquilibriumSet,
}

/// Evaluate the reduced-order model's analytical equilibria.
///
/// This is a mathematical reference, not a physical feasibility authority.
/// Near-fold binary64 cases are deliberately returned as ambiguous.
pub fn evaluate_equilibria(
    params: BusParameters,
) -> Result<EquilibriumReport, ModelError> {
    let vs = params.source_voltage_v;
    let rs = params.source_resistance_ohm;
    let capacitance = params.bus_capacitance_f;
    let power = params.load_power_w;

    // Ratio form avoids unnecessarily forming Vs^2 and Rs*P before division.
    let loading_fraction = 4.0 * (rs / vs) * (power / vs);
    if !loading_fraction.is_finite() || loading_fraction <= 0.0 {
        return Err(ModelError::NonFiniteDerivedValue("power_loading_fraction"));
    }

    let margin = 1.0 - loading_fraction;
    if !margin.is_finite() {
        return Err(ModelError::NonFiniteDerivedValue("normalized_fold_margin"));
    }

    let critical_denominator = 4.0 * (rs / vs);
    let critical_power = if critical_denominator.is_finite() && critical_denominator > 0.0 {
        let candidate = vs / critical_denominator;
        candidate.is_finite().then_some(candidate)
    } else {
        None
    };

    let equilibria = if margin == 0.0 {
        let voltage = vs / 2.0;
        let current = power / voltage;
        let point = EquilibriumPoint {
            voltage_v: voltage,
            current_a: current,
            eigenvalue_per_s: 0.0,
            stability: LocalStability::NonHyperbolicFold,
        };
        validate_point(point)?;
        EquilibriumSet::CriticalFold { equilibrium: point }
    } else if margin.abs() <= NEAR_FOLD_GUARD {
        EquilibriumSet::NearFoldAmbiguous {
            normalized_fold_margin: margin,
        }
    } else if margin < 0.0 {
        EquilibriumSet::NoPositiveEquilibrium
    } else {
        let root_gap = margin.sqrt();
        let high_voltage = (vs / 2.0) * (1.0 + root_gap);
        // Rationalized lower root avoids subtracting nearly equal values.
        let low_voltage = (vs / 2.0) * (loading_fraction / (1.0 + root_gap));

        let impedance_capacitance = rs * capacitance;
        if !impedance_capacitance.is_finite() || impedance_capacitance <= 0.0 {
            return Err(ModelError::NonFiniteDerivedValue(
                "source_resistance_times_capacitance",
            ));
        }

        let high_current = power / high_voltage;
        let low_current = power / low_voltage;
        let high_eigenvalue =
            -(vs * root_gap) / (impedance_capacitance * high_voltage);
        let low_eigenvalue =
            (vs * root_gap) / (impedance_capacitance * low_voltage);

        let high = EquilibriumPoint {
            voltage_v: high_voltage,
            current_a: high_current,
            eigenvalue_per_s: high_eigenvalue,
            stability: LocalStability::LocallyAsymptoticallyStable,
        };
        let low = EquilibriumPoint {
            voltage_v: low_voltage,
            current_a: low_current,
            eigenvalue_per_s: low_eigenvalue,
            stability: LocalStability::Unstable,
        };
        validate_point(high)?;
        validate_point(low)?;
        EquilibriumSet::Two {
            high_voltage: high,
            low_voltage: low,
        }
    };

    Ok(EquilibriumReport {
        normalized_fold_margin: margin,
        power_loading_fraction: loading_fraction,
        critical_load_power_w: critical_power,
        equilibria,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialStateBehavior {
    ConvergesToHighVoltageEquilibrium,
    RemainsAtUnstableEquilibrium,
    RemainsAtStableEquilibrium,
    MovesTowardLowerVoltageDomainBoundary,
    ApproachesFoldFromAbove,
    RemainsAtFold,
    LeavesFoldTowardLowerVoltageDomainBoundary,
    NoPositiveEquilibrium,
    NearFoldClassificationAmbiguous,
}

/// Classify phase-line behavior for a strictly positive initial voltage.
/// This result is specific to the one-state ideal CPL model.
pub fn classify_initial_voltage(
    report: EquilibriumReport,
    initial_voltage_v: f64,
) -> Result<InitialStateBehavior, ModelError> {
    if !initial_voltage_v.is_finite() {
        return Err(ModelError::NonFiniteInitialVoltage);
    }
    if initial_voltage_v <= 0.0 {
        return Err(ModelError::NonPositiveInitialVoltage);
    }

    Ok(match report.equilibria {
        EquilibriumSet::Two { high_voltage, low_voltage } => {
            if initial_voltage_v < low_voltage.voltage_v {
                InitialStateBehavior::MovesTowardLowerVoltageDomainBoundary
            } else if initial_voltage_v == low_voltage.voltage_v {
                InitialStateBehavior::RemainsAtUnstableEquilibrium
            } else if initial_voltage_v == high_voltage.voltage_v {
                InitialStateBehavior::RemainsAtStableEquilibrium
            } else {
                InitialStateBehavior::ConvergesToHighVoltageEquilibrium
            }
        }
        EquilibriumSet::CriticalFold { equilibrium } => {
            if initial_voltage_v > equilibrium.voltage_v {
                InitialStateBehavior::ApproachesFoldFromAbove
            } else if initial_voltage_v == equilibrium.voltage_v {
                InitialStateBehavior::RemainsAtFold
            } else {
                InitialStateBehavior::LeavesFoldTowardLowerVoltageDomainBoundary
            }
        }
        EquilibriumSet::NoPositiveEquilibrium => InitialStateBehavior::NoPositiveEquilibrium,
        EquilibriumSet::NearFoldAmbiguous { .. } => {
            InitialStateBehavior::NearFoldClassificationAmbiguous
        }
    })
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), ModelError> {
    if !value.is_finite() {
        return Err(ModelError::NonFiniteParameter(name));
    }
    if value <= 0.0 {
        return Err(ModelError::NonPositiveParameter(name));
    }
    Ok(())
}

fn validate_point(point: EquilibriumPoint) -> Result<(), ModelError> {
    if !point.voltage_v.is_finite()
        || !point.current_a.is_finite()
        || !point.eigenvalue_per_s.is_finite()
        || point.voltage_v <= 0.0
        || point.current_a <= 0.0
    {
        return Err(ModelError::NonFiniteDerivedValue("equilibrium_point"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard(power_w: f64) -> BusParameters {
        BusParameters::new(100.0, 1.0, 0.01, power_w)
            .expect("fixture parameters are positive and finite")
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn stable_and_unstable_roots_match_frozen_reference() {
        let report = evaluate_equilibria(standard(1875.0)).unwrap();
        close(report.normalized_fold_margin, 0.25, 1e-15);
        close(report.power_loading_fraction, 0.75, 1e-15);
        close(report.critical_load_power_w.unwrap(), 2500.0, 1e-12);

        let EquilibriumSet::Two { high_voltage, low_voltage } = report.equilibria else {
            panic!("expected two equilibria");
        };

        close(high_voltage.voltage_v, 75.0, 1e-12);
        close(high_voltage.current_a, 25.0, 1e-12);
        close(high_voltage.eigenvalue_per_s, -200.0 / 3.0, 1e-12);
        assert_eq!(high_voltage.stability, LocalStability::LocallyAsymptoticallyStable);

        close(low_voltage.voltage_v, 25.0, 1e-12);
        close(low_voltage.current_a, 75.0, 1e-12);
        close(low_voltage.eigenvalue_per_s, 200.0, 1e-12);
        assert_eq!(low_voltage.stability, LocalStability::Unstable);
    }

    #[test]
    fn near_fold_case_preserves_both_distinct_roots() {
        let report = evaluate_equilibria(standard(2499.0)).unwrap();
        close(report.normalized_fold_margin, 1.0 / 2500.0, 1e-14);

        let EquilibriumSet::Two { high_voltage, low_voltage } = report.equilibria else {
            panic!("expected two equilibria");
        };

        close(high_voltage.voltage_v, 51.0, 1e-10);
        close(low_voltage.voltage_v, 49.0, 1e-10);
        close(high_voltage.current_a, 49.0, 1e-10);
        close(low_voltage.current_a, 51.0, 1e-10);
        close(high_voltage.eigenvalue_per_s, -200.0 / 51.0, 1e-10);
        close(low_voltage.eigenvalue_per_s, 200.0 / 49.0, 1e-10);
    }

    #[test]
    fn exact_fold_is_critical_and_not_robustly_stable() {
        let report = evaluate_equilibria(standard(2500.0)).unwrap();
        assert_eq!(report.normalized_fold_margin, 0.0);

        let EquilibriumSet::CriticalFold { equilibrium } = report.equilibria else {
            panic!("expected critical fold");
        };

        close(equilibrium.voltage_v, 50.0, 1e-12);
        close(equilibrium.current_a, 50.0, 1e-12);
        assert_eq!(equilibrium.eigenvalue_per_s, 0.0);
        assert_eq!(equilibrium.stability, LocalStability::NonHyperbolicFold);
    }

    #[test]
    fn above_fold_load_has_no_positive_equilibrium() {
        let report = evaluate_equilibria(standard(2501.0)).unwrap();
        close(report.normalized_fold_margin, -1.0 / 2500.0, 1e-14);
        assert_eq!(report.equilibria, EquilibriumSet::NoPositiveEquilibrium);
    }

    #[test]
    fn phase_line_identifies_high_voltage_basin_and_unstable_boundary() {
        let report = evaluate_equilibria(standard(1875.0)).unwrap();

        assert_eq!(
            classify_initial_voltage(report, 30.0).unwrap(),
            InitialStateBehavior::ConvergesToHighVoltageEquilibrium
        );
        assert_eq!(
            classify_initial_voltage(report, 25.0).unwrap(),
            InitialStateBehavior::RemainsAtUnstableEquilibrium
        );
        assert_eq!(
            classify_initial_voltage(report, 20.0).unwrap(),
            InitialStateBehavior::MovesTowardLowerVoltageDomainBoundary
        );
        assert_eq!(
            classify_initial_voltage(report, 75.0).unwrap(),
            InitialStateBehavior::RemainsAtStableEquilibrium
        );
        assert_eq!(
            classify_initial_voltage(report, 90.0).unwrap(),
            InitialStateBehavior::ConvergesToHighVoltageEquilibrium
        );
    }

    #[test]
    fn fold_phase_line_is_semistable() {
        let report = evaluate_equilibria(standard(2500.0)).unwrap();

        assert_eq!(
            classify_initial_voltage(report, 60.0).unwrap(),
            InitialStateBehavior::ApproachesFoldFromAbove
        );
        assert_eq!(
            classify_initial_voltage(report, 50.0).unwrap(),
            InitialStateBehavior::RemainsAtFold
        );
        assert_eq!(
            classify_initial_voltage(report, 40.0).unwrap(),
            InitialStateBehavior::LeavesFoldTowardLowerVoltageDomainBoundary
        );
    }

    #[test]
    fn values_too_close_to_fold_are_not_classified() {
        let report = evaluate_equilibria(standard(2500.0 * (1.0 + 2.0 * f64::EPSILON)))
            .unwrap();
        assert!(matches!(report.equilibria, EquilibriumSet::NearFoldAmbiguous { .. }));
        assert_eq!(
            classify_initial_voltage(report, 50.0).unwrap(),
            InitialStateBehavior::NearFoldClassificationAmbiguous
        );
    }

    #[test]
    fn rejects_non_positive_and_non_finite_parameters() {
        assert_eq!(
            BusParameters::new(0.0, 1.0, 0.01, 10.0).unwrap_err(),
            ModelError::NonPositiveParameter("source_voltage_v")
        );
        assert_eq!(
            BusParameters::new(100.0, -1.0, 0.01, 10.0).unwrap_err(),
            ModelError::NonPositiveParameter("source_resistance_ohm")
        );
        assert_eq!(
            BusParameters::new(100.0, 1.0, f64::NAN, 10.0).unwrap_err(),
            ModelError::NonFiniteParameter("bus_capacitance_f")
        );
        assert_eq!(
            BusParameters::new(100.0, 1.0, 0.01, f64::INFINITY).unwrap_err(),
            ModelError::NonFiniteParameter("load_power_w")
        );
    }

    #[test]
    fn rejects_invalid_initial_voltage() {
        let report = evaluate_equilibria(standard(1875.0)).unwrap();
        assert_eq!(
            classify_initial_voltage(report, 0.0).unwrap_err(),
            ModelError::NonPositiveInitialVoltage
        );
        assert_eq!(
            classify_initial_voltage(report, f64::NAN).unwrap_err(),
            ModelError::NonFiniteInitialVoltage
        );
    }
}
