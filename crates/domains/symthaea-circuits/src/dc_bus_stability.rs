// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Reduced-order DC-bus stability reference for a Thevenin source and
//! constant-power load.
//!
//! This is an analytical research model, not a converter, battery, protection,
//! aircraft-network, flight-control, or certification model. Its purpose is to
//! make equilibrium and local-stability boundaries independently testable.
//!
//! Model: C dV/dt = (Vs - V)/Rs - P/V.
//!
//! The binary64 discriminant calculation includes an ambiguity band. Values
//! inside that band return NumericallyIndeterminate rather than being promoted
//! to a stable or feasible result. This guard is not a formally directed-rounded
//! interval enclosure.

/// Stable identifier for this reduced-order model contract.
pub const MODEL_ID: &str = "thevenin-cpl-dc-link-v0.1";

/// Physical parameters for the one-state Thevenin / constant-power-load model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Parameters {
    /// Thevenin source voltage, volts.
    pub source_voltage_v: f64,
    /// Thevenin source resistance, ohms.
    pub source_resistance_ohm: f64,
    /// Lumped DC-link capacitance, farads.
    pub bus_capacitance_f: f64,
    /// Constant-power load, watts.
    pub load_power_w: f64,
}

/// Input validation or representability failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisError {
    /// A named input was NaN or infinite.
    NonFiniteInput(&'static str),
    /// A named physical input was not strictly positive.
    NonPositiveInput(&'static str),
    /// The initial bus voltage was NaN or infinite.
    NonFiniteInitialVoltage,
    /// The initial bus voltage was zero or negative.
    NonPositiveInitialVoltage,
    /// An intermediate overflowed, underflowed, or left the supported range.
    UnrepresentableIntermediate,
}

/// Local stability classification for an equilibrium of this scalar model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalStability {
    /// Perturbations decay locally toward this equilibrium.
    LocallyAsymptoticallyStable,
    /// Perturbations grow locally away from this equilibrium.
    Unstable,
}

/// A distinct equilibrium and its small-signal properties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equilibrium {
    /// Equilibrium bus voltage, volts.
    pub voltage_v: f64,
    /// Source/load current at equilibrium, amperes.
    pub current_a: f64,
    /// Scalar linearization eigenvalue, inverse seconds.
    pub eigenvalue_per_s: f64,
    /// Local stability in the stated one-state model.
    pub stability: LocalStability,
}

/// Initial-voltage phase-line classification when two distinct roots exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialVoltageBasin {
    /// Initial voltage is below the unstable root and moves toward the model boundary.
    BelowUnstableEquilibrium,
    /// Initial voltage equals the unstable equilibrium in the represented model.
    AtUnstableEquilibrium,
    /// Initial voltage is numerically close to the unstable-root basin boundary.
    NearUnstableEquilibriumBoundary,
    /// Initial voltage lies between roots and moves toward the upper root.
    BetweenEquilibriaTowardStableRoot,
    /// Initial voltage equals the upper equilibrium in the represented model.
    AtStableEquilibrium,
    /// Initial voltage is numerically close to the upper equilibrium.
    NearStableEquilibrium,
    /// Initial voltage is above the upper root and moves toward it.
    AboveStableEquilibriumTowardStableRoot,
}

/// Equilibrium classification under the supported floating-point calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EquilibriumResult {
    /// Two distinct positive roots exist in the reduced model.
    TwoEquilibria {
        lower: Equilibrium,
        upper: Equilibrium,
    },
    /// The discriminant is clearly negative; no real equilibrium exists.
    NoEquilibrium,
    /// The discriminant lies within the numerical ambiguity band.
    ///
    /// This includes the exact fold as represented by ordinary binary64.
    /// Callers must not treat it as a feasibility PASS.
    NumericallyIndeterminate,
}

/// Result of the reduced-order equilibrium analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Analysis {
    /// Floating-point discriminant estimate, V².
    pub discriminant_v2: f64,
    /// Heuristic arithmetic ambiguity guard, V²; not a formal enclosure.
    pub ambiguity_guard_v2: f64,
    /// Dimensionless Delta / Vs² diagnostic when representable.
    ///
    /// Positive values indicate two roots, zero the fold, negative values no
    /// equilibrium. This is not a dynamic damping or certification margin.
    pub normalized_fold_margin: f64,
    /// Equilibrium classification.
    pub result: EquilibriumResult,
    /// Initial-voltage classification for two-root cases only.
    pub initial_voltage_basin: Option<InitialVoltageBasin>,
}

/// Analyze the equilibria of the reduced Thevenin / constant-power-load model.
///
/// The caller supplies an initial voltage only for phase-line classification;
/// this function does not integrate a transient trajectory.
///
/// The discriminant ambiguity band intentionally withholds a positive
/// disposition near collapse. This is a research reference, not a formally
/// qualified interval-arithmetic kernel.
pub fn analyze(parameters: Parameters, initial_voltage_v: f64) -> Result<Analysis, AnalysisError> {
    validate_positive("source_voltage_v", parameters.source_voltage_v)?;
    validate_positive("source_resistance_ohm", parameters.source_resistance_ohm)?;
    validate_positive("bus_capacitance_f", parameters.bus_capacitance_f)?;
    validate_positive("load_power_w", parameters.load_power_w)?;

    if !initial_voltage_v.is_finite() {
        return Err(AnalysisError::NonFiniteInitialVoltage);
    }
    if initial_voltage_v <= 0.0 {
        return Err(AnalysisError::NonPositiveInitialVoltage);
    }

    let source_square = parameters.source_voltage_v * parameters.source_voltage_v;
    let resistance_power = parameters.source_resistance_ohm * parameters.load_power_w;
    let load_term = 4.0 * resistance_power;

    if !source_square.is_finite()
        || !resistance_power.is_finite()
        || !load_term.is_finite()
        || source_square == 0.0
        || resistance_power == 0.0
        || load_term == 0.0
    {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let scale = source_square.abs() + load_term.abs();
    if !scale.is_finite() || scale == 0.0 || !scale.is_normal() {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    // FMA rounds the source-square subtraction once. The load term has already
    // been rounded by its two multiplications, so retain an ambiguity guard.
    let discriminant_v2 = parameters
        .source_voltage_v
        .mul_add(parameters.source_voltage_v, -load_term);
    if !discriminant_v2.is_finite() {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let ambiguity_guard_v2 = 16.0 * f64::EPSILON * scale;
    if !ambiguity_guard_v2.is_finite() || ambiguity_guard_v2 == 0.0 {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let normalized_fold_margin = discriminant_v2 / source_square;
    if !normalized_fold_margin.is_finite() {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    if discriminant_v2.abs() <= ambiguity_guard_v2 {
        return Ok(Analysis {
            discriminant_v2,
            ambiguity_guard_v2,
            normalized_fold_margin,
            result: EquilibriumResult::NumericallyIndeterminate,
            initial_voltage_basin: None,
        });
    }

    if discriminant_v2 < 0.0 {
        return Ok(Analysis {
            discriminant_v2,
            ambiguity_guard_v2,
            normalized_fold_margin,
            result: EquilibriumResult::NoEquilibrium,
            initial_voltage_basin: None,
        });
    }

    let root_discriminant = discriminant_v2.sqrt();
    let upper_voltage_v = (parameters.source_voltage_v * 0.5) + (root_discriminant * 0.5);

    // Use the product of roots (Rs * P) to avoid cancellation in Vs - sqrt(Delta).
    let lower_voltage_v = resistance_power / upper_voltage_v;

    if !root_discriminant.is_finite()
        || !upper_voltage_v.is_finite()
        || !lower_voltage_v.is_finite()
        || upper_voltage_v <= 0.0
        || lower_voltage_v <= 0.0
    {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let time_resistance = parameters.source_resistance_ohm * parameters.bus_capacitance_f;
    if !time_resistance.is_finite() || time_resistance <= 0.0 {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let lower_denominator = time_resistance * lower_voltage_v;
    let upper_denominator = time_resistance * upper_voltage_v;
    if !lower_denominator.is_finite()
        || !upper_denominator.is_finite()
        || lower_denominator <= 0.0
        || upper_denominator <= 0.0
    {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let lower_current_a = parameters.load_power_w / lower_voltage_v;
    let upper_current_a = parameters.load_power_w / upper_voltage_v;
    let lower_eigenvalue_per_s = root_discriminant / lower_denominator;
    let upper_eigenvalue_per_s = -root_discriminant / upper_denominator;

    if !lower_current_a.is_finite()
        || !upper_current_a.is_finite()
        || !lower_eigenvalue_per_s.is_finite()
        || !upper_eigenvalue_per_s.is_finite()
    {
        return Err(AnalysisError::UnrepresentableIntermediate);
    }

    let lower = Equilibrium {
        voltage_v: lower_voltage_v,
        current_a: lower_current_a,
        eigenvalue_per_s: lower_eigenvalue_per_s,
        stability: LocalStability::Unstable,
    };
    let upper = Equilibrium {
        voltage_v: upper_voltage_v,
        current_a: upper_current_a,
        eigenvalue_per_s: upper_eigenvalue_per_s,
        stability: LocalStability::LocallyAsymptoticallyStable,
    };

    let initial_voltage_basin = Some(classify_initial_voltage(
        initial_voltage_v,
        lower_voltage_v,
        upper_voltage_v,
    ));

    Ok(Analysis {
        discriminant_v2,
        ambiguity_guard_v2,
        normalized_fold_margin,
        result: EquilibriumResult::TwoEquilibria { lower, upper },
        initial_voltage_basin,
    })
}

fn validate_positive(name: &'static str, value: f64) -> Result<(), AnalysisError> {
    if !value.is_finite() {
        return Err(AnalysisError::NonFiniteInput(name));
    }
    if value <= 0.0 {
        return Err(AnalysisError::NonPositiveInput(name));
    }
    Ok(())
}

fn classify_initial_voltage(
    initial_voltage_v: f64,
    lower_voltage_v: f64,
    upper_voltage_v: f64,
) -> InitialVoltageBasin {
    if initial_voltage_v == lower_voltage_v {
        return InitialVoltageBasin::AtUnstableEquilibrium;
    }
    if initial_voltage_v == upper_voltage_v {
        return InitialVoltageBasin::AtStableEquilibrium;
    }

    let lower_scale = initial_voltage_v.abs().max(lower_voltage_v.abs());
    let upper_scale = initial_voltage_v.abs().max(upper_voltage_v.abs());
    let lower_guard = 16.0 * f64::EPSILON * lower_scale;
    let upper_guard = 16.0 * f64::EPSILON * upper_scale;

    if (initial_voltage_v - lower_voltage_v).abs() <= lower_guard {
        InitialVoltageBasin::NearUnstableEquilibriumBoundary
    } else if (initial_voltage_v - upper_voltage_v).abs() <= upper_guard {
        InitialVoltageBasin::NearStableEquilibrium
    } else if initial_voltage_v < lower_voltage_v {
        InitialVoltageBasin::BelowUnstableEquilibrium
    } else if initial_voltage_v < upper_voltage_v {
        InitialVoltageBasin::BetweenEquilibriaTowardStableRoot
    } else {
        InitialVoltageBasin::AboveStableEquilibriumTowardStableRoot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parameters(load_power_w: f64) -> Parameters {
        Parameters {
            source_voltage_v: 100.0,
            source_resistance_ohm: 1.0,
            bus_capacitance_f: 0.01,
            load_power_w,
        }
    }

    #[test]
    fn stable_case_reproduces_both_equilibria_and_currents() {
        let analysis = analyze(parameters(1_875.0), 40.0).expect("valid fixture");
        assert_eq!(analysis.discriminant_v2, 2_500.0);
        assert!((analysis.normalized_fold_margin - 0.25).abs() < 1e-15);

        let EquilibriumResult::TwoEquilibria { lower, upper } = analysis.result else {
            panic!("expected two equilibria");
        };
        assert!((lower.voltage_v - 25.0).abs() < 1e-12);
        assert!((upper.voltage_v - 75.0).abs() < 1e-12);
        assert!((lower.current_a - 75.0).abs() < 1e-12);
        assert!((upper.current_a - 25.0).abs() < 1e-12);
        assert!((lower.eigenvalue_per_s - 200.0).abs() < 1e-10);
        assert!((upper.eigenvalue_per_s + (200.0 / 3.0)).abs() < 1e-10);
        assert_eq!(lower.stability, LocalStability::Unstable);
        assert_eq!(upper.stability, LocalStability::LocallyAsymptoticallyStable);
        assert_eq!(
            analysis.initial_voltage_basin,
            Some(InitialVoltageBasin::BetweenEquilibriaTowardStableRoot)
        );
    }

    #[test]
    fn near_fold_case_retains_distinct_roots_and_signed_eigenvalues() {
        let analysis = analyze(parameters(2_499.0), 50.0).expect("valid fixture");
        assert_eq!(analysis.discriminant_v2, 4.0);
        let EquilibriumResult::TwoEquilibria { lower, upper } = analysis.result else {
            panic!("expected two equilibria");
        };
        assert!((lower.voltage_v - 49.0).abs() < 1e-12);
        assert!((upper.voltage_v - 51.0).abs() < 1e-12);
        assert!((lower.eigenvalue_per_s - (200.0 / 49.0)).abs() < 1e-10);
        assert!((upper.eigenvalue_per_s + (200.0 / 51.0)).abs() < 1e-10);
    }

    #[test]
    fn exact_fold_is_withheld_as_numerically_indeterminate() {
        let analysis = analyze(parameters(2_500.0), 60.0).expect("valid fixture");
        assert_eq!(analysis.discriminant_v2, 0.0);
        assert_eq!(analysis.result, EquilibriumResult::NumericallyIndeterminate);
        assert_eq!(analysis.initial_voltage_basin, None);
        assert_eq!(analysis.normalized_fold_margin, 0.0);
    }

    #[test]
    fn above_fold_load_has_no_equilibrium() {
        let analysis = analyze(parameters(2_501.0), 60.0).expect("valid fixture");
        assert_eq!(analysis.discriminant_v2, -4.0);
        assert_eq!(analysis.result, EquilibriumResult::NoEquilibrium);
        assert_eq!(analysis.initial_voltage_basin, None);
    }

    #[test]
    fn initial_voltage_phase_line_is_explicit() {
        let below = analyze(parameters(1_875.0), 20.0).expect("valid fixture");
        let at_lower = analyze(parameters(1_875.0), 25.0).expect("valid fixture");
        let between = analyze(parameters(1_875.0), 40.0).expect("valid fixture");
        let at_upper = analyze(parameters(1_875.0), 75.0).expect("valid fixture");
        let above = analyze(parameters(1_875.0), 90.0).expect("valid fixture");

        assert_eq!(
            below.initial_voltage_basin,
            Some(InitialVoltageBasin::BelowUnstableEquilibrium)
        );
        assert_eq!(
            at_lower.initial_voltage_basin,
            Some(InitialVoltageBasin::AtUnstableEquilibrium)
        );
        assert_eq!(
            between.initial_voltage_basin,
            Some(InitialVoltageBasin::BetweenEquilibriaTowardStableRoot)
        );
        assert_eq!(
            at_upper.initial_voltage_basin,
            Some(InitialVoltageBasin::AtStableEquilibrium)
        );
        assert_eq!(
            above.initial_voltage_basin,
            Some(InitialVoltageBasin::AboveStableEquilibriumTowardStableRoot)
        );
    }

    #[test]
    fn near_basin_boundaries_are_not_silently_classified() {
        let near_lower =
            analyze(parameters(1_875.0), 25.0 + f64::EPSILON * 128.0).expect("valid fixture");
        assert_eq!(
            near_lower.initial_voltage_basin,
            Some(InitialVoltageBasin::NearUnstableEquilibriumBoundary)
        );
        let near_upper =
            analyze(parameters(1_875.0), 75.0 - f64::EPSILON * 256.0).expect("valid fixture");
        assert_eq!(
            near_upper.initial_voltage_basin,
            Some(InitialVoltageBasin::NearStableEquilibrium)
        );
    }

    #[test]
    fn input_validation_fails_closed() {
        let mut p = parameters(1_875.0);
        p.source_voltage_v = f64::NAN;
        assert_eq!(
            analyze(p, 50.0),
            Err(AnalysisError::NonFiniteInput("source_voltage_v"))
        );

        p = parameters(1_875.0);
        p.source_resistance_ohm = 0.0;
        assert_eq!(
            analyze(p, 50.0),
            Err(AnalysisError::NonPositiveInput("source_resistance_ohm"))
        );

        p = parameters(1_875.0);
        p.bus_capacitance_f = -1.0;
        assert_eq!(
            analyze(p, 50.0),
            Err(AnalysisError::NonPositiveInput("bus_capacitance_f"))
        );

        p = parameters(f64::INFINITY);
        assert_eq!(
            analyze(p, 50.0),
            Err(AnalysisError::NonFiniteInput("load_power_w"))
        );
    }

    #[test]
    fn initial_voltage_must_be_finite_and_positive() {
        let p = parameters(1_875.0);
        assert_eq!(
            analyze(p, f64::INFINITY),
            Err(AnalysisError::NonFiniteInitialVoltage)
        );
        assert_eq!(
            analyze(p, 0.0),
            Err(AnalysisError::NonPositiveInitialVoltage)
        );
    }

    #[test]
    fn near_fold_ambiguity_does_not_become_a_false_pass() {
        let p = parameters(2_500.0 + f64::EPSILON * 8_192.0);
        let analysis = analyze(p, 60.0).expect("finite but boundary-ambiguous input");
        assert_eq!(
            analysis.result,
            EquilibriumResult::NumericallyIndeterminate
        );
    }

    #[test]
    fn scaled_voltage_resistance_capacitance_case_preserves_analytic_scaling() {
        let p = Parameters {
            source_voltage_v: 400.0,
            source_resistance_ohm: 0.02,
            bus_capacitance_f: 0.005,
            load_power_w: 1_500_000.0,
        };
        let analysis = analyze(p, 200.0).expect("scaled valid fixture");
        assert_eq!(analysis.discriminant_v2, 40_000.0);
        assert!((analysis.normalized_fold_margin - 0.25).abs() < 1e-15);

        let EquilibriumResult::TwoEquilibria { lower, upper } = analysis.result else {
            panic!("expected two equilibria");
        };
        assert!((lower.voltage_v - 100.0).abs() < 1e-12);
        assert!((upper.voltage_v - 300.0).abs() < 1e-12);
        assert!((lower.current_a - 15_000.0).abs() < 1e-9);
        assert!((upper.current_a - 5_000.0).abs() < 1e-9);
        assert!((lower.eigenvalue_per_s - 20_000.0).abs() < 1e-8);
        assert!((upper.eigenvalue_per_s + (20_000.0 / 3.0)).abs() < 1e-8);
        assert_eq!(
            analysis.initial_voltage_basin,
            Some(InitialVoltageBasin::BetweenEquilibriaTowardStableRoot)
        );
    }

    #[test]
    fn overflowing_intermediates_are_rejected() {
        let p = Parameters {
            source_voltage_v: 1e200,
            source_resistance_ohm: 1.0,
            bus_capacitance_f: 0.01,
            load_power_w: 100.0,
        };
        assert_eq!(
            analyze(p, 50.0),
            Err(AnalysisError::UnrepresentableIntermediate)
        );
    }
}
