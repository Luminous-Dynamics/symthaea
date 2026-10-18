// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic hard-bound propagation for frequency-resolved capacitor screening.
//!
//! This module composes the existing impedance-sweep/current-spectrum records. It
//! accepts only explicitly declared hard intervals, never standard uncertainties or
//! confidence intervals relabelled as deterministic bounds. The first version only
//! supports exact measured sweep knots: interpolation discrepancy is not assumed zero.
//! Results bound only the supplied quantities and equations, not unknown model error,
//! temperature transfer, safety margin, component lifetime or qualification.

use crate::capacitor_spectrum::{
    CapacitorCurrentSpectrum, CapacitorImpedanceSweep, CapacitorSpectrumConstraintViolation,
    CapacitorSpectrumDisposition, CapacitorSpectrumError, CapacitorSpectrumScreeningReport,
    CapacitorSpectrumScreeningRequirements, ImpedanceEvaluationKind,
    calculate_capacitor_spectrum_metrics, screen_capacitor_spectrum,
};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// A documented deterministic hard interval with a source/method reference.
///
/// This type is not a confidence interval or standard uncertainty. The caller is
/// responsible for ensuring the reference really supports a hard guarantee.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeclaredHardInterval {
    /// Inclusive lower endpoint.
    pub lower: f64,
    /// Inclusive upper endpoint.
    pub upper: f64,
    /// Method/receipt reference that justifies the deterministic bound.
    pub method_reference: String,
}

impl DeclaredHardInterval {
    /// Validate a finite, ordered hard interval that contains the nominal value.
    pub fn validate_contains(&self, nominal: f64) -> Result<(), CapacitorSpectrumBoundsError> {
        if !self.lower.is_finite() || !self.upper.is_finite() || self.lower > self.upper {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "interval endpoints must be finite and lower must not exceed upper",
            ));
        }
        if self.method_reference.trim().is_empty() {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "hard interval method reference is missing",
            ));
        }
        if !nominal.is_finite() || nominal < self.lower || nominal > self.upper {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "the nominal value must lie within its declared hard interval",
            ));
        }
        Ok(())
    }
}

/// Hard intervals for measured real/imaginary impedance at one exact sweep knot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexImpedanceHardBounds {
    /// Exact measured-knot frequency in hertz.
    pub frequency_hz: f64,
    /// Hard interval for the real part of impedance in ohms.
    pub real_ohm: DeclaredHardInterval,
    /// Hard interval for the imaginary part of impedance in ohms.
    pub imaginary_ohm: DeclaredHardInterval,
}

impl ComplexImpedanceHardBounds {
    /// Validate interval ordering, nominal containment and passive real impedance.
    pub fn validate_against(
        &self,
        sweep: &CapacitorImpedanceSweep,
    ) -> Result<(), CapacitorSpectrumBoundsError> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "impedance-bound frequency must be finite and positive",
            ));
        }
        let point = sweep
            .points
            .iter()
            .find(|point| point.frequency_hz == self.frequency_hz)
            .ok_or(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "impedance bounds must refer to an exact measured sweep knot",
            ))?;
        self.real_ohm.validate_contains(point.real_ohm)?;
        self.imaginary_ohm.validate_contains(point.imaginary_ohm)?;
        if self.real_ohm.lower < 0.0 {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "real-impedance lower bound must be non-negative for this passive-device contract",
            ));
        }
        Ok(())
    }
}

/// Hard bounds for the RMS magnitude of one exact positive-frequency current component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CurrentRmsHardBounds {
    /// Exact spectral component frequency in hertz.
    pub frequency_hz: f64,
    /// Deterministic interval for current RMS magnitude in amperes.
    pub current_rms_a: DeclaredHardInterval,
}

impl CurrentRmsHardBounds {
    /// Validate bounds against the source-bound current-spectrum component.
    pub fn validate_against(
        &self,
        spectrum: &CapacitorCurrentSpectrum,
    ) -> Result<(), CapacitorSpectrumBoundsError> {
        if !self.frequency_hz.is_finite() || self.frequency_hz <= 0.0 {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "current-bound frequency must be finite and positive",
            ));
        }
        let component = spectrum
            .components
            .iter()
            .find(|component| component.frequency_hz == self.frequency_hz)
            .ok_or(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "current bounds must refer to an exact spectrum component",
            ))?;
        self.current_rms_a.validate_contains(component.current_rms_a)?;
        if self.current_rms_a.lower < 0.0 {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "RMS current lower bound must be non-negative",
            ));
        }
        Ok(())
    }
}

/// Closed numeric bounds for a derived metric.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericInterval {
    /// Inclusive lower endpoint.
    pub lower: f64,
    /// Inclusive upper endpoint.
    pub upper: f64,
}

/// Derived bounds for a single exact frequency component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedSpectrumComponentMetrics {
    /// Exact current-spectrum frequency in hertz.
    pub frequency_hz: f64,
    /// Hard-bounded current RMS magnitude in amperes.
    pub current_rms_a: NumericInterval,
    /// Hard-bounded real impedance in ohms.
    pub impedance_real_ohm: NumericInterval,
    /// Hard-bounded imaginary impedance in ohms.
    pub impedance_imaginary_ohm: NumericInterval,
    /// Conservative interval for impedance magnitude in ohms.
    pub impedance_magnitude_ohm: NumericInterval,
    /// Conservative interval for this component's RMS ripple-voltage contribution, in volts.
    pub ripple_voltage_rms_v: NumericInterval,
    /// Conservative interval for this component's real-power dissipation, in watts.
    pub dissipated_power_w: NumericInterval,
    /// Evidence state: bounded path requires exact measured knots, not nominal interpolation.
    pub evaluation_kind: ImpedanceEvaluationKind,
}

/// Aggregate hard-bounded metrics over a source-bound AC current spectrum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedCapacitorSpectrumMetrics {
    /// Quadrature total RMS current bounds, in amperes.
    pub total_current_rms_a: NumericInterval,
    /// Quadrature aggregate RMS ripple voltage bounds, in volts.
    pub total_ripple_voltage_rms_v: NumericInterval,
    /// Sum of per-frequency real-power loss bounds, in watts.
    pub total_loss_power_w: NumericInterval,
    /// Bounds for the maximum component impedance magnitude, in ohms.
    pub maximum_component_impedance_ohm: NumericInterval,
    /// First-order steady-state temperature-rise bounds, if thermal resistance is supplied.
    pub modeled_temperature_rise_k: Option<NumericInterval>,
    /// Per-frequency bounded metrics retaining each source frequency.
    pub components: Vec<BoundedSpectrumComponentMetrics>,
}

impl BoundedCapacitorSpectrumMetrics {
    fn validate_finite(&self) -> Result<(), CapacitorSpectrumBoundsError> {
        let intervals = [
            self.total_current_rms_a,
            self.total_ripple_voltage_rms_v,
            self.total_loss_power_w,
            self.maximum_component_impedance_ohm,
        ];
        if intervals.iter().any(|interval| {
            !interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower < 0.0
                || interval.upper < interval.lower
        }) || self.modeled_temperature_rise_k.is_some_and(|interval| {
            !interval.lower.is_finite()
                || !interval.upper.is_finite()
                || interval.lower < 0.0
                || interval.upper < interval.lower
        }) {
            return Err(CapacitorSpectrumBoundsError::NonFiniteCalculatedMetric);
        }
        for component in &self.components {
            let intervals = [
                component.current_rms_a,
                component.impedance_real_ohm,
                component.impedance_magnitude_ohm,
                component.ripple_voltage_rms_v,
                component.dissipated_power_w,
            ];
            if intervals.iter().any(|interval| {
                !interval.lower.is_finite()
                    || !interval.upper.is_finite()
                    || interval.lower < 0.0
                    || interval.upper < interval.lower
            }) || !component.impedance_imaginary_ohm.lower.is_finite()
                || !component.impedance_imaginary_ohm.upper.is_finite()
                || component.impedance_imaginary_ohm.upper < component.impedance_imaginary_ohm.lower
            {
                return Err(CapacitorSpectrumBoundsError::NonFiniteCalculatedMetric);
            }
        }
        Ok(())
    }
}

/// Conservative disposition for the upper-limit constraints in the spectrum screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundedSpectrumDisposition {
    /// All declared upper-limit constraints are satisfied by the full derived intervals.
    PassesConservativeBounds,
    /// One or more interval upper endpoints exceed a declared limit; this is not proof the physical device fails.
    CannotGuaranteeConstraints,
    /// Evidence or an applicability condition does not support interval evaluation.
    NotEvaluable,
}

/// Why a bounded decision could not be computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundedSpectrumLimitation {
    /// The nominal screening gate has a condition/evidence limitation.
    NominalScreenNotEvaluable,
    /// No hard impedance interval was supplied for an active spectrum frequency.
    MissingImpedanceHardBounds,
    /// No hard RMS-current interval was supplied for an active spectrum frequency.
    MissingCurrentHardBounds,
    /// At least one active frequency is interpolated rather than an exact measured sweep knot.
    InterpolatedFrequencyNeedsHardDiscrepancyBound,
}

/// Which declared upper-limit requirement is not guaranteed by the calculated interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundedSpectrumConstraintViolation {
    /// The upper bound of maximum component impedance exceeds its limit.
    ImpedanceNotGuaranteedBelowMaximum,
    /// Upper bound of total real-power loss exceeds its limit.
    TotalLossNotGuaranteedBelowMaximum,
    /// Upper bound of aggregate RMS ripple voltage exceeds its limit.
    RippleVoltageNotGuaranteedBelowMaximum,
    /// Upper bound of the first-order temperature rise exceeds its limit.
    TemperatureRiseNotGuaranteedBelowMaximum,
}

/// Auditable hard-bound report retaining nominal screen and bounded metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundedCapacitorSpectrumScreeningReport {
    /// Nominal screen report, preserved without replacing its disposition.
    pub nominal_report: CapacitorSpectrumScreeningReport,
    /// Decision based on deterministic interval upper endpoints.
    pub disposition: BoundedSpectrumDisposition,
    /// Missing-evidence or applicability reasons preventing a bounded decision.
    pub limitations: Vec<BoundedSpectrumLimitation>,
    /// Upper-limit constraints whose satisfaction cannot be guaranteed.
    pub violations: Vec<BoundedSpectrumConstraintViolation>,
    /// Hard-bounded metrics; absent when evidence or applicability prevents evaluation.
    pub bounded_metrics: Option<BoundedCapacitorSpectrumMetrics>,
}

/// Error while validating interval provenance or calculating bounded metrics.
#[derive(Debug, Clone, PartialEq)]
pub enum CapacitorSpectrumBoundsError {
    /// Invalid or untraceable hard interval inputs.
    InvalidHardBounds(&'static str),
    /// Nominal spectrum screening input/evaluation error.
    Nominal(CapacitorSpectrumError),
    /// Derived interval endpoint is non-finite or malformed.
    NonFiniteCalculatedMetric,
}

impl fmt::Display for CapacitorSpectrumBoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHardBounds(message) => {
                write!(f, "invalid capacitor spectrum hard bounds: {message}")
            }
            Self::Nominal(error) => write!(f, "nominal capacitor spectrum error: {error}"),
            Self::NonFiniteCalculatedMetric => {
                write!(f, "non-finite or invalid derived capacitor spectrum interval")
            }
        }
    }
}

impl Error for CapacitorSpectrumBoundsError {}

impl From<CapacitorSpectrumError> for CapacitorSpectrumBoundsError {
    fn from(value: CapacitorSpectrumError) -> Self {
        Self::Nominal(value)
    }
}

fn validate_hard_bound_collections(
    sweep: &CapacitorImpedanceSweep,
    spectrum: &CapacitorCurrentSpectrum,
    impedance_bounds: &[ComplexImpedanceHardBounds],
    current_bounds: &[CurrentRmsHardBounds],
) -> Result<(), CapacitorSpectrumBoundsError> {
    let mut last_frequency = 0.0;
    for bound in impedance_bounds {
        if bound.frequency_hz <= last_frequency {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "impedance-bound records must have strictly increasing unique frequencies",
            ));
        }
        bound.validate_against(sweep)?;
        last_frequency = bound.frequency_hz;
    }
    last_frequency = 0.0;
    for bound in current_bounds {
        if bound.frequency_hz <= last_frequency {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "current-bound records must have strictly increasing unique frequencies",
            ));
        }
        bound.validate_against(spectrum)?;
        last_frequency = bound.frequency_hz;
    }
    Ok(())
}

fn abs_interval_minimum(lower: f64, upper: f64) -> f64 {
    if lower <= 0.0 && upper >= 0.0 {
        0.0
    } else {
        lower.abs().min(upper.abs())
    }
}

/// Calculate conservative intervals from hard bounds at exact measured sweep knots.
///
/// No interpolation is allowed here. Exact nominal knots may have uncertainty; points
/// between knots need a validated interpolation-discrepancy bound before they can be
/// admitted to this robust path. Each real impedance and current magnitude is
/// non-negative, so the power interval is formed from the lower/lower and upper/upper
/// products. The result is conservative with respect to supplied intervals and the
/// declared model only.
pub fn calculate_bounded_capacitor_spectrum_metrics(
    sweep: &CapacitorImpedanceSweep,
    spectrum: &CapacitorCurrentSpectrum,
    impedance_bounds: &[ComplexImpedanceHardBounds],
    current_bounds: &[CurrentRmsHardBounds],
    thermal_resistance_k_per_w: Option<f64>,
) -> Result<BoundedCapacitorSpectrumMetrics, CapacitorSpectrumBoundsError> {
    sweep.validate()?;
    spectrum.validate()?;
    validate_hard_bound_collections(sweep, spectrum, impedance_bounds, current_bounds)?;
    if thermal_resistance_k_per_w.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
            "thermal resistance must be finite and positive when supplied",
        ));
    }

    let mut components = Vec::with_capacity(spectrum.components.len());
    let mut current_lower_squared = 0.0;
    let mut current_upper_squared = 0.0;
    let mut voltage_lower_squared = 0.0;
    let mut voltage_upper_squared = 0.0;
    let mut total_loss_lower_w = 0.0;
    let mut total_loss_upper_w = 0.0;
    let mut maximum_impedance_lower_ohm: f64 = 0.0;
    let mut maximum_impedance_upper_ohm: f64 = 0.0;

    for current in &spectrum.components {
        let point = sweep
            .points
            .iter()
            .find(|point| point.frequency_hz == current.frequency_hz)
            .ok_or(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "bounded calculation requires every current component to match an exact measured sweep knot",
            ))?;
        let impedance_bound = impedance_bounds
            .iter()
            .find(|bound| bound.frequency_hz == current.frequency_hz)
            .ok_or(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "hard impedance interval missing for an active current-spectrum frequency",
            ))?;
        let current_bound = current_bounds
            .iter()
            .find(|bound| bound.frequency_hz == current.frequency_hz)
            .ok_or(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "hard RMS-current interval missing for an active current-spectrum frequency",
            ))?;

        impedance_bound.validate_against(sweep)?;
        current_bound.validate_against(spectrum)?;

        let i = &current_bound.current_rms_a;
        let r = &impedance_bound.real_ohm;
        let x = &impedance_bound.imaginary_ohm;
        let minimum_abs_x = abs_interval_minimum(x.lower, x.upper);
        let maximum_abs_x = x.lower.abs().max(x.upper.abs());
        let z_magnitude = NumericInterval {
            lower: r.lower.hypot(minimum_abs_x),
            upper: r.upper.hypot(maximum_abs_x),
        };
        let ripple_voltage_rms_v = NumericInterval {
            lower: i.lower * z_magnitude.lower,
            upper: i.upper * z_magnitude.upper,
        };
        let dissipated_power_w = NumericInterval {
            lower: i.lower.powi(2) * r.lower,
            upper: i.upper.powi(2) * r.upper,
        };

        current_lower_squared += i.lower.powi(2);
        current_upper_squared += i.upper.powi(2);
        voltage_lower_squared += ripple_voltage_rms_v.lower.powi(2);
        voltage_upper_squared += ripple_voltage_rms_v.upper.powi(2);
        total_loss_lower_w += dissipated_power_w.lower;
        total_loss_upper_w += dissipated_power_w.upper;
        maximum_impedance_lower_ohm = maximum_impedance_lower_ohm.max(z_magnitude.lower);
        maximum_impedance_upper_ohm = maximum_impedance_upper_ohm.max(z_magnitude.upper);

        if point.frequency_hz != current.frequency_hz {
            return Err(CapacitorSpectrumBoundsError::InvalidHardBounds(
                "spectrum frequency did not map to its exact measured impedance knot",
            ));
        }
        components.push(BoundedSpectrumComponentMetrics {
            frequency_hz: current.frequency_hz,
            current_rms_a: NumericInterval {
                lower: i.lower,
                upper: i.upper,
            },
            impedance_real_ohm: NumericInterval {
                lower: r.lower,
                upper: r.upper,
            },
            impedance_imaginary_ohm: NumericInterval {
                lower: x.lower,
                upper: x.upper,
            },
            impedance_magnitude_ohm: z_magnitude,
            ripple_voltage_rms_v,
            dissipated_power_w,
            evaluation_kind: ImpedanceEvaluationKind::MeasuredKnot,
        });
    }

    let total_loss_power_w = NumericInterval {
        lower: total_loss_lower_w,
        upper: total_loss_upper_w,
    };
    let metrics = BoundedCapacitorSpectrumMetrics {
        total_current_rms_a: NumericInterval {
            lower: current_lower_squared.sqrt(),
            upper: current_upper_squared.sqrt(),
        },
        total_ripple_voltage_rms_v: NumericInterval {
            lower: voltage_lower_squared.sqrt(),
            upper: voltage_upper_squared.sqrt(),
        },
        total_loss_power_w,
        maximum_component_impedance_ohm: NumericInterval {
            lower: maximum_impedance_lower_ohm,
            upper: maximum_impedance_upper_ohm,
        },
        modeled_temperature_rise_k: thermal_resistance_k_per_w.map(|resistance| NumericInterval {
            lower: total_loss_power_w.lower * resistance,
            upper: total_loss_power_w.upper * resistance,
        }),
        components,
    };
    metrics.validate_finite()?;
    Ok(metrics)
}

/// Screen a spectrum using nominal constraints and deterministic hard bounds.
///
/// A conservative pass means the upper endpoints satisfy the declared upper-limit
/// requirements throughout the supplied hard intervals. An upper bound that exceeds
/// a requirement produces CannotGuaranteeConstraints; it does not prove the physical
/// device fails. Any active frequency without an exact measured knot or any needed hard
/// interval makes the result NotEvaluable rather than granting a robust pass.
pub fn screen_capacitor_spectrum_bounded(
    sweep: &CapacitorImpedanceSweep,
    spectrum: &CapacitorCurrentSpectrum,
    requirements: &CapacitorSpectrumScreeningRequirements,
    impedance_bounds: &[ComplexImpedanceHardBounds],
    current_bounds: &[CurrentRmsHardBounds],
) -> Result<BoundedCapacitorSpectrumScreeningReport, CapacitorSpectrumBoundsError> {
    let nominal_report =
        screen_capacitor_spectrum(sweep, spectrum, requirements).map_err(|error| {
            CapacitorSpectrumBoundsError::Nominal(error)
        })?;
    if nominal_report.disposition == CapacitorSpectrumDisposition::NotEvaluable {
        return Ok(BoundedCapacitorSpectrumScreeningReport {
            nominal_report,
            disposition: BoundedSpectrumDisposition::NotEvaluable,
            limitations: vec![BoundedSpectrumLimitation::NominalScreenNotEvaluable],
            violations: Vec::new(),
            bounded_metrics: None,
        });
    }

    validate_hard_bound_collections(sweep, spectrum, impedance_bounds, current_bounds)?;
    let mut limitations = Vec::new();
    for component in &spectrum.components {
        if !sweep
            .points
            .iter()
            .any(|point| point.frequency_hz == component.frequency_hz)
        {
            limitations.push(
                BoundedSpectrumLimitation::InterpolatedFrequencyNeedsHardDiscrepancyBound,
            );
            continue;
        }
        if !impedance_bounds
            .iter()
            .any(|bound| bound.frequency_hz == component.frequency_hz)
        {
            limitations.push(BoundedSpectrumLimitation::MissingImpedanceHardBounds);
        }
        if !current_bounds
            .iter()
            .any(|bound| bound.frequency_hz == component.frequency_hz)
        {
            limitations.push(BoundedSpectrumLimitation::MissingCurrentHardBounds);
        }
    }
    limitations.sort_by_key(|limitation| *limitation as u8);
    limitations.dedup();
    if !limitations.is_empty() {
        return Ok(BoundedCapacitorSpectrumScreeningReport {
            nominal_report,
            disposition: BoundedSpectrumDisposition::NotEvaluable,
            limitations,
            violations: Vec::new(),
            bounded_metrics: None,
        });
    }

    let bounded_metrics = calculate_bounded_capacitor_spectrum_metrics(
        sweep,
        spectrum,
        impedance_bounds,
        current_bounds,
        requirements.thermal_resistance_k_per_w,
    )?;
    let mut violations = Vec::new();
    if requirements.max_impedance_magnitude_ohm.is_some_and(|maximum| {
        bounded_metrics.maximum_component_impedance_ohm.upper > maximum
    }) {
        violations.push(
            BoundedSpectrumConstraintViolation::ImpedanceNotGuaranteedBelowMaximum,
        );
    }
    if bounded_metrics.total_loss_power_w.upper > requirements.max_total_loss_power_w {
        violations.push(BoundedSpectrumConstraintViolation::TotalLossNotGuaranteedBelowMaximum);
    }
    if requirements.max_total_ripple_voltage_rms_v.is_some_and(|maximum| {
        bounded_metrics.total_ripple_voltage_rms_v.upper > maximum
    }) {
        violations.push(
            BoundedSpectrumConstraintViolation::RippleVoltageNotGuaranteedBelowMaximum,
        );
    }
    if requirements.max_temperature_rise_k.is_some_and(|maximum| {
        bounded_metrics
            .modeled_temperature_rise_k
            .is_none_or(|rise| rise.upper > maximum)
    }) {
        violations.push(
            BoundedSpectrumConstraintViolation::TemperatureRiseNotGuaranteedBelowMaximum,
        );
    }

    let disposition = if violations.is_empty() {
        BoundedSpectrumDisposition::PassesConservativeBounds
    } else {
        BoundedSpectrumDisposition::CannotGuaranteeConstraints
    };
    Ok(BoundedCapacitorSpectrumScreeningReport {
        nominal_report,
        disposition,
        limitations: Vec::new(),
        violations,
        bounded_metrics: Some(bounded_metrics),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capacitor_spectrum::{
        CapacitorDeviceEvidenceKind, CapacitorSpectrumScreeningRequirements, CurrentSpectrumComponent,
        ImpedanceSweepPoint,
    };

    fn hard_interval(lower: f64, upper: f64, method: &str) -> DeclaredHardInterval {
        DeclaredHardInterval {
            lower,
            upper,
            method_reference: method.to_string(),
        }
    }

    fn sweep() -> CapacitorImpedanceSweep {
        CapacitorImpedanceSweep {
            device_id: "synthetic-bounded-sweep".to_string(),
            characterization_temperature_k: 298.15,
            dc_bias_v: None,
            source_reference: "test-fixture://not-real-bounded-sweep".to_string(),
            method_reference: "synthetic deterministic fixture".to_string(),
            evidence_kind: CapacitorDeviceEvidenceKind::SyntheticFixture,
            points: vec![
                ImpedanceSweepPoint {
                    frequency_hz: 1.0e3,
                    real_ohm: 0.1,
                    imaginary_ohm: -1.0,
                },
                ImpedanceSweepPoint {
                    frequency_hz: 1.0e4,
                    real_ohm: 0.2,
                    imaginary_ohm: -0.5,
                },
            ],
        }
    }

    fn spectrum() -> CapacitorCurrentSpectrum {
        CapacitorCurrentSpectrum {
            source_reference: "test-fixture://not-real-bounded-spectrum".to_string(),
            method_reference: "synthetic RMS spectrum fixture".to_string(),
            components: vec![
                CurrentSpectrumComponent {
                    frequency_hz: 1.0e3,
                    current_rms_a: 0.5,
                },
                CurrentSpectrumComponent {
                    frequency_hz: 1.0e4,
                    current_rms_a: 0.2,
                },
            ],
        }
    }

    fn impedance_bounds() -> Vec<ComplexImpedanceHardBounds> {
        vec![
            ComplexImpedanceHardBounds {
                frequency_hz: 1.0e3,
                real_ohm: hard_interval(0.09, 0.11, "synthetic hard R interval"),
                imaginary_ohm: hard_interval(-1.1, -0.9, "synthetic hard X interval"),
            },
            ComplexImpedanceHardBounds {
                frequency_hz: 1.0e4,
                real_ohm: hard_interval(0.18, 0.22, "synthetic hard R interval"),
                imaginary_ohm: hard_interval(-0.6, -0.4, "synthetic hard X interval"),
            },
        ]
    }

    fn current_bounds() -> Vec<CurrentRmsHardBounds> {
        vec![
            CurrentRmsHardBounds {
                frequency_hz: 1.0e3,
                current_rms_a: hard_interval(0.45, 0.55, "synthetic hard current interval"),
            },
            CurrentRmsHardBounds {
                frequency_hz: 1.0e4,
                current_rms_a: hard_interval(0.18, 0.22, "synthetic hard current interval"),
            },
        ]
    }

    fn requirements() -> CapacitorSpectrumScreeningRequirements {
        CapacitorSpectrumScreeningRequirements {
            operating_temperature_k: 298.15,
            max_temperature_deviation_k: 1.0,
            operating_dc_bias_v: None,
            max_absolute_dc_bias_deviation_v: None,
            max_impedance_magnitude_ohm: Some(2.0),
            max_total_loss_power_w: 0.08,
            max_total_ripple_voltage_rms_v: Some(1.0),
            thermal_resistance_k_per_w: Some(5.0),
            max_temperature_rise_k: Some(0.5),
            accepted_evidence_kinds: vec![CapacitorDeviceEvidenceKind::SyntheticFixture],
        }
    }

    #[test]
    fn hard_interval_must_contain_nominal_value() {
        assert!(hard_interval(0.2, 0.3, "declared").validate_contains(0.1).is_err());
    }

    #[test]
    fn passive_impedance_bound_rejects_negative_real_lower_endpoint() {
        let mut bounds = impedance_bounds();
        bounds[0].real_ohm = hard_interval(-0.01, 0.11, "not passive");
        assert!(matches!(
            bounds[0].validate_against(&sweep()),
            Err(CapacitorSpectrumBoundsError::InvalidHardBounds(_))
        ));
    }

    #[test]
    fn hard_metric_intervals_enclose_nominal_spectrum_metrics() {
        let s = sweep();
        let i = spectrum();
        let nominal = calculate_capacitor_spectrum_metrics(&s, &i, Some(5.0)).unwrap();
        let bounded = calculate_bounded_capacitor_spectrum_metrics(
            &s,
            &i,
            &impedance_bounds(),
            &current_bounds(),
            Some(5.0),
        )
        .unwrap();
        assert!(bounded.total_loss_power_w.lower <= nominal.total_loss_power_w);
        assert!(bounded.total_loss_power_w.upper >= nominal.total_loss_power_w);
        assert!(bounded.total_ripple_voltage_rms_v.lower <= nominal.total_ripple_voltage_rms_v);
        assert!(bounded.total_ripple_voltage_rms_v.upper >= nominal.total_ripple_voltage_rms_v);
        assert!(bounded.modeled_temperature_rise_k.unwrap().upper >= nominal.modeled_temperature_rise_k.unwrap());
    }

    #[test]
    fn hard_bounds_can_pass_when_all_upper_limits_are_respected() {
        let report = screen_capacitor_spectrum_bounded(
            &sweep(),
            &spectrum(),
            &requirements(),
            &impedance_bounds(),
            &current_bounds(),
        )
        .unwrap();
        assert_eq!(report.disposition, BoundedSpectrumDisposition::PassesConservativeBounds);
        assert!(report.violations.is_empty());
        assert!(report.bounded_metrics.is_some());
    }

    #[test]
    fn nominal_pass_cannot_hide_worst_case_loss_failure() {
        let mut req = requirements();
        req.max_total_loss_power_w = 0.04;
        let report = screen_capacitor_spectrum_bounded(
            &sweep(),
            &spectrum(),
            &req,
            &impedance_bounds(),
            &current_bounds(),
        )
        .unwrap();
        assert_eq!(
            report.nominal_report.disposition,
            CapacitorSpectrumDisposition::PassesSpectrumScreen
        );
        assert_eq!(
            report.disposition,
            BoundedSpectrumDisposition::CannotGuaranteeConstraints
        );
        assert!(report
            .violations
            .contains(&BoundedSpectrumConstraintViolation::TotalLossNotGuaranteedBelowMaximum));
    }

    #[test]
    fn interpolation_without_an_error_bound_is_not_evaluable() {
        let mut current = spectrum();
        current.components[0].frequency_hz = 3.0e3;
        let mut bounded_currents = current_bounds();
        bounded_currents[0].frequency_hz = 3.0e3;
        let report = screen_capacitor_spectrum_bounded(
            &sweep(),
            &current,
            &requirements(),
            &impedance_bounds(),
            &bounded_currents,
        )
        .unwrap();
        assert_eq!(report.disposition, BoundedSpectrumDisposition::NotEvaluable);
        assert!(report.limitations.contains(
            &BoundedSpectrumLimitation::InterpolatedFrequencyNeedsHardDiscrepancyBound
        ));
        assert!(report.bounded_metrics.is_none());
    }

    #[test]
    fn missing_active_current_interval_is_not_evaluable() {
        let mut currents = current_bounds();
        currents.pop();
        let report = screen_capacitor_spectrum_bounded(
            &sweep(),
            &spectrum(),
            &requirements(),
            &impedance_bounds(),
            &currents,
        )
        .unwrap();
        assert_eq!(report.disposition, BoundedSpectrumDisposition::NotEvaluable);
        assert!(report.limitations.contains(&BoundedSpectrumLimitation::MissingCurrentHardBounds));
    }

    #[test]
    fn missing_active_impedance_interval_is_not_evaluable() {
        let mut impedances = impedance_bounds();
        impedances.pop();
        let report = screen_capacitor_spectrum_bounded(
            &sweep(),
            &spectrum(),
            &requirements(),
            &impedances,
            &current_bounds(),
        )
        .unwrap();
        assert_eq!(report.disposition, BoundedSpectrumDisposition::NotEvaluable);
        assert!(report.limitations.contains(&BoundedSpectrumLimitation::MissingImpedanceHardBounds));
    }

    #[test]
    fn invalid_or_duplicate_hard_bound_frequencies_are_rejected() {
        let mut bounds = impedance_bounds();
        bounds[1].frequency_hz = bounds[0].frequency_hz;
        assert!(matches!(
            validate_hard_bound_collections(&sweep(), &spectrum(), &bounds, &current_bounds()),
            Err(CapacitorSpectrumBoundsError::InvalidHardBounds(_))
        ));
    }

    #[test]
    fn upper_ripple_voltage_violation_is_reported_independently() {
        let mut req = requirements();
        req.max_total_ripple_voltage_rms_v = Some(0.1);
        req.max_total_loss_power_w = 100.0;
        req.max_impedance_magnitude_ohm = Some(100.0);
        req.max_temperature_rise_k = Some(100.0);
        let report = screen_capacitor_spectrum_bounded(
            &sweep(),
            &spectrum(),
            &req,
            &impedance_bounds(),
            &current_bounds(),
        )
        .unwrap();
        assert!(report.violations.contains(
            &BoundedSpectrumConstraintViolation::RippleVoltageNotGuaranteedBelowMaximum
        ));
        assert!(!report.violations.contains(
            &BoundedSpectrumConstraintViolation::TotalLossNotGuaranteedBelowMaximum
        ));
    }

    #[test]
    fn hard_bound_validation_rejects_empty_method_reference() {
        let mut bounds = current_bounds();
        bounds[0].current_rms_a.method_reference.clear();
        assert!(matches!(
            bounds[0].validate_against(&spectrum()),
            Err(CapacitorSpectrumBoundsError::InvalidHardBounds(_))
        ));
    }

    #[test]
    fn nonfinite_bound_endpoint_is_rejected() {
        let mut bounds = impedance_bounds();
        bounds[0].imaginary_ohm.upper = f64::INFINITY;
        assert!(matches!(
            bounds[0].validate_against(&sweep()),
            Err(CapacitorSpectrumBoundsError::InvalidHardBounds(_))
        ));
    }
}
