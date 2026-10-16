// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Interval-integrated first-law accounting for transient energy-conversion tests.
//!
//! Convention: positive stored-energy change means the system ended the interval
//! with more energy stored; negative means stored energy was released. The
//! balance is:
//!
//! input = net useful output + auxiliary consumption + rejected/lost energy
//!         + change in stored energy.
//!
//! This accounting primitive does not model the transient physics itself. The
//! supplied values must be integrated/measured over the same registered interval
//! and system boundary. It prevents a battery or thermal store being discharged
//! during a test and having that contribution silently misreported as engine
//! conversion efficiency.

/// Integrated energy flows over a single, explicitly bounded interval, in joules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyIntervalBalance {
    /// Total external energy crossing into the registered system boundary.
    pub input_energy_j: f64,
    /// Useful energy delivered to the mission outside the system boundary.
    pub net_useful_output_energy_j: f64,
    /// Energy consumed by internal auxiliaries and not delivered to the mission.
    pub auxiliary_energy_j: f64,
    /// Rejected heat and other energy leaving without useful output.
    pub rejected_or_loss_energy_j: f64,
    /// Final stored energy minus initial stored energy; positive means storage charged.
    pub stored_energy_change_j: f64,
}

/// Validated interval-integrated energy accounting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyIntervalReport {
    /// Input plus energy released from storage during this interval.
    pub energy_throughput_j: f64,
    /// Input minus output, auxiliaries, losses, and stored-energy change.
    pub residual_energy_j: f64,
    /// Absolute residual divided by energy throughput.
    pub relative_balance_error: f64,
}

/// Failure reasons for interval energy-balance validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnergyIntervalError {
    /// An input, flow, stored-energy change, or tolerance is NaN or infinite.
    NonFiniteInput,
    /// External input, output, auxiliary consumption, or rejected/lost energy cannot be negative.
    NegativeEnergyFlow,
    /// Relative tolerance must be finite and in the range [0, 1).
    InvalidTolerance,
    /// The interval contains no external input or net discharge from storage.
    NoEnergyThroughput,
    /// Summing the supplied energy flows overflowed the numeric representation.
    NumericOverflow,
    /// First-law residual exceeds the declared relative tolerance.
    FirstLawImbalance { relative_error: f64, tolerance: f64 },
}

impl std::fmt::Display for EnergyIntervalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput => f.write_str("all energy flows and tolerance must be finite"),
            Self::NegativeEnergyFlow => f.write_str("input, output, auxiliary, and loss energy flows cannot be negative"),
            Self::InvalidTolerance => f.write_str("relative tolerance must be finite and in the range [0, 1)"),
            Self::NoEnergyThroughput => f.write_str("interval must contain external input or net storage discharge"),
            Self::NumericOverflow => f.write_str("summing interval energy flows overflowed"),
            Self::FirstLawImbalance { relative_error, tolerance } => write!(
                f,
                "first-law interval balance error {relative_error:.6e} exceeds tolerance {tolerance:.6e}"
            ),
        }
    }
}

impl std::error::Error for EnergyIntervalError {}

impl EnergyIntervalBalance {
    /// Validate first-law closure over the interval.
    ///
    /// The error tolerance is relative to total energy throughput, defined as
    /// external input plus net energy released from storage. Choose this
    /// tolerance before examining candidate performance, and justify it from
    /// numerical and/or measurement uncertainty. This method intentionally does
    /// not report an efficiency: storage discharge may make output exceed the
    /// external input in an interval without violating energy conservation.
    pub fn evaluate(
        self,
        relative_tolerance: f64,
    ) -> Result<EnergyIntervalReport, EnergyIntervalError> {
        let flows = [
            self.input_energy_j,
            self.net_useful_output_energy_j,
            self.auxiliary_energy_j,
            self.rejected_or_loss_energy_j,
            self.stored_energy_change_j,
            relative_tolerance,
        ];
        if flows.iter().any(|value| !value.is_finite()) {
            return Err(EnergyIntervalError::NonFiniteInput);
        }
        if self.input_energy_j < 0.0
            || self.net_useful_output_energy_j < 0.0
            || self.auxiliary_energy_j < 0.0
            || self.rejected_or_loss_energy_j < 0.0
        {
            return Err(EnergyIntervalError::NegativeEnergyFlow);
        }
        if !(0.0..1.0).contains(&relative_tolerance) {
            return Err(EnergyIntervalError::InvalidTolerance);
        }

        let stored_discharge_j = (-self.stored_energy_change_j).max(0.0);
        let energy_throughput_j = self.input_energy_j + stored_discharge_j;
        if !energy_throughput_j.is_finite() {
            return Err(EnergyIntervalError::NumericOverflow);
        }
        if energy_throughput_j <= 0.0 {
            return Err(EnergyIntervalError::NoEnergyThroughput);
        }

        let accounted_energy_j = self.net_useful_output_energy_j
            + self.auxiliary_energy_j
            + self.rejected_or_loss_energy_j
            + self.stored_energy_change_j;
        if !accounted_energy_j.is_finite() {
            return Err(EnergyIntervalError::NumericOverflow);
        }

        let residual_energy_j = self.input_energy_j - accounted_energy_j;
        if !residual_energy_j.is_finite() {
            return Err(EnergyIntervalError::NumericOverflow);
        }
        let relative_balance_error = residual_energy_j.abs() / energy_throughput_j;
        if !relative_balance_error.is_finite() {
            return Err(EnergyIntervalError::NumericOverflow);
        }
        if relative_balance_error > relative_tolerance {
            return Err(EnergyIntervalError::FirstLawImbalance {
                relative_error: relative_balance_error,
                tolerance: relative_tolerance,
            });
        }

        Ok(EnergyIntervalReport {
            energy_throughput_j,
            residual_energy_j,
            relative_balance_error,
        })
    }
}

/// Checked aggregate of individually validated intervals in one registered duty cycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyDutyCycleReport {
    /// Sum of all external input energy.
    pub total_input_energy_j: f64,
    /// Sum of useful energy delivered over the duty cycle.
    pub total_net_useful_output_energy_j: f64,
    /// Sum of auxiliary energy consumed.
    pub total_auxiliary_energy_j: f64,
    /// Sum of rejected and other loss energy.
    pub total_rejected_or_loss_energy_j: f64,
    /// Net stored-energy change from the start to end of the duty cycle.
    pub net_stored_energy_change_j: f64,
    /// Total input plus energy released from storage across intervals.
    pub total_energy_throughput_j: f64,
    /// Aggregate residual after every individual interval has passed closure.
    pub residual_energy_j: f64,
    /// Aggregate residual divided by total energy throughput.
    pub relative_balance_error: f64,
    /// Useful output divided by external input, available only when net storage
    /// change is within the registered relative tolerance of total throughput.
    /// This is a mission-cycle ratio, not a component or fuel-tank-to-wheel metric.
    pub net_cycle_efficiency: Option<f64>,
}

/// Errors from evaluating a sequence of transient energy intervals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnergyDutyCycleError {
    /// No intervals were supplied.
    EmptyCycle,
    /// An individual interval failed validation before aggregation.
    InvalidInterval {
        /// Zero-based position in the supplied interval sequence.
        index: usize,
        /// The original interval validation failure.
        error: EnergyIntervalError,
    },
    /// A sum of individually finite intervals overflowed.
    NumericOverflow,
    /// The aggregate conservation check failed.
    AggregateImbalance(EnergyIntervalError),
}

impl std::fmt::Display for EnergyDutyCycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCycle => f.write_str("duty cycle must contain at least one interval"),
            Self::InvalidInterval { index, error } => {
                write!(f, "duty-cycle interval {index} is invalid: {error}")
            }
            Self::NumericOverflow => f.write_str("aggregating duty-cycle energy flows overflowed"),
            Self::AggregateImbalance(error) => write!(f, "aggregate duty-cycle balance failed: {error}"),
        }
    }
}

impl std::error::Error for EnergyDutyCycleError {}

/// Validate each interval first, then aggregate raw energies across the duty cycle.
///
/// Validating each interval before summation prevents positive and negative
/// residuals from cancelling and concealing invalid observations. The aggregate
/// check is performed as a second gate. This function expects already integrated
/// energy quantities (joules); it does not infer duration-weighted energy from
/// power readings or apply any system-boundary conversions.
pub fn evaluate_duty_cycle(
    intervals: &[EnergyIntervalBalance],
    relative_tolerance: f64,
) -> Result<EnergyDutyCycleReport, EnergyDutyCycleError> {
    if intervals.is_empty() {
        return Err(EnergyDutyCycleError::EmptyCycle);
    }

    let mut total_input_energy_j = 0.0;
    let mut total_net_useful_output_energy_j = 0.0;
    let mut total_auxiliary_energy_j = 0.0;
    let mut total_rejected_or_loss_energy_j = 0.0;
    let mut net_stored_energy_change_j = 0.0;

    for (index, interval) in intervals.iter().copied().enumerate() {
        interval
            .evaluate(relative_tolerance)
            .map_err(|error| EnergyDutyCycleError::InvalidInterval { index, error })?;

        total_input_energy_j += interval.input_energy_j;
        total_net_useful_output_energy_j += interval.net_useful_output_energy_j;
        total_auxiliary_energy_j += interval.auxiliary_energy_j;
        total_rejected_or_loss_energy_j += interval.rejected_or_loss_energy_j;
        net_stored_energy_change_j += interval.stored_energy_change_j;

        if [
            total_input_energy_j,
            total_net_useful_output_energy_j,
            total_auxiliary_energy_j,
            total_rejected_or_loss_energy_j,
            net_stored_energy_change_j,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(EnergyDutyCycleError::NumericOverflow);
        }
    }

    let aggregate = EnergyIntervalBalance {
        input_energy_j: total_input_energy_j,
        net_useful_output_energy_j: total_net_useful_output_energy_j,
        auxiliary_energy_j: total_auxiliary_energy_j,
        rejected_or_loss_energy_j: total_rejected_or_loss_energy_j,
        stored_energy_change_j: net_stored_energy_change_j,
    };
    let aggregate_report = aggregate
        .evaluate(relative_tolerance)
        .map_err(EnergyDutyCycleError::AggregateImbalance)?;

    let storage_is_balanced =
        net_stored_energy_change_j.abs()
            <= relative_tolerance * aggregate_report.energy_throughput_j;
    // A tiny net discharge within tolerance can still inflate output/input. Be
    // conservative: only expose the ratio when storage is balanced and does not
    // show a net discharge.
    let net_cycle_efficiency = if storage_is_balanced
        && net_stored_energy_change_j >= 0.0
        && total_input_energy_j > 0.0
    {
        Some(total_net_useful_output_energy_j / total_input_energy_j)
    } else {
        None
    };

    Ok(EnergyDutyCycleReport {
        total_input_energy_j,
        total_net_useful_output_energy_j,
        total_auxiliary_energy_j,
        total_rejected_or_loss_energy_j,
        net_stored_energy_change_j,
        total_energy_throughput_j: aggregate_report.energy_throughput_j,
        residual_energy_j: aggregate_report.residual_energy_j,
        relative_balance_error: aggregate_report.relative_balance_error,
        net_cycle_efficiency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_without_storage_change_matches_closed_steady_balance() {
        let report = EnergyIntervalBalance {
            input_energy_j: 1000.0,
            net_useful_output_energy_j: 300.0,
            auxiliary_energy_j: 20.0,
            rejected_or_loss_energy_j: 680.0,
            stored_energy_change_j: 0.0,
        }
        .evaluate(1e-9)
        .unwrap();

        assert!((report.energy_throughput_j - 1000.0).abs() < 1e-12);
        assert!(report.residual_energy_j.abs() < 1e-12);
        assert!(report.relative_balance_error < 1e-12);
    }

    #[test]
    fn energy_accumulated_in_storage_is_counted() {
        let report = EnergyIntervalBalance {
            input_energy_j: 1200.0,
            net_useful_output_energy_j: 300.0,
            auxiliary_energy_j: 20.0,
            rejected_or_loss_energy_j: 680.0,
            stored_energy_change_j: 200.0,
        }
        .evaluate(1e-9)
        .unwrap();

        assert!(report.residual_energy_j.abs() < 1e-12);
        assert!((report.energy_throughput_j - 1200.0).abs() < 1e-12);
    }

    #[test]
    fn storage_discharge_is_counted_as_energy_source_not_free_efficiency() {
        let report = EnergyIntervalBalance {
            input_energy_j: 800.0,
            net_useful_output_energy_j: 300.0,
            auxiliary_energy_j: 20.0,
            rejected_or_loss_energy_j: 680.0,
            stored_energy_change_j: -200.0,
        }
        .evaluate(1e-9)
        .unwrap();

        assert!((report.energy_throughput_j - 1000.0).abs() < 1e-12);
        assert!(report.residual_energy_j.abs() < 1e-12);
        // No external-input-only efficiency is returned for this transient interval.
    }

    #[test]
    fn storage_only_discharge_can_have_a_closed_balance() {
        let report = EnergyIntervalBalance {
            input_energy_j: 0.0,
            net_useful_output_energy_j: 70.0,
            auxiliary_energy_j: 5.0,
            rejected_or_loss_energy_j: 25.0,
            stored_energy_change_j: -100.0,
        }
        .evaluate(1e-9)
        .unwrap();

        assert!((report.energy_throughput_j - 100.0).abs() < 1e-12);
        assert!(report.residual_energy_j.abs() < 1e-12);
    }

    #[test]
    fn non_closing_or_invalid_interval_is_rejected() {
        let imbalance = EnergyIntervalBalance {
            input_energy_j: 1000.0,
            net_useful_output_energy_j: 300.0,
            auxiliary_energy_j: 20.0,
            rejected_or_loss_energy_j: 600.0,
            stored_energy_change_j: 0.0,
        };
        assert!(matches!(
            imbalance.evaluate(1e-6),
            Err(EnergyIntervalError::FirstLawImbalance { .. })
        ));

        let negative_output = EnergyIntervalBalance {
            net_useful_output_energy_j: -1.0,
            ..imbalance
        };
        assert_eq!(
            negative_output.evaluate(1e-6),
            Err(EnergyIntervalError::NegativeEnergyFlow)
        );

        let negative_input = EnergyIntervalBalance {
            input_energy_j: -1.0,
            net_useful_output_energy_j: 0.0,
            auxiliary_energy_j: 0.0,
            rejected_or_loss_energy_j: 0.0,
            stored_energy_change_j: 0.0,
        };
        assert_eq!(
            negative_input.evaluate(1e-6),
            Err(EnergyIntervalError::NegativeEnergyFlow)
        );

        let no_throughput = EnergyIntervalBalance {
            input_energy_j: 0.0,
            net_useful_output_energy_j: 0.0,
            auxiliary_energy_j: 0.0,
            rejected_or_loss_energy_j: 0.0,
            stored_energy_change_j: 0.0,
        };
        assert_eq!(
            no_throughput.evaluate(1e-6),
            Err(EnergyIntervalError::NoEnergyThroughput)
        );
        assert_eq!(
            no_throughput.evaluate(-1e-6),
            Err(EnergyIntervalError::InvalidTolerance)
        );
        assert_eq!(
            no_throughput.evaluate(1.0),
            Err(EnergyIntervalError::InvalidTolerance)
        );
    }

    #[test]
    fn duty_cycle_efficiency_requires_charge_sustaining_storage() {
        let intervals = [
            EnergyIntervalBalance {
                input_energy_j: 200.0,
                net_useful_output_energy_j: 50.0,
                auxiliary_energy_j: 10.0,
                rejected_or_loss_energy_j: 40.0,
                stored_energy_change_j: 100.0,
            },
            EnergyIntervalBalance {
                input_energy_j: 100.0,
                net_useful_output_energy_j: 50.0,
                auxiliary_energy_j: 10.0,
                rejected_or_loss_energy_j: 140.0,
                stored_energy_change_j: -100.0,
            },
        ];
        let report = evaluate_duty_cycle(&intervals, 1e-9).unwrap();

        assert!((report.total_input_energy_j - 300.0).abs() < 1e-12);
        assert!((report.total_net_useful_output_energy_j - 100.0).abs() < 1e-12);
        assert!(report.net_stored_energy_change_j.abs() < 1e-12);
        assert!((report.net_cycle_efficiency.unwrap() - (1.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn non_charge_sustaining_duty_cycle_does_not_claim_cycle_efficiency() {
        let intervals = [EnergyIntervalBalance {
            input_energy_j: 200.0,
            net_useful_output_energy_j: 50.0,
            auxiliary_energy_j: 10.0,
            rejected_or_loss_energy_j: 40.0,
            stored_energy_change_j: 100.0,
        }];
        let report = evaluate_duty_cycle(&intervals, 1e-9).unwrap();
        assert_eq!(report.net_cycle_efficiency, None);
    }

    #[test]
    fn small_net_storage_discharge_does_not_inflate_cycle_efficiency() {
        let intervals = [EnergyIntervalBalance {
            input_energy_j: 1000.0,
            net_useful_output_energy_j: 300.0,
            auxiliary_energy_j: 20.0,
            rejected_or_loss_energy_j: 680.00001,
            stored_energy_change_j: -0.00001,
        }];
        let report = evaluate_duty_cycle(&intervals, 1e-6).unwrap();
        assert!(report.net_stored_energy_change_j.abs()
            <= 1e-6 * report.total_energy_throughput_j);
        assert_eq!(report.net_cycle_efficiency, None);
    }

    #[test]
    fn duty_cycle_rejects_individually_unbalanced_intervals_before_aggregation() {
        // Residuals are +1 J and -1 J; aggregating first would hide both errors.
        let intervals = [
            EnergyIntervalBalance {
                input_energy_j: 100.0,
                net_useful_output_energy_j: 10.0,
                auxiliary_energy_j: 0.0,
                rejected_or_loss_energy_j: 89.0,
                stored_energy_change_j: 0.0,
            },
            EnergyIntervalBalance {
                input_energy_j: 100.0,
                net_useful_output_energy_j: 10.0,
                auxiliary_energy_j: 0.0,
                rejected_or_loss_energy_j: 91.0,
                stored_energy_change_j: 0.0,
            },
        ];
        assert!(matches!(
            evaluate_duty_cycle(&intervals, 1e-6),
            Err(EnergyDutyCycleError::InvalidInterval { index: 0, error: EnergyIntervalError::FirstLawImbalance { .. } })
        ));
    }

    #[test]
    fn empty_duty_cycle_is_rejected() {
        assert_eq!(
            evaluate_duty_cycle(&[], 1e-6),
            Err(EnergyDutyCycleError::EmptyCycle)
        );
    }

    #[test]
    fn overflow_and_non_finite_values_fail_closed() {
        let overflow = EnergyIntervalBalance {
            input_energy_j: f64::MAX,
            net_useful_output_energy_j: f64::MAX,
            auxiliary_energy_j: 0.0,
            rejected_or_loss_energy_j: 0.0,
            stored_energy_change_j: -f64::MAX,
        };
        assert_eq!(
            overflow.evaluate(1e-6),
            Err(EnergyIntervalError::NumericOverflow)
        );

        let non_finite = EnergyIntervalBalance {
            stored_energy_change_j: f64::NAN,
            ..overflow
        };
        assert_eq!(
            non_finite.evaluate(1e-6),
            Err(EnergyIntervalError::NonFiniteInput)
        );
    }
}
