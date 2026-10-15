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
    /// Output, auxiliary consumption, or rejected/lost energy cannot be negative.
    NegativeEnergyFlow,
    /// Relative tolerance must be finite and non-negative.
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
            Self::NegativeEnergyFlow => f.write_str("output, auxiliary, and loss energy flows cannot be negative"),
            Self::InvalidTolerance => f.write_str("relative tolerance must be finite and non-negative"),
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
        if self.net_useful_output_energy_j < 0.0
            || self.auxiliary_energy_j < 0.0
            || self.rejected_or_loss_energy_j < 0.0
        {
            return Err(EnergyIntervalError::NegativeEnergyFlow);
        }
        if relative_tolerance < 0.0 {
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
