// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Steady-state system-boundary energy accounting.
//!
//! The four power flows use mutually exclusive categories:
//! input = net useful output + auxiliary consumption + rejected/lost energy.
//! Auxiliary power is not also counted as useful output. This makes net
//! efficiency comparable only when competing systems use the same boundary and
//! account for auxiliary loads in the same way.
//!
//! This module is not a transient model: energy storage accumulation, changing
//! thermal inventories, and time-varying duty cycles must be modelled separately
//! or integrated over an explicitly defined measurement interval.

/// A steady-state energy balance expressed as power flows in watts.
///
/// All flows are non-negative. The output categories are mutually exclusive:
/// net useful output, auxiliary consumption, and rejected/other loss.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBalance {
    /// Total power entering the registered system boundary.
    pub input_power_w: f64,
    /// Useful output available to the mission after auxiliary consumption.
    pub net_useful_output_power_w: f64,
    /// Power consumed by internal auxiliaries and not available to the mission.
    pub auxiliary_power_w: f64,
    /// Rejected heat and all other energy leaving without useful output.
    pub rejected_or_loss_power_w: f64,
}

/// A checked steady-state energy-accounting result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBalanceReport {
    /// Total output before splitting it into mission-usable and auxiliary power.
    pub gross_output_power_w: f64,
    /// Input minus the sum of all three exclusive output categories.
    pub residual_power_w: f64,
    /// Absolute residual divided by input power.
    pub relative_balance_error: f64,
    /// Net useful output divided by total input.
    pub net_efficiency: f64,
    /// Net useful output plus auxiliaries, divided by total input.
    pub gross_efficiency: f64,
}

/// Failure reasons for validating steady-state energy flows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EnergyBalanceError {
    /// An input, flow, or tolerance is NaN or infinite.
    NonFiniteInput,
    /// Total input power must be strictly positive.
    NonPositiveInputPower,
    /// Output or loss flows cannot be negative.
    NegativePowerFlow,
    /// Relative tolerance must be finite and in the range [0, 1).
    InvalidTolerance,
    /// Reported gross output exceeds reported input, which is not admissible.
    OutputExceedsInput,
    /// Conservation residual exceeds the caller's declared relative tolerance.
    FirstLawImbalance { relative_error: f64, tolerance: f64 },
}

impl std::fmt::Display for EnergyBalanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteInput => f.write_str("all power flows and tolerance must be finite"),
            Self::NonPositiveInputPower => f.write_str("input power must be greater than zero"),
            Self::NegativePowerFlow => f.write_str("output and loss power flows cannot be negative"),
            Self::InvalidTolerance => f.write_str("relative tolerance must be finite and in the range [0, 1)"),
            Self::OutputExceedsInput => f.write_str("gross output cannot exceed input power"),
            Self::FirstLawImbalance { relative_error, tolerance } => write!(
                f,
                "first-law balance error {relative_error:.6e} exceeds tolerance {tolerance:.6e}"
            ),
        }
    }
}

impl std::error::Error for EnergyBalanceError {}

impl EnergyBalance {
    /// Validate the balance and calculate gross/net performance.
    ///
    /// The tolerance is relative to input power; a value of 1e-6 means a
    /// residual no larger than one part per million of input power. Choose the
    /// tolerance before evaluating candidate results and justify it from the
    /// numerical solver or measurement uncertainty; do not loosen it to pass a
    /// preferred candidate.
    pub fn evaluate(
        self,
        relative_tolerance: f64,
    ) -> Result<EnergyBalanceReport, EnergyBalanceError> {
        let flows = [
            self.input_power_w,
            self.net_useful_output_power_w,
            self.auxiliary_power_w,
            self.rejected_or_loss_power_w,
            relative_tolerance,
        ];
        if flows.iter().any(|v| !v.is_finite()) {
            return Err(EnergyBalanceError::NonFiniteInput);
        }
        if self.input_power_w <= 0.0 {
            return Err(EnergyBalanceError::NonPositiveInputPower);
        }
        if self.net_useful_output_power_w < 0.0
            || self.auxiliary_power_w < 0.0
            || self.rejected_or_loss_power_w < 0.0
        {
            return Err(EnergyBalanceError::NegativePowerFlow);
        }
        if !(0.0..1.0).contains(&relative_tolerance) {
            return Err(EnergyBalanceError::InvalidTolerance);
        }

        let gross_output_power_w =
            self.net_useful_output_power_w + self.auxiliary_power_w;
        if !gross_output_power_w.is_finite() {
            return Err(EnergyBalanceError::NonFiniteInput);
        }
        if gross_output_power_w > self.input_power_w {
            return Err(EnergyBalanceError::OutputExceedsInput);
        }

        let accounted_power_w = gross_output_power_w + self.rejected_or_loss_power_w;
        if !accounted_power_w.is_finite() {
            return Err(EnergyBalanceError::NonFiniteInput);
        }
        let residual_power_w = self.input_power_w - accounted_power_w;
        let relative_balance_error = residual_power_w.abs() / self.input_power_w;
        if relative_balance_error > relative_tolerance {
            return Err(EnergyBalanceError::FirstLawImbalance {
                relative_error: relative_balance_error,
                tolerance: relative_tolerance,
            });
        }

        Ok(EnergyBalanceReport {
            gross_output_power_w,
            residual_power_w,
            relative_balance_error,
            net_efficiency: self.net_useful_output_power_w / self.input_power_w,
            gross_efficiency: gross_output_power_w / self.input_power_w,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_balance_reports_net_and_gross_efficiency() {
        let report = EnergyBalance {
            input_power_w: 1000.0,
            net_useful_output_power_w: 300.0,
            auxiliary_power_w: 20.0,
            rejected_or_loss_power_w: 680.0,
        }
        .evaluate(1e-9)
        .unwrap();

        assert!((report.gross_output_power_w - 320.0).abs() < 1e-12);
        assert!((report.net_efficiency - 0.30).abs() < 1e-12);
        assert!((report.gross_efficiency - 0.32).abs() < 1e-12);
        assert!(report.relative_balance_error < 1e-12);
    }

    #[test]
    fn small_residual_is_accepted_only_within_declared_tolerance() {
        let balance = EnergyBalance {
            input_power_w: 1000.0,
            net_useful_output_power_w: 300.0,
            auxiliary_power_w: 20.0,
            rejected_or_loss_power_w: 679.999,
        };
        assert!(balance.evaluate(1e-6).is_ok());
        assert!(matches!(
            balance.evaluate(1e-7),
            Err(EnergyBalanceError::FirstLawImbalance { .. })
        ));
    }

    #[test]
    fn first_law_violation_is_rejected() {
        let error = EnergyBalance {
            input_power_w: 1000.0,
            net_useful_output_power_w: 300.0,
            auxiliary_power_w: 20.0,
            rejected_or_loss_power_w: 600.0,
        }
        .evaluate(1e-6)
        .unwrap_err();
        assert!(matches!(
            error,
            EnergyBalanceError::FirstLawImbalance { .. }
        ));
    }

    #[test]
    fn impossible_output_and_invalid_flows_are_rejected() {
        let above_input = EnergyBalance {
            input_power_w: 100.0,
            net_useful_output_power_w: 100.0,
            auxiliary_power_w: 1.0,
            rejected_or_loss_power_w: 0.0,
        };
        assert_eq!(
            above_input.evaluate(1e-6),
            Err(EnergyBalanceError::OutputExceedsInput)
        );

        let negative = EnergyBalance {
            input_power_w: 100.0,
            net_useful_output_power_w: 90.0,
            auxiliary_power_w: -1.0,
            rejected_or_loss_power_w: 11.0,
        };
        assert_eq!(
            negative.evaluate(1e-6),
            Err(EnergyBalanceError::NegativePowerFlow)
        );
    }

    #[test]
    fn non_finite_and_non_positive_inputs_are_rejected() {
        let nan = EnergyBalance {
            input_power_w: f64::NAN,
            net_useful_output_power_w: 1.0,
            auxiliary_power_w: 0.0,
            rejected_or_loss_power_w: 0.0,
        };
        assert_eq!(
            nan.evaluate(1e-6),
            Err(EnergyBalanceError::NonFiniteInput)
        );

        let zero = EnergyBalance {
            input_power_w: 0.0,
            net_useful_output_power_w: 0.0,
            auxiliary_power_w: 0.0,
            rejected_or_loss_power_w: 0.0,
        };
        assert_eq!(
            zero.evaluate(1e-6),
            Err(EnergyBalanceError::NonPositiveInputPower)
        );
        assert_eq!(
            zero.evaluate(f64::NAN),
            Err(EnergyBalanceError::NonFiniteInput)
        );
        let valid_flows = EnergyBalance {
            input_power_w: 100.0,
            net_useful_output_power_w: 0.0,
            auxiliary_power_w: 0.0,
            rejected_or_loss_power_w: 100.0,
        };
        assert_eq!(
            valid_flows.evaluate(-1e-6),
            Err(EnergyBalanceError::InvalidTolerance)
        );
        assert_eq!(
            valid_flows.evaluate(1.0),
            Err(EnergyBalanceError::InvalidTolerance)
        );
    }
}
