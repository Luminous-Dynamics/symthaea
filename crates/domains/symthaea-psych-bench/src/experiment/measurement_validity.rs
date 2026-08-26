// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Construct-validity guard for SYM-ARCH-002A6 acquisition criteria.
//!
//! A learning threshold is not claim-bearing if a known chance/majority/shortcut
//! reference can already meet it. This audit makes that boundary explicit while
//! leaving inferential uncertainty to A2 and shortcut discovery to A4.

use crate::experiment_measurement::LearningCriterion;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcquisitionCriterionStatus {
    Admissible,
    ReferenceConfounded,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcquisitionCriterionAudit {
    pub criterion: LearningCriterion,
    /// Frozen accuracy ceiling of the preregistered simple reference/control.
    pub reference_accuracy_ceiling: f64,
    /// Minimum practical accuracy excess required above the reference ceiling.
    pub minimum_excess_over_reference: f64,
    /// Integer number correct actually needed in one rolling window.
    pub required_correct_per_window: usize,
    /// Finite-window accuracy represented by `required_correct_per_window`.
    pub effective_accuracy_threshold: f64,
    pub effective_excess_over_reference: f64,
    pub status: AcquisitionCriterionStatus,
    pub qualifiers: Vec<String>,
}

fn validate_probability(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(format!("{name} must be finite in [0,1]"));
    }
    Ok(())
}

/// Audit a frozen acquisition criterion against a preregistered reference ceiling.
///
/// This is a construct-validity guard, not a significance test. An admissible
/// result still requires A2 uncertainty/power analysis and A4 benchmark controls.
pub fn audit_acquisition_criterion(
    criterion: LearningCriterion,
    reference_accuracy_ceiling: f64,
    minimum_excess_over_reference: f64,
) -> Result<AcquisitionCriterionAudit, String> {
    criterion.validate()?;
    validate_probability("reference_accuracy_ceiling", reference_accuracy_ceiling)?;
    validate_probability(
        "minimum_excess_over_reference",
        minimum_excess_over_reference,
    )?;

    let raw_required = criterion.accuracy_threshold * criterion.window_size as f64;
    let required_correct_per_window = (raw_required - 1e-12)
        .ceil()
        .clamp(0.0, criterion.window_size as f64) as usize;
    let effective_accuracy_threshold =
        required_correct_per_window as f64 / criterion.window_size as f64;
    let effective_excess_over_reference =
        effective_accuracy_threshold - reference_accuracy_ceiling;

    let mut qualifiers = Vec::new();
    if effective_accuracy_threshold <= reference_accuracy_ceiling + 1e-12 {
        qualifiers.push(format!(
            "effective criterion {:.6} does not exceed frozen reference ceiling {:.6}",
            effective_accuracy_threshold, reference_accuracy_ceiling
        ));
    }
    if effective_excess_over_reference + 1e-12 < minimum_excess_over_reference {
        qualifiers.push(format!(
            "effective excess {:.6} is below required practical margin {:.6}",
            effective_excess_over_reference, minimum_excess_over_reference
        ));
    }

    Ok(AcquisitionCriterionAudit {
        criterion,
        reference_accuracy_ceiling,
        minimum_excess_over_reference,
        required_correct_per_window,
        effective_accuracy_threshold,
        effective_excess_over_reference,
        status: if qualifiers.is_empty() {
            AcquisitionCriterionStatus::Admissible
        } else {
            AcquisitionCriterionStatus::ReferenceConfounded
        },
        qualifiers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn criterion_above_reference_and_margin_is_admissible() {
        let audit = audit_acquisition_criterion(
            LearningCriterion {
                window_size: 20,
                accuracy_threshold: 0.80,
                consecutive_windows: 2,
            },
            0.50,
            0.10,
        )
        .unwrap();
        assert_eq!(audit.required_correct_per_window, 16);
        assert!((audit.effective_accuracy_threshold - 0.80).abs() < 1e-12);
        assert_eq!(audit.status, AcquisitionCriterionStatus::Admissible);
        assert!(audit.qualifiers.is_empty());
    }

    #[test]
    fn shortcut_level_threshold_is_reference_confounded() {
        let audit = audit_acquisition_criterion(
            LearningCriterion {
                window_size: 20,
                accuracy_threshold: 0.60,
                consecutive_windows: 3,
            },
            0.65,
            0.05,
        )
        .unwrap();
        assert_eq!(audit.status, AcquisitionCriterionStatus::ReferenceConfounded);
        assert!(!audit.qualifiers.is_empty());
    }

    #[test]
    fn finite_window_resolution_is_reported_not_hidden() {
        let audit = audit_acquisition_criterion(
            LearningCriterion {
                window_size: 4,
                accuracy_threshold: 0.81,
                consecutive_windows: 1,
            },
            0.50,
            0.10,
        )
        .unwrap();
        assert_eq!(audit.required_correct_per_window, 4);
        assert_eq!(audit.effective_accuracy_threshold, 1.0);
        assert_eq!(audit.status, AcquisitionCriterionStatus::Admissible);
    }

    #[test]
    fn practical_margin_can_fail_even_when_reference_is_exceeded() {
        let audit = audit_acquisition_criterion(
            LearningCriterion {
                window_size: 20,
                accuracy_threshold: 0.70,
                consecutive_windows: 2,
            },
            0.65,
            0.10,
        )
        .unwrap();
        assert_eq!(audit.status, AcquisitionCriterionStatus::ReferenceConfounded);
        assert!(audit
            .qualifiers
            .iter()
            .any(|qualifier| qualifier.contains("practical margin")));
    }

    #[test]
    fn audit_rejects_invalid_reference_contract() {
        let criterion = LearningCriterion {
            window_size: 20,
            accuracy_threshold: 0.80,
            consecutive_windows: 2,
        };
        assert!(audit_acquisition_criterion(criterion, 1.1, 0.05).is_err());
        assert!(audit_acquisition_criterion(criterion, 0.5, -0.01).is_err());
    }
}
