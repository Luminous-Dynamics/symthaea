// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed calibration-domain binding for MAGI / calibrated decision paths.
//!
//! CAL-004A keeps two concepts separate:
//!
//! ```text
//! declared typed domain   -> authority-eligible calibration cohort
//! inferred legacy domain  -> compatibility/modeling only
//! ```
//!
//! A free-form action string may still provide a useful modeling fallback, but
//! it must remain visibly distinct from an explicitly declared domain. Likewise,
//! a declared domain is not automatically calibrated: the matching declared
//! cohort must contain enough resolved observations for ECE to be measured.

use super::magi_integration::WorldGroundedSelfModel;
use super::world_prediction::{PredictionDomain, WorldActionContext};

/// How the prediction/calibration domain for an action was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationDomainSource {
    /// The action explicitly declared the exact typed domain.
    Declared,
    /// Compatibility-only inference from the free-form action type.
    InferredCompatibility,
}

/// Domain selected for modeling together with the provenance of that selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundPredictionDomain {
    pub domain: PredictionDomain,
    pub source: CalibrationDomainSource,
}

impl BoundPredictionDomain {
    /// Bind an action to a domain without erasing whether the binding was typed
    /// or inferred from the historical action-string heuristic.
    pub fn from_action(action: &WorldActionContext) -> Self {
        match action.declared_prediction_domain() {
            Some(domain) => Self {
                domain,
                source: CalibrationDomainSource::Declared,
            },
            None => Self {
                domain: action.prediction_domain_for_modeling(),
                source: CalibrationDomainSource::InferredCompatibility,
            },
        }
    }

    /// Only explicit typed bindings are eligible to select an authority-bearing
    /// calibration cohort.
    pub const fn authority_eligible(self) -> bool {
        matches!(self.source, CalibrationDomainSource::Declared)
    }
}

/// Evidence state for the exact declared domain of an action.
///
/// This is intentionally not a boolean `is_calibrated` flag. An action may be
/// unbound, may have a declared domain with too little evidence, or may have an
/// actually measured calibration cohort. Consumers must handle those states
/// separately.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeclaredCalibrationEvidence {
    /// No explicit typed domain was supplied. Heuristic inference is not
    /// authority-bearing calibration evidence.
    Unbound,

    /// A typed domain exists, but the explicitly-declared cohort has not yet
    /// reached the calibration measurement threshold.
    Insufficient {
        domain: PredictionDomain,
        sample_count: usize,
        minimum_required: usize,
    },

    /// ECE was actually measured from the explicitly-declared matching cohort.
    Measured {
        domain: PredictionDomain,
        sample_count: usize,
        accuracy: f64,
        ece: f64,
    },
}

impl DeclaredCalibrationEvidence {
    pub const fn is_measured(self) -> bool {
        matches!(self, Self::Measured { .. })
    }

    pub const fn domain(self) -> Option<PredictionDomain> {
        match self {
            Self::Unbound => None,
            Self::Insufficient { domain, .. } | Self::Measured { domain, .. } => Some(domain),
        }
    }
}

impl WorldGroundedSelfModel {
    /// Return the domain used for compatibility/modeling together with binding
    /// provenance. An explicit typed domain always wins over string inference.
    pub fn calibration_domain_binding(
        &self,
        action: &WorldActionContext,
    ) -> BoundPredictionDomain {
        BoundPredictionDomain::from_action(action)
    }

    /// Project the action onto the session-local, explicitly-declared
    /// calibration cohort that CAL-002A made authority-eligible.
    ///
    /// This deliberately does not read the ordinary per-domain aggregate for
    /// authority decisions because that aggregate may include legacy inferred
    /// records. It also fails closed after restart because detailed declared
    /// cohort history is intentionally not yet persisted.
    pub fn declared_calibration_evidence(
        &self,
        action: &WorldActionContext,
    ) -> DeclaredCalibrationEvidence {
        let Some(domain) = action.declared_prediction_domain() else {
            return DeclaredCalibrationEvidence::Unbound;
        };

        let cohort = self.calibration().declared_domain_calibration(domain);
        let minimum_required = self.calibration().config().min_predictions_for_ece;

        match cohort.ece {
            Some(ece) if cohort.sample_count >= minimum_required => {
                DeclaredCalibrationEvidence::Measured {
                    domain,
                    sample_count: cohort.sample_count,
                    accuracy: cohort.accuracy,
                    ece,
                }
            }
            _ => DeclaredCalibrationEvidence::Insufficient {
                domain,
                sample_count: cohort.sample_count,
                minimum_required,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::recursive_improvement::{
        OutcomeCategory, RiskTier, WorldGroundedConfig,
    };

    fn config_with_small_calibration_gate() -> WorldGroundedConfig {
        let mut config = WorldGroundedConfig::default();
        config.calibration.min_predictions_for_ece = 4;
        config.calibration.rolling_window = 8;
        config
    }

    fn resolve(
        model: &mut WorldGroundedSelfModel,
        action: WorldActionContext,
        confidence: f64,
        correct: bool,
    ) {
        let prediction = model.predict(
            "typed-domain calibration regression",
            OutcomeCategory::Success,
            confidence,
            action,
        );
        let observed = if correct {
            OutcomeCategory::Success
        } else {
            OutcomeCategory::SafeFailure
        };
        assert!(model
            .resolve_prediction(&prediction.id, observed, 1.0)
            .is_some());
    }

    #[test]
    fn explicit_domain_wins_without_erasing_binding_provenance() {
        let model = WorldGroundedSelfModel::with_defaults();
        let action = WorldActionContext::new("compile", "invoke external tool")
            .with_prediction_domain(PredictionDomain::ToolUse);

        let binding = model.calibration_domain_binding(&action);
        assert_eq!(binding.domain, PredictionDomain::ToolUse);
        assert_eq!(binding.source, CalibrationDomainSource::Declared);
        assert!(binding.authority_eligible());
    }

    #[test]
    fn legacy_inference_remains_visible_and_non_authorizing() {
        let model = WorldGroundedSelfModel::with_defaults();
        let action = WorldActionContext::new("compile", "compile source");

        let binding = model.calibration_domain_binding(&action);
        assert_eq!(binding.domain, PredictionDomain::CodeExecution);
        assert_eq!(
            binding.source,
            CalibrationDomainSource::InferredCompatibility
        );
        assert!(!binding.authority_eligible());
        assert_eq!(
            model.declared_calibration_evidence(&action),
            DeclaredCalibrationEvidence::Unbound
        );
    }

    #[test]
    fn inferred_history_cannot_satisfy_declared_cohort_measurement() {
        let mut model = WorldGroundedSelfModel::new(config_with_small_calibration_gate());

        for _ in 0..4 {
            resolve(
                &mut model,
                WorldActionContext::new("compile", "legacy inferred compile")
                    .with_risk_tier(RiskTier::Observation),
                0.75,
                true,
            );
        }

        let declared_action = WorldActionContext::new("anything", "typed code action")
            .with_risk_tier(RiskTier::Observation)
            .with_prediction_domain(PredictionDomain::CodeExecution);

        assert_eq!(
            model.declared_calibration_evidence(&declared_action),
            DeclaredCalibrationEvidence::Insufficient {
                domain: PredictionDomain::CodeExecution,
                sample_count: 0,
                minimum_required: 4,
            }
        );
    }

    #[test]
    fn matching_declared_history_becomes_measured_only_after_threshold() {
        let mut model = WorldGroundedSelfModel::new(config_with_small_calibration_gate());

        let action = || {
            WorldActionContext::new("compile", "typed tool action")
                .with_risk_tier(RiskTier::Observation)
                .with_prediction_domain(PredictionDomain::ToolUse)
        };

        for correct in [true, true, true] {
            resolve(&mut model, action(), 0.75, correct);
        }
        assert_eq!(
            model.declared_calibration_evidence(&action()),
            DeclaredCalibrationEvidence::Insufficient {
                domain: PredictionDomain::ToolUse,
                sample_count: 3,
                minimum_required: 4,
            }
        );

        resolve(&mut model, action(), 0.75, false);
        match model.declared_calibration_evidence(&action()) {
            DeclaredCalibrationEvidence::Measured {
                domain,
                sample_count,
                accuracy,
                ece,
            } => {
                assert_eq!(domain, PredictionDomain::ToolUse);
                assert_eq!(sample_count, 4);
                assert!((accuracy - 0.75).abs() < 1e-12);
                assert!(ece.is_finite());
            }
            other => panic!("expected measured declared cohort, got {other:?}"),
        }
    }
}
