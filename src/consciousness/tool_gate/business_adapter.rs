// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Business-facing adapter for the conscious tool gate.
//!
//! The ordinary tool gate answers an internal cognitive/safety question. Business
//! execution additionally requires independently owned institutional authority,
//! policy, coordination, and runtime-precondition gates. This module deliberately
//! avoids exposing `GateDecision::Allowed` as business authorization.

use super::types::{GateDecision, GateResult, RiskLevel};

/// Cognitive disposition suitable for composition with external business gates.
///
/// `AcceptableForFurtherGating` means only that Symthaea's internal cognitive
/// gate does not currently block the candidate. It never means institutionally
/// authorized, legally permitted, policy-compliant, or executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusinessCognitiveClearance {
    /// Candidate may proceed to independent policy/authority/runtime gates.
    AcceptableForFurtherGating,
    /// Candidate is blocked until epistemic confidence is improved.
    NeedsMoreEvidence,
    /// Candidate is blocked pending stronger cognitive state or explicit review.
    NeedsReview,
}

/// Stable business-facing projection of a `GateResult`.
///
/// This struct intentionally carries no capability token, authority lease,
/// approval identity, execution permission, or mutation handle.
#[derive(Debug, Clone, PartialEq)]
pub struct BusinessCognitiveAssessment {
    pub clearance: BusinessCognitiveClearance,
    pub risk_level: RiskLevel,
    pub required_phi: f64,
    pub actual_phi_eff: f64,
    pub required_confidence: f64,
    pub actual_confidence: f64,
}

impl BusinessCognitiveAssessment {
    /// Convert an internal tool-gate result into a non-authorizing business view.
    pub fn from_gate_result(result: &GateResult) -> Self {
        let clearance = match result.decision {
            GateDecision::Allowed => BusinessCognitiveClearance::AcceptableForFurtherGating,
            GateDecision::InsufficientConfidence { .. } => {
                BusinessCognitiveClearance::NeedsMoreEvidence
            }
            GateDecision::InsufficientPhi { .. } => BusinessCognitiveClearance::NeedsReview,
        };

        Self {
            clearance,
            risk_level: result.risk_level,
            required_phi: result.required_phi,
            actual_phi_eff: result.actual_phi_eff,
            required_confidence: result.required_confidence,
            actual_confidence: result.actual_confidence,
        }
    }

    /// Business actions always require authority from outside the cognitive gate.
    pub const fn requires_external_authority(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::tool_gate::FallbackStrategy;

    fn result(decision: GateDecision) -> GateResult {
        GateResult {
            decision,
            required_phi: 0.5,
            required_confidence: 0.4,
            actual_phi_eff: 0.7,
            actual_confidence: 0.8,
            risk_level: RiskLevel::Elevated,
            fallback: Some(FallbackStrategy::NoFallback {
                reason: "fixture".into(),
            }),
        }
    }

    #[test]
    fn allowed_gate_is_only_acceptable_for_further_gating() {
        let assessment = BusinessCognitiveAssessment::from_gate_result(&result(GateDecision::Allowed));
        assert_eq!(
            assessment.clearance,
            BusinessCognitiveClearance::AcceptableForFurtherGating
        );
        assert!(assessment.requires_external_authority());
    }

    #[test]
    fn insufficient_confidence_requests_more_evidence() {
        let assessment = BusinessCognitiveAssessment::from_gate_result(&result(
            GateDecision::InsufficientConfidence {
                current: 0.2,
                required: 0.4,
            },
        ));
        assert_eq!(
            assessment.clearance,
            BusinessCognitiveClearance::NeedsMoreEvidence
        );
        assert!(assessment.requires_external_authority());
    }

    #[test]
    fn insufficient_phi_never_becomes_business_authority() {
        let assessment = BusinessCognitiveAssessment::from_gate_result(&result(
            GateDecision::InsufficientPhi {
                current: 0.2,
                required: 0.5,
            },
        ));
        assert_eq!(assessment.clearance, BusinessCognitiveClearance::NeedsReview);
        assert!(assessment.requires_external_authority());
    }
}
