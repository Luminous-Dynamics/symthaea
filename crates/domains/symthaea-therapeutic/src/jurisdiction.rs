// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deployment-supplied jurisdiction policy for crisis resources and reporting.
//!
//! The crate does not infer legal duties from location and does not silently
//! fall back to US resources. A deployment must provide a reviewed, current
//! policy before localized crisis instructions or reporting workflows are used.

use crate::safety::CrisisType;
use serde::{Deserialize, Serialize};

/// Validated jurisdiction identifier such as `US-TX` or `ZA-GP`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JurisdictionId(String);

impl JurisdictionId {
    pub fn new(value: impl Into<String>) -> Result<Self, JurisdictionPolicyError> {
        let value = value.into();
        let valid = !value.is_empty()
            && value.len() <= 24
            && value
                .bytes()
                .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'-');
        if !valid {
            return Err(JurisdictionPolicyError::InvalidJurisdictionId);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Crisis-resource category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CrisisResourceKind {
    EmergencyServices,
    CrisisLine,
    TextService,
    DomesticViolenceService,
    ChildProtectionService,
    SubstanceEmergencyService,
    LocalMentalHealthService,
}

/// A reviewed resource supplied by the deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrisisResource {
    pub kind: CrisisResourceKind,
    pub label: String,
    pub instructions: String,
    pub languages: Vec<String>,
    pub verified_at_unix: u64,
    pub review_due_unix: u64,
    pub active: bool,
}

/// Action specified by deployment legal and clinical governance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReportingAction {
    NoAutomatedReport,
    SeekQualifiedHumanReview,
    ImmediateQualifiedHumanReview,
}

/// Reviewed rule for a potentially reportable crisis category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MandatoryReportingRule {
    pub crisis_type: CrisisType,
    pub action: ReportingAction,
    pub legal_basis_reference: String,
    pub review_due_unix: u64,
}

/// Complete deployment jurisdiction policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JurisdictionPolicy {
    pub jurisdiction: JurisdictionId,
    pub policy_version: String,
    pub reviewed_at_unix: u64,
    pub review_due_unix: u64,
    pub emergency_preamble: String,
    pub resources: Vec<CrisisResource>,
    pub reporting_rules: Vec<MandatoryReportingRule>,
}

/// Fail-closed policy validation errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum JurisdictionPolicyError {
    InvalidJurisdictionId,
    MissingPolicyVersion,
    MissingEmergencyPreamble,
    PolicyExpired,
    NoActiveEmergencyResource,
    ResourceIncomplete,
    ResourceExpired,
    ReportingRuleIncomplete,
    ReportingRuleExpired,
}

impl JurisdictionPolicy {
    /// Validate that the policy is complete and current at the supplied Unix time.
    pub fn validate(&self, now_unix: u64) -> Result<(), JurisdictionPolicyError> {
        if self.policy_version.trim().is_empty() {
            return Err(JurisdictionPolicyError::MissingPolicyVersion);
        }
        if self.emergency_preamble.trim().is_empty() {
            return Err(JurisdictionPolicyError::MissingEmergencyPreamble);
        }
        if self.review_due_unix <= now_unix {
            return Err(JurisdictionPolicyError::PolicyExpired);
        }

        let mut has_emergency = false;
        for resource in self.resources.iter().filter(|resource| resource.active) {
            if resource.label.trim().is_empty() || resource.instructions.trim().is_empty() {
                return Err(JurisdictionPolicyError::ResourceIncomplete);
            }
            if resource.review_due_unix <= now_unix {
                return Err(JurisdictionPolicyError::ResourceExpired);
            }
            if resource.kind == CrisisResourceKind::EmergencyServices {
                has_emergency = true;
            }
        }
        if !has_emergency {
            return Err(JurisdictionPolicyError::NoActiveEmergencyResource);
        }

        for rule in &self.reporting_rules {
            if rule.legal_basis_reference.trim().is_empty() {
                return Err(JurisdictionPolicyError::ReportingRuleIncomplete);
            }
            if rule.review_due_unix <= now_unix {
                return Err(JurisdictionPolicyError::ReportingRuleExpired);
            }
        }
        Ok(())
    }

    /// Render only reviewed, active crisis resources after validating the policy.
    pub fn crisis_resource_lines(
        &self,
        now_unix: u64,
    ) -> Result<Vec<String>, JurisdictionPolicyError> {
        self.validate(now_unix)?;
        let mut lines = Vec::with_capacity(self.resources.len() + 1);
        lines.push(self.emergency_preamble.clone());
        lines.extend(
            self.resources
                .iter()
                .filter(|resource| resource.active)
                .map(|resource| format!("{}: {}", resource.label, resource.instructions)),
        );
        Ok(lines)
    }

    /// Return the reviewed reporting action, never an inferred legal conclusion.
    pub fn reporting_action(
        &self,
        crisis_type: CrisisType,
        now_unix: u64,
    ) -> Result<ReportingAction, JurisdictionPolicyError> {
        self.validate(now_unix)?;
        Ok(self
            .reporting_rules
            .iter()
            .find(|rule| rule.crisis_type == crisis_type)
            .map_or(ReportingAction::SeekQualifiedHumanReview, |rule| {
                rule.action
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_policy() -> JurisdictionPolicy {
        JurisdictionPolicy {
            jurisdiction: JurisdictionId::new("ZA-GP").unwrap(),
            policy_version: "za-gp-2026-01".to_string(),
            reviewed_at_unix: 10,
            review_due_unix: 1_000,
            emergency_preamble: "Use verified local emergency resources.".to_string(),
            resources: vec![CrisisResource {
                kind: CrisisResourceKind::EmergencyServices,
                label: "Emergency services".to_string(),
                instructions: "Call the deployment-configured emergency number".to_string(),
                languages: vec!["en".to_string()],
                verified_at_unix: 10,
                review_due_unix: 1_000,
                active: true,
            }],
            reporting_rules: vec![MandatoryReportingRule {
                crisis_type: CrisisType::ChildAbuse,
                action: ReportingAction::ImmediateQualifiedHumanReview,
                legal_basis_reference: "deployment-legal-register-1".to_string(),
                review_due_unix: 1_000,
            }],
        }
    }

    #[test]
    fn current_reviewed_policy_is_usable() {
        let policy = valid_policy();
        assert!(policy.validate(100).is_ok());
        assert_eq!(policy.crisis_resource_lines(100).unwrap().len(), 2);
    }

    #[test]
    fn expired_policy_fails_closed() {
        let policy = valid_policy();
        assert_eq!(
            policy.validate(1_000),
            Err(JurisdictionPolicyError::PolicyExpired)
        );
    }

    #[test]
    fn reporting_defaults_to_human_review_not_automatic_reporting() {
        let policy = valid_policy();
        assert_eq!(
            policy.reporting_action(CrisisType::Psychosis, 100).unwrap(),
            ReportingAction::SeekQualifiedHumanReview
        );
    }
}
