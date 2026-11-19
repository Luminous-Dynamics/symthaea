//! Non-compensatory qualification of completed Aqua bathing services.
//!
//! This module is a decision-contract layer. It checks claim scope, evidence completeness,
//! authority, protocol identity and matched conditions. It does not calculate physical
//! performance, water/energy totals, statistical confidence intervals, hygiene, accessibility,
//! safety or certification. Those decisions remain with the named independent/owner assessors.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_text(field: &'static str, value: &str) -> Result<(), ServiceQualificationError> {
    if value.trim().is_empty() {
        return Err(ServiceQualificationError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), ServiceQualificationError> {
    if !is_sha256(value) {
        return Err(ServiceQualificationError::InvalidDigest { field });
    }
    Ok(())
}

/// Independent assessment domain required by a declared product claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualificationDomain {
    MeasurementValidity,
    TaskCompletionAndComfort,
    FixtureMethodPerformance,
    ThermalStability,
    WaterQuality,
    HygieneAndExposure,
    LocalSafety,
    AccessibilityAndUsability,
    WaterSavingsPerCompletedTask,
    EnergySavingsPerCompletedTask,
    Serviceability,
    CleaningBurden,
    ContinuityAndRecovery,
}

/// The assessor / canonical owner permitted to issue evidence for a domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QualificationAuthority {
    MycelixMetrology,
    IndependentTaskAssessor,
    IndependentTestLaboratory,
    MycelixWaterQuality,
    IndependentHygieneExposureAssessor,
    IndependentDeviceAuthority,
    IndependentAccessibilityAssessor,
    SymtropyResourceClosure,
    IndependentServiceabilityAssessor,
}

impl QualificationDomain {
    pub fn expected_authority(self) -> QualificationAuthority {
        match self {
            Self::MeasurementValidity => QualificationAuthority::MycelixMetrology,
            Self::TaskCompletionAndComfort => QualificationAuthority::IndependentTaskAssessor,
            Self::FixtureMethodPerformance | Self::ThermalStability => {
                QualificationAuthority::IndependentTestLaboratory
            }
            Self::WaterQuality => QualificationAuthority::MycelixWaterQuality,
            Self::HygieneAndExposure => {
                QualificationAuthority::IndependentHygieneExposureAssessor
            }
            Self::LocalSafety | Self::ContinuityAndRecovery => {
                QualificationAuthority::IndependentDeviceAuthority
            }
            Self::AccessibilityAndUsability => {
                QualificationAuthority::IndependentAccessibilityAssessor
            }
            Self::WaterSavingsPerCompletedTask | Self::EnergySavingsPerCompletedTask => {
                QualificationAuthority::SymtropyResourceClosure
            }
            Self::Serviceability | Self::CleaningBurden => {
                QualificationAuthority::IndependentServiceabilityAssessor
            }
        }
    }
}

/// Currentness is asserted by the canonical evidence owner after its own
/// issuer-authentication, revocation and validity checks. Aqua preserves that state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationEvidenceStatus {
    Current,
    Stale,
    Revoked,
    Ambiguous,
}

/// A claim profile fixes which independent domains must be assessed.
/// More ambitious claims require more evidence; profiles cannot be weakened at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationClaim {
    FunctionalService,
    AccessibleService,
    WaterEfficiency,
    WaterAndEnergyEfficiency,
    LifecycleEfficiency,
    ResilientService,
    FullPlatform,
}

const FUNCTIONAL_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
];

const ACCESSIBLE_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
    QualificationDomain::AccessibilityAndUsability,
];

const WATER_EFFICIENCY_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
    QualificationDomain::WaterSavingsPerCompletedTask,
];

const WATER_AND_ENERGY_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
    QualificationDomain::WaterSavingsPerCompletedTask,
    QualificationDomain::EnergySavingsPerCompletedTask,
];

const LIFECYCLE_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
    QualificationDomain::AccessibilityAndUsability,
    QualificationDomain::WaterSavingsPerCompletedTask,
    QualificationDomain::EnergySavingsPerCompletedTask,
    QualificationDomain::Serviceability,
    QualificationDomain::CleaningBurden,
];

const RESILIENT_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
    QualificationDomain::ContinuityAndRecovery,
];

const FULL_PLATFORM_DOMAINS: &[QualificationDomain] = &[
    QualificationDomain::MeasurementValidity,
    QualificationDomain::TaskCompletionAndComfort,
    QualificationDomain::FixtureMethodPerformance,
    QualificationDomain::ThermalStability,
    QualificationDomain::WaterQuality,
    QualificationDomain::HygieneAndExposure,
    QualificationDomain::LocalSafety,
    QualificationDomain::AccessibilityAndUsability,
    QualificationDomain::WaterSavingsPerCompletedTask,
    QualificationDomain::EnergySavingsPerCompletedTask,
    QualificationDomain::Serviceability,
    QualificationDomain::CleaningBurden,
    QualificationDomain::ContinuityAndRecovery,
];

impl QualificationClaim {
    pub fn required_domains(self) -> &'static [QualificationDomain] {
        match self {
            Self::FunctionalService => FUNCTIONAL_DOMAINS,
            Self::AccessibleService => ACCESSIBLE_DOMAINS,
            Self::WaterEfficiency => WATER_EFFICIENCY_DOMAINS,
            Self::WaterAndEnergyEfficiency => WATER_AND_ENERGY_DOMAINS,
            Self::LifecycleEfficiency => LIFECYCLE_DOMAINS,
            Self::ResilientService => RESILIENT_DOMAINS,
            Self::FullPlatform => FULL_PLATFORM_DOMAINS,
        }
    }
}

/// One frozen specialist method and its method-specific conditions within a service trial.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationMethodBinding {
    pub domain: QualificationDomain,
    pub method_sha256: String,
    pub conditions_sha256: String,
}

/// Claim identity and the frozen trial context against which every receipt is bound.
/// Each domain may use a different specialist method and method-specific conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceQualificationProfile {
    pub profile_id: String,
    pub claim: QualificationClaim,
    pub protocol_sha256: String,
    pub conditions_sha256: String,
    pub method_bindings: Vec<QualificationMethodBinding>,
}

impl ServiceQualificationProfile {
    fn method_binding(&self, domain: QualificationDomain) -> Option<&QualificationMethodBinding> {
        self.method_bindings
            .iter()
            .find(|binding| binding.domain == domain)
    }

    pub fn validate(&self) -> Result<(), ServiceQualificationError> {
        require_text("profile_id", &self.profile_id)?;
        require_digest("profile.protocol_sha256", &self.protocol_sha256)?;
        require_digest("profile.conditions_sha256", &self.conditions_sha256)?;

        let required = self.claim.required_domains();
        let mut domains = HashSet::with_capacity(required.len());
        for domain in required {
            if !domains.insert(*domain) {
                return Err(ServiceQualificationError::DuplicateRequiredDomain {
                    domain: *domain,
                });
            }
        }

        let mut bound = HashSet::with_capacity(required.len());
        for binding in &self.method_bindings {
            if !required.contains(&binding.domain) {
                return Err(ServiceQualificationError::UnexpectedMethodBinding {
                    domain: binding.domain,
                });
            }
            if !bound.insert(binding.domain) {
                return Err(ServiceQualificationError::DuplicateMethodBinding {
                    domain: binding.domain,
                });
            }
            require_digest("method_binding.method_sha256", &binding.method_sha256)?;
            require_digest(
                "method_binding.conditions_sha256",
                &binding.conditions_sha256,
            )?;
        }

        for domain in required {
            if !bound.contains(domain) {
                return Err(ServiceQualificationError::MissingMethodBinding {
                    domain: *domain,
                });
            }
        }
        Ok(())
    }
}

/// A source-pinned assessment receipt. The receipt binds its method and test conditions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationEvidenceRef {
    pub authority: QualificationAuthority,
    /// Exact Aqua run identity to which this assessment was issued.
    pub run_id: String,
    /// Stable principal identity returned by the canonical authority registry.
    pub issuer_principal_id: String,
    /// Receipt from the canonical owner that it authenticated and checked this evidence.
    pub authority_verification_receipt_sha256: String,
    pub status: QualificationEvidenceStatus,
    pub record_id: String,
    pub evidence_sha256: String,
    pub method_sha256: String,
    pub method_conditions_sha256: String,
    pub trial_conditions_sha256: String,
}

impl QualificationEvidenceRef {
    fn validate(
        &self,
        domain: QualificationDomain,
        profile: &ServiceQualificationProfile,
        expected_run_id: &str,
    ) -> Result<(), ServiceQualificationError> {
        require_text("assessment.run_id", &self.run_id)?;
        if self.run_id != expected_run_id {
            return Err(ServiceQualificationError::EvidenceRunMismatch { domain });
        }
        require_text("assessment.issuer_principal_id", &self.issuer_principal_id)?;
        require_digest(
            "assessment.authority_verification_receipt_sha256",
            &self.authority_verification_receipt_sha256,
        )?;
        require_text("assessment.record_id", &self.record_id)?;
        require_digest("assessment.evidence_sha256", &self.evidence_sha256)?;
        require_digest("assessment.method_sha256", &self.method_sha256)?;
        require_digest(
            "assessment.method_conditions_sha256",
            &self.method_conditions_sha256,
        )?;
        require_digest(
            "assessment.trial_conditions_sha256",
            &self.trial_conditions_sha256,
        )?;

        let expected = domain.expected_authority();
        if self.authority != expected {
            return Err(ServiceQualificationError::WrongAuthority {
                domain,
                expected,
                actual: self.authority,
            });
        }
        if self.status != QualificationEvidenceStatus::Current {
            return Err(ServiceQualificationError::EvidenceNotCurrent {
                domain,
                status: self.status,
            });
        }
        let binding = profile
            .method_binding(domain)
            .ok_or(ServiceQualificationError::MissingMethodBinding { domain })?;
        if self.method_sha256 != binding.method_sha256 {
            return Err(ServiceQualificationError::EvidenceMethodMismatch { domain });
        }
        if self.method_conditions_sha256 != binding.conditions_sha256 {
            return Err(ServiceQualificationError::EvidenceConditionsMismatch { domain });
        }
        if self.trial_conditions_sha256 != profile.conditions_sha256 {
            return Err(ServiceQualificationError::EvidenceTrialConditionsMismatch { domain });
        }
        Ok(())
    }
}

/// These are assessor decisions, not locally calculated scores.
/// Indeterminate is never promoted to Pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssessmentOutcome {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainAssessment {
    pub domain: QualificationDomain,
    pub outcome: AssessmentOutcome,
    pub evidence: QualificationEvidenceRef,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceQualificationRecord {
    pub run_id: String,
    /// Canonical digest of market variant, hardware, installation, cleaning protocol and firmware.
    pub device_configuration_sha256: String,
    pub profile_id: String,
    pub protocol_sha256: String,
    pub conditions_sha256: String,
    pub assessments: Vec<DomainAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceQualificationError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    DuplicateRequiredDomain { domain: QualificationDomain },
    MissingMethodBinding { domain: QualificationDomain },
    UnexpectedMethodBinding { domain: QualificationDomain },
    DuplicateMethodBinding { domain: QualificationDomain },
    ProfileMismatch,
    ProtocolMismatch,
    ConditionsMismatch,
    MissingDomain { domain: QualificationDomain },
    UnexpectedDomain { domain: QualificationDomain },
    DuplicateDomain { domain: QualificationDomain },
    DuplicateEvidenceRecord { record_id: String },
    EvidenceNotCurrent {
        domain: QualificationDomain,
        status: QualificationEvidenceStatus,
    },
    EvidenceRunMismatch { domain: QualificationDomain },
    WrongAuthority {
        domain: QualificationDomain,
        expected: QualificationAuthority,
        actual: QualificationAuthority,
    },
    EvidenceMethodMismatch { domain: QualificationDomain },
    EvidenceConditionsMismatch { domain: QualificationDomain },
    EvidenceTrialConditionsMismatch { domain: QualificationDomain },
    SameRunCompared,
}

impl Display for ServiceQualificationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => {
                write!(f, "missing service-qualification field: {field}")
            }
            Self::InvalidDigest { field } => {
                write!(f, "invalid service-qualification SHA-256: {field}")
            }
            Self::DuplicateRequiredDomain { domain } => {
                write!(f, "claim profile repeats required domain: {domain:?}")
            }
            Self::MissingMethodBinding { domain } => {
                write!(f, "claim profile has no method binding for {domain:?}")
            }
            Self::UnexpectedMethodBinding { domain } => {
                write!(f, "claim profile has an unrequested method binding for {domain:?}")
            }
            Self::DuplicateMethodBinding { domain } => {
                write!(f, "claim profile has duplicate method bindings for {domain:?}")
            }
            Self::ProfileMismatch => write!(f, "qualification record uses a different claim profile"),
            Self::ProtocolMismatch => write!(f, "qualification record uses a different protocol"),
            Self::ConditionsMismatch => {
                write!(f, "qualification record uses different matched conditions")
            }
            Self::MissingDomain { domain } => {
                write!(f, "required independent assessment is missing: {domain:?}")
            }
            Self::UnexpectedDomain { domain } => {
                write!(f, "assessment is outside the declared claim profile: {domain:?}")
            }
            Self::DuplicateDomain { domain } => {
                write!(f, "claim domain has more than one assessment: {domain:?}")
            }
            Self::DuplicateEvidenceRecord { record_id } => {
                write!(f, "independent evidence record is reused: {record_id}")
            }
            Self::EvidenceNotCurrent { domain, status } => {
                write!(f, "evidence for {domain:?} is not current: {status:?}")
            }
            Self::EvidenceRunMismatch { domain } => {
                write!(f, "evidence for {domain:?} is bound to a different run")
            }
            Self::WrongAuthority {
                domain,
                expected,
                actual,
            } => write!(
                f,
                "assessment authority mismatch for {domain:?}: expected {expected:?}, got {actual:?}"
            ),
            Self::EvidenceMethodMismatch { domain } => {
                write!(f, "evidence method does not match the profile for {domain:?}")
            }
            Self::EvidenceConditionsMismatch { domain } => {
                write!(f, "evidence method conditions do not match the profile for {domain:?}")
            }
            Self::EvidenceTrialConditionsMismatch { domain } => {
                write!(f, "evidence trial conditions do not match the profile for {domain:?}")
            }
            Self::SameRunCompared => {
                write!(f, "a service run cannot be compared with itself")
            }
        }
    }
}

impl Error for ServiceQualificationError {}

impl ServiceQualificationRecord {
    pub fn validate(
        &self,
        profile: &ServiceQualificationProfile,
    ) -> Result<(), ServiceQualificationError> {
        profile.validate()?;
        require_text("run_id", &self.run_id)?;
        require_text("record.profile_id", &self.profile_id)?;
        require_digest(
            "record.device_configuration_sha256",
            &self.device_configuration_sha256,
        )?;
        require_digest("record.protocol_sha256", &self.protocol_sha256)?;
        require_digest("record.conditions_sha256", &self.conditions_sha256)?;

        if self.profile_id != profile.profile_id {
            return Err(ServiceQualificationError::ProfileMismatch);
        }
        if self.protocol_sha256 != profile.protocol_sha256 {
            return Err(ServiceQualificationError::ProtocolMismatch);
        }
        if self.conditions_sha256 != profile.conditions_sha256 {
            return Err(ServiceQualificationError::ConditionsMismatch);
        }

        let required = profile.claim.required_domains();
        let mut seen = HashSet::with_capacity(required.len());
        let mut evidence_record_ids = HashSet::with_capacity(self.assessments.len());
        for assessment in &self.assessments {
            if !required.contains(&assessment.domain) {
                return Err(ServiceQualificationError::UnexpectedDomain {
                    domain: assessment.domain,
                });
            }
            if !seen.insert(assessment.domain) {
                return Err(ServiceQualificationError::DuplicateDomain {
                    domain: assessment.domain,
                });
            }
            if !evidence_record_ids.insert(assessment.evidence.record_id.clone()) {
                return Err(ServiceQualificationError::DuplicateEvidenceRecord {
                    record_id: assessment.evidence.record_id.clone(),
                });
            }
            assessment
                .evidence
                .validate(assessment.domain, profile, &self.run_id)?;
        }

        for domain in required {
            if !seen.contains(domain) {
                return Err(ServiceQualificationError::MissingDomain { domain: *domain });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationDisposition {
    Qualified,
    Failed,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceQualificationResult {
    pub disposition: QualificationDisposition,
    pub failed_domains: Vec<QualificationDomain>,
    pub indeterminate_domains: Vec<QualificationDomain>,
}

/// Evaluate completeness and independent decisions only. No cross-domain score is produced.
pub fn evaluate_service_qualification(
    profile: &ServiceQualificationProfile,
    record: &ServiceQualificationRecord,
) -> Result<ServiceQualificationResult, ServiceQualificationError> {
    record.validate(profile)?;

    let failed_domains = record
        .assessments
        .iter()
        .filter(|assessment| assessment.outcome == AssessmentOutcome::Fail)
        .map(|assessment| assessment.domain)
        .collect::<Vec<_>>();
    let indeterminate_domains = record
        .assessments
        .iter()
        .filter(|assessment| assessment.outcome == AssessmentOutcome::Indeterminate)
        .map(|assessment| assessment.domain)
        .collect::<Vec<_>>();

    let disposition = if !failed_domains.is_empty() {
        QualificationDisposition::Failed
    } else if !indeterminate_domains.is_empty() {
        QualificationDisposition::Indeterminate
    } else {
        QualificationDisposition::Qualified
    };

    Ok(ServiceQualificationResult {
        disposition,
        failed_domains,
        indeterminate_domains,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchedComparisonDisposition {
    CandidateQualified,
    CandidateFailed,
    CandidateIndeterminate,
    ControlNotQualified,
    ControlIndeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedServiceQualificationResult {
    pub disposition: MatchedComparisonDisposition,
    pub control: ServiceQualificationResult,
    pub candidate: ServiceQualificationResult,
}

/// Qualify a matched candidate only when the control and candidate share the frozen
/// task/protocol conditions and the control itself is qualified. A failed or inconclusive
/// control cannot support a comparative improvement claim.
pub fn evaluate_matched_service_qualification(
    profile: &ServiceQualificationProfile,
    control: &ServiceQualificationRecord,
    candidate: &ServiceQualificationRecord,
) -> Result<MatchedServiceQualificationResult, ServiceQualificationError> {
    control.validate(profile)?;
    candidate.validate(profile)?;

    if control.run_id == candidate.run_id {
        return Err(ServiceQualificationError::SameRunCompared);
    }
    if control.profile_id != candidate.profile_id {
        return Err(ServiceQualificationError::ProfileMismatch);
    }
    if control.protocol_sha256 != candidate.protocol_sha256 {
        return Err(ServiceQualificationError::ProtocolMismatch);
    }
    if control.conditions_sha256 != candidate.conditions_sha256 {
        return Err(ServiceQualificationError::ConditionsMismatch);
    }

    let control_result = evaluate_service_qualification(profile, control)?;
    let candidate_result = evaluate_service_qualification(profile, candidate)?;

    let disposition = match control_result.disposition {
        QualificationDisposition::Failed => MatchedComparisonDisposition::ControlNotQualified,
        QualificationDisposition::Indeterminate => {
            MatchedComparisonDisposition::ControlIndeterminate
        }
        QualificationDisposition::Qualified => match candidate_result.disposition {
            QualificationDisposition::Qualified => {
                MatchedComparisonDisposition::CandidateQualified
            }
            QualificationDisposition::Failed => MatchedComparisonDisposition::CandidateFailed,
            QualificationDisposition::Indeterminate => {
                MatchedComparisonDisposition::CandidateIndeterminate
            }
        },
    };

    Ok(MatchedServiceQualificationResult {
        disposition,
        control: control_result,
        candidate: candidate_result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn profile(claim: QualificationClaim) -> ServiceQualificationProfile {
        ServiceQualificationProfile {
            profile_id: "AQS1-FUNCTIONAL-001".to_owned(),
            claim,
            protocol_sha256: digest('a'),
            conditions_sha256: digest('b'),
            method_bindings: claim
                .required_domains()
                .iter()
                .map(|domain| QualificationMethodBinding {
                    domain: *domain,
                    method_sha256: digest('d'),
                    conditions_sha256: digest('e'),
                })
                .collect(),
        }
    }

    fn assessment(
        run_id: &str,
        domain: QualificationDomain,
        outcome: AssessmentOutcome,
        profile: &ServiceQualificationProfile,
    ) -> DomainAssessment {
        DomainAssessment {
            domain,
            outcome,
            evidence: QualificationEvidenceRef {
                authority: domain.expected_authority(),
                run_id: run_id.to_owned(),
                issuer_principal_id: format!("issuer-{domain:?}"),
                authority_verification_receipt_sha256: digest('f'),
                status: QualificationEvidenceStatus::Current,
                record_id: format!("assessment-{domain:?}"),
                evidence_sha256: digest('c'),
                method_sha256: profile
                    .method_binding(domain)
                    .expect("profile method")
                    .method_sha256
                    .clone(),
                method_conditions_sha256: profile
                    .method_binding(domain)
                    .expect("profile method")
                    .conditions_sha256
                    .clone(),
                trial_conditions_sha256: profile.conditions_sha256.clone(),
            },
        }
    }

    fn record(
        run_id: &str,
        profile: &ServiceQualificationProfile,
        outcome: AssessmentOutcome,
    ) -> ServiceQualificationRecord {
        ServiceQualificationRecord {
            run_id: run_id.to_owned(),
            device_configuration_sha256: digest('6'),
            profile_id: profile.profile_id.clone(),
            protocol_sha256: profile.protocol_sha256.clone(),
            conditions_sha256: profile.conditions_sha256.clone(),
            assessments: profile
                .claim
                .required_domains()
                .iter()
                .map(|domain| assessment(run_id, *domain, outcome, profile))
                .collect(),
        }
    }

    #[test]
    fn evidence_must_be_bound_to_the_exact_run_id() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.run_id = "other-run".to_owned();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceRunMismatch {
                domain: QualificationDomain::MeasurementValidity
            })
        );
    }

    #[test]
    fn stale_evidence_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.status = QualificationEvidenceStatus::Stale;
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceNotCurrent {
                domain: QualificationDomain::MeasurementValidity,
                status: QualificationEvidenceStatus::Stale
            })
        );
    }

    #[test]
    fn revoked_evidence_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.status = QualificationEvidenceStatus::Revoked;
        assert!(matches!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceNotCurrent {
                status: QualificationEvidenceStatus::Revoked,
                ..
            })
        ));
    }

    #[test]
    fn ambiguous_evidence_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.status = QualificationEvidenceStatus::Ambiguous;
        assert!(matches!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceNotCurrent {
                status: QualificationEvidenceStatus::Ambiguous,
                ..
            })
        ));
    }

    #[test]
    fn blank_issuer_principal_identity_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.issuer_principal_id.clear();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::MissingField {
                field: "assessment.issuer_principal_id"
            })
        );
    }

    #[test]
    fn invalid_authority_verification_receipt_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0]
            .evidence
            .authority_verification_receipt_sha256 = "invalid".to_owned();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::InvalidDigest {
                field: "assessment.authority_verification_receipt_sha256"
            })
        );
    }

    #[test]
    fn evidence_record_cannot_be_reused_for_two_domains() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        let first_record_id = record.assessments[0].evidence.record_id.clone();
        record.assessments[1].evidence.record_id = first_record_id.clone();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::DuplicateEvidenceRecord {
                record_id: first_record_id
            })
        );
    }

    #[test]
    fn profile_rejects_a_missing_domain_method_binding() {
        let mut profile = profile(QualificationClaim::FunctionalService);
        profile.method_bindings.pop();
        assert_eq!(
            profile.validate(),
            Err(ServiceQualificationError::MissingMethodBinding {
                domain: QualificationDomain::LocalSafety
            })
        );
    }

    #[test]
    fn profile_rejects_duplicate_domain_method_bindings() {
        let mut profile = profile(QualificationClaim::FunctionalService);
        let first_binding = profile.method_bindings[0].clone();
        profile.method_bindings.push(first_binding);
        assert!(matches!(
            profile.validate(),
            Err(ServiceQualificationError::DuplicateMethodBinding { .. })
        ));
    }

    #[test]
    fn profile_rejects_method_binding_outside_claim_scope() {
        let mut profile = profile(QualificationClaim::FunctionalService);
        profile.method_bindings.push(QualificationMethodBinding {
            domain: QualificationDomain::AccessibilityAndUsability,
            method_sha256: digest('f'),
            conditions_sha256: digest('e'),
        });
        assert_eq!(
            profile.validate(),
            Err(ServiceQualificationError::UnexpectedMethodBinding {
                domain: QualificationDomain::AccessibilityAndUsability
            })
        );
    }

    #[test]
    fn each_domain_can_bind_its_own_specialist_method() {
        let mut profile = profile(QualificationClaim::FunctionalService);
        let fixture_binding = profile
            .method_bindings
            .iter_mut()
            .find(|binding| binding.domain == QualificationDomain::FixtureMethodPerformance)
            .expect("fixture method binding");
        fixture_binding.method_sha256 = digest('f');
        fixture_binding.conditions_sha256 = digest('g');

        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(record.validate(&profile), Ok(()));

        let metrology = record
            .assessments
            .iter()
            .find(|assessment| assessment.domain == QualificationDomain::MeasurementValidity)
            .expect("metrology assessment");
        let fixture = record
            .assessments
            .iter()
            .find(|assessment| assessment.domain == QualificationDomain::FixtureMethodPerformance)
            .expect("fixture performance assessment");

        assert_ne!(
            metrology.evidence.method_sha256,
            fixture.evidence.method_sha256
        );
        assert_ne!(
            metrology.evidence.method_conditions_sha256,
            fixture.evidence.method_conditions_sha256
        );
    }

    #[test]
    fn receipt_trial_conditions_are_bound_to_the_matched_service_profile() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.trial_conditions_sha256 = digest('z');
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceTrialConditionsMismatch {
                domain: QualificationDomain::MeasurementValidity
            })
        );
    }

    #[test]
    fn complete_functional_service_profile_qualifies() {
        let profile = profile(QualificationClaim::FunctionalService);
        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_service_qualification(&profile, &record)
                .expect("valid qualification")
                .disposition,
            QualificationDisposition::Qualified
        );
    }

    #[test]
    fn functional_profile_has_seven_non_compensatory_domains() {
        let profile = profile(QualificationClaim::FunctionalService);
        assert_eq!(profile.claim.required_domains().len(), 7);
    }

    #[test]
    fn missing_required_domain_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments.pop();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::MissingDomain {
                domain: QualificationDomain::LocalSafety
            })
        );
    }

    #[test]
    fn unexpected_domain_outside_claim_profile_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments.push(assessment(
            QualificationDomain::AccessibilityAndUsability,
            AssessmentOutcome::Pass,
            &profile,
        ));
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::UnexpectedDomain {
                domain: QualificationDomain::AccessibilityAndUsability
            })
        );
    }

    #[test]
    fn duplicate_assessment_domain_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        let first_assessment = record.assessments[0].clone();
        record.assessments.push(first_assessment);
        assert!(matches!(
            record.validate(&profile),
            Err(ServiceQualificationError::DuplicateDomain { .. })
        ));
    }

    #[test]
    fn incorrect_authority_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.authority =
            QualificationAuthority::IndependentTaskAssessor;
        assert!(matches!(
            record.validate(&profile),
            Err(ServiceQualificationError::WrongAuthority { .. })
        ));
    }

    #[test]
    fn evidence_method_must_match_the_frozen_profile() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.method_sha256 = digest('z');
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceMethodMismatch {
                domain: QualificationDomain::MeasurementValidity
            })
        );
    }

    #[test]
    fn evidence_conditions_must_match_the_frozen_profile() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.method_conditions_sha256 = digest('z');
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::EvidenceConditionsMismatch {
                domain: QualificationDomain::MeasurementValidity
            })
        );
    }

    #[test]
    fn invalid_evidence_digest_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.evidence_sha256 = "invalid".to_owned();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::InvalidDigest {
                field: "assessment.evidence_sha256"
            })
        );
    }

    #[test]
    fn blank_assessment_record_identity_is_rejected() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].evidence.record_id.clear();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::MissingField {
                field: "assessment.record_id"
            })
        );
    }

    #[test]
    fn failed_endpoint_prevents_qualification() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[2].outcome = AssessmentOutcome::Fail;
        let result = evaluate_service_qualification(&profile, &record).expect("valid record");
        assert_eq!(result.disposition, QualificationDisposition::Failed);
        assert_eq!(
            result.failed_domains,
            vec![QualificationDomain::FixtureMethodPerformance]
        );
    }

    #[test]
    fn indeterminate_endpoint_is_not_promoted_to_pass() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[1].outcome = AssessmentOutcome::Indeterminate;
        let result = evaluate_service_qualification(&profile, &record).expect("valid record");
        assert_eq!(result.disposition, QualificationDisposition::Indeterminate);
        assert_eq!(
            result.indeterminate_domains,
            vec![QualificationDomain::TaskCompletionAndComfort]
        );
    }

    #[test]
    fn failure_dominates_indeterminate_when_both_are_present() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.assessments[0].outcome = AssessmentOutcome::Indeterminate;
        record.assessments[3].outcome = AssessmentOutcome::Fail;
        let result = evaluate_service_qualification(&profile, &record).expect("valid record");
        assert_eq!(result.disposition, QualificationDisposition::Failed);
        assert_eq!(result.failed_domains.len(), 1);
        assert_eq!(result.indeterminate_domains.len(), 1);
    }

    #[test]
    fn water_efficiency_claim_requires_completed_task_water_savings_evidence() {
        let profile = profile(QualificationClaim::WaterEfficiency);
        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert!(profile
            .claim
            .required_domains()
            .contains(&QualificationDomain::WaterSavingsPerCompletedTask));
        assert_eq!(
            evaluate_service_qualification(&profile, &record)
                .expect("complete evidence")
                .disposition,
            QualificationDisposition::Qualified
        );
    }

    #[test]
    fn water_and_energy_claim_requires_both_resource_domains() {
        let profile = profile(QualificationClaim::WaterAndEnergyEfficiency);
        let required = profile.claim.required_domains();
        assert!(required.contains(&QualificationDomain::WaterSavingsPerCompletedTask));
        assert!(required.contains(&QualificationDomain::EnergySavingsPerCompletedTask));
        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_service_qualification(&profile, &record)
                .expect("complete evidence")
                .disposition,
            QualificationDisposition::Qualified
        );
    }

    #[test]
    fn resource_savings_must_be_assessed_by_symtropy_resource_closure() {
        let profile = profile(QualificationClaim::WaterEfficiency);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        let assessment = record
            .assessments
            .iter_mut()
            .find(|assessment| {
                assessment.domain == QualificationDomain::WaterSavingsPerCompletedTask
            })
            .expect("water savings domain");
        assessment.evidence.authority = QualificationAuthority::IndependentTestLaboratory;
        assert!(matches!(
            record.validate(&profile),
            Err(ServiceQualificationError::WrongAuthority {
                domain: QualificationDomain::WaterSavingsPerCompletedTask,
                ..
            })
        ));
    }

    #[test]
    fn accessible_claim_requires_accessibility_assessment() {
        let profile = profile(QualificationClaim::AccessibleService);
        assert!(profile
            .claim
            .required_domains()
            .contains(&QualificationDomain::AccessibilityAndUsability));
        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_service_qualification(&profile, &record)
                .expect("complete evidence")
                .disposition,
            QualificationDisposition::Qualified
        );
    }

    #[test]
    fn lifecycle_claim_requires_serviceability_and_cleaning_burden() {
        let profile = profile(QualificationClaim::LifecycleEfficiency);
        let required = profile.claim.required_domains();
        assert!(required.contains(&QualificationDomain::Serviceability));
        assert!(required.contains(&QualificationDomain::CleaningBurden));
        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_service_qualification(&profile, &record)
                .expect("complete evidence")
                .disposition,
            QualificationDisposition::Qualified
        );
    }

    #[test]
    fn full_platform_profile_covers_all_domains() {
        let profile = profile(QualificationClaim::FullPlatform);
        assert_eq!(profile.claim.required_domains().len(), 13);
        let record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_service_qualification(&profile, &record)
                .expect("complete evidence")
                .disposition,
            QualificationDisposition::Qualified
        );
    }

    #[test]
    fn profile_requires_identity_and_valid_digests() {
        let mut profile = profile(QualificationClaim::FunctionalService);
        profile.profile_id.clear();
        assert_eq!(
            profile.validate(),
            Err(ServiceQualificationError::MissingField {
                field: "profile_id"
            })
        );
        profile.profile_id = "profile".to_owned();
        profile.protocol_sha256 = "bad".to_owned();
        assert_eq!(
            profile.validate(),
            Err(ServiceQualificationError::InvalidDigest {
                field: "profile.protocol_sha256"
            })
        );
    }

    #[test]
    fn record_must_match_claim_profile_id() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.profile_id = "other-profile".to_owned();
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::ProfileMismatch)
        );
    }

    #[test]
    fn record_must_match_profile_protocol() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.protocol_sha256 = digest('z');
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::ProtocolMismatch)
        );
    }

    #[test]
    fn record_must_match_profile_conditions() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut record = record("candidate-01", &profile, AssessmentOutcome::Pass);
        record.conditions_sha256 = digest('z');
        assert_eq!(
            record.validate(&profile),
            Err(ServiceQualificationError::ConditionsMismatch)
        );
    }

    #[test]
    fn matched_pair_qualifies_only_when_both_runs_pass() {
        let profile = profile(QualificationClaim::FunctionalService);
        let control = record("control-01", &profile, AssessmentOutcome::Pass);
        let candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        let result = evaluate_matched_service_qualification(&profile, &control, &candidate)
            .expect("matched runs");
        assert_eq!(
            result.disposition,
            MatchedComparisonDisposition::CandidateQualified
        );
    }

    #[test]
    fn unqualified_control_prevents_comparative_claim() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut control = record("control-01", &profile, AssessmentOutcome::Pass);
        control.assessments[0].outcome = AssessmentOutcome::Fail;
        let candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        let result = evaluate_matched_service_qualification(&profile, &control, &candidate)
            .expect("matched runs");
        assert_eq!(
            result.disposition,
            MatchedComparisonDisposition::ControlNotQualified
        );
    }

    #[test]
    fn indeterminate_control_prevents_comparative_claim() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut control = record("control-01", &profile, AssessmentOutcome::Pass);
        control.assessments[0].outcome = AssessmentOutcome::Indeterminate;
        let candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        let result = evaluate_matched_service_qualification(&profile, &control, &candidate)
            .expect("matched runs");
        assert_eq!(
            result.disposition,
            MatchedComparisonDisposition::ControlIndeterminate
        );
    }

    #[test]
    fn candidate_failure_is_not_a_qualified_comparison() {
        let profile = profile(QualificationClaim::FunctionalService);
        let control = record("control-01", &profile, AssessmentOutcome::Pass);
        let mut candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        candidate.assessments[4].outcome = AssessmentOutcome::Fail;
        let result = evaluate_matched_service_qualification(&profile, &control, &candidate)
            .expect("matched runs");
        assert_eq!(
            result.disposition,
            MatchedComparisonDisposition::CandidateFailed
        );
    }

    #[test]
    fn candidate_indeterminate_is_not_a_qualified_comparison() {
        let profile = profile(QualificationClaim::FunctionalService);
        let control = record("control-01", &profile, AssessmentOutcome::Pass);
        let mut candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        candidate.assessments[5].outcome = AssessmentOutcome::Indeterminate;
        let result = evaluate_matched_service_qualification(&profile, &control, &candidate)
            .expect("matched runs");
        assert_eq!(
            result.disposition,
            MatchedComparisonDisposition::CandidateIndeterminate
        );
    }

    #[test]
    fn identical_run_ids_cannot_form_a_matched_comparison() {
        let profile = profile(QualificationClaim::FunctionalService);
        let control = record("same-run", &profile, AssessmentOutcome::Pass);
        let candidate = record("same-run", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_matched_service_qualification(&profile, &control, &candidate),
            Err(ServiceQualificationError::SameRunCompared)
        );
    }

    #[test]
    fn matched_comparison_rejects_different_conditions() {
        let profile = profile(QualificationClaim::FunctionalService);
        let control = record("control-01", &profile, AssessmentOutcome::Pass);
        let mut candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        candidate.conditions_sha256 = digest('z');
        assert_eq!(
            evaluate_matched_service_qualification(&profile, &control, &candidate),
            Err(ServiceQualificationError::ConditionsMismatch)
        );
    }

    #[test]
    fn matched_comparison_rejects_blank_run_identity() {
        let profile = profile(QualificationClaim::FunctionalService);
        let mut control = record("control-01", &profile, AssessmentOutcome::Pass);
        control.run_id.clear();
        let candidate = record("candidate-01", &profile, AssessmentOutcome::Pass);
        assert_eq!(
            evaluate_matched_service_qualification(&profile, &control, &candidate),
            Err(ServiceQualificationError::MissingField { field: "run_id" })
        );
    }
}
