//! Evidence-backed installation eligibility for Aqua architectures.
//!
//! This module consumes assessments from canonical owners and competent site assessors.
//! It does not calculate hydraulic, electrical, plumbing, thermal, water-quality, hygiene,
//! structural or regulatory outcomes. It does not rank architectures or authorize equipment.
//! Missing, stale, revoked, ambiguous or out-of-scope evidence is Indeterminate, never Eligible.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::service_qualification::QualificationEvidenceStatus;

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_text(field: &'static str, value: &str) -> Result<(), SiteFitError> {
    if value.trim().is_empty() {
        return Err(SiteFitError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), SiteFitError> {
    if !is_sha256(value) {
        return Err(SiteFitError::InvalidDigest { field });
    }
    Ok(())
}

/// Complete architecture classes considered by the first site-fit contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AquaArchitecture {
    DirectFlowPassive,
    DirectFlowWithDwhR,
    PowerModulatedDirectFlowWithDwhR,
    SameSessionRecirculation,
    SameSessionRecirculationWithDwhR,
}

/// Independent site capability that must be evidenced before an architecture is eligible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SiteCapabilityDomain {
    DynamicSupplyEnvelope,
    PerInletSupplyCapacity,
    WaterTemperatureEnvelope,
    WaterChemistryAndScale,
    ElectricalSupplyAndProtection,
    ContinuousPowerAvailability,
    DrainAndWasteGeometry,
    InstallationSpaceAndRouting,
    OperatingEnvironmentAndFreezeProtection,
    ThermalReceiverAndHeatSinkCompatibility,
    ServiceAndMaintenanceCapacity,
    WetPathHygieneAndCleaning,
    BackflowAndCrossConnectionProtection,
    NetworkConnectivity,
}

/// Canonical evidence owner or competent assessor expected for each site capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SiteCapabilityAuthority {
    SymtropyHydraulicAuthority,
    SymtropyThermalAuthority,
    MycelixWaterQuality,
    IndependentElectricalAssessor,
    CompetentInstallationSurveyor,
    CompetentPlumbingAssessor,
    IndependentServiceabilityAssessor,
    IndependentHygieneExposureAssessor,
    IndependentSiteInfrastructureAssessor,
}

impl SiteCapabilityDomain {
    pub fn expected_authority(self) -> SiteCapabilityAuthority {
        match self {
            Self::DynamicSupplyEnvelope | Self::PerInletSupplyCapacity => {
                SiteCapabilityAuthority::SymtropyHydraulicAuthority
            }
            Self::WaterTemperatureEnvelope
            | Self::ThermalReceiverAndHeatSinkCompatibility => {
                SiteCapabilityAuthority::SymtropyThermalAuthority
            }
            Self::WaterChemistryAndScale => SiteCapabilityAuthority::MycelixWaterQuality,
            Self::ElectricalSupplyAndProtection | Self::ContinuousPowerAvailability => {
                SiteCapabilityAuthority::IndependentElectricalAssessor
            }
            Self::DrainAndWasteGeometry
            | Self::InstallationSpaceAndRouting
            | Self::OperatingEnvironmentAndFreezeProtection => {
                SiteCapabilityAuthority::CompetentInstallationSurveyor
            }
            Self::ServiceAndMaintenanceCapacity => {
                SiteCapabilityAuthority::IndependentServiceabilityAssessor
            }
            Self::WetPathHygieneAndCleaning => {
                SiteCapabilityAuthority::IndependentHygieneExposureAssessor
            }
            Self::BackflowAndCrossConnectionProtection => {
                SiteCapabilityAuthority::CompetentPlumbingAssessor
            }
            Self::NetworkConnectivity => {
                SiteCapabilityAuthority::IndependentSiteInfrastructureAssessor
            }
        }
    }
}

const DIRECT_FLOW_BASE: &[SiteCapabilityDomain] = &[
    SiteCapabilityDomain::DynamicSupplyEnvelope,
    SiteCapabilityDomain::WaterTemperatureEnvelope,
    SiteCapabilityDomain::WaterChemistryAndScale,
    SiteCapabilityDomain::DrainAndWasteGeometry,
    SiteCapabilityDomain::InstallationSpaceAndRouting,
    SiteCapabilityDomain::ServiceAndMaintenanceCapacity,
];

const DIRECT_FLOW_DWHR: &[SiteCapabilityDomain] = &[
    SiteCapabilityDomain::DynamicSupplyEnvelope,
    SiteCapabilityDomain::WaterTemperatureEnvelope,
    SiteCapabilityDomain::WaterChemistryAndScale,
    SiteCapabilityDomain::DrainAndWasteGeometry,
    SiteCapabilityDomain::InstallationSpaceAndRouting,
    SiteCapabilityDomain::ThermalReceiverAndHeatSinkCompatibility,
    SiteCapabilityDomain::ServiceAndMaintenanceCapacity,
];

const POWER_MODULATED_DWHR: &[SiteCapabilityDomain] = &[
    SiteCapabilityDomain::DynamicSupplyEnvelope,
    SiteCapabilityDomain::WaterTemperatureEnvelope,
    SiteCapabilityDomain::WaterChemistryAndScale,
    SiteCapabilityDomain::ElectricalSupplyAndProtection,
    SiteCapabilityDomain::ContinuousPowerAvailability,
    SiteCapabilityDomain::DrainAndWasteGeometry,
    SiteCapabilityDomain::InstallationSpaceAndRouting,
    SiteCapabilityDomain::ThermalReceiverAndHeatSinkCompatibility,
    SiteCapabilityDomain::ServiceAndMaintenanceCapacity,
];

const SAME_SESSION_RECIRCULATION: &[SiteCapabilityDomain] = &[
    SiteCapabilityDomain::DynamicSupplyEnvelope,
    SiteCapabilityDomain::PerInletSupplyCapacity,
    SiteCapabilityDomain::WaterTemperatureEnvelope,
    SiteCapabilityDomain::WaterChemistryAndScale,
    SiteCapabilityDomain::ElectricalSupplyAndProtection,
    SiteCapabilityDomain::ContinuousPowerAvailability,
    SiteCapabilityDomain::DrainAndWasteGeometry,
    SiteCapabilityDomain::InstallationSpaceAndRouting,
    SiteCapabilityDomain::ServiceAndMaintenanceCapacity,
    SiteCapabilityDomain::WetPathHygieneAndCleaning,
    SiteCapabilityDomain::BackflowAndCrossConnectionProtection,
];

const SAME_SESSION_RECIRCULATION_DWHR: &[SiteCapabilityDomain] = &[
    SiteCapabilityDomain::DynamicSupplyEnvelope,
    SiteCapabilityDomain::PerInletSupplyCapacity,
    SiteCapabilityDomain::WaterTemperatureEnvelope,
    SiteCapabilityDomain::WaterChemistryAndScale,
    SiteCapabilityDomain::ElectricalSupplyAndProtection,
    SiteCapabilityDomain::ContinuousPowerAvailability,
    SiteCapabilityDomain::DrainAndWasteGeometry,
    SiteCapabilityDomain::InstallationSpaceAndRouting,
    SiteCapabilityDomain::ThermalReceiverAndHeatSinkCompatibility,
    SiteCapabilityDomain::ServiceAndMaintenanceCapacity,
    SiteCapabilityDomain::WetPathHygieneAndCleaning,
    SiteCapabilityDomain::BackflowAndCrossConnectionProtection,
];

impl AquaArchitecture {
    fn baseline_domains(self) -> &'static [SiteCapabilityDomain] {
        match self {
            Self::DirectFlowPassive => DIRECT_FLOW_BASE,
            Self::DirectFlowWithDwhR => DIRECT_FLOW_DWHR,
            Self::PowerModulatedDirectFlowWithDwhR => POWER_MODULATED_DWHR,
            Self::SameSessionRecirculation => SAME_SESSION_RECIRCULATION,
            Self::SameSessionRecirculationWithDwhR => SAME_SESSION_RECIRCULATION_DWHR,
        }
    }
}

/// One frozen method and its conditions for a site-capability domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteFitMethodBinding {
    pub domain: SiteCapabilityDomain,
    pub method_sha256: String,
    pub method_conditions_sha256: String,
}

/// Fixed site/configuration scope for one architecture eligibility decision.
///
/// Additional required domains can add product-specific constraints (for example network
/// connectivity when the exact product requires it), but cannot remove mandatory domains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteFitProfile {
    pub site_profile_id: String,
    pub site_revision: String,
    pub site_conditions_sha256: String,
    pub architecture_profile_id: String,
    pub architecture: AquaArchitecture,
    pub device_configuration_sha256: String,
    pub requirements_sha256: String,
    pub additional_required_domains: Vec<SiteCapabilityDomain>,
    pub method_bindings: Vec<SiteFitMethodBinding>,
}

impl SiteFitProfile {
    pub fn required_domains(&self) -> Vec<SiteCapabilityDomain> {
        let mut required = self.architecture.baseline_domains().to_vec();
        required.extend(self.additional_required_domains.iter().copied());
        required
    }

    fn method_binding(&self, domain: SiteCapabilityDomain) -> Option<&SiteFitMethodBinding> {
        self.method_bindings
            .iter()
            .find(|binding| binding.domain == domain)
    }

    pub fn validate(&self) -> Result<(), SiteFitError> {
        require_text("site_profile_id", &self.site_profile_id)?;
        require_text("site_revision", &self.site_revision)?;
        require_digest("site_conditions_sha256", &self.site_conditions_sha256)?;
        require_text("architecture_profile_id", &self.architecture_profile_id)?;
        require_digest(
            "device_configuration_sha256",
            &self.device_configuration_sha256,
        )?;
        require_digest("requirements_sha256", &self.requirements_sha256)?;

        let baseline = self.architecture.baseline_domains();
        let mut required = HashSet::with_capacity(baseline.len());
        for domain in baseline {
            if !required.insert(*domain) {
                return Err(SiteFitError::DuplicateRequiredDomain { domain: *domain });
            }
        }

        for domain in &self.additional_required_domains {
            if !required.insert(*domain) {
                return Err(SiteFitError::DuplicateAdditionalDomain { domain: *domain });
            }
        }

        let mut bound = HashSet::with_capacity(required.len());
        for binding in &self.method_bindings {
            if !required.contains(&binding.domain) {
                return Err(SiteFitError::UnexpectedMethodBinding {
                    domain: binding.domain,
                });
            }
            if !bound.insert(binding.domain) {
                return Err(SiteFitError::DuplicateMethodBinding {
                    domain: binding.domain,
                });
            }
            require_digest("method_binding.method_sha256", &binding.method_sha256)?;
            require_digest(
                "method_binding.method_conditions_sha256",
                &binding.method_conditions_sha256,
            )?;
        }

        for domain in required {
            if !bound.contains(&domain) {
                return Err(SiteFitError::MissingMethodBinding { domain });
            }
        }
        Ok(())
    }
}

/// Evidence identity and scope returned by the canonical owner / competent assessor.
///
/// The issuer service must authenticate the issuer and produce the verification receipt.
/// This module only checks that the receipt is present and correctly bound; it does not
/// implement signature verification or trust-registry lookup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteFitEvidenceRef {
    pub domain: SiteCapabilityDomain,
    pub authority: SiteCapabilityAuthority,
    pub issuer_principal_id: String,
    pub authority_verification_receipt_sha256: String,
    pub status: QualificationEvidenceStatus,
    pub record_id: String,
    pub evidence_sha256: String,
    pub method_sha256: String,
    pub method_conditions_sha256: String,
    pub site_profile_id: String,
    pub site_revision: String,
    pub site_conditions_sha256: String,
    pub architecture_profile_id: String,
    pub architecture: AquaArchitecture,
    pub device_configuration_sha256: String,
    pub requirements_sha256: String,
}

impl SiteFitEvidenceRef {
    fn validate_structure(&self) -> Result<(), SiteFitError> {
        require_text("evidence.issuer_principal_id", &self.issuer_principal_id)?;
        require_text("evidence.record_id", &self.record_id)?;
        require_digest(
            "evidence.authority_verification_receipt_sha256",
            &self.authority_verification_receipt_sha256,
        )?;
        require_digest("evidence.evidence_sha256", &self.evidence_sha256)?;
        require_digest("evidence.method_sha256", &self.method_sha256)?;
        require_digest(
            "evidence.method_conditions_sha256",
            &self.method_conditions_sha256,
        )?;
        require_text("evidence.site_profile_id", &self.site_profile_id)?;
        require_text("evidence.site_revision", &self.site_revision)?;
        require_digest(
            "evidence.site_conditions_sha256",
            &self.site_conditions_sha256,
        )?;
        require_text(
            "evidence.architecture_profile_id",
            &self.architecture_profile_id,
        )?;
        require_digest(
            "evidence.device_configuration_sha256",
            &self.device_configuration_sha256,
        )?;
        require_digest(
            "evidence.requirements_sha256",
            &self.requirements_sha256,
        )?;
        Ok(())
    }
}

/// Independent result for one required site capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteCapabilityOutcome {
    Satisfied,
    Unsatisfied,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteCapabilityAssessment {
    pub domain: SiteCapabilityDomain,
    pub outcome: SiteCapabilityOutcome,
    pub evidence: Option<SiteFitEvidenceRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteFitDisposition {
    Eligible,
    Ineligible,
    Indeterminate,
}

/// Why a required site capability cannot currently support an Eligible decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteFitIssueKind {
    MissingAssessment,
    MissingEvidence,
    EvidenceNotCurrent(QualificationEvidenceStatus),
    EvidenceDomainMismatch {
        actual: SiteCapabilityDomain,
    },
    WrongAuthority {
        expected: SiteCapabilityAuthority,
        actual: SiteCapabilityAuthority,
    },
    SiteProfileMismatch,
    SiteRevisionMismatch,
    SiteConditionsMismatch,
    ArchitectureProfileMismatch,
    ArchitectureMismatch,
    DeviceConfigurationMismatch,
    RequirementsMismatch,
    MethodMismatch,
    MethodConditionsMismatch,
    IndeterminateAssessment,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteFitIssue {
    pub domain: SiteCapabilityDomain,
    pub kind: SiteFitIssueKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteFitResult {
    pub disposition: SiteFitDisposition,
    pub satisfied_domains: Vec<SiteCapabilityDomain>,
    pub unsatisfied_domains: Vec<SiteCapabilityDomain>,
    pub indeterminate_issues: Vec<SiteFitIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SiteFitError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    DuplicateRequiredDomain { domain: SiteCapabilityDomain },
    DuplicateAdditionalDomain { domain: SiteCapabilityDomain },
    MissingMethodBinding { domain: SiteCapabilityDomain },
    UnexpectedMethodBinding { domain: SiteCapabilityDomain },
    DuplicateMethodBinding { domain: SiteCapabilityDomain },
    UnexpectedAssessment { domain: SiteCapabilityDomain },
    DuplicateAssessment { domain: SiteCapabilityDomain },
    DuplicateEvidenceRecord { record_id: String },
}

impl Display for SiteFitError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing site-fit field: {field}"),
            Self::InvalidDigest { field } => {
                write!(f, "invalid site-fit SHA-256: {field}")
            }
            Self::DuplicateRequiredDomain { domain } => {
                write!(f, "architecture repeats a required domain: {domain:?}")
            }
            Self::DuplicateAdditionalDomain { domain } => {
                write!(f, "additional requirements repeat a domain: {domain:?}")
            }
            Self::MissingMethodBinding { domain } => {
                write!(f, "site-fit profile has no method binding for {domain:?}")
            }
            Self::UnexpectedMethodBinding { domain } => {
                write!(f, "site-fit profile has an out-of-scope method binding: {domain:?}")
            }
            Self::DuplicateMethodBinding { domain } => {
                write!(f, "site-fit profile repeats a method binding: {domain:?}")
            }
            Self::UnexpectedAssessment { domain } => {
                write!(f, "assessment is outside the architecture scope: {domain:?}")
            }
            Self::DuplicateAssessment { domain } => {
                write!(f, "site capability has duplicate assessments: {domain:?}")
            }
            Self::DuplicateEvidenceRecord { record_id } => {
                write!(f, "site evidence record is reused: {record_id}")
            }
        }
    }
}

impl Error for SiteFitError {}

/// Evaluate whether one exact architecture and device configuration is supported by a
/// measured site evidence packet. No thresholds are calculated here: assessors provide
/// Satisfied / Unsatisfied / Indeterminate against their declared, versioned envelope.
pub fn evaluate_site_fit(
    profile: &SiteFitProfile,
    assessments: &[SiteCapabilityAssessment],
) -> Result<SiteFitResult, SiteFitError> {
    profile.validate()?;

    let required = profile.required_domains();
    let mut seen_domains = HashSet::with_capacity(assessments.len());
    let mut evidence_record_ids = HashSet::with_capacity(assessments.len());

    for assessment in assessments {
        if !required.contains(&assessment.domain) {
            return Err(SiteFitError::UnexpectedAssessment {
                domain: assessment.domain,
            });
        }
        if !seen_domains.insert(assessment.domain) {
            return Err(SiteFitError::DuplicateAssessment {
                domain: assessment.domain,
            });
        }
        if let Some(evidence) = &assessment.evidence {
            evidence.validate_structure()?;
            if !evidence_record_ids.insert(evidence.record_id.clone()) {
                return Err(SiteFitError::DuplicateEvidenceRecord {
                    record_id: evidence.record_id.clone(),
                });
            }
        }
    }

    let mut satisfied_domains = Vec::new();
    let mut unsatisfied_domains = Vec::new();
    let mut indeterminate_issues = Vec::new();

    for domain in required {
        let Some(assessment) = assessments
            .iter()
            .find(|assessment| assessment.domain == domain)
        else {
            indeterminate_issues.push(SiteFitIssue {
                domain,
                kind: SiteFitIssueKind::MissingAssessment,
            });
            continue;
        };

        let Some(evidence) = &assessment.evidence else {
            indeterminate_issues.push(SiteFitIssue {
                domain,
                kind: SiteFitIssueKind::MissingEvidence,
            });
            continue;
        };

        if evidence.status != QualificationEvidenceStatus::Current {
            indeterminate_issues.push(SiteFitIssue {
                domain,
                kind: SiteFitIssueKind::EvidenceNotCurrent(evidence.status),
            });
            continue;
        }

        let expected_authority = domain.expected_authority();
        if evidence.domain != domain {
            indeterminate_issues.push(SiteFitIssue {
                domain,
                kind: SiteFitIssueKind::EvidenceDomainMismatch {
                    actual: evidence.domain,
                },
            });
            continue;
        }
        if evidence.authority != expected_authority {
            indeterminate_issues.push(SiteFitIssue {
                domain,
                kind: SiteFitIssueKind::WrongAuthority {
                    expected: expected_authority,
                    actual: evidence.authority,
                },
            });
            continue;
        }

        let scope_issue = if evidence.site_profile_id != profile.site_profile_id {
            Some(SiteFitIssueKind::SiteProfileMismatch)
        } else if evidence.site_revision != profile.site_revision {
            Some(SiteFitIssueKind::SiteRevisionMismatch)
        } else if evidence.site_conditions_sha256 != profile.site_conditions_sha256 {
            Some(SiteFitIssueKind::SiteConditionsMismatch)
        } else if evidence.architecture_profile_id != profile.architecture_profile_id {
            Some(SiteFitIssueKind::ArchitectureProfileMismatch)
        } else if evidence.architecture != profile.architecture {
            Some(SiteFitIssueKind::ArchitectureMismatch)
        } else if evidence.device_configuration_sha256 != profile.device_configuration_sha256 {
            Some(SiteFitIssueKind::DeviceConfigurationMismatch)
        } else if evidence.requirements_sha256 != profile.requirements_sha256 {
            Some(SiteFitIssueKind::RequirementsMismatch)
        } else {
            let binding = profile
                .method_binding(domain)
                .expect("validated profile has one binding for every required domain");
            if evidence.method_sha256 != binding.method_sha256 {
                Some(SiteFitIssueKind::MethodMismatch)
            } else if evidence.method_conditions_sha256 != binding.method_conditions_sha256 {
                Some(SiteFitIssueKind::MethodConditionsMismatch)
            } else {
                None
            }
        };

        if let Some(kind) = scope_issue {
            indeterminate_issues.push(SiteFitIssue { domain, kind });
            continue;
        }

        match assessment.outcome {
            SiteCapabilityOutcome::Satisfied => satisfied_domains.push(domain),
            SiteCapabilityOutcome::Unsatisfied => unsatisfied_domains.push(domain),
            SiteCapabilityOutcome::Indeterminate => {
                indeterminate_issues.push(SiteFitIssue {
                    domain,
                    kind: SiteFitIssueKind::IndeterminateAssessment,
                });
            }
        }
    }

    let disposition = if !unsatisfied_domains.is_empty() {
        SiteFitDisposition::Ineligible
    } else if !indeterminate_issues.is_empty() {
        SiteFitDisposition::Indeterminate
    } else {
        SiteFitDisposition::Eligible
    };

    Ok(SiteFitResult {
        disposition,
        satisfied_domains,
        unsatisfied_domains,
        indeterminate_issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn profile(architecture: AquaArchitecture) -> SiteFitProfile {
        let required = architecture.baseline_domains();
        SiteFitProfile {
            site_profile_id: "SITE-DALLAS-001".to_owned(),
            site_revision: "survey-r3".to_owned(),
            site_conditions_sha256: digest('a'),
            architecture_profile_id: format!("PROFILE-{architecture:?}"),
            architecture,
            device_configuration_sha256: digest('b'),
            requirements_sha256: digest('c'),
            additional_required_domains: Vec::new(),
            method_bindings: required
                .iter()
                .map(|domain| SiteFitMethodBinding {
                    domain: *domain,
                    method_sha256: digest('d'),
                    method_conditions_sha256: digest('e'),
                })
                .collect(),
        }
    }

    fn evidence(
        domain: SiteCapabilityDomain,
        profile: &SiteFitProfile,
        record_id: &str,
    ) -> SiteFitEvidenceRef {
        let binding = profile
            .method_binding(domain)
            .expect("required site domain has a method binding");
        SiteFitEvidenceRef {
            domain,
            authority: domain.expected_authority(),
            issuer_principal_id: format!("issuer-{domain:?}"),
            authority_verification_receipt_sha256: digest('f'),
            status: QualificationEvidenceStatus::Current,
            record_id: record_id.to_owned(),
            evidence_sha256: digest('9'),
            method_sha256: binding.method_sha256.clone(),
            method_conditions_sha256: binding.method_conditions_sha256.clone(),
            site_profile_id: profile.site_profile_id.clone(),
            site_revision: profile.site_revision.clone(),
            site_conditions_sha256: profile.site_conditions_sha256.clone(),
            architecture_profile_id: profile.architecture_profile_id.clone(),
            architecture: profile.architecture,
            device_configuration_sha256: profile.device_configuration_sha256.clone(),
            requirements_sha256: profile.requirements_sha256.clone(),
        }
    }

    fn assessments(
        profile: &SiteFitProfile,
        outcome: SiteCapabilityOutcome,
    ) -> Vec<SiteCapabilityAssessment> {
        profile
            .required_domains()
            .into_iter()
            .map(|domain| SiteCapabilityAssessment {
                domain,
                outcome,
                evidence: Some(evidence(
                    domain,
                    profile,
                    &format!("site-evidence-{domain:?}"),
                )),
            })
            .collect()
    }

    #[test]
    fn complete_direct_flow_site_is_eligible() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let result =
            evaluate_site_fit(&profile, &assessments(&profile, SiteCapabilityOutcome::Satisfied))
                .expect("valid site fit");
        assert_eq!(result.disposition, SiteFitDisposition::Eligible);
        assert_eq!(result.satisfied_domains.len(), 6);
    }

    #[test]
    fn passive_direct_flow_does_not_require_electrical_or_recirc_hygiene_domains() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let required = profile.required_domains();
        assert!(!required.contains(&SiteCapabilityDomain::ElectricalSupplyAndProtection));
        assert!(!required.contains(&SiteCapabilityDomain::ContinuousPowerAvailability));
        assert!(!required.contains(&SiteCapabilityDomain::WetPathHygieneAndCleaning));
        assert!(!required.contains(&SiteCapabilityDomain::PerInletSupplyCapacity));
    }

    #[test]
    fn same_session_recirculation_requires_power_per_inlet_and_hygiene() {
        let profile = profile(AquaArchitecture::SameSessionRecirculation);
        let required = profile.required_domains();
        assert!(required.contains(&SiteCapabilityDomain::ElectricalSupplyAndProtection));
        assert!(required.contains(&SiteCapabilityDomain::ContinuousPowerAvailability));
        assert!(required.contains(&SiteCapabilityDomain::PerInletSupplyCapacity));
        assert!(required.contains(&SiteCapabilityDomain::WetPathHygieneAndCleaning));
        assert!(required.contains(&SiteCapabilityDomain::BackflowAndCrossConnectionProtection));
    }

    #[test]
    fn passive_dwhR_requires_a_qualified_thermal_receiver() {
        let profile = profile(AquaArchitecture::DirectFlowWithDwhR);
        assert!(profile
            .required_domains()
            .contains(&SiteCapabilityDomain::ThermalReceiverAndHeatSinkCompatibility));
        assert!(!profile
            .required_domains()
            .contains(&SiteCapabilityDomain::ElectricalSupplyAndProtection));
    }

    #[test]
    fn power_modulated_dwhR_requires_electrical_supply_and_continuity() {
        let profile = profile(AquaArchitecture::PowerModulatedDirectFlowWithDwhR);
        let required = profile.required_domains();
        assert!(required.contains(&SiteCapabilityDomain::ElectricalSupplyAndProtection));
        assert!(required.contains(&SiteCapabilityDomain::ContinuousPowerAvailability));
        assert!(required.contains(&SiteCapabilityDomain::ThermalReceiverAndHeatSinkCompatibility));
    }

    #[test]
    fn recirculation_with_dwhR_requires_both_architecture_specific_sets() {
        let profile = profile(AquaArchitecture::SameSessionRecirculationWithDwhR);
        let required = profile.required_domains();
        assert!(required.contains(&SiteCapabilityDomain::WetPathHygieneAndCleaning));
        assert!(required.contains(&SiteCapabilityDomain::ThermalReceiverAndHeatSinkCompatibility));
        assert!(required.contains(&SiteCapabilityDomain::ElectricalSupplyAndProtection));
    }

    #[test]
    fn exact_product_can_add_network_connectivity_as_a_prerequisite() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        profile.additional_required_domains.push(SiteCapabilityDomain::NetworkConnectivity);
        profile.method_bindings.push(SiteFitMethodBinding {
            domain: SiteCapabilityDomain::NetworkConnectivity,
            method_sha256: digest('1'),
            method_conditions_sha256: digest('2'),
        });
        assert_eq!(
            evaluate_site_fit(&profile, &assessments(&profile, SiteCapabilityOutcome::Satisfied))
                .expect("product-specific network prerequisite"),
            SiteFitResult {
                disposition: SiteFitDisposition::Eligible,
                satisfied_domains: profile.required_domains(),
                unsatisfied_domains: Vec::new(),
                indeterminate_issues: Vec::new(),
            }
        );
    }

    #[test]
    fn missing_required_assessment_is_indeterminate_not_eligible() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments.pop();
        let result = evaluate_site_fit(&profile, &assessments).expect("incomplete site record");
        assert_eq!(result.disposition, SiteFitDisposition::Indeterminate);
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::MissingAssessment
        );
    }

    #[test]
    fn missing_evidence_is_indeterminate_not_satisfied() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence = None;
        let result = evaluate_site_fit(&profile, &assessments).expect("missing evidence");
        assert_eq!(result.disposition, SiteFitDisposition::Indeterminate);
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::MissingEvidence
        );
    }

    #[test]
    fn explicit_unsatisfied_prerequisite_makes_architecture_ineligible() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].outcome = SiteCapabilityOutcome::Unsatisfied;
        let result = evaluate_site_fit(&profile, &assessments).expect("valid negative decision");
        assert_eq!(result.disposition, SiteFitDisposition::Ineligible);
        assert_eq!(
            result.unsatisfied_domains,
            vec![SiteCapabilityDomain::DynamicSupplyEnvelope]
        );
    }

    #[test]
    fn indeterminate_assessor_outcome_is_not_promoted_to_satisfied() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].outcome = SiteCapabilityOutcome::Indeterminate;
        let result = evaluate_site_fit(&profile, &assessments).expect("valid unresolved decision");
        assert_eq!(result.disposition, SiteFitDisposition::Indeterminate);
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::IndeterminateAssessment
        );
    }

    #[test]
    fn stale_evidence_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").status =
            QualificationEvidenceStatus::Stale;
        let result = evaluate_site_fit(&profile, &assessments).expect("stale evidence");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::EvidenceNotCurrent(QualificationEvidenceStatus::Stale)
        );
    }

    #[test]
    fn revoked_evidence_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").status =
            QualificationEvidenceStatus::Revoked;
        let result = evaluate_site_fit(&profile, &assessments).expect("revoked evidence");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::EvidenceNotCurrent(QualificationEvidenceStatus::Revoked)
        );
    }

    #[test]
    fn ambiguous_evidence_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").status =
            QualificationEvidenceStatus::Ambiguous;
        let result = evaluate_site_fit(&profile, &assessments).expect("ambiguous evidence");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::EvidenceNotCurrent(QualificationEvidenceStatus::Ambiguous)
        );
    }

    #[test]
    fn evidence_from_another_site_profile_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").site_profile_id =
            "OTHER-SITE".to_owned();
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong site scope");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::SiteProfileMismatch
        );
    }

    #[test]
    fn evidence_from_another_site_revision_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").site_revision =
            "survey-r2".to_owned();
        let result = evaluate_site_fit(&profile, &assessments).expect("old site revision");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::SiteRevisionMismatch
        );
    }

    #[test]
    fn changed_site_condition_digest_invalidates_prior_evidence() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").site_conditions_sha256 = digest('8');
        let result = evaluate_site_fit(&profile, &assessments).expect("changed site conditions");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::SiteConditionsMismatch
        );
    }

    #[test]
    fn evidence_for_another_architecture_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").architecture =
            AquaArchitecture::SameSessionRecirculation;
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong architecture");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::ArchitectureMismatch
        );
    }

    #[test]
    fn evidence_for_another_architecture_profile_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0]
            .evidence
            .as_mut()
            .expect("evidence")
            .architecture_profile_id = "OTHER-PROFILE".to_owned();
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong profile");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::ArchitectureProfileMismatch
        );
    }

    #[test]
    fn evidence_for_another_device_configuration_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").device_configuration_sha256 =
            digest('8');
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong device");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::DeviceConfigurationMismatch
        );
    }

    #[test]
    fn evidence_for_another_requirements_revision_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").requirements_sha256 = digest('8');
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong requirements");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::RequirementsMismatch
        );
    }

    #[test]
    fn evidence_for_another_capability_domain_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").domain =
            SiteCapabilityDomain::WaterChemistryAndScale;
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong evidence domain");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::EvidenceDomainMismatch {
                actual: SiteCapabilityDomain::WaterChemistryAndScale
            }
        );
    }

    #[test]
    fn wrong_authority_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").authority =
            SiteCapabilityAuthority::MycelixWaterQuality;
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong authority");
        assert!(matches!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::WrongAuthority { .. }
        ));
    }

    #[test]
    fn wrong_site_method_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").method_sha256 = digest('8');
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong method");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::MethodMismatch
        );
    }

    #[test]
    fn wrong_method_conditions_are_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0]
            .evidence
            .as_mut()
            .expect("evidence")
            .method_conditions_sha256 = digest('8');
        let result = evaluate_site_fit(&profile, &assessments).expect("wrong method conditions");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::MethodConditionsMismatch
        );
    }

    #[test]
    fn duplicate_assessment_domain_is_rejected() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        let duplicate = assessments[0].clone();
        assessments.push(duplicate);
        assert!(matches!(
            evaluate_site_fit(&profile, &assessments),
            Err(SiteFitError::DuplicateAssessment { .. })
        ));
    }

    #[test]
    fn duplicate_evidence_record_id_is_rejected() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        let record_id = assessments[0]
            .evidence
            .as_ref()
            .expect("evidence")
            .record_id
            .clone();
        assessments[1]
            .evidence
            .as_mut()
            .expect("evidence")
            .record_id = record_id;
        assert!(matches!(
            evaluate_site_fit(&profile, &assessments),
            Err(SiteFitError::DuplicateEvidenceRecord { .. })
        ));
    }

    #[test]
    fn unexpected_assessment_domain_is_rejected() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments.push(SiteCapabilityAssessment {
            domain: SiteCapabilityDomain::NetworkConnectivity,
            outcome: SiteCapabilityOutcome::Satisfied,
            evidence: None,
        });
        assert!(matches!(
            evaluate_site_fit(&profile, &assessments),
            Err(SiteFitError::UnexpectedAssessment {
                domain: SiteCapabilityDomain::NetworkConnectivity
            })
        ));
    }

    #[test]
    fn malformed_evidence_digest_is_rejected() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].evidence.as_mut().expect("evidence").evidence_sha256 =
            "not-a-digest".to_owned();
        assert_eq!(
            evaluate_site_fit(&profile, &assessments),
            Err(SiteFitError::InvalidDigest {
                field: "evidence.evidence_sha256"
            })
        );
    }

    #[test]
    fn malformed_authority_verification_receipt_is_rejected() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0]
            .evidence
            .as_mut()
            .expect("evidence")
            .authority_verification_receipt_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            evaluate_site_fit(&profile, &assessments),
            Err(SiteFitError::InvalidDigest {
                field: "evidence.authority_verification_receipt_sha256"
            })
        );
    }

    #[test]
    fn duplicate_additional_requirement_is_rejected() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        profile
            .additional_required_domains
            .push(SiteCapabilityDomain::NetworkConnectivity);
        profile
            .additional_required_domains
            .push(SiteCapabilityDomain::NetworkConnectivity);
        assert_eq!(
            profile.validate(),
            Err(SiteFitError::DuplicateAdditionalDomain {
                domain: SiteCapabilityDomain::NetworkConnectivity
            })
        );
    }

    #[test]
    fn additional_requirement_cannot_duplicate_architecture_baseline() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        profile
            .additional_required_domains
            .push(SiteCapabilityDomain::DynamicSupplyEnvelope);
        assert_eq!(
            profile.validate(),
            Err(SiteFitError::DuplicateAdditionalDomain {
                domain: SiteCapabilityDomain::DynamicSupplyEnvelope
            })
        );
    }

    #[test]
    fn missing_method_binding_is_rejected_in_profile() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        profile.method_bindings.pop();
        assert!(matches!(
            profile.validate(),
            Err(SiteFitError::MissingMethodBinding { .. })
        ));
    }

    #[test]
    fn duplicate_method_binding_is_rejected_in_profile() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        let first_binding = profile.method_bindings[0].clone();
        profile.method_bindings.push(first_binding);
        assert!(matches!(
            profile.validate(),
            Err(SiteFitError::DuplicateMethodBinding { .. })
        ));
    }

    #[test]
    fn out_of_scope_method_binding_is_rejected_in_profile() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        profile.method_bindings.push(SiteFitMethodBinding {
            domain: SiteCapabilityDomain::NetworkConnectivity,
            method_sha256: digest('1'),
            method_conditions_sha256: digest('2'),
        });
        assert!(matches!(
            profile.validate(),
            Err(SiteFitError::UnexpectedMethodBinding { .. })
        ));
    }

    #[test]
    fn malformed_site_profile_digest_is_rejected() {
        let mut profile = profile(AquaArchitecture::DirectFlowPassive);
        profile.site_conditions_sha256 = "invalid".to_owned();
        assert_eq!(
            profile.validate(),
            Err(SiteFitError::InvalidDigest {
                field: "site_conditions_sha256"
            })
        );
    }

    #[test]
    fn explicit_ineligibility_dominates_other_indeterminate_capabilities() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].outcome = SiteCapabilityOutcome::Unsatisfied;
        assessments[1].evidence.as_mut().expect("evidence").status =
            QualificationEvidenceStatus::Stale;
        let result = evaluate_site_fit(&profile, &assessments).expect("mixed site evidence");
        assert_eq!(result.disposition, SiteFitDisposition::Ineligible);
        assert_eq!(result.unsatisfied_domains.len(), 1);
        assert_eq!(result.indeterminate_issues.len(), 1);
    }

    #[test]
    fn one_site_can_be_eligible_while_another_is_ineligible() {
        let profile_a = profile(AquaArchitecture::DirectFlowPassive);
        let mut profile_b = profile_a.clone();
        profile_b.site_profile_id = "SITE-LOW-HEAD-002".to_owned();
        profile_b.site_revision = "survey-r1".to_owned();
        profile_b.site_conditions_sha256 = digest('8');

        let evidence_a = assessments(&profile_a, SiteCapabilityOutcome::Satisfied);
        let mut evidence_b = assessments(&profile_b, SiteCapabilityOutcome::Satisfied);
        evidence_b[0].outcome = SiteCapabilityOutcome::Unsatisfied;

        let result_a = evaluate_site_fit(&profile_a, &evidence_a).expect("site A");
        let result_b = evaluate_site_fit(&profile_b, &evidence_b).expect("site B");
        assert_eq!(result_a.disposition, SiteFitDisposition::Eligible);
        assert_eq!(result_b.disposition, SiteFitDisposition::Ineligible);
    }

    #[test]
    fn operating_environment_is_not_imposed_on_every_architecture() {
        let direct = profile(AquaArchitecture::DirectFlowPassive);
        let recirculation = profile(AquaArchitecture::SameSessionRecirculation);
        assert!(!direct
            .required_domains()
            .contains(&SiteCapabilityDomain::OperatingEnvironmentAndFreezeProtection));
        assert!(!recirculation
            .required_domains()
            .contains(&SiteCapabilityDomain::OperatingEnvironmentAndFreezeProtection));
    }

    #[test]
    fn exact_recirculation_product_can_add_ambient_and_freeze_protection() {
        let mut orbital_profile = profile(AquaArchitecture::SameSessionRecirculation);
        orbital_profile
            .additional_required_domains
            .push(SiteCapabilityDomain::OperatingEnvironmentAndFreezeProtection);
        orbital_profile.method_bindings.push(SiteFitMethodBinding {
            domain: SiteCapabilityDomain::OperatingEnvironmentAndFreezeProtection,
            method_sha256: digest('3'),
            method_conditions_sha256: digest('4'),
        });

        let result = evaluate_site_fit(
            &orbital_profile,
            &assessments(&orbital_profile, SiteCapabilityOutcome::Satisfied),
        )
        .expect("exact product ambient prerequisite");
        assert_eq!(result.disposition, SiteFitDisposition::Eligible);
        assert!(result.satisfied_domains.contains(
            &SiteCapabilityDomain::OperatingEnvironmentAndFreezeProtection
        ));
    }

    #[test]
    fn network_connectivity_is_required_only_when_exact_profile_adds_it() {
        let direct = profile(AquaArchitecture::DirectFlowPassive);
        assert!(!direct
            .required_domains()
            .contains(&SiteCapabilityDomain::NetworkConnectivity));

        let mut connected_product = profile(AquaArchitecture::SameSessionRecirculation);
        connected_product
            .additional_required_domains
            .push(SiteCapabilityDomain::NetworkConnectivity);
        connected_product.method_bindings.push(SiteFitMethodBinding {
            domain: SiteCapabilityDomain::NetworkConnectivity,
            method_sha256: digest('1'),
            method_conditions_sha256: digest('2'),
        });
        assert!(connected_product
            .required_domains()
            .contains(&SiteCapabilityDomain::NetworkConnectivity));
    }

    #[test]
    fn all_architecture_profiles_have_distinct_required_domain_sets() {
        let architectures = [
            AquaArchitecture::DirectFlowPassive,
            AquaArchitecture::DirectFlowWithDwhR,
            AquaArchitecture::PowerModulatedDirectFlowWithDwhR,
            AquaArchitecture::SameSessionRecirculation,
            AquaArchitecture::SameSessionRecirculationWithDwhR,
        ];
        let sets = architectures
            .iter()
            .map(|architecture| {
                let mut domains = architecture.baseline_domains().to_vec();
                domains.sort_by_key(|domain| format!("{domain:?}"));
                domains
            })
            .collect::<Vec<_>>();
        for left in 0..sets.len() {
            for right in (left + 1)..sets.len() {
                assert_ne!(sets[left], sets[right]);
            }
        }
    }

    #[test]
    fn empty_assessment_packet_is_indeterminate() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let result = evaluate_site_fit(&profile, &[]).expect("empty site evidence");
        assert_eq!(result.disposition, SiteFitDisposition::Indeterminate);
        assert_eq!(result.indeterminate_issues.len(), profile.required_domains().len());
    }

    #[test]
    fn current_evidence_with_indeterminate_owner_result_is_not_eligible() {
        let profile = profile(AquaArchitecture::DirectFlowPassive);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0].outcome = SiteCapabilityOutcome::Indeterminate;
        let result = evaluate_site_fit(&profile, &assessments).expect("indeterminate owner result");
        assert_eq!(result.disposition, SiteFitDisposition::Indeterminate);
    }

    #[test]
    fn wrong_site_record_cannot_be_reused_for_another_device_revision() {
        let profile = profile(AquaArchitecture::SameSessionRecirculation);
        let mut assessments = assessments(&profile, SiteCapabilityOutcome::Satisfied);
        assessments[0]
            .evidence
            .as_mut()
            .expect("evidence")
            .device_configuration_sha256 = digest('7');
        let result = evaluate_site_fit(&profile, &assessments).expect("different recirculation device");
        assert_eq!(
            result.indeterminate_issues[0].kind,
            SiteFitIssueKind::DeviceConfigurationMismatch
        );
    }
}
