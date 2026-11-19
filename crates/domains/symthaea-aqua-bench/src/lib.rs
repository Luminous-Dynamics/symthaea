//! Product-level benchmark adapter for Luminous Aqua.
//!
//! This crate validates matched test context and binds an Aqua run to canonical owner
//! evidence. It deliberately does not calculate water/energy ledgers, hydraulic or
//! thermal states, water quality, metrology admission, safety, hygiene, or comfort.
//! Symtropy, Mycelix, Sol-Atlas, and independent endpoint assessors remain authoritative
//! for those results. Missing evidence is an error; it must never be replaced by zero.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub mod pressure_curve;
pub mod service_continuity;
pub mod serviceability;
pub mod service_qualification;
pub mod site_fit;
pub mod spray_field;
pub mod standard_methods;
pub mod task_phase;

pub const CONTRACT_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceOwner {
    SymtropyWaterServiceT0,
    SymtropyResourceClosure,
    MycelixMetrology,
    MycelixWaterQuality,
    IndependentTaskOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceRef {
    pub owner: EvidenceOwner,
    pub record_id: String,
    pub sha256: String,
}

impl EvidenceRef {
    fn validate(
        &self,
        field: &'static str,
        expected_owner: EvidenceOwner,
    ) -> Result<(), ContractError> {
        if self.owner != expected_owner {
            return Err(ContractError::WrongEvidenceOwner { field });
        }
        if self.record_id.trim().is_empty() {
            return Err(ContractError::MissingField { field });
        }
        if !is_sha256(&self.sha256) {
            return Err(ContractError::InvalidDigest { field });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    MissingField { field: &'static str },
    InvalidDigest { field: &'static str },
    WrongEvidenceOwner { field: &'static str },
    ZeroDuration,
    SameRunCompared,
    ContextMismatch { field: &'static str },
    MissingEnergyReceipt,
}

impl Display for ContractError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing required Aqua field: {field}"),
            Self::InvalidDigest { field } => write!(f, "invalid SHA-256 digest for: {field}"),
            Self::WrongEvidenceOwner { field } => {
                write!(f, "evidence owner mismatch for: {field}")
            }
            Self::ZeroDuration => write!(f, "task duration must be greater than zero"),
            Self::SameRunCompared => write!(f, "a run cannot be compared with itself"),
            Self::ContextMismatch { field } => {
                write!(f, "matched benchmark context differs at: {field}")
            }
            Self::MissingEnergyReceipt => {
                write!(
                    f,
                    "energy comparison requires a Symtropy resource-closure receipt"
                )
            }
        }
    }
}

impl Error for ContractError {}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), ContractError> {
    if value.trim().is_empty() {
        return Err(ContractError::MissingField { field });
    }
    Ok(())
}

fn require_digest(field: &'static str, value: &str) -> Result<(), ContractError> {
    if !is_sha256(value) {
        return Err(ContractError::InvalidDigest { field });
    }
    Ok(())
}

/// Conditions that must be identical for a matched control/candidate comparison.
///
/// Digests bind the full versioned profiles and their source data. The adapter checks
/// equality and syntax only; it does not validate the underlying physical observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkConditions {
    pub task_profile_id: String,
    /// Canonical digest of the complete matched trial condition bundle consumed by
    /// independent endpoint assessments, including the exact task and shared conditions.
    pub conditions_bundle_sha256: String,
    pub duration_seconds: u32,
    pub inlet_water_profile_sha256: String,
    pub delivery_conditions_sha256: String,
    pub room_conditions_sha256: String,
    pub measurement_protocol_sha256: String,
    pub accounting_boundary_sha256: String,
}

impl BenchmarkConditions {
    pub fn validate(&self) -> Result<(), ContractError> {
        require_nonempty("task_profile_id", &self.task_profile_id)?;
        require_digest(
            "conditions_bundle_sha256",
            &self.conditions_bundle_sha256,
        )?;
        if self.duration_seconds == 0 {
            return Err(ContractError::ZeroDuration);
        }
        require_digest(
            "inlet_water_profile_sha256",
            &self.inlet_water_profile_sha256,
        )?;
        require_digest(
            "delivery_conditions_sha256",
            &self.delivery_conditions_sha256,
        )?;
        require_digest("room_conditions_sha256", &self.room_conditions_sha256)?;
        require_digest(
            "measurement_protocol_sha256",
            &self.measurement_protocol_sha256,
        )?;
        require_digest(
            "accounting_boundary_sha256",
            &self.accounting_boundary_sha256,
        )?;
        Ok(())
    }

    fn compare_with(&self, other: &Self) -> Result<(), ContractError> {
        if self.task_profile_id != other.task_profile_id {
            return Err(ContractError::ContextMismatch {
                field: "task_profile_id",
            });
        }
        if self.conditions_bundle_sha256 != other.conditions_bundle_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "conditions_bundle_sha256",
            });
        }
        if self.duration_seconds != other.duration_seconds {
            return Err(ContractError::ContextMismatch {
                field: "duration_seconds",
            });
        }
        if self.inlet_water_profile_sha256 != other.inlet_water_profile_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "inlet_water_profile_sha256",
            });
        }
        if self.delivery_conditions_sha256 != other.delivery_conditions_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "delivery_conditions_sha256",
            });
        }
        if self.room_conditions_sha256 != other.room_conditions_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "room_conditions_sha256",
            });
        }
        if self.measurement_protocol_sha256 != other.measurement_protocol_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "measurement_protocol_sha256",
            });
        }
        if self.accounting_boundary_sha256 != other.accounting_boundary_sha256 {
            return Err(ContractError::ContextMismatch {
                field: "accounting_boundary_sha256",
            });
        }
        Ok(())
    }
}

/// Device-specific identity. Configuration and cleaning protocol may differ between
/// comparators; they must be recorded and bound to each run, not forced to be identical.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceConfiguration {
    pub market_variant_id: String,
    /// Canonical digest covering market variant, hardware, installation, cleaning and firmware.
    pub device_configuration_sha256: String,
    pub hardware_configuration_sha256: String,
    pub installation_sha256: String,
    pub cleaning_protocol_sha256: String,
    pub firmware_revision: Option<String>,
}

impl DeviceConfiguration {
    pub fn validate(&self) -> Result<(), ContractError> {
        require_nonempty("market_variant_id", &self.market_variant_id)?;
        require_digest(
            "device_configuration_sha256",
            &self.device_configuration_sha256,
        )?;
        require_digest(
            "hardware_configuration_sha256",
            &self.hardware_configuration_sha256,
        )?;
        require_digest("installation_sha256", &self.installation_sha256)?;
        require_digest("cleaning_protocol_sha256", &self.cleaning_protocol_sha256)?;
        if let Some(revision) = &self.firmware_revision {
            require_nonempty("firmware_revision", revision)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AquaRunEvidence {
    pub water_service_t0: EvidenceRef,
    pub metrology_admission: EvidenceRef,
    pub water_quality_assessment: EvidenceRef,
    pub task_outcome_assessment: EvidenceRef,
    pub energy_resource_closure: Option<EvidenceRef>,
}

impl AquaRunEvidence {
    pub fn validate(&self, scope: ComparisonScope) -> Result<(), ContractError> {
        self.water_service_t0
            .validate("water_service_t0", EvidenceOwner::SymtropyWaterServiceT0)?;
        self.metrology_admission
            .validate("metrology_admission", EvidenceOwner::MycelixMetrology)?;
        self.water_quality_assessment.validate(
            "water_quality_assessment",
            EvidenceOwner::MycelixWaterQuality,
        )?;
        self.task_outcome_assessment.validate(
            "task_outcome_assessment",
            EvidenceOwner::IndependentTaskOutcome,
        )?;

        match (&self.energy_resource_closure, scope) {
            (Some(receipt), _) => receipt.validate(
                "energy_resource_closure",
                EvidenceOwner::SymtropyResourceClosure,
            )?,
            (None, ComparisonScope::WaterAndEnergy) => {
                return Err(ContractError::MissingEnergyReceipt);
            }
            (None, ComparisonScope::WaterServiceOnly) => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonScope {
    WaterServiceOnly,
    WaterAndEnergy,
}

/// One Aqua test run bound to canonical subsystem and independent outcome evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AquaRunRecord {
    pub run_id: String,
    pub conditions: BenchmarkConditions,
    pub device: DeviceConfiguration,
    pub evidence: AquaRunEvidence,
}

impl AquaRunRecord {
    pub fn validate(&self, scope: ComparisonScope) -> Result<(), ContractError> {
        require_nonempty("run_id", &self.run_id)?;
        self.conditions.validate()?;
        self.device.validate()?;
        self.evidence.validate(scope)
    }
}

/// Check whether two independently evidenced runs are suitable for a matched comparison.
///
/// This returns no performance score and does not certify non-inferiority. It only checks
/// that the task, duration, supply, room, measurement protocol, and accounting boundary
/// match, and that both runs carry the required owner receipts for the requested scope.
pub fn validate_matched_comparison(
    control: &AquaRunRecord,
    candidate: &AquaRunRecord,
    scope: ComparisonScope,
) -> Result<(), ContractError> {
    control.validate(scope)?;
    candidate.validate(scope)?;
    if control.run_id == candidate.run_id {
        return Err(ContractError::SameRunCompared);
    }
    control.conditions.compare_with(&candidate.conditions)
}

/// Error returned by the composed matched-run and service-qualification gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonSide {
    Control,
    Candidate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualifiedComparisonError {
    Benchmark(ContractError),
    ServiceQualification(crate::service_qualification::ServiceQualificationError),
    SiteFit(crate::site_fit::SiteFitError),
    SiteContextMismatch { field: &'static str },
    SiteNotEligible {
        side: ComparisonSide,
        result: crate::site_fit::SiteFitResult,
    },
}

impl Display for QualifiedComparisonError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Benchmark(error) => write!(f, "matched benchmark evidence failed: {error}"),
            Self::ServiceQualification(error) => {
                write!(f, "completed-service qualification failed: {error}")
            }
            Self::SiteFit(error) => write!(f, "site-fit evidence failed: {error}"),
            Self::SiteContextMismatch { field } => {
                write!(f, "control/candidate site profiles differ: {field}")
            }
            Self::SiteNotEligible { side, result } => write!(
                f,
                "{side:?} installation is not eligible: {:?} ({} unresolved domain(s), {} unsatisfied domain(s))",
                result.disposition,
                result.indeterminate_issues.len(),
                result.unsatisfied_domains.len()
            ),
        }
    }
}

impl Error for QualifiedComparisonError {}

impl From<ContractError> for QualifiedComparisonError {
    fn from(error: ContractError) -> Self {
        Self::Benchmark(error)
    }
}

impl From<crate::service_qualification::ServiceQualificationError> for QualifiedComparisonError {
    fn from(error: crate::service_qualification::ServiceQualificationError) -> Self {
        Self::ServiceQualification(error)
    }
}

impl From<crate::site_fit::SiteFitError> for QualifiedComparisonError {
    fn from(error: crate::site_fit::SiteFitError) -> Self {
        Self::SiteFit(error)
    }
}

/// Compose canonical Aqua run-receipt validation with independent completed-service decisions.
///
/// This is the preferred comparison entry point for product claims. It first validates the
/// matched task/resource context and canonical subsystem receipts, then requires both the control
/// and candidate to have complete, current, domain-specific independent assessments. It returns
/// no physical score and does not calculate non-inferiority or superiority.
pub fn validate_qualified_matched_comparison(
    control: &AquaRunRecord,
    candidate: &AquaRunRecord,
    scope: ComparisonScope,
    qualification_profile: &crate::service_qualification::ServiceQualificationProfile,
    control_qualification: &crate::service_qualification::ServiceQualificationRecord,
    candidate_qualification: &crate::service_qualification::ServiceQualificationRecord,
) -> Result<
    crate::service_qualification::MatchedServiceQualificationResult,
    QualifiedComparisonError,
> {
    validate_matched_comparison(control, candidate, scope)?;
    qualification_profile
        .validate()
        .map_err(QualifiedComparisonError::ServiceQualification)?;
    if qualification_profile.conditions_sha256 != control.conditions.conditions_bundle_sha256
        || qualification_profile.conditions_sha256
            != candidate.conditions.conditions_bundle_sha256
    {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "qualification_conditions_sha256",
            },
        ));
    }
    if control_qualification.run_id != control.run_id {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "control_qualification.run_id",
            },
        ));
    }
    if candidate_qualification.run_id != candidate.run_id {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "candidate_qualification.run_id",
            },
        ));
    }
    if control_qualification.device_configuration_sha256
        != control.device.device_configuration_sha256
    {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "control_device_configuration_sha256",
            },
        ));
    }
    if candidate_qualification.device_configuration_sha256
        != candidate.device.device_configuration_sha256
    {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "candidate_device_configuration_sha256",
            },
        ));
    }
    crate::service_qualification::evaluate_matched_service_qualification(
        qualification_profile,
        control_qualification,
        candidate_qualification,
    )
    .map_err(QualifiedComparisonError::from)
}

/// Result of a matched service comparison after both exact installations pass site fit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteQualifiedComparisonResult {
    pub control_site_fit: crate::site_fit::SiteFitResult,
    pub candidate_site_fit: crate::site_fit::SiteFitResult,
    pub service_qualification: crate::service_qualification::MatchedServiceQualificationResult,
}

/// Preferred site-aware product comparison entry point.
///
/// Both control and candidate must be Eligible at the same surveyed site revision.
/// Their product/architecture configurations may differ, but each site's evidence must bind
/// to that run's exact device digest. Only then are canonical run receipts and independent
/// completed-service decisions evaluated. This function performs no site engineering.
pub fn validate_site_qualified_matched_comparison(
    control: &AquaRunRecord,
    candidate: &AquaRunRecord,
    scope: ComparisonScope,
    qualification_profile: &crate::service_qualification::ServiceQualificationProfile,
    control_qualification: &crate::service_qualification::ServiceQualificationRecord,
    candidate_qualification: &crate::service_qualification::ServiceQualificationRecord,
    control_site_profile: &crate::site_fit::SiteFitProfile,
    control_site_assessments: &[crate::site_fit::SiteCapabilityAssessment],
    candidate_site_profile: &crate::site_fit::SiteFitProfile,
    candidate_site_assessments: &[crate::site_fit::SiteCapabilityAssessment],
) -> Result<SiteQualifiedComparisonResult, QualifiedComparisonError> {
    if control_site_profile.site_profile_id != candidate_site_profile.site_profile_id {
        return Err(QualifiedComparisonError::SiteContextMismatch {
            field: "site_profile_id",
        });
    }
    if control_site_profile.site_revision != candidate_site_profile.site_revision {
        return Err(QualifiedComparisonError::SiteContextMismatch {
            field: "site_revision",
        });
    }
    if control_site_profile.site_conditions_sha256
        != candidate_site_profile.site_conditions_sha256
    {
        return Err(QualifiedComparisonError::SiteContextMismatch {
            field: "site_conditions_sha256",
        });
    }
    if control_site_profile.device_configuration_sha256
        != control.device.device_configuration_sha256
    {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "control_site_device_configuration_sha256",
            },
        ));
    }
    if candidate_site_profile.device_configuration_sha256
        != candidate.device.device_configuration_sha256
    {
        return Err(QualifiedComparisonError::Benchmark(
            ContractError::ContextMismatch {
                field: "candidate_site_device_configuration_sha256",
            },
        ));
    }

    let control_site_fit =
        crate::site_fit::evaluate_site_fit(control_site_profile, control_site_assessments)?;
    if control_site_fit.disposition != crate::site_fit::SiteFitDisposition::Eligible {
        return Err(QualifiedComparisonError::SiteNotEligible {
            side: ComparisonSide::Control,
            result: control_site_fit,
        });
    }

    let candidate_site_fit =
        crate::site_fit::evaluate_site_fit(candidate_site_profile, candidate_site_assessments)?;
    if candidate_site_fit.disposition != crate::site_fit::SiteFitDisposition::Eligible {
        return Err(QualifiedComparisonError::SiteNotEligible {
            side: ComparisonSide::Candidate,
            result: candidate_site_fit,
        });
    }

    let service_qualification = validate_qualified_matched_comparison(
        control,
        candidate,
        scope,
        qualification_profile,
        control_qualification,
        candidate_qualification,
    )?;

    Ok(SiteQualifiedComparisonResult {
        control_site_fit,
        candidate_site_fit,
        service_qualification,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn evidence(owner: EvidenceOwner, id: &str, byte: char) -> EvidenceRef {
        EvidenceRef {
            owner,
            record_id: id.to_owned(),
            sha256: digest(byte),
        }
    }

    fn run(run_id: &str, market_variant_id: &str, cleaning_digest: char) -> AquaRunRecord {
        AquaRunRecord {
            run_id: run_id.to_owned(),
            conditions: BenchmarkConditions {
                task_profile_id: "STANDARD_SHOWER".to_owned(),
                conditions_bundle_sha256: digest('b'),
                duration_seconds: 600,
                inlet_water_profile_sha256: digest('a'),
                delivery_conditions_sha256: digest('b'),
                room_conditions_sha256: digest('c'),
                measurement_protocol_sha256: digest('d'),
                accounting_boundary_sha256: digest('e'),
            },
            device: DeviceConfiguration {
                market_variant_id: market_variant_id.to_owned(),
                device_configuration_sha256: digest('6'),
                hardware_configuration_sha256: digest('f'),
                installation_sha256: digest('1'),
                cleaning_protocol_sha256: digest(cleaning_digest),
                firmware_revision: Some("1.0.0".to_owned()),
            },
            evidence: AquaRunEvidence {
                water_service_t0: evidence(EvidenceOwner::SymtropyWaterServiceT0, "water-run", '2'),
                metrology_admission: evidence(
                    EvidenceOwner::MycelixMetrology,
                    "metrology-run",
                    '3',
                ),
                water_quality_assessment: evidence(
                    EvidenceOwner::MycelixWaterQuality,
                    "quality-run",
                    '4',
                ),
                task_outcome_assessment: evidence(
                    EvidenceOwner::IndependentTaskOutcome,
                    "outcome-run",
                    '5',
                ),
                energy_resource_closure: Some(evidence(
                    EvidenceOwner::SymtropyResourceClosure,
                    "energy-run",
                    '6',
                )),
            },
        }
    }

    fn qualification_profile(
        claim: crate::service_qualification::QualificationClaim,
    ) -> crate::service_qualification::ServiceQualificationProfile {
        use crate::service_qualification::QualificationMethodBinding;
        crate::service_qualification::ServiceQualificationProfile {
            profile_id: "AQS1-QUALIFICATION-001".to_owned(),
            claim,
            protocol_sha256: digest('a'),
            conditions_sha256: digest('b'),
            method_bindings: claim
                .required_domains()
                .iter()
                .map(|domain| QualificationMethodBinding {
                    domain: *domain,
                    method_sha256: digest('c'),
                    conditions_sha256: digest('d'),
                })
                .collect(),
        }
    }

    fn service_qualification_record(
        run_id: &str,
        profile: &crate::service_qualification::ServiceQualificationProfile,
        outcome: crate::service_qualification::AssessmentOutcome,
    ) -> crate::service_qualification::ServiceQualificationRecord {
        use crate::service_qualification::{
            DomainAssessment, QualificationEvidenceRef, ServiceQualificationRecord,
        };

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
                .map(|domain| {
                    let binding = profile
                        .method_bindings
                        .iter()
                        .find(|binding| binding.domain == *domain)
                        .expect("domain method binding");
                    DomainAssessment {
                        domain: *domain,
                        outcome,
                        evidence: QualificationEvidenceRef {
                            authority: domain.expected_authority(),
                            run_id: run_id.to_owned(),
                            issuer_principal_id: format!("issuer-{domain:?}"),
                            authority_verification_receipt_sha256: digest('e'),
                            status: crate::service_qualification::QualificationEvidenceStatus::Current,
                            record_id: format!("{run_id}-{domain:?}"),
                            evidence_sha256: digest('f'),
                            method_sha256: binding.method_sha256.clone(),
                            method_conditions_sha256: binding.conditions_sha256.clone(),
                            trial_conditions_sha256: profile.conditions_sha256.clone(),
                        },
                    }
                })
                .collect(),
        }
    }

    #[test]
    fn rejects_different_canonical_condition_bundles() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.conditions_bundle_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly
            ),
            Err(ContractError::ContextMismatch {
                field: "conditions_bundle_sha256"
            })
        );
    }

    #[test]
    fn rejects_invalid_device_configuration_bundle_digest() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.device.device_configuration_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::InvalidDigest {
                field: "device_configuration_sha256"
            })
        );
    }

    #[test]
    fn allows_matched_task_conditions_with_different_device_and_cleaning_profiles() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterAndEnergy,),
            Ok(())
        );
    }

    fn site_fit_profile(
        architecture: crate::site_fit::AquaArchitecture,
        device_configuration_sha256: &str,
    ) -> crate::site_fit::SiteFitProfile {
        use crate::site_fit::{SiteFitMethodBinding, SiteFitProfile};
        let mut profile = SiteFitProfile {
            site_profile_id: "SITE-SHARED-001".to_owned(),
            site_revision: "survey-r4".to_owned(),
            site_conditions_sha256: digest('9'),
            architecture_profile_id: format!("SITE-PROFILE-{architecture:?}"),
            architecture,
            device_configuration_sha256: device_configuration_sha256.to_owned(),
            requirements_sha256: digest('8'),
            additional_required_domains: Vec::new(),
            method_bindings: Vec::new(),
        };
        profile.method_bindings = profile
            .required_domains()
            .into_iter()
            .map(|domain| SiteFitMethodBinding {
                domain,
                method_sha256: digest('c'),
                method_conditions_sha256: digest('d'),
            })
            .collect();
        profile
    }

    fn site_fit_assessments(
        profile: &crate::site_fit::SiteFitProfile,
        outcome: crate::site_fit::SiteCapabilityOutcome,
    ) -> Vec<crate::site_fit::SiteCapabilityAssessment> {
        use crate::site_fit::{SiteCapabilityAssessment, SiteFitEvidenceRef};
        profile
            .required_domains()
            .into_iter()
            .map(|domain| {
                let binding = profile
                    .method_bindings
                    .iter()
                    .find(|binding| binding.domain == domain)
                    .expect("site capability method binding");
                SiteCapabilityAssessment {
                    domain,
                    outcome,
                    evidence: Some(SiteFitEvidenceRef {
                        domain,
                        authority: domain.expected_authority(),
                        issuer_principal_id: format!("site-issuer-{domain:?}"),
                        authority_verification_receipt_sha256: digest('e'),
                        status: crate::service_qualification::QualificationEvidenceStatus::Current,
                        record_id: format!("site-evidence-{domain:?}"),
                        evidence_sha256: digest('f'),
                        method_sha256: binding.method_sha256.clone(),
                        method_conditions_sha256: binding.method_conditions_sha256.clone(),
                        site_profile_id: profile.site_profile_id.clone(),
                        site_revision: profile.site_revision.clone(),
                        site_conditions_sha256: profile.site_conditions_sha256.clone(),
                        architecture_profile_id: profile.architecture_profile_id.clone(),
                        architecture: profile.architecture,
                        device_configuration_sha256: profile.device_configuration_sha256.clone(),
                        requirements_sha256: profile.requirements_sha256.clone(),
                    }),
                }
            })
            .collect()
    }

    fn site_fit_profile(
        architecture: crate::site_fit::AquaArchitecture,
        device_configuration_sha256: &str,
    ) -> crate::site_fit::SiteFitProfile {
        use crate::site_fit::{SiteFitMethodBinding, SiteFitProfile};
        let mut profile = SiteFitProfile {
            site_profile_id: "SITE-SHARED-001".to_owned(),
            site_revision: "survey-r4".to_owned(),
            site_conditions_sha256: digest('9'),
            architecture_profile_id: format!("SITE-PROFILE-{architecture:?}"),
            architecture,
            device_configuration_sha256: device_configuration_sha256.to_owned(),
            requirements_sha256: digest('8'),
            additional_required_domains: Vec::new(),
            method_bindings: Vec::new(),
        };
        profile.method_bindings = profile
            .required_domains()
            .into_iter()
            .map(|domain| SiteFitMethodBinding {
                domain,
                method_sha256: digest('c'),
                method_conditions_sha256: digest('d'),
            })
            .collect();
        profile
    }

    fn site_fit_assessments(
        profile: &crate::site_fit::SiteFitProfile,
        outcome: crate::site_fit::SiteCapabilityOutcome,
    ) -> Vec<crate::site_fit::SiteCapabilityAssessment> {
        use crate::site_fit::{SiteCapabilityAssessment, SiteFitEvidenceRef};
        profile
            .required_domains()
            .into_iter()
            .map(|domain| {
                let binding = profile
                    .method_bindings
                    .iter()
                    .find(|binding| binding.domain == domain)
                    .expect("site capability method binding");
                SiteCapabilityAssessment {
                    domain,
                    outcome,
                    evidence: Some(SiteFitEvidenceRef {
                        domain,
                        authority: domain.expected_authority(),
                        issuer_principal_id: format!("site-issuer-{domain:?}"),
                        authority_verification_receipt_sha256: digest('e'),
                        status: crate::service_qualification::QualificationEvidenceStatus::Current,
                        record_id: format!("site-evidence-{domain:?}"),
                        evidence_sha256: digest('f'),
                        method_sha256: binding.method_sha256.clone(),
                        method_conditions_sha256: binding.method_conditions_sha256.clone(),
                        site_profile_id: profile.site_profile_id.clone(),
                        site_revision: profile.site_revision.clone(),
                        site_conditions_sha256: profile.site_conditions_sha256.clone(),
                        architecture_profile_id: profile.architecture_profile_id.clone(),
                        architecture: profile.architecture,
                        device_configuration_sha256: profile.device_configuration_sha256.clone(),
                        requirements_sha256: profile.requirements_sha256.clone(),
                    }),
                }
            })
            .collect()
    }

    fn site_fit_profile(
        architecture: crate::site_fit::AquaArchitecture,
        device_configuration_sha256: &str,
    ) -> crate::site_fit::SiteFitProfile {
        use crate::site_fit::{SiteFitMethodBinding, SiteFitProfile};
        let mut profile = SiteFitProfile {
            site_profile_id: "SITE-SHARED-001".to_owned(),
            site_revision: "survey-r4".to_owned(),
            site_conditions_sha256: digest('9'),
            architecture_profile_id: format!("SITE-PROFILE-{architecture:?}"),
            architecture,
            device_configuration_sha256: device_configuration_sha256.to_owned(),
            requirements_sha256: digest('8'),
            additional_required_domains: Vec::new(),
            method_bindings: Vec::new(),
        };
        profile.method_bindings = profile
            .required_domains()
            .into_iter()
            .map(|domain| SiteFitMethodBinding {
                domain,
                method_sha256: digest('c'),
                method_conditions_sha256: digest('d'),
            })
            .collect();
        profile
    }

    fn site_fit_assessments(
        profile: &crate::site_fit::SiteFitProfile,
        outcome: crate::site_fit::SiteCapabilityOutcome,
    ) -> Vec<crate::site_fit::SiteCapabilityAssessment> {
        use crate::site_fit::{SiteCapabilityAssessment, SiteFitEvidenceRef};
        profile
            .required_domains()
            .into_iter()
            .map(|domain| {
                let binding = profile
                    .method_bindings
                    .iter()
                    .find(|binding| binding.domain == domain)
                    .expect("site capability method binding");
                SiteCapabilityAssessment {
                    domain,
                    outcome,
                    evidence: Some(SiteFitEvidenceRef {
                        domain,
                        authority: domain.expected_authority(),
                        issuer_principal_id: format!("site-issuer-{domain:?}"),
                        authority_verification_receipt_sha256: digest('e'),
                        status: crate::service_qualification::QualificationEvidenceStatus::Current,
                        record_id: format!("site-evidence-{domain:?}"),
                        evidence_sha256: digest('f'),
                        method_sha256: binding.method_sha256.clone(),
                        method_conditions_sha256: binding.method_conditions_sha256.clone(),
                        site_profile_id: profile.site_profile_id.clone(),
                        site_revision: profile.site_revision.clone(),
                        site_conditions_sha256: profile.site_conditions_sha256.clone(),
                        architecture_profile_id: profile.architecture_profile_id.clone(),
                        architecture: profile.architecture,
                        device_configuration_sha256: profile.device_configuration_sha256.clone(),
                        requirements_sha256: profile.requirements_sha256.clone(),
                    }),
                }
            })
            .collect()
    }

    #[test]
    fn site_qualified_comparison_accepts_two_eligible_architectures_at_one_site() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let service_profile = qualification_profile(
            crate::service_qualification::QualificationClaim::WaterAndEnergyEfficiency,
        );
        let control_qualification = service_qualification_record(
            "control-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_qualification = service_qualification_record(
            "candidate-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let control_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::DirectFlowPassive,
            &control.device.device_configuration_sha256,
        );
        let candidate_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::SameSessionRecirculation,
            &candidate.device.device_configuration_sha256,
        );
        let control_site_assessments = site_fit_assessments(
            &control_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        let candidate_site_assessments = site_fit_assessments(
            &candidate_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );

        let result = validate_site_qualified_matched_comparison(
            &control,
            &candidate,
            ComparisonScope::WaterAndEnergy,
            &service_profile,
            &control_qualification,
            &candidate_qualification,
            &control_site,
            &control_site_assessments,
            &candidate_site,
            &candidate_site_assessments,
        )
        .expect("both architectures fit the same site and both service records qualify");

        assert_eq!(
            result.control_site_fit.disposition,
            crate::site_fit::SiteFitDisposition::Eligible
        );
        assert_eq!(
            result.candidate_site_fit.disposition,
            crate::site_fit::SiteFitDisposition::Eligible
        );
        assert_eq!(
            result.service_qualification.disposition,
            crate::service_qualification::MatchedComparisonDisposition::CandidateQualified
        );
    }

    #[test]
    fn site_qualified_comparison_rejects_ineligible_candidate() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let service_profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_qualification = service_qualification_record(
            "control-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_qualification = service_qualification_record(
            "candidate-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let control_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::DirectFlowPassive,
            &control.device.device_configuration_sha256,
        );
        let candidate_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::SameSessionRecirculation,
            &candidate.device.device_configuration_sha256,
        );
        let control_evidence = site_fit_assessments(
            &control_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        let mut candidate_evidence = site_fit_assessments(
            &candidate_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        candidate_evidence[0].outcome = crate::site_fit::SiteCapabilityOutcome::Unsatisfied;

        let result = validate_site_qualified_matched_comparison(
            &control,
            &candidate,
            ComparisonScope::WaterServiceOnly,
            &service_profile,
            &control_qualification,
            &candidate_qualification,
            &control_site,
            &control_evidence,
            &candidate_site,
            &candidate_evidence,
        );

        assert!(matches!(
            result,
            Err(QualifiedComparisonError::SiteNotEligible {
                side: ComparisonSide::Candidate,
                result
            }) if result.disposition == crate::site_fit::SiteFitDisposition::Ineligible
        ));
    }

    #[test]
    fn site_qualified_comparison_rejects_indeterminate_site_evidence() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let service_profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_qualification = service_qualification_record(
            "control-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_qualification = service_qualification_record(
            "candidate-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let control_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::DirectFlowPassive,
            &control.device.device_configuration_sha256,
        );
        let candidate_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::SameSessionRecirculation,
            &candidate.device.device_configuration_sha256,
        );
        let control_evidence = site_fit_assessments(
            &control_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        let mut candidate_evidence = site_fit_assessments(
            &candidate_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        candidate_evidence[0].evidence.as_mut().expect("evidence").status =
            crate::service_qualification::QualificationEvidenceStatus::Stale;

        let result = validate_site_qualified_matched_comparison(
            &control,
            &candidate,
            ComparisonScope::WaterServiceOnly,
            &service_profile,
            &control_qualification,
            &candidate_qualification,
            &control_site,
            &control_evidence,
            &candidate_site,
            &candidate_evidence,
        );

        assert!(matches!(
            result,
            Err(QualifiedComparisonError::SiteNotEligible {
                side: ComparisonSide::Candidate,
                result
            }) if result.disposition == crate::site_fit::SiteFitDisposition::Indeterminate
        ));
    }

    #[test]
    fn site_qualified_comparison_rejects_different_site_surveys() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let service_profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_qualification = service_qualification_record(
            "control-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_qualification = service_qualification_record(
            "candidate-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let control_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::DirectFlowPassive,
            &control.device.device_configuration_sha256,
        );
        let mut candidate_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::SameSessionRecirculation,
            &candidate.device.device_configuration_sha256,
        );
        candidate_site.site_conditions_sha256 = digest('0');
        let control_evidence = site_fit_assessments(
            &control_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        let candidate_evidence = site_fit_assessments(
            &candidate_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );

        assert_eq!(
            validate_site_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &service_profile,
                &control_qualification,
                &candidate_qualification,
                &control_site,
                &control_evidence,
                &candidate_site,
                &candidate_evidence,
            ),
            Err(QualifiedComparisonError::SiteContextMismatch {
                field: "site_conditions_sha256"
            })
        );
    }

    #[test]
    fn site_qualified_comparison_rejects_wrong_device_binding() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let service_profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_qualification = service_qualification_record(
            "control-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_qualification = service_qualification_record(
            "candidate-001",
            &service_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let control_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::DirectFlowPassive,
            &control.device.device_configuration_sha256,
        );
        let candidate_site = site_fit_profile(
            crate::site_fit::AquaArchitecture::SameSessionRecirculation,
            &digest('0'),
        );
        let control_evidence = site_fit_assessments(
            &control_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );
        let candidate_evidence = site_fit_assessments(
            &candidate_site,
            crate::site_fit::SiteCapabilityOutcome::Satisfied,
        );

        assert_eq!(
            validate_site_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &service_profile,
                &control_qualification,
                &candidate_qualification,
                &control_site,
                &control_evidence,
                &candidate_site,
                &candidate_evidence,
            ),
            Err(QualifiedComparisonError::Benchmark(
                ContractError::ContextMismatch {
                    field: "candidate_site_device_configuration_sha256"
                }
            ))
        );
    }

    #[test]
    fn qualified_comparison_requires_both_owner_receipts_and_service_assessments() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::WaterAndEnergyEfficiency,
        );
        let control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );

        let result = validate_qualified_matched_comparison(
            &control,
            &candidate,
            ComparisonScope::WaterAndEnergy,
            &profile,
            &control_assessment,
            &candidate_assessment,
        )
        .expect("both matched runs and independent service assessments qualify");

        assert_eq!(
            result.disposition,
            crate::service_qualification::MatchedComparisonDisposition::CandidateQualified
        );
    }

    #[test]
    fn qualified_comparison_rejects_service_evidence_from_another_run() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_assessment = service_qualification_record(
            "different-control-run",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );

        assert_eq!(
            validate_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &profile,
                &control_assessment,
                &candidate_assessment,
            ),
            Err(QualifiedComparisonError::Benchmark(
                ContractError::ContextMismatch {
                    field: "control_qualification.run_id"
                }
            ))
        );
    }

    #[test]
    fn qualified_comparison_rejects_service_evidence_from_another_device_configuration() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let mut control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        control_assessment.device_configuration_sha256 = digest('z');
        let candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );

        assert_eq!(
            validate_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &profile,
                &control_assessment,
                &candidate_assessment,
            ),
            Err(QualifiedComparisonError::Benchmark(
                ContractError::ContextMismatch {
                    field: "control_device_configuration_sha256"
                }
            ))
        );
    }

    #[test]
    fn qualified_comparison_rejects_missing_canonical_energy_receipt() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.energy_resource_closure = None;
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::WaterAndEnergyEfficiency,
        );
        let control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );

        assert_eq!(
            validate_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterAndEnergy,
                &profile,
                &control_assessment,
                &candidate_assessment,
            ),
            Err(QualifiedComparisonError::Benchmark(
                ContractError::MissingEnergyReceipt
            ))
        );
    }

    #[test]
    fn qualified_comparison_reports_candidate_endpoint_failure() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let mut candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        candidate_assessment.assessments[2].outcome =
            crate::service_qualification::AssessmentOutcome::Fail;

        let result = validate_qualified_matched_comparison(
            &control,
            &candidate,
            ComparisonScope::WaterServiceOnly,
            &profile,
            &control_assessment,
            &candidate_assessment,
        )
        .expect("well-formed endpoint failure is a decision, not a structural error");

        assert_eq!(
            result.disposition,
            crate::service_qualification::MatchedComparisonDisposition::CandidateFailed
        );
    }

    #[test]
    fn qualified_comparison_rejects_indeterminate_control() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let mut control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        control_assessment.assessments[0].outcome =
            crate::service_qualification::AssessmentOutcome::Indeterminate;
        let candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );

        let result = validate_qualified_matched_comparison(
            &control,
            &candidate,
            ComparisonScope::WaterServiceOnly,
            &profile,
            &control_assessment,
            &candidate_assessment,
        )
        .expect("indeterminate control is a valid unresolved decision");

        assert_eq!(
            result.disposition,
            crate::service_qualification::MatchedComparisonDisposition::ControlIndeterminate
        );
    }

    #[test]
    fn qualified_comparison_requires_service_profile_to_bind_run_condition_bundle() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let mut unrelated_profile = profile.clone();
        unrelated_profile.conditions_sha256 = digest('z');
        let unrelated_control = service_qualification_record(
            "control-001",
            &unrelated_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let unrelated_candidate = service_qualification_record(
            "candidate-001",
            &unrelated_profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );

        assert_eq!(
            validate_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &unrelated_profile,
                &unrelated_control,
                &unrelated_candidate,
            ),
            Err(QualifiedComparisonError::Benchmark(
                ContractError::ContextMismatch {
                    field: "qualification_conditions_sha256"
                }
            ))
        );

        // The bound profile remains valid for the same run context.
        assert_eq!(
            validate_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &profile,
                &control_assessment,
                &candidate_assessment,
            )
            .expect("bound profile")
            .disposition,
            crate::service_qualification::MatchedComparisonDisposition::CandidateQualified
        );
    }

    #[test]
    fn qualified_comparison_rejects_divergent_independent_trial_conditions() {
        let control = run("control-001", "incumbent-shower", '7');
        let candidate = run("candidate-001", "aqua-test-article", '8');
        let profile = qualification_profile(
            crate::service_qualification::QualificationClaim::FunctionalService,
        );
        let control_assessment = service_qualification_record(
            "control-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        let mut candidate_assessment = service_qualification_record(
            "candidate-001",
            &profile,
            crate::service_qualification::AssessmentOutcome::Pass,
        );
        candidate_assessment.conditions_sha256 = digest('z');

        assert_eq!(
            validate_qualified_matched_comparison(
                &control,
                &candidate,
                ComparisonScope::WaterServiceOnly,
                &profile,
                &control_assessment,
                &candidate_assessment,
            ),
            Err(QualifiedComparisonError::ServiceQualification(
                crate::service_qualification::ServiceQualificationError::ConditionsMismatch
            ))
        );
    }

    #[test]
    fn rejects_different_task_profiles() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.task_profile_id = "HAIR_WASH".to_owned();
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly,),
            Err(ContractError::ContextMismatch {
                field: "task_profile_id",
            })
        );
    }

    #[test]
    fn rejects_different_delivery_conditions() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.delivery_conditions_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly,),
            Err(ContractError::ContextMismatch {
                field: "delivery_conditions_sha256",
            })
        );
    }

    #[test]
    fn rejects_zero_duration() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.duration_seconds = 0;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::ZeroDuration)
        );
    }

    #[test]
    fn rejects_missing_water_service_receipt() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.water_service_t0.record_id.clear();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::MissingField {
                field: "water_service_t0",
            })
        );
    }

    #[test]
    fn rejects_wrong_evidence_owner() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.metrology_admission.owner = EvidenceOwner::MycelixWaterQuality;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::WrongEvidenceOwner {
                field: "metrology_admission",
            })
        );
    }

    #[test]
    fn rejects_invalid_sha256_reference() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.water_quality_assessment.sha256 = "not-a-digest".to_owned();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::InvalidDigest {
                field: "water_quality_assessment",
            })
        );
    }

    #[test]
    fn energy_comparison_requires_canonical_resource_receipt() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.energy_resource_closure = None;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterAndEnergy),
            Err(ContractError::MissingEnergyReceipt)
        );
    }

    #[test]
    fn water_only_comparison_does_not_require_energy_receipt() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.energy_resource_closure = None;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Ok(())
        );
    }

    #[test]
    fn rejects_comparing_a_run_to_itself() {
        let candidate = run("same-run", "aqua-test-article", '8');
        assert_eq!(
            validate_matched_comparison(&candidate, &candidate, ComparisonScope::WaterServiceOnly,),
            Err(ContractError::SameRunCompared)
        );
    }

    #[test]
    fn rejects_different_task_duration() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.duration_seconds = 599;
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "duration_seconds"
            })
        );
    }

    #[test]
    fn rejects_different_inlet_water_profile() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.inlet_water_profile_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "inlet_water_profile_sha256"
            })
        );
    }

    #[test]
    fn rejects_different_room_conditions() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.room_conditions_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "room_conditions_sha256"
            })
        );
    }

    #[test]
    fn rejects_different_measurement_protocol() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.measurement_protocol_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "measurement_protocol_sha256"
            })
        );
    }

    #[test]
    fn rejects_different_accounting_boundary() {
        let control = run("control-001", "incumbent-shower", '7');
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.conditions.accounting_boundary_sha256 = digest('9');
        assert_eq!(
            validate_matched_comparison(&control, &candidate, ComparisonScope::WaterServiceOnly),
            Err(ContractError::ContextMismatch {
                field: "accounting_boundary_sha256"
            })
        );
    }

    #[test]
    fn rejects_missing_independent_task_outcome_evidence() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.evidence.task_outcome_assessment.record_id.clear();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::MissingField {
                field: "task_outcome_assessment"
            })
        );
    }

    #[test]
    fn rejects_wrong_energy_resource_owner() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate
            .evidence
            .energy_resource_closure
            .as_mut()
            .unwrap()
            .owner = EvidenceOwner::SymtropyWaterServiceT0;
        assert_eq!(
            candidate.validate(ComparisonScope::WaterAndEnergy),
            Err(ContractError::WrongEvidenceOwner {
                field: "energy_resource_closure"
            })
        );
    }

    #[test]
    fn rejects_invalid_device_cleaning_protocol_digest() {
        let mut candidate = run("candidate-001", "aqua-test-article", '8');
        candidate.device.cleaning_protocol_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            candidate.validate(ComparisonScope::WaterServiceOnly),
            Err(ContractError::InvalidDigest {
                field: "cleaning_protocol_sha256"
            })
        );
    }
}
