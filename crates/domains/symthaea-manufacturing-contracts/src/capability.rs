use crate::{canonical_token, hash_field, ManufacturingContractError};
use serde::{Deserialize, Serialize};
use symthaea_manufacturing_process::ProcessDefinitionId;

const CAPABILITY_DOMAIN: &str = "symthaea-manufacturing-contracts::capability-v1";
const REQUIREMENT_DOMAIN: &str = "symthaea-manufacturing-contracts::capability-requirement-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CapabilityEvidenceClassV1 {
    UnknownOrUnavailable,
    Declared,
    ManufacturerSpecified,
    ObservedCapability,
    QualifiedUnderProfile,
    ProductionQualifiedUnderProfile,
}

impl CapabilityEvidenceClassV1 {
    fn tag(self) -> &'static str {
        match self {
            Self::UnknownOrUnavailable => "unknown-or-unavailable",
            Self::Declared => "declared",
            Self::ManufacturerSpecified => "manufacturer-specified",
            Self::ObservedCapability => "observed-capability",
            Self::QualifiedUnderProfile => "qualified-under-profile",
            Self::ProductionQualifiedUnderProfile => "production-qualified-under-profile",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityValidityV1 {
    pub revision: String,
    pub valid_from_unix_s: u64,
    pub valid_until_unix_s: u64,
}

impl CapabilityValidityV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("capability.validity.revision", &self.revision)?;
        if self.valid_from_unix_s >= self.valid_until_unix_s {
            return Err(ManufacturingContractError::Invalid(
                "capability validity window must be ordered",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessCapabilityProfileV1 {
    pub resource_subject_ref: String,
    pub process_id: ProcessDefinitionId,
    pub envelope_profile_ref: String,
    pub configuration_profile_ref: String,
    pub evidence_class: CapabilityEvidenceClassV1,
    pub evidence_ref: String,
    pub validity: CapabilityValidityV1,
    pub display_label: Option<String>,
}

impl ProcessCapabilityProfileV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("capability.resource_subject_ref", &self.resource_subject_ref)?;
        canonical_token("capability.process_id", &self.process_id.0)?;
        canonical_token("capability.envelope_profile_ref", &self.envelope_profile_ref)?;
        canonical_token(
            "capability.configuration_profile_ref",
            &self.configuration_profile_ref,
        )?;
        canonical_token("capability.evidence_ref", &self.evidence_ref)?;
        self.validity.validate()
    }

    pub fn capability_id(&self) -> Result<String, ManufacturingContractError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, CAPABILITY_DOMAIN);
        for value in [
            self.resource_subject_ref.as_str(),
            self.process_id.0.as_str(),
            self.envelope_profile_ref.as_str(),
            self.configuration_profile_ref.as_str(),
            self.evidence_class.tag(),
            self.evidence_ref.as_str(),
            self.validity.revision.as_str(),
        ] {
            hash_field(&mut hasher, value);
        }
        hasher.update(&self.validity.valid_from_unix_s.to_le_bytes());
        hasher.update(&self.validity.valid_until_unix_s.to_le_bytes());
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessCapabilityRequirementV1 {
    pub process_id: ProcessDefinitionId,
    pub envelope_profile_ref: String,
    pub minimum_evidence: CapabilityEvidenceClassV1,
}

impl ProcessCapabilityRequirementV1 {
    pub fn validate(&self) -> Result<(), ManufacturingContractError> {
        canonical_token("capability_requirement.process_id", &self.process_id.0)?;
        canonical_token(
            "capability_requirement.envelope_profile_ref",
            &self.envelope_profile_ref,
        )
    }

    pub fn requirement_id(&self) -> Result<String, ManufacturingContractError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, REQUIREMENT_DOMAIN);
        hash_field(&mut hasher, &self.process_id.0);
        hash_field(&mut hasher, &self.envelope_profile_ref);
        hash_field(&mut hasher, self.minimum_evidence.tag());
        Ok(hasher.finalize().to_hex().to_string())
    }

    pub fn evaluate(
        &self,
        candidate: &ProcessCapabilityProfileV1,
        evaluation_time_unix_s: u64,
        external_refs_resolved: bool,
    ) -> CapabilityMatchV1 {
        if self.validate().is_err() || candidate.validate().is_err() {
            return CapabilityMatchV1::Unknown;
        }
        if candidate.process_id != self.process_id {
            return CapabilityMatchV1::ProcessMismatch;
        }
        if candidate.envelope_profile_ref != self.envelope_profile_ref {
            return CapabilityMatchV1::EnvelopeMismatch;
        }
        if evaluation_time_unix_s < candidate.validity.valid_from_unix_s
            || evaluation_time_unix_s >= candidate.validity.valid_until_unix_s
        {
            return CapabilityMatchV1::ExpiredOrStale;
        }
        if !external_refs_resolved {
            return CapabilityMatchV1::UnresolvedExternalRefs;
        }
        if candidate.evidence_class < self.minimum_evidence {
            return CapabilityMatchV1::CompatibleButWeakerEvidence;
        }
        CapabilityMatchV1::ExactAdmitted
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityMatchV1 {
    ExactAdmitted,
    CompatibleButWeakerEvidence,
    UnresolvedExternalRefs,
    ProcessMismatch,
    EnvelopeMismatch,
    ExpiredOrStale,
    Unknown,
}
