// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed claim-derivation projections for MEL-EPI.
//!
//! This module records a structural declaration that an exact derivation
//! profile's prerequisite claim *declarations*, preservation rules, provenance
//! links, and output ceiling are internally consistent.
//!
//! It deliberately does not establish that any prerequisite claim was admitted
//! by its source-native verifier. Generic structural consistency is not an
//! epistemic credential.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::claim_scope::{
    ClaimDeclarationStatusV1, ClaimScopeError, ClaimScopeProjectionV1,
};
use crate::namespaced_code::{NamespacedCodeError, NamespacedCodeV1};
use crate::provenance::{
    ProvenanceAssociationRelationV1, ProvenanceGraphError, ProvenanceGraphV1,
    ProvenanceNodeClassV1, ProvenanceNodeReferenceV1, ProvenanceRelationV1,
    ValidatedProvenanceGraphV1,
};
use crate::semantic_evidence::{
    EvidenceSemanticIdV1, SemanticTranscriptError, SemanticTranscriptV1,
};
use crate::source_reference::SemanticPreservationV1;
use crate::validated_projection::ValidatedClaimScopeProjectionV1;

pub const CLAIM_DERIVATION_PROFILE_VERSION_V1: &str =
    "melothaea-claim-derivation-profile-projection-v1";
pub const CLAIM_DERIVATION_RECEIPT_VERSION_V1: &str =
    "melothaea-claim-derivation-receipt-projection-v1";

const PRESERVATION_REQUIRE_LOSSLESS_V1: &str = "require-lossless-under-profile";
const PRESERVATION_ALLOW_PROJECTED_V1: &str = "allow-projected-with-loss";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputPreservationRequirementV1 {
    RequireLosslessUnderProfile,
    AllowProjectedWithLoss {
        allowed_loss_codes: Vec<NamespacedCodeV1>,
    },
}

impl InputPreservationRequirementV1 {
    fn validate(&self) -> Result<(), ClaimDerivationError> {
        if let Self::AllowProjectedWithLoss { allowed_loss_codes } = self {
            if allowed_loss_codes.is_empty() {
                return Err(ClaimDerivationError::EmptyAllowedLossCodes);
            }
            validate_sorted_codes(
                allowed_loss_codes,
                ClaimDerivationCodeSetKind::AllowedLossCodes,
            )?;
        }
        Ok(())
    }

    fn mode_code(&self) -> &'static str {
        match self {
            Self::RequireLosslessUnderProfile => PRESERVATION_REQUIRE_LOSSLESS_V1,
            Self::AllowProjectedWithLoss { .. } => PRESERVATION_ALLOW_PROJECTED_V1,
        }
    }

    fn allowed_loss_bytes(&self) -> Vec<Vec<u8>> {
        match self {
            Self::RequireLosslessUnderProfile => Vec::new(),
            Self::AllowProjectedWithLoss { allowed_loss_codes } => allowed_loss_codes
                .iter()
                .map(|code| code.as_str().as_bytes().to_vec())
                .collect(),
        }
    }

    fn accepts(&self, preservation: &SemanticPreservationV1) -> bool {
        match (self, preservation) {
            (Self::RequireLosslessUnderProfile, SemanticPreservationV1::LosslessUnderProfile) => {
                true
            }
            (Self::RequireLosslessUnderProfile, _) => false,
            (Self::AllowProjectedWithLoss { .. }, SemanticPreservationV1::LosslessUnderProfile) => {
                true
            }
            (
                Self::AllowProjectedWithLoss { allowed_loss_codes },
                SemanticPreservationV1::ProjectedWithLoss { loss_codes },
            ) => loss_codes
                .iter()
                .all(|loss| allowed_loss_codes.binary_search(loss).is_ok()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDerivationInputRequirementV1 {
    pub role_id: NamespacedCodeV1,
    pub required_scope_profile_id: NamespacedCodeV1,
    pub required_claims: Vec<NamespacedCodeV1>,
    pub preservation: InputPreservationRequirementV1,
}

impl ClaimDerivationInputRequirementV1 {
    fn validate(&self) -> Result<(), ClaimDerivationError> {
        self.role_id
            .validate()
            .map_err(ClaimDerivationError::InvalidCode)?;
        self.required_scope_profile_id
            .validate()
            .map_err(ClaimDerivationError::InvalidCode)?;
        if self.required_claims.is_empty() {
            return Err(ClaimDerivationError::EmptyRequiredClaims {
                role: self.role_id.as_str().to_owned(),
            });
        }
        validate_sorted_codes(
            &self.required_claims,
            ClaimDerivationCodeSetKind::RequiredClaims,
        )?;
        self.preservation.validate()?;
        Ok(())
    }
}

/// Shape-only declaration of one fail-closed claim-transfer theorem/profile.
///
/// A valid profile projection describes prerequisites and an output ceiling. It
/// is ordinary data and does not prove scientific justification or source
/// admission of any concrete prerequisite claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDerivationProfileProjectionV1 {
    pub profile_version: String,
    pub profile_id: NamespacedCodeV1,
    pub mechanism_activity_id: EvidenceSemanticIdV1,
    pub input_requirements: Vec<ClaimDerivationInputRequirementV1>,
    pub output_scope_profile_id: NamespacedCodeV1,
    pub output_claims: Vec<NamespacedCodeV1>,
    pub explicit_nonclaims: Vec<NamespacedCodeV1>,
}

impl ClaimDerivationProfileProjectionV1 {
    pub fn validate(&self) -> Result<(), ClaimDerivationError> {
        if self.profile_version != CLAIM_DERIVATION_PROFILE_VERSION_V1 {
            return Err(ClaimDerivationError::WrongProfileVersion {
                found: self.profile_version.clone(),
            });
        }
        self.profile_id
            .validate()
            .map_err(ClaimDerivationError::InvalidCode)?;
        self.mechanism_activity_id
            .validate()
            .map_err(ClaimDerivationError::InvalidMechanismSemanticId)?;
        self.output_scope_profile_id
            .validate()
            .map_err(ClaimDerivationError::InvalidCode)?;

        if self.input_requirements.is_empty() {
            return Err(ClaimDerivationError::EmptyInputRequirements);
        }
        for requirement in &self.input_requirements {
            requirement.validate()?;
        }
        for pair in self.input_requirements.windows(2) {
            if pair[0].role_id >= pair[1].role_id {
                return Err(ClaimDerivationError::InputRequirementsNotStrictlyIncreasing {
                    previous: pair[0].role_id.as_str().to_owned(),
                    next: pair[1].role_id.as_str().to_owned(),
                });
            }
        }

        if self.output_claims.is_empty() {
            return Err(ClaimDerivationError::EmptyOutputClaims);
        }
        validate_sorted_codes(
            &self.output_claims,
            ClaimDerivationCodeSetKind::OutputClaims,
        )?;
        validate_sorted_codes(
            &self.explicit_nonclaims,
            ClaimDerivationCodeSetKind::ExplicitNonclaims,
        )?;
        for claim in &self.output_claims {
            if self.explicit_nonclaims.binary_search(claim).is_ok() {
                return Err(ClaimDerivationError::OutputClaimOverlap {
                    claim: claim.as_str().to_owned(),
                });
            }
        }
        Ok(())
    }

    pub fn semantic_payload(&self) -> Result<SemanticTranscriptV1, ClaimDerivationError> {
        self.validate()?;
        let requirements = self
            .input_requirements
            .iter()
            .map(input_requirement_bytes)
            .collect::<Result<Vec<_>, _>>()?;

        let mut payload = SemanticTranscriptV1::new();
        payload
            .push_utf8(1, &self.profile_version)
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_utf8(2, self.profile_id.as_str())
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_bytes(3, &semantic_id_bytes(&self.mechanism_activity_id)?)
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_sequence(4, &requirements)
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_utf8(5, self.output_scope_profile_id.as_str())
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_sequence(6, &code_bytes(&self.output_claims))
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_sequence(7, &code_bytes(&self.explicit_nonclaims))
            .map_err(ClaimDerivationError::Transcript)?;
        Ok(payload)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDerivationInputProjectionV1 {
    pub role_id: NamespacedCodeV1,
    pub claim_scope: ClaimScopeProjectionV1,
}

impl ClaimDerivationInputProjectionV1 {
    fn validate(&self) -> Result<(), ClaimDerivationError> {
        self.role_id
            .validate()
            .map_err(ClaimDerivationError::InvalidCode)?;
        self.claim_scope
            .validate()
            .map_err(ClaimDerivationError::InvalidClaimScope)?;
        Ok(())
    }
}

/// Structural receipt projection for one concrete claim derivation.
///
/// Validation establishes only that exact generic declarations satisfy the
/// projected profile's structural prerequisites and provenance links. It does
/// not establish that any required input claim was source-admitted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDerivationReceiptProjectionV1 {
    pub receipt_version: String,
    pub profile: ClaimDerivationProfileProjectionV1,
    pub inputs: Vec<ClaimDerivationInputProjectionV1>,
    pub output: ClaimScopeProjectionV1,
    pub provenance: ProvenanceGraphV1,
}

impl ClaimDerivationReceiptProjectionV1 {
    pub fn validate(&self) -> Result<(), ClaimDerivationError> {
        if self.receipt_version != CLAIM_DERIVATION_RECEIPT_VERSION_V1 {
            return Err(ClaimDerivationError::WrongReceiptVersion {
                found: self.receipt_version.clone(),
            });
        }
        self.profile.validate()?;
        self.output
            .validate()
            .map_err(ClaimDerivationError::InvalidClaimScope)?;
        self.provenance
            .validate()
            .map_err(ClaimDerivationError::InvalidProvenance)?;

        if self.inputs.len() != self.profile.input_requirements.len() {
            return Err(ClaimDerivationError::InputRoleSetMismatch);
        }
        for input in &self.inputs {
            input.validate()?;
        }
        for pair in self.inputs.windows(2) {
            if pair[0].role_id >= pair[1].role_id {
                return Err(ClaimDerivationError::InputsNotStrictlyIncreasing {
                    previous: pair[0].role_id.as_str().to_owned(),
                    next: pair[1].role_id.as_str().to_owned(),
                });
            }
        }

        for (requirement, input) in self
            .profile
            .input_requirements
            .iter()
            .zip(self.inputs.iter())
        {
            if requirement.role_id != input.role_id {
                return Err(ClaimDerivationError::InputRoleSetMismatch);
            }
            if requirement.required_scope_profile_id != input.claim_scope.scope_profile_id {
                return Err(ClaimDerivationError::InputScopeProfileMismatch {
                    role: input.role_id.as_str().to_owned(),
                    expected: requirement.required_scope_profile_id.as_str().to_owned(),
                    found: input.claim_scope.scope_profile_id.as_str().to_owned(),
                });
            }
            if !requirement
                .preservation
                .accepts(&input.claim_scope.source_ref.preservation)
            {
                return Err(ClaimDerivationError::InputPreservationRejected {
                    role: input.role_id.as_str().to_owned(),
                });
            }
            for required_claim in &requirement.required_claims {
                match input
                    .claim_scope
                    .declaration_status(required_claim)
                    .map_err(ClaimDerivationError::InvalidClaimScope)?
                {
                    ClaimDeclarationStatusV1::DeclaredEstablished => {}
                    ClaimDeclarationStatusV1::ExplicitNonclaim => {
                        return Err(ClaimDerivationError::PrerequisiteExplicitNonclaim {
                            role: input.role_id.as_str().to_owned(),
                            claim: required_claim.as_str().to_owned(),
                        });
                    }
                    ClaimDeclarationStatusV1::NotDeclared => {
                        return Err(ClaimDerivationError::PrerequisiteNotDeclared {
                            role: input.role_id.as_str().to_owned(),
                            claim: required_claim.as_str().to_owned(),
                        });
                    }
                }
            }
        }

        if self.output.scope_profile_id != self.profile.output_scope_profile_id {
            return Err(ClaimDerivationError::OutputScopeProfileMismatch);
        }
        if self.output.establishes != self.profile.output_claims {
            return Err(ClaimDerivationError::OutputClaimsMismatch);
        }
        if self.output.explicit_nonclaims != self.profile.explicit_nonclaims {
            return Err(ClaimDerivationError::OutputNonclaimsMismatch);
        }

        let output_ref = ProvenanceNodeReferenceV1::SourceRef(self.output.source_ref.clone());
        if !provenance_contains_reference(&self.provenance, &output_ref) {
            return Err(ClaimDerivationError::OutputSourceMissingFromProvenance);
        }

        let mechanism_ref =
            ProvenanceNodeReferenceV1::SemanticId(self.profile.mechanism_activity_id.clone());
        if provenance_node_class(&self.provenance, &mechanism_ref)
            != Some(ProvenanceNodeClassV1::Activity)
        {
            return Err(ClaimDerivationError::MechanismActivityMissingFromProvenance);
        }
        if !has_generated_by_edge(&self.provenance, &output_ref, &mechanism_ref) {
            return Err(ClaimDerivationError::OutputNotGeneratedByMechanism);
        }

        for input in &self.inputs {
            if input.claim_scope.source_ref == self.output.source_ref {
                return Err(ClaimDerivationError::InputEqualsOutputSource {
                    role: input.role_id.as_str().to_owned(),
                });
            }
            let input_ref =
                ProvenanceNodeReferenceV1::SourceRef(input.claim_scope.source_ref.clone());
            if !provenance_contains_reference(&self.provenance, &input_ref) {
                return Err(ClaimDerivationError::InputSourceMissingFromProvenance {
                    role: input.role_id.as_str().to_owned(),
                });
            }
            if !has_derivation_path(&self.provenance, &output_ref, &input_ref) {
                return Err(ClaimDerivationError::InputNotDerivationAncestor {
                    role: input.role_id.as_str().to_owned(),
                });
            }
        }

        Ok(())
    }

    pub fn semantic_payload(&self) -> Result<SemanticTranscriptV1, ClaimDerivationError> {
        self.validate()?;
        let input_bytes = self
            .inputs
            .iter()
            .map(input_projection_bytes)
            .collect::<Result<Vec<_>, _>>()?;

        let mut payload = SemanticTranscriptV1::new();
        payload
            .push_utf8(1, &self.receipt_version)
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_bytes(2, self.profile.semantic_payload()?.as_bytes())
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_sequence(3, &input_bytes)
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_bytes(
                4,
                self.output
                    .semantic_payload()
                    .map_err(ClaimDerivationError::InvalidClaimScope)?
                    .as_bytes(),
            )
            .map_err(ClaimDerivationError::Transcript)?;
        payload
            .push_bytes(
                5,
                self.provenance
                    .semantic_payload()
                    .map_err(ClaimDerivationError::InvalidProvenance)?
                    .as_bytes(),
            )
            .map_err(ClaimDerivationError::Transcript)?;
        Ok(payload)
    }

    pub fn from_validated_parts(
        profile: ClaimDerivationProfileProjectionV1,
        inputs: Vec<(NamespacedCodeV1, &ValidatedClaimScopeProjectionV1)>,
        output: &ValidatedClaimScopeProjectionV1,
        provenance: &ValidatedProvenanceGraphV1,
    ) -> Result<Self, ClaimDerivationError> {
        let raw = Self {
            receipt_version: CLAIM_DERIVATION_RECEIPT_VERSION_V1.into(),
            profile,
            inputs: inputs
                .into_iter()
                .map(|(role_id, claim_scope)| ClaimDerivationInputProjectionV1 {
                    role_id,
                    claim_scope: claim_scope.as_raw().clone(),
                })
                .collect(),
            output: output.as_raw().clone(),
            provenance: provenance.as_raw().clone(),
        };
        raw.validate()?;
        Ok(raw)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedClaimDerivationReceiptProjectionV1 {
    raw: ClaimDerivationReceiptProjectionV1,
    semantic_payload: SemanticTranscriptV1,
}

impl ValidatedClaimDerivationReceiptProjectionV1 {
    pub fn as_raw(&self) -> &ClaimDerivationReceiptProjectionV1 {
        &self.raw
    }

    pub fn into_raw(self) -> ClaimDerivationReceiptProjectionV1 {
        self.raw
    }

    pub fn semantic_payload(&self) -> &SemanticTranscriptV1 {
        &self.semantic_payload
    }
}

impl TryFrom<ClaimDerivationReceiptProjectionV1>
    for ValidatedClaimDerivationReceiptProjectionV1
{
    type Error = ClaimDerivationError;

    fn try_from(raw: ClaimDerivationReceiptProjectionV1) -> Result<Self, Self::Error> {
        raw.validate()?;
        let semantic_payload = raw.semantic_payload()?;
        Ok(Self {
            raw,
            semantic_payload,
        })
    }
}

impl TryFrom<&ClaimDerivationReceiptProjectionV1>
    for ValidatedClaimDerivationReceiptProjectionV1
{
    type Error = ClaimDerivationError;

    fn try_from(raw: &ClaimDerivationReceiptProjectionV1) -> Result<Self, Self::Error> {
        Self::try_from(raw.clone())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClaimDerivationCodeSetKind {
    AllowedLossCodes,
    RequiredClaims,
    OutputClaims,
    ExplicitNonclaims,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClaimDerivationError {
    WrongProfileVersion { found: String },
    WrongReceiptVersion { found: String },
    InvalidCode(NamespacedCodeError),
    InvalidMechanismSemanticId(SemanticTranscriptError),
    InvalidClaimScope(ClaimScopeError),
    InvalidProvenance(ProvenanceGraphError),
    EmptyAllowedLossCodes,
    EmptyInputRequirements,
    EmptyRequiredClaims { role: String },
    EmptyOutputClaims,
    CodesNotStrictlyIncreasing {
        kind: &'static str,
        previous: String,
        next: String,
    },
    InputRequirementsNotStrictlyIncreasing { previous: String, next: String },
    InputsNotStrictlyIncreasing { previous: String, next: String },
    OutputClaimOverlap { claim: String },
    InputRoleSetMismatch,
    InputScopeProfileMismatch {
        role: String,
        expected: String,
        found: String,
    },
    InputPreservationRejected { role: String },
    PrerequisiteExplicitNonclaim { role: String, claim: String },
    PrerequisiteNotDeclared { role: String, claim: String },
    OutputScopeProfileMismatch,
    OutputClaimsMismatch,
    OutputNonclaimsMismatch,
    OutputSourceMissingFromProvenance,
    MechanismActivityMissingFromProvenance,
    OutputNotGeneratedByMechanism,
    InputEqualsOutputSource { role: String },
    InputSourceMissingFromProvenance { role: String },
    InputNotDerivationAncestor { role: String },
    Transcript(SemanticTranscriptError),
}

impl fmt::Display for ClaimDerivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongProfileVersion { found } => write!(
                f,
                "unsupported claim-derivation profile version: {found}"
            ),
            Self::WrongReceiptVersion { found } => write!(
                f,
                "unsupported claim-derivation receipt version: {found}"
            ),
            Self::InvalidCode(error) => write!(f, "invalid namespaced code: {error}"),
            Self::InvalidMechanismSemanticId(error) => write!(
                f,
                "invalid mechanism activity semantic id: {error}"
            ),
            Self::InvalidClaimScope(error) => write!(f, "invalid claim scope: {error}"),
            Self::InvalidProvenance(error) => write!(f, "invalid provenance graph: {error}"),
            Self::EmptyAllowedLossCodes => write!(
                f,
                "AllowProjectedWithLoss requires at least one explicitly allowed loss code"
            ),
            Self::EmptyInputRequirements => {
                write!(f, "derivation profile requires at least one input")
            }
            Self::EmptyRequiredClaims { role } => write!(
                f,
                "input role {role} requires at least one prerequisite claim"
            ),
            Self::EmptyOutputClaims => {
                write!(f, "derivation profile requires at least one output claim")
            }
            Self::CodesNotStrictlyIncreasing {
                kind,
                previous,
                next,
            } => write!(
                f,
                "{kind} must be strictly increasing and unique: previous={previous}, next={next}"
            ),
            Self::InputRequirementsNotStrictlyIncreasing { previous, next } => write!(
                f,
                "input requirements must be strictly increasing and unique by role: previous={previous}, next={next}"
            ),
            Self::InputsNotStrictlyIncreasing { previous, next } => write!(
                f,
                "receipt inputs must be strictly increasing and unique by role: previous={previous}, next={next}"
            ),
            Self::OutputClaimOverlap { claim } => write!(
                f,
                "output claim cannot also be an explicit nonclaim: {claim}"
            ),
            Self::InputRoleSetMismatch => {
                write!(f, "receipt input role set does not exactly match profile")
            }
            Self::InputScopeProfileMismatch {
                role,
                expected,
                found,
            } => write!(
                f,
                "input role {role} claim-scope profile mismatch: expected={expected}, found={found}"
            ),
            Self::InputPreservationRejected { role } => write!(
                f,
                "input role {role} preservation state is not allowed by the derivation profile"
            ),
            Self::PrerequisiteExplicitNonclaim { role, claim } => write!(
                f,
                "input role {role} explicitly marks prerequisite {claim} as a nonclaim"
            ),
            Self::PrerequisiteNotDeclared { role, claim } => write!(
                f,
                "input role {role} does not declare prerequisite {claim} established"
            ),
            Self::OutputScopeProfileMismatch => write!(
                f,
                "output claim-scope profile does not match derivation profile"
            ),
            Self::OutputClaimsMismatch => write!(
                f,
                "output positive claim set does not exactly match derivation profile ceiling"
            ),
            Self::OutputNonclaimsMismatch => write!(
                f,
                "output explicit nonclaim set does not exactly match derivation profile"
            ),
            Self::OutputSourceMissingFromProvenance => write!(
                f,
                "output source reference is absent from provenance graph"
            ),
            Self::MechanismActivityMissingFromProvenance => write!(
                f,
                "derivation mechanism activity is absent or not typed Activity in provenance graph"
            ),
            Self::OutputNotGeneratedByMechanism => write!(
                f,
                "output is not linked to the exact mechanism activity by GeneratedBy"
            ),
            Self::InputEqualsOutputSource { role } => write!(
                f,
                "input role {role} uses the exact output source reference"
            ),
            Self::InputSourceMissingFromProvenance { role } => write!(
                f,
                "input role {role} exact source reference is absent from provenance graph"
            ),
            Self::InputNotDerivationAncestor { role } => write!(
                f,
                "input role {role} is not a derivation ancestor of the output"
            ),
            Self::Transcript(error) => write!(f, "claim-derivation transcript error: {error}"),
        }
    }
}

impl std::error::Error for ClaimDerivationError {}

fn validate_sorted_codes(
    codes: &[NamespacedCodeV1],
    kind: ClaimDerivationCodeSetKind,
) -> Result<(), ClaimDerivationError> {
    for code in codes {
        code.validate().map_err(ClaimDerivationError::InvalidCode)?;
    }
    for pair in codes.windows(2) {
        if pair[0] >= pair[1] {
            return Err(ClaimDerivationError::CodesNotStrictlyIncreasing {
                kind: code_set_name(kind),
                previous: pair[0].as_str().to_owned(),
                next: pair[1].as_str().to_owned(),
            });
        }
    }
    Ok(())
}

fn code_set_name(kind: ClaimDerivationCodeSetKind) -> &'static str {
    match kind {
        ClaimDerivationCodeSetKind::AllowedLossCodes => "allowed loss codes",
        ClaimDerivationCodeSetKind::RequiredClaims => "required claims",
        ClaimDerivationCodeSetKind::OutputClaims => "output claims",
        ClaimDerivationCodeSetKind::ExplicitNonclaims => "explicit nonclaims",
    }
}

fn code_bytes(codes: &[NamespacedCodeV1]) -> Vec<Vec<u8>> {
    codes
        .iter()
        .map(|code| code.as_str().as_bytes().to_vec())
        .collect()
}

fn semantic_id_bytes(id: &EvidenceSemanticIdV1) -> Result<Vec<u8>, ClaimDerivationError> {
    id.validate()
        .map_err(ClaimDerivationError::InvalidMechanismSemanticId)?;
    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, &id.digest_algorithm)
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(2, &id.transcript_version)
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(3, &id.namespace)
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(4, &id.schema_version)
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(5, &id.profile_id)
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(6, &id.record_id)
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_sha256_digest_reference(
            7,
            id.digest_bytes()
                .map_err(ClaimDerivationError::InvalidMechanismSemanticId)?,
        )
        .map_err(ClaimDerivationError::Transcript)?;
    Ok(payload.as_bytes().to_vec())
}

fn input_requirement_bytes(
    requirement: &ClaimDerivationInputRequirementV1,
) -> Result<Vec<u8>, ClaimDerivationError> {
    requirement.validate()?;
    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, requirement.role_id.as_str())
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(2, requirement.required_scope_profile_id.as_str())
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_sequence(3, &code_bytes(&requirement.required_claims))
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_utf8(4, requirement.preservation.mode_code())
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_sequence(5, &requirement.preservation.allowed_loss_bytes())
        .map_err(ClaimDerivationError::Transcript)?;
    Ok(payload.as_bytes().to_vec())
}

fn input_projection_bytes(
    input: &ClaimDerivationInputProjectionV1,
) -> Result<Vec<u8>, ClaimDerivationError> {
    input.validate()?;
    let mut payload = SemanticTranscriptV1::new();
    payload
        .push_utf8(1, input.role_id.as_str())
        .map_err(ClaimDerivationError::Transcript)?;
    payload
        .push_bytes(
            2,
            input
                .claim_scope
                .semantic_payload()
                .map_err(ClaimDerivationError::InvalidClaimScope)?
                .as_bytes(),
        )
        .map_err(ClaimDerivationError::Transcript)?;
    Ok(payload.as_bytes().to_vec())
}

fn provenance_contains_reference(
    graph: &ProvenanceGraphV1,
    reference: &ProvenanceNodeReferenceV1,
) -> bool {
    graph.nodes.iter().any(|node| &node.reference == reference)
}

fn provenance_node_class(
    graph: &ProvenanceGraphV1,
    reference: &ProvenanceNodeReferenceV1,
) -> Option<ProvenanceNodeClassV1> {
    graph
        .nodes
        .iter()
        .find(|node| &node.reference == reference)
        .map(|node| node.class)
}

fn has_generated_by_edge(
    graph: &ProvenanceGraphV1,
    output: &ProvenanceNodeReferenceV1,
    mechanism: &ProvenanceNodeReferenceV1,
) -> bool {
    graph.edges.iter().any(|edge| {
        edge.relation
            == ProvenanceRelationV1::Association(ProvenanceAssociationRelationV1::GeneratedBy)
            && &edge.from == output
            && &edge.to == mechanism
    })
}

fn has_derivation_path(
    graph: &ProvenanceGraphV1,
    output: &ProvenanceNodeReferenceV1,
    input: &ProvenanceNodeReferenceV1,
) -> bool {
    let mut frontier = vec![output.clone()];
    let mut visited = Vec::<ProvenanceNodeReferenceV1>::new();

    while let Some(current) = frontier.pop() {
        if visited.iter().any(|seen| seen == &current) {
            continue;
        }
        visited.push(current.clone());
        for edge in &graph.edges {
            if edge.relation.is_derivation() && edge.from == current {
                if &edge.to == input {
                    return true;
                }
                if !visited.iter().any(|seen| seen == &edge.to) {
                    frontier.push(edge.to.clone());
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim_scope::CLAIM_SCOPE_PROJECTION_VERSION_V1;
    use crate::provenance::{
        PROVENANCE_GRAPH_VERSION_V1, ProvenanceDerivationRelationV1, ProvenanceEdgeV1,
        ProvenanceNodeV1,
    };
    use crate::source_reference::{
        EVIDENCE_SOURCE_REF_VERSION_V1, EvidenceSourceRefV1, SourceNativeCommitmentV1,
    };

    fn code(value: &str) -> NamespacedCodeV1 {
        NamespacedCodeV1::new(value).unwrap()
    }

    fn id(namespace: &str, record: &str, byte: u8) -> EvidenceSemanticIdV1 {
        EvidenceSemanticIdV1::from_sha256_digest(
            namespace,
            "v1",
            "projection-v1",
            record,
            [byte; 32],
        )
        .unwrap()
    }

    fn source_ref(record: &str, semantic_byte: u8, native_byte: u8) -> EvidenceSourceRefV1 {
        EvidenceSourceRefV1 {
            source_ref_version: EVIDENCE_SOURCE_REF_VERSION_V1.into(),
            semantic_id: id("muse.derivation-test", record, semantic_byte),
            native_commitment: SourceNativeCommitmentV1 {
                scheme_id: code("muse.test.commitment-v1"),
                sha256_hex: format!("{native_byte:02x}").repeat(32),
            },
            preservation: SemanticPreservationV1::LosslessUnderProfile,
        }
    }

    fn claim_scope(
        source: EvidenceSourceRefV1,
        scope_profile: &str,
        claims: &[&str],
        nonclaims: &[&str],
    ) -> ClaimScopeProjectionV1 {
        let mut establishes: Vec<_> = claims.iter().map(|value| code(value)).collect();
        establishes.sort();
        let mut explicit_nonclaims: Vec<_> = nonclaims.iter().map(|value| code(value)).collect();
        explicit_nonclaims.sort();
        ClaimScopeProjectionV1 {
            claim_scope_version: CLAIM_SCOPE_PROJECTION_VERSION_V1.into(),
            source_ref: source,
            scope_profile_id: code(scope_profile),
            establishes,
            explicit_nonclaims,
        }
    }

    fn profile(mechanism_activity_id: EvidenceSemanticIdV1) -> ClaimDerivationProfileProjectionV1 {
        ClaimDerivationProfileProjectionV1 {
            profile_version: CLAIM_DERIVATION_PROFILE_VERSION_V1.into(),
            profile_id: code("muse.test.derivation-profile-v1"),
            mechanism_activity_id,
            input_requirements: vec![
                ClaimDerivationInputRequirementV1 {
                    role_id: code("input.plan"),
                    required_scope_profile_id: code("muse.plan.claim-profile-v1"),
                    required_claims: vec![code("muse.plan.bound")],
                    preservation: InputPreservationRequirementV1::RequireLosslessUnderProfile,
                },
                ClaimDerivationInputRequirementV1 {
                    role_id: code("input.results"),
                    required_scope_profile_id: code("muse.results.claim-profile-v1"),
                    required_claims: vec![code("muse.results.rederived")],
                    preservation: InputPreservationRequirementV1::AllowProjectedWithLoss {
                        allowed_loss_codes: vec![code("muse.loss.display-metadata")],
                    },
                },
            ],
            output_scope_profile_id: code("muse.analysis.claim-profile-v1"),
            output_claims: vec![code("muse.analysis.confirmatory-under-profile")],
            explicit_nonclaims: vec![code("muse.analysis.causal-generalization")],
        }
    }

    fn canonical_receipt() -> ClaimDerivationReceiptProjectionV1 {
        let plan = claim_scope(
            source_ref("plan", 0x11, 0x61),
            "muse.plan.claim-profile-v1",
            &["muse.plan.bound"],
            &[],
        );
        let results = claim_scope(
            source_ref("results", 0x22, 0x62),
            "muse.results.claim-profile-v1",
            &["muse.results.rederived"],
            &[],
        );
        let output = claim_scope(
            source_ref("analysis", 0x33, 0x63),
            "muse.analysis.claim-profile-v1",
            &["muse.analysis.confirmatory-under-profile"],
            &["muse.analysis.causal-generalization"],
        );
        let mechanism = id("muse.derivation-activity", "analysis-run", 0x44);

        let plan_ref = ProvenanceNodeReferenceV1::SourceRef(plan.source_ref.clone());
        let results_ref = ProvenanceNodeReferenceV1::SourceRef(results.source_ref.clone());
        let output_ref = ProvenanceNodeReferenceV1::SourceRef(output.source_ref.clone());
        let mechanism_ref = ProvenanceNodeReferenceV1::SemanticId(mechanism.clone());

        let mut nodes = vec![
            ProvenanceNodeV1 {
                class: ProvenanceNodeClassV1::EvidenceArtifact,
                reference: plan_ref.clone(),
            },
            ProvenanceNodeV1 {
                class: ProvenanceNodeClassV1::EvidenceArtifact,
                reference: results_ref.clone(),
            },
            ProvenanceNodeV1 {
                class: ProvenanceNodeClassV1::EvidenceArtifact,
                reference: output_ref.clone(),
            },
            ProvenanceNodeV1::from_semantic_id(
                ProvenanceNodeClassV1::Activity,
                mechanism.clone(),
            ),
        ];
        nodes.sort_by_key(|node| node_reference_bytes(&node.reference).unwrap());

        let edges = vec![
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Derivation(
                    ProvenanceDerivationRelationV1::DerivedFrom,
                ),
                from: output_ref.clone(),
                to: plan_ref,
            },
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Derivation(
                    ProvenanceDerivationRelationV1::DerivedFrom,
                ),
                from: output_ref.clone(),
                to: results_ref,
            },
            ProvenanceEdgeV1 {
                relation: ProvenanceRelationV1::Association(
                    ProvenanceAssociationRelationV1::GeneratedBy,
                ),
                from: output_ref,
                to: mechanism_ref,
            },
        ];

        ClaimDerivationReceiptProjectionV1 {
            receipt_version: CLAIM_DERIVATION_RECEIPT_VERSION_V1.into(),
            profile: profile(mechanism),
            inputs: vec![
                ClaimDerivationInputProjectionV1 {
                    role_id: code("input.plan"),
                    claim_scope: plan,
                },
                ClaimDerivationInputProjectionV1 {
                    role_id: code("input.results"),
                    claim_scope: results,
                },
            ],
            output,
            provenance: ProvenanceGraphV1 {
                graph_version: PROVENANCE_GRAPH_VERSION_V1.into(),
                nodes,
                edges,
            },
        }
    }

    #[test]
    fn canonical_receipt_is_structurally_valid() {
        let receipt = canonical_receipt();
        receipt.validate().unwrap();
        assert!(!receipt.semantic_payload().unwrap().as_bytes().is_empty());
    }

    #[test]
    fn absent_prerequisite_fails_closed() {
        let mut receipt = canonical_receipt();
        receipt.inputs[0].claim_scope.establishes.clear();
        assert!(matches!(
            receipt.validate(),
            Err(ClaimDerivationError::PrerequisiteNotDeclared { .. })
        ));
    }

    #[test]
    fn explicit_nonclaim_cannot_satisfy_prerequisite() {
        let mut receipt = canonical_receipt();
        receipt.inputs[0].claim_scope.establishes.clear();
        receipt.inputs[0].claim_scope.explicit_nonclaims = vec![code("muse.plan.bound")];
        assert!(matches!(
            receipt.validate(),
            Err(ClaimDerivationError::PrerequisiteExplicitNonclaim { .. })
        ));
    }

    #[test]
    fn output_cannot_exceed_profile_claim_ceiling() {
        let mut receipt = canonical_receipt();
        receipt
            .output
            .establishes
            .push(code("muse.analysis.artistic-quality"));
        receipt.output.establishes.sort();
        assert_eq!(
            receipt.validate(),
            Err(ClaimDerivationError::OutputClaimsMismatch)
        );
    }

    #[test]
    fn projected_input_requires_explicit_loss_allowance() {
        let mut receipt = canonical_receipt();
        receipt.inputs[0].claim_scope.source_ref.preservation =
            SemanticPreservationV1::ProjectedWithLoss {
                loss_codes: vec![code("muse.loss.display-metadata")],
            };
        assert!(matches!(
            receipt.validate(),
            Err(ClaimDerivationError::InputPreservationRejected { .. })
        ));
    }

    #[test]
    fn preservation_policy_accepts_lossless_and_only_registered_losses() {
        let requirement = InputPreservationRequirementV1::AllowProjectedWithLoss {
            allowed_loss_codes: vec![code("muse.loss.display-metadata")],
        };
        assert!(requirement.accepts(&SemanticPreservationV1::LosslessUnderProfile));
        assert!(requirement.accepts(&SemanticPreservationV1::ProjectedWithLoss {
            loss_codes: vec![code("muse.loss.display-metadata")],
        }));
        assert!(!requirement.accepts(&SemanticPreservationV1::ProjectedWithLoss {
            loss_codes: vec![code("muse.loss.scientific-semantics")],
        }));
    }

    #[test]
    fn unregistered_loss_rejects() {
        let mut receipt = canonical_receipt();
        receipt.inputs[1].claim_scope.source_ref.preservation =
            SemanticPreservationV1::ProjectedWithLoss {
                loss_codes: vec![code("muse.loss.scientific-semantics")],
            };
        assert!(matches!(
            receipt.validate(),
            Err(ClaimDerivationError::InputPreservationRejected { .. })
        ));
    }

    #[test]
    fn provenance_derivation_path_is_required() {
        let mut receipt = canonical_receipt();
        let target = ProvenanceNodeReferenceV1::SourceRef(
            receipt.inputs[0].claim_scope.source_ref.clone(),
        );
        receipt
            .provenance
            .edges
            .retain(|edge| !(edge.relation.is_derivation() && edge.to == target));
        assert!(matches!(
            receipt.validate(),
            Err(ClaimDerivationError::InputNotDerivationAncestor { .. })
        ));
    }

    #[test]
    fn exact_mechanism_generated_by_link_is_required() {
        let mut receipt = canonical_receipt();
        receipt.provenance.edges.retain(|edge| {
            edge.relation
                != ProvenanceRelationV1::Association(
                    ProvenanceAssociationRelationV1::GeneratedBy,
                )
        });
        assert_eq!(
            receipt.validate(),
            Err(ClaimDerivationError::OutputNotGeneratedByMechanism)
        );
    }

    #[test]
    fn validated_receipt_remains_declaration_only_typestate() {
        let receipt = canonical_receipt();
        let validated =
            ValidatedClaimDerivationReceiptProjectionV1::try_from(receipt.clone()).unwrap();
        assert_eq!(
            validated.as_raw().output.establishes,
            receipt.profile.output_claims
        );
        assert!(!validated.semantic_payload().as_bytes().is_empty());
    }
}
