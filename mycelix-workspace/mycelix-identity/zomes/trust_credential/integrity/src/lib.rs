// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Trust Credential Integrity Zome
//!
//! Defines entry types for K-Vector trust credentials with ZKP proofs.
//! Enables privacy-preserving trust attestations where:
//! - The K-Vector commitment is public (can be verified)
//! - The actual K-Vector values are private (hidden via ZKP)
//! - The trust score can be proven in a range without revealing exact value
//!
//! Integration with kvector-zkp library:
//! - KVectorWitness.commitment() produces the 32-byte commitment
//! - KVectorRangeProof proves values are in valid [0,1] range
//! - Proofs are generated off-chain and verified on-chain

use hdi::prelude::*;

/// K-Vector Trust Credential
///
/// A verifiable credential that attests to an agent's K-Vector trust profile
/// without revealing the individual component values.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustCredential {
    /// Unique credential identifier
    pub id: String,
    /// Subject's DID (who this credential is about)
    pub subject_did: String,
    /// Issuer's DID (who issued this credential)
    pub issuer_did: String,
    /// K-Vector commitment hash (32 bytes, SHA3-256)
    /// This binds to the actual K-Vector without revealing values
    pub kvector_commitment: Vec<u8>,
    /// STARK proof that K-Vector components are in valid [0,1] range
    /// Serialized proof from kvector-zkp library
    pub range_proof: Vec<u8>,
    /// Proven trust score range (e.g., [0.5, 0.7])
    /// Proves score is within this range without revealing exact value
    pub trust_score_range: TrustScoreRange,
    /// Trust tier derived from K-Vector (for governance thresholds)
    pub trust_tier: TrustTier,
    /// Credential issuance timestamp
    pub issued_at: Timestamp,
    /// Credential expiration (None = never)
    pub expires_at: Option<Timestamp>,
    /// Whether credential has been revoked
    pub revoked: bool,
    /// Revocation reason if revoked
    pub revocation_reason: Option<String>,
    /// Previous credential this supersedes (for updates)
    pub supersedes: Option<String>,
}

/// Trust score range (privacy-preserving)
///
/// Proves the trust score falls within a range without revealing exact value.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TrustScoreRange {
    /// Lower bound (inclusive)
    pub lower: f32,
    /// Upper bound (inclusive)
    pub upper: f32,
}

/// Trust tiers for governance participation
///
/// Derived from K-Vector trust_score() with defined thresholds.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TrustTier {
    /// Trust score < 0.3 - Observer only, cannot vote
    Observer,
    /// Trust score >= 0.3 - Basic participation
    Basic,
    /// Trust score >= 0.4 - Can vote on major proposals
    Standard,
    /// Trust score >= 0.6 - Can propose and vote on constitutional changes
    Elevated,
    /// Trust score >= 0.8 - Full governance rights including emergency powers
    Guardian,
}

impl TrustTier {
    /// Get the minimum trust score for this tier
    pub fn min_score(&self) -> f32 {
        match self {
            TrustTier::Observer => 0.0,
            TrustTier::Basic => 0.3,
            TrustTier::Standard => 0.4,
            TrustTier::Elevated => 0.6,
            TrustTier::Guardian => 0.8,
        }
    }

    /// Determine tier from trust score
    pub fn from_score(score: f32) -> Self {
        if score >= 0.8 {
            TrustTier::Guardian
        } else if score >= 0.6 {
            TrustTier::Elevated
        } else if score >= 0.4 {
            TrustTier::Standard
        } else if score >= 0.3 {
            TrustTier::Basic
        } else {
            TrustTier::Observer
        }
    }
}

/// K-Vector Attestation Request
///
/// A request for someone to attest to components of their K-Vector.
/// Used when an issuer needs to verify specific trust properties.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AttestationRequest {
    /// Request identifier
    pub id: String,
    /// Who is requesting attestation
    pub requester_did: String,
    /// Who should provide attestation
    pub subject_did: String,
    /// Which K-Vector components need attestation
    pub components: Vec<KVectorComponent>,
    /// Minimum acceptable trust score
    pub min_trust_score: Option<f32>,
    /// Minimum acceptable tier
    pub min_tier: Option<TrustTier>,
    /// Purpose of the attestation
    pub purpose: String,
    /// Request expiration
    pub expires_at: Timestamp,
    /// Request status
    pub status: AttestationStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
}

/// K-Vector component identifiers
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum KVectorComponent {
    /// k_r: Reputation
    Reputation,
    /// k_a: Activity
    Activity,
    /// k_i: Integrity
    Integrity,
    /// k_p: Performance
    Performance,
    /// k_m: Membership duration
    Membership,
    /// k_s: Stake weight
    Stake,
    /// k_h: Historical consistency
    History,
    /// k_topo: Network topology contribution
    Topology,
}

/// Status of an attestation request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AttestationStatus {
    /// Waiting for response
    Pending,
    /// Attestation provided
    Fulfilled,
    /// Request was declined
    Declined,
    /// Request expired
    Expired,
    /// Request was cancelled
    Cancelled,
}

/// Trust Credential Presentation
///
/// A selective disclosure presentation of a trust credential.
/// Can reveal specific attributes while keeping others private.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustPresentation {
    /// Presentation identifier
    pub id: String,
    /// Reference to source credential
    pub credential_id: String,
    /// Subject's DID
    pub subject_did: String,
    /// Disclosed trust tier (always disclosed)
    pub disclosed_tier: TrustTier,
    /// Disclosed trust score range (if disclosed)
    pub disclosed_range: Option<TrustScoreRange>,
    /// Presentation proof (derived from original proof)
    pub presentation_proof: Vec<u8>,
    /// Who this presentation is for
    pub verifier_did: Option<String>,
    /// Purpose of presentation
    pub purpose: String,
    /// Presentation timestamp
    pub presented_at: Timestamp,
    /// Single-use nonce to prevent replay
    pub nonce: Vec<u8>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    TrustCredential(TrustCredential),
    AttestationRequest(AttestationRequest),
    TrustPresentation(TrustPresentation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Subject to their trust credentials
    SubjectToCredential,
    /// Issuer to credentials they issued
    IssuerToCredential,
    /// Subject to attestation requests they received
    SubjectToRequest,
    /// Credential to presentations derived from it
    CredentialToPresentation,
    /// Trust tier anchor to credentials in that tier
    TierToCredential,
}

/// Main validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::TrustCredential(cred) => validate_create_credential(action, cred),
                EntryTypes::AttestationRequest(req) => validate_create_request(action, req),
                EntryTypes::TrustPresentation(pres) => validate_create_presentation(action, pres),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::TrustCredential(cred) => validate_update_credential(action, cred),
                EntryTypes::AttestationRequest(req) => validate_update_request(action, req),
                EntryTypes::TrustPresentation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Trust presentations cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::SubjectToCredential => Ok(ValidateCallbackResult::Valid),
            LinkTypes::IssuerToCredential => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SubjectToRequest => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CredentialToPresentation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TierToCredential => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate trust credential creation
fn validate_create_credential(
    _action: Create,
    cred: TrustCredential,
) -> ExternResult<ValidateCallbackResult> {
    // Subject must be a valid DID
    if !cred.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    // Issuer must be a valid DID
    if !cred.issuer_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer must be a valid DID".into(),
        ));
    }

    // K-Vector commitment must be 32 bytes (SHA3-256)
    if cred.kvector_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment must be 32 bytes".into(),
        ));
    }

    // Range proof must not be empty
    if cred.range_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Range proof cannot be empty".into(),
        ));
    }

    // Trust score range must be valid
    if cred.trust_score_range.lower < 0.0 || cred.trust_score_range.upper > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range must be within [0, 1]".into(),
        ));
    }

    if cred.trust_score_range.lower > cred.trust_score_range.upper {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range lower bound cannot exceed upper bound".into(),
        ));
    }

    // Trust tier must be consistent with range
    let tier_min = cred.trust_tier.min_score();
    if cred.trust_score_range.upper < tier_min {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range is inconsistent with claimed tier".into(),
        ));
    }

    // New credentials cannot be revoked
    if cred.revoked {
        return Ok(ValidateCallbackResult::Invalid(
            "New credentials cannot be created in revoked state".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate trust credential update (for revocation)
fn validate_update_credential(
    _action: Update,
    cred: TrustCredential,
) -> ExternResult<ValidateCallbackResult> {
    // Can only update to revoke or supersede
    // Note: Full validation would check original entry
    // Here we just ensure updated entry is valid

    if cred.kvector_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment must be 32 bytes".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate attestation request creation
fn validate_create_request(
    _action: Create,
    req: AttestationRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Requester must be a valid DID
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    // Subject must be a valid DID
    if !req.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    // Cannot request attestation from yourself
    if req.requester_did == req.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot request attestation from yourself".into(),
        ));
    }

    // New requests must be pending
    if req.status != AttestationStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New requests must have Pending status".into(),
        ));
    }

    // Min trust score must be valid if specified
    if let Some(score) = req.min_trust_score {
        if score < 0.0 || score > 1.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Minimum trust score must be in [0, 1]".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate attestation request update
fn validate_update_request(
    _action: Update,
    req: AttestationRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Basic validation for updated request
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate trust presentation creation
fn validate_create_presentation(
    _action: Create,
    pres: TrustPresentation,
) -> ExternResult<ValidateCallbackResult> {
    // Subject must be a valid DID
    if !pres.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    // Presentation proof must not be empty
    if pres.presentation_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Presentation proof cannot be empty".into(),
        ));
    }

    // Nonce must be present for replay protection
    if pres.nonce.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Nonce is required for replay protection".into(),
        ));
    }

    // If range is disclosed, it must be valid
    if let Some(ref range) = pres.disclosed_range {
        if range.lower < 0.0 || range.upper > 1.0 || range.lower > range.upper {
            return Ok(ValidateCallbackResult::Invalid(
                "Disclosed range must be valid".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}
