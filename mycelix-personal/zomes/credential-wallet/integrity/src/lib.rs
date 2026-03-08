//! Credential Wallet Integrity Zome
//!
//! Defines entry types for storing verifiable credentials (VCs) issued
//! by any cluster (Civic, Commons, or external issuers). Credentials
//! are stored privately; selective presentation happens via personal_bridge.
//!
//! Also includes K-Vector trust credential types migrated from `mycelix-identity`
//! for privacy-preserving trust attestations with ZKP proofs.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};
use personal_types::{
    AttestationStatus, CredentialType, KVectorComponent, TrustScoreRange, TrustTier,
};

// ============================================================================
// Generic Credential Storage
// ============================================================================

/// A verifiable credential stored in the agent's wallet.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StoredCredential {
    /// Type of credential.
    pub credential_type: CredentialType,
    /// The credential payload (JSON-LD, JWT, or custom format).
    pub credential_data: String,
    /// DID or agent key of the issuer.
    pub issuer: String,
    /// When the credential was issued.
    pub issued_at: Timestamp,
    /// When the credential expires (None = no expiry).
    pub expires_at: Option<Timestamp>,
    /// Whether this credential has been revoked by the holder.
    pub revoked: bool,
}

/// A proof derived from a stored credential for selective presentation.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CredentialProof {
    /// Hash of the source credential.
    pub credential_hash: ActionHash,
    /// The proof payload (e.g., ZK proof, signed subset).
    pub proof_data: String,
    /// What was proven (human-readable description).
    pub claim: String,
    /// When this proof was generated.
    pub created_at: Timestamp,
}

// ============================================================================
// K-Vector Trust Credentials
// ============================================================================

/// K-Vector Trust Credential
///
/// A verifiable credential that attests to an agent's K-Vector trust profile
/// without revealing the individual component values. The commitment is
/// public (verifiable) while the actual K-Vector values remain private (hidden via ZKP).
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustCredential {
    /// Unique credential identifier.
    pub id: String,
    /// Subject's DID (who this credential is about).
    pub subject_did: String,
    /// Issuer's DID (who issued this credential).
    pub issuer_did: String,
    /// K-Vector commitment hash (32 bytes, SHA3-256).
    pub kvector_commitment: Vec<u8>,
    /// STARK proof that K-Vector components are in valid [0,1] range.
    pub range_proof: Vec<u8>,
    /// Proven trust score range (e.g., [0.5, 0.7]).
    pub trust_score_range: TrustScoreRange,
    /// Trust tier derived from K-Vector (for governance thresholds).
    pub trust_tier: TrustTier,
    /// Credential issuance timestamp.
    pub issued_at: Timestamp,
    /// Credential expiration (None = never).
    pub expires_at: Option<Timestamp>,
    /// Whether credential has been revoked.
    pub revoked: bool,
    /// Revocation reason if revoked.
    pub revocation_reason: Option<String>,
    /// When the credential was revoked (for audit trails).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revoked_at: Option<Timestamp>,
    /// Previous credential this supersedes (for updates).
    pub supersedes: Option<String>,
}

/// K-Vector Attestation Request
///
/// A request for an agent to attest to components of their K-Vector.
/// Used when governance or another cluster needs to verify specific trust properties.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AttestationRequest {
    /// Request identifier.
    pub id: String,
    /// Who is requesting attestation.
    pub requester_did: String,
    /// Who should provide attestation.
    pub subject_did: String,
    /// Which K-Vector components need attestation.
    pub components: Vec<KVectorComponent>,
    /// Minimum acceptable trust score.
    pub min_trust_score: Option<f32>,
    /// Minimum acceptable tier.
    pub min_tier: Option<TrustTier>,
    /// Purpose of the attestation.
    pub purpose: String,
    /// Request expiration.
    pub expires_at: Timestamp,
    /// Request status.
    pub status: AttestationStatus,
    /// Creation timestamp.
    pub created_at: Timestamp,
}

/// Trust Credential Presentation
///
/// A selective disclosure presentation of a trust credential.
/// Can reveal specific attributes while keeping others private.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustPresentation {
    /// Presentation identifier.
    pub id: String,
    /// Reference to source credential.
    pub credential_id: String,
    /// Subject's DID.
    pub subject_did: String,
    /// Disclosed trust tier (always disclosed).
    pub disclosed_tier: TrustTier,
    /// Disclosed trust score range (if disclosed).
    pub disclosed_range: Option<TrustScoreRange>,
    /// Presentation proof (derived from original proof).
    pub presentation_proof: Vec<u8>,
    /// Who this presentation is for.
    pub verifier_did: Option<String>,
    /// Purpose of presentation.
    pub purpose: String,
    /// Presentation timestamp.
    pub presented_at: Timestamp,
    /// Single-use nonce to prevent replay.
    pub nonce: Vec<u8>,
}

// ============================================================================
// Entry & Link Type Registration
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    StoredCredential(StoredCredential),
    CredentialProof(CredentialProof),
    TrustCredential(TrustCredential),
    AttestationRequest(AttestationRequest),
    TrustPresentation(TrustPresentation),
}

#[hdk_link_types]
pub enum LinkTypes {
    // Generic credential links
    AgentToCredentials,
    AgentToProofs,
    CredentialTypeToCredential,
    // Trust credential links
    SubjectToTrustCredential,
    IssuerToTrustCredential,
    SubjectToRequest,
    CredentialToPresentation,
    TierToTrustCredential,
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::StoredCredential(cred) => validate_credential(&cred),
                EntryTypes::CredentialProof(proof) => validate_proof(&proof),
                EntryTypes::TrustCredential(cred) => validate_create_trust_credential(action, cred),
                EntryTypes::AttestationRequest(req) => validate_create_request(req),
                EntryTypes::TrustPresentation(pres) => validate_create_presentation(pres),
            },
            OpEntry::UpdateEntry { app_entry, action, .. } => match app_entry {
                EntryTypes::StoredCredential(cred) => validate_credential(&cred),
                EntryTypes::CredentialProof(proof) => validate_proof(&proof),
                EntryTypes::TrustCredential(cred) => validate_update_trust_credential(action, cred),
                EntryTypes::AttestationRequest(req) => validate_update_request(action, req),
                EntryTypes::TrustPresentation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Trust presentations cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::AgentToCredentials
                | LinkTypes::AgentToProofs
                | LinkTypes::CredentialTypeToCredential
                | LinkTypes::SubjectToTrustCredential
                | LinkTypes::IssuerToTrustCredential
                | LinkTypes::SubjectToRequest
                | LinkTypes::CredentialToPresentation
                | LinkTypes::TierToTrustCredential => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) | FlatOp::RegisterAgentActivity(_) => {
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

// --- Generic credential validation ---

fn validate_credential(cred: &StoredCredential) -> ExternResult<ValidateCallbackResult> {
    if cred.credential_data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "StoredCredential credential_data cannot be empty".into(),
        ));
    }
    if cred.credential_data.len() > 65536 {
        return Ok(ValidateCallbackResult::Invalid(
            "StoredCredential credential_data must be <= 65536 bytes".into(),
        ));
    }
    if cred.issuer.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "StoredCredential issuer cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_proof(proof: &CredentialProof) -> ExternResult<ValidateCallbackResult> {
    if proof.proof_data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "CredentialProof proof_data cannot be empty".into(),
        ));
    }
    if proof.proof_data.len() > 32768 {
        return Ok(ValidateCallbackResult::Invalid(
            "CredentialProof proof_data must be <= 32768 bytes".into(),
        ));
    }
    if proof.claim.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "CredentialProof claim cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// --- Trust credential validation ---

fn validate_create_trust_credential(
    action: Create,
    cred: TrustCredential,
) -> ExternResult<ValidateCallbackResult> {
    // Issuer must be the action author (prevent impersonation)
    let expected_issuer = format!("did:mycelix:{}", action.author);
    if cred.issuer_did != expected_issuer {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Issuer DID must match action author. Expected '{}', got '{}'",
                expected_issuer, cred.issuer_did
            ),
        ));
    }

    // Subject must be a valid DID
    if !cred.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
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

    // Trust score range must be finite and valid
    if !cred.trust_score_range.lower.is_finite() || !cred.trust_score_range.upper.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score range values must be finite (no NaN or Infinity)".into(),
        ));
    }
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
    if (cred.trust_score_range.upper as f64) < tier_min {
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

    // New credentials must not have a revocation timestamp
    if cred.revoked_at.is_some() {
        return Ok(ValidateCallbackResult::Invalid(
            "New credentials cannot have a revocation timestamp".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_trust_credential(
    action: Update,
    cred: TrustCredential,
) -> ExternResult<ValidateCallbackResult> {
    if cred.kvector_commitment.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment must be 32 bytes".into(),
        ));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: TrustCredential = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original trust credential not found".into()
        )))?;

    // Immutable fields
    if cred.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust credential ID cannot be changed".into(),
        ));
    }
    if cred.subject_did != original.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject DID cannot be changed".into(),
        ));
    }
    if cred.issuer_did != original.issuer_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuer DID cannot be changed".into(),
        ));
    }
    if cred.kvector_commitment != original.kvector_commitment {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector commitment cannot be changed".into(),
        ));
    }
    if cred.issued_at != original.issued_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Issuance timestamp cannot be changed".into(),
        ));
    }

    // Revocation is irreversible
    if original.revoked && !cred.revoked {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust credential revocation is irreversible".into(),
        ));
    }

    // If being revoked, revoked_at must be set
    if cred.revoked && cred.revoked_at.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Revoked credentials must have a revoked_at timestamp".into(),
        ));
    }

    // revoked_at is immutable once set
    if let Some(original_ts) = &original.revoked_at {
        match &cred.revoked_at {
            Some(new_ts) if new_ts != original_ts => {
                return Ok(ValidateCallbackResult::Invalid(
                    "Revocation timestamp cannot be changed once set".into(),
                ));
            }
            None => {
                return Ok(ValidateCallbackResult::Invalid(
                    "Revocation timestamp cannot be removed".into(),
                ));
            }
            _ => {}
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_request(req: AttestationRequest) -> ExternResult<ValidateCallbackResult> {
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    if !req.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    if req.requester_did == req.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot request attestation from yourself".into(),
        ));
    }

    if req.status != AttestationStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New requests must have Pending status".into(),
        ));
    }

    if let Some(score) = req.min_trust_score {
        if !score.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Minimum trust score must be finite (no NaN or Infinity)".into(),
            ));
        }
        if !(0.0..=1.0).contains(&score) {
            return Ok(ValidateCallbackResult::Invalid(
                "Minimum trust score must be in [0, 1]".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_request(
    action: Update,
    req: AttestationRequest,
) -> ExternResult<ValidateCallbackResult> {
    if !req.requester_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester must be a valid DID".into(),
        ));
    }

    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: AttestationRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original attestation request not found".into()
        )))?;

    // Immutable fields
    if req.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation request ID cannot be changed".into(),
        ));
    }
    if req.requester_did != original.requester_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Requester DID cannot be changed".into(),
        ));
    }
    if req.subject_did != original.subject_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject DID cannot be changed".into(),
        ));
    }

    // State machine: Pending → Fulfilled/Declined/Expired/Cancelled
    let valid = match (&original.status, &req.status) {
        (AttestationStatus::Pending, AttestationStatus::Fulfilled)
        | (AttestationStatus::Pending, AttestationStatus::Declined)
        | (AttestationStatus::Pending, AttestationStatus::Expired)
        | (AttestationStatus::Pending, AttestationStatus::Cancelled) => true,
        (a, b) if a == b => true,
        _ => false,
    };

    if !valid {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid attestation request status transition".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_presentation(pres: TrustPresentation) -> ExternResult<ValidateCallbackResult> {
    if !pres.subject_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject must be a valid DID".into(),
        ));
    }

    if pres.presentation_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Presentation proof cannot be empty".into(),
        ));
    }

    if pres.nonce.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Nonce is required for replay protection".into(),
        ));
    }

    if let Some(ref range) = pres.disclosed_range {
        if !range.lower.is_finite() || !range.upper.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Disclosed range values must be finite (no NaN or Infinity)".into(),
            ));
        }
        if range.lower < 0.0 || range.upper > 1.0 || range.lower > range.upper {
            return Ok(ValidateCallbackResult::Invalid(
                "Disclosed range must be valid".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// Pure verification logic (no HDK calls, fully testable)
// ============================================================================

/// Verify trust credential properties without HDK calls.
/// Returns (commitment_valid, range_valid, tier_consistent, proof_format_valid).
pub fn verify_credential_pure(
    kvector_commitment: &[u8],
    trust_score_range: &TrustScoreRange,
    trust_tier: &TrustTier,
    range_proof: &[u8],
) -> (bool, bool, bool, bool) {
    let commitment_valid =
        kvector_commitment.len() == 32 && kvector_commitment.iter().any(|&b| b != 0);

    let range_valid = trust_score_range.lower.is_finite()
        && trust_score_range.upper.is_finite()
        && trust_score_range.lower >= 0.0
        && trust_score_range.upper <= 1.0
        && trust_score_range.lower <= trust_score_range.upper;

    let mid_score =
        (trust_score_range.lower as f64 + trust_score_range.upper as f64) / 2.0;
    let expected_tier = TrustTier::from_score(mid_score);
    let tier_consistent = range_valid && *trust_tier == expected_tier;

    let proof_format_valid = !range_proof.is_empty();

    (commitment_valid, range_valid, tier_consistent, proof_format_valid)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_credential(ctype: CredentialType) -> StoredCredential {
        StoredCredential {
            credential_type: ctype,
            credential_data: r#"{"vc":"test"}"#.into(),
            issuer: "did:key:z6MkTest".into(),
            issued_at: Timestamp::from_micros(0),
            expires_at: None,
            revoked: false,
        }
    }

    fn make_proof() -> CredentialProof {
        CredentialProof {
            credential_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            proof_data: r#"{"zk_proof":"abc123"}"#.into(),
            claim: "holder has valid identity credential".into(),
            created_at: Timestamp::from_micros(0),
        }
    }

    fn valid_commitment() -> Vec<u8> {
        let mut c = vec![0u8; 32];
        c[0] = 1;
        c
    }

    // --- Generic credential tests ---

    #[test]
    fn valid_identity_credential_passes() {
        let c = make_credential(CredentialType::Identity);
        assert!(matches!(validate_credential(&c).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn valid_fl_credential_passes() {
        let c = make_credential(CredentialType::FederatedLearning);
        assert!(matches!(validate_credential(&c).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn empty_credential_data_rejected() {
        let mut c = make_credential(CredentialType::Identity);
        c.credential_data = String::new();
        match validate_credential(&c).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn oversized_credential_data_rejected() {
        let mut c = make_credential(CredentialType::Identity);
        c.credential_data = "x".repeat(65537);
        match validate_credential(&c).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("65536")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn empty_issuer_rejected() {
        let mut c = make_credential(CredentialType::Identity);
        c.issuer = String::new();
        match validate_credential(&c).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn valid_proof_passes() {
        let p = make_proof();
        assert!(matches!(validate_proof(&p).unwrap(), ValidateCallbackResult::Valid));
    }

    #[test]
    fn empty_proof_data_rejected() {
        let mut p = make_proof();
        p.proof_data = String::new();
        match validate_proof(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn oversized_proof_data_rejected() {
        let mut p = make_proof();
        p.proof_data = "x".repeat(32769);
        match validate_proof(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("32768")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn empty_claim_rejected() {
        let mut p = make_proof();
        p.claim = String::new();
        match validate_proof(&p).unwrap() {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("empty")),
            other => panic!("expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn credential_serde_roundtrip() {
        let c = make_credential(CredentialType::Governance);
        let json = serde_json::to_string(&c).unwrap();
        let back: StoredCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(back, c);
    }

    #[test]
    fn proof_serde_roundtrip() {
        let p = make_proof();
        let json = serde_json::to_string(&p).unwrap();
        let back: CredentialProof = serde_json::from_str(&json).unwrap();
        assert_eq!(back, p);
    }

    // --- Trust credential pure verification tests ---

    #[test]
    fn verify_valid_trust_credential() {
        let (cv, rv, tc, pv) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.3, upper: 0.39 },
            &TrustTier::Basic, // mid = 0.345 → Basic
            &[1, 2, 3],
        );
        assert!(cv && rv && tc && pv);
    }

    #[test]
    fn verify_commitment_too_short() {
        let (cv, _, _, _) = verify_credential_pure(
            &[1u8; 16],
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer,
            &[1],
        );
        assert!(!cv);
    }

    #[test]
    fn verify_commitment_all_zeros() {
        let (cv, _, _, _) = verify_credential_pure(
            &[0u8; 32],
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer,
            &[1],
        );
        assert!(!cv);
    }

    #[test]
    fn verify_range_invalid_nan() {
        let (_, rv, _, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: f32::NAN, upper: 0.5 },
            &TrustTier::Basic,
            &[1],
        );
        assert!(!rv);
    }

    #[test]
    fn verify_range_inverted() {
        let (_, rv, _, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.8, upper: 0.2 },
            &TrustTier::Basic,
            &[1],
        );
        assert!(!rv);
    }

    #[test]
    fn verify_range_exceeds_one() {
        let (_, rv, _, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.5, upper: 1.5 },
            &TrustTier::Elevated,
            &[1],
        );
        assert!(!rv);
    }

    #[test]
    fn verify_tier_mismatch() {
        // mid = 0.5 → Standard, but claiming Guardian
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.4, upper: 0.59 },
            &TrustTier::Guardian,
            &[1],
        );
        assert!(!tc);
    }

    #[test]
    fn verify_tier_observer() {
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer, // mid = 0.1 → Observer
            &[1],
        );
        assert!(tc);
    }

    #[test]
    fn verify_tier_guardian() {
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.85, upper: 0.95 },
            &TrustTier::Guardian, // mid = 0.9 → Guardian
            &[1],
        );
        assert!(tc);
    }

    #[test]
    fn verify_tier_elevated() {
        let (_, _, tc, _) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.6, upper: 0.79 },
            &TrustTier::Elevated, // mid = 0.695 → Elevated
            &[1],
        );
        assert!(tc);
    }

    #[test]
    fn verify_empty_proof() {
        let (_, _, _, pv) = verify_credential_pure(
            &valid_commitment(),
            &TrustScoreRange { lower: 0.0, upper: 0.2 },
            &TrustTier::Observer,
            &[],
        );
        assert!(!pv);
    }

    #[test]
    fn trust_credential_serde_roundtrip() {
        let cred = TrustCredential {
            id: "trust-cred:test:123".into(),
            subject_did: "did:mycelix:alice".into(),
            issuer_did: "did:mycelix:bob".into(),
            kvector_commitment: valid_commitment(),
            range_proof: vec![1, 2, 3],
            trust_score_range: TrustScoreRange { lower: 0.4, upper: 0.6 },
            trust_tier: TrustTier::Standard,
            issued_at: Timestamp::from_micros(1_000_000),
            expires_at: None,
            revoked: false,
            revocation_reason: None,
            revoked_at: None,
            supersedes: None,
        };
        let json = serde_json::to_string(&cred).unwrap();
        let back: TrustCredential = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cred);
    }

    #[test]
    fn attestation_request_serde_roundtrip() {
        let req = AttestationRequest {
            id: "req:alice:bob:123".into(),
            requester_did: "did:mycelix:alice".into(),
            subject_did: "did:mycelix:bob".into(),
            components: vec![KVectorComponent::Reputation, KVectorComponent::Integrity],
            min_trust_score: Some(0.4),
            min_tier: Some(TrustTier::Standard),
            purpose: "governance vote".into(),
            expires_at: Timestamp::from_micros(2_000_000),
            status: AttestationStatus::Pending,
            created_at: Timestamp::from_micros(1_000_000),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: AttestationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn trust_presentation_serde_roundtrip() {
        let pres = TrustPresentation {
            id: "pres:alice:123".into(),
            credential_id: "trust-cred:alice:456".into(),
            subject_did: "did:mycelix:alice".into(),
            disclosed_tier: TrustTier::Standard,
            disclosed_range: Some(TrustScoreRange { lower: 0.4, upper: 0.6 }),
            presentation_proof: vec![1, 2, 3],
            verifier_did: Some("did:mycelix:governance".into()),
            purpose: "voting eligibility".into(),
            presented_at: Timestamp::from_micros(1_000_000),
            nonce: vec![42, 43, 44],
        };
        let json = serde_json::to_string(&pres).unwrap();
        let back: TrustPresentation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, pres);
    }

    // --- Validation edge case tests ---

    #[test]
    fn entry_types_all_exist() {
        let _cred = UnitEntryTypes::StoredCredential;
        let _proof = UnitEntryTypes::CredentialProof;
        let _trust = UnitEntryTypes::TrustCredential;
        let _req = UnitEntryTypes::AttestationRequest;
        let _pres = UnitEntryTypes::TrustPresentation;
    }

    #[test]
    fn link_types_all_exist() {
        let _creds = LinkTypes::AgentToCredentials;
        let _proofs = LinkTypes::AgentToProofs;
        let _type_to_cred = LinkTypes::CredentialTypeToCredential;
        let _subj = LinkTypes::SubjectToTrustCredential;
        let _issuer = LinkTypes::IssuerToTrustCredential;
        let _req = LinkTypes::SubjectToRequest;
        let _pres = LinkTypes::CredentialToPresentation;
        let _tier = LinkTypes::TierToTrustCredential;
    }

    // --- is_finite() guard tests ---

    #[test]
    fn verify_credential_pure_rejects_nan_scores() {
        let commitment = vec![1u8; 32];
        let range = TrustScoreRange { lower: f32::NAN, upper: 0.5 };
        let (_, range_valid, _, _) = verify_credential_pure(
            &commitment, &range, &TrustTier::Basic, &[1, 2, 3],
        );
        assert!(!range_valid, "NaN lower should fail range validation");

        let range2 = TrustScoreRange { lower: 0.3, upper: f32::NAN };
        let (_, range_valid2, _, _) = verify_credential_pure(
            &commitment, &range2, &TrustTier::Basic, &[1, 2, 3],
        );
        assert!(!range_valid2, "NaN upper should fail range validation");
    }

    #[test]
    fn verify_credential_pure_rejects_infinity_scores() {
        let commitment = vec![1u8; 32];
        let range = TrustScoreRange { lower: f32::INFINITY, upper: 0.5 };
        let (_, range_valid, _, _) = verify_credential_pure(
            &commitment, &range, &TrustTier::Basic, &[1, 2, 3],
        );
        assert!(!range_valid, "Infinity lower should fail range validation");

        let range2 = TrustScoreRange { lower: 0.3, upper: f32::NEG_INFINITY };
        let (_, range_valid2, _, _) = verify_credential_pure(
            &commitment, &range2, &TrustTier::Basic, &[1, 2, 3],
        );
        assert!(!range_valid2, "Negative infinity upper should fail range validation");
    }
}
