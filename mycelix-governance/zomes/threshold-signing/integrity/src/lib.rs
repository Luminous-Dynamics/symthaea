//! Threshold Signing Integrity Zome
//!
//! Entry types and validation for DKG-based threshold signatures.
//! Enables collective signing of governance decisions by a validator committee.
//!
//! Integration with feldman-dkg library:
//! - Committee members run DKG ceremony off-chain
//! - Public commitments and key shares are stored on-chain
//! - Threshold signatures are verified and stored for governance finality

use hdi::prelude::*;

/// Anchor for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A signing committee formed through DKG
///
/// Members coordinate off-chain using feldman-dkg to generate threshold keys.
/// The public key and commitments are stored on-chain for verification.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SigningCommittee {
    /// Unique committee identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Threshold (t) for t-of-n signing
    pub threshold: u32,
    /// Total members (n)
    pub member_count: u32,
    /// DKG ceremony phase
    pub phase: DkgPhase,
    /// Combined public key (from DKG result)
    pub public_key: Option<Vec<u8>>,
    /// Public polynomial commitments (Feldman VSS)
    pub commitments: Vec<Vec<u8>>,
    /// Governance scope (what this committee can sign)
    pub scope: CommitteeScope,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Whether committee is active
    pub active: bool,
    /// Epoch number (for key rotation)
    pub epoch: u32,
}

/// DKG ceremony phases
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DkgPhase {
    /// Waiting for members to register
    Registration,
    /// Collecting deals from members
    Dealing,
    /// Verifying and complaints
    Verification,
    /// DKG complete, ready to sign
    Complete,
    /// Committee disbanded
    Disbanded,
}

/// Governance scope for signing committees
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CommitteeScope {
    /// Can sign any governance decision
    All,
    /// Only constitutional amendments
    Constitutional,
    /// Only treasury operations
    Treasury,
    /// Only protocol upgrades
    Protocol,
    /// Custom scope with specific proposal types
    Custom(Vec<String>),
}

/// A member of a signing committee
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommitteeMember {
    /// Reference to parent committee
    pub committee_id: String,
    /// Member's participant ID (from DKG)
    pub participant_id: u32,
    /// Member's agent public key
    pub agent: AgentPubKey,
    /// Member's DID
    pub member_did: String,
    /// K-Vector trust score at time of joining
    pub trust_score: f64,
    /// Member's public key share (from DKG)
    pub public_share: Option<Vec<u8>>,
    /// Member's VSS commitments
    pub vss_commitment: Option<Vec<u8>>,
    /// Whether member has submitted their deal
    pub deal_submitted: bool,
    /// Whether member is qualified after verification
    pub qualified: bool,
    /// Registration timestamp
    pub registered_at: Timestamp,
}

/// A threshold signature on a governance decision
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ThresholdSignature {
    /// Unique signature identifier
    pub id: String,
    /// Committee that signed
    pub committee_id: String,
    /// What is being signed (proposal ID, tally hash, etc.)
    pub signed_content_hash: Vec<u8>,
    /// Human-readable description of signed content
    pub signed_content_description: String,
    /// Combined threshold signature
    pub signature: Vec<u8>,
    /// Number of signers who contributed
    pub signer_count: u32,
    /// IDs of participating signers
    pub signers: Vec<u32>,
    /// Whether signature has been verified
    pub verified: bool,
    /// Signature timestamp
    pub signed_at: Timestamp,
}

/// Individual signature share from a committee member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SignatureShare {
    /// Reference to the threshold signature being built
    pub signature_id: String,
    /// Committee member's participant ID
    pub participant_id: u32,
    /// Member's agent public key
    pub signer: AgentPubKey,
    /// Partial signature share
    pub share: Vec<u8>,
    /// Content hash being signed
    pub content_hash: Vec<u8>,
    /// Timestamp of share submission
    pub submitted_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    SigningCommittee(SigningCommittee),
    CommitteeMember(CommitteeMember),
    ThresholdSignature(ThresholdSignature),
    SignatureShare(SignatureShare),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Committee to its members
    CommitteeToMember,
    /// Committee to signatures it has produced
    CommitteeToSignature,
    /// Signature to its shares
    SignatureToShare,
    /// Agent to committees they belong to
    AgentToCommittee,
    /// Epoch anchor for committee versioning
    EpochToCommittee,
}

/// Validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SigningCommittee(committee) => {
                    validate_create_committee(action, committee)
                }
                EntryTypes::CommitteeMember(member) => validate_create_member(action, member),
                EntryTypes::ThresholdSignature(sig) => validate_create_signature(action, sig),
                EntryTypes::SignatureShare(share) => validate_create_share(action, share),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SigningCommittee(committee) => {
                    validate_update_committee(action, committee)
                }
                EntryTypes::CommitteeMember(member) => validate_update_member(action, member),
                EntryTypes::ThresholdSignature(_) => Ok(ValidateCallbackResult::Invalid(
                    "Threshold signatures cannot be updated".into(),
                )),
                EntryTypes::SignatureShare(_) => Ok(ValidateCallbackResult::Invalid(
                    "Signature shares cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::CommitteeToMember => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CommitteeToSignature => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SignatureToShare => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToCommittee => Ok(ValidateCallbackResult::Valid),
            LinkTypes::EpochToCommittee => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate committee creation
fn validate_create_committee(
    _action: Create,
    committee: SigningCommittee,
) -> ExternResult<ValidateCallbackResult> {
    // Threshold must be positive
    if committee.threshold == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold must be at least 1".into(),
        ));
    }

    // Threshold must be <= member count
    if committee.threshold > committee.member_count {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold cannot exceed member count".into(),
        ));
    }

    // Must have at least 3 members for meaningful threshold
    if committee.member_count < 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Committee must have at least 3 members".into(),
        ));
    }

    // New committees start in Registration phase
    if committee.phase != DkgPhase::Registration {
        return Ok(ValidateCallbackResult::Invalid(
            "New committees must start in Registration phase".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for committee update — testable without HDI
pub fn check_committee_update_validity(committee: &SigningCommittee) -> Result<(), String> {
    if committee.phase == DkgPhase::Disbanded && committee.active {
        return Err("Cannot reactivate disbanded committee".into());
    }

    if committee.phase == DkgPhase::Complete {
        let pk_bytes = match committee.public_key {
            Some(ref bytes) => bytes,
            None => {
                return Err("Complete committee must have a public key".into());
            }
        };

        if feldman_dkg::Commitment::from_bytes(pk_bytes).is_err() {
            return Err("Public key is not a valid secp256k1 point".into());
        }

        if (committee.commitments.len() as u32) < committee.threshold {
            return Err(format!(
                "Need at least {} commitment sets, got {}",
                committee.threshold,
                committee.commitments.len()
            ));
        }

        for (i, cs_bytes) in committee.commitments.iter().enumerate() {
            if feldman_dkg::CommitmentSet::from_bytes(cs_bytes).is_err() {
                return Err(format!("Invalid commitment set at index {}", i));
            }
        }
    }

    Ok(())
}

/// Validate committee update
fn validate_update_committee(
    _action: Update,
    committee: SigningCommittee,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_committee_update_validity(&committee) {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for committee member — testable without HDI
pub fn check_member_validity(member: &CommitteeMember) -> Result<(), String> {
    if !member.member_did.starts_with("did:") {
        return Err("Member must have a valid DID".into());
    }
    if member.trust_score < 0.0 {
        return Err("Trust score cannot be negative".into());
    }
    if member.participant_id == 0 {
        return Err("Participant ID must be positive".into());
    }
    if let Some(ref vss_bytes) = member.vss_commitment {
        match feldman_dkg::CommitmentSet::from_bytes(vss_bytes) {
            Ok(cs) => {
                if cs.is_empty() {
                    return Err("VSS commitment set must not be empty".into());
                }
            }
            Err(e) => {
                return Err(format!("Invalid VSS commitment: {}", e));
            }
        }
    }
    Ok(())
}

/// Validate member creation
fn validate_create_member(
    _action: Create,
    member: CommitteeMember,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_member_validity(&member) {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate member update
fn validate_update_member(
    _action: Update,
    member: CommitteeMember,
) -> ExternResult<ValidateCallbackResult> {
    // Cannot change participant ID
    // (Would check against original in production)

    // Trust score must remain non-negative
    if member.trust_score < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score cannot be negative".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for threshold signature — testable without HDI
pub fn check_signature_validity(sig: &ThresholdSignature) -> Result<(), String> {
    if sig.signer_count == 0 {
        return Err("Signature must have at least one signer".into());
    }
    if sig.signer_count as usize != sig.signers.len() {
        return Err("Signer count must match signers list length".into());
    }
    if sig.signed_content_hash.is_empty() {
        return Err("Signed content hash cannot be empty".into());
    }
    if sig.signature.is_empty() {
        return Err("Signature cannot be empty".into());
    }
    if sig.signature.len() < 64 {
        return Err(format!(
            "Signature too short: expected at least 64 bytes (ECDSA r||s), got {}",
            sig.signature.len()
        ));
    }
    Ok(())
}

/// Validate threshold signature creation
fn validate_create_signature(
    _action: Create,
    sig: ThresholdSignature,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_signature_validity(&sig) {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate signature share creation
fn validate_create_share(
    _action: Create,
    share: SignatureShare,
) -> ExternResult<ValidateCallbackResult> {
    // Participant ID must be positive
    if share.participant_id == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Participant ID must be positive".into(),
        ));
    }

    // Share must not be empty
    if share.share.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature share cannot be empty".into(),
        ));
    }

    // Content hash must not be empty
    if share.content_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Content hash cannot be empty".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a valid CommitmentSet with `n` commitments and return its serialized bytes
    fn make_valid_commitment_set_bytes(n: usize) -> Vec<u8> {
        let commitments: Vec<feldman_dkg::Commitment> = (1..=n)
            .map(|i| feldman_dkg::Commitment::new(&feldman_dkg::Scalar::from_u64(i as u64)))
            .collect();
        feldman_dkg::CommitmentSet::new(commitments).to_bytes()
    }

    /// Helper: create a valid secp256k1 public key (33-byte compressed SEC1 point)
    fn make_valid_public_key() -> Vec<u8> {
        feldman_dkg::Commitment::new(&feldman_dkg::Scalar::from_u64(42)).to_bytes()
    }

    /// Helper: create a minimal CommitteeMember for testing
    fn make_test_member() -> CommitteeMember {
        CommitteeMember {
            committee_id: "test-committee".into(),
            participant_id: 1,
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            member_did: "did:key:z123".into(),
            trust_score: 0.8,
            public_share: None,
            vss_commitment: None,
            deal_submitted: false,
            qualified: false,
            registered_at: Timestamp::from_micros(0),
        }
    }

    /// Helper: create a minimal SigningCommittee for testing
    fn make_test_committee_complete(
        public_key: Option<Vec<u8>>,
        commitments: Vec<Vec<u8>>,
    ) -> SigningCommittee {
        SigningCommittee {
            id: "test-committee".into(),
            name: "Test".into(),
            threshold: 2,
            member_count: 3,
            phase: DkgPhase::Complete,
            public_key,
            commitments,
            scope: CommitteeScope::All,
            created_at: Timestamp::from_micros(0),
            active: true,
            epoch: 1,
        }
    }

    // --- VSS Commitment Tests ---

    #[test]
    fn test_valid_vss_commitment_accepted() {
        let mut member = make_test_member();
        member.vss_commitment = Some(make_valid_commitment_set_bytes(3));
        assert!(check_member_validity(&member).is_ok());
    }

    #[test]
    fn test_invalid_vss_commitment_rejected() {
        let mut member = make_test_member();
        member.vss_commitment = Some(vec![0xDE, 0xAD, 0xBE, 0xEF, 0xFF]);
        let result = check_member_validity(&member);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid VSS commitment"));
    }

    #[test]
    fn test_empty_vss_commitment_rejected() {
        let mut member = make_test_member();
        // A CommitmentSet with count=0: 4 zero bytes (BE u32 = 0)
        member.vss_commitment = Some(vec![0, 0, 0, 0]);
        let result = check_member_validity(&member);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must not be empty"));
    }

    // --- Committee Complete Validation Tests ---

    #[test]
    fn test_complete_committee_valid_public_key() {
        let pk = make_valid_public_key();
        let cs = make_valid_commitment_set_bytes(2);
        let committee = make_test_committee_complete(Some(pk), vec![cs.clone(), cs]);
        assert!(check_committee_update_validity(&committee).is_ok());
    }

    #[test]
    fn test_complete_committee_invalid_public_key() {
        // 33 garbage bytes — not a valid curve point
        let committee = make_test_committee_complete(
            Some(vec![0xFF; 33]),
            vec![make_valid_commitment_set_bytes(2)],
        );
        let result = check_committee_update_validity(&committee);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not a valid secp256k1 point"));
    }

    #[test]
    fn test_complete_committee_missing_public_key() {
        let committee = make_test_committee_complete(
            None,
            vec![make_valid_commitment_set_bytes(2)],
        );
        let result = check_committee_update_validity(&committee);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must have a public key"));
    }

    #[test]
    fn test_complete_committee_insufficient_commitments() {
        let pk = make_valid_public_key();
        // threshold=2 but only 1 commitment set
        let committee = make_test_committee_complete(
            Some(pk),
            vec![make_valid_commitment_set_bytes(2)],
        );
        let result = check_committee_update_validity(&committee);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Need at least 2"));
    }

    // --- Signature Validation Tests ---

    #[test]
    fn test_signature_too_short_rejected() {
        let sig = ThresholdSignature {
            id: "sig-1".into(),
            committee_id: "test".into(),
            signed_content_hash: vec![1; 32],
            signed_content_description: "test".into(),
            signature: vec![0u8; 32], // too short
            signer_count: 1,
            signers: vec![1],
            verified: false,
            signed_at: Timestamp::from_micros(0),
        };
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_signature_minimum_length_accepted() {
        let sig = ThresholdSignature {
            id: "sig-1".into(),
            committee_id: "test".into(),
            signed_content_hash: vec![1; 32],
            signed_content_description: "test".into(),
            signature: vec![0u8; 64], // minimum valid length
            signer_count: 1,
            signers: vec![1],
            verified: false,
            signed_at: Timestamp::from_micros(0),
        };
        assert!(check_signature_validity(&sig).is_ok());
    }
}
