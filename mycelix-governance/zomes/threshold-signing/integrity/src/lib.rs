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

/// Validate committee update
fn validate_update_committee(
    _action: Update,
    committee: SigningCommittee,
) -> ExternResult<ValidateCallbackResult> {
    // Cannot change threshold after creation
    // (This would require checking original, simplified here)

    // Cannot reactivate disbanded committee
    if committee.phase == DkgPhase::Disbanded && committee.active {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot reactivate disbanded committee".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate member creation
fn validate_create_member(
    _action: Create,
    member: CommitteeMember,
) -> ExternResult<ValidateCallbackResult> {
    // Member DID must be valid
    if !member.member_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Member must have a valid DID".into(),
        ));
    }

    // Trust score must be non-negative
    if member.trust_score < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score cannot be negative".into(),
        ));
    }

    // Participant ID must be positive
    if member.participant_id == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Participant ID must be positive".into(),
        ));
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

/// Validate threshold signature creation
fn validate_create_signature(
    _action: Create,
    sig: ThresholdSignature,
) -> ExternResult<ValidateCallbackResult> {
    // Must have at least one signer
    if sig.signer_count == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature must have at least one signer".into(),
        ));
    }

    // Signer count must match signers list
    if sig.signer_count as usize != sig.signers.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Signer count must match signers list length".into(),
        ));
    }

    // Content hash must not be empty
    if sig.signed_content_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Signed content hash cannot be empty".into(),
        ));
    }

    // Signature must not be empty
    if sig.signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Signature cannot be empty".into(),
        ));
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
