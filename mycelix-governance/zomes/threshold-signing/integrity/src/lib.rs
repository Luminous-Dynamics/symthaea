// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

/// Threshold signature algorithm selection
///
/// Supports classical ECDSA, post-quantum ML-DSA-65, or hybrid (both).
#[derive(Serialize, Deserialize, Debug, Default, Clone, PartialEq)]
pub enum ThresholdSignatureAlgorithm {
    /// Classical ECDSA (secp256k1)
    #[default]
    Ecdsa,
    /// Post-quantum ML-DSA-65 (FIPS 204)
    MlDsa65,
    /// Hybrid: both ECDSA and ML-DSA-65 (defense-in-depth)
    HybridEcdsaMlDsa65,
}

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
    /// Minimum Phi (consciousness) score required to join committee (0.0-1.0)
    #[serde(default)]
    pub min_phi: Option<f64>,
    /// Signature algorithm used by this committee
    #[serde(default)]
    pub signature_algorithm: ThresholdSignatureAlgorithm,
    /// If true, PQ signature is mandatory (no graceful degradation to ECDSA-only)
    #[serde(default)]
    pub pq_required: bool,
}

/// DKG ceremony phases
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DkgPhase {
    /// Waiting for members to register
    Registration,
    /// Collecting hash commitments (PQ commit-reveal protocol)
    CommitmentCollection,
    /// Collecting deals from members
    Dealing,
    /// Verifying and complaints
    Verification,
    /// DKG complete, ready to sign
    Complete,
    /// Committee disbanded
    Disbanded,
}

/// Severity level for DKG protocol violations.
///
/// Typed enum replaces freeform String to prevent parsing errors
/// and ensure consistent severity handling across locales.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ViolationSeverity {
    /// Minor protocol deviation, no security impact
    Minor,
    /// Moderate violation, potential for degraded security
    Moderate,
    /// Severe violation, active security threat
    Severe,
    /// Critical violation, immediate committee compromise risk
    Critical,
}

impl std::fmt::Display for ViolationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Minor => write!(f, "minor"),
            Self::Moderate => write!(f, "moderate"),
            Self::Severe => write!(f, "severe"),
            Self::Critical => write!(f, "critical"),
        }
    }
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
    /// ML-KEM-768 encapsulation (public) key for encrypted DKG deals
    #[serde(default)]
    pub ml_kem_encapsulation_key: Option<Vec<u8>>,
    /// ML-KEM-768 encrypted deal shares (for PQ-protected DKG)
    #[serde(default)]
    pub encrypted_shares: Option<Vec<u8>>,
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
    /// Combined threshold signature (ECDSA)
    pub signature: Vec<u8>,
    /// Post-quantum signature (ML-DSA-65), if using hybrid or PQ-only
    #[serde(default)]
    pub pq_signature: Option<Vec<u8>>,
    /// Signature algorithm used
    #[serde(default)]
    pub signature_algorithm: ThresholdSignatureAlgorithm,
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

/// A violation report from a DKG ceremony
///
/// Records protocol violations for reputation slashing and accountability.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DkgViolationReport {
    /// Committee where violation occurred
    pub committee_id: String,
    /// Participant who committed the violation
    pub participant_id: u32,
    /// Type of violation
    pub violation_type: String,
    /// Severity level
    pub severity: ViolationSeverity,
    /// Penalty score (0.0-1.0)
    pub penalty_score: f64,
    /// Epoch when violation occurred
    pub epoch: u32,
    /// Agent reporting the violation
    pub reporter: AgentPubKey,
    /// Timestamp of report
    pub reported_at: Timestamp,
}

/// Hash commitment from a DKG dealer (commit-reveal protocol)
///
/// In the Hybrid commitment scheme, each dealer first commits SHA-256 hashes
/// of their shares before revealing the actual shares. This prevents a quantum
/// attacker from using Shor's algorithm against Feldman commitments.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DkgHashCommitment {
    /// Committee this commitment belongs to
    pub committee_id: String,
    /// Participant ID of the dealer
    pub dealer_participant_id: u32,
    /// Serialized HashCommitmentSet from feldman-dkg
    pub commitment_set_bytes: Vec<u8>,
    /// Submission timestamp
    pub submitted_at: Timestamp,
}

/// Hash reveal from a DKG dealer (commit-reveal protocol)
///
/// After all commitments are collected, dealers reveal the salts used
/// to create the hash commitments. Recipients verify H(share || dealer_id || salt)
/// matches the committed hash.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DkgHashReveal {
    /// Committee this reveal belongs to
    pub committee_id: String,
    /// Participant ID of the dealer
    pub dealer_participant_id: u32,
    /// Serialized HashReveal from feldman-dkg
    pub reveal_bytes: Vec<u8>,
    /// Submission timestamp
    pub submitted_at: Timestamp,
}

/// Post-quantum attestor registration for hybrid signing
///
/// Stores the ML-DSA-65 public key for the designated PQ attestor of a committee.
/// The attestor is selected deterministically per epoch.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PqAttestor {
    /// Committee this attestor serves
    pub committee_id: String,
    /// Epoch for which this attestor is designated
    pub epoch: u32,
    /// Participant ID of the attestor
    pub participant_id: u32,
    /// Agent public key of the attestor
    pub agent: AgentPubKey,
    /// ML-DSA-65 public key (2,592 bytes)
    pub ml_dsa_public_key: Vec<u8>,
    /// Registration timestamp
    pub registered_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    SigningCommittee(SigningCommittee),
    CommitteeMember(CommitteeMember),
    ThresholdSignature(ThresholdSignature),
    SignatureShare(SignatureShare),
    DkgViolationReport(DkgViolationReport),
    PqAttestor(PqAttestor),
    DkgHashCommitment(DkgHashCommitment),
    DkgHashReveal(DkgHashReveal),
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
    /// Proposal ID to its threshold signature
    ProposalToSignature,
    /// Committee to violation reports
    CommitteeToViolation,
    /// Participant anchor to violation reports (global, cross-committee)
    ParticipantToViolation,
    /// Committee to its PQ attestor
    CommitteeToAttestor,
    /// Committee to hash commitments (commit-reveal protocol)
    CommitteeToHashCommitment,
    /// Committee to hash reveals (commit-reveal protocol)
    CommitteeToHashReveal,
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
                EntryTypes::DkgViolationReport(report) => {
                    validate_create_violation_report(action, report)
                }
                EntryTypes::PqAttestor(attestor) => validate_create_pq_attestor(action, attestor),
                EntryTypes::DkgHashCommitment(commitment) => {
                    validate_create_hash_commitment(action, commitment)
                }
                EntryTypes::DkgHashReveal(reveal) => validate_create_hash_reveal(action, reveal),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SigningCommittee(committee) => {
                    validate_update_committee(action, committee)
                }
                EntryTypes::CommitteeMember(member) => {
                    validate_update_member(action, member, original_action_hash)
                }
                EntryTypes::ThresholdSignature(_) => Ok(ValidateCallbackResult::Invalid(
                    "Threshold signatures cannot be updated".into(),
                )),
                EntryTypes::SignatureShare(_) => Ok(ValidateCallbackResult::Invalid(
                    "Signature shares cannot be updated".into(),
                )),
                EntryTypes::DkgViolationReport(_) => Ok(ValidateCallbackResult::Invalid(
                    "Violation reports cannot be updated".into(),
                )),
                EntryTypes::PqAttestor(_) => Ok(ValidateCallbackResult::Invalid(
                    "PQ attestor registrations cannot be updated".into(),
                )),
                EntryTypes::DkgHashCommitment(_) => Ok(ValidateCallbackResult::Invalid(
                    "Hash commitments cannot be updated".into(),
                )),
                EntryTypes::DkgHashReveal(_) => Ok(ValidateCallbackResult::Invalid(
                    "Hash reveals cannot be updated".into(),
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
            LinkTypes::ProposalToSignature => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CommitteeToViolation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ParticipantToViolation => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CommitteeToAttestor => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CommitteeToHashCommitment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CommitteeToHashReveal => Ok(ValidateCallbackResult::Valid),
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

    // Validate min_phi if set
    if let Some(min_phi) = committee.min_phi {
        if !(0.0..=1.0).contains(&min_phi) {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "min_phi must be between 0.0 and 1.0, got {}",
                min_phi
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for committee update -- testable without HDI
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

/// Pure validation for committee member -- testable without HDI
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
    // ML-KEM-768 encapsulation key must be exactly 1,184 bytes if present
    if let Some(ref ek_bytes) = member.ml_kem_encapsulation_key {
        if ek_bytes.len() != 1184 {
            return Err(format!(
                "ML-KEM-768 encapsulation key must be exactly 1184 bytes, got {}",
                ek_bytes.len()
            ));
        }
    }
    // Encrypted shares must be a valid EncryptedDeal if present
    if let Some(ref enc_bytes) = member.encrypted_shares {
        if enc_bytes.is_empty() {
            return Err("Encrypted shares cannot be empty".into());
        }
        feldman_dkg::EncryptedDeal::from_bytes(enc_bytes)
            .map_err(|e| format!("Invalid encrypted shares: {}", e))?;
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
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Fetch original member to enforce immutable fields
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_member: CommitteeMember = original_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize original member: {}",
                e
            )))
        })?
        .ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "Original member record has no entry".into()
            ))
        })?;

    // participant_id is immutable -- cannot change after creation
    if member.participant_id != original_member.participant_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change participant_id after creation".into(),
        ));
    }

    // committee_id is immutable -- cannot move member between committees
    if member.committee_id != original_member.committee_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change committee_id after creation".into(),
        ));
    }

    // agent is immutable -- cannot reassign member identity
    if member.agent != original_member.agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change agent after creation".into(),
        ));
    }

    // Trust score must remain non-negative
    if member.trust_score < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust score cannot be negative".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for threshold signature -- testable without HDI
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

    // Algorithm-specific signature validation
    match sig.signature_algorithm {
        ThresholdSignatureAlgorithm::Ecdsa => {
            if sig.signature.is_empty() {
                return Err("Signature cannot be empty".into());
            }
            if sig.signature.len() < 64 {
                return Err(format!(
                    "ECDSA signature too short: expected at least 64 bytes (r||s), got {}",
                    sig.signature.len()
                ));
            }
        }
        ThresholdSignatureAlgorithm::MlDsa65 => {
            let pq_sig = sig
                .pq_signature
                .as_ref()
                .ok_or("ML-DSA-65 algorithm requires pq_signature field".to_string())?;
            if pq_sig.len() != 3309 {
                return Err(format!(
                    "ML-DSA-65 signature must be exactly 3309 bytes, got {}",
                    pq_sig.len()
                ));
            }
        }
        ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65 => {
            // ECDSA part
            if sig.signature.is_empty() {
                return Err("Hybrid mode requires ECDSA signature".into());
            }
            if sig.signature.len() < 64 {
                return Err(format!(
                    "Hybrid ECDSA signature too short: expected at least 64 bytes, got {}",
                    sig.signature.len()
                ));
            }
            // ML-DSA-65 part
            let pq_sig = sig
                .pq_signature
                .as_ref()
                .ok_or("Hybrid mode requires pq_signature field for ML-DSA-65".to_string())?;
            if pq_sig.len() != 3309 {
                return Err(format!(
                    "Hybrid ML-DSA-65 signature must be exactly 3309 bytes, got {}",
                    pq_sig.len()
                ));
            }
        }
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

/// Pure validation for violation report -- testable without HDI
pub fn check_violation_report_validity(report: &DkgViolationReport) -> Result<(), String> {
    // Severity is now a typed enum — no string parsing needed.
    // Validate penalty aligns with severity level.
    let min_penalty = match report.severity {
        ViolationSeverity::Minor => 0.0,
        ViolationSeverity::Moderate => 0.1,
        ViolationSeverity::Severe => 0.3,
        ViolationSeverity::Critical => 0.5,
    };
    if report.penalty_score < min_penalty {
        return Err(format!(
            "Penalty score {:.2} too low for {:?} severity (minimum: {:.2})",
            report.penalty_score, report.severity, min_penalty
        ));
    }

    // Penalty score must be in [0.0, 1.0]
    if !report.penalty_score.is_finite() || report.penalty_score < 0.0 || report.penalty_score > 1.0
    {
        return Err(format!(
            "Penalty score must be between 0.0 and 1.0, got {}",
            report.penalty_score
        ));
    }

    // Participant ID must be positive
    if report.participant_id == 0 {
        return Err("Participant ID must be positive".into());
    }

    // Committee ID must not be empty
    if report.committee_id.is_empty() {
        return Err("Committee ID cannot be empty".into());
    }

    // Violation type must not be empty
    if report.violation_type.is_empty() {
        return Err("Violation type cannot be empty".into());
    }

    Ok(())
}

/// Validate violation report creation
fn validate_create_violation_report(
    _action: Create,
    report: DkgViolationReport,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_violation_report_validity(&report) {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for PQ attestor -- testable without HDI
pub fn check_pq_attestor_validity(attestor: &PqAttestor) -> Result<(), String> {
    if attestor.committee_id.is_empty() {
        return Err("Committee ID cannot be empty".into());
    }
    if attestor.participant_id == 0 {
        return Err("Participant ID must be positive".into());
    }
    // ML-DSA-65 public key must be exactly 1,952 bytes
    if attestor.ml_dsa_public_key.len() != 1952 {
        return Err(format!(
            "ML-DSA-65 public key must be exactly 1952 bytes, got {}",
            attestor.ml_dsa_public_key.len()
        ));
    }
    Ok(())
}

/// Pure validation for DKG hash commitment -- testable without HDI
pub fn check_hash_commitment_validity(commitment: &DkgHashCommitment) -> Result<(), String> {
    if commitment.committee_id.is_empty() {
        return Err("Committee ID cannot be empty".into());
    }
    if commitment.dealer_participant_id == 0 {
        return Err("Dealer participant ID must be positive".into());
    }
    if commitment.commitment_set_bytes.is_empty() {
        return Err("Hash commitment set bytes cannot be empty".into());
    }
    feldman_dkg::HashCommitmentSet::from_bytes(&commitment.commitment_set_bytes)
        .map_err(|e| format!("Invalid hash commitment set: {}", e))?;
    Ok(())
}

/// Validate DKG hash commitment creation
fn validate_create_hash_commitment(
    _action: Create,
    commitment: DkgHashCommitment,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_hash_commitment_validity(&commitment) {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Pure validation for DKG hash reveal -- testable without HDI
pub fn check_hash_reveal_validity(reveal: &DkgHashReveal) -> Result<(), String> {
    if reveal.committee_id.is_empty() {
        return Err("Committee ID cannot be empty".into());
    }
    if reveal.dealer_participant_id == 0 {
        return Err("Dealer participant ID must be positive".into());
    }
    if reveal.reveal_bytes.is_empty() {
        return Err("Hash reveal bytes cannot be empty".into());
    }
    // Validate deserialization
    serde_json::from_slice::<feldman_dkg::HashReveal>(&reveal.reveal_bytes)
        .map_err(|e| format!("Invalid hash reveal: {}", e))?;
    Ok(())
}

/// Validate DKG hash reveal creation
fn validate_create_hash_reveal(
    _action: Create,
    reveal: DkgHashReveal,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_hash_reveal_validity(&reveal) {
        return Ok(ValidateCallbackResult::Invalid(reason));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate PQ attestor creation
fn validate_create_pq_attestor(
    _action: Create,
    attestor: PqAttestor,
) -> ExternResult<ValidateCallbackResult> {
    if let Err(reason) = check_pq_attestor_validity(&attestor) {
        return Ok(ValidateCallbackResult::Invalid(reason));
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
            ml_kem_encapsulation_key: None,
            encrypted_shares: None,
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
            min_phi: None,
            signature_algorithm: ThresholdSignatureAlgorithm::default(),
            pq_required: false,
        }
    }

    /// Helper: create a minimal ThresholdSignature for testing
    fn make_test_signature(algorithm: ThresholdSignatureAlgorithm) -> ThresholdSignature {
        ThresholdSignature {
            id: "sig-1".into(),
            committee_id: "test".into(),
            signed_content_hash: vec![1; 32],
            signed_content_description: "test".into(),
            signature: vec![0u8; 64],
            pq_signature: None,
            signature_algorithm: algorithm,
            signer_count: 1,
            signers: vec![1],
            verified: false,
            signed_at: Timestamp::from_micros(0),
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
        // 33 garbage bytes -- not a valid curve point
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
        let committee =
            make_test_committee_complete(None, vec![make_valid_commitment_set_bytes(2)]);
        let result = check_committee_update_validity(&committee);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must have a public key"));
    }

    #[test]
    fn test_complete_committee_insufficient_commitments() {
        let pk = make_valid_public_key();
        // threshold=2 but only 1 commitment set
        let committee =
            make_test_committee_complete(Some(pk), vec![make_valid_commitment_set_bytes(2)]);
        let result = check_committee_update_validity(&committee);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Need at least 2"));
    }

    // --- Signature Validation Tests ---

    #[test]
    fn test_ecdsa_signature_too_short_rejected() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::Ecdsa);
        sig.signature = vec![0u8; 32]; // too short
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_ecdsa_signature_minimum_length_accepted() {
        let sig = make_test_signature(ThresholdSignatureAlgorithm::Ecdsa);
        assert!(check_signature_validity(&sig).is_ok());
    }

    #[test]
    fn test_ml_dsa65_signature_accepted() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::MlDsa65);
        sig.pq_signature = Some(vec![0u8; 3309]); // ML-DSA-65 minimum
        assert!(check_signature_validity(&sig).is_ok());
    }

    #[test]
    fn test_ml_dsa65_signature_too_short_rejected() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::MlDsa65);
        sig.pq_signature = Some(vec![0u8; 100]); // too short
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("ML-DSA-65 signature must be exactly 3309"));
    }

    #[test]
    fn test_ml_dsa65_missing_pq_signature_rejected() {
        let sig = make_test_signature(ThresholdSignatureAlgorithm::MlDsa65);
        // pq_signature is None
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires pq_signature"));
    }

    #[test]
    fn test_hybrid_signature_accepted() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65);
        sig.pq_signature = Some(vec![0u8; 3309]);
        assert!(check_signature_validity(&sig).is_ok());
    }

    #[test]
    fn test_hybrid_missing_ecdsa_rejected() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65);
        sig.signature = vec![]; // empty ECDSA
        sig.pq_signature = Some(vec![0u8; 3309]);
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires ECDSA"));
    }

    #[test]
    fn test_hybrid_missing_pq_rejected() {
        let sig = make_test_signature(ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65);
        // pq_signature is None
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires pq_signature"));
    }

    // --- Violation Report Tests ---

    #[test]
    fn test_valid_violation_report() {
        let report = DkgViolationReport {
            committee_id: "committee-1".into(),
            participant_id: 3,
            violation_type: "InvalidShare".into(),
            severity: ViolationSeverity::Moderate,
            penalty_score: 0.15,
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        assert!(check_violation_report_validity(&report).is_ok());
    }

    #[test]
    fn test_violation_report_penalty_below_severity_minimum() {
        // Severe requires penalty >= 0.3
        let report = DkgViolationReport {
            committee_id: "committee-1".into(),
            participant_id: 3,
            violation_type: "InvalidShare".into(),
            severity: ViolationSeverity::Severe,
            penalty_score: 0.15, // too low for Severe
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        let result = check_violation_report_validity(&report);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too low"));
    }

    #[test]
    fn test_violation_report_penalty_out_of_range() {
        let report = DkgViolationReport {
            committee_id: "committee-1".into(),
            participant_id: 3,
            violation_type: "InvalidShare".into(),
            severity: ViolationSeverity::Severe,
            penalty_score: 1.5, // out of range
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        let result = check_violation_report_validity(&report);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("between 0.0 and 1.0"));
    }

    #[test]
    fn test_violation_report_nan_penalty_rejected() {
        let report = DkgViolationReport {
            committee_id: "committee-1".into(),
            participant_id: 3,
            violation_type: "InvalidShare".into(),
            severity: ViolationSeverity::Minor,
            penalty_score: f64::NAN,
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        assert!(check_violation_report_validity(&report).is_err());
    }

    #[test]
    fn test_violation_report_zero_participant_rejected() {
        let report = DkgViolationReport {
            committee_id: "committee-1".into(),
            participant_id: 0, // invalid
            violation_type: "InvalidShare".into(),
            severity: ViolationSeverity::Minor,
            penalty_score: 0.05,
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        assert!(check_violation_report_validity(&report).is_err());
    }

    #[test]
    fn test_all_severity_levels_accepted() {
        let levels = [
            (ViolationSeverity::Minor, 0.05),
            (ViolationSeverity::Moderate, 0.15),
            (ViolationSeverity::Severe, 0.35),
            (ViolationSeverity::Critical, 0.55),
        ];
        for (severity, penalty) in &levels {
            let report = DkgViolationReport {
                committee_id: "committee-1".into(),
                participant_id: 1,
                violation_type: "DealTimeout".into(),
                severity: severity.clone(),
                penalty_score: *penalty,
                epoch: 1,
                reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
                reported_at: Timestamp::from_micros(0),
            };
            assert!(
                check_violation_report_validity(&report).is_ok(),
                "Severity {:?} with penalty {} should be accepted",
                severity,
                penalty
            );
        }
    }

    // --- ViolationSeverity penalty alignment ---

    #[test]
    fn test_minor_penalty_at_zero_accepted() {
        let report = DkgViolationReport {
            committee_id: "c-1".into(),
            participant_id: 1,
            violation_type: "LateResponse".into(),
            severity: ViolationSeverity::Minor,
            penalty_score: 0.0, // Minor allows 0.0
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        assert!(check_violation_report_validity(&report).is_ok());
    }

    #[test]
    fn test_moderate_penalty_below_minimum_rejected() {
        let report = DkgViolationReport {
            committee_id: "c-1".into(),
            participant_id: 1,
            violation_type: "InvalidShare".into(),
            severity: ViolationSeverity::Moderate,
            penalty_score: 0.05, // Below 0.1 minimum for Moderate
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        let result = check_violation_report_validity(&report);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too low"));
    }

    #[test]
    fn test_critical_penalty_at_boundary_accepted() {
        let report = DkgViolationReport {
            committee_id: "c-1".into(),
            participant_id: 1,
            violation_type: "Equivocation".into(),
            severity: ViolationSeverity::Critical,
            penalty_score: 0.5, // Exactly at minimum for Critical
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        assert!(check_violation_report_validity(&report).is_ok());
    }

    #[test]
    fn test_critical_penalty_below_boundary_rejected() {
        let report = DkgViolationReport {
            committee_id: "c-1".into(),
            participant_id: 1,
            violation_type: "Equivocation".into(),
            severity: ViolationSeverity::Critical,
            penalty_score: 0.49, // Just below 0.5 minimum for Critical
            epoch: 1,
            reporter: AgentPubKey::from_raw_36(vec![0u8; 36]),
            reported_at: Timestamp::from_micros(0),
        };
        let result = check_violation_report_validity(&report);
        assert!(result.is_err());
    }

    #[test]
    fn test_violation_severity_display() {
        assert_eq!(ViolationSeverity::Minor.to_string(), "minor");
        assert_eq!(ViolationSeverity::Moderate.to_string(), "moderate");
        assert_eq!(ViolationSeverity::Severe.to_string(), "severe");
        assert_eq!(ViolationSeverity::Critical.to_string(), "critical");
    }

    #[test]
    fn test_violation_severity_serde_roundtrip() {
        let original = ViolationSeverity::Critical;
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ViolationSeverity = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    // =========================================================================
    // END-TO-END DKG INTEGRATION TEST
    // =========================================================================
    // Runs a real 3-of-5 feldman-dkg ceremony and validates all outputs
    // through the integrity validators -- proving the full crypto pipeline.

    #[test]
    fn test_e2e_dkg_ceremony_through_validators() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let threshold = 3usize;
        let n_members = 5usize;

        // Step 1: Run a real DKG ceremony
        let config = feldman_dkg::DkgConfig::new(threshold, n_members).expect("valid DKG config");
        let mut ceremony = feldman_dkg::DkgCeremony::new(config, 1000);

        let mut rng = StdRng::seed_from_u64(42);

        // Register all participants (auto-transitions to Dealing when all registered)
        for i in 1..=(n_members as u32) {
            ceremony
                .add_participant(feldman_dkg::ParticipantId(i), 1000)
                .expect("add participant");
        }

        // Each participant generates a deal
        let mut participants: Vec<feldman_dkg::Participant> = (1..=(n_members as u32))
            .map(|i| {
                feldman_dkg::Participant::new(feldman_dkg::ParticipantId(i), threshold, n_members)
                    .unwrap()
            })
            .collect();

        let deals: Vec<feldman_dkg::dealer::Deal> = participants
            .iter_mut()
            .map(|p| p.generate_deal(&mut rng).unwrap())
            .collect();

        // Submit all deals to the ceremony
        for (i, deal) in deals.iter().enumerate() {
            ceremony
                .submit_deal(
                    feldman_dkg::ParticipantId((i + 1) as u32),
                    deal.clone(),
                    1001,
                )
                .expect("submit deal");
        }

        // Finalize ceremony
        let result = ceremony.finalize().expect("ceremony finalize");
        let combined_pk = result.public_key.to_bytes();

        // Step 2: Collect all commitment sets from participants
        let commitment_sets: Vec<Vec<u8>> = deals
            .iter()
            .map(|deal| deal.commitments.to_bytes())
            .collect();

        // Step 3: Validate each member's VSS commitment through check_member_validity
        for (i, cs_bytes) in commitment_sets.iter().enumerate() {
            let mut member = make_test_member();
            member.participant_id = (i + 1) as u32;
            member.vss_commitment = Some(cs_bytes.clone());
            assert!(
                check_member_validity(&member).is_ok(),
                "Member {}'s VSS commitment should be valid",
                i + 1
            );
        }

        // Step 4: Validate completed committee through check_committee_update_validity
        let committee = SigningCommittee {
            id: "e2e-test".into(),
            name: "E2E Test".into(),
            threshold: threshold as u32,
            member_count: n_members as u32,
            phase: DkgPhase::Complete,
            public_key: Some(combined_pk.clone()),
            commitments: commitment_sets.clone(),
            scope: CommitteeScope::All,
            created_at: Timestamp::from_micros(0),
            active: true,
            epoch: 1,
            min_phi: Some(0.4),
            signature_algorithm: ThresholdSignatureAlgorithm::default(),
            pq_required: false,
        };
        assert!(
            check_committee_update_validity(&committee).is_ok(),
            "Completed committee with real DKG data should pass validation"
        );

        // Step 5: Verify public key is a valid secp256k1 point
        assert_eq!(
            combined_pk.len(),
            33,
            "Combined PK should be 33-byte compressed SEC1"
        );
        assert!(
            feldman_dkg::Commitment::from_bytes(&combined_pk).is_ok(),
            "Combined PK should be a valid curve point"
        );

        // Step 6: Verify committee with wrong threshold fails
        let bad_committee = SigningCommittee {
            threshold: 6, // more than commitments available
            ..committee.clone()
        };
        assert!(
            check_committee_update_validity(&bad_committee).is_err(),
            "Committee with threshold > commitments should fail"
        );

        // Step 7: Verify committee with corrupt public key fails
        let bad_pk_committee = SigningCommittee {
            public_key: Some(vec![0xFF; 33]),
            ..committee.clone()
        };
        assert!(
            check_committee_update_validity(&bad_pk_committee).is_err(),
            "Committee with corrupt public key should fail"
        );

        // Step 8: Verify committee with corrupt commitment set fails
        let mut bad_commitments = commitment_sets;
        bad_commitments[2] = vec![0xAA; 40]; // corrupt one commitment set
        let bad_cs_committee = SigningCommittee {
            commitments: bad_commitments,
            ..committee
        };
        assert!(
            check_committee_update_validity(&bad_cs_committee).is_err(),
            "Committee with corrupt commitment set should fail"
        );
    }

    #[test]
    fn test_min_phi_validation() {
        // Valid min_phi
        let committee = SigningCommittee {
            id: "test".into(),
            name: "Test".into(),
            threshold: 2,
            member_count: 3,
            phase: DkgPhase::Complete,
            public_key: Some(make_valid_public_key()),
            commitments: vec![
                make_valid_commitment_set_bytes(2),
                make_valid_commitment_set_bytes(2),
            ],
            scope: CommitteeScope::All,
            created_at: Timestamp::from_micros(0),
            active: true,
            epoch: 1,
            min_phi: Some(0.4),
            signature_algorithm: ThresholdSignatureAlgorithm::default(),
            pq_required: false,
        };
        assert!(check_committee_update_validity(&committee).is_ok());

        // None min_phi is valid (no consciousness gate)
        let no_phi = SigningCommittee {
            min_phi: None,
            ..committee.clone()
        };
        assert!(check_committee_update_validity(&no_phi).is_ok());
    }

    #[test]
    fn test_signature_algorithm_default_is_ecdsa() {
        assert_eq!(
            ThresholdSignatureAlgorithm::default(),
            ThresholdSignatureAlgorithm::Ecdsa,
        );
    }

    // --- PQ Attestor Validation Tests ---

    #[test]
    fn test_valid_pq_attestor() {
        let attestor = PqAttestor {
            committee_id: "committee-1".into(),
            epoch: 1,
            participant_id: 2,
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            ml_dsa_public_key: vec![0u8; 1952], // ML-DSA-65 PK size
            registered_at: Timestamp::from_micros(0),
        };
        assert!(check_pq_attestor_validity(&attestor).is_ok());
    }

    #[test]
    fn test_pq_attestor_short_key_rejected() {
        let attestor = PqAttestor {
            committee_id: "committee-1".into(),
            epoch: 1,
            participant_id: 2,
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            ml_dsa_public_key: vec![0u8; 100], // too short
            registered_at: Timestamp::from_micros(0),
        };
        let result = check_pq_attestor_validity(&attestor);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be exactly 1952"));
    }

    #[test]
    fn test_pq_attestor_oversized_key_rejected() {
        let attestor = PqAttestor {
            committee_id: "committee-1".into(),
            epoch: 1,
            participant_id: 2,
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            ml_dsa_public_key: vec![0u8; 2000], // too long
            registered_at: Timestamp::from_micros(0),
        };
        let result = check_pq_attestor_validity(&attestor);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be exactly 1952"));
    }

    #[test]
    fn test_pq_attestor_zero_participant_rejected() {
        let attestor = PqAttestor {
            committee_id: "committee-1".into(),
            epoch: 1,
            participant_id: 0,
            agent: AgentPubKey::from_raw_36(vec![0u8; 36]),
            ml_dsa_public_key: vec![0u8; 1952],
            registered_at: Timestamp::from_micros(0),
        };
        assert!(check_pq_attestor_validity(&attestor).is_err());
    }

    // --- ML-KEM Encapsulation Key Validation Tests ---

    #[test]
    fn test_ml_kem_encapsulation_key_valid() {
        let mut member = make_test_member();
        member.ml_kem_encapsulation_key = Some(vec![0u8; 1184]);
        assert!(check_member_validity(&member).is_ok());
    }

    #[test]
    fn test_ml_kem_encapsulation_key_wrong_size_rejected() {
        let mut member = make_test_member();
        member.ml_kem_encapsulation_key = Some(vec![0u8; 100]);
        let result = check_member_validity(&member);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("1184 bytes"));
    }

    #[test]
    fn test_ml_kem_encapsulation_key_oversized_rejected() {
        let mut member = make_test_member();
        member.ml_kem_encapsulation_key = Some(vec![0u8; 2000]);
        let result = check_member_validity(&member);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("1184 bytes"));
    }

    #[test]
    fn test_ml_kem_encapsulation_key_none_accepted() {
        let member = make_test_member();
        assert!(member.ml_kem_encapsulation_key.is_none());
        assert!(check_member_validity(&member).is_ok());
    }

    // --- ML-DSA-65 Exact Signature Size Tests ---

    #[test]
    fn test_ml_dsa65_signature_oversized_rejected() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::MlDsa65);
        sig.pq_signature = Some(vec![0u8; 4000]); // too long
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be exactly 3309"));
    }

    #[test]
    fn test_hybrid_ml_dsa65_oversized_rejected() {
        let mut sig = make_test_signature(ThresholdSignatureAlgorithm::HybridEcdsaMlDsa65);
        sig.pq_signature = Some(vec![0u8; 4000]); // too long
        let result = check_signature_validity(&sig);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be exactly 3309"));
    }
}
