//! LUCID Privacy Integrity Zome
//!
//! Selective sharing and access control.
//! Default: Private. Explicit sharing required.

use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Visibility level for a thought
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Visibility {
    /// Only the owner can see
    Private,
    /// Specific agents can see
    SharedWith(Vec<AgentPubKey>),
    /// Members of a group can see
    GroupMembers(String),
    /// Federated to other hApps
    Federated(Vec<String>),
    /// Publicly visible
    Public,
}

impl Default for Visibility {
    fn default() -> Self {
        Visibility::Private
    }
}

/// Sharing policy for a thought
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SharingPolicy {
    /// The thought this policy applies to
    pub thought_id: String,
    /// Visibility level
    pub visibility: Visibility,
    /// When this policy expires (optional)
    pub expires_at: Option<Timestamp>,
    /// Whether this policy can be further delegated
    pub delegatable: bool,
    /// Created timestamp
    pub created_at: Timestamp,
    /// Last updated
    pub updated_at: Timestamp,
}

/// An access grant to a specific agent
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AccessGrant {
    /// Unique identifier
    pub id: String,
    /// The thought being shared
    pub thought_id: String,
    /// The agent receiving access
    pub grantee: AgentPubKey,
    /// Who granted access
    pub grantor: AgentPubKey,
    /// Operations allowed (read, comment, etc.)
    pub permissions: Vec<String>,
    /// When this grant expires
    pub expires_at: Option<Timestamp>,
    /// Whether grantee can share further
    pub can_delegate: bool,
    /// Revoked flag
    pub revoked: bool,
    /// Created timestamp
    pub created_at: Timestamp,
}

/// Access log entry for audit
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AccessLog {
    /// The thought accessed
    pub thought_id: String,
    /// Who accessed it
    pub accessor: AgentPubKey,
    /// Type of access
    pub access_type: String,
    /// Timestamp
    pub timestamp: Timestamp,
}

/// Zero-knowledge proof attestation for anonymous participation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ZkProofAttestation {
    /// Unique identifier for this attestation
    pub proof_id: String,
    /// Type of proof (anonymous_belief, reputation_range, vote_eligibility)
    pub proof_type: String,
    /// IPFS CID where the full proof is stored (proofs too large for DHT)
    pub proof_cid: String,
    /// Hash of public inputs for quick verification
    pub public_inputs_hash: String,
    /// Hash of the subject being attested (belief, vote, etc.)
    pub subject_hash: String,
    /// When this attestation was verified
    pub verified_at: Timestamp,
    /// Agent who verified the proof (trusted verifier service)
    pub verifier: AgentPubKey,
    /// Whether the proof was valid
    pub verified: bool,
    /// Optional expiration
    pub expires_at: Option<Timestamp>,
    /// Additional metadata
    pub metadata: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    SharingPolicy(SharingPolicy),
    AccessGrant(AccessGrant),
    AccessLog(AccessLog),
    ZkProofAttestation(ZkProofAttestation),
}

#[hdk_link_types]
pub enum LinkTypes {
    ThoughtToPolicy,
    ThoughtToGrants,
    AgentToGrants,
    ThoughtToAccessLog,
    /// Links from subject hash to ZK attestations
    SubjectToAttestation,
    /// Links from verifier to their attestations
    VerifierToAttestation,
    /// Links from proof type anchor to attestations
    ProofTypeToAttestation,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::SharingPolicy(p) => {
                    if p.thought_id.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("thought_id required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::AccessGrant(g) => {
                    if g.thought_id.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("thought_id required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                EntryTypes::AccessLog(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ZkProofAttestation(att) => {
                    if att.proof_id.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("proof_id required".into()));
                    }
                    if att.proof_type.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("proof_type required".into()));
                    }
                    if att.proof_cid.is_empty() {
                        return Ok(ValidateCallbackResult::Invalid("proof_cid required".into()));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
