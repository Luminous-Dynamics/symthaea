//! Debris Bounties Integrity Zome
//!
//! Implements the "Kessler Cleanup Market" - a decentralized bounty system
//! for debris removal. Organizations can post bounties on specific debris
//! objects, and removal services can claim them upon verified removal.
//!
//! # How It Works
//!
//! 1. **Bounty Creation**: Operator posts bounty on debris threatening their assets
//! 2. **Bounty Aggregation**: Multiple parties can contribute to same bounty
//! 3. **Removal Claim**: Service provider claims intent to remove
//! 4. **Verification**: Network verifies debris is no longer tracked
//! 5. **Payout**: Bounty released to remover (via external settlement)

use hdi::prelude::*;
use mycelix_space_shared::SpaceTimestamp;

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    /// Bounty on a debris object
    DebrisBounty(DebrisBounty),

    /// Contribution to a bounty
    BountyContribution(BountyContribution),

    /// Claim to remove debris
    RemovalClaim(RemovalClaim),

    /// Verification of removal
    RemovalVerification(RemovalVerification),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Bounties for an object
    ObjectBounties,
    /// Contributions to a bounty
    BountyContributions,
    /// Claims on a bounty
    BountyClaims,
    /// All active bounties
    ActiveBounties,
    /// Bounties by contributor
    ContributorBounties,
}

/// A bounty for debris removal
#[hdk_entry_helper]
#[derive(Clone)]
pub struct DebrisBounty {
    /// Unique bounty ID
    pub bounty_id: String,

    /// Target debris NORAD ID
    pub debris_norad_id: u32,

    /// Why this debris is a problem
    pub justification: String,

    /// Bounty amount (in smallest currency unit)
    pub amount: u64,

    /// Currency/token identifier
    pub currency: String,

    /// Expiration time (bounty void after this)
    pub expires_at: Option<SpaceTimestamp>,

    /// Bounty status
    pub status: BountyStatus,

    /// Creator
    pub creator: AgentPubKey,

    /// Created at
    pub created_at: SpaceTimestamp,

    /// Requirements for claiming
    pub requirements: RemovalRequirements,
}

/// Bounty status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BountyStatus {
    /// Open for claims
    Open,
    /// Claimed by a remover
    Claimed,
    /// Removal in progress
    InProgress,
    /// Pending verification
    PendingVerification,
    /// Successfully completed
    Completed,
    /// Expired without completion
    Expired,
    /// Cancelled by creator
    Cancelled,
}

/// Requirements for claiming the bounty
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RemovalRequirements {
    /// Minimum trust level to claim
    pub min_trust_level: u8,

    /// Required removal method
    pub allowed_methods: Vec<RemovalMethod>,

    /// Deadline for completion after claiming
    pub completion_deadline_days: u32,

    /// Number of independent verifications needed
    pub verification_threshold: u32,
}

/// Methods for debris removal
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum RemovalMethod {
    /// Controlled deorbit
    Deorbit,
    /// Capture and removal
    Capture,
    /// Deflection to graveyard orbit
    GraveyardOrbit,
    /// Any method
    Any,
}

/// Contribution to a bounty (allows multiple funders)
#[hdk_entry_helper]
#[derive(Clone)]
pub struct BountyContribution {
    /// Bounty being contributed to
    pub bounty_id: String,

    /// Amount contributed
    pub amount: u64,

    /// Currency
    pub currency: String,

    /// Contributor
    pub contributor: AgentPubKey,

    /// Message/reason
    pub message: Option<String>,

    /// Contributed at
    pub contributed_at: SpaceTimestamp,
}

/// Claim to remove debris
#[hdk_entry_helper]
#[derive(Clone)]
pub struct RemovalClaim {
    /// Bounty being claimed
    pub bounty_id: String,

    /// Organization claiming
    pub claimer: AgentPubKey,

    /// Organization name
    pub organization: String,

    /// Proposed removal method
    pub method: RemovalMethod,

    /// Estimated completion date
    pub estimated_completion: SpaceTimestamp,

    /// Mission plan summary
    pub mission_plan: String,

    /// Claim status
    pub status: ClaimStatus,

    /// Claimed at
    pub claimed_at: SpaceTimestamp,
}

/// Claim status
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ClaimStatus {
    /// Pending approval
    Pending,
    /// Approved, work can begin
    Approved,
    /// Work in progress
    InProgress,
    /// Submitted for verification
    Submitted,
    /// Completed and verified
    Completed,
    /// Failed or abandoned
    Failed,
    /// Rejected
    Rejected,
}

/// Verification of successful removal
#[hdk_entry_helper]
#[derive(Clone)]
pub struct RemovalVerification {
    /// Claim being verified
    pub claim_id: ActionHash,

    /// Verifier
    pub verifier: AgentPubKey,

    /// Verification result
    pub verified: bool,

    /// Evidence/reasoning
    pub evidence: VerificationEvidence,

    /// Verified at
    pub verified_at: SpaceTimestamp,
}

/// Evidence for verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationEvidence {
    /// Last observed time (should be before deorbit)
    pub last_observed: Option<SpaceTimestamp>,

    /// Predicted reentry time
    pub predicted_reentry: Option<SpaceTimestamp>,

    /// Number of sensors that lost track
    pub sensors_lost_track: u32,

    /// Hash of supporting data
    pub data_hash: Option<[u8; 32]>,

    /// Textual notes
    pub notes: String,
}

#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::DebrisBounty(bounty) => validate_bounty(&bounty),
                    EntryTypes::BountyContribution(contrib) => validate_contribution(&contrib),
                    EntryTypes::RemovalClaim(claim) => validate_claim(&claim),
                    EntryTypes::RemovalVerification(verif) => validate_verification(&verif),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_bounty(bounty: &DebrisBounty) -> ExternResult<ValidateCallbackResult> {
    // NORAD ID must be valid
    if bounty.debris_norad_id == 0 || bounty.debris_norad_id > 999999 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid debris NORAD ID".to_string()
        ));
    }

    // Amount must be positive
    if bounty.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Bounty amount must be positive".to_string()
        ));
    }

    // Justification must not be empty
    if bounty.justification.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Justification cannot be empty".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_contribution(contrib: &BountyContribution) -> ExternResult<ValidateCallbackResult> {
    if contrib.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contribution amount must be positive".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_claim(claim: &RemovalClaim) -> ExternResult<ValidateCallbackResult> {
    if claim.organization.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Organization name cannot be empty".to_string()
        ));
    }

    if claim.mission_plan.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Mission plan cannot be empty".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_verification(_verif: &RemovalVerification) -> ExternResult<ValidateCallbackResult> {
    // Basic validation - more complex verification logic in coordinator
    Ok(ValidateCallbackResult::Valid)
}
