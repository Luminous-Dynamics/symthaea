//! Trust Integrity Zome
//!
//! Implements Multi-Agent Trust Logic (MATL) for Mycelix Music.
//! - Artist verification through web-of-trust
//! - CDN node reputation scoring
//! - Byzantine detection integration from Mycelix-Core

use hdi::prelude::*;

/// Trust claim - one agent vouches for another
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrustClaim {
    /// Agent making the claim
    pub from: AgentPubKey,
    /// Agent being vouched for
    pub to: AgentPubKey,
    /// Type of trust claim
    pub claim_type: TrustClaimType,
    /// Confidence level (0-1000, basis points)
    pub confidence_bps: u32,
    /// Evidence supporting the claim (IPFS hash, URL, etc.)
    pub evidence: Option<String>,
    /// Timestamp
    pub created_at: Timestamp,
    /// Expiry (if applicable)
    pub expires_at: Option<Timestamp>,
    /// Whether this claim is still active
    pub active: bool,
}

/// Types of trust claims
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum TrustClaimType {
    /// Identity verification (real artist)
    IdentityVerification,
    /// Content authenticity (original work)
    ContentAuthenticity,
    /// Quality attestation (good music)
    QualityAttestation,
    /// CDN reliability (node uptime)
    CdnReliability,
    /// Payment reliability (pays on time)
    PaymentReliability,
    /// General endorsement
    GeneralEndorsement,
}

/// Artist verification status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerificationStatus {
    /// Artist being verified
    pub artist: AgentPubKey,
    /// Overall trust score (0-1000)
    pub trust_score: u32,
    /// Verification tier
    pub tier: VerificationTier,
    /// Number of vouches
    pub vouch_count: u32,
    /// Last computed timestamp
    pub computed_at: Timestamp,
}

/// Verification tiers
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum VerificationTier {
    /// No verification
    Unverified,
    /// Community vouched (3+ claims)
    CommunityVerified,
    /// Highly trusted (10+ claims, high scores)
    Trusted,
    /// Platform verified (official verification)
    PlatformVerified,
    /// Founding artist
    FoundingArtist,
}

/// CDN node reputation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CdnNodeReputation {
    /// Node's agent pub key
    pub node: AgentPubKey,
    /// Ethereum address (for staking/rewards)
    pub eth_address: String,
    /// IPFS peer ID
    pub ipfs_peer_id: String,
    /// Geographic region
    pub region: String,
    /// Total bytes served
    pub bytes_served: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average latency (ms)
    pub avg_latency_ms: u32,
    /// Uptime percentage (basis points)
    pub uptime_bps: u32,
    /// PoGQ score from Mycelix-Core
    pub pogq_score: f64,
    /// Last activity
    pub last_active: Timestamp,
    /// Stake amount (in wei)
    pub stake_amount: u64,
    /// Slashing events
    pub slash_count: u32,
}

/// Service quality report (for CDN nodes)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ServiceQualityReport {
    /// Reporter (listener)
    pub reporter: AgentPubKey,
    /// CDN node being reported on
    pub node: AgentPubKey,
    /// Song that was served
    pub song_hash: ActionHash,
    /// Latency experienced (ms)
    pub latency_ms: u32,
    /// Was the request successful?
    pub success: bool,
    /// Error code if failed
    pub error_code: Option<String>,
    /// Timestamp
    pub reported_at: Timestamp,
}

/// Byzantine behavior report
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ByzantineReport {
    /// Reporter
    pub reporter: AgentPubKey,
    /// Accused agent
    pub accused: AgentPubKey,
    /// Type of misbehavior
    pub behavior_type: ByzantineBehavior,
    /// Evidence
    pub evidence: String,
    /// Severity (0-100)
    pub severity: u8,
    /// Timestamp
    pub reported_at: Timestamp,
    /// Resolution status
    pub status: ReportStatus,
}

/// Types of Byzantine behavior
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum ByzantineBehavior {
    /// Serving corrupted content
    ContentCorruption,
    /// Claiming plays that didn't happen
    FakePlayClaims,
    /// Node serving wrong content
    WrongContent,
    /// Replay attacks
    ReplayAttack,
    /// Sybil attack (multiple fake identities)
    SybilAttack,
    /// Other
    Other,
}

/// Report status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum ReportStatus {
    /// Under review
    Pending,
    /// Confirmed misbehavior
    Confirmed,
    /// Dismissed
    Dismissed,
    /// Slashing executed
    Slashed,
}

/// Link types
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> Trust claims they made
    AgentToClaimsMade,
    /// Agent -> Trust claims about them
    AgentToClaimsReceived,
    /// Agent -> Verification status
    AgentToVerification,
    /// CDN node -> Reputation
    NodeToReputation,
    /// Agent -> Quality reports made
    AgentToReports,
    /// Byzantine reports anchor
    ByzantineReports,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    TrustClaim(TrustClaim),
    VerificationStatus(VerificationStatus),
    CdnNodeReputation(CdnNodeReputation),
    ServiceQualityReport(ServiceQualityReport),
    ByzantineReport(ByzantineReport),
}

/// Validation
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::TrustClaim(claim) => validate_trust_claim(claim, action),
                EntryTypes::VerificationStatus(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::CdnNodeReputation(rep) => validate_cdn_reputation(rep, action),
                EntryTypes::ServiceQualityReport(report) => {
                    validate_quality_report(report, action)
                }
                EntryTypes::ByzantineReport(report) => validate_byzantine_report(report, action),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_trust_claim(claim: TrustClaim, action: Create) -> ExternResult<ValidateCallbackResult> {
    // From must match author
    if claim.from != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust claim 'from' must match action author".to_string(),
        ));
    }

    // Cannot vouch for self
    if claim.from == claim.to {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot create trust claim for self".to_string(),
        ));
    }

    // Confidence must be valid
    if claim.confidence_bps > 1000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be 0-1000 basis points".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_cdn_reputation(
    rep: CdnNodeReputation,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    // Node must match author (nodes register themselves)
    if rep.node != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "CDN node must match action author".to_string(),
        ));
    }

    // Validate Ethereum address format
    if !rep.eth_address.starts_with("0x") || rep.eth_address.len() != 42 {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid Ethereum address format".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_quality_report(
    report: ServiceQualityReport,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    // Reporter must match author
    if report.reporter != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Reporter must match action author".to_string(),
        ));
    }

    // Cannot report self
    if report.reporter == report.node {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot report on self".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_byzantine_report(
    report: ByzantineReport,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    // Reporter must match author
    if report.reporter != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Reporter must match action author".to_string(),
        ));
    }

    // Cannot report self
    if report.reporter == report.accused {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot report self".to_string(),
        ));
    }

    // Must have evidence
    if report.evidence.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Byzantine report must include evidence".to_string(),
        ));
    }

    // Severity must be valid
    if report.severity > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Severity must be 0-100".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
