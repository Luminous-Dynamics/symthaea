// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Staking Integrity Zome
//!
//! Defines entry types for tri-asset staking with cryptographic verification:
//! - Tri-asset stakes (MYC + ETH + USDC)
//! - Slashing conditions and evidence
//! - Cryptographic proofs for cross-chain verification
//! - K-Vector-based stake weighting

use hdi::prelude::*;

/// Tri-Asset Stake Position
///
/// Represents a validator's stake across three assets:
/// - MYC: Native governance token ($10K minimum)
/// - ETH: Ethereum for network security (32 ETH minimum)
/// - USDC: Stablecoin for liquidity ($50K minimum)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TriAssetStake {
    /// Unique stake identifier
    pub id: String,
    /// Staker's DID
    pub staker_did: String,
    /// MYC token amount (in smallest unit, e.g., 10000 * 10^18)
    pub myc_amount: u128,
    /// ETH amount (in wei, e.g., 32 * 10^18)
    pub eth_amount: u128,
    /// USDC amount (in 6 decimals, e.g., 50000 * 10^6)
    pub usdc_amount: u128,
    /// Total weighted stake value in USD
    pub total_value_usd: f64,
    /// K-Vector trust score of staker (affects stake weight)
    pub kvector_trust_score: f32,
    /// Stake weight multiplier (1.0-2.0 based on K-Vector)
    pub stake_weight: f32,
    /// Stake creation timestamp
    pub staked_at: Timestamp,
    /// Unbonding period end (None if not unbonding)
    pub unbonding_until: Option<Timestamp>,
    /// Current stake status
    pub status: StakeStatus,
    /// Cross-chain proof of MYC lock
    pub myc_lock_proof: Vec<u8>,
    /// Cross-chain proof of ETH lock
    pub eth_lock_proof: Vec<u8>,
    /// Cross-chain proof of USDC lock
    pub usdc_lock_proof: Vec<u8>,
    /// Accumulated rewards (in MYC)
    pub pending_rewards: u128,
    /// Last reward claim timestamp
    pub last_reward_claim: Timestamp,
}

/// Stake status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum StakeStatus {
    /// Active and earning rewards
    Active,
    /// In unbonding period (21 days)
    Unbonding,
    /// Withdrawn after unbonding
    Withdrawn,
    /// Partially slashed
    Slashed,
    /// Fully slashed (jailed)
    Jailed,
}

/// Slashing Event
///
/// Records a slashing event with cryptographic evidence
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SlashingEvent {
    /// Unique event identifier
    pub id: String,
    /// Stake ID being slashed
    pub stake_id: String,
    /// Staker DID
    pub staker_did: String,
    /// Slashing reason
    pub reason: SlashingReason,
    /// Percentage of stake slashed (0-100)
    pub slash_percentage: u8,
    /// Amount slashed per asset
    pub myc_slashed: u128,
    pub eth_slashed: u128,
    pub usdc_slashed: u128,
    /// Cryptographic evidence hash
    pub evidence_hash: Vec<u8>,
    /// Evidence data (serialized SlashingEvidence)
    pub evidence: Vec<u8>,
    /// Timestamp of slashing
    pub slashed_at: Timestamp,
    /// Whether staker is jailed (cannot restake)
    pub jailed: bool,
    /// Jail release timestamp (if jailed)
    pub jail_release: Option<Timestamp>,
}

/// Reasons for slashing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SlashingReason {
    /// Double signing blocks
    DoubleSigning,
    /// Prolonged downtime
    Downtime,
    /// Byzantine behavior in consensus
    ByzantineConsensus,
    /// Invalid gradient submission (FL)
    InvalidGradient,
    /// Cartel/collusion detected
    CartelActivity,
    /// Invalid cross-chain proof
    InvalidCrossChainProof,
    /// Governance manipulation
    GovernanceManipulation,
}

impl SlashingReason {
    /// Get default slash percentage for this reason
    pub fn default_slash_percentage(&self) -> u8 {
        match self {
            SlashingReason::DoubleSigning => 100,     // Full slash, jail
            SlashingReason::Downtime => 5,            // Minor
            SlashingReason::ByzantineConsensus => 50, // Severe
            SlashingReason::InvalidGradient => 10,    // Moderate
            SlashingReason::CartelActivity => 75,     // Very severe
            SlashingReason::InvalidCrossChainProof => 25,
            SlashingReason::GovernanceManipulation => 50,
        }
    }

    /// Check if this reason results in jailing
    pub fn results_in_jail(&self) -> bool {
        matches!(
            self,
            SlashingReason::DoubleSigning | SlashingReason::CartelActivity
        )
    }
}

/// Cryptographic slashing evidence
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SlashingEvidence {
    /// Type of evidence
    pub evidence_type: EvidenceType,
    /// Block/round number where violation occurred
    pub violation_height: u64,
    /// Conflicting data (e.g., two signed blocks)
    pub conflicting_data: Vec<Vec<u8>>,
    /// Signatures proving the violation
    pub signatures: Vec<Vec<u8>>,
    /// Timestamp of violation
    pub violation_time: u64,
    /// Merkle proof of inclusion (if applicable)
    pub merkle_proof: Option<Vec<u8>>,
    /// Witnesses who reported the violation
    pub reporters: Vec<String>,
}

/// Types of slashing evidence
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EvidenceType {
    /// Two different blocks signed at same height
    DoubleSignEvidence,
    /// Missed blocks attestation
    DowntimeEvidence,
    /// Invalid state transition proof
    InvalidStateProof,
    /// Gradient analysis showing Byzantine behavior
    GradientAnalysis,
    /// Graph clustering showing cartel
    CartelProof,
    /// Failed cross-chain verification
    CrossChainVerificationFailure,
}

/// Cross-Chain Lock Proof
///
/// Cryptographic proof that assets are locked on their native chain
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CrossChainLockProof {
    /// Unique proof identifier
    pub id: String,
    /// Stake this proof belongs to
    pub stake_id: String,
    /// Chain where assets are locked
    pub source_chain: ChainType,
    /// Asset locked
    pub asset: AssetType,
    /// Amount locked
    pub amount: u128,
    /// Lock contract address
    pub lock_contract: String,
    /// Transaction hash of lock
    pub lock_tx_hash: Vec<u8>,
    /// Block number of lock
    pub lock_block: u64,
    /// Merkle proof of transaction inclusion
    pub merkle_proof: Vec<u8>,
    /// Block header hash
    pub block_header_hash: Vec<u8>,
    /// Validator signatures on block (for light client verification)
    pub validator_signatures: Vec<Vec<u8>>,
    /// Proof creation timestamp
    pub created_at: Timestamp,
    /// Proof expiration (must be refreshed)
    pub expires_at: Timestamp,
    /// Verification status
    pub verified: bool,
}

/// Supported chains
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ChainType {
    Ethereum,
    Polygon,
    Arbitrum,
    Optimism,
    Holochain,
}

/// Asset types
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssetType {
    MYC,
    ETH,
    USDC,
    WETH,
}

/// Escrow with cryptographic release conditions
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CryptoEscrow {
    /// Unique escrow identifier
    pub id: String,
    /// Depositor DID
    pub depositor_did: String,
    /// Beneficiary DID
    pub beneficiary_did: String,
    /// Amount escrowed per asset
    pub myc_amount: u128,
    pub eth_amount: u128,
    pub usdc_amount: u128,
    /// Escrow purpose
    pub purpose: String,
    /// Release conditions
    pub conditions: Vec<ReleaseCondition>,
    /// Number of conditions that must be met
    pub required_conditions: u8,
    /// Conditions currently met
    pub met_conditions: Vec<u8>,
    /// Hash commitment of release secret (for hash-lock)
    pub hash_lock: Option<Vec<u8>>,
    /// Timelock expiration
    pub timelock: Option<Timestamp>,
    /// Multi-sig threshold (e.g., 2-of-3)
    pub multisig_threshold: Option<u8>,
    /// Multi-sig signers
    pub multisig_signers: Vec<String>,
    /// Collected signatures
    pub collected_signatures: Vec<EscrowSignature>,
    /// Escrow status
    pub status: EscrowStatus,
    /// Creation timestamp
    pub created_at: Timestamp,
    /// Release timestamp
    pub released_at: Option<Timestamp>,
}

/// Release conditions for escrow
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ReleaseCondition {
    /// Time-based release
    Timelock { release_time: i64 },
    /// Hash preimage reveal
    HashLock {
        hash: Vec<u8>,
        hash_type: EscrowHashType,
    },
    /// Multi-signature approval
    MultiSig { threshold: u8, signers: Vec<String> },
    /// Oracle price attestation
    OraclePrice {
        oracle_id: String,
        asset: String,
        min_price: f64,
    },
    /// Governance proposal passed
    GovernanceApproval { proposal_id: String },
    /// K-Vector threshold met
    TrustThreshold { min_trust_score: f32 },
    /// External contract call (verified via cross-chain proof)
    ExternalContractState {
        chain: ChainType,
        contract: String,
        expected_state: Vec<u8>,
    },
}

/// Hash types for hash-lock
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EscrowHashType {
    Sha256,
    Sha3_256,
    Blake2b,
    Keccak256,
}

/// Signature on escrow release
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EscrowSignature {
    pub signer_did: String,
    pub signature: Vec<u8>,
    pub signed_at: i64,
}

/// Escrow status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EscrowStatus {
    /// Awaiting condition fulfillment
    Pending,
    /// Conditions met, awaiting release
    Releasable,
    /// Released to beneficiary
    Released,
    /// Refunded to depositor
    Refunded,
    /// Disputed
    Disputed,
    /// Expired without resolution
    Expired,
}

/// Staking rewards distribution
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RewardDistribution {
    /// Distribution ID
    pub id: String,
    /// Epoch number
    pub epoch: u64,
    /// Total rewards distributed (in MYC)
    pub total_rewards: u128,
    /// Per-staker allocations
    pub allocations: Vec<RewardAllocation>,
    /// Distribution timestamp
    pub distributed_at: Timestamp,
    /// Merkle root of allocations (for efficient claiming)
    pub merkle_root: Vec<u8>,
}

/// Individual reward allocation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RewardAllocation {
    pub stake_id: String,
    pub staker_did: String,
    pub amount: u128,
    pub merkle_proof: Vec<u8>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    TriAssetStake(TriAssetStake),
    SlashingEvent(SlashingEvent),
    CrossChainLockProof(CrossChainLockProof),
    CryptoEscrow(CryptoEscrow),
    RewardDistribution(RewardDistribution),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Staker to their stakes
    StakerToStake,
    /// Stake to slashing events
    StakeToSlashing,
    /// Stake to cross-chain proofs
    StakeToProofs,
    /// Escrow to depositor
    DepositorToEscrow,
    /// Escrow to beneficiary
    BeneficiaryToEscrow,
    /// Epoch to reward distribution
    EpochToRewards,
    /// Active stakes anchor
    ActiveStakes,
}

/// Validation callback
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::TriAssetStake(stake) => validate_create_stake(action, stake),
                EntryTypes::SlashingEvent(event) => validate_slashing_event(action, event),
                EntryTypes::CrossChainLockProof(proof) => validate_cross_chain_proof(action, proof),
                EntryTypes::CryptoEscrow(escrow) => validate_escrow(action, escrow),
                EntryTypes::RewardDistribution(dist) => validate_reward_distribution(action, dist),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::TriAssetStake(stake) => validate_update_stake(action, stake),
                EntryTypes::CryptoEscrow(escrow) => validate_update_escrow(action, escrow),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::StakerToStake => Ok(ValidateCallbackResult::Valid),
            LinkTypes::StakeToSlashing => Ok(ValidateCallbackResult::Valid),
            LinkTypes::StakeToProofs => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DepositorToEscrow => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BeneficiaryToEscrow => Ok(ValidateCallbackResult::Valid),
            LinkTypes::EpochToRewards => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ActiveStakes => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate stake creation
fn validate_create_stake(
    _action: Create,
    stake: TriAssetStake,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !stake.staker_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Staker must be a valid DID".into(),
        ));
    }

    // Validate minimum stake amounts
    // MYC: $10K minimum (assuming 1 MYC = $1, 18 decimals)
    const MIN_MYC: u128 = 10_000 * 10u128.pow(18);
    if stake.myc_amount < MIN_MYC {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "MYC stake must be at least {} (10K MYC)",
            MIN_MYC
        )));
    }

    // ETH: 32 ETH minimum (18 decimals)
    const MIN_ETH: u128 = 32 * 10u128.pow(18);
    if stake.eth_amount < MIN_ETH {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "ETH stake must be at least {} (32 ETH)",
            MIN_ETH
        )));
    }

    // USDC: $50K minimum (6 decimals)
    const MIN_USDC: u128 = 50_000 * 10u128.pow(6);
    if stake.usdc_amount < MIN_USDC {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "USDC stake must be at least {} (50K USDC)",
            MIN_USDC
        )));
    }

    // Validate K-Vector trust score range
    if stake.kvector_trust_score < 0.0 || stake.kvector_trust_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector trust score must be in [0, 1]".into(),
        ));
    }

    // Validate stake weight range
    if stake.stake_weight < 1.0 || stake.stake_weight > 2.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Stake weight must be in [1.0, 2.0]".into(),
        ));
    }

    // Cross-chain proofs must not be empty
    if stake.myc_lock_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "MYC lock proof is required".into(),
        ));
    }
    if stake.eth_lock_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "ETH lock proof is required".into(),
        ));
    }
    if stake.usdc_lock_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "USDC lock proof is required".into(),
        ));
    }

    // New stake must be Active
    if stake.status != StakeStatus::Active {
        return Ok(ValidateCallbackResult::Invalid(
            "New stake must have Active status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate stake update
fn validate_update_stake(
    _action: Update,
    stake: TriAssetStake,
) -> ExternResult<ValidateCallbackResult> {
    // Can only update stake weight or status
    if stake.kvector_trust_score < 0.0 || stake.kvector_trust_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "K-Vector trust score must be in [0, 1]".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate slashing event
fn validate_slashing_event(
    _action: Create,
    event: SlashingEvent,
) -> ExternResult<ValidateCallbackResult> {
    // Slash percentage must be valid
    if event.slash_percentage > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Slash percentage cannot exceed 100".into(),
        ));
    }

    // Evidence hash must be 32 bytes
    if event.evidence_hash.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Evidence hash must be 32 bytes".into(),
        ));
    }

    // Evidence must not be empty
    if event.evidence.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Slashing evidence is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate cross-chain proof
fn validate_cross_chain_proof(
    _action: Create,
    proof: CrossChainLockProof,
) -> ExternResult<ValidateCallbackResult> {
    // Transaction hash must be 32 bytes
    if proof.lock_tx_hash.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction hash must be 32 bytes".into(),
        ));
    }

    // Block header hash must be 32 bytes
    if proof.block_header_hash.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Block header hash must be 32 bytes".into(),
        ));
    }

    // Merkle proof must not be empty
    if proof.merkle_proof.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Merkle proof is required".into(),
        ));
    }

    // Must have at least one validator signature
    if proof.validator_signatures.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one validator signature is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate escrow
fn validate_escrow(_action: Create, escrow: CryptoEscrow) -> ExternResult<ValidateCallbackResult> {
    // Depositor must be a valid DID
    if !escrow.depositor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Depositor must be a valid DID".into(),
        ));
    }

    // Beneficiary must be a valid DID
    if !escrow.beneficiary_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Beneficiary must be a valid DID".into(),
        ));
    }

    // Must have at least one condition
    if escrow.conditions.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one release condition is required".into(),
        ));
    }

    // Required conditions cannot exceed total conditions
    if escrow.required_conditions as usize > escrow.conditions.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Required conditions cannot exceed total conditions".into(),
        ));
    }

    // New escrow must be Pending
    if escrow.status != EscrowStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New escrow must have Pending status".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate escrow update
fn validate_update_escrow(
    _action: Update,
    escrow: CryptoEscrow,
) -> ExternResult<ValidateCallbackResult> {
    // Basic validation
    if !escrow.depositor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Depositor must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate reward distribution
fn validate_reward_distribution(
    _action: Create,
    dist: RewardDistribution,
) -> ExternResult<ValidateCallbackResult> {
    // Merkle root must be 32 bytes
    if dist.merkle_root.len() != 32 {
        return Ok(ValidateCallbackResult::Invalid(
            "Merkle root must be 32 bytes".into(),
        ));
    }

    // Must have at least one allocation
    if dist.allocations.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "At least one allocation is required".into(),
        ));
    }

    // Sum of allocations must equal total rewards
    let sum: u128 = dist.allocations.iter().map(|a| a.amount).sum();
    if sum != dist.total_rewards {
        return Ok(ValidateCallbackResult::Invalid(
            "Sum of allocations must equal total rewards".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
