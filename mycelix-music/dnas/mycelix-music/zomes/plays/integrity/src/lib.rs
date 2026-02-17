//! Plays Integrity Zome
//!
//! Defines validation rules for play records.
//! CRITICAL: Plays are recorded on the listener's source chain - ZERO COST.
//! Only aggregated settlements touch the blockchain.

use hdi::prelude::*;

/// Play record - stored on listener's source chain (FREE!)
/// This is the magic of Holochain - each play is just a local entry.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PlayRecord {
    /// Song action hash (reference to catalog)
    pub song_hash: ActionHash,
    /// Artist's agent public key (for aggregation)
    pub artist: AgentPubKey,
    /// Timestamp of the play
    pub played_at: Timestamp,
    /// Duration listened (seconds) - for partial play tracking
    pub duration_listened: u32,
    /// Total song duration (for completion percentage)
    pub song_duration: u32,
    /// Strategy that was active at play time
    pub strategy_id: String,
    /// Calculated micro-payment amount (in wei equivalent)
    pub amount_owed: u64,
    /// Whether this play has been included in a settlement batch
    pub settled: bool,
    /// Settlement batch hash (if settled)
    pub settlement_hash: Option<ActionHash>,
}

/// Play attestation - signed by listener, can be verified
/// Used when plays need to be proven to others
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PlayAttestation {
    /// The play record this attests to
    pub play_hash: ActionHash,
    /// Song hash for quick lookup
    pub song_hash: ActionHash,
    /// Artist who should receive payment
    pub artist: AgentPubKey,
    /// Amount owed for this play
    pub amount_owed: u64,
    /// Signature of the play record by listener
    pub listener_signature: Vec<u8>,
}

/// Settlement batch - aggregates many plays for efficient on-chain settlement
/// This is what actually touches the blockchain - batched for efficiency
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SettlementBatch {
    /// Artist receiving payment
    pub artist: AgentPubKey,
    /// Total plays in this batch
    pub play_count: u64,
    /// Total amount to settle (in wei)
    pub total_amount: u64,
    /// Play record hashes included
    pub play_hashes: Vec<ActionHash>,
    /// Merkle root of play hashes (for efficient verification)
    pub merkle_root: Vec<u8>,
    /// When this batch was created
    pub created_at: Timestamp,
    /// On-chain settlement status
    pub status: SettlementStatus,
    /// Transaction hash if settled on-chain
    pub tx_hash: Option<String>,
}

/// Settlement status
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum SettlementStatus {
    /// Batch created, awaiting settlement
    Pending,
    /// Settlement submitted to blockchain
    Submitted,
    /// Settlement confirmed on-chain
    Confirmed,
    /// Settlement failed (will retry)
    Failed,
}

/// Link types for plays
#[hdk_link_types]
pub enum LinkTypes {
    /// Listener -> Their play records
    ListenerToPlays,
    /// Song -> Play records
    SongToPlays,
    /// Artist -> Settlement batches
    ArtistToSettlements,
    /// Play -> Settlement batch
    PlayToSettlement,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    PlayRecord(PlayRecord),
    PlayAttestation(PlayAttestation),
    SettlementBatch(SettlementBatch),
}

/// Validation
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::PlayRecord(play) => validate_create_play(play, action),
                EntryTypes::PlayAttestation(attestation) => {
                    validate_create_attestation(attestation, action)
                }
                EntryTypes::SettlementBatch(batch) => validate_create_settlement(batch, action),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_play(play: PlayRecord, _action: Create) -> ExternResult<ValidateCallbackResult> {
    // Duration listened cannot exceed song duration
    if play.duration_listened > play.song_duration {
        return Ok(ValidateCallbackResult::Invalid(
            "Duration listened cannot exceed song duration".to_string(),
        ));
    }

    // Plays cannot be pre-settled
    if play.settled {
        return Ok(ValidateCallbackResult::Invalid(
            "New plays must have settled=false".to_string(),
        ));
    }

    // Play must reference a song
    if play.song_hash.as_ref().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Play must reference a song".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_attestation(
    _attestation: PlayAttestation,
    _action: Create,
) -> ExternResult<ValidateCallbackResult> {
    // Attestation signature verification would happen here
    // For now, accept all attestations
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_settlement(
    batch: SettlementBatch,
    _action: Create,
) -> ExternResult<ValidateCallbackResult> {
    // Settlement must have plays
    if batch.play_count == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Settlement batch must contain at least one play".to_string(),
        ));
    }

    // Play count must match hashes
    if batch.play_count as usize != batch.play_hashes.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Play count must match number of play hashes".to_string(),
        ));
    }

    // New settlements must be pending
    if batch.status != SettlementStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "New settlements must have Pending status".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
