use hdi::prelude::*;

/// A content entry for the resonance feed.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ContentEntry {
    /// Content title.
    pub title: String,
    /// Content body (or URI to content).
    pub body: String,
    /// Content domain (e.g., "water", "governance", "knowledge").
    pub domain: String,
    /// HDV embedding truncated to 32 bytes (256 bits) for resonance.
    pub hdv_truncated: Vec<u8>,
    /// Content hash (BLAKE3).
    pub content_hash: Vec<u8>,
    /// Creation timestamp (µs since epoch).
    pub created_at: u64,
}

/// A resonance vote on content (replaces "likes" with consciousness-aligned signal).
#[hdk_entry_helper]
#[derive(Clone)]
pub struct ResonanceVote {
    /// Action hash of the content being voted on.
    pub content_hash: ActionHash,
    /// Resonance strength [-1.0, 1.0] (negative = dissonance).
    pub resonance: f64,
    /// Domain of resonance (e.g., "personal", "community", "truth").
    pub domain: String,
    /// Timestamp (µs since epoch).
    pub timestamp_us: u64,
}

/// Anchor entry for deterministic link bases.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(name = "ContentEntry", visibility = "public")]
    ContentEntry(ContentEntry),
    #[entry_type(name = "ResonanceVote", visibility = "public")]
    ResonanceVote(ResonanceVote),
    #[entry_type(name = "Anchor", visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllContent,
    DomainContent,
    ContentToVotes,
    AgentToContent,
    AgentToVotes,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::ContentEntry(entry) => {
                        if entry.title.is_empty() || entry.title.len() > 256 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Title must be 1-256 chars".to_string(),
                            ));
                        }
                        if entry.body.len() > 65536 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Body too long (max 64KB)".to_string(),
                            ));
                        }
                        if entry.domain.len() > 64 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Domain too long (max 64)".to_string(),
                            ));
                        }
                        if entry.hdv_truncated.len() > 32 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "HDV truncation must be <= 32 bytes".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::ResonanceVote(vote) => {
                        if vote.resonance < -1.0 || vote.resonance > 1.0 {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Resonance must be in [-1.0, 1.0]".to_string(),
                            ));
                        }
                        Ok(ValidateCallbackResult::Valid)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. }
        | FlatOp::RegisterDeleteLink { .. }
        | FlatOp::StoreRecord(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_)
        | FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
    }
}
