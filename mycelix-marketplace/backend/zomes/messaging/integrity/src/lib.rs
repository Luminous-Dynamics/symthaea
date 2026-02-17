/// Messaging Integrity Zome
///
/// This zome provides P2P encrypted messaging for the Mycelix-Marketplace.
/// Features:
/// - End-to-end encrypted messages
/// - Conversation threading (by listing or transaction)
/// - Rich media attachments (IPFS)
/// - Read receipts and typing indicators
/// - MATL-gated messaging (spam prevention)
/// - Message search and filtering

use hdi::prelude::*;

/// Message entry - represents a single message in a conversation
///
/// Messages are encrypted client-side before being stored on the DHT.
/// Only the sender and recipient can decrypt the content.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Message {
    /// Sender's public key
    pub sender: AgentPubKey,

    /// Recipient's public key
    pub recipient: AgentPubKey,

    /// Encrypted message content (AES-256-GCM)
    /// Structure when decrypted: {"text": "...", "attachments": [...]}
    pub encrypted_content: String,

    /// Optional: Link to listing this message relates to
    pub listing_hash: Option<ActionHash>,

    /// Optional: Link to transaction this message relates to
    pub transaction_hash: Option<ActionHash>,

    /// Conversation thread ID (hash of first message in thread)
    pub conversation_id: ActionHash,

    /// Message sent timestamp
    pub sent_at: u64,

    /// Message read timestamp (None if unread)
    pub read_at: Option<u64>,

    /// Message type for UI rendering
    pub message_type: MessageType,

    /// Epistemic classification
    pub epistemic: EpistemicClassification,
}

/// Message types for different UI contexts
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum MessageType {
    /// Regular text message
    Text,

    /// System message (transaction update, etc.)
    System,

    /// Offer message (price negotiation)
    Offer,

    /// Question about listing
    Question,

    /// Review request
    ReviewRequest,
}

/// Conversation metadata - summary of a message thread
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Conversation {
    /// Participants in conversation (usually 2)
    pub participants: Vec<AgentPubKey>,

    /// Related listing (if any)
    pub listing_hash: Option<ActionHash>,

    /// Related transaction (if any)
    pub transaction_hash: Option<ActionHash>,

    /// Conversation subject/title
    pub subject: String,

    /// First message hash (conversation ID)
    pub first_message_hash: ActionHash,

    /// Last message hash
    pub last_message_hash: ActionHash,

    /// Total message count
    pub message_count: u32,

    /// Unread count for each participant
    pub unread_counts: Vec<(AgentPubKey, u32)>,

    /// Conversation started timestamp
    pub started_at: u64,

    /// Last activity timestamp
    pub last_activity_at: u64,

    /// Conversation status
    pub status: ConversationStatus,
}

/// Conversation status
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum ConversationStatus {
    /// Active conversation
    Active,

    /// Archived by one or both participants
    Archived,

    /// Muted notifications
    Muted,

    /// Blocked (spam/abuse)
    Blocked,
}

/// Read receipt - tracks when messages are read
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReadReceipt {
    /// Message that was read
    pub message_hash: ActionHash,

    /// Who read it
    pub reader: AgentPubKey,

    /// When it was read
    pub read_at: u64,
}

/// Typing indicator - ephemeral signal that user is typing
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct TypingIndicator {
    /// Conversation ID
    pub conversation_id: ActionHash,

    /// Who is typing
    pub typer: AgentPubKey,

    /// Timestamp (expires after 5 seconds)
    pub timestamp: u64,
}

/// Message attachment metadata (actual file on IPFS)
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct Attachment {
    /// IPFS CID
    pub ipfs_cid: String,

    /// Original filename
    pub filename: String,

    /// MIME type
    pub mime_type: String,

    /// File size in bytes
    pub size_bytes: u64,

    /// Optional thumbnail CID (for images/videos)
    pub thumbnail_cid: Option<String>,
}

/// Epistemic classification for messages
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub struct EpistemicClassification {
    /// Empirical level
    pub empirical: EmpiricalLevel,

    /// Normative level
    pub normative: NormativeLevel,

    /// Materiality level
    pub materiality: MaterialityLevel,
}

/// Empirical classification levels
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum EmpiricalLevel {
    /// E1: Testimonial (sender claims, recipient trusts)
    E1Testimonial,

    /// E2: Privately verifiable (both can verify)
    E2PrivateVerify,

    /// E3: Publicly verifiable (anyone can verify)
    E3PublicVerify,
}

/// Normative classification levels
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum NormativeLevel {
    /// N1: Communal (buyer-seller agreement)
    N1Communal,

    /// N2: Marketplace (whole marketplace standards)
    N2Marketplace,

    /// N3: Universal (global standards)
    N3Universal,
}

/// Materiality classification levels
#[derive(Clone, PartialEq, Serialize, Deserialize, Debug)]
pub enum MaterialityLevel {
    /// M1: Temporal (ephemeral messages)
    M1Temporal,

    /// M2: Persistent (archived conversations)
    M2Persistent,

    /// M3: Permanent (dispute evidence)
    M3Permanent,
}

/// Entry types for messaging zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Message(Message),
    Conversation(Conversation),
    ReadReceipt(ReadReceipt),
}

/// Link types for messaging relationships
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> Conversations (as participant)
    AgentToConversations,

    /// Agent -> Sent Messages
    AgentToSentMessages,

    /// Agent -> Received Messages
    AgentToReceivedMessages,

    /// Conversation -> Messages
    ConversationToMessages,

    /// Message -> Read Receipts
    MessageToReadReceipts,

    /// Listing -> Conversations
    ListingToConversations,

    /// Transaction -> Conversations
    TransactionToConversations,
}

/// Validation rules for messages

/// Validate message creation
pub fn validate_create_message(message: Message, action: &Create) -> ExternResult<ValidateCallbackResult> {
    // Verify sender matches agent creating entry
    if message.sender != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Sender must match creating agent".to_string()
        ));
    }

    // Verify timestamp is reasonable (within 5 minutes of action timestamp)
    let action_time = action.timestamp.as_micros() as u64;
    let five_minutes = 300_000_000_u64; // 5 min in microseconds

    if message.sent_at > action_time + five_minutes || message.sent_at < action_time.saturating_sub(five_minutes) {
        return Ok(ValidateCallbackResult::Invalid(
            "Message timestamp out of valid range".to_string()
        ));
    }

    // Verify encrypted content is not empty
    if message.encrypted_content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Message content cannot be empty".to_string()
        ));
    }

    // Verify encrypted content is not too large (10KB limit)
    if message.encrypted_content.len() > 10_000 {
        return Ok(ValidateCallbackResult::Invalid(
            "Message content too large (max 10KB)".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate conversation creation
pub fn validate_create_conversation(conversation: Conversation) -> ExternResult<ValidateCallbackResult> {
    // Verify at least 2 participants
    if conversation.participants.len() < 2 {
        return Ok(ValidateCallbackResult::Invalid(
            "Conversation must have at least 2 participants".to_string()
        ));
    }

    // Verify subject is not empty
    if conversation.subject.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Conversation subject cannot be empty".to_string()
        ));
    }

    // Verify timestamps are valid
    if conversation.last_activity_at < conversation.started_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Last activity cannot be before start time".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate read receipt creation
pub fn validate_create_read_receipt(receipt: ReadReceipt, action: &Create) -> ExternResult<ValidateCallbackResult> {
    // Verify reader matches creating agent
    if receipt.reader != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Reader must match creating agent".to_string()
        ));
    }

    // Verify timestamp is reasonable
    let action_time = action.timestamp.as_micros() as u64;
    let five_minutes = 300_000_000_u64;

    if receipt.read_at > action_time + five_minutes {
        return Ok(ValidateCallbackResult::Invalid(
            "Read timestamp cannot be in the future".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// Validation function dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Message(message) => validate_create_message(message, &action),
                EntryTypes::Conversation(conversation) => validate_create_conversation(conversation),
                EntryTypes::ReadReceipt(receipt) => validate_create_read_receipt(receipt, &action),
            },
            OpEntry::UpdateEntry {
                app_entry, ..
            } => match app_entry {
                EntryTypes::Message(_message) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Conversation(_conversation) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ReadReceipt(_receipt) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterUpdate(update_entry) => match update_entry {
            OpUpdate::Entry { app_entry, .. } => match app_entry {
                EntryTypes::Message(_message) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Conversation(_conversation) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ReadReceipt(_receipt) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDelete(_delete_entry) => {
            // Allow deletion of messages, conversations, receipts
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => {
            // Validate link creation based on type
            match link_type {
                LinkTypes::AgentToConversations => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AgentToSentMessages => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AgentToReceivedMessages => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ConversationToMessages => Ok(ValidateCallbackResult::Valid),
                LinkTypes::MessageToReadReceipts => Ok(ValidateCallbackResult::Valid),
                LinkTypes::ListingToConversations => Ok(ValidateCallbackResult::Valid),
                LinkTypes::TransactionToConversations => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
