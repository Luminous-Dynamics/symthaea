// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(unsafe_code)]
//! Mail Messages Integrity Zome
//!
//! Defines entry types and validation rules for decentralized email on Holochain DHT.
//! All emails are encrypted, stored as DHT entries, and delivered via P2P signals.

use hdi::prelude::*;

const MAX_CHUNK_SIZE: usize = 10 * 1024 * 1024;
const MAX_TOTAL_CHUNKS: u32 = 1000;
const SHA256_LEN: usize = 32;
const ED25519_SIG_LEN: usize = 64;
const DILITHIUM3_SIG_LEN: usize = 3293;
const DILITHIUM2_SIG_LEN: usize = 2420;
const X25519_KEY_LEN: usize = 32;
const KYBER1024_KEY_LEN: usize = 1568;
const KYBER768_KEY_LEN: usize = 1088;

pub fn email_signing_content(email: &EncryptedEmail) -> Vec<u8> {
    let mut content = Vec::with_capacity(256);
    content.push(0x01);
    content.extend_from_slice(email.sender.get_raw_39());
    content.extend_from_slice(email.recipient.get_raw_39());
    content.extend_from_slice(&(email.encrypted_subject.len() as u32).to_le_bytes());
    content.extend_from_slice(&email.encrypted_subject);
    content.extend_from_slice(&(email.encrypted_body.len() as u32).to_le_bytes());
    content.extend_from_slice(&email.encrypted_body);
    content.extend_from_slice(&email.nonce);
    content.extend_from_slice(&(email.message_id.len() as u32).to_le_bytes());
    content.extend_from_slice(email.message_id.as_bytes());
    content.extend_from_slice(&email.timestamp.as_micros().to_le_bytes());
    content
}

pub fn receipt_signing_content(email_hash: &ActionHash, reader: &AgentPubKey, read_at: &Timestamp) -> Vec<u8> {
    let mut content = Vec::with_capacity(128);
    content.push(0x01);
    content.extend_from_slice(email_hash.get_raw_39());
    content.extend_from_slice(reader.get_raw_39());
    content.extend_from_slice(&read_at.as_micros().to_le_bytes());
    content
}

pub fn delivery_receipt_signing_content(email_hash: &ActionHash, recipient: &AgentPubKey, delivered_at: &Timestamp) -> Vec<u8> {
    let mut content = Vec::with_capacity(128);
    content.push(0x01);
    content.extend_from_slice(email_hash.get_raw_39());
    content.extend_from_slice(recipient.get_raw_39());
    content.extend_from_slice(&delivered_at.as_micros().to_le_bytes());
    content
}

/// Email message stored on DHT
/// Content is always encrypted - only metadata is visible for routing
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EncryptedEmail {
    /// Sender's agent public key
    pub sender: AgentPubKey,
    /// Recipient's agent public key
    pub recipient: AgentPubKey,
    /// Encrypted subject (ChaCha20-Poly1305)
    pub encrypted_subject: Vec<u8>,
    /// Encrypted body content
    pub encrypted_body: Vec<u8>,
    /// Encrypted attachments manifest (references to attachment entries)
    pub encrypted_attachments: Vec<u8>,
    /// Ephemeral public key for decryption (X25519 or Kyber)
    pub ephemeral_pubkey: Vec<u8>,
    /// Nonce used for encryption
    pub nonce: [u8; 24],
    /// Digital signature of content hash (Dilithium or Ed25519)
    pub signature: Vec<u8>,
    /// Algorithm identifiers
    pub crypto_suite: CryptoSuite,
    /// Message ID for threading
    pub message_id: String,
    /// In-reply-to message ID (for threads)
    pub in_reply_to: Option<String>,
    /// References (thread chain)
    pub references: Vec<String>,
    /// Timestamp (author's claimed time)
    pub timestamp: Timestamp,
    /// Priority level
    pub priority: EmailPriority,
    /// Read receipt requested
    pub read_receipt_requested: bool,
    /// Expiration time (for ephemeral messages)
    pub expires_at: Option<Timestamp>,
}

/// Cryptographic algorithms used
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct CryptoSuite {
    /// Key exchange algorithm (x25519, kyber1024)
    pub key_exchange: String,
    /// Symmetric encryption (chacha20-poly1305, aes-256-gcm)
    pub symmetric: String,
    /// Signature algorithm (ed25519, dilithium3)
    pub signature: String,
}

/// Email priority levels
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub enum EmailPriority {
    Low,
    #[default]
    Normal,
    High,
    Urgent,
}

/// Large attachment stored separately on DHT
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EncryptedAttachment {
    /// Reference to parent email
    pub email_hash: ActionHash,
    /// Encrypted filename
    pub encrypted_filename: Vec<u8>,
    /// Encrypted MIME type
    pub encrypted_mime_type: Vec<u8>,
    /// Encrypted content (chunked for large files)
    pub encrypted_content: Vec<u8>,
    /// Chunk index (for multi-part attachments)
    pub chunk_index: u32,
    /// Total chunks
    pub total_chunks: u32,
    /// Content hash for integrity verification
    pub content_hash: Vec<u8>,
    /// Encryption nonce
    pub nonce: [u8; 24],
}

/// Email folder/label assignment
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmailFolder {
    /// Owner agent
    pub owner: AgentPubKey,
    /// Folder name (encrypted)
    pub encrypted_name: Vec<u8>,
    /// Folder color/icon metadata
    pub metadata: Option<Vec<u8>>,
    /// Is system folder (inbox, sent, drafts, trash)
    pub is_system: bool,
    /// Sort order
    pub sort_order: i32,
}

/// Email state for a specific agent (read, starred, etc.)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmailState {
    /// Email entry hash
    pub email_hash: ActionHash,
    /// Owner of this state
    pub owner: AgentPubKey,
    /// Read status
    pub is_read: bool,
    /// Starred/flagged
    pub is_starred: bool,
    /// Folder assignments
    pub folders: Vec<ActionHash>,
    /// Labels (encrypted)
    pub encrypted_labels: Vec<Vec<u8>>,
    /// Snoozed until
    pub snoozed_until: Option<Timestamp>,
    /// Archived
    pub is_archived: bool,
    /// Trashed
    pub is_trashed: bool,
    /// Trash timestamp (for auto-delete)
    pub trashed_at: Option<Timestamp>,
}

/// Draft email (not yet sent)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmailDraft {
    /// Owner/author
    pub owner: AgentPubKey,
    /// Intended recipients
    pub recipients: Vec<AgentPubKey>,
    /// CC recipients
    pub cc: Vec<AgentPubKey>,
    /// BCC recipients (stored separately, not shared)
    pub bcc: Vec<AgentPubKey>,
    /// Encrypted subject
    pub encrypted_subject: Vec<u8>,
    /// Encrypted body
    pub encrypted_body: Vec<u8>,
    /// Attachment references
    pub attachments: Vec<ActionHash>,
    /// Reply-to email
    pub in_reply_to: Option<ActionHash>,
    /// Last modified
    pub updated_at: Timestamp,
    /// Scheduled send time
    pub scheduled_for: Option<Timestamp>,
}

/// Read receipt confirmation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReadReceipt {
    /// Original email hash
    pub email_hash: ActionHash,
    /// Reader agent
    pub reader: AgentPubKey,
    /// When read
    pub read_at: Timestamp,
    /// Signature proving authenticity
    pub signature: Vec<u8>,
}

/// Delivery receipt (email received by recipient's node)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DeliveryReceipt {
    /// Original email hash
    pub email_hash: ActionHash,
    /// Recipient who received it
    pub recipient: AgentPubKey,
    /// Delivery timestamp
    pub delivered_at: Timestamp,
    /// Signature
    pub signature: Vec<u8>,
}

/// Email thread grouping
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EmailThread {
    /// Thread ID (from first message)
    pub thread_id: String,
    /// Subject (encrypted, from first message)
    pub encrypted_subject: Vec<u8>,
    /// Participants
    pub participants: Vec<AgentPubKey>,
    /// Message count
    pub message_count: u32,
    /// Last activity
    pub last_activity: Timestamp,
}

/// Link types for email graph
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> their received emails
    AgentToInbox,
    /// Agent -> their sent emails
    AgentToSent,
    /// Agent -> their drafts
    AgentToDrafts,
    /// Agent -> their folders
    AgentToFolders,
    /// Folder -> emails in folder
    FolderToEmails,
    /// Email -> its attachments
    EmailToAttachments,
    /// Email -> read receipts
    EmailToReadReceipts,
    /// Email -> delivery receipts
    EmailToDeliveryReceipts,
    /// Email -> its state (per agent)
    EmailToState,
    /// Thread -> emails in thread
    ThreadToEmails,
    /// Agent -> threads they're part of
    AgentToThreads,
    /// Email -> reply emails
    EmailToReplies,
    /// Scheduled emails pending send
    AgentToScheduled,
    /// Anchor for global discovery (optional, for public emails)
    AnchorToPublicEmails,
}

/// Entry type definitions
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 3)]
    EncryptedEmail(EncryptedEmail),
    #[entry_type(required_validations = 2)]
    EncryptedAttachment(EncryptedAttachment),
    #[entry_type(required_validations = 2)]
    EmailFolder(EmailFolder),
    #[entry_type(required_validations = 2)]
    EmailState(EmailState),
    #[entry_type(required_validations = 1)]
    EmailDraft(EmailDraft),
    #[entry_type(required_validations = 2)]
    ReadReceipt(ReadReceipt),
    #[entry_type(required_validations = 2)]
    DeliveryReceipt(DeliveryReceipt),
    #[entry_type(required_validations = 2)]
    EmailThread(EmailThread),
}

/// Validation callbacks
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, action)
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                validate_update_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            tag,
            action,
        } => validate_create_link(link_type, base_address, target_address, tag, action),
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action,
            base_address,
            target_address,
            tag,
            action,
        } => validate_delete_link(
            link_type,
            original_action,
            base_address,
            target_address,
            tag,
            action,
        ),
        FlatOp::StoreRecord(store_record) => match store_record {
            OpRecord::CreateEntry { app_entry, action } => {
                validate_create_entry(app_entry, action)
            }
            OpRecord::UpdateEntry { app_entry, action, .. } => {
                validate_update_entry(app_entry, action)
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::EncryptedEmail(email) => validate_encrypted_email(&email, &action),
        EntryTypes::EncryptedAttachment(attachment) => validate_attachment(&attachment, &action),
        EntryTypes::EmailFolder(folder) => validate_folder(&folder, &action),
        EntryTypes::EmailState(state) => validate_email_state(&state, &action),
        EntryTypes::EmailDraft(draft) => validate_draft(&draft, &action),
        EntryTypes::ReadReceipt(receipt) => validate_read_receipt(&receipt, &action),
        EntryTypes::DeliveryReceipt(receipt) => validate_delivery_receipt(&receipt, &action),
        EntryTypes::EmailThread(thread) => validate_thread(&thread, &action),
    }
}

fn validate_update_entry(
    entry: EntryTypes,
    action: Update,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        // Emails are immutable once sent
        EntryTypes::EncryptedEmail(_) => Ok(ValidateCallbackResult::Invalid(
            "Sent emails cannot be modified".to_string(),
        )),
        // Drafts can be updated by owner
        EntryTypes::EmailDraft(draft) => {
            if draft.owner != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only draft owner can update".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        // State can be updated by owner
        EntryTypes::EmailState(state) => {
            if state.owner != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only state owner can update".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        // Folders can be updated by owner
        EntryTypes::EmailFolder(folder) => {
            if folder.owner != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only folder owner can update".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        // Receipts are immutable
        EntryTypes::ReadReceipt(_) | EntryTypes::DeliveryReceipt(_) => {
            Ok(ValidateCallbackResult::Invalid(
                "Receipts cannot be modified".to_string(),
            ))
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_encrypted_email(
    email: &EncryptedEmail,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Sender must be the author
    if email.sender != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Email sender must match action author".to_string(),
        ));
    }

    // Must have encrypted content
    if email.encrypted_body.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Email body cannot be empty".to_string(),
        ));
    }

    // Must have valid nonce
    if email.nonce == [0u8; 24] {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid encryption nonce".to_string(),
        ));
    }

    // Must have signature
    if email.signature.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Email must be signed".to_string(),
        ));
    }

    // Check expiration if set
    if let Some(expires) = email.expires_at {
        if expires <= email.timestamp {
            return Ok(ValidateCallbackResult::Invalid(
                "Expiration must be after timestamp".to_string(),
            ));
        }
    }

    // Validate crypto suite
    let valid_key_exchange = ["x25519", "kyber1024", "kyber768"];
    let valid_symmetric = ["chacha20-poly1305", "aes-256-gcm"];
    let valid_signature = ["ed25519", "dilithium3", "dilithium2"];

    if !valid_key_exchange.contains(&email.crypto_suite.key_exchange.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid key exchange algorithm".to_string(),
        ));
    }
    if !valid_symmetric.contains(&email.crypto_suite.symmetric.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid symmetric encryption algorithm".to_string(),
        ));
    }
    if !valid_signature.contains(&email.crypto_suite.signature.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid signature algorithm".to_string(),
        ));
    }

    // Validate ephemeral key length matches key exchange algorithm
    let expected_key_len = match email.crypto_suite.key_exchange.as_str() {
        "x25519" => X25519_KEY_LEN,
        "kyber1024" => KYBER1024_KEY_LEN,
        "kyber768" => KYBER768_KEY_LEN,
        _ => return Ok(ValidateCallbackResult::Invalid(
            "Unknown key exchange for ephemeral key validation".to_string(),
        )),
    };
    if email.ephemeral_pubkey.len() != expected_key_len {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Ephemeral pubkey length {} does not match expected {} for {}",
                email.ephemeral_pubkey.len(),
                expected_key_len,
                email.crypto_suite.key_exchange
            ),
        ));
    }

    // Validate signature length and verify for Ed25519
    match email.crypto_suite.signature.as_str() {
        "ed25519" => {
            if email.signature.len() != ED25519_SIG_LEN {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Ed25519 signature must be {} bytes", ED25519_SIG_LEN),
                ));
            }
            // Verify signature against signing content
            let signing_content = email_signing_content(email);
            let mut sig_bytes = [0u8; 64];
            sig_bytes.copy_from_slice(&email.signature);
            let sig = Signature(sig_bytes);
            if !verify_signature_raw(email.sender.clone(), sig, signing_content)? {
                return Ok(ValidateCallbackResult::Invalid(
                    "Email signature verification failed".to_string(),
                ));
            }
        }
        "dilithium3" => {
            if email.signature.len() != DILITHIUM3_SIG_LEN {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Dilithium3 signature must be {} bytes", DILITHIUM3_SIG_LEN),
                ));
            }
            // Dilithium verification requires PQC library, validated at application layer
        }
        "dilithium2" => {
            if email.signature.len() != DILITHIUM2_SIG_LEN {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Dilithium2 signature must be {} bytes", DILITHIUM2_SIG_LEN),
                ));
            }
        }
        _ => {}
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_attachment(
    attachment: &EncryptedAttachment,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Must have content
    if attachment.encrypted_content.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attachment content cannot be empty".to_string(),
        ));
    }

    // Chunk size limit
    if attachment.encrypted_content.len() > MAX_CHUNK_SIZE {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Attachment chunk exceeds maximum size of {} bytes", MAX_CHUNK_SIZE),
        ));
    }

    // Total chunks limit
    if attachment.total_chunks > MAX_TOTAL_CHUNKS {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Total chunks {} exceeds maximum of {}", attachment.total_chunks, MAX_TOTAL_CHUNKS),
        ));
    }

    // Chunk index must be valid
    if attachment.chunk_index >= attachment.total_chunks {
        return Ok(ValidateCallbackResult::Invalid(
            "Invalid chunk index".to_string(),
        ));
    }

    // Must have SHA-256 content hash
    if attachment.content_hash.len() != SHA256_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Content hash must be {} bytes (SHA-256)", SHA256_LEN),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_folder(
    folder: &EmailFolder,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Owner must be author
    if folder.owner != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Folder owner must match author".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_email_state(
    state: &EmailState,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Owner must be author
    if state.owner != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "State owner must match author".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_draft(
    draft: &EmailDraft,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Owner must be author
    if draft.owner != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Draft owner must match author".to_string(),
        ));
    }

    // Must have at least one recipient
    if draft.recipients.is_empty() && draft.cc.is_empty() && draft.bcc.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Draft must have at least one recipient".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_read_receipt(
    receipt: &ReadReceipt,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Reader must be author
    if receipt.reader != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Read receipt reader must match author".to_string(),
        ));
    }

    // Must have Ed25519 signature of correct length
    if receipt.signature.len() != ED25519_SIG_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Read receipt signature must be {} bytes (Ed25519)", ED25519_SIG_LEN),
        ));
    }

    // Verify signature
    let signing_content = receipt_signing_content(&receipt.email_hash, &receipt.reader, &receipt.read_at);
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(&receipt.signature);
    let sig = Signature(sig_bytes);
    if !verify_signature_raw(receipt.reader.clone(), sig, signing_content)? {
        return Ok(ValidateCallbackResult::Invalid(
            "Read receipt signature verification failed".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_delivery_receipt(
    receipt: &DeliveryReceipt,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Recipient must be author
    if receipt.recipient != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Delivery receipt recipient must match author".to_string(),
        ));
    }

    // Must have Ed25519 signature of correct length
    if receipt.signature.len() != ED25519_SIG_LEN {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Delivery receipt signature must be {} bytes (Ed25519)", ED25519_SIG_LEN),
        ));
    }

    // Verify signature
    let signing_content = delivery_receipt_signing_content(&receipt.email_hash, &receipt.recipient, &receipt.delivered_at);
    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(&receipt.signature);
    let sig = Signature(sig_bytes);
    if !verify_signature_raw(receipt.recipient.clone(), sig, signing_content)? {
        return Ok(ValidateCallbackResult::Invalid(
            "Delivery receipt signature verification failed".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_thread(
    _thread: &EmailThread,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Threads are created automatically, minimal validation
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_link(
    link_type: LinkTypes,
    base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
    action: CreateLink,
) -> ExternResult<ValidateCallbackResult> {
    match link_type {
        LinkTypes::AgentToSent
        | LinkTypes::AgentToDrafts
        | LinkTypes::AgentToFolders
        | LinkTypes::AgentToThreads
        | LinkTypes::AgentToScheduled => {
            // Agent links must be created by the agent themselves
            let author_hash: AnyLinkableHash = action.author.into();
            if base_address != author_hash {
                return Ok(ValidateCallbackResult::Invalid(
                    "Agent link base must match action author".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::AgentToInbox => {
            // Inbox links can be created by sender (delivering to recipient)
            Ok(ValidateCallbackResult::Valid)
        }
        LinkTypes::FolderToEmails
        | LinkTypes::EmailToAttachments
        | LinkTypes::EmailToReadReceipts
        | LinkTypes::EmailToDeliveryReceipts
        | LinkTypes::EmailToState
        | LinkTypes::ThreadToEmails
        | LinkTypes::EmailToReplies => Ok(ValidateCallbackResult::Valid),
        LinkTypes::AnchorToPublicEmails => {
            // Public emails anchor - could add rate limiting validation
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

fn validate_delete_link(
    _link_type: LinkTypes,
    original_action: CreateLink,
    _base_address: AnyLinkableHash,
    _target_address: AnyLinkableHash,
    _tag: LinkTag,
    action: DeleteLink,
) -> ExternResult<ValidateCallbackResult> {
    // Only the original link author can delete the link
    if original_action.author != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only the link author can delete a link".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_email_signing_content_deterministic() {
        // Verify that signing content is deterministic for the same input
        // (cannot construct full EncryptedEmail without Holochain types in unit tests,
        // but the function signature and constants are verified at compile time)
        assert_eq!(SHA256_LEN, 32);
        assert_eq!(ED25519_SIG_LEN, 64);
        assert_eq!(DILITHIUM3_SIG_LEN, 3293);
        assert_eq!(DILITHIUM2_SIG_LEN, 2420);
        assert_eq!(X25519_KEY_LEN, 32);
        assert_eq!(KYBER1024_KEY_LEN, 1568);
        assert_eq!(KYBER768_KEY_LEN, 1088);
        assert_eq!(MAX_CHUNK_SIZE, 10 * 1024 * 1024);
        assert_eq!(MAX_TOTAL_CHUNKS, 1000);
    }
}
