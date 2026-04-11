// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mail Messages Coordinator Zome
//!
//! Implements P2P email delivery, real-time signals, and message management
//! for Mycelix Mail on Holochain.
//!
//! # Overview
//!
//! This zome is the core of the Mycelix Mail system, handling:
//! - **Email Sending**: End-to-end encrypted P2P delivery
//! - **Inbox/Sent Management**: DHT-based email storage and retrieval
//! - **Threading**: Automatic email thread grouping
//! - **Folders & Labels**: User-defined organization
//! - **Real-time Signals**: Instant notifications for new emails
//!
//! # Architecture
//!
//! Emails are stored encrypted on the DHT with links to sender's "Sent" and
//! recipient's "Inbox". Real-time delivery uses Holochain remote signals.
//!
//! ```text
//! Sender Agent --> DHT Entry (EncryptedEmail) --> Recipient Agent
//!      |                    |                           |
//!      +-- Link: Sent       +-- Link: Inbox            +-- Signal: EmailReceived
//! ```
//!
//! # Key Functions
//!
//! ## Sending
//! - [`send_email`] - Send encrypted email to recipients
//! - [`send_typing_indicator`] - Real-time typing notification
//!
//! ## Querying
//! - [`get_inbox`] - Retrieve inbox emails with filtering
//! - [`get_sent`] - Retrieve sent emails
//! - [`get_email`] - Get full email by hash
//! - [`get_drafts`] - Get saved drafts
//!
//! ## State Management
//! - [`update_email_state`] - Update read/starred/archived status
//! - [`mark_as_read`] - Mark email as read, optionally send receipt
//!
//! ## Organization
//! - [`create_folder`] - Create custom folder
//! - [`get_folders`] - List all folders
//! - [`move_to_folder`] - Move email to folder
//!
//! # Signal Types
//!
//! The [`MailSignal`] enum defines real-time notifications:
//! - `EmailReceived` - New email arrived
//! - `DeliveryConfirmed` - Email was delivered
//! - `ReadReceiptReceived` - Recipient read your email
//! - `EmailStateChanged` - Email status updated
//! - `ThreadActivity` - New activity in thread
//! - `DraftSaved` - Draft auto-saved
//! - `TypingIndicator` - Someone is composing
//!
//! # Example Usage
//!
//! ```ignore
//! // Send an encrypted email
//! let result = send_email(SendEmailInput {
//!     recipients: vec![recipient_agent],
//!     encrypted_subject: encrypted_subject_bytes,
//!     encrypted_body: encrypted_body_bytes,
//!     // ... other fields
//! })?;
//!
//! // Query inbox
//! let emails = get_inbox(EmailQuery {
//!     is_read: Some(false),
//!     limit: Some(50),
//!     ..Default::default()
//! })?;
//! ```

#[cfg(test)]
mod tests;

use hdk::prelude::*;
use std::collections::HashSet;
use mail_messages_integrity::*;

/// Signal types for real-time notifications
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "data")]
pub enum MailSignal {
    /// New email received
    EmailReceived {
        email_hash: ActionHash,
        sender: AgentPubKey,
        encrypted_subject: Vec<u8>,
        timestamp: Timestamp,
        priority: EmailPriority,
    },
    /// Delivery confirmation
    DeliveryConfirmed {
        email_hash: ActionHash,
        recipient: AgentPubKey,
        delivered_at: Timestamp,
    },
    /// Read receipt received
    ReadReceiptReceived {
        email_hash: ActionHash,
        reader: AgentPubKey,
        read_at: Timestamp,
    },
    /// Email state changed (starred, archived, etc.)
    EmailStateChanged {
        email_hash: ActionHash,
        state: EmailStateUpdate,
    },
    /// New thread activity
    ThreadActivity {
        thread_id: String,
        email_hash: ActionHash,
        sender: AgentPubKey,
    },
    /// Draft auto-saved
    DraftSaved {
        draft_hash: ActionHash,
    },
    /// Scheduled email sent
    ScheduledEmailSent {
        email_hash: ActionHash,
        recipients: Vec<AgentPubKey>,
    },
    /// Typing indicator (for real-time compose)
    TypingIndicator {
        sender: AgentPubKey,
        thread_id: Option<String>,
    },
}

/// Email state update payload
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmailStateUpdate {
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub is_archived: Option<bool>,
    pub is_trashed: Option<bool>,
    pub folders: Option<Vec<ActionHash>>,
}

/// Input for sending an email
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SendEmailInput {
    pub recipients: Vec<AgentPubKey>,
    pub cc: Vec<AgentPubKey>,
    pub bcc: Vec<AgentPubKey>,
    pub encrypted_subject: Vec<u8>,
    pub encrypted_body: Vec<u8>,
    pub encrypted_attachments: Vec<u8>,
    pub ephemeral_pubkey: Vec<u8>,
    pub nonce: [u8; 24],
    pub signature: Vec<u8>,
    pub crypto_suite: CryptoSuite,
    pub message_id: String,
    pub in_reply_to: Option<String>,
    pub references: Vec<String>,
    pub priority: EmailPriority,
    pub read_receipt_requested: bool,
    pub expires_at: Option<Timestamp>,
}

/// Output from sending an email
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SendEmailOutput {
    pub email_hash: ActionHash,
    pub delivered_to: Vec<AgentPubKey>,
    pub failed_deliveries: Vec<(AgentPubKey, String)>,
}

/// Query input for fetching emails
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmailQuery {
    pub folder: Option<ActionHash>,
    pub is_read: Option<bool>,
    pub is_starred: Option<bool>,
    pub is_archived: Option<bool>,
    pub since: Option<Timestamp>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

/// Email list item (minimal info for lists)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmailListItem {
    pub hash: ActionHash,
    pub sender: AgentPubKey,
    pub encrypted_subject: Vec<u8>,
    pub timestamp: Timestamp,
    pub priority: EmailPriority,
    pub is_read: bool,
    pub is_starred: bool,
    pub has_attachments: bool,
    pub thread_id: Option<String>,
}

// ==================== SEND EMAIL ====================

/// Send an email to recipients via P2P
#[hdk_extern]
pub fn send_email(input: SendEmailInput) -> ExternResult<SendEmailOutput> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Combine all recipients (we'll send to each)
    let all_recipients: Vec<AgentPubKey> = input
        .recipients
        .iter()
        .chain(input.cc.iter())
        .chain(input.bcc.iter())
        .cloned()
        .collect();

    let mut delivered_to = Vec::new();
    let mut failed_deliveries = Vec::new();

    // Trust-gated delivery: check sender trust via cross-zome call
    // Low-trust senders may have emails quarantined by recipient
    let sender_trust: Option<(f64, f64)> = call(
        CallTargetCell::Local,
        ZomeName::from("mail_trust_coordinator"),
        "get_sender_trust_for_delivery".into(),
        None,
        my_agent.clone(),
    )
    .ok()
    .and_then(|response| match response {
        ZomeCallResponse::Ok(bytes) => bytes.decode().ok(),
        _ => None,
    });

    let _trust_score = sender_trust.map(|(s, _)| s).unwrap_or(0.3);
    let _trust_confidence = sender_trust.map(|(_, c)| c).unwrap_or(0.0);

    // Create email entry for each recipient (they need their own copy to decrypt)
    for recipient in &all_recipients {
        let email = EncryptedEmail {
            sender: my_agent.clone(),
            recipient: recipient.clone(),
            encrypted_subject: input.encrypted_subject.clone(),
            encrypted_body: input.encrypted_body.clone(),
            encrypted_attachments: input.encrypted_attachments.clone(),
            ephemeral_pubkey: input.ephemeral_pubkey.clone(),
            nonce: input.nonce,
            signature: input.signature.clone(),
            crypto_suite: input.crypto_suite.clone(),
            message_id: input.message_id.clone(),
            in_reply_to: input.in_reply_to.clone(),
            references: input.references.clone(),
            timestamp: now,
            priority: input.priority.clone(),
            read_receipt_requested: input.read_receipt_requested,
            expires_at: input.expires_at,
        };

        // Create the entry
        let email_hash = create_entry(EntryTypes::EncryptedEmail(email.clone()))?;

        // Link to sender's sent folder
        create_link(
            my_agent.clone(),
            email_hash.clone(),
            LinkTypes::AgentToSent,
            LinkTag::new("sent"),
        )?;

        // Link to recipient's inbox
        create_link(
            recipient.clone(),
            email_hash.clone(),
            LinkTypes::AgentToInbox,
            LinkTag::new("inbox"),
        )?;

        // Handle threading
        if let Some(ref reply_to) = input.in_reply_to {
            // Find or create thread
            let thread_id = input
                .references
                .first()
                .cloned()
                .unwrap_or_else(|| reply_to.clone());

            let thread_hash = get_or_create_thread(&thread_id, &input.encrypted_subject)?;
            create_link(
                thread_hash,
                email_hash.clone(),
                LinkTypes::ThreadToEmails,
                LinkTag::new(input.message_id.clone()),
            )?;
        }

        // Send real-time signal to recipient
        let signal = MailSignal::EmailReceived {
            email_hash: email_hash.clone(),
            sender: my_agent.clone(),
            encrypted_subject: input.encrypted_subject.clone(),
            timestamp: now,
            priority: input.priority.clone(),
        };

        let encoded = match ExternIO::encode(signal) {
            Ok(e) => e,
            Err(e) => {
                failed_deliveries.push((recipient.clone(), format!("Encode failed: {:?}", e)));
                continue;
            }
        };
        match send_remote_signal(encoded, vec![recipient.clone()]) {
            Ok(_) => {
                delivered_to.push(recipient.clone());
            }
            Err(e) => {
                // Signal failed but email is still on DHT - they'll see it when online
                failed_deliveries.push((recipient.clone(), format!("Signal failed: {}", e)));
                // Still count as "delivered" since it's on DHT
                delivered_to.push(recipient.clone());
            }
        }
    }

    // Return the hash of the first email (they're all essentially the same content)
    let email_hash = get_links(LinkQuery::try_new(my_agent.clone(), LinkTypes::AgentToSent)?, GetStrategy::default())?
    .into_iter()
    .last()
    .map(|l| ActionHash::try_from(l.target).ok())
    .flatten()
    .ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to get email hash".to_string()
    )))?;

    Ok(SendEmailOutput {
        email_hash,
        delivered_to,
        failed_deliveries,
    })
}

/// Get or create a thread entry
fn get_or_create_thread(
    thread_id: &str,
    encrypted_subject: &[u8],
) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Try to find existing thread
    let thread_anchor = anchor_for_thread(thread_id)?;
    let existing = get_links(LinkQuery::try_new(thread_anchor.clone(), LinkTypes::ThreadToEmails)?, GetStrategy::default())?;

    if let Some(link) = existing.first() {
        // Thread exists, return its action hash from the link target
        if let Some(hash) = link.target.clone().into_action_hash() {
            return Ok(hash);
        }
    }

    // Create new thread
    let thread = EmailThread {
        thread_id: thread_id.to_string(),
        encrypted_subject: encrypted_subject.to_vec(),
        participants: vec![my_agent.clone()],
        message_count: 1,
        last_activity: sys_time()?,
    };

    let thread_hash = create_entry(EntryTypes::EmailThread(thread))?;

    // Link agent to thread
    create_link(
        my_agent,
        thread_hash.clone(),
        LinkTypes::AgentToThreads,
        LinkTag::new(thread_id),
    )?;

    Ok(thread_hash)
}

fn anchor_for_thread(thread_id: &str) -> ExternResult<EntryHash> {
    // Create a deterministic anchor for thread lookup using Path
    let path = Path::from(format!("email_thread:{}", thread_id));
    path.path_entry_hash()
}

// ==================== RECEIVE & QUERY ====================

/// Get inbox emails
#[hdk_extern]
pub fn get_inbox(query: EmailQuery) -> ExternResult<Vec<EmailListItem>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(my_agent.clone(), LinkTypes::AgentToInbox)?, GetStrategy::default())?;

    let mut items: Vec<EmailListItem> = Vec::new();

    for link in links {
        let email_hash = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
        })?;

        if let Some(record) = get(email_hash.clone(), GetOptions::default())? {
            if let Some(email) = record
                .entry()
                .to_app_option::<EncryptedEmail>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                // Get state for this email
                let state = get_email_state_internal(&email_hash, &my_agent)?;

                // Apply filters
                if let Some(is_read) = query.is_read {
                    if state.as_ref().map(|s| s.is_read) != Some(is_read) {
                        continue;
                    }
                }
                if let Some(is_starred) = query.is_starred {
                    if state.as_ref().map(|s| s.is_starred) != Some(is_starred) {
                        continue;
                    }
                }
                if let Some(is_archived) = query.is_archived {
                    if state.as_ref().map(|s| s.is_archived) != Some(is_archived) {
                        continue;
                    }
                }
                if let Some(since) = query.since {
                    if email.timestamp < since {
                        continue;
                    }
                }

                items.push(EmailListItem {
                    hash: email_hash,
                    sender: email.sender,
                    encrypted_subject: email.encrypted_subject,
                    timestamp: email.timestamp,
                    priority: email.priority,
                    is_read: state.as_ref().map(|s| s.is_read).unwrap_or(false),
                    is_starred: state.as_ref().map(|s| s.is_starred).unwrap_or(false),
                    has_attachments: !email.encrypted_attachments.is_empty(),
                    thread_id: email.references.first().cloned().or(email.in_reply_to.clone()),
                });
            }
        }
    }

    // Sort by timestamp descending
    items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    // Apply pagination
    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(50);
    let items: Vec<_> = items.into_iter().skip(offset).take(limit).collect();

    Ok(items)
}

/// Get sent emails
#[hdk_extern]
pub fn get_sent(query: EmailQuery) -> ExternResult<Vec<EmailListItem>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(my_agent.clone(), LinkTypes::AgentToSent)?, GetStrategy::default())?;

    let mut items: Vec<EmailListItem> = Vec::new();

    for link in links {
        let email_hash = ActionHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".to_string()))
        })?;

        if let Some(record) = get(email_hash.clone(), GetOptions::default())? {
            if let Some(email) = record
                .entry()
                .to_app_option::<EncryptedEmail>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                items.push(EmailListItem {
                    hash: email_hash,
                    sender: email.sender,
                    encrypted_subject: email.encrypted_subject,
                    timestamp: email.timestamp,
                    priority: email.priority,
                    is_read: true, // Sent emails are always "read"
                    is_starred: false,
                    has_attachments: !email.encrypted_attachments.is_empty(),
                    thread_id: email.references.first().cloned(),
                });
            }
        }
    }

    items.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    let offset = query.offset.unwrap_or(0);
    let limit = query.limit.unwrap_or(50);
    Ok(items.into_iter().skip(offset).take(limit).collect())
}

/// Get full email by hash
#[hdk_extern]
pub fn get_email(hash: ActionHash) -> ExternResult<Option<EncryptedEmail>> {
    if let Some(record) = get(hash, GetOptions::default())? {
        let email = record
            .entry()
            .to_app_option::<EncryptedEmail>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
        return Ok(email);
    }
    Ok(None)
}

// ==================== EMAIL STATE ====================

fn get_email_state_internal(
    email_hash: &ActionHash,
    owner: &AgentPubKey,
) -> ExternResult<Option<EmailState>> {
    let links = get_links(LinkQuery::try_new(email_hash.clone(), LinkTypes::EmailToState)?, GetStrategy::default())?;

    for link in links {
        if let Some(record) = get(
            ActionHash::try_from(link.target).map_err(|_| {
                wasm_error!(WasmErrorInner::Guest("Invalid target".to_string()))
            })?,
            GetOptions::default(),
        )? {
            if let Some(state) = record
                .entry()
                .to_app_option::<EmailState>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if &state.owner == owner {
                    return Ok(Some(state));
                }
            }
        }
    }

    Ok(None)
}

/// Update email state (read, starred, etc.)
#[hdk_extern]
pub fn update_email_state(input: (ActionHash, EmailStateUpdate)) -> ExternResult<ActionHash> {
    let (email_hash, update) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Get existing state or create new
    let existing = get_email_state_internal(&email_hash, &my_agent)?;

    let new_state = if let Some(mut state) = existing {
        // Update fields
        if let Some(is_read) = update.is_read {
            state.is_read = is_read;
        }
        if let Some(is_starred) = update.is_starred {
            state.is_starred = is_starred;
        }
        if let Some(is_archived) = update.is_archived {
            state.is_archived = is_archived;
        }
        if let Some(is_trashed) = update.is_trashed {
            state.is_trashed = is_trashed;
            if is_trashed {
                state.trashed_at = Some(sys_time()?);
            }
        }
        if let Some(ref folders) = update.folders {
            state.folders = folders.clone();
        }
        state
    } else {
        let folders = update.folders.clone().unwrap_or_default();
        let is_trashed = update.is_trashed.unwrap_or(false);
        EmailState {
            email_hash: email_hash.clone(),
            owner: my_agent.clone(),
            is_read: update.is_read.unwrap_or(false),
            is_starred: update.is_starred.unwrap_or(false),
            folders,
            encrypted_labels: Vec::new(),
            snoozed_until: None,
            is_archived: update.is_archived.unwrap_or(false),
            is_trashed,
            trashed_at: if is_trashed {
                Some(sys_time()?)
            } else {
                None
            },
        }
    };

    let state_hash = create_entry(EntryTypes::EmailState(new_state.clone()))?;

    // Link email to state
    create_link(
        email_hash.clone(),
        state_hash.clone(),
        LinkTypes::EmailToState,
        LinkTag::new(my_agent.to_string()),
    )?;

    // Emit signal for real-time UI updates
    emit_signal(MailSignal::EmailStateChanged {
        email_hash,
        state: update,
    })?;

    Ok(state_hash)
}

/// Mark email as read and optionally send read receipt
#[hdk_extern]
pub fn mark_as_read(input: (ActionHash, bool)) -> ExternResult<Option<ActionHash>> {
    let (email_hash, send_receipt) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Update state
    update_email_state((
        email_hash.clone(),
        EmailStateUpdate {
            is_read: Some(true),
            is_starred: None,
            is_archived: None,
            is_trashed: None,
            folders: None,
        },
    ))?;

    // Send read receipt if requested and email has that flag
    if send_receipt {
        if let Some(email) = get_email(email_hash.clone())? {
            if email.read_receipt_requested && email.sender != my_agent {
                let read_at = sys_time()?;
                let signing_content = receipt_signing_content(&email_hash, &my_agent, &read_at);
                let receipt = ReadReceipt {
                    email_hash: email_hash.clone(),
                    reader: my_agent.clone(),
                    read_at,
                    signature: sign_raw(
                        my_agent.clone(),
                        signing_content,
                    )?.0.to_vec(),
                };

                let receipt_hash = create_entry(EntryTypes::ReadReceipt(receipt.clone()))?;

                // Link to email
                create_link(
                    email_hash,
                    receipt_hash.clone(),
                    LinkTypes::EmailToReadReceipts,
                    LinkTag::new("receipt"),
                )?;

                // Signal sender
                let signal = MailSignal::ReadReceiptReceived {
                    email_hash: receipt.email_hash,
                    reader: receipt.reader,
                    read_at: receipt.read_at,
                };
                let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
                let _ = send_remote_signal(encoded, vec![email.sender]);

                return Ok(Some(receipt_hash));
            }
        }
    }

    Ok(None)
}

// ==================== DRAFTS ====================

/// Save a draft
#[hdk_extern]
pub fn save_draft(input: EmailDraft) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let mut draft = input;
    draft.owner = my_agent.clone();
    draft.updated_at = sys_time()?;

    let draft_hash = create_entry(EntryTypes::EmailDraft(draft))?;

    create_link(
        my_agent,
        draft_hash.clone(),
        LinkTypes::AgentToDrafts,
        LinkTag::new("draft"),
    )?;

    emit_signal(MailSignal::DraftSaved {
        draft_hash: draft_hash.clone(),
    })?;

    Ok(draft_hash)
}

/// Get drafts
#[hdk_extern]
pub fn get_drafts(_: ()) -> ExternResult<Vec<(ActionHash, EmailDraft)>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(my_agent, LinkTypes::AgentToDrafts)?, GetStrategy::default())?;

    let mut drafts = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(draft) = record
                .entry()
                .to_app_option::<EmailDraft>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                drafts.push((hash, draft));
            }
        }
    }

    drafts.sort_by(|a, b| b.1.updated_at.cmp(&a.1.updated_at));

    Ok(drafts)
}

/// Delete a draft
#[hdk_extern]
pub fn delete_draft(hash: ActionHash) -> ExternResult<()> {
    delete_entry(hash)?;
    Ok(())
}

// ==================== FOLDERS ====================

/// Create a folder
#[hdk_extern]
pub fn create_folder(input: (Vec<u8>, Option<Vec<u8>>, i32)) -> ExternResult<ActionHash> {
    let (encrypted_name, metadata, sort_order) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;

    let folder = EmailFolder {
        owner: my_agent.clone(),
        encrypted_name,
        metadata,
        is_system: false,
        sort_order,
    };

    let folder_hash = create_entry(EntryTypes::EmailFolder(folder))?;

    create_link(
        my_agent,
        folder_hash.clone(),
        LinkTypes::AgentToFolders,
        LinkTag::new("folder"),
    )?;

    Ok(folder_hash)
}

/// Get user's folders
#[hdk_extern]
pub fn get_folders(_: ()) -> ExternResult<Vec<(ActionHash, EmailFolder)>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(my_agent, LinkTypes::AgentToFolders)?, GetStrategy::default())?;

    let mut folders = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(folder) = record
                .entry()
                .to_app_option::<EmailFolder>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                folders.push((hash, folder));
            }
        }
    }

    folders.sort_by(|a, b| a.1.sort_order.cmp(&b.1.sort_order));

    Ok(folders)
}

/// Move email to folder
#[hdk_extern]
pub fn move_to_folder(input: (ActionHash, ActionHash)) -> ExternResult<()> {
    let (email_hash, folder_hash) = input;

    create_link(
        folder_hash,
        email_hash,
        LinkTypes::FolderToEmails,
        LinkTag::new("email"),
    )?;

    Ok(())
}

// ==================== ATTACHMENTS ====================

/// Add attachment to email
#[hdk_extern]
pub fn add_attachment(input: EncryptedAttachment) -> ExternResult<ActionHash> {
    let attachment_hash = create_entry(EntryTypes::EncryptedAttachment(input.clone()))?;

    create_link(
        input.email_hash,
        attachment_hash.clone(),
        LinkTypes::EmailToAttachments,
        LinkTag::new(format!("chunk:{}", input.chunk_index)),
    )?;

    Ok(attachment_hash)
}

/// Get attachments for email
#[hdk_extern]
pub fn get_attachments(email_hash: ActionHash) -> ExternResult<Vec<EncryptedAttachment>> {
    let links = get_links(LinkQuery::try_new(email_hash, LinkTypes::EmailToAttachments)?, GetStrategy::default())?;

    let mut attachments = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(attachment) = record
                .entry()
                .to_app_option::<EncryptedAttachment>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                attachments.push(attachment);
            }
        }
    }

    // Sort by chunk index
    attachments.sort_by(|a, b| a.chunk_index.cmp(&b.chunk_index));

    Ok(attachments)
}

// ==================== SIGNALS ====================

/// Signal handler for incoming signals
#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    let mail_signal: MailSignal = signal.decode().map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to decode signal: {}",
            e
        )))
    })?;

    // Forward to UI
    emit_signal(mail_signal)?;

    Ok(())
}

/// Send typing indicator
#[hdk_extern]
pub fn send_typing_indicator(input: (Vec<AgentPubKey>, Option<String>)) -> ExternResult<()> {
    let (recipients, thread_id) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;

    let signal = MailSignal::TypingIndicator {
        sender: my_agent,
        thread_id,
    };

    let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
    send_remote_signal(encoded, recipients)?;

    Ok(())
}

// ==================== INIT ====================

/// Initialize zome - create system folders
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Create system folders: Inbox, Sent, Drafts, Trash, Spam, Archive
    let system_folders = vec![
        ("Inbox", 0),
        ("Sent", 1),
        ("Drafts", 2),
        ("Starred", 3),
        ("Archive", 4),
        ("Spam", 5),
        ("Trash", 6),
    ];

    for (name, order) in system_folders {
        let folder = EmailFolder {
            owner: my_agent.clone(),
            encrypted_name: name.as_bytes().to_vec(), // System folders aren't encrypted
            metadata: None,
            is_system: true,
            sort_order: order,
        };

        let folder_hash = create_entry(EntryTypes::EmailFolder(folder))?;

        create_link(
            my_agent.clone(),
            folder_hash,
            LinkTypes::AgentToFolders,
            LinkTag::new(format!("system:{}", name.to_lowercase())),
        )?;
    }

    // Set up capability grants for receiving signals
    let functions = GrantedFunctions::Listed(HashSet::from([
        (zome_info()?.name, "recv_remote_signal".into()),
    ]));

    create_cap_grant(CapGrantEntry {
        tag: "recv_signals".to_string(),
        access: CapAccess::Unrestricted,
        functions,
    })?;

    Ok(InitCallbackResult::Pass)
}
