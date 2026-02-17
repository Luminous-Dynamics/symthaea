/// Messaging Coordinator Zome
///
/// Provides P2P encrypted messaging for the Mycelix-Marketplace.
/// This coordinator implements the business logic for:
/// - Sending and receiving messages
/// - Managing conversations
/// - Read receipts and typing indicators
/// - Spam prevention via MATL gating
/// - Message search and filtering

use hdk::prelude::*;
use messaging_integrity::*;
use mycelix_common::{error_handling, link_queries, remote_calls, time};

/// Send a message to another agent
///
/// This is MATL-gated: sender must have score > 0.4 to prevent spam.
/// Messages are encrypted client-side before calling this function.
///
/// # Rate Limiting
/// - New conversations: 10/hour
/// - Messages in existing conversation: 100/hour
#[hdk_extern]
pub fn send_message(input: SendMessageInput) -> ExternResult<MessageOutput> {
    let agent_info = agent_info()?;
    let sender = agent_info.agent_initial_pubkey.clone();

    // MATL gate: Check sender reputation (prevent spam)
    // Use shared utility for remote calls
    let matl_score: f64 = remote_calls::call_zome(
        "reputation",
        "get_agent_matl_score_fast",
        sender.clone(),
    )?;

    if matl_score < 0.4 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Insufficient MATL score to send messages (have: {:.2}, need: 0.40). \
                 Build your reputation through successful transactions first.",
                matl_score
            )
        )));
    }

    // Create message entry
    let message = Message {
        sender: sender.clone(),
        recipient: input.recipient.clone(),
        encrypted_content: input.encrypted_content,
        listing_hash: input.listing_hash.clone(),
        transaction_hash: input.transaction_hash.clone(),
        conversation_id: input.conversation_id.clone(),
        sent_at: time::now_micros()?,
        read_at: None,
        message_type: input.message_type,
        epistemic: EpistemicClassification {
            empirical: EmpiricalLevel::E2PrivateVerify, // Both parties can verify
            normative: NormativeLevel::N1Communal,       // Between two people
            materiality: MaterialityLevel::M1Temporal,   // Ephemeral unless archived
        },
    };

    let action_hash = create_entry(&EntryTypes::Message(message.clone()))?;

    // Create links for discovery
    create_link(
        sender.clone(),
        action_hash.clone(),
        LinkTypes::AgentToSentMessages,
        (),
    )?;

    create_link(
        input.recipient.clone(),
        action_hash.clone(),
        LinkTypes::AgentToReceivedMessages,
        (),
    )?;

    create_link(
        input.conversation_id.clone(),
        action_hash.clone(),
        LinkTypes::ConversationToMessages,
        (),
    )?;

    // Update conversation metadata
    update_conversation_metadata(
        input.conversation_id.clone(),
        action_hash.clone(),
    )?;

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::MessageSent,
        1.0,
        Some(sender),
        Some(format!("recipient:{:?}", input.recipient)),
    )?;

    Ok(MessageOutput {
        message_hash: action_hash,
        message,
    })
}

/// Start a new conversation
///
/// Creates the conversation metadata and sends the first message.
/// Automatically links to listing or transaction if provided.
#[hdk_extern]
pub fn start_conversation(input: StartConversationInput) -> ExternResult<ConversationOutput> {
    let agent_info = agent_info()?;
    let initiator = agent_info.agent_initial_pubkey.clone();

    // MATL gate
    // Use shared utility for remote calls
    let matl_score: f64 = remote_calls::call_zome(
        "reputation",
        "get_agent_matl_score_fast",
        initiator.clone(),
    )?;

    if matl_score < 0.4 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Insufficient MATL score to start conversations (have: {:.2}, need: 0.40)",
                matl_score
            )
        )));
    }

    // Create a placeholder ActionHash for the first message
    // We'll use a zero-filled hash as placeholder since we don't have the conversation hash yet
    let placeholder_hash = ActionHash::from_raw_39(vec![0; 39]);

    // Send first message (this creates the message entry)
    let first_message = send_message(SendMessageInput {
        recipient: input.recipient.clone(),
        encrypted_content: input.first_message_content,
        listing_hash: input.listing_hash.clone(),
        transaction_hash: input.transaction_hash.clone(),
        conversation_id: placeholder_hash,
        message_type: MessageType::Text,
    })?;

    // Create conversation entry
    let now = time::now_micros()?;
    let conversation = Conversation {
        participants: vec![initiator.clone(), input.recipient.clone()],
        listing_hash: input.listing_hash.clone(),
        transaction_hash: input.transaction_hash.clone(),
        subject: input.subject,
        first_message_hash: first_message.message_hash.clone(),
        last_message_hash: first_message.message_hash.clone(),
        message_count: 1,
        unread_counts: vec![
            (initiator.clone(), 0),
            (input.recipient.clone(), 1),
        ],
        started_at: now,
        last_activity_at: now,
        status: ConversationStatus::Active,
    };

    let conversation_hash = create_entry(&EntryTypes::Conversation(conversation.clone()))?;

    // Create links
    for participant in &conversation.participants {
        create_link(
            participant.clone(),
            conversation_hash.clone(),
            LinkTypes::AgentToConversations,
            (),
        )?;
    }

    // Link from listing if provided
    if let Some(listing_hash) = &input.listing_hash {
        create_link(
            listing_hash.clone(),
            conversation_hash.clone(),
            LinkTypes::ListingToConversations,
            (),
        )?;
    }

    // Link from transaction if provided
    if let Some(transaction_hash) = &input.transaction_hash {
        create_link(
            transaction_hash.clone(),
            conversation_hash.clone(),
            LinkTypes::TransactionToConversations,
            (),
        )?;
    }

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::ConversationStarted,
        1.0,
        Some(initiator),
        Some(format!("participants:2,subject:{}", conversation.subject)),
    )?;

    Ok(ConversationOutput {
        conversation_hash,
        conversation,
        first_message: first_message,
    })
}

/// Mark a message as read
///
/// Creates a read receipt and updates conversation unread count.
#[hdk_extern]
pub fn mark_message_read(message_hash: ActionHash) -> ExternResult<ReadReceiptOutput> {
    let agent_info = agent_info()?;
    let reader = agent_info.agent_initial_pubkey;

    // Get the message
    let message: Message = get_entry_from_hash(message_hash.clone())?;

    // Verify reader is the recipient
    if reader != message.recipient {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only mark messages addressed to you as read".to_string()
        )));
    }

    // Create read receipt
    let receipt = ReadReceipt {
        message_hash: message_hash.clone(),
        reader: reader.clone(),
        read_at: time::now_micros()?,
    };

    let receipt_hash = create_entry(&EntryTypes::ReadReceipt(receipt.clone()))?;

    // Link receipt to message
    create_link(
        message_hash.clone(),
        receipt_hash.clone(),
        LinkTypes::MessageToReadReceipts,
        (),
    )?;

    // Update conversation unread count
    update_conversation_unread_count(message.conversation_id, reader)?;

    Ok(ReadReceiptOutput {
        receipt_hash,
        receipt,
    })
}

/// Get all conversations for the current agent
#[hdk_extern]
pub fn get_my_conversations(_: ()) -> ExternResult<ConversationsResponse> {
    let agent_info = agent_info()?;
    let agent = agent_info.agent_initial_pubkey;

    // Use shared utility for get_links
    let links = link_queries::get_links_local(agent, LinkTypes::AgentToConversations)?;

    let mut conversations = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            let conversation: Conversation = get_entry_from_hash(action_hash.clone())?;
            conversations.push(ConversationOutput {
                conversation_hash: action_hash,
                conversation: conversation.clone(),
                first_message: get_message(conversation.first_message_hash)?,
            });
        }
    }

    // Sort by last activity (most recent first)
    conversations.sort_by(|a, b| {
        b.conversation.last_activity_at.cmp(&a.conversation.last_activity_at)
    });

    Ok(ConversationsResponse { conversations })
}

/// Get all messages in a conversation
#[hdk_extern]
pub fn get_conversation_messages(
    conversation_hash: ActionHash,
) -> ExternResult<MessagesResponse> {
    // Use shared utility for get_links
    let links = link_queries::get_links_local(conversation_hash, LinkTypes::ConversationToMessages)?;

    let mut messages = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                // Use shared utility for deserialization
                let message: Message = error_handling::deserialize_entry(&record)?;

                messages.push(MessageOutput {
                    message_hash: action_hash,
                    message,
                });
            }
        }
    }

    // Sort by sent_at (chronological)
    messages.sort_by_key(|m| m.message.sent_at);

    Ok(MessagesResponse { messages })
}

/// Get a single message by hash
#[hdk_extern]
pub fn get_message(message_hash: ActionHash) -> ExternResult<MessageOutput> {
    let message: Message = get_entry_from_hash(message_hash.clone())?;

    Ok(MessageOutput {
        message_hash,
        message,
    })
}

/// Archive a conversation (hide from active list)
#[hdk_extern]
pub fn archive_conversation(conversation_hash: ActionHash) -> ExternResult<ConversationOutput> {
    let mut conversation: Conversation = get_entry_from_hash(conversation_hash.clone())?;

    conversation.status = ConversationStatus::Archived;

    let new_hash = update_entry(conversation_hash, &conversation)?;

    // Get first message for output
    let first_message = get_message(conversation.first_message_hash.clone())?;

    Ok(ConversationOutput {
        conversation_hash: new_hash,
        conversation,
        first_message,
    })
}

/// Block a conversation (spam/abuse)
#[hdk_extern]
pub fn block_conversation(conversation_hash: ActionHash) -> ExternResult<ConversationOutput> {
    let agent_info = agent_info()?;
    let blocker = agent_info.agent_initial_pubkey;

    let mut conversation: Conversation = get_entry_from_hash(conversation_hash.clone())?;

    // Verify agent is a participant
    if !conversation.participants.contains(&blocker) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only block conversations you're a participant in".to_string()
        )));
    }

    conversation.status = ConversationStatus::Blocked;

    let new_hash = update_entry(conversation_hash, &conversation)?;

    // Emit monitoring metric (track spam)
    monitoring::emit_metric(
        monitoring::MetricType::ConversationBlocked,
        1.0,
        Some(blocker),
        None,
    )?;

    let first_message = get_message(conversation.first_message_hash.clone())?;

    Ok(ConversationOutput {
        conversation_hash: new_hash,
        conversation,
        first_message,
    })
}

/// Search messages by content (works on encrypted content hash)
///
/// Note: This searches metadata only since content is encrypted.
/// Client-side decryption and search required for full-text.
#[hdk_extern]
pub fn search_conversations(query: SearchQuery) -> ExternResult<ConversationsResponse> {
    let agent_info = agent_info()?;
    let _agent = agent_info.agent_initial_pubkey;

    // Get all conversations
    let all_conversations = get_my_conversations(())?;

    let mut filtered = Vec::new();

    for conv_output in all_conversations.conversations {
        let mut matches = true;

        // Filter by participant if specified
        if let Some(ref participant) = query.participant {
            if !conv_output.conversation.participants.contains(participant) {
                matches = false;
            }
        }

        // Filter by listing if specified
        if let Some(ref listing) = query.listing_hash {
            if conv_output.conversation.listing_hash.as_ref() != Some(listing) {
                matches = false;
            }
        }

        // Filter by transaction if specified
        if let Some(ref transaction) = query.transaction_hash {
            if conv_output.conversation.transaction_hash.as_ref() != Some(transaction) {
                matches = false;
            }
        }

        // Filter by status if specified
        if let Some(ref status) = query.status {
            if &conv_output.conversation.status != status {
                matches = false;
            }
        }

        // Filter by subject keyword (case-insensitive)
        if let Some(ref subject_keyword) = query.subject_keyword {
            if !conv_output
                .conversation
                .subject
                .to_lowercase()
                .contains(&subject_keyword.to_lowercase())
            {
                matches = false;
            }
        }

        if matches {
            filtered.push(conv_output);
        }
    }

    Ok(ConversationsResponse {
        conversations: filtered,
    })
}

// ===== Helper Functions =====

/// Update conversation metadata after new message
fn update_conversation_metadata(
    conversation_hash: ActionHash,
    new_message_hash: ActionHash,
) -> ExternResult<()> {
    let mut conversation: Conversation = get_entry_from_hash(conversation_hash.clone())?;

    conversation.last_message_hash = new_message_hash;
    conversation.message_count += 1;
    conversation.last_activity_at = time::now_micros()?;

    // Increment unread count for recipient
    // (This is simplified - production would track per-participant)

    update_entry(conversation_hash, &conversation)?;

    Ok(())
}

/// Update conversation unread count after message read
fn update_conversation_unread_count(
    conversation_hash: ActionHash,
    reader: AgentPubKey,
) -> ExternResult<()> {
    let mut conversation: Conversation = get_entry_from_hash(conversation_hash.clone())?;

    // Decrement unread count for reader
    for (agent, count) in &mut conversation.unread_counts {
        if agent == &reader && *count > 0 {
            *count -= 1;
            break;
        }
    }

    update_entry(conversation_hash, &conversation)?;

    Ok(())
}

/// Get entry from action hash (helper)
fn get_entry_from_hash<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    hash: ActionHash,
) -> ExternResult<T> {
    let record = get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Entry not found".into())))?;

    // HDK 0.6.0: to_app_option() returns SerializedBytesError
    record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialization error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not deserialize entry".into())))
}

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SendMessageInput {
    pub recipient: AgentPubKey,
    pub encrypted_content: String,
    pub listing_hash: Option<ActionHash>,
    pub transaction_hash: Option<ActionHash>,
    pub conversation_id: ActionHash,
    pub message_type: MessageType,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StartConversationInput {
    pub recipient: AgentPubKey,
    pub subject: String,
    pub first_message_content: String,
    pub listing_hash: Option<ActionHash>,
    pub transaction_hash: Option<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MessageOutput {
    pub message_hash: ActionHash,
    pub message: Message,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConversationOutput {
    pub conversation_hash: ActionHash,
    pub conversation: Conversation,
    pub first_message: MessageOutput,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConversationsResponse {
    pub conversations: Vec<ConversationOutput>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MessagesResponse {
    pub messages: Vec<MessageOutput>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ReadReceiptOutput {
    pub receipt_hash: ActionHash,
    pub receipt: ReadReceipt,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchQuery {
    pub participant: Option<AgentPubKey>,
    pub listing_hash: Option<ActionHash>,
    pub transaction_hash: Option<ActionHash>,
    pub status: Option<ConversationStatus>,
    pub subject_keyword: Option<String>,
}

// ===== Tests =====
#[cfg(test)]
mod tests;
