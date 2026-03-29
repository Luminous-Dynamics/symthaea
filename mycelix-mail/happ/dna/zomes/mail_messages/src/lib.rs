// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use mycelix_mail_integrity::*;

/// Input required to register the caller's DID inside the DNA
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterDidInput {
    pub did: String,
}

/// Register the caller's DID so other agents can resolve their AgentPubKey.
#[hdk_extern]
pub fn register_my_did(input: RegisterDidInput) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_initial_pubkey;
    let did = input.did.trim();
    if did.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID cannot be empty".into()
        )));
    }

    let path = did_path(did);
    path.ensure()?;
    let path_hash = path.path_entry_hash()?;

    let existing = get_links(
        GetLinksInputBuilder::try_new(path_hash.clone(), LinkTypes::DidBindingLink)?.build(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "DID {} already registered",
            did
        ))));
    }

    let binding = DidBinding {
        did: did.to_string(),
        agent_pub_key: agent.clone(),
    };
    let binding_hash = create_entry(EntryTypes::DidBinding(binding))?;

    create_link(
        path_hash,
        binding_hash.clone(),
        LinkTypes::DidBindingLink,
        (),
    )?;

    Ok(binding_hash)
}

/// Send a mail message
/// Creates the message entry and links it to both sender's outbox and recipient's inbox
#[hdk_extern]
pub fn send_message(message: MailMessage) -> ExternResult<ActionHash> {
    debug!(
        "Sending message from {} to {}",
        message.from_did, message.to_did
    );

    // Get the agent info to verify sender
    let agent_info = agent_info()?;

    // Create the message entry on the sender's source chain
    let message_hash = create_entry(EntryTypes::MailMessage(message.clone()))?;

    // Link to sender's outbox (for "sent mail" folder)
    create_link(
        agent_info.agent_initial_pubkey.clone(),
        message_hash.clone(),
        LinkTypes::FromOutbox,
        (),
    )?;

    // Resolve DID to AgentPubKey stored on the DHT
    let recipient_pubkey = resolve_did_to_pubkey(&message.to_did)?;

    create_link(
        recipient_pubkey,
        message_hash.clone(),
        LinkTypes::ToInbox,
        (),
    )?;

    // If this is a reply, link to parent message
    if let Some(thread_id) = &message.thread_id {
        if let Some(parent_hash) = parse_thread_id(thread_id)? {
            create_link(
                parent_hash,
                message_hash.clone(),
                LinkTypes::ThreadReply,
                (),
            )?;
        }
    }

    debug!("Message sent successfully: {:?}", message_hash);
    Ok(message_hash)
}

/// Get all messages in the inbox
/// Returns all messages linked to the current agent's inbox
#[hdk_extern]
pub fn get_inbox(_: ()) -> ExternResult<Vec<MailMessage>> {
    let agent_info = agent_info()?;

    // Get all links pointing to this agent's inbox
    let links = get_links(
        GetLinksInputBuilder::try_new(agent_info.agent_initial_pubkey, LinkTypes::ToInbox)?.build(),
    )?;

    let mut messages = Vec::new();

    // Fetch each message
    for link in links {
        if let Some(message) = get_message_from_link(link)? {
            messages.push(message);
        }
    }

    // Sort by timestamp (newest first)
    messages.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(messages)
}

/// Get sent messages from outbox
#[hdk_extern]
pub fn get_outbox(_: ()) -> ExternResult<Vec<MailMessage>> {
    let agent_info = agent_info()?;

    let links = get_links(
        GetLinksInputBuilder::try_new(agent_info.agent_initial_pubkey, LinkTypes::FromOutbox)?
            .build(),
    )?;

    let mut messages = Vec::new();

    for link in links {
        if let Some(message) = get_message_from_link(link)? {
            messages.push(message);
        }
    }

    // Sort by timestamp (newest first)
    messages.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(messages)
}

/// Get messages in a specific thread
#[hdk_extern]
pub fn get_thread(parent_hash: ActionHash) -> ExternResult<Vec<MailMessage>> {
    let links =
        get_links(GetLinksInputBuilder::try_new(parent_hash, LinkTypes::ThreadReply)?.build())?;

    let mut messages = Vec::new();

    for link in links {
        if let Some(message) = get_message_from_link(link)? {
            messages.push(message);
        }
    }

    // Sort by timestamp (oldest first for threads)
    messages.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    Ok(messages)
}

/// Get a single message by its hash
#[hdk_extern]
pub fn get_message(message_hash: ActionHash) -> ExternResult<Option<MailMessage>> {
    let record = get(message_hash, GetOptions::default())?;

    match record {
        Some(record) => {
            let message: MailMessage = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Deserialization error: {:?}",
                        e
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))?;
            Ok(Some(message))
        }
        None => Ok(None),
    }
}

/// Delete a message (creates a delete action)
#[hdk_extern]
pub fn delete_message(message_hash: ActionHash) -> ExternResult<ActionHash> {
    delete_entry(message_hash)
}

/// Resolve a DID to a DidBinding (public wrapper for external callers)
/// Returns the DidBinding entry containing both DID and AgentPubKey
#[hdk_extern]
pub fn resolve_did(did: String) -> ExternResult<Option<DidBinding>> {
    let path = did_path(&did);
    let path_hash = path.path_entry_hash()?;

    let links =
        get_links(GetLinksInputBuilder::try_new(path_hash, LinkTypes::DidBindingLink)?.build())?;

    if let Some(link) = links.first() {
        let hash_any_dht: AnyDhtHash =
            ActionHash::from_raw_39(link.target.get_raw_39().to_vec()).into();
        if let Some(record) = get(hash_any_dht, GetOptions::default())? {
            let binding: DidBinding = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Deserialization error: {:?}",
                        e
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid DID binding entry".into()
                )))?;

            return Ok(Some(binding));
        }
    }

    Ok(None)
}

/// List all registered DIDs (for admin/debug purposes)
#[hdk_extern]
pub fn list_registered_dids(_: ()) -> ExternResult<Vec<String>> {
    let path = Path::from("did_index");
    let children = path.children()?;

    let dids: Vec<String> = children
        .into_iter()
        .filter_map(|link| {
            // Extract DID from path component
            let path_str = String::from_utf8(link.tag.into_inner()).ok()?;
            Some(path_str)
        })
        .collect();

    Ok(dids)
}

// === Helper Functions ===

/// Helper function to get a message from a link
fn get_message_from_link(link: Link) -> ExternResult<Option<MailMessage>> {
    // The link target is an AnyLinkableHash which we convert to ActionHash then to AnyDhtHash
    let hash_any_dht: AnyDhtHash =
        ActionHash::from_raw_39(link.target.get_raw_39().to_vec()).into();
    let record = get(hash_any_dht, GetOptions::default())?;

    match record {
        Some(record) => {
            let message: MailMessage = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Deserialization error: {:?}",
                        e
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))?;
            Ok(Some(message))
        }
        None => Ok(None),
    }
}

/// Resolve a DID to an AgentPubKey
/// In production, this would query the DHT for DID -> PubKey mappings
/// For MVP, we'll use a simplified mock
fn resolve_did_to_pubkey(did: &str) -> ExternResult<AgentPubKey> {
    let path = did_path(did);
    let path_hash = path.path_entry_hash()?;

    let links =
        get_links(GetLinksInputBuilder::try_new(path_hash, LinkTypes::DidBindingLink)?.build())?;

    if let Some(link) = links.first() {
        let hash_any_dht: AnyDhtHash =
            ActionHash::from_raw_39(link.target.get_raw_39().to_vec()).into();
        if let Some(record) = get(hash_any_dht, GetOptions::default())? {
            let binding: DidBinding = record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Deserialization error: {:?}",
                        e
                    )))
                })?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid DID binding entry".into()
                )))?;

            return Ok(binding.agent_pub_key);
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "DID {} is not registered in this DNA",
        did
    ))))
}

/// Parse a thread ID into an ActionHash
fn parse_thread_id(thread_id: &str) -> ExternResult<Option<ActionHash>> {
    // Thread ID format: "msg_<base64_encoded_hash>"
    if let Some(_hash_str) = thread_id.strip_prefix("msg_") {
        // Decode base64 and convert to ActionHash
        // For MVP, we'll return None
        // In production, implement proper parsing
        debug!("Parsing thread ID: {}", thread_id);
        Ok(None)
    } else {
        Ok(None)
    }
}

fn did_path(did: &str) -> Path {
    Path::from(format!("did_index.{}", did))
}
