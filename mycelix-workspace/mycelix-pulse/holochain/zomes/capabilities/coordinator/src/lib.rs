// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(unsafe_code)]
//! Capabilities Coordinator Zome
//!
//! Manages fine-grained access control, shared mailboxes, and delegation.

use hdk::prelude::*;
use std::collections::{BTreeSet, HashSet};
use mail_capabilities_integrity::*;

/// Signal types for capability events
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "data")]
pub enum CapabilitySignal {
    /// New capability granted
    CapabilityGranted {
        capability_hash: ActionHash,
        grantor: AgentPubKey,
        access_type: MailboxAccessType,
    },
    /// Capability revoked
    CapabilityRevoked {
        capability_id: String,
        grantor: AgentPubKey,
        reason: Option<String>,
    },
    /// Added to shared mailbox
    AddedToSharedMailbox {
        mailbox_hash: ActionHash,
        mailbox_name: String,
        role: SharedMailboxRole,
    },
    /// New email in shared mailbox
    SharedMailboxNewEmail {
        mailbox_hash: ActionHash,
        email_hash: ActionHash,
    },
    /// Capability used (audit notification)
    CapabilityUsed {
        capability_id: String,
        action: AuditAction,
    },
}

// ==================== GRANT CAPABILITY ====================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GrantCapabilityInput {
    pub grantee: AgentPubKey,
    pub access_type: MailboxAccessType,
    pub permissions: MailboxPermissions,
    pub restrictions: Option<AccessRestrictions>,
    pub expires_at: Option<Timestamp>,
}

/// Grant a new capability to another agent
#[hdk_extern]
pub fn grant_capability(input: GrantCapabilityInput) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Generate unique ID and secret
    let id = format!("cap_{}_{}", my_agent, now.as_micros());
    let secret = random_bytes(32)?;
    // Use secret bytes directly as the hash (32 bytes from random source is sufficient)
    let secret_hash = secret.to_vec();

    let capability = MailboxCapability {
        id: id.clone(),
        grantor: my_agent.clone(),
        grantee: input.grantee.clone(),
        access_type: input.access_type.clone(),
        permissions: input.permissions,
        restrictions: input.restrictions,
        granted_at: now,
        expires_at: input.expires_at,
        revoked: false,
        revocation_reason: None,
        secret_hash,
    };

    let cap_hash = create_entry(EntryTypes::MailboxCapability(capability))?;

    // Link from grantor
    create_link(
        my_agent.clone(),
        cap_hash.clone(),
        LinkTypes::AgentToGrantedCapabilities,
        LinkTag::new(format!("to:{}", input.grantee)),
    )?;

    // Link to grantee
    create_link(
        input.grantee.clone(),
        cap_hash.clone(),
        LinkTypes::AgentToReceivedCapabilities,
        LinkTag::new(format!("from:{}", my_agent)),
    )?;

    // Also create Holochain capability grant for zome function access
    let functions = determine_granted_functions(&input.access_type)?;
    let secret_arr: [u8; 64] = {
        let mut arr = [0u8; 64];
        let bytes = secret.into_vec();
        arr[..bytes.len().min(64)].copy_from_slice(&bytes[..bytes.len().min(64)]);
        arr
    };
    create_cap_grant(CapGrantEntry {
        tag: id,
        access: CapAccess::Assigned {
            secret: CapSecret::from(secret_arr),
            assignees: BTreeSet::from([input.grantee.clone()]),
        },
        functions,
    })?;

    // Signal to grantee
    let signal = CapabilitySignal::CapabilityGranted {
        capability_hash: cap_hash.clone(),
        grantor: my_agent,
        access_type: input.access_type,
    };
    let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
    let _ = send_remote_signal(encoded, vec![input.grantee]);

    // Audit log
    log_capability_action(&cap_hash, AuditAction::GrantCapability, true, None)?;

    Ok(cap_hash)
}

fn determine_granted_functions(access_type: &MailboxAccessType) -> ExternResult<GrantedFunctions> {
    let zome_name = zome_info()?.name;

    Ok(match access_type {
        MailboxAccessType::FullAccess => GrantedFunctions::All,
        MailboxAccessType::ReadOnly => GrantedFunctions::Listed(HashSet::from([
            (zome_name.clone(), "get_emails".into()),
            (zome_name.clone(), "get_email".into()),
            (zome_name.clone(), "get_folders".into()),
            (zome_name, "get_attachments".into()),
        ])),
        MailboxAccessType::SendAs => GrantedFunctions::Listed(HashSet::from([
            (zome_name.clone(), "send_email_as".into()),
            (zome_name, "get_drafts".into()),
        ])),
        _ => GrantedFunctions::Listed(HashSet::new()),
    })
}

// ==================== REVOKE CAPABILITY ====================

/// Revoke a granted capability
#[hdk_extern]
pub fn revoke_capability(input: (ActionHash, Option<String>)) -> ExternResult<ActionHash> {
    let (cap_hash, reason) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Get capability
    let record = get(cap_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Capability not found".to_string()
        )))?;

    let mut capability: MailboxCapability = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid capability".to_string()
        )))?;

    // Verify ownership
    if capability.grantor != my_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only grantor can revoke capability".to_string()
        )));
    }

    // Mark as revoked
    capability.revoked = true;
    capability.revocation_reason = reason.clone();

    let new_hash = update_entry(cap_hash.clone(), EntryTypes::MailboxCapability(capability.clone()))?;

    // NOTE: delete_cap_grant in HDK 0.6 takes ActionHash, not CapSecret.
    // The capability is already marked revoked=true above, which is the authoritative check.
    // TODO: Track cap grant ActionHash at creation time to enable proper deletion here.

    // Signal to grantee
    let signal = CapabilitySignal::CapabilityRevoked {
        capability_id: capability.id,
        grantor: my_agent,
        reason,
    };
    let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
    let _ = send_remote_signal(encoded, vec![capability.grantee]);

    // Audit log
    log_capability_action(&cap_hash, AuditAction::RevokeCapability, true, None)?;

    Ok(new_hash)
}

// ==================== VERIFY CAPABILITY ====================

/// Verify if a capability is valid for an action
#[hdk_extern]
pub fn verify_capability(input: (ActionHash, AuditAction)) -> ExternResult<bool> {
    let (cap_hash, action) = input;
    let caller = agent_info()?.agent_initial_pubkey;

    let record = get(cap_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Capability not found".to_string()
        )))?;

    let capability: MailboxCapability = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid capability".to_string()
        )))?;

    // Check if revoked
    if capability.revoked {
        return Ok(false);
    }

    // Check if caller is grantee
    if capability.grantee != caller {
        return Ok(false);
    }

    // Check expiration
    if let Some(expires) = capability.expires_at {
        if expires < sys_time()? {
            return Ok(false);
        }
    }

    // Check if action is permitted
    let permitted = match action {
        AuditAction::ReadEmail => capability.permissions.can_read,
        AuditAction::SendEmail => capability.permissions.can_send,
        AuditAction::DeleteEmail => capability.permissions.can_delete,
        AuditAction::MoveEmail => capability.permissions.can_move,
        AuditAction::CreateFolder => capability.permissions.can_create_folders,
        AuditAction::AccessAttachment => capability.permissions.can_view_attachments,
        AuditAction::ModifySettings => capability.permissions.can_modify_settings,
        AuditAction::GrantCapability => capability.permissions.can_delegate,
        AuditAction::ModifyTrust => capability.permissions.can_modify_trust,
        _ => true, // Default allow for non-specific actions
    };

    Ok(permitted)
}

// ==================== SHARED MAILBOXES ====================

/// Create a shared mailbox
#[hdk_extern]
pub fn create_shared_mailbox(input: (String, String, SharedMailboxSettings)) -> ExternResult<ActionHash> {
    let (name, email_address, settings) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let id = format!("shared_{}_{}", my_agent, now.as_micros());

    let mailbox = SharedMailbox {
        id,
        name: name.clone(),
        owner: my_agent.clone(),
        email_address,
        members: vec![SharedMailboxMember {
            agent: my_agent.clone(),
            role: SharedMailboxRole::Owner,
            permissions: MailboxPermissions {
                can_read: true,
                can_send: true,
                can_delete: true,
                can_move: true,
                can_create_folders: true,
                can_manage_labels: true,
                can_view_attachments: true,
                can_download_attachments: true,
                can_manage_rules: true,
                can_delegate: true,
                can_modify_settings: true,
                can_view_trust: true,
                can_modify_trust: true,
            },
            added_at: now,
            added_by: my_agent.clone(),
        }],
        created_at: now,
        is_active: true,
        settings,
    };

    let mailbox_hash = create_entry(EntryTypes::SharedMailbox(mailbox))?;

    // Link to owner
    create_link(
        my_agent,
        mailbox_hash.clone(),
        LinkTypes::AgentToOwnedSharedMailboxes,
        LinkTag::new(name.to_string()),
    )?;

    Ok(mailbox_hash)
}

/// Add member to shared mailbox
#[hdk_extern]
pub fn add_shared_mailbox_member(
    input: (ActionHash, AgentPubKey, SharedMailboxRole, MailboxPermissions),
) -> ExternResult<ActionHash> {
    let (mailbox_hash, new_member, role, permissions) = input;
    let my_agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get mailbox
    let record = get(mailbox_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Shared mailbox not found".to_string()
        )))?;

    let mut mailbox: SharedMailbox = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid mailbox".to_string()
        )))?;

    // Verify caller has permission to add members
    let caller_member = mailbox
        .members
        .iter()
        .find(|m| m.agent == my_agent)
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Not a member of this mailbox".to_string()
        )))?;

    match caller_member.role {
        SharedMailboxRole::Owner | SharedMailboxRole::Admin => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Insufficient permissions to add members".to_string()
            )));
        }
    }

    // Add new member
    let member = SharedMailboxMember {
        agent: new_member.clone(),
        role: role.clone(),
        permissions,
        added_at: now,
        added_by: my_agent,
    };

    mailbox.members.push(member);

    let new_hash = update_entry(mailbox_hash.clone(), EntryTypes::SharedMailbox(mailbox.clone()))?;

    // Link to new member
    create_link(
        new_member.clone(),
        mailbox_hash,
        LinkTypes::AgentToMemberSharedMailboxes,
        LinkTag::new(mailbox.name.to_string()),
    )?;

    // Signal to new member
    let signal = CapabilitySignal::AddedToSharedMailbox {
        mailbox_hash: new_hash.clone(),
        mailbox_name: mailbox.name,
        role,
    };
    let encoded = ExternIO::encode(signal).map_err(|e| wasm_error!(WasmErrorInner::Serialize(e)))?;
    let _ = send_remote_signal(encoded, vec![new_member]);

    Ok(new_hash)
}

/// Get shared mailboxes user is member of
#[hdk_extern]
pub fn get_my_shared_mailboxes(_: ()) -> ExternResult<Vec<(ActionHash, SharedMailbox)>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let mut mailboxes = Vec::new();

    // Get owned mailboxes
    let owned_links = get_links(LinkQuery::try_new(my_agent.clone(), LinkTypes::AgentToOwnedSharedMailboxes)?, GetStrategy::default())?;

    // Get member mailboxes
    let member_links = get_links(LinkQuery::try_new(my_agent, LinkTypes::AgentToMemberSharedMailboxes)?, GetStrategy::default())?;

    for link in owned_links.into_iter().chain(member_links.into_iter()) {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(mailbox) = record
                .entry()
                .to_app_option::<SharedMailbox>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                mailboxes.push((hash, mailbox));
            }
        }
    }

    Ok(mailboxes)
}

// ==================== AUDIT LOGGING ====================

fn log_capability_action(
    capability_hash: &ActionHash,
    action: AuditAction,
    success: bool,
    error: Option<String>,
) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let log = CapabilityAuditLog {
        capability_id: capability_hash.to_string(),
        actor: my_agent,
        action,
        resource: Some(capability_hash.clone()),
        timestamp: sys_time()?,
        success,
        error,
        context: None,
    };

    let log_hash = create_entry(EntryTypes::CapabilityAuditLog(log))?;

    create_link(
        capability_hash.clone(),
        log_hash.clone(),
        LinkTypes::CapabilityToAuditLogs,
        LinkTag::new("audit"),
    )?;

    Ok(log_hash)
}

/// Log an action using a capability
#[hdk_extern]
pub fn log_action(input: (ActionHash, AuditAction, bool, Option<String>)) -> ExternResult<ActionHash> {
    let (cap_hash, action, success, error) = input;

    // Verify capability first
    let verified = verify_capability((cap_hash.clone(), action.clone()))?;
    if !verified && success {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot log successful action for invalid capability".to_string()
        )));
    }

    log_capability_action(&cap_hash, action, success, error)
}

/// Get audit logs for a capability
#[hdk_extern]
pub fn get_audit_logs(capability_hash: ActionHash) -> ExternResult<Vec<CapabilityAuditLog>> {
    let links = get_links(LinkQuery::try_new(capability_hash, LinkTypes::CapabilityToAuditLogs)?, GetStrategy::default())?;

    let mut logs = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(log) = record
                .entry()
                .to_app_option::<CapabilityAuditLog>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                logs.push(log);
            }
        }
    }

    // Sort by timestamp
    logs.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

    Ok(logs)
}

// ==================== QUERY CAPABILITIES ====================

/// Get capabilities I've granted
#[hdk_extern]
pub fn get_granted_capabilities(_: ()) -> ExternResult<Vec<(ActionHash, MailboxCapability)>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(my_agent, LinkTypes::AgentToGrantedCapabilities)?, GetStrategy::default())?;

    let mut capabilities = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(cap) = record
                .entry()
                .to_app_option::<MailboxCapability>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                capabilities.push((hash, cap));
            }
        }
    }

    Ok(capabilities)
}

/// Get capabilities I've received
#[hdk_extern]
pub fn get_received_capabilities(_: ()) -> ExternResult<Vec<(ActionHash, MailboxCapability)>> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    let links = get_links(LinkQuery::try_new(my_agent, LinkTypes::AgentToReceivedCapabilities)?, GetStrategy::default())?;

    let mut capabilities = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".to_string())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            if let Some(cap) = record
                .entry()
                .to_app_option::<MailboxCapability>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                // Only return non-revoked, non-expired
                if !cap.revoked {
                    if let Some(expires) = cap.expires_at {
                        if expires > sys_time()? {
                            capabilities.push((hash, cap));
                        }
                    } else {
                        capabilities.push((hash, cap));
                    }
                }
            }
        }
    }

    Ok(capabilities)
}

// ==================== SIGNAL HANDLING ====================

#[hdk_extern]
pub fn recv_remote_signal(signal: ExternIO) -> ExternResult<()> {
    let cap_signal: CapabilitySignal = signal.decode().map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to decode signal: {}",
            e
        )))
    })?;

    emit_signal(cap_signal)?;

    Ok(())
}

// ==================== INIT ====================

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Grant capability for receiving signals
    let functions = GrantedFunctions::Listed(HashSet::from([
        (zome_info()?.name, "recv_remote_signal".into()),
    ]));

    create_cap_grant(CapGrantEntry {
        tag: "recv_cap_signals".to_string(),
        access: CapAccess::Unrestricted,
        functions,
    })?;

    Ok(InitCallbackResult::Pass)
}
