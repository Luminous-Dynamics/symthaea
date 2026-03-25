// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capabilities Integrity Zome
//!
//! Fine-grained access control for Mycelix Mail using Holochain capabilities.
//! Enables shared mailboxes, delegated access, and time-limited permissions.

use hdi::prelude::*;

/// Capability grant for mailbox access
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MailboxCapability {
    /// Unique identifier for this capability
    pub id: String,
    /// Grantor (mailbox owner)
    pub grantor: AgentPubKey,
    /// Grantee (who receives access)
    pub grantee: AgentPubKey,
    /// Type of access granted
    pub access_type: MailboxAccessType,
    /// Specific permissions
    pub permissions: MailboxPermissions,
    /// Optional resource restrictions
    pub restrictions: Option<AccessRestrictions>,
    /// When capability was granted
    pub granted_at: Timestamp,
    /// Expiration time
    pub expires_at: Option<Timestamp>,
    /// Whether capability has been revoked
    pub revoked: bool,
    /// Revocation reason if revoked
    pub revocation_reason: Option<String>,
    /// Secret for capability verification
    pub secret_hash: Vec<u8>,
}

/// Type of mailbox access
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum MailboxAccessType {
    /// Full access (owner equivalent)
    FullAccess,
    /// Read-only access
    ReadOnly,
    /// Send on behalf of
    SendAs,
    /// Manage specific folder
    FolderAccess { folder_hash: ActionHash },
    /// Access to specific thread
    ThreadAccess { thread_id: String },
    /// Temporary out-of-office delegation
    OutOfOffice,
    /// Admin access for organization
    OrganizationAdmin,
    /// Custom access type
    Custom(String),
}

/// Granular permissions
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct MailboxPermissions {
    /// Can read emails
    pub can_read: bool,
    /// Can send emails
    pub can_send: bool,
    /// Can delete emails
    pub can_delete: bool,
    /// Can move emails between folders
    pub can_move: bool,
    /// Can create folders
    pub can_create_folders: bool,
    /// Can manage labels
    pub can_manage_labels: bool,
    /// Can view attachments
    pub can_view_attachments: bool,
    /// Can download attachments
    pub can_download_attachments: bool,
    /// Can manage filters/rules
    pub can_manage_rules: bool,
    /// Can grant sub-capabilities
    pub can_delegate: bool,
    /// Can modify settings
    pub can_modify_settings: bool,
    /// Can view trust scores
    pub can_view_trust: bool,
    /// Can modify trust attestations
    pub can_modify_trust: bool,
}

/// Restrictions on access
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AccessRestrictions {
    /// Only specific folders
    pub folder_whitelist: Option<Vec<ActionHash>>,
    /// Excluded folders
    pub folder_blacklist: Option<Vec<ActionHash>>,
    /// Only emails from specific senders
    pub sender_whitelist: Option<Vec<String>>,
    /// Maximum emails accessible
    pub max_emails: Option<u32>,
    /// Only emails after this date
    pub date_from: Option<Timestamp>,
    /// Only emails before this date
    pub date_to: Option<Timestamp>,
    /// IP/network restrictions (for enterprise)
    pub network_restrictions: Option<Vec<String>>,
    /// Require 2FA for access
    pub require_2fa: bool,
    /// Audit all access
    pub audit_required: bool,
}

/// Shared mailbox configuration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SharedMailbox {
    /// Unique identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Owner agent
    pub owner: AgentPubKey,
    /// Email address for this mailbox
    pub email_address: String,
    /// Members with access
    pub members: Vec<SharedMailboxMember>,
    /// When created
    pub created_at: Timestamp,
    /// Whether active
    pub is_active: bool,
    /// Mailbox settings
    pub settings: SharedMailboxSettings,
}

/// Member of a shared mailbox
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct SharedMailboxMember {
    /// Member agent
    pub agent: AgentPubKey,
    /// Their role
    pub role: SharedMailboxRole,
    /// Their specific permissions
    pub permissions: MailboxPermissions,
    /// When added
    pub added_at: Timestamp,
    /// Who added them
    pub added_by: AgentPubKey,
}

/// Roles in shared mailbox
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum SharedMailboxRole {
    Owner,
    Admin,
    Editor,
    Contributor,
    Viewer,
}

/// Settings for shared mailbox
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Default)]
pub struct SharedMailboxSettings {
    /// Send notifications on new mail
    pub notify_all_on_new_mail: bool,
    /// Allow members to send as mailbox
    pub allow_send_as: bool,
    /// Require approval for external sends
    pub require_send_approval: bool,
    /// Auto-assign emails to members
    pub auto_assign: bool,
    /// Assignment strategy
    pub assignment_strategy: Option<AssignmentStrategy>,
}

/// Strategy for auto-assigning emails
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum AssignmentStrategy {
    RoundRobin,
    LeastBusy,
    ByKeyword { rules: Vec<(String, AgentPubKey)> },
    BySender { rules: Vec<(String, AgentPubKey)> },
    Random,
}

/// Capability audit log entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CapabilityAuditLog {
    /// Capability that was used
    pub capability_id: String,
    /// Who used it
    pub actor: AgentPubKey,
    /// What action was taken
    pub action: AuditAction,
    /// Resource affected
    pub resource: Option<ActionHash>,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Success or failure
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Additional context
    pub context: Option<String>,
}

/// Actions that can be audited
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub enum AuditAction {
    ReadEmail,
    SendEmail,
    DeleteEmail,
    MoveEmail,
    CreateFolder,
    DeleteFolder,
    ModifySettings,
    GrantCapability,
    RevokeCapability,
    AccessAttachment,
    ModifyTrust,
    ExportData,
    Login,
    Logout,
    Custom(String),
}

/// Delegation chain tracking
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DelegationChain {
    /// Original grantor
    pub root_grantor: AgentPubKey,
    /// Chain of delegations
    pub chain: Vec<DelegationHop>,
    /// Current capability hash
    pub current_capability: ActionHash,
    /// Maximum chain length allowed
    pub max_depth: u8,
}

/// Single hop in delegation chain
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DelegationHop {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub capability_hash: ActionHash,
    pub timestamp: Timestamp,
}

/// Link types
#[hdk_link_types]
pub enum LinkTypes {
    /// Agent -> capabilities they've granted
    AgentToGrantedCapabilities,
    /// Agent -> capabilities they've received
    AgentToReceivedCapabilities,
    /// Capability -> audit logs
    CapabilityToAuditLogs,
    /// Shared mailbox -> members
    SharedMailboxToMembers,
    /// Agent -> shared mailboxes they own
    AgentToOwnedSharedMailboxes,
    /// Agent -> shared mailboxes they're member of
    AgentToMemberSharedMailboxes,
    /// Capability -> delegation chain
    CapabilityToDelegationChain,
}

/// Entry types
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(required_validations = 2)]
    MailboxCapability(MailboxCapability),
    #[entry_type(required_validations = 2)]
    SharedMailbox(SharedMailbox),
    #[entry_type(required_validations = 1)]
    CapabilityAuditLog(CapabilityAuditLog),
    #[entry_type(required_validations = 2)]
    DelegationChain(DelegationChain),
}

// ==================== VALIDATION ====================

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
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(
    entry: EntryTypes,
    action: Create,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        EntryTypes::MailboxCapability(cap) => validate_capability(&cap, &action),
        EntryTypes::SharedMailbox(mailbox) => validate_shared_mailbox(&mailbox, &action),
        EntryTypes::CapabilityAuditLog(log) => validate_audit_log(&log, &action),
        EntryTypes::DelegationChain(chain) => validate_delegation_chain(&chain, &action),
    }
}

fn validate_update_entry(
    entry: EntryTypes,
    action: Update,
) -> ExternResult<ValidateCallbackResult> {
    match entry {
        // Capabilities can only be updated by grantor (to revoke)
        EntryTypes::MailboxCapability(cap) => {
            if cap.grantor != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only grantor can update capability".to_string(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        // Shared mailboxes can be updated by owner or admin
        EntryTypes::SharedMailbox(mailbox) => {
            if mailbox.owner != action.author {
                // Check if the author is an admin member
                let is_admin = mailbox.members.iter().any(|m| {
                    m.agent == action.author && matches!(m.role, SharedMailboxRole::Admin)
                });
                if !is_admin {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Only mailbox owner or admin can update shared mailbox".to_string(),
                    ));
                }
            }
            Ok(ValidateCallbackResult::Valid)
        }
        // Audit logs are immutable
        EntryTypes::CapabilityAuditLog(_) => Ok(ValidateCallbackResult::Invalid(
            "Audit logs cannot be modified".to_string(),
        )),
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_capability(
    cap: &MailboxCapability,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Grantor must be author
    if cap.grantor != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Grantor must match author".to_string(),
        ));
    }

    // Can't grant to self
    if cap.grantor == cap.grantee {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot grant capability to self".to_string(),
        ));
    }

    // Must have secret hash
    if cap.secret_hash.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Capability must have secret hash".to_string(),
        ));
    }

    // If has expiration, must be in future
    if let Some(expires) = cap.expires_at {
        if expires <= cap.granted_at {
            return Ok(ValidateCallbackResult::Invalid(
                "Expiration must be after grant time".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_shared_mailbox(
    mailbox: &SharedMailbox,
    action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Owner must be author
    if mailbox.owner != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must match author".to_string(),
        ));
    }

    // Must have name
    if mailbox.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shared mailbox must have name".to_string(),
        ));
    }

    // Must have email address
    if mailbox.email_address.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shared mailbox must have email address".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_audit_log(
    _log: &CapabilityAuditLog,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Audit logs have minimal validation - they're append-only
    Ok(ValidateCallbackResult::Valid)
}

fn validate_delegation_chain(
    chain: &DelegationChain,
    _action: &Create,
) -> ExternResult<ValidateCallbackResult> {
    // Chain must not exceed max depth
    if chain.chain.len() > chain.max_depth as usize {
        return Ok(ValidateCallbackResult::Invalid(
            "Delegation chain exceeds maximum depth".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
