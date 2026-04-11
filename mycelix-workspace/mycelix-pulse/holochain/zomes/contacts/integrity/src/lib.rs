// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contacts Integrity Zome
//!
//! Entry types and validation for contacts and groups.

use hdi::prelude::*;

/// Contact entry with full metadata
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contact {
    pub id: String,
    pub display_name: String,
    pub nickname: Option<String>,
    pub emails: Vec<ContactEmail>,
    pub phones: Vec<ContactPhone>,
    pub addresses: Vec<ContactAddress>,
    pub organization: Option<String>,
    pub title: Option<String>,
    pub notes: Option<String>,
    pub avatar: Option<String>,
    pub groups: Vec<String>,
    pub labels: Vec<String>,
    pub agent_pub_key: Option<AgentPubKey>,
    pub is_favorite: bool,
    pub is_blocked: bool,
    pub metadata: ContactMetadata,
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContactEmail {
    pub email: String,
    pub email_type: String,
    pub is_primary: bool,
    pub verified: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContactPhone {
    pub number: String,
    pub phone_type: String,
    pub is_primary: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContactAddress {
    pub street: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub country: Option<String>,
    pub address_type: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContactMetadata {
    pub source: String,
    pub last_email_sent: Option<u64>,
    pub last_email_received: Option<u64>,
    pub email_count: u32,
    pub response_rate: Option<f64>,
    pub average_response_time: Option<u64>,
}

/// Contact group
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ContactGroup {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub color: Option<String>,
    pub icon: Option<String>,
    pub created_at: u64,
}

/// Group membership link
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GroupMembership {
    pub contact_id: String,
    pub group_id: String,
    pub added_at: u64,
}

/// Blocked contact record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BlockedContact {
    pub contact_hash: ActionHash,
    pub reason: Option<String>,
    pub blocked_at: u64,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Contact(Contact),
    ContactGroup(ContactGroup),
    GroupMembership(GroupMembership),
    BlockedContact(BlockedContact),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllContacts,
    AllGroups,
    ContactToGroups,
    GroupToContacts,
    AgentToContact,
    BlockedContacts,
    EmailToContact,
}

/// Validate contact entry
fn validate_create_contact(_action: Create, contact: Contact) -> ExternResult<ValidateCallbackResult> {
    // Validate display name
    if contact.display_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Display name cannot be empty".to_string(),
        ));
    }

    // Validate at least one email if any emails provided
    for email in &contact.emails {
        if !email.email.contains('@') {
            return Ok(ValidateCallbackResult::Invalid(
                format!("Invalid email format: {}", email.email),
            ));
        }
    }

    // Validate ID format
    if contact.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact ID cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate contact update
fn validate_update_contact(contact: Contact) -> ExternResult<ValidateCallbackResult> {
    // Same validation as create
    if contact.display_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Display name cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate group entry
fn validate_create_contact_group(group: ContactGroup) -> ExternResult<ValidateCallbackResult> {
    if group.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Group name cannot be empty".to_string(),
        ));
    }

    if group.name.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Group name too long (max 100 characters)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate membership
fn validate_create_group_membership(membership: GroupMembership) -> ExternResult<ValidateCallbackResult> {
    if membership.contact_id.is_empty() || membership.group_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact ID and Group ID are required".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Contact(contact) => validate_create_contact(action, contact),
                EntryTypes::ContactGroup(group) => validate_create_contact_group(group),
                EntryTypes::GroupMembership(membership) => {
                    validate_create_group_membership(membership)
                }
                EntryTypes::BlockedContact(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Contact(contact) => validate_update_contact(contact),
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
