// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contacts Coordinator Zome
//!
//! CRUD operations for contacts and groups.

#[cfg(test)]
mod tests;

use hdk::prelude::*;
use contacts_integrity::*;

/// Anchor for all contacts
const ALL_CONTACTS_ANCHOR: &str = "all_contacts";
/// Anchor for all groups
const ALL_GROUPS_ANCHOR: &str = "all_groups";
/// Anchor for blocked contacts
const BLOCKED_ANCHOR: &str = "blocked_contacts";

// ==================== CONTACT CRUD ====================

/// Create a new contact
#[hdk_extern]
pub fn create_contact(contact: Contact) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::Contact(contact.clone()))?;

    // Link to all contacts anchor
    let anchor = anchor_hash(ALL_CONTACTS_ANCHOR)?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::AllContacts,
        (),
    )?;

    // Link by email for lookup
    for email in &contact.emails {
        let email_anchor = email_anchor_hash(&email.email)?;
        create_link(
            email_anchor,
            action_hash.clone(),
            LinkTypes::EmailToContact,
            (),
        )?;
    }

    // Link from agent if provided
    if let Some(agent) = &contact.agent_pub_key {
        create_link(
            agent.clone(),
            action_hash.clone(),
            LinkTypes::AgentToContact,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get a contact by hash
#[hdk_extern]
pub fn get_contact(hash: ActionHash) -> ExternResult<Option<Contact>> {
    match get(hash, GetOptions::default())? {
        Some(record) => {
            let contact: Contact = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(e))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Contact not found".to_string()
                )))?;
            Ok(Some(contact))
        }
        None => Ok(None),
    }
}

/// Update a contact
#[hdk_extern]
pub fn update_contact(input: UpdateContactInput) -> ExternResult<ActionHash> {
    update_entry(input.hash, EntryTypes::Contact(input.contact))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateContactInput {
    pub hash: ActionHash,
    pub contact: Contact,
}

/// Delete a contact
#[hdk_extern]
pub fn delete_contact(hash: ActionHash) -> ExternResult<ActionHash> {
    delete_entry(hash)
}

/// Get all contacts
#[hdk_extern]
pub fn get_all_contacts(_: ()) -> ExternResult<Vec<Contact>> {
    let anchor = anchor_hash(ALL_CONTACTS_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AllContacts)?, GetStrategy::default())?;

    let mut contacts = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(contact) = get_contact(hash)? {
                contacts.push(contact);
            }
        }
    }

    // Sort by display name
    contacts.sort_by(|a, b| a.display_name.cmp(&b.display_name));

    Ok(contacts)
}

/// Get contact by email
#[hdk_extern]
pub fn get_contact_by_email(email: String) -> ExternResult<Option<Contact>> {
    let email_anchor = email_anchor_hash(&email)?;
    let links = get_links(LinkQuery::try_new(email_anchor, LinkTypes::EmailToContact)?, GetStrategy::default())?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            return get_contact(hash);
        }
    }

    Ok(None)
}

/// Get contact by agent
#[hdk_extern]
pub fn get_contact_by_agent(agent: AgentPubKey) -> ExternResult<Option<Contact>> {
    let links = get_links(LinkQuery::try_new(agent, LinkTypes::AgentToContact)?, GetStrategy::default())?;

    if let Some(link) = links.first() {
        if let Some(hash) = link.target.clone().into_action_hash() {
            return get_contact(hash);
        }
    }

    Ok(None)
}

// ==================== GROUPS ====================

/// Create a contact group
#[hdk_extern]
pub fn create_group(group: ContactGroup) -> ExternResult<ActionHash> {
    let action_hash = create_entry(EntryTypes::ContactGroup(group))?;

    let anchor = anchor_hash(ALL_GROUPS_ANCHOR)?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::AllGroups,
        (),
    )?;

    Ok(action_hash)
}

/// Get all groups
#[hdk_extern]
pub fn get_all_groups(_: ()) -> ExternResult<Vec<ContactGroup>> {
    let anchor = anchor_hash(ALL_GROUPS_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::AllGroups)?, GetStrategy::default())?;

    let mut groups = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(group) = record.entry().to_app_option().map_err(|e| wasm_error!(e))? {
                    groups.push(group);
                }
            }
        }
    }

    groups.sort_by(|a: &ContactGroup, b: &ContactGroup| a.name.cmp(&b.name));

    Ok(groups)
}

/// Add contact to group
#[hdk_extern]
pub fn add_to_group(input: GroupMembershipInput) -> ExternResult<ActionHash> {
    let membership = GroupMembership {
        contact_id: input.contact_hash.to_string(),
        group_id: input.group_id.clone(),
        added_at: sys_time()?.as_micros() as u64,
    };

    let action_hash = create_entry(EntryTypes::GroupMembership(membership))?;

    // Create bidirectional links
    create_link(
        input.contact_hash.clone(),
        action_hash.clone(),
        LinkTypes::ContactToGroups,
        input.group_id.as_bytes().to_vec(),
    )?;

    Ok(action_hash)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupMembershipInput {
    pub contact_hash: ActionHash,
    pub group_id: String,
}

/// Remove contact from group
#[hdk_extern]
pub fn remove_from_group(input: GroupMembershipInput) -> ExternResult<()> {
    let links = get_links(LinkQuery::try_new(input.contact_hash.clone(), LinkTypes::ContactToGroups)?, GetStrategy::default())?;

    for link in links {
        // Check if this link is for the specified group
        if link.tag.0 == input.group_id.as_bytes() {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }

    Ok(())
}

/// Get contacts in group
#[hdk_extern]
pub fn get_group_members(group_id: String) -> ExternResult<Vec<Contact>> {
    // Get all contacts and filter by group membership
    let contacts = get_all_contacts(())?;
    let members: Vec<Contact> = contacts
        .into_iter()
        .filter(|c| c.groups.contains(&group_id))
        .collect();

    Ok(members)
}

// ==================== BLOCKING ====================

/// Block a contact
#[hdk_extern]
pub fn block_contact(hash: ActionHash) -> ExternResult<ActionHash> {
    let blocked = BlockedContact {
        contact_hash: hash.clone(),
        reason: None,
        blocked_at: sys_time()?.as_micros() as u64,
    };

    let action_hash = create_entry(EntryTypes::BlockedContact(blocked))?;

    let anchor = anchor_hash(BLOCKED_ANCHOR)?;
    create_link(
        anchor,
        action_hash.clone(),
        LinkTypes::BlockedContacts,
        (),
    )?;

    Ok(action_hash)
}

/// Unblock a contact
#[hdk_extern]
pub fn unblock_contact(hash: ActionHash) -> ExternResult<()> {
    let anchor = anchor_hash(BLOCKED_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::BlockedContacts)?, GetStrategy::default())?;

    for link in links {
        if let Some(blocked_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(blocked_hash.clone(), GetOptions::default())? {
                if let Some(blocked) = record
                    .entry()
                    .to_app_option::<BlockedContact>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if blocked.contact_hash == hash {
                        delete_link(link.create_link_hash, GetOptions::default())?;
                        delete_entry(blocked_hash)?;
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Get blocked contacts
#[hdk_extern]
pub fn get_blocked_contacts(_: ()) -> ExternResult<Vec<ActionHash>> {
    let anchor = anchor_hash(BLOCKED_ANCHOR)?;
    let links = get_links(LinkQuery::try_new(anchor, LinkTypes::BlockedContacts)?, GetStrategy::default())?;

    let mut blocked = Vec::new();
    for link in links {
        if let Some(hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(bc) = record
                    .entry()
                    .to_app_option::<BlockedContact>()
                    .map_err(|e| wasm_error!(e))?
                {
                    blocked.push(bc.contact_hash);
                }
            }
        }
    }

    Ok(blocked)
}

// ==================== HELPERS ====================

fn anchor_hash(name: &str) -> ExternResult<EntryHash> {
    let path = Path::from(name);
    path.path_entry_hash()
}

fn email_anchor_hash(email: &str) -> ExternResult<EntryHash> {
    let normalized = email.to_lowercase();
    let path = Path::from(format!("email:{}", normalized));
    path.path_entry_hash()
}
