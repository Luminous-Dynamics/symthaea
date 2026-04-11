// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contacts Zome for Mycelix Mail
//!
//! Manages contacts and contact discovery on Holochain DHT

use hdk::prelude::*;

/// Contact entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Contact {
    /// The agent this contact refers to
    pub agent: AgentPubKey,
    /// Local alias/display name
    pub alias: Option<String>,
    /// Personal notes about this contact
    pub notes: Option<String>,
    /// Contact groups/tags
    pub tags: Vec<String>,
    /// Whether this contact is blocked
    pub is_blocked: bool,
    /// Whether this is a favorite contact
    pub is_favorite: bool,
}

/// Contact with computed data
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ContactWithInfo {
    pub contact: Contact,
    pub action_hash: ActionHash,
    pub trust_score: Option<f64>,
    pub profile_name: Option<String>,
    pub profile_email: Option<String>,
    pub created_at: Timestamp,
}

/// Input for creating a contact
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateContactInput {
    pub agent: AgentPubKey,
    pub alias: Option<String>,
    pub notes: Option<String>,
    pub tags: Vec<String>,
}

/// Input for updating a contact
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateContactInput {
    pub action_hash: ActionHash,
    pub alias: Option<String>,
    pub notes: Option<String>,
    pub tags: Option<Vec<String>>,
    pub is_blocked: Option<bool>,
    pub is_favorite: Option<bool>,
}

entry_defs![
    PathEntry::entry_def(),
    Contact::entry_def()
];

#[hdk_link_types]
pub enum LinkTypes {
    /// Link from owner to their contacts
    OwnerToContacts,
    /// Link from tag to contacts
    TagToContacts,
    /// Link to blocked contacts
    BlockedContacts,
    /// Link to favorite contacts
    FavoriteContacts,
}

// ============================================================================
// Zome Functions
// ============================================================================

/// Add a new contact
#[hdk_extern]
pub fn add_contact(input: CreateContactInput) -> ExternResult<ActionHash> {
    let owner = agent_info()?.agent_initial_pubkey;

    // Cannot add yourself as contact
    if owner == input.agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot add yourself as a contact".to_string()
        )));
    }

    // Check if contact already exists
    let existing = get_contact_for_agent(input.agent.clone())?;
    if existing.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Contact already exists".to_string()
        )));
    }

    let contact = Contact {
        agent: input.agent.clone(),
        alias: input.alias,
        notes: input.notes,
        tags: input.tags.clone(),
        is_blocked: false,
        is_favorite: false,
    };

    let action_hash = create_entry(&contact)?;

    // Link owner to contact
    create_link(
        owner.clone(),
        action_hash.clone(),
        LinkTypes::OwnerToContacts,
        input.agent.get_raw_39().to_vec(),
    )?;

    // Create tag links
    for tag in input.tags {
        let tag_path = Path::from(format!("tags/{}/{}", owner, tag.to_lowercase()));
        tag_path.ensure()?;
        create_link(
            tag_path.path_entry_hash()?,
            action_hash.clone(),
            LinkTypes::TagToContacts,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get all contacts
#[hdk_extern]
pub fn get_contacts(_: ()) -> ExternResult<Vec<ContactWithInfo>> {
    let owner = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        GetLinksInputBuilder::try_new(owner, LinkTypes::OwnerToContacts)?
            .build(),
    )?;

    let mut contacts = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
        })?;

        if let Some(contact_info) = get_contact_by_hash(action_hash)? {
            contacts.push(contact_info);
        }
    }

    // Sort by favorite first, then by alias/name
    contacts.sort_by(|a, b| {
        match (a.contact.is_favorite, b.contact.is_favorite) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => {
                let name_a = a.contact.alias.as_ref()
                    .or(a.profile_name.as_ref())
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                let name_b = b.contact.alias.as_ref()
                    .or(b.profile_name.as_ref())
                    .map(|s| s.to_lowercase())
                    .unwrap_or_default();
                name_a.cmp(&name_b)
            }
        }
    });

    Ok(contacts)
}

/// Get contact for a specific agent
#[hdk_extern]
pub fn get_contact_for_agent(agent: AgentPubKey) -> ExternResult<Option<ContactWithInfo>> {
    let owner = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        GetLinksInputBuilder::try_new(owner, LinkTypes::OwnerToContacts)?
            .build(),
    )?;

    for link in links {
        // Check tag (which contains the agent pubkey)
        if link.tag.into_inner() == agent.get_raw_39().to_vec() {
            let action_hash = ActionHash::try_from(link.target).map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
            })?;
            return get_contact_by_hash(action_hash);
        }
    }

    Ok(None)
}

/// Get contact by action hash
fn get_contact_by_hash(action_hash: ActionHash) -> ExternResult<Option<ContactWithInfo>> {
    if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
        if let Some(contact) = record
            .entry()
            .to_app_option::<Contact>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {:?}", e))))?
        {
            // Try to get trust score (call to trust zome would go here)
            // For now, return None
            let trust_score = None;

            // Try to get profile info (call to profiles zome would go here)
            let profile_name = None;
            let profile_email = None;

            return Ok(Some(ContactWithInfo {
                contact,
                action_hash,
                trust_score,
                profile_name,
                profile_email,
                created_at: record.action().timestamp(),
            }));
        }
    }

    Ok(None)
}

/// Update a contact
#[hdk_extern]
pub fn update_contact(input: UpdateContactInput) -> ExternResult<ActionHash> {
    let current = get_contact_by_hash(input.action_hash.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contact not found".to_string())))?;

    let updated = Contact {
        agent: current.contact.agent,
        alias: input.alias.or(current.contact.alias),
        notes: input.notes.or(current.contact.notes),
        tags: input.tags.unwrap_or(current.contact.tags),
        is_blocked: input.is_blocked.unwrap_or(current.contact.is_blocked),
        is_favorite: input.is_favorite.unwrap_or(current.contact.is_favorite),
    };

    update_entry(input.action_hash, &updated)
}

/// Block a contact
#[hdk_extern]
pub fn block_contact(action_hash: ActionHash) -> ExternResult<ActionHash> {
    let owner = agent_info()?.agent_initial_pubkey;

    let current = get_contact_by_hash(action_hash.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contact not found".to_string())))?;

    let updated = Contact {
        is_blocked: true,
        ..current.contact
    };

    let new_hash = update_entry(action_hash, &updated)?;

    // Add to blocked list
    create_link(
        owner,
        new_hash.clone(),
        LinkTypes::BlockedContacts,
        (),
    )?;

    Ok(new_hash)
}

/// Unblock a contact
#[hdk_extern]
pub fn unblock_contact(action_hash: ActionHash) -> ExternResult<ActionHash> {
    let current = get_contact_by_hash(action_hash.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contact not found".to_string())))?;

    let updated = Contact {
        is_blocked: false,
        ..current.contact
    };

    update_entry(action_hash, &updated)
}

/// Get blocked contacts
#[hdk_extern]
pub fn get_blocked_contacts(_: ()) -> ExternResult<Vec<ContactWithInfo>> {
    let owner = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        GetLinksInputBuilder::try_new(owner, LinkTypes::BlockedContacts)?
            .build(),
    )?;

    let mut contacts = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
        })?;

        if let Some(contact) = get_contact_by_hash(action_hash)? {
            if contact.contact.is_blocked {
                contacts.push(contact);
            }
        }
    }

    Ok(contacts)
}

/// Toggle favorite status
#[hdk_extern]
pub fn toggle_favorite(action_hash: ActionHash) -> ExternResult<ActionHash> {
    let owner = agent_info()?.agent_initial_pubkey;

    let current = get_contact_by_hash(action_hash.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contact not found".to_string())))?;

    let new_favorite = !current.contact.is_favorite;

    let updated = Contact {
        is_favorite: new_favorite,
        ..current.contact
    };

    let new_hash = update_entry(action_hash, &updated)?;

    // Update favorite links
    if new_favorite {
        create_link(
            owner,
            new_hash.clone(),
            LinkTypes::FavoriteContacts,
            (),
        )?;
    }

    Ok(new_hash)
}

/// Get favorite contacts
#[hdk_extern]
pub fn get_favorite_contacts(_: ()) -> ExternResult<Vec<ContactWithInfo>> {
    let owner = agent_info()?.agent_initial_pubkey;

    let links = get_links(
        GetLinksInputBuilder::try_new(owner, LinkTypes::FavoriteContacts)?
            .build(),
    )?;

    let mut contacts = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
        })?;

        if let Some(contact) = get_contact_by_hash(action_hash)? {
            if contact.contact.is_favorite {
                contacts.push(contact);
            }
        }
    }

    Ok(contacts)
}

/// Get contacts by tag
#[hdk_extern]
pub fn get_contacts_by_tag(tag: String) -> ExternResult<Vec<ContactWithInfo>> {
    let owner = agent_info()?.agent_initial_pubkey;
    let tag_path = Path::from(format!("tags/{}/{}", owner, tag.to_lowercase()));

    let links = get_links(
        GetLinksInputBuilder::try_new(tag_path.path_entry_hash()?, LinkTypes::TagToContacts)?
            .build(),
    )?;

    let mut contacts = Vec::new();

    for link in links {
        let action_hash = ActionHash::try_from(link.target).map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!("Invalid hash: {:?}", e)))
        })?;

        if let Some(contact) = get_contact_by_hash(action_hash)? {
            contacts.push(contact);
        }
    }

    Ok(contacts)
}

/// Delete a contact
#[hdk_extern]
pub fn delete_contact(action_hash: ActionHash) -> ExternResult<ActionHash> {
    delete_entry(action_hash)
}

// ============================================================================
// Validation
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<Contact, ()>()? {
        FlatOp::StoreEntry(store_entry) => {
            match store_entry {
                OpEntry::CreateEntry { entry, .. } => {
                    if let Entry::App(app_entry) = entry {
                        let _contact: Contact = app_entry.into_sb().try_into()?;
                        // Add validation logic here
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
