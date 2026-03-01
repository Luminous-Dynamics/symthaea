use hdk::prelude::*;
use registry_integrity::*;
use serde::{Deserialize, Serialize};

// ── Signals ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum RegistrySignal {
    DependencyRegistered {
        dependency_id: String,
        name: String,
        ecosystem: String,
    },
    DependencyUpdated {
        dependency_id: String,
    },
    DependencyVerified {
        dependency_id: String,
    },
}

// ── Input Types ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateDependencyInput {
    pub original_action_hash: ActionHash,
    pub dependency: DependencyIdentity,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginationInput {
    pub offset: u64,
    pub limit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedDependencies {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BulkRegisterResult {
    pub registered: Vec<Record>,
    pub skipped: Vec<String>,
}

// ── Init ─────────────────────────────────────────────────────────────

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    let anchor = Anchor("all_deps".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

// ── Helpers ──────────────────────────────────────────────────────────

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
}

fn require_author(maintainer_did: &str) -> ExternResult<()> {
    let info = agent_info()?;
    let expected_did = format!("did:mycelix:{}", info.agent_initial_pubkey);
    if maintainer_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the maintainer can perform this action".into()
        )));
    }
    Ok(())
}

fn resolve_links(base: EntryHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(base, link_type)?,
        GetStrategy::default(),
    )?;
    let mut records = Vec::new();
    for link in links {
        let entry_hash = EntryHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ── Externs ──────────────────────────────────────────────────────────

#[hdk_extern]
pub fn register_dependency(dep: DependencyIdentity) -> ExternResult<Record> {
    // Duplicate check: reject if ID already registered
    let id_tag = format!("dep:{}", dep.id);
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::DependencyById)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Dependency '{}' is already registered",
            dep.id
        ))));
    }

    let action_hash =
        create_entry(&EntryTypes::DependencyIdentity(dep.clone()))?;
    let entry_hash = hash_entry(&dep)?;

    // Link: all_deps → dependency
    create_link(
        anchor_hash("all_deps")?,
        entry_hash.clone(),
        LinkTypes::AllDependencies,
        (),
    )?;

    // Link: eco:{ecosystem} → dependency
    let eco_tag = format!("eco:{}", dep.ecosystem);
    create_link(
        anchor_hash(&eco_tag)?,
        entry_hash.clone(),
        LinkTypes::EcosystemToDependency,
        (),
    )?;

    // Link: maint:{did} → dependency
    let maint_tag = format!("maint:{}", dep.maintainer_did);
    create_link(
        anchor_hash(&maint_tag)?,
        entry_hash.clone(),
        LinkTypes::MaintainerToDependency,
        (),
    )?;

    // Link: dep:{id} → dependency (O(1) lookup)
    let id_tag = format!("dep:{}", dep.id);
    create_link(
        anchor_hash(&id_tag)?,
        entry_hash,
        LinkTypes::DependencyById,
        (),
    )?;

    let _ = emit_signal(&RegistrySignal::DependencyRegistered {
        dependency_id: dep.id.clone(),
        name: dep.name.clone(),
        ecosystem: dep.ecosystem.to_string(),
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch newly created dependency".into()
        )))
}

#[hdk_extern]
pub fn update_dependency(input: UpdateDependencyInput) -> ExternResult<Record> {
    // Author-only: verify caller is the maintainer
    require_author(&input.dependency.maintainer_did)?;

    let action_hash = update_entry(
        input.original_action_hash,
        &EntryTypes::DependencyIdentity(input.dependency.clone()),
    )?;

    let _ = emit_signal(&RegistrySignal::DependencyUpdated {
        dependency_id: input.dependency.id.clone(),
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch updated dependency".into()
        )))
}

#[hdk_extern]
pub fn get_dependency(id: String) -> ExternResult<Option<Record>> {
    let id_tag = format!("dep:{}", id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::DependencyById)?,
        GetStrategy::default(),
    )?;

    match links.first() {
        Some(link) => {
            let entry_hash =
                EntryHash::try_from(link.target.clone()).map_err(|_| {
                    wasm_error!(WasmErrorInner::Guest(
                        "Invalid link target".into()
                    ))
                })?;
            get(entry_hash, GetOptions::default())
        }
        None => Ok(None),
    }
}

#[hdk_extern]
pub fn get_all_dependencies(_: ()) -> ExternResult<Vec<Record>> {
    resolve_links(anchor_hash("all_deps")?, LinkTypes::AllDependencies)
}

#[hdk_extern]
pub fn get_dependencies_by_ecosystem(
    ecosystem: String,
) -> ExternResult<Vec<Record>> {
    let eco_tag = format!("eco:{}", ecosystem);
    resolve_links(anchor_hash(&eco_tag)?, LinkTypes::EcosystemToDependency)
}

#[hdk_extern]
pub fn get_maintainer_dependencies(did: String) -> ExternResult<Vec<Record>> {
    let maint_tag = format!("maint:{}", did);
    resolve_links(
        anchor_hash(&maint_tag)?,
        LinkTypes::MaintainerToDependency,
    )
}

#[hdk_extern]
pub fn verify_dependency(id: String) -> ExternResult<Record> {
    let record = get_dependency(id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest(format!(
            "Dependency '{}' not found",
            id
        ))
    ))?;

    let mut dep: DependencyIdentity = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize dependency: {}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Record has no entry".into()
        )))?;

    // Author-only: verify caller is the maintainer
    require_author(&dep.maintainer_did)?;

    dep.verified = true;

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::DependencyIdentity(dep),
    )?;

    let _ = emit_signal(&RegistrySignal::DependencyVerified {
        dependency_id: id,
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch verified dependency".into()
        )))
}

// ── Batch Registration ──────────────────────────────────────────────

#[hdk_extern]
pub fn bulk_register_dependencies(
    deps: Vec<DependencyIdentity>,
) -> ExternResult<BulkRegisterResult> {
    let mut registered = Vec::new();
    let mut skipped = Vec::new();

    for dep in deps {
        let id_tag = format!("dep:{}", dep.id);
        let existing = get_links(
            LinkQuery::try_new(anchor_hash(&id_tag)?, LinkTypes::DependencyById)?,
            GetStrategy::default(),
        )?;
        if !existing.is_empty() {
            skipped.push(dep.id.clone());
            continue;
        }

        let action_hash =
            create_entry(&EntryTypes::DependencyIdentity(dep.clone()))?;
        let entry_hash = hash_entry(&dep)?;

        create_link(
            anchor_hash("all_deps")?,
            entry_hash.clone(),
            LinkTypes::AllDependencies,
            (),
        )?;

        let eco_tag = format!("eco:{}", dep.ecosystem);
        create_link(
            anchor_hash(&eco_tag)?,
            entry_hash.clone(),
            LinkTypes::EcosystemToDependency,
            (),
        )?;

        let maint_tag = format!("maint:{}", dep.maintainer_did);
        create_link(
            anchor_hash(&maint_tag)?,
            entry_hash.clone(),
            LinkTypes::MaintainerToDependency,
            (),
        )?;

        create_link(
            anchor_hash(&id_tag)?,
            entry_hash,
            LinkTypes::DependencyById,
            (),
        )?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            registered.push(record);
        }
    }

    if !registered.is_empty() {
        let _ = emit_signal(&RegistrySignal::DependencyRegistered {
            dependency_id: format!("batch:{}", registered.len()),
            name: format!("{} dependencies", registered.len()),
            ecosystem: "Mixed".to_string(),
        });
    }

    Ok(BulkRegisterResult {
        registered,
        skipped,
    })
}

// ── Paginated Queries ───────────────────────────────────────────────

#[hdk_extern]
pub fn get_all_dependencies_paginated(
    input: PaginationInput,
) -> ExternResult<PaginatedDependencies> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("all_deps")?,
            LinkTypes::AllDependencies,
        )?,
        GetStrategy::default(),
    )?;
    let total = links.len() as u64;

    let page_links: Vec<_> = links
        .into_iter()
        .skip(input.offset as usize)
        .take(input.limit as usize)
        .collect();

    let mut items = Vec::new();
    for link in page_links {
        let entry_hash = EntryHash::try_from(link.target).map_err(|_| {
            wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
        })?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            items.push(record);
        }
    }

    Ok(PaginatedDependencies {
        items,
        total,
        offset: input.offset,
        limit: input.limit,
    })
}
