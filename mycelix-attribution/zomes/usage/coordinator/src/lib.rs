use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use usage_integrity::*;

/// Mirror type for deserializing DependencyIdentity from registry zome.
/// Only the `id` field is needed for top-N queries.
/// Must implement TryFrom<SerializedBytes> for to_app_option().
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DependencyIdRef {
    id: String,
}

holochain_serialized_bytes::holochain_serial!(DependencyIdRef);

// ── Signals ──────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum UsageSignal {
    UsageRecorded {
        dependency_id: String,
        user_did: String,
    },
    AttestationSubmitted {
        dependency_id: String,
        user_did: String,
    },
    AttestationVerified {
        attestation_id: String,
        dependency_id: String,
    },
}

// ── Input Types ──────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VerifyAttestationInput {
    pub original_action_hash: ActionHash,
    pub verifier_pubkey: Vec<u8>,
    pub verifier_signature: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginationInput {
    pub offset: u64,
    pub limit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedUsage {
    pub items: Vec<Record>,
    pub total: u64,
    pub offset: u64,
    pub limit: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TopDependency {
    pub dependency_id: String,
    pub usage_count: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedUsageInput {
    pub id: String,
    pub pagination: PaginationInput,
}

// ── Init ─────────────────────────────────────────────────────────────

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

// ── Helpers ──────────────────────────────────────────────────────────

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
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

// ── Usage Receipt Externs ────────────────────────────────────────────

#[hdk_extern]
pub fn record_usage(receipt: UsageReceipt) -> ExternResult<Record> {
    let action_hash =
        create_entry(&EntryTypes::UsageReceipt(receipt.clone()))?;
    let entry_hash = hash_entry(&receipt)?;

    // Link: usage:{dep_id} → receipt
    let dep_tag = format!("usage:{}", receipt.dependency_id);
    create_link(
        anchor_hash(&dep_tag)?,
        entry_hash.clone(),
        LinkTypes::DependencyToUsageReceipts,
        (),
    )?;

    // Link: user_usage:{did} → receipt
    let user_tag = format!("user_usage:{}", receipt.user_did);
    create_link(
        anchor_hash(&user_tag)?,
        entry_hash,
        LinkTypes::UserToUsageReceipts,
        (),
    )?;

    let _ = emit_signal(&UsageSignal::UsageRecorded {
        dependency_id: receipt.dependency_id.clone(),
        user_did: receipt.user_did.clone(),
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch newly created usage receipt".into()
        )))
}

#[hdk_extern]
pub fn get_dependency_usage(dep_id: String) -> ExternResult<Vec<Record>> {
    let dep_tag = format!("usage:{}", dep_id);
    resolve_links(
        anchor_hash(&dep_tag)?,
        LinkTypes::DependencyToUsageReceipts,
    )
}

#[hdk_extern]
pub fn get_user_usage(did: String) -> ExternResult<Vec<Record>> {
    let user_tag = format!("user_usage:{}", did);
    resolve_links(
        anchor_hash(&user_tag)?,
        LinkTypes::UserToUsageReceipts,
    )
}

#[hdk_extern]
pub fn get_usage_count(dep_id: String) -> ExternResult<u64> {
    let dep_tag = format!("usage:{}", dep_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&dep_tag)?,
            LinkTypes::DependencyToUsageReceipts,
        )?,
        GetStrategy::default(),
    )?;
    Ok(links.len() as u64)
}

// ── Usage Attestation Externs ────────────────────────────────────────

#[hdk_extern]
pub fn submit_usage_attestation(
    att: UsageAttestation,
) -> ExternResult<Record> {
    let action_hash =
        create_entry(&EntryTypes::UsageAttestation(att.clone()))?;
    let entry_hash = hash_entry(&att)?;

    // Link: attest:{dep_id} → attestation
    let dep_tag = format!("attest:{}", att.dependency_id);
    create_link(
        anchor_hash(&dep_tag)?,
        entry_hash.clone(),
        LinkTypes::DependencyToAttestations,
        (),
    )?;

    // Link: user_attest:{did} → attestation
    let user_tag = format!("user_attest:{}", att.user_did);
    create_link(
        anchor_hash(&user_tag)?,
        entry_hash,
        LinkTypes::UserToAttestations,
        (),
    )?;

    let _ = emit_signal(&UsageSignal::AttestationSubmitted {
        dependency_id: att.dependency_id.clone(),
        user_did: att.user_did.clone(),
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch newly created attestation".into()
        )))
}

#[hdk_extern]
pub fn verify_usage_attestation(
    input: VerifyAttestationInput,
) -> ExternResult<Record> {
    let record = get(
        input.original_action_hash.clone(),
        GetOptions::default(),
    )?
    .ok_or(wasm_error!(WasmErrorInner::Guest(
        "Attestation not found".into()
    )))?;

    let mut att: UsageAttestation = record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize attestation: {}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Record has no entry".into()
        )))?;

    att.verified = true;
    att.verifier_pubkey = Some(input.verifier_pubkey);
    att.verifier_signature = Some(input.verifier_signature);

    let action_hash = update_entry(
        input.original_action_hash,
        &EntryTypes::UsageAttestation(att.clone()),
    )?;

    let _ = emit_signal(&UsageSignal::AttestationVerified {
        attestation_id: att.id.clone(),
        dependency_id: att.dependency_id.clone(),
    });

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not fetch verified attestation".into()
        )))
}

#[hdk_extern]
pub fn get_dependency_attestations(
    dep_id: String,
) -> ExternResult<Vec<Record>> {
    let dep_tag = format!("attest:{}", dep_id);
    let all = resolve_links(
        anchor_hash(&dep_tag)?,
        LinkTypes::DependencyToAttestations,
    )?;

    // Filter out expired attestations
    let now = sys_time()?;
    let mut active = Vec::new();
    for record in all {
        let att: Option<UsageAttestation> = record
            .entry()
            .to_app_option()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to deserialize attestation: {}",
                    e
                )))
            })?;
        match att {
            Some(a) if a.expires_at.is_some_and(|exp| exp < now) => {
                // Expired — skip
            }
            _ => active.push(record),
        }
    }
    Ok(active)
}

// ── Paginated Usage Queries ─────────────────────────────────────────

#[hdk_extern]
pub fn get_dependency_usage_paginated(
    input: PaginatedUsageInput,
) -> ExternResult<PaginatedUsage> {
    let dep_tag = format!("usage:{}", input.id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&dep_tag)?,
            LinkTypes::DependencyToUsageReceipts,
        )?,
        GetStrategy::default(),
    )?;
    let total = links.len() as u64;

    let page_links: Vec<_> = links
        .into_iter()
        .skip(input.pagination.offset as usize)
        .take(input.pagination.limit as usize)
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

    Ok(PaginatedUsage {
        items,
        total,
        offset: input.pagination.offset,
        limit: input.pagination.limit,
    })
}

// ── Top-N Most Used Dependencies ────────────────────────────────────

#[hdk_extern]
pub fn get_top_dependencies(limit: u64) -> ExternResult<Vec<TopDependency>> {
    // Get all dependencies via cross-zome call to registry
    let encoded = ExternIO::encode(()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to encode payload: {}",
            e
        )))
    })?;

    let dep_records: Vec<Record> = match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_all_dependencies"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or_default(),
        _ => return Ok(Vec::new()),
    };

    // For each dependency, count usage links
    let mut scored: Vec<TopDependency> = Vec::new();
    for record in dep_records {
        let dep: Option<DependencyIdRef> = record
            .entry()
            .to_app_option()
            .ok()
            .flatten();
        if let Some(d) = dep {
            let dep_tag = format!("usage:{}", d.id);
            let count = get_links(
                LinkQuery::try_new(
                    anchor_hash(&dep_tag)?,
                    LinkTypes::DependencyToUsageReceipts,
                )?,
                GetStrategy::default(),
            )?
            .len() as u64;

            if count > 0 {
                scored.push(TopDependency {
                    dependency_id: d.id,
                    usage_count: count,
                });
            }
        }
    }

    // Sort descending by usage count
    scored.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
    scored.truncate(limit as usize);

    Ok(scored)
}

// ── Maintainer Notification ─────────────────────────────────────────

// ── Maintainer Notification ─────────────────────────────────────────
// Signals are emitted locally via record_usage's emit_signal(UsageRecorded).
// All connected UI clients receive signals and can filter by dependency_id
// to show notifications to the maintainer.
//
// For remote/push notifications to offline maintainers, a future integration
// with the identity hApp would resolve DID→AgentPubKey for remote_signal().
