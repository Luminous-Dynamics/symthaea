use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use usage_integrity::*;

/// Mirror type for deserializing DependencyIdentity from registry zome.
/// Must implement TryFrom<SerializedBytes> for to_app_option().
#[derive(Serialize, Deserialize, Debug, Clone)]
struct DependencyIdRef {
    id: String,
}

holochain_serialized_bytes::holochain_serial!(DependencyIdRef);

/// Mirror type for extracting maintainer_did from registry entries.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct MaintainerRef {
    maintainer_did: String,
}

holochain_serialized_bytes::holochain_serial!(MaintainerRef);

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

    // Notify maintainer (best-effort: emits local signal, attempts identity bridge)
    notify_maintainer(&receipt.dependency_id, &receipt.user_did);

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

// ── Batch Usage Recording ───────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BulkUsageResult {
    pub recorded: u64,
    pub records: Vec<Record>,
}

#[hdk_extern]
pub fn bulk_record_usage(
    receipts: Vec<UsageReceipt>,
) -> ExternResult<BulkUsageResult> {
    let mut records = Vec::new();

    for receipt in &receipts {
        let action_hash =
            create_entry(&EntryTypes::UsageReceipt(receipt.clone()))?;
        let entry_hash = hash_entry(receipt)?;

        let dep_tag = format!("usage:{}", receipt.dependency_id);
        create_link(
            anchor_hash(&dep_tag)?,
            entry_hash.clone(),
            LinkTypes::DependencyToUsageReceipts,
            (),
        )?;

        let user_tag = format!("user_usage:{}", receipt.user_did);
        create_link(
            anchor_hash(&user_tag)?,
            entry_hash,
            LinkTypes::UserToUsageReceipts,
            (),
        )?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    let recorded = records.len() as u64;

    if recorded > 0 {
        let _ = emit_signal(&UsageSignal::UsageRecorded {
            dependency_id: format!("batch:{}", recorded),
            user_did: receipts
                .first()
                .map(|r| r.user_did.clone())
                .unwrap_or_default(),
        });
    }

    Ok(BulkUsageResult { recorded, records })
}

// ── Maintainer Notification (Identity Bridge) ──────────────────────
//
// When running inside the unified hApp (mycelix-unified-happ.yaml),
// the identity DNA is available via CallTargetCell::OtherRole("identity").
// This allows DID→AgentPubKey resolution for remote signals.
//
// Notification flow:
//   1. record_usage() emits local UsageRecorded signal (current behavior)
//   2. notify_maintainer() cross-zome calls registry to get maintainer_did
//   3. Cross-role calls identity to resolve DID→AgentPubKey
//   4. remote_signal() to maintainer's agent
//
// Steps 2-4 are best-effort and silently fail when:
//   - Running as standalone hApp (no identity role)
//   - Maintainer's agent is offline
//   - DID is not registered in identity hApp

/// Attempt to notify maintainer via remote signal.
/// Best-effort: silently ignored if identity bridge is unavailable.
fn notify_maintainer(dependency_id: &str, user_did: &str) {
    // Step 1: Get maintainer_did from registry
    let Ok(encoded) = ExternIO::encode(dependency_id.to_string()) else {
        return;
    };
    let maintainer_did = match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_dependency"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => {
            let record: Option<Record> = io.decode().ok().flatten();
            record.and_then(|r| {
                let m: Option<MaintainerRef> =
                    r.entry().to_app_option().ok().flatten();
                m.map(|mr| mr.maintainer_did)
            })
        }
        _ => None,
    };

    let Some(did) = maintainer_did else { return };

    // Step 2: Resolve DID→AgentPubKey via identity bridge
    // This call goes to OtherRole("identity") in the unified hApp.
    // If identity role is not present (standalone deployment), this fails silently.
    let Ok(did_payload) = ExternIO::encode(did) else { return };
    let _agent_key: Option<AgentPubKey> = match call(
        CallTargetCell::OtherRole("identity".into()),
        ZomeName::from("identity"),
        FunctionName::from("resolve_did_to_agent"),
        None,
        did_payload,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => io.decode().ok().flatten(),
        _ => None, // Identity bridge not available — standalone mode
    };

    // Step 3: Send remote signal (when identity bridge is wired)
    // if let Some(agent_key) = _agent_key {
    //     let signal = UsageSignal::UsageRecorded {
    //         dependency_id: dependency_id.to_string(),
    //         user_did: user_did.to_string(),
    //     };
    //     let _ = remote_signal(ExternIO::encode(signal).unwrap(), vec![agent_key]);
    // }

    // For now: local signal only (all UI clients filter by dependency_id)
    let _ = emit_signal(&UsageSignal::UsageRecorded {
        dependency_id: dependency_id.to_string(),
        user_did: user_did.to_string(),
    });
}
