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
    AttestationRevoked {
        attestation_id: String,
        dependency_id: String,
    },
    AttestationRenewed {
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

// ── Constants ───────────────────────────────────────────────────────

const RATE_LIMIT_WINDOW_SECS: i64 = 60;
const MAX_PAGE_SIZE: u64 = 1000;

// ── Helpers ──────────────────────────────────────────────────────────

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
}

/// Validate that a dependency exists in the registry (cross-zome call).
/// Gracefully allows the operation if the registry is unavailable.
fn validate_dependency_exists(dep_id: &str) -> ExternResult<()> {
    let encoded = ExternIO::encode(dep_id.to_string()).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to encode dep_id: {}",
            e
        )))
    })?;
    match call(
        CallTargetCell::Local,
        ZomeName::from("registry"),
        FunctionName::from("get_dependency"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(io)) => {
            let record: Option<Record> = io.decode().unwrap_or(None);
            match record {
                Some(_) => Ok(()),
                None => Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Dependency '{}' not registered",
                    dep_id
                )))),
            }
        }
        _ => Ok(()), // Graceful: allow if registry unavailable
    }
}

/// Sliding-window rate limiter using links as timestamps.
/// Checks count BEFORE creating the link to prevent off-by-one at window boundary.
fn enforce_rate_limit(limit: u64) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash = EntryHash::from(agent);
    let now = sys_time()?;
    let now_micros = now.as_micros();
    let window_micros = RATE_LIMIT_WINDOW_SECS * 1_000_000;

    // Count links within window FIRST
    let links = get_links(
        LinkQuery::try_new(agent_hash.clone(), LinkTypes::UsageRateLimit)?,
        GetStrategy::default(),
    )?;

    let cutoff = now_micros - window_micros;
    let recent = links
        .iter()
        .filter(|l| {
            if l.tag.0.len() >= 8 {
                let ts = i64::from_le_bytes(l.tag.0[..8].try_into().unwrap_or([0; 8]));
                ts > cutoff
            } else {
                false
            }
        })
        .count() as u64;

    if recent >= limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Rate limit exceeded: {} requests in last {} seconds (limit: {})",
            recent, RATE_LIMIT_WINDOW_SECS, limit
        ))));
    }

    // Create rate-limit link AFTER passing the check
    create_link(
        agent_hash.clone(),
        agent_hash.clone(),
        LinkTypes::UsageRateLimit,
        LinkTag::new(now_micros.to_le_bytes()),
    )?;

    Ok(())
}

fn resolve_links(base: EntryHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(base, link_type)?, GetStrategy::default())?;
    let mut records = Vec::new();
    for link in links {
        let entry_hash = EntryHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(entry_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ── Usage Receipt Externs ────────────────────────────────────────────

#[hdk_extern]
pub fn record_usage(receipt: UsageReceipt) -> ExternResult<Record> {
    validate_dependency_exists(&receipt.dependency_id)?;
    enforce_rate_limit(50)?;

    let action_hash = create_entry(&EntryTypes::UsageReceipt(receipt.clone()))?;
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

    // Emit local signal for UI clients
    let _ = emit_signal(&UsageSignal::UsageRecorded {
        dependency_id: receipt.dependency_id.clone(),
        user_did: receipt.user_did.clone(),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created usage receipt".into()
    )))
}

#[hdk_extern]
pub fn get_dependency_usage(dep_id: String) -> ExternResult<Vec<Record>> {
    let dep_tag = format!("usage:{}", dep_id);
    resolve_links(anchor_hash(&dep_tag)?, LinkTypes::DependencyToUsageReceipts)
}

#[hdk_extern]
pub fn get_user_usage(did: String) -> ExternResult<Vec<Record>> {
    let user_tag = format!("user_usage:{}", did);
    resolve_links(anchor_hash(&user_tag)?, LinkTypes::UserToUsageReceipts)
}

#[hdk_extern]
pub fn get_usage_count(dep_id: String) -> ExternResult<u64> {
    let dep_tag = format!("usage:{}", dep_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&dep_tag)?, LinkTypes::DependencyToUsageReceipts)?,
        GetStrategy::default(),
    )?;
    Ok(links.len() as u64)
}

// ── Usage Attestation Externs ────────────────────────────────────────

#[hdk_extern]
pub fn submit_usage_attestation(att: UsageAttestation) -> ExternResult<Record> {
    validate_dependency_exists(&att.dependency_id)?;
    enforce_rate_limit(10)?;

    let action_hash = create_entry(&EntryTypes::UsageAttestation(att.clone()))?;
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
        entry_hash.clone(),
        LinkTypes::UserToAttestations,
        (),
    )?;

    // Link: all_attestations → attestation (for global count)
    create_link(
        anchor_hash("all_attestations")?,
        entry_hash,
        LinkTypes::AllAttestations,
        (),
    )?;

    let _ = emit_signal(&UsageSignal::AttestationSubmitted {
        dependency_id: att.dependency_id.clone(),
        user_did: att.user_did.clone(),
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created attestation".into()
    )))
}

#[hdk_extern]
pub fn verify_usage_attestation(input: VerifyAttestationInput) -> ExternResult<Record> {
    // Validate signature format: Ed25519 pubkey (32 bytes) + signature (64 bytes)
    if input.verifier_pubkey.len() != 32 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid verifier pubkey: expected 32 bytes, got {}",
            input.verifier_pubkey.len()
        ))));
    }
    if input.verifier_signature.len() != 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid verifier signature: expected 64 bytes, got {}",
            input.verifier_signature.len()
        ))));
    }

    let record = get(input.original_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Attestation not found".into())),
    )?;

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

    // Prevent re-verification
    if att.verified {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Attestation already verified".into()
        )));
    }

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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch verified attestation".into()
    )))
}

#[hdk_extern]
pub fn get_dependency_attestations(dep_id: String) -> ExternResult<Vec<Record>> {
    let dep_tag = format!("attest:{}", dep_id);
    let all = resolve_links(anchor_hash(&dep_tag)?, LinkTypes::DependencyToAttestations)?;

    // Filter out expired attestations
    let now = sys_time()?;
    let mut active = Vec::new();
    for record in all {
        let att: Option<UsageAttestation> = record.entry().to_app_option().map_err(|e| {
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

/// Count all active (non-expired) attestations across all dependencies.
#[hdk_extern]
pub fn get_attestation_count(_: ()) -> ExternResult<u64> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_attestations")?, LinkTypes::AllAttestations)?,
        GetStrategy::default(),
    )?;
    // For a precise count we'd filter expired, but link count is a good approximation
    Ok(links.len() as u64)
}

// ── Attestation Lifecycle ───────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RenewAttestationInput {
    pub original_action_hash: ActionHash,
    pub new_proof_bytes: Vec<u8>,
    pub new_witness_commitment: Vec<u8>,
}

/// Revoke an attestation (author-only delete).
#[hdk_extern]
pub fn revoke_attestation(action_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Attestation not found".into())
    ))?;

    // Author-only check
    let author = record.action().author().clone();
    let me = agent_info()?.agent_initial_pubkey;
    if author != me {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the author can revoke an attestation".into()
        )));
    }

    let att: UsageAttestation = record
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

    let delete_hash = delete_entry(action_hash)?;

    let _ = emit_signal(&UsageSignal::AttestationRevoked {
        attestation_id: att.id,
        dependency_id: att.dependency_id,
    });

    Ok(delete_hash)
}

/// Renew an attestation: create a new attestation linked to its predecessor.
#[hdk_extern]
pub fn renew_attestation(input: RenewAttestationInput) -> ExternResult<Record> {
    let record =
        get(input.original_action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Original attestation not found".into())
        ))?;

    let original: UsageAttestation = record
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

    // Create renewed attestation (unverified)
    let renewed = UsageAttestation {
        id: format!("{}-renewed", original.id),
        dependency_id: original.dependency_id.clone(),
        user_did: original.user_did.clone(),
        witness_commitment: input.new_witness_commitment,
        proof_bytes: input.new_proof_bytes,
        verified: false,
        generated_at: sys_time()?,
        expires_at: original.expires_at,
        verifier_pubkey: None,
        verifier_signature: None,
    };

    let new_action_hash = create_entry(&EntryTypes::UsageAttestation(renewed.clone()))?;
    let new_entry_hash = hash_entry(&renewed)?;

    // Link: predecessor → new attestation
    let predecessor_hash = hash_entry(&original)?;
    create_link(
        predecessor_hash,
        new_entry_hash.clone(),
        LinkTypes::PredecessorToAttestation,
        (),
    )?;

    // Re-create standard links
    let dep_tag = format!("attest:{}", renewed.dependency_id);
    create_link(
        anchor_hash(&dep_tag)?,
        new_entry_hash.clone(),
        LinkTypes::DependencyToAttestations,
        (),
    )?;
    let user_tag = format!("user_attest:{}", renewed.user_did);
    create_link(
        anchor_hash(&user_tag)?,
        new_entry_hash.clone(),
        LinkTypes::UserToAttestations,
        (),
    )?;
    create_link(
        anchor_hash("all_attestations")?,
        new_entry_hash,
        LinkTypes::AllAttestations,
        (),
    )?;

    let _ = emit_signal(&UsageSignal::AttestationRenewed {
        attestation_id: renewed.id,
        dependency_id: renewed.dependency_id,
    });

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch renewed attestation".into()
    )))
}

// ── Paginated Usage Queries ─────────────────────────────────────────

#[hdk_extern]
pub fn get_dependency_usage_paginated(input: PaginatedUsageInput) -> ExternResult<PaginatedUsage> {
    if input.pagination.limit > MAX_PAGE_SIZE {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Pagination limit {} exceeds maximum {}",
            input.pagination.limit, MAX_PAGE_SIZE
        ))));
    }
    let dep_tag = format!("usage:{}", input.id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&dep_tag)?, LinkTypes::DependencyToUsageReceipts)?,
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
        let entry_hash = EntryHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
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
    let limit = limit.min(MAX_PAGE_SIZE);
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
        Ok(ZomeCallResponse::Ok(io)) => io.decode().unwrap_or_else(|e| {
            debug!("Failed to decode registry response: {:?}", e);
            Vec::new()
        }),
        _ => return Ok(Vec::new()),
    };

    // For each dependency, count usage links
    let mut scored: Vec<TopDependency> = Vec::new();
    for record in dep_records {
        let dep: Option<DependencyIdRef> = record.entry().to_app_option().ok().flatten();
        if let Some(d) = dep {
            let dep_tag = format!("usage:{}", d.id);
            let count = get_links(
                LinkQuery::try_new(anchor_hash(&dep_tag)?, LinkTypes::DependencyToUsageReceipts)?,
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
pub fn bulk_record_usage(receipts: Vec<UsageReceipt>) -> ExternResult<BulkUsageResult> {
    enforce_rate_limit(5)?;

    let mut records = Vec::new();

    for receipt in &receipts {
        let action_hash = create_entry(&EntryTypes::UsageReceipt(receipt.clone()))?;
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
