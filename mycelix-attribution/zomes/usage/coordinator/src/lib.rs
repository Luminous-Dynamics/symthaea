// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    pub has_more: bool,
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
const MAX_RENEWAL_DEPTH: u32 = 5;

// ── Helpers ──────────────────────────────────────────────────────────
// NOTE: anchor_hash/resolve_links are intentionally duplicated across
// registry, usage, and reciprocity coordinators. Each zome uses its own
// Anchor/LinkTypes from its integrity crate, preventing shared extraction.

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
        other => {
            debug!(
                "validate_dependency_exists: registry call returned non-Ok response: {:?}",
                other
            );
            Ok(()) // Graceful: allow if registry unavailable
        }
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

    // Authorization: only the attestation author can submit verification results.
    // The author runs the off-chain verifier binary, which produces a signed verdict,
    // then submits the verifier's pubkey + signature back to the DHT.
    let author = record.action().author().clone();
    let me = agent_info()?.agent_initial_pubkey;
    if author != me {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the attestation author can submit verification results".into()
        )));
    }

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

    let now = sys_time()?;
    let mut active = 0u64;
    for link in links {
        let entry_hash = match EntryHash::try_from(link.target) {
            Ok(h) => h,
            Err(_) => {
                active += 1; // Count if we can't resolve (link exists)
                continue;
            }
        };
        match get(entry_hash, GetOptions::default())? {
            Some(record) => {
                let att: Option<UsageAttestation> = record.entry().to_app_option().ok().flatten();
                match att {
                    Some(a) if a.expires_at.is_some_and(|exp| exp < now) => {
                        // Expired — don't count
                    }
                    _ => active += 1,
                }
            }
            None => active += 1, // Link exists but entry not found — count it
        }
    }
    Ok(active)
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

    // Enforce renewal depth limit by counting "-renewed" suffixes
    let depth = original.id.matches("-renewed").count() as u32;
    if depth >= MAX_RENEWAL_DEPTH {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Renewal depth limit exceeded: max {} renewals per attestation chain",
            MAX_RENEWAL_DEPTH
        ))));
    }

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

    let has_more = input.pagination.offset + input.pagination.limit < total;
    Ok(PaginatedUsage {
        items,
        total,
        offset: input.pagination.offset,
        limit: input.pagination.limit,
        has_more,
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

/// Pure computation of has_more flag (testable without HDK).
#[cfg(test)]
fn compute_has_more(offset: u64, limit: u64, total: u64) -> bool {
    offset + limit < total
}

/// Pure validation of pagination limit (testable without HDK).
#[cfg(test)]
fn validate_page_limit(limit: u64) -> Result<(), String> {
    if limit > MAX_PAGE_SIZE {
        Err(format!(
            "Pagination limit {} exceeds maximum {}",
            limit, MAX_PAGE_SIZE
        ))
    } else {
        Ok(())
    }
}

/// Pure validation of verifier pubkey length.
#[cfg(test)]
fn validate_verifier_pubkey_len(len: usize) -> Result<(), String> {
    if len != 32 {
        Err(format!(
            "Invalid verifier pubkey: expected 32 bytes, got {}",
            len
        ))
    } else {
        Ok(())
    }
}

/// Pure validation of verifier signature length.
#[cfg(test)]
fn validate_verifier_signature_len(len: usize) -> Result<(), String> {
    if len != 64 {
        Err(format!(
            "Invalid verifier signature: expected 64 bytes, got {}",
            len
        ))
    } else {
        Ok(())
    }
}

/// Pure computation of renewal depth from attestation ID.
#[cfg(test)]
fn compute_renewal_depth(id: &str) -> u32 {
    id.matches("-renewed").count() as u32
}

/// Check whether renewal depth exceeds the limit.
#[cfg(test)]
fn exceeds_renewal_depth(id: &str) -> bool {
    compute_renewal_depth(id) >= MAX_RENEWAL_DEPTH
}

// ── Unit Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Constants ───────────────────────────────────────────────────

    #[test]
    fn test_max_page_size_is_1000() {
        assert_eq!(MAX_PAGE_SIZE, 1000);
    }

    #[test]
    fn test_rate_limit_window_is_60_seconds() {
        assert_eq!(RATE_LIMIT_WINDOW_SECS, 60);
    }

    #[test]
    fn test_max_renewal_depth_is_5() {
        assert_eq!(MAX_RENEWAL_DEPTH, 5);
    }

    // ── Pagination validation ───────────────────────────────────────

    #[test]
    fn test_validate_page_limit_within_max() {
        assert!(validate_page_limit(500).is_ok());
    }

    #[test]
    fn test_validate_page_limit_at_max() {
        assert!(validate_page_limit(MAX_PAGE_SIZE).is_ok());
    }

    #[test]
    fn test_validate_page_limit_exceeds_max() {
        let err = validate_page_limit(MAX_PAGE_SIZE + 1).unwrap_err();
        assert!(err.contains("exceeds maximum"), "error: {}", err);
    }

    #[test]
    fn test_validate_page_limit_zero() {
        assert!(validate_page_limit(0).is_ok());
    }

    // ── has_more computation ────────────────────────────────────────

    #[test]
    fn test_has_more_true_when_remaining() {
        assert!(compute_has_more(0, 10, 20));
    }

    #[test]
    fn test_has_more_false_at_exact_end() {
        assert!(!compute_has_more(10, 10, 20));
    }

    #[test]
    fn test_has_more_false_past_end() {
        assert!(!compute_has_more(15, 10, 20));
    }

    #[test]
    fn test_has_more_false_empty() {
        assert!(!compute_has_more(0, 10, 0));
    }

    // ── Verifier field validation ───────────────────────────────────

    #[test]
    fn test_verifier_pubkey_valid_32_bytes() {
        assert!(validate_verifier_pubkey_len(32).is_ok());
    }

    #[test]
    fn test_verifier_pubkey_wrong_size() {
        assert!(validate_verifier_pubkey_len(31).is_err());
        assert!(validate_verifier_pubkey_len(33).is_err());
        assert!(validate_verifier_pubkey_len(0).is_err());
    }

    #[test]
    fn test_verifier_signature_valid_64_bytes() {
        assert!(validate_verifier_signature_len(64).is_ok());
    }

    #[test]
    fn test_verifier_signature_wrong_size() {
        assert!(validate_verifier_signature_len(63).is_err());
        assert!(validate_verifier_signature_len(65).is_err());
        assert!(validate_verifier_signature_len(0).is_err());
    }

    // ── Renewal depth ───────────────────────────────────────────────

    #[test]
    fn test_renewal_depth_zero() {
        assert_eq!(compute_renewal_depth("attest-001"), 0);
    }

    #[test]
    fn test_renewal_depth_one() {
        assert_eq!(compute_renewal_depth("attest-001-renewed"), 1);
    }

    #[test]
    fn test_renewal_depth_multiple() {
        assert_eq!(
            compute_renewal_depth("attest-001-renewed-renewed-renewed"),
            3
        );
    }

    #[test]
    fn test_renewal_depth_at_limit() {
        let id = "attest-001-renewed-renewed-renewed-renewed-renewed";
        assert_eq!(compute_renewal_depth(id), 5);
        assert!(exceeds_renewal_depth(id));
    }

    #[test]
    fn test_renewal_depth_below_limit() {
        let id = "attest-001-renewed-renewed-renewed-renewed";
        assert_eq!(compute_renewal_depth(id), 4);
        assert!(!exceeds_renewal_depth(id));
    }

    #[test]
    fn test_renewal_depth_suffix_match() {
        // "-renewed" as substring match — "attest-renewedfoo" still contains "-renewed"
        assert_eq!(compute_renewal_depth("attest-renewedfoo"), 1);
        // Clean suffix
        assert_eq!(compute_renewal_depth("attest-renewed"), 1);
        // No match at all
        assert_eq!(compute_renewal_depth("attest-renew"), 0);
    }

    // ── Top dependency sorting ──────────────────────────────────────

    #[test]
    fn test_top_dependency_sort_descending() {
        let mut deps = vec![
            TopDependency {
                dependency_id: "a".into(),
                usage_count: 10,
            },
            TopDependency {
                dependency_id: "b".into(),
                usage_count: 50,
            },
            TopDependency {
                dependency_id: "c".into(),
                usage_count: 30,
            },
        ];
        deps.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        assert_eq!(deps[0].dependency_id, "b");
        assert_eq!(deps[1].dependency_id, "c");
        assert_eq!(deps[2].dependency_id, "a");
    }

    #[test]
    fn test_top_dependency_truncate() {
        let mut deps: Vec<TopDependency> = (0..20)
            .map(|i| TopDependency {
                dependency_id: format!("dep-{}", i),
                usage_count: i,
            })
            .collect();
        deps.sort_by(|a, b| b.usage_count.cmp(&a.usage_count));
        let limit = 5u64.min(MAX_PAGE_SIZE);
        deps.truncate(limit as usize);
        assert_eq!(deps.len(), 5);
        assert_eq!(deps[0].usage_count, 19);
    }

    // ── Serde roundtrip tests ───────────────────────────────────────

    #[test]
    fn test_pagination_input_serde_roundtrip() {
        let input = PaginationInput {
            offset: 42,
            limit: 100,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: PaginationInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.offset, 42);
        assert_eq!(back.limit, 100);
    }

    #[test]
    fn test_top_dependency_serde_roundtrip() {
        let td = TopDependency {
            dependency_id: "crate:serde".into(),
            usage_count: 1234,
        };
        let json = serde_json::to_string(&td).unwrap();
        let back: TopDependency = serde_json::from_str(&json).unwrap();
        assert_eq!(back.dependency_id, "crate:serde");
        assert_eq!(back.usage_count, 1234);
    }

    #[test]
    fn test_bulk_usage_result_serde_roundtrip() {
        let result = BulkUsageResult {
            recorded: 5,
            records: vec![],
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: BulkUsageResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.recorded, 5);
        assert!(back.records.is_empty());
    }

    #[test]
    fn test_usage_signal_serde_roundtrip() {
        let signals = vec![
            UsageSignal::UsageRecorded {
                dependency_id: "dep1".into(),
                user_did: "did:mycelix:user1".into(),
            },
            UsageSignal::AttestationSubmitted {
                dependency_id: "dep1".into(),
                user_did: "did:mycelix:user1".into(),
            },
            UsageSignal::AttestationVerified {
                attestation_id: "att1".into(),
                dependency_id: "dep1".into(),
            },
            UsageSignal::AttestationRevoked {
                attestation_id: "att1".into(),
                dependency_id: "dep1".into(),
            },
            UsageSignal::AttestationRenewed {
                attestation_id: "att1".into(),
                dependency_id: "dep1".into(),
            },
        ];
        for sig in signals {
            let json = serde_json::to_string(&sig).unwrap();
            let back: UsageSignal = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&back).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[test]
    fn test_paginated_usage_input_serde_roundtrip() {
        let input = PaginatedUsageInput {
            id: "crate:serde".into(),
            pagination: PaginationInput {
                offset: 0,
                limit: 50,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: PaginatedUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "crate:serde");
        assert_eq!(back.pagination.limit, 50);
    }
}
