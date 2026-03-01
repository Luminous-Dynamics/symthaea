use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use usage_integrity::*;

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
    resolve_links(
        anchor_hash(&dep_tag)?,
        LinkTypes::DependencyToAttestations,
    )
}
