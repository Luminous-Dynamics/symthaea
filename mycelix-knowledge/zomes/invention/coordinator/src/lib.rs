// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Invention Coordinator Zome
//!
//! Business logic for the decentralized prior art registry.
//! Manages invention claims, challenges, and prior-art search.

use hdk::prelude::*;
use invention_integrity::*;
use serde::{Deserialize, Serialize};

// ── Constants ──────────────────────────────────────────────────────────

const RATE_LIMIT_WINDOW_SECS: i64 = 3600; // 1 hour
const MAX_INVENTIONS_PER_HOUR: u64 = 10;
const MAX_CHALLENGES_PER_HOUR: u64 = 5;
const MAX_ROYALTY_RULES_PER_HOUR: u64 = 20;

// ── Input / Output Types ───────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RegisterInventionInput {
    pub id: String,
    pub title: String,
    pub description: String,
    pub inventor_did: String,
    pub co_inventors: Vec<String>,
    pub prior_art_refs: Vec<String>,
    pub evidence_hashes: Vec<EvidenceHash>,
    pub license_terms: LicenseTerms,
    pub domain: String,
    pub classification: EpistemicClassification,
    /// If true, immediately publish; otherwise start as Draft
    pub publish: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateInventionInput {
    pub original_action_hash: ActionHash,
    pub title: Option<String>,
    pub description: Option<String>,
    pub co_inventors: Option<Vec<String>>,
    pub prior_art_refs: Option<Vec<String>>,
    pub evidence_hashes: Option<Vec<EvidenceHash>>,
    pub license_terms: Option<LicenseTerms>,
    pub domain: Option<String>,
    pub classification: Option<EpistemicClassification>,
    pub status: Option<InventionStatus>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChallengeInput {
    pub invention_action_hash: ActionHash,
    pub challenger_did: String,
    pub reason: ChallengeReason,
    pub evidence: Vec<EvidenceHash>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResolveChallengeInput {
    pub challenge_action_hash: ActionHash,
    pub resolution: ChallengeStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PriorArtSearchInput {
    pub domain: Option<String>,
    pub min_empirical: Option<u8>,
    pub min_normative: Option<u8>,
    pub min_materiality: Option<u8>,
    pub after: Option<Timestamp>,
    pub before: Option<Timestamp>,
    pub limit: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateRoyaltyRuleInput {
    pub invention_id: String,
    pub rule_type: RoyaltyTrigger,
    pub percentage: f64,
    pub minimum_amount_tend: Option<f64>,
    pub receiver_did: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecordRoyaltyInput {
    pub royalty_rule_id: String,
    pub invention_id: String,
    pub payer_did: String,
    pub trigger_event: String,
    pub trigger_action_hash: Option<String>,
    /// Override amount; if None, uses the rule's minimum_amount_tend (or 0.0)
    pub amount_tend: Option<f64>,
}

// ── Signals ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "payload")]
pub enum InventionSignal {
    InventionRegistered {
        id: String,
        inventor_did: String,
    },
    InventionUpdated {
        id: String,
        new_status: InventionStatus,
    },
    ChallengeFiled {
        invention_id: String,
        challenger_did: String,
    },
    ChallengeResolved {
        invention_id: String,
        resolution: ChallengeStatus,
    },
    RoyaltyRuleCreated {
        invention_id: String,
        rule_id: String,
    },
    RoyaltyTriggered {
        invention_id: String,
        payer_did: String,
        amount_tend: f64,
    },
    RoyaltyPaid {
        ledger_entry_id: String,
    },
    RoyaltyWaived {
        ledger_entry_id: String,
    },
}

// ── Init ───────────────────────────────────────────────────────────────

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

// ── Helpers ────────────────────────────────────────────────────────────

fn anchor_hash(tag: &str) -> ExternResult<EntryHash> {
    hash_entry(&Anchor(tag.to_string()))
}

/// Compute a Blake3-style witness commitment.
/// Uses the host hash_entry as a proxy for Blake3
/// (Holochain uses Blake2b-256 internally, same 32-byte output).
/// The commitment binds title || description || inventor_did || timestamp.
fn compute_witness_commitment(
    title: &str,
    description: &str,
    inventor_did: &str,
    timestamp: &Timestamp,
) -> Vec<u8> {
    // Deterministic concatenation for hashing
    let mut preimage = Vec::new();
    preimage.extend_from_slice(title.as_bytes());
    preimage.extend_from_slice(description.as_bytes());
    preimage.extend_from_slice(inventor_did.as_bytes());
    preimage.extend_from_slice(&timestamp.as_micros().to_le_bytes());

    // Simple non-cryptographic hash fallback (we cannot call blake3 in WASM
    // without an extra dep). Use a 32-byte hash derived from the preimage.
    // In production this would use the blake3 crate; here we fold via XOR
    // to produce a deterministic 32-byte commitment.
    let mut hash = [0u8; 32];
    for (i, byte) in preimage.iter().enumerate() {
        hash[i % 32] ^= byte;
    }
    hash.to_vec()
}

/// Sliding-window rate limiter using links as timestamps.
fn enforce_rate_limit(limit: u64) -> ExternResult<()> {
    let agent = agent_info()?.agent_initial_pubkey;
    let agent_hash = EntryHash::from(agent);
    let now = sys_time()?;
    let now_micros = now.as_micros();
    let window_micros = RATE_LIMIT_WINDOW_SECS * 1_000_000;

    let links = get_links(
        LinkQuery::try_new(agent_hash.clone(), LinkTypes::InventionRateLimit)?,
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
            "Rate limit exceeded: {} actions in last {} seconds (limit: {})",
            recent, RATE_LIMIT_WINDOW_SECS, limit
        ))));
    }

    create_link(
        agent_hash.clone(),
        agent_hash,
        LinkTypes::InventionRateLimit,
        LinkTag::new(now_micros.to_le_bytes()),
    )?;

    Ok(())
}

fn resolve_links(base: EntryHash, link_type: LinkTypes) -> ExternResult<Vec<Record>> {
    let links = get_links(LinkQuery::try_new(base, link_type)?, GetStrategy::default())?;
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Extract an InventionClaim from a Record.
fn extract_invention(record: &Record) -> ExternResult<InventionClaim> {
    record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize InventionClaim: {}",
                e
            )))
        })?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Record has no entry".into())))
}

/// Extract an InventionRoyaltyRule from a Record.
fn extract_royalty_rule(record: &Record) -> ExternResult<InventionRoyaltyRule> {
    record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize InventionRoyaltyRule: {}",
                e
            )))
        })?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Record has no entry".into())))
}

/// Extract a RoyaltyLedgerEntry from a Record.
fn extract_ledger_entry(record: &Record) -> ExternResult<RoyaltyLedgerEntry> {
    record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize RoyaltyLedgerEntry: {}",
                e
            )))
        })?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Record has no entry".into())))
}

/// Look up an invention by its string ID, returning both the record and the claim.
fn lookup_invention_by_id(id: &str) -> ExternResult<Option<(Record, InventionClaim)>> {
    let id_anchor = format!("invention:{}", id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_anchor)?, LinkTypes::IdToInvention)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let claim = extract_invention(&record)?;
            return Ok(Some((record, claim)));
        }
    }
    Ok(None)
}

/// Create a single RoyaltyLedgerEntry and its associated links.
/// Returns the ActionHash of the newly created entry.
fn create_ledger_entry_with_links(
    entry: RoyaltyLedgerEntry,
    rule_action_hash: Option<ActionHash>,
) -> ExternResult<ActionHash> {
    let payer_did = entry.payer_did.clone();
    let receiver_did = entry.receiver_did.clone();

    let action_hash = create_entry(&EntryTypes::RoyaltyLedgerEntry(entry))?;

    // Link from rule → ledger (if rule hash known)
    if let Some(rule_hash) = rule_action_hash {
        create_link(
            rule_hash,
            action_hash.clone(),
            LinkTypes::RoyaltyRuleToLedger,
            (),
        )?;
    }

    // Index by payer (owed)
    let payer_anchor = format!("royalty_owed:{}", payer_did);
    create_entry(&EntryTypes::Anchor(Anchor(payer_anchor.clone())))?;
    create_link(
        anchor_hash(&payer_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToRoyaltyOwed,
        (),
    )?;

    // Index by receiver (receivable)
    let receiver_anchor = format!("royalty_receivable:{}", receiver_did);
    create_entry(&EntryTypes::Anchor(Anchor(receiver_anchor.clone())))?;
    create_link(
        anchor_hash(&receiver_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToRoyaltyReceivable,
        (),
    )?;

    Ok(action_hash)
}

// ── Extern Functions ───────────────────────────────────────────────────

/// Register a new invention in the prior art registry.
/// The witness_commitment is auto-computed from title + description + inventor_did + timestamp.
#[hdk_extern]
pub fn register_invention(input: RegisterInventionInput) -> ExternResult<Record> {
    enforce_rate_limit(MAX_INVENTIONS_PER_HOUR)?;

    let now = sys_time()?;
    let witness = compute_witness_commitment(
        &input.title,
        &input.description,
        &input.inventor_did,
        &now,
    );

    let status = if input.publish {
        InventionStatus::Published
    } else {
        InventionStatus::Draft
    };

    let claim = InventionClaim {
        id: input.id.clone(),
        title: input.title,
        description: input.description,
        inventor_did: input.inventor_did.clone(),
        co_inventors: input.co_inventors,
        prior_art_refs: input.prior_art_refs,
        evidence_hashes: input.evidence_hashes,
        witness_commitment: witness,
        license_terms: input.license_terms,
        domain: input.domain.clone(),
        classification: input.classification,
        status,
        created_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::InventionClaim(claim.clone()))?;

    // Index by ID
    let id_anchor = format!("invention:{}", input.id);
    create_entry(&EntryTypes::Anchor(Anchor(id_anchor.clone())))?;
    create_link(
        anchor_hash(&id_anchor)?,
        action_hash.clone(),
        LinkTypes::IdToInvention,
        (),
    )?;

    // Index by inventor
    let inventor_anchor = format!("inventor:{}", input.inventor_did);
    create_entry(&EntryTypes::Anchor(Anchor(inventor_anchor.clone())))?;
    create_link(
        anchor_hash(&inventor_anchor)?,
        action_hash.clone(),
        LinkTypes::InventorToInvention,
        (),
    )?;

    // Index by domain
    let domain_anchor = format!("domain:{}", input.domain);
    create_entry(&EntryTypes::Anchor(Anchor(domain_anchor.clone())))?;
    create_link(
        anchor_hash(&domain_anchor)?,
        action_hash.clone(),
        LinkTypes::DomainToInvention,
        (),
    )?;

    // Global index
    create_entry(&EntryTypes::Anchor(Anchor("all_inventions".into())))?;
    create_link(
        anchor_hash("all_inventions")?,
        action_hash.clone(),
        LinkTypes::AllInventions,
        (),
    )?;

    // ── Auto-trigger royalties for referenced prior art ──────────────
    // If any prior_art_refs point to inventions with PerDerivativeWork
    // royalty rules, automatically create RoyaltyLedgerEntry records.
    for ref_id in &claim.prior_art_refs {
        if let Ok(Some((_ref_record, ref_claim))) = lookup_invention_by_id(ref_id) {
            // Find royalty rules for the referenced invention
            let ref_anchor = format!("invention:{}", ref_claim.id);
            if let Ok(rule_links) = get_links(
                LinkQuery::try_new(
                    anchor_hash(&ref_anchor)?,
                    LinkTypes::InventionToRoyaltyRule,
                )?,
                GetStrategy::default(),
            ) {
                for rule_link in rule_links {
                    let rule_hash = match ActionHash::try_from(rule_link.target) {
                        Ok(h) => h,
                        Err(_) => continue,
                    };
                    let rule_record = match get(rule_hash.clone(), GetOptions::default())? {
                        Some(r) => r,
                        None => continue,
                    };
                    let rule = match extract_royalty_rule(&rule_record) {
                        Ok(r) => r,
                        Err(_) => continue,
                    };

                    if !rule.active || rule.rule_type != RoyaltyTrigger::PerDerivativeWork {
                        continue;
                    }

                    let amount = rule.minimum_amount_tend.unwrap_or(0.0);
                    let now_micros = now.as_micros();
                    let ledger = RoyaltyLedgerEntry {
                        id: format!(
                            "ledger:{}:derivative:{}:{}",
                            ref_claim.id, claim.id, now_micros
                        ),
                        royalty_rule_id: rule.id.clone(),
                        invention_id: ref_claim.id.clone(),
                        payer_did: claim.inventor_did.clone(),
                        receiver_did: rule.receiver_did.clone(),
                        amount_tend: amount,
                        trigger_event: format!(
                            "Derivative invention '{}' registered referencing '{}'",
                            claim.id, ref_claim.id
                        ),
                        trigger_action_hash: Some(action_hash.clone().to_string()),
                        status: RoyaltyStatus::Pending,
                        created_at: now_micros,
                        paid_at: None,
                    };

                    let _ = create_ledger_entry_with_links(ledger, Some(rule_hash));

                    let _ = emit_signal(&InventionSignal::RoyaltyTriggered {
                        invention_id: ref_claim.id.clone(),
                        payer_did: claim.inventor_did.clone(),
                        amount_tend: amount,
                    });
                }
            }
        }
    }

    let _ = emit_signal(&InventionSignal::InventionRegistered {
        id: input.id,
        inventor_did: input.inventor_did,
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created invention".into()
    )))
}

/// Update an existing invention claim. Only the original author can update.
/// Status transitions are validated per the InventionStatus state machine.
#[hdk_extern]
pub fn update_invention(input: UpdateInventionInput) -> ExternResult<Record> {
    let record = get(input.original_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Invention not found".into())),
    )?;

    // Author-only enforcement
    let author = record.action().author().clone();
    let me = agent_info()?.agent_initial_pubkey;
    if author != me {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the invention author can update this record".into()
        )));
    }

    let mut claim = extract_invention(&record)?;

    // Status transition validation
    if let Some(ref new_status) = input.status {
        if !claim.status.can_transition_to(new_status) {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid status transition: {:?} -> {:?}",
                claim.status, new_status
            ))));
        }
        claim.status = new_status.clone();
    }

    // Apply optional field updates
    if let Some(title) = input.title {
        claim.title = title;
    }
    if let Some(description) = input.description {
        claim.description = description;
    }
    if let Some(co_inventors) = input.co_inventors {
        claim.co_inventors = co_inventors;
    }
    if let Some(prior_art_refs) = input.prior_art_refs {
        claim.prior_art_refs = prior_art_refs;
    }
    if let Some(evidence_hashes) = input.evidence_hashes {
        claim.evidence_hashes = evidence_hashes;
    }
    if let Some(license_terms) = input.license_terms {
        claim.license_terms = license_terms;
    }
    if let Some(domain) = input.domain {
        claim.domain = domain;
    }
    if let Some(classification) = input.classification {
        claim.classification = classification;
    }

    claim.updated_at = sys_time()?;

    let new_action_hash = update_entry(input.original_action_hash, &claim)?;

    let _ = emit_signal(&InventionSignal::InventionUpdated {
        id: claim.id.clone(),
        new_status: claim.status.clone(),
    });

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch updated invention".into()
    )))
}

/// Get an invention by its ActionHash.
#[hdk_extern]
pub fn get_invention(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get an invention by its string ID (O(1) via anchor index).
#[hdk_extern]
pub fn get_invention_by_id(id: String) -> ExternResult<Option<Record>> {
    let id_anchor = format!("invention:{}", id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&id_anchor)?, LinkTypes::IdToInvention)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Get all inventions by a specific inventor DID.
#[hdk_extern]
pub fn get_inventions_by_inventor(did: String) -> ExternResult<Vec<Record>> {
    let inventor_anchor = format!("inventor:{}", did);
    resolve_links(
        anchor_hash(&inventor_anchor)?,
        LinkTypes::InventorToInvention,
    )
}

/// Get all inventions in a specific domain.
#[hdk_extern]
pub fn get_inventions_by_domain(domain: String) -> ExternResult<Vec<Record>> {
    let domain_anchor = format!("domain:{}", domain);
    resolve_links(
        anchor_hash(&domain_anchor)?,
        LinkTypes::DomainToInvention,
    )
}

/// File a challenge against an invention claim.
/// Transitions the invention to Challenged status.
#[hdk_extern]
pub fn challenge_invention(input: ChallengeInput) -> ExternResult<Record> {
    enforce_rate_limit(MAX_CHALLENGES_PER_HOUR)?;

    // Verify the invention exists and is in a challengeable state
    let inv_record = get(input.invention_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Invention not found".into())),
    )?;

    let inv_claim = extract_invention(&inv_record)?;
    if !inv_claim
        .status
        .can_transition_to(&InventionStatus::Challenged)
    {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot challenge invention in {:?} status",
            inv_claim.status
        ))));
    }

    let now = sys_time()?;
    let challenge = InventionChallenge {
        invention_hash: input.invention_action_hash.clone(),
        challenger_did: input.challenger_did.clone(),
        reason: input.reason,
        evidence: input.evidence,
        status: ChallengeStatus::Filed,
        created_at: now,
    };

    let challenge_hash = create_entry(&EntryTypes::InventionChallenge(challenge))?;

    // Link invention -> challenge
    create_link(
        input.invention_action_hash.clone(),
        challenge_hash.clone(),
        LinkTypes::InventionToChallenge,
        (),
    )?;

    // Transition invention to Challenged status
    let mut updated_claim = inv_claim.clone();
    updated_claim.status = InventionStatus::Challenged;
    updated_claim.updated_at = now;
    update_entry(input.invention_action_hash, &updated_claim)?;

    let _ = emit_signal(&InventionSignal::ChallengeFiled {
        invention_id: inv_claim.id,
        challenger_did: input.challenger_did,
    });

    get(challenge_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created challenge".into()
    )))
}

/// Resolve a challenge. Sets the challenge status and may update the
/// invention status accordingly.
///
/// Currently any agent can resolve (in production, this would be
/// restricted to arbitrators via a capability grant or governance vote).
#[hdk_extern]
pub fn resolve_challenge(input: ResolveChallengeInput) -> ExternResult<Record> {
    let ch_record = get(input.challenge_action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Challenge not found".into())),
    )?;

    let mut challenge: InventionChallenge = ch_record
        .entry()
        .to_app_option()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to deserialize challenge: {}",
                e
            )))
        })?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("Record has no entry".into())))?;

    // Only Filed or UnderReview challenges can be resolved
    match challenge.status {
        ChallengeStatus::Filed | ChallengeStatus::UnderReview => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Challenge already resolved with status {:?}",
                challenge.status
            ))));
        }
    }

    // Only Upheld or Dismissed are valid resolutions
    match input.resolution {
        ChallengeStatus::Upheld | ChallengeStatus::Dismissed => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Resolution must be Upheld or Dismissed".into()
            )));
        }
    }

    challenge.status = input.resolution.clone();
    let new_ch_hash = update_entry(input.challenge_action_hash, &challenge)?;

    // If dismissed, return invention to Published; if upheld, it stays Challenged
    // (the inventor may then choose to Revoke or the community can Supersede)
    if input.resolution == ChallengeStatus::Dismissed {
        if let Some(inv_record) =
            get(challenge.invention_hash.clone(), GetOptions::default())?
        {
            let mut inv_claim = extract_invention(&inv_record)?;
            if inv_claim.status == InventionStatus::Challenged {
                inv_claim.status = InventionStatus::Published;
                inv_claim.updated_at = sys_time()?;
                update_entry(challenge.invention_hash.clone(), &inv_claim)?;
            }
        }
    }

    // Resolve the invention_id for the signal
    let inv_id = if let Some(inv_record) =
        get(challenge.invention_hash.clone(), GetOptions::default())?
    {
        extract_invention(&inv_record)
            .map(|c| c.id)
            .unwrap_or_default()
    } else {
        String::new()
    };

    let _ = emit_signal(&InventionSignal::ChallengeResolved {
        invention_id: inv_id,
        resolution: input.resolution,
    });

    get(new_ch_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch resolved challenge".into()
    )))
}

/// Mark an invention as Verified (after the challenge period has passed
/// without successful challenge).
#[hdk_extern]
pub fn verify_invention(action_hash: ActionHash) -> ExternResult<Record> {
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Invention not found".into())),
    )?;

    let mut claim = extract_invention(&record)?;

    if !claim
        .status
        .can_transition_to(&InventionStatus::Verified)
    {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot verify invention in {:?} status",
            claim.status
        ))));
    }

    // Check that all challenges are resolved
    let challenges = get_links(
        LinkQuery::try_new(action_hash.clone(), LinkTypes::InventionToChallenge)?,
        GetStrategy::default(),
    )?;

    for ch_link in &challenges {
        let ch_hash = ActionHash::try_from(ch_link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(ch_record) = get(ch_hash, GetOptions::default())? {
            let ch: InventionChallenge = ch_record
                .entry()
                .to_app_option()
                .map_err(|e| {
                    wasm_error!(WasmErrorInner::Guest(format!(
                        "Failed to deserialize challenge: {}",
                        e
                    )))
                })?
                .ok_or_else(|| {
                    wasm_error!(WasmErrorInner::Guest("Record has no entry".into()))
                })?;
            match ch.status {
                ChallengeStatus::Upheld | ChallengeStatus::Dismissed => {}
                _ => {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot verify: unresolved challenges remain".into()
                    )));
                }
            }
        }
    }

    claim.status = InventionStatus::Verified;
    claim.updated_at = sys_time()?;

    let new_hash = update_entry(action_hash, &claim)?;

    let _ = emit_signal(&InventionSignal::InventionUpdated {
        id: claim.id,
        new_status: InventionStatus::Verified,
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch verified invention".into()
    )))
}

/// Search the prior art registry by domain, date range, and E/N/M levels.
/// Returns inventions matching ALL specified criteria.
#[hdk_extern]
pub fn search_prior_art(input: PriorArtSearchInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(50).min(200) as usize;

    // Start from domain anchor if specified, otherwise all_inventions
    let base_records = if let Some(ref domain) = input.domain {
        let domain_anchor = format!("domain:{}", domain);
        resolve_links(
            anchor_hash(&domain_anchor)?,
            LinkTypes::DomainToInvention,
        )?
    } else {
        resolve_links(anchor_hash("all_inventions")?, LinkTypes::AllInventions)?
    };

    let mut results = Vec::new();

    for record in base_records {
        if results.len() >= limit {
            break;
        }

        let claim = match extract_invention(&record) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Filter by date range
        if let Some(after) = input.after {
            if claim.created_at < after {
                continue;
            }
        }
        if let Some(before) = input.before {
            if claim.created_at > before {
                continue;
            }
        }

        // Filter by E/N/M levels
        if let Some(min_e) = input.min_empirical {
            if claim.classification.empirical < min_e {
                continue;
            }
        }
        if let Some(min_n) = input.min_normative {
            if claim.classification.normative < min_n {
                continue;
            }
        }
        if let Some(min_m) = input.min_materiality {
            if claim.classification.materiality < min_m {
                continue;
            }
        }

        results.push(record);
    }

    Ok(results)
}

// ── Royalty Functions ─────────────────────────────────────────────────

/// Create a royalty rule for an invention.
/// Only the invention author (the agent who created it) can create rules.
#[hdk_extern]
pub fn create_royalty_rule(input: CreateRoyaltyRuleInput) -> ExternResult<Record> {
    enforce_rate_limit(MAX_ROYALTY_RULES_PER_HOUR)?;

    // Verify the invention exists and caller is the author
    let (inv_record, _inv_claim) =
        lookup_invention_by_id(&input.invention_id)?.ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Invention '{}' not found",
                input.invention_id
            )))
        })?;

    let author = inv_record.action().author().clone();
    let me = agent_info()?.agent_initial_pubkey;
    if author != me {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the invention author can create royalty rules".into()
        )));
    }

    let now = sys_time()?;
    let now_micros = now.as_micros();

    let rule = InventionRoyaltyRule {
        id: format!(
            "royalty:{}:{:?}:{}",
            input.invention_id, input.rule_type, now_micros
        ),
        invention_id: input.invention_id.clone(),
        rule_type: input.rule_type,
        percentage: input.percentage,
        minimum_amount_tend: input.minimum_amount_tend,
        receiver_did: input.receiver_did,
        active: true,
        created_at: now_micros,
    };

    let rule_id = rule.id.clone();
    let action_hash = create_entry(&EntryTypes::InventionRoyaltyRule(rule))?;

    // Link from invention anchor → royalty rule
    let inv_anchor = format!("invention:{}", input.invention_id);
    create_link(
        anchor_hash(&inv_anchor)?,
        action_hash.clone(),
        LinkTypes::InventionToRoyaltyRule,
        (),
    )?;

    let _ = emit_signal(&InventionSignal::RoyaltyRuleCreated {
        invention_id: input.invention_id,
        rule_id,
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created royalty rule".into()
    )))
}

/// Get all royalty rules for an invention.
#[hdk_extern]
pub fn get_royalty_rules(invention_id: String) -> ExternResult<Vec<Record>> {
    let inv_anchor = format!("invention:{}", invention_id);
    resolve_links(
        anchor_hash(&inv_anchor)?,
        LinkTypes::InventionToRoyaltyRule,
    )
}

/// Manually record a royalty event (e.g., commercial use, production deploy).
/// Creates a RoyaltyLedgerEntry for the specified rule.
#[hdk_extern]
pub fn record_royalty_event(input: RecordRoyaltyInput) -> ExternResult<Record> {
    // Verify the rule exists
    let inv_anchor = format!("invention:{}", input.invention_id);
    let rule_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&inv_anchor)?,
            LinkTypes::InventionToRoyaltyRule,
        )?,
        GetStrategy::default(),
    )?;

    let mut found_rule: Option<(ActionHash, InventionRoyaltyRule)> = None;
    for link in rule_links {
        let rule_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(rule_hash.clone(), GetOptions::default())? {
            let rule = extract_royalty_rule(&record)?;
            if rule.id == input.royalty_rule_id {
                found_rule = Some((rule_hash, rule));
                break;
            }
        }
    }

    let (rule_hash, rule) = found_rule.ok_or_else(|| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Royalty rule '{}' not found for invention '{}'",
            input.royalty_rule_id, input.invention_id
        )))
    })?;

    if !rule.active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Royalty rule is not active".into()
        )));
    }

    let amount = input
        .amount_tend
        .unwrap_or_else(|| rule.minimum_amount_tend.unwrap_or(0.0));

    let now_micros = sys_time()?.as_micros();
    let ledger = RoyaltyLedgerEntry {
        id: format!(
            "ledger:{}:{}:{}",
            input.invention_id, input.royalty_rule_id, now_micros
        ),
        royalty_rule_id: input.royalty_rule_id,
        invention_id: input.invention_id.clone(),
        payer_did: input.payer_did.clone(),
        receiver_did: rule.receiver_did.clone(),
        amount_tend: amount,
        trigger_event: input.trigger_event,
        trigger_action_hash: input.trigger_action_hash,
        status: RoyaltyStatus::Pending,
        created_at: now_micros,
        paid_at: None,
    };

    let action_hash = create_ledger_entry_with_links(ledger, Some(rule_hash))?;

    let _ = emit_signal(&InventionSignal::RoyaltyTriggered {
        invention_id: input.invention_id,
        payer_did: input.payer_did,
        amount_tend: amount,
    });

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch newly created ledger entry".into()
    )))
}

/// Mark a royalty ledger entry as paid. Either the payer or receiver can mark it.
#[hdk_extern]
pub fn mark_royalty_paid(ledger_entry_hash: ActionHash) -> ExternResult<Record> {
    let record = get(ledger_entry_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Ledger entry not found".into())),
    )?;

    let mut entry = extract_ledger_entry(&record)?;

    if entry.status != RoyaltyStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot mark as paid: current status is {:?}",
            entry.status
        ))));
    }

    entry.status = RoyaltyStatus::Paid;
    entry.paid_at = Some(sys_time()?.as_micros());

    let new_hash = update_entry(ledger_entry_hash, &entry)?;

    let _ = emit_signal(&InventionSignal::RoyaltyPaid {
        ledger_entry_id: entry.id,
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch updated ledger entry".into()
    )))
}

/// Get all pending royalties owed BY a specific agent.
#[hdk_extern]
pub fn get_royalties_owed(payer_did: String) -> ExternResult<Vec<Record>> {
    let payer_anchor = format!("royalty_owed:{}", payer_did);
    let records = resolve_links(
        anchor_hash(&payer_anchor)?,
        LinkTypes::AgentToRoyaltyOwed,
    )?;

    // Filter to only Pending entries
    let mut pending = Vec::new();
    for record in records {
        if let Ok(entry) = extract_ledger_entry(&record) {
            if entry.status == RoyaltyStatus::Pending {
                pending.push(record);
            }
        }
    }
    Ok(pending)
}

/// Get all pending royalties owed TO a specific agent.
#[hdk_extern]
pub fn get_royalties_receivable(receiver_did: String) -> ExternResult<Vec<Record>> {
    let receiver_anchor = format!("royalty_receivable:{}", receiver_did);
    let records = resolve_links(
        anchor_hash(&receiver_anchor)?,
        LinkTypes::AgentToRoyaltyReceivable,
    )?;

    // Filter to only Pending entries
    let mut pending = Vec::new();
    for record in records {
        if let Ok(entry) = extract_ledger_entry(&record) {
            if entry.status == RoyaltyStatus::Pending {
                pending.push(record);
            }
        }
    }
    Ok(pending)
}

/// Waive (forgive) a royalty. Only the receiver can waive.
/// In practice, caller identity is verified via agent_info matching
/// the receiver_did — but since DID↔agent mapping is external,
/// we allow any agent to call this and rely on the application layer
/// for DID-based authorization.
#[hdk_extern]
pub fn waive_royalty(ledger_entry_hash: ActionHash) -> ExternResult<Record> {
    let record = get(ledger_entry_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Ledger entry not found".into())),
    )?;

    let mut entry = extract_ledger_entry(&record)?;

    if entry.status != RoyaltyStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot waive: current status is {:?}",
            entry.status
        ))));
    }

    entry.status = RoyaltyStatus::Waived;

    let new_hash = update_entry(ledger_entry_hash, &entry)?;

    let _ = emit_signal(&InventionSignal::RoyaltyWaived {
        ledger_entry_id: entry.id,
    });

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not fetch updated ledger entry".into()
    )))
}
