#![deny(unsafe_code)]
//! Staking Coordinator Zome
//!
//! Business logic for SAP-based collateral staking with MYCEL weighting:
//! - Stake creation with SAP collateral and MYCEL score
//! - Slashing with cryptographic evidence
//! - Escrow with multiple release conditions
//! - Reward distribution with Merkle proofs

use hdk::prelude::*;
use mycelix_finance_shared::{
    anchor_hash, follow_update_chain, links_to_records, verify_caller_is_did,
    verify_governance_or_bootstrap_from_links, GOVERNANCE_AGENTS_ANCHOR,
};
use staking_integrity::*;

/// Anchor for active stakes
const ACTIVE_STAKES_ANCHOR: &str = "active_stakes";

fn verify_governance_or_bootstrap() -> ExternResult<()> {
    let gov_links = get_links(
        LinkQuery::try_new(
            anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
            LinkTypes::GovernanceAgents,
        )?,
        GetStrategy::default(),
    )?;
    verify_governance_or_bootstrap_from_links(gov_links)
}

/// Register a governance agent. Only existing governance agents can register
/// new ones (or anyone during bootstrap when no agents exist yet).
#[hdk_extern]
pub fn register_governance_agent(agent: AgentPubKey) -> ExternResult<ActionHash> {
    verify_governance_or_bootstrap()?;
    create_link(
        anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
        agent,
        LinkTypes::GovernanceAgents,
        (),
    )
}

/// Compute a 32-byte Blake2b hash from arbitrary bytes.
fn compute_bytes_hash(data: &[u8]) -> Vec<u8> {
    blake2b_simd::Params::new()
        .hash_length(32)
        .hash(data)
        .as_bytes()
        .to_vec()
}

// =============================================================================
// Collateral Staking (SAP + MYCEL)
// =============================================================================

/// Input for creating a collateral stake
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateStakeInput {
    pub staker_did: String,
    pub sap_amount: u64,
}

/// Create a new collateral stake
///
/// SAP collateral with MYCEL-weighted influence.
/// Stake weight = 1.0 + mycel_score (range: 1.0-2.0).
/// MYCEL score is fetched from the recognition zome (not caller-provided).
#[hdk_extern]
pub fn create_stake(input: CreateStakeInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.staker_did)?;
    let now = sys_time()?;
    let stake_id = format!("stake:{}:{}", input.staker_did, now.as_micros());

    // Fetch verified MYCEL score from recognition zome
    let mycel_score = fetch_verified_mycel_score(&input.staker_did)?;

    // Calculate stake weight: 1.0 + mycel_score (range: 1.0-2.0)
    let stake_weight = 1.0 + mycel_score;

    let stake = CollateralStake {
        id: stake_id.clone(),
        staker_did: input.staker_did.clone(),
        sap_amount: input.sap_amount,
        mycel_score,
        stake_weight,
        staked_at: now,
        unbonding_until: None,
        status: StakeStatus::Active,
        pending_rewards: 0,
        last_reward_claim: now,
    };

    let action_hash = create_entry(&EntryTypes::CollateralStake(stake))?;

    // Link staker to stake
    create_link(
        anchor_hash(&format!("staker:{}", input.staker_did))?,
        action_hash.clone(),
        LinkTypes::StakerToStake,
        (),
    )?;

    // Link to active stakes anchor
    create_link(
        anchor_hash(ACTIVE_STAKES_ANCHOR)?,
        action_hash.clone(),
        LinkTypes::ActiveStakes,
        (),
    )?;

    // Index by stake ID for O(1) lookup
    create_link(
        anchor_hash(&stake_id)?,
        action_hash.clone(),
        LinkTypes::StakeIdToStake,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Stake record not found after creation for staker {}",
        input.staker_did
    ))))
}

/// Look up a stake by its ID using the StakeIdToStake link index.
/// Returns (CollateralStake, Record) or an error if not found.
/// Follows the update chain to return the latest version.
fn find_stake_by_id(stake_id: &str) -> ExternResult<(CollateralStake, Record)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(stake_id)?, LinkTypes::StakeIdToStake)?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Stake not found for ID {}",
            stake_id
        ))))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let stake = record
        .entry()
        .to_app_option::<CollateralStake>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Stake entry missing".into()
        )))?;
    Ok((stake, record))
}

/// Look up an escrow by its ID using the EscrowIdToEscrow link index.
/// Returns (CryptoEscrow, Record) or an error if not found.
/// Follows the update chain to return the latest version.
fn find_escrow_by_id(escrow_id: &str) -> ExternResult<(CryptoEscrow, Record)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(escrow_id)?, LinkTypes::EscrowIdToEscrow)?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Escrow not found for ID {}",
            escrow_id
        ))))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let escrow = record
        .entry()
        .to_app_option::<CryptoEscrow>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Escrow entry missing".into()
        )))?;
    Ok((escrow, record))
}

/// Begin unbonding a stake
#[hdk_extern]
pub fn begin_unbonding(stake_id: String) -> ExternResult<Record> {
    let now = sys_time()?;
    // 21-day unbonding period
    let unbonding_end =
        Timestamp::from_micros(now.as_micros() as i64 + (21 * 24 * 3600 * 1_000_000));

    let (stake, record) = find_stake_by_id(&stake_id)?;

    if stake.status != StakeStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Stake {} is {:?}, expected Active for unbonding",
            stake_id, stake.status
        ))));
    }

    let updated = CollateralStake {
        status: StakeStatus::Unbonding,
        unbonding_until: Some(unbonding_end),
        ..stake
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CollateralStake(updated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Stake record not found after unbonding for stake {}",
        stake_id
    ))))
}

/// Complete withdrawal after unbonding period
#[hdk_extern]
pub fn withdraw_stake(stake_id: String) -> ExternResult<Record> {
    let now = sys_time()?;
    let (stake, record) = find_stake_by_id(&stake_id)?;

    if stake.status != StakeStatus::Unbonding {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Stake {} is {:?}, expected Unbonding for withdrawal",
            stake_id, stake.status
        ))));
    }

    // Check if unbonding period is complete
    if let Some(unbonding_until) = stake.unbonding_until {
        if (now.as_micros() as i64) < unbonding_until.as_micros() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Unbonding period not complete".into()
            )));
        }
    }

    let updated = CollateralStake {
        status: StakeStatus::Withdrawn,
        ..stake
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CollateralStake(updated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Stake record not found after withdrawal for stake {}",
        stake_id
    ))))
}

/// Update stake MYCEL score
#[hdk_extern]
pub fn update_stake_mycel(input: UpdateMycelInput) -> ExternResult<Record> {
    let (stake, record) = find_stake_by_id(&input.stake_id)?;

    if stake.status != StakeStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Stake {} is {:?}, expected Active for MYCEL update",
            input.stake_id, stake.status
        ))));
    }

    // Fetch verified MYCEL score from recognition zome
    let mycel_score = fetch_verified_mycel_score(&stake.staker_did)?;
    let new_weight = 1.0 + mycel_score;
    let updated = CollateralStake {
        mycel_score,
        stake_weight: new_weight,
        ..stake
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CollateralStake(updated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Stake record not found after MYCEL update for stake {}",
        input.stake_id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMycelInput {
    pub stake_id: String,
}

/// Fetch a member's verified MYCEL score from the recognition zome.
/// Falls back to 0.0 (minimum weight) if recognition zome is unreachable.
fn fetch_verified_mycel_score(staker_did: &str) -> ExternResult<f32> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("recognition"),
        FunctionName::from("get_mycel_score"),
        None,
        staker_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct MycelState {
                mycel_score: f64,
            }
            match result.decode::<MycelState>() {
                Ok(state) => Ok((state.mycel_score as f32).clamp(0.0, 1.0)),
                Err(e) => {
                    debug!(
                        "fetch_verified_mycel_score: decode error for {}: {:?}, defaulting to 0.0",
                        staker_did, e
                    );
                    Ok(0.0)
                }
            }
        }
        Ok(other) => {
            debug!(
                "fetch_verified_mycel_score: recognition returned {:?} for {}, defaulting to 0.0",
                other, staker_did
            );
            Ok(0.0)
        }
        Err(e) => {
            debug!("fetch_verified_mycel_score: recognition unreachable for {}: {:?}, defaulting to 0.0", staker_did, e);
            Ok(0.0) // Recognition unreachable → minimum weight
        }
    }
}

// =============================================================================
// Slashing
// =============================================================================

/// Input for slashing a stake
#[derive(Serialize, Deserialize, Debug)]
pub struct SlashStakeInput {
    pub stake_id: String,
    pub reason: SlashingReason,
    pub evidence: SlashingEvidence,
    /// Override default slash percentage (optional)
    pub custom_slash_percentage: Option<u8>,
}

/// Slash a stake with cryptographic evidence.
/// Restricted to authorized governance agents (or any agent during bootstrap).
#[hdk_extern]
pub fn slash_stake(input: SlashStakeInput) -> ExternResult<Record> {
    verify_governance_or_bootstrap()?;
    let now = sys_time()?;

    // Serialize and hash evidence
    let evidence_bytes = serde_json::to_vec(&input.evidence)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    let evidence_hash = compute_bytes_hash(&evidence_bytes);

    let (stake, record) = find_stake_by_id(&input.stake_id)?;

    if stake.status != StakeStatus::Active && stake.status != StakeStatus::Unbonding {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot slash a {:?} stake — only Active or Unbonding stakes can be slashed",
            stake.status
        ))));
    }

    let slash_pct = input
        .custom_slash_percentage
        .unwrap_or_else(|| input.reason.default_slash_percentage());

    // Calculate slashed SAP amount
    let sap_slashed = (stake.sap_amount as u128 * slash_pct as u128 / 100) as u64;

    let jailed = input.reason.results_in_jail();
    let jail_release = if jailed {
        // 7-day jail period
        Some(Timestamp::from_micros(
            now.as_micros() as i64 + (7 * 24 * 3600 * 1_000_000),
        ))
    } else {
        None
    };

    // Create slashing event
    let event_id = format!("slash:{}:{}", input.stake_id, now.as_micros());
    let slashing_event = SlashingEvent {
        id: event_id.clone(),
        stake_id: input.stake_id.clone(),
        staker_did: stake.staker_did.clone(),
        reason: input.reason.clone(),
        slash_percentage: slash_pct,
        sap_slashed,
        evidence_hash: evidence_hash.clone(),
        evidence: evidence_bytes,
        slashed_at: now,
        jailed,
        jail_release,
    };

    let event_hash = create_entry(&EntryTypes::SlashingEvent(slashing_event))?;

    // Link stake to slashing event
    create_link(
        anchor_hash(&format!("stake:{}", input.stake_id))?,
        event_hash.clone(),
        LinkTypes::StakeToSlashing,
        (),
    )?;

    // Update stake
    let new_status = if jailed {
        StakeStatus::Jailed
    } else {
        StakeStatus::Slashed
    };

    let updated_stake = CollateralStake {
        sap_amount: stake.sap_amount - sap_slashed,
        status: new_status,
        ..stake
    };

    update_entry(
        record.action_address().clone(),
        &EntryTypes::CollateralStake(updated_stake),
    )?;

    get(event_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Slashing event record not found after creation for stake {}",
        input.stake_id
    ))))
}

// =============================================================================
// Crypto Escrow
// =============================================================================

/// Input for creating an escrow
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateEscrowInput {
    pub depositor_did: String,
    pub beneficiary_did: String,
    pub sap_amount: u64,
    pub purpose: String,
    pub conditions: Vec<ReleaseCondition>,
    pub required_conditions: u8,
    pub hash_lock: Option<Vec<u8>>,
    pub timelock: Option<i64>,
    pub multisig_threshold: Option<u8>,
    pub multisig_signers: Vec<String>,
}

/// Create a crypto escrow with release conditions
#[hdk_extern]
pub fn create_escrow(input: CreateEscrowInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.depositor_did)?;
    let now = sys_time()?;
    let escrow_id = format!(
        "escrow:{}:{}:{}",
        input.depositor_did,
        input.beneficiary_did,
        now.as_micros()
    );

    let timelock = input.timelock.map(Timestamp::from_micros);

    let escrow = CryptoEscrow {
        id: escrow_id.clone(),
        depositor_did: input.depositor_did.clone(),
        beneficiary_did: input.beneficiary_did.clone(),
        sap_amount: input.sap_amount,
        purpose: input.purpose,
        conditions: input.conditions,
        required_conditions: input.required_conditions,
        met_conditions: Vec::new(),
        hash_lock: input.hash_lock,
        timelock,
        multisig_threshold: input.multisig_threshold,
        multisig_signers: input.multisig_signers,
        collected_signatures: Vec::new(),
        status: EscrowStatus::Pending,
        created_at: now,
        released_at: None,
    };

    let action_hash = create_entry(&EntryTypes::CryptoEscrow(escrow))?;

    // Link depositor to escrow
    create_link(
        anchor_hash(&format!("depositor:{}", input.depositor_did))?,
        action_hash.clone(),
        LinkTypes::DepositorToEscrow,
        (),
    )?;

    // Link beneficiary to escrow
    create_link(
        anchor_hash(&format!("beneficiary:{}", input.beneficiary_did))?,
        action_hash.clone(),
        LinkTypes::BeneficiaryToEscrow,
        (),
    )?;

    // Index by escrow ID for O(1) lookup
    create_link(
        anchor_hash(&escrow_id)?,
        action_hash.clone(),
        LinkTypes::EscrowIdToEscrow,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Escrow record not found after creation for escrow {}",
        escrow_id
    ))))
}

/// Reveal hash preimage to satisfy hash-lock condition
#[hdk_extern]
pub fn reveal_hash_preimage(input: RevealPreimageInput) -> ExternResult<Record> {
    let (escrow, record) = find_escrow_by_id(&input.escrow_id)?;

    if escrow.status != EscrowStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Escrow {} is {:?}, expected Pending for hash preimage reveal",
            input.escrow_id, escrow.status
        ))));
    }

    // Verify hash preimage
    if escrow.hash_lock.is_some() {
        // Find hash-lock condition
        for (i, condition) in escrow.conditions.iter().enumerate() {
            if let ReleaseCondition::HashLock { hash, hash_type } = condition {
                // Verify the preimage
                let computed_hash = compute_hash(&input.preimage, hash_type);
                if &computed_hash != hash {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Invalid hash preimage".into()
                    )));
                }

                // Mark condition as met
                let mut met_conditions = escrow.met_conditions.clone();
                if !met_conditions.contains(&(i as u8)) {
                    met_conditions.push(i as u8);
                }

                // Check if escrow is now releasable
                let status = if met_conditions.len() >= escrow.required_conditions as usize {
                    EscrowStatus::Releasable
                } else {
                    EscrowStatus::Pending
                };

                let updated = CryptoEscrow {
                    met_conditions,
                    status,
                    ..escrow
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CryptoEscrow(updated),
                )?;

                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest(format!(
                        "Escrow record not found after hash preimage reveal for escrow {}",
                        input.escrow_id
                    ))
                ));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "No hash-lock condition found for escrow {}",
        input.escrow_id
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevealPreimageInput {
    pub escrow_id: String,
    pub preimage: Vec<u8>,
}

/// Compute hash for hash-lock verification.
/// All hash types use Blake2b-256 (the only hash available in WASM without
/// additional dependencies). Callers must generate their hash-locks using the
/// same algorithm.
fn compute_hash(preimage: &[u8], _hash_type: &EscrowHashType) -> Vec<u8> {
    blake2b_simd::Params::new()
        .hash_length(32)
        .hash(preimage)
        .as_bytes()
        .to_vec()
}

/// Collect all EscrowSignatureEntry records linked to an escrow's signature anchor.
///
/// Batch-optimized: EscrowSignatureEntry entries are immutable (write-once
/// signatures), so bare get() via links_to_records is equivalent to
/// follow_update_chain. The records are then deserialized in-memory.
fn get_escrow_signature_entries(
    escrow_id: &str,
) -> ExternResult<Vec<(ActionHash, EscrowSignatureEntry)>> {
    let sig_anchor = anchor_hash(&format!("escrow-sigs:{}", escrow_id))?;
    let links = get_links(
        LinkQuery::try_new(sig_anchor, LinkTypes::EscrowToSignatures)?,
        GetStrategy::default(),
    )?;

    let records = links_to_records(links)?;
    let mut entries = Vec::new();
    for record in records {
        let ah = record.action_address().clone();
        if let Some(sig_entry) = record
            .entry()
            .to_app_option::<EscrowSignatureEntry>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Deserialization error: {:?}",
                    e
                )))
            })?
        {
            entries.push((ah, sig_entry));
        }
    }
    Ok(entries)
}

/// Add multi-sig signature to escrow (RC-18: race-condition-safe)
///
/// Creates an immutable `EscrowSignatureEntry` and links it to the escrow
/// signature anchor. Duplicate detection is done AFTER creation for
/// idempotency — if a duplicate is found, the link is deleted and Ok is
/// returned.
#[hdk_extern]
pub fn add_escrow_signature(input: AddSignatureInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let (escrow, _record) = find_escrow_by_id(&input.escrow_id)?;

    if escrow.status != EscrowStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Escrow {} is {:?}, expected Pending for signature addition",
            input.escrow_id, escrow.status
        ))));
    }

    // Verify signer is authorized
    if !escrow.multisig_signers.contains(&input.signer_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Signer not authorized".into()
        )));
    }

    // Create immutable signature entry
    let sig_entry = EscrowSignatureEntry {
        escrow_id: input.escrow_id.clone(),
        signer_did: input.signer_did.clone(),
        signature: input.signature,
        timestamp: now,
    };

    let sig_action_hash = create_entry(&EntryTypes::EscrowSignatureEntry(sig_entry))?;

    // Link from escrow signature anchor to this signature entry
    let sig_anchor = anchor_hash(&format!("escrow-sigs:{}", input.escrow_id))?;
    let link_hash = create_link(
        sig_anchor,
        sig_action_hash.clone(),
        LinkTypes::EscrowToSignatures,
        (),
    )?;

    // Check for duplicate signer (AFTER creating, for idempotency)
    let existing_sigs = get_escrow_signature_entries(&input.escrow_id)?;
    let duplicates: Vec<_> = existing_sigs
        .iter()
        .filter(|(ah, s)| s.signer_did == input.signer_did && *ah != sig_action_hash)
        .collect();

    if !duplicates.is_empty() {
        // Another entry for this signer already exists — remove our link
        delete_link(link_hash, GetOptions::default())?;
        // Return the existing record
        let (existing_ah, _) = &duplicates[0];
        return get(existing_ah.clone(), GetOptions::default())?.ok_or(wasm_error!(
            WasmErrorInner::Guest(format!(
                "Existing signature record not found for escrow {} signer {}",
                input.escrow_id, input.signer_did
            ))
        ));
    }

    get(sig_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Signature record not found after creation for escrow {} signer {}",
        input.escrow_id, input.signer_did
    ))))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddSignatureInput {
    pub escrow_id: String,
    pub signer_did: String,
    pub signature: Vec<u8>,
}

/// Release escrow to beneficiary
///
/// Verifies the multi-sig threshold by counting unique signer DIDs from
/// linked `EscrowSignatureEntry` entries (RC-18 fix).
#[hdk_extern]
pub fn release_escrow(escrow_id: String) -> ExternResult<Record> {
    let now = sys_time()?;
    let (escrow, record) = find_escrow_by_id(&escrow_id)?;

    // Allow release if already marked Releasable, OR if still Pending but
    // the multi-sig threshold is now met via linked signature entries.
    if escrow.status != EscrowStatus::Releasable && escrow.status != EscrowStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Escrow is not in a releasable state".into()
        )));
    }

    // Count unique signers from linked EscrowSignatureEntry entries
    if let Some(threshold) = escrow.multisig_threshold {
        let sig_entries = get_escrow_signature_entries(&escrow_id)?;
        let mut unique_signers = std::collections::HashSet::new();
        for (_ah, sig) in &sig_entries {
            unique_signers.insert(sig.signer_did.clone());
        }
        if unique_signers.len() < threshold as usize {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Multi-sig threshold not met: {} of {} required signatures",
                unique_signers.len(),
                threshold
            ))));
        }
    }

    // Also check non-multisig met_conditions if required
    // (legacy path: met_conditions covers hash-lock, timelock, etc.)
    let mut met_conditions = escrow.met_conditions.clone();

    // If multisig threshold is met, mark the MultiSig condition as met
    if escrow.multisig_threshold.is_some() {
        for (i, condition) in escrow.conditions.iter().enumerate() {
            if matches!(condition, ReleaseCondition::MultiSig { .. }) {
                if !met_conditions.contains(&(i as u8)) {
                    met_conditions.push(i as u8);
                }
                break;
            }
        }
    }

    if met_conditions.len() < escrow.required_conditions as usize {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Not all required conditions are met".into()
        )));
    }

    let updated = CryptoEscrow {
        status: EscrowStatus::Released,
        released_at: Some(now),
        met_conditions,
        ..escrow
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CryptoEscrow(updated),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "Escrow record not found after release for escrow {}",
        escrow_id
    ))))
}

// =============================================================================
// Query Functions
// =============================================================================

/// Input for paginated staker queries
#[derive(Serialize, Deserialize, Debug)]
pub struct PaginatedDidInput {
    pub did: String,
    pub limit: Option<usize>,
}

/// Get all stakes for a staker (paginated, default limit 100)
///
// NOTE: Sequential follow_update_chain required — entries are mutable (status transitions:
// Active → Unbonding → Withdrawn, or Active → Slashed/Jailed)
#[hdk_extern]
pub fn get_staker_stakes(input: PaginatedDidInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(100);
    let query = LinkQuery::try_new(
        anchor_hash(&format!("staker:{}", input.did))?,
        LinkTypes::StakerToStake,
    )?;
    let links = get_links(query, GetStrategy::default())?;

    let mut stakes = Vec::new();
    for link in links.into_iter().take(limit) {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        stakes.push(follow_update_chain(ah)?);
    }

    Ok(stakes)
}

/// Get all active stakes (paginated, default limit 100)
///
// NOTE: Sequential follow_update_chain required — entries are mutable (status transitions).
/// Filters to only return currently-Active stakes (skips Withdrawn, Slashed, etc.).
#[hdk_extern]
pub fn get_active_stakes(limit: Option<usize>) -> ExternResult<Vec<Record>> {
    let limit = limit.unwrap_or(100);
    let query = LinkQuery::try_new(anchor_hash(ACTIVE_STAKES_ANCHOR)?, LinkTypes::ActiveStakes)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut stakes = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let record = follow_update_chain(ah)?;
        if let Some(stake) = record
            .entry()
            .to_app_option::<CollateralStake>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if stake.status == StakeStatus::Active {
                stakes.push(record);
                if stakes.len() >= limit {
                    break;
                }
            }
        }
    }

    Ok(stakes)
}

/// Get escrows for a depositor (paginated, default limit 100)
///
// NOTE: Sequential follow_update_chain required — entries are mutable (conditions met, status changes).
#[hdk_extern]
pub fn get_depositor_escrows(input: PaginatedDidInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(100);
    let query = LinkQuery::try_new(
        anchor_hash(&format!("depositor:{}", input.did))?,
        LinkTypes::DepositorToEscrow,
    )?;
    let links = get_links(query, GetStrategy::default())?;

    let mut escrows = Vec::new();
    for link in links.into_iter().take(limit) {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        escrows.push(follow_update_chain(ah)?);
    }

    Ok(escrows)
}

/// Get escrows for a beneficiary (paginated, default limit 100)
///
// NOTE: Sequential follow_update_chain required — entries are mutable (conditions met, status changes).
#[hdk_extern]
pub fn get_beneficiary_escrows(input: PaginatedDidInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("beneficiary:{}", input.did))?,
            LinkTypes::BeneficiaryToEscrow,
        )?,
        GetStrategy::default(),
    )?;

    let mut escrows = Vec::new();
    for link in links.into_iter().take(limit) {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        escrows.push(follow_update_chain(ah)?);
    }

    Ok(escrows)
}

/// Calculate total weighted stake in the network
#[hdk_extern]
pub fn get_total_weighted_stake(_: ()) -> ExternResult<f64> {
    let stakes = get_active_stakes(None)?;
    let mut total = 0.0;

    for record in stakes {
        if let Some(stake) = record
            .entry()
            .to_app_option::<CollateralStake>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            total += stake.sap_amount as f64 * stake.stake_weight as f64;
        }
    }

    Ok(total)
}
