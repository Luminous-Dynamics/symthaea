//! Staking Coordinator Zome
//!
//! Business logic for SAP-based collateral staking with MYCEL weighting:
//! - Stake creation with SAP collateral and MYCEL score
//! - Slashing with cryptographic evidence
//! - Escrow with multiple release conditions
//! - Reward distribution with Merkle proofs

use hdk::prelude::*;
use staking_integrity::*;

/// Anchor for active stakes
const ACTIVE_STAKES_ANCHOR: &str = "active_stakes";

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    // Create a deterministic entry hash from the anchor string
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    anchor_str.hash(&mut hasher);
    let h1 = hasher.finish();
    hasher.write_u64(h1);
    let h2 = hasher.finish();
    hasher.write_u64(h2);
    let h3 = hasher.finish();
    hasher.write_u64(h3);
    let h4 = hasher.finish();

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&h1.to_le_bytes());
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result[16..24].copy_from_slice(&h3.to_le_bytes());
    result[24..32].copy_from_slice(&h4.to_le_bytes());

    Ok(EntryHash::from_raw_32(result.to_vec()))
}

/// Compute a 32-byte hash from arbitrary bytes
fn compute_bytes_hash(data: &[u8]) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let h1 = hasher.finish();
    hasher.write_u64(h1);
    let h2 = hasher.finish();
    hasher.write_u64(h2);
    let h3 = hasher.finish();
    hasher.write_u64(h3);
    let h4 = hasher.finish();

    let mut result = Vec::with_capacity(32);
    result.extend_from_slice(&h1.to_le_bytes());
    result.extend_from_slice(&h2.to_le_bytes());
    result.extend_from_slice(&h3.to_le_bytes());
    result.extend_from_slice(&h4.to_le_bytes());
    result
}

// =============================================================================
// Collateral Staking (SAP + MYCEL)
// =============================================================================

/// Input for creating a collateral stake
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateStakeInput {
    pub staker_did: String,
    pub sap_amount: u64,
    pub mycel_score: f32,
}

/// Create a new collateral stake
///
/// SAP collateral with MYCEL-weighted influence.
/// Stake weight = 1.0 + mycel_score (range: 1.0-2.0).
#[hdk_extern]
pub fn create_stake(input: CreateStakeInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let stake_id = format!("stake:{}:{}", input.staker_did, now.as_micros());

    // Clamp mycel_score to [0.0, 1.0]
    let mycel_score = input.mycel_score.clamp(0.0, 1.0);

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find stake".into())))
}

/// Begin unbonding a stake
#[hdk_extern]
pub fn begin_unbonding(stake_id: String) -> ExternResult<Record> {
    let now = sys_time()?;
    // 21-day unbonding period
    let unbonding_end = Timestamp::from_micros(
        now.as_micros() as i64 + (21 * 24 * 3600 * 1_000_000)
    );

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CollateralStake)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(stake) = record.entry().to_app_option::<CollateralStake>().ok().flatten() {
            if stake.id == stake_id && stake.status == StakeStatus::Active {
                let updated = CollateralStake {
                    status: StakeStatus::Unbonding,
                    unbonding_until: Some(unbonding_end),
                    ..stake
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CollateralStake(updated),
                )?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Active stake not found".into())))
}

/// Complete withdrawal after unbonding period
#[hdk_extern]
pub fn withdraw_stake(stake_id: String) -> ExternResult<Record> {
    let now = sys_time()?;

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CollateralStake)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(stake) = record.entry().to_app_option::<CollateralStake>().ok().flatten() {
            if stake.id == stake_id && stake.status == StakeStatus::Unbonding {
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

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Unbonding stake not found".into())))
}

/// Update stake MYCEL score
#[hdk_extern]
pub fn update_stake_mycel(input: UpdateMycelInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CollateralStake)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(stake) = record.entry().to_app_option::<CollateralStake>().ok().flatten() {
            if stake.id == input.stake_id && stake.status == StakeStatus::Active {
                let mycel_score = input.new_mycel_score.clamp(0.0, 1.0);
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

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Active stake not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMycelInput {
    pub stake_id: String,
    pub new_mycel_score: f32,
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

/// Slash a stake with cryptographic evidence
#[hdk_extern]
pub fn slash_stake(input: SlashStakeInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Serialize and hash evidence
    let evidence_bytes = serde_json::to_vec(&input.evidence)
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;
    let evidence_hash = compute_bytes_hash(&evidence_bytes);

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CollateralStake)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(stake) = record.entry().to_app_option::<CollateralStake>().ok().flatten() {
            if stake.id == input.stake_id {
                let slash_pct = input.custom_slash_percentage
                    .unwrap_or_else(|| input.reason.default_slash_percentage());

                // Calculate slashed SAP amount
                let sap_slashed = (stake.sap_amount as u128 * slash_pct as u128 / 100) as u64;

                let jailed = input.reason.results_in_jail();
                let jail_release = if jailed {
                    // 7-day jail period
                    Some(Timestamp::from_micros(
                        now.as_micros() as i64 + (7 * 24 * 3600 * 1_000_000)
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

                return get(event_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Stake not found".into())))
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
    let now = sys_time()?;
    let escrow_id = format!("escrow:{}:{}:{}", input.depositor_did, input.beneficiary_did, now.as_micros());

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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Reveal hash preimage to satisfy hash-lock condition
#[hdk_extern]
pub fn reveal_hash_preimage(input: RevealPreimageInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CryptoEscrow)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(escrow) = record.entry().to_app_option::<CryptoEscrow>().ok().flatten() {
            if escrow.id == input.escrow_id && escrow.status == EscrowStatus::Pending {
                // Verify hash preimage
                if let Some(_hash_lock) = &escrow.hash_lock {
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

                            return get(action_hash, GetOptions::default())?
                                .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
                        }
                    }
                }

                return Err(wasm_error!(WasmErrorInner::Guest(
                    "No hash-lock condition found".into()
                )));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Escrow not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevealPreimageInput {
    pub escrow_id: String,
    pub preimage: Vec<u8>,
}

/// Compute hash for hash-lock verification
fn compute_hash(preimage: &[u8], hash_type: &EscrowHashType) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simplified hash implementation (in production, use proper crypto libraries)
    match hash_type {
        EscrowHashType::Sha256 | EscrowHashType::Sha3_256 | EscrowHashType::Blake2b | EscrowHashType::Keccak256 => {
            let mut hasher = DefaultHasher::new();
            preimage.hash(&mut hasher);
            let h1 = hasher.finish();
            hasher.write_u64(h1);
            let h2 = hasher.finish();
            hasher.write_u64(h2);
            let h3 = hasher.finish();
            hasher.write_u64(h3);
            let h4 = hasher.finish();

            let mut result = Vec::with_capacity(32);
            result.extend_from_slice(&h1.to_le_bytes());
            result.extend_from_slice(&h2.to_le_bytes());
            result.extend_from_slice(&h3.to_le_bytes());
            result.extend_from_slice(&h4.to_le_bytes());
            result
        }
    }
}

/// Add multi-sig signature to escrow
#[hdk_extern]
pub fn add_escrow_signature(input: AddSignatureInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CryptoEscrow)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(escrow) = record.entry().to_app_option::<CryptoEscrow>().ok().flatten() {
            if escrow.id == input.escrow_id && escrow.status == EscrowStatus::Pending {
                // Verify signer is authorized
                if !escrow.multisig_signers.contains(&input.signer_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Signer not authorized".into()
                    )));
                }

                // Check if already signed
                if escrow.collected_signatures.iter().any(|s| s.signer_did == input.signer_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Already signed".into()
                    )));
                }

                let signature = EscrowSignature {
                    signer_did: input.signer_did,
                    signature: input.signature,
                    signed_at: now.as_micros() as i64,
                };

                let mut collected = escrow.collected_signatures.clone();
                collected.push(signature);

                // Check if multi-sig threshold is met
                let mut met_conditions = escrow.met_conditions.clone();
                if let Some(threshold) = escrow.multisig_threshold {
                    if collected.len() >= threshold as usize {
                        // Find and mark the multi-sig condition as met
                        for (i, condition) in escrow.conditions.iter().enumerate() {
                            if matches!(condition, ReleaseCondition::MultiSig { .. }) {
                                if !met_conditions.contains(&(i as u8)) {
                                    met_conditions.push(i as u8);
                                }
                                break;
                            }
                        }
                    }
                }

                let status = if met_conditions.len() >= escrow.required_conditions as usize {
                    EscrowStatus::Releasable
                } else {
                    EscrowStatus::Pending
                };

                let updated = CryptoEscrow {
                    collected_signatures: collected,
                    met_conditions,
                    status,
                    ..escrow
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CryptoEscrow(updated),
                )?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Escrow not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddSignatureInput {
    pub escrow_id: String,
    pub signer_did: String,
    pub signature: Vec<u8>,
}

/// Release escrow to beneficiary
#[hdk_extern]
pub fn release_escrow(escrow_id: String) -> ExternResult<Record> {
    let now = sys_time()?;

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CryptoEscrow)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(escrow) = record.entry().to_app_option::<CryptoEscrow>().ok().flatten() {
            if escrow.id == escrow_id && escrow.status == EscrowStatus::Releasable {
                let updated = CryptoEscrow {
                    status: EscrowStatus::Released,
                    released_at: Some(now),
                    ..escrow
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CryptoEscrow(updated),
                )?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest("Releasable escrow not found".into())))
}

// =============================================================================
// Query Functions
// =============================================================================

/// Get all stakes for a staker
#[hdk_extern]
pub fn get_staker_stakes(staker_did: String) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::try_new(
        anchor_hash(&format!("staker:{}", staker_did))?,
        LinkTypes::StakerToStake,
    )?;
    let links = get_links(query, GetStrategy::default())?;

    let mut stakes = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            stakes.push(record);
        }
    }

    Ok(stakes)
}

/// Get all active stakes
#[hdk_extern]
pub fn get_active_stakes(_: ()) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::try_new(
        anchor_hash(ACTIVE_STAKES_ANCHOR)?,
        LinkTypes::ActiveStakes,
    )?;
    let links = get_links(query, GetStrategy::default())?;

    let mut stakes = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
            if let Some(stake) = record.entry().to_app_option::<CollateralStake>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if stake.status == StakeStatus::Active {
                    stakes.push(record);
                }
            }
        }
    }

    Ok(stakes)
}

/// Get escrows for a depositor
#[hdk_extern]
pub fn get_depositor_escrows(depositor_did: String) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::try_new(
        anchor_hash(&format!("depositor:{}", depositor_did))?,
        LinkTypes::DepositorToEscrow,
    )?;
    let links = get_links(query, GetStrategy::default())?;

    let mut escrows = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            escrows.push(record);
        }
    }

    Ok(escrows)
}

/// Get escrows for a beneficiary
#[hdk_extern]
pub fn get_beneficiary_escrows(beneficiary_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("beneficiary:{}", beneficiary_did))?,
            LinkTypes::BeneficiaryToEscrow,
        )?,
        GetStrategy::default(),
    )?;

    let mut escrows = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah, GetOptions::default())? {
            escrows.push(record);
        }
    }

    Ok(escrows)
}

/// Calculate total weighted stake in the network
#[hdk_extern]
pub fn get_total_weighted_stake(_: ()) -> ExternResult<f64> {
    let stakes = get_active_stakes(())?;
    let mut total = 0.0;

    for record in stakes {
        if let Some(stake) = record.entry().to_app_option::<CollateralStake>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            total += stake.sap_amount as f64 * stake.stake_weight as f64;
        }
    }

    Ok(total)
}
