//! Staking Coordinator Zome
//!
//! Business logic for SAP-based collateral staking with MYCEL weighting:
//! - Stake creation with SAP collateral and MYCEL score
//! - Slashing with cryptographic evidence
//! - Escrow with multiple release conditions
//! - Reward distribution with Merkle proofs

use hdk::prelude::*;
use mycelix_finance_shared::{anchor_hash, verify_caller_is_did};
use staking_integrity::*;

/// Anchor for active stakes
const ACTIVE_STAKES_ANCHOR: &str = "active_stakes";

const GOVERNANCE_AGENTS_ANCHOR: &str = "governance_agents";

/// Check if the calling agent is a registered governance agent.
/// If no governance agents are registered yet (bootstrap), allow any agent.
fn verify_governance_or_bootstrap() -> ExternResult<()> {
    let caller = agent_info()?.agent_initial_pubkey;
    let gov_links = get_links(
        LinkQuery::try_new(
            anchor_hash(GOVERNANCE_AGENTS_ANCHOR)?,
            LinkTypes::GovernanceAgents,
        )?,
        GetStrategy::default(),
    )?;

    // Bootstrap: if no governance agents registered yet, allow anyone
    if gov_links.is_empty() {
        return Ok(());
    }

    // Check if caller is in the governance list
    for link in gov_links {
        if let Ok(agent) = AgentPubKey::try_from(link.target) {
            if agent == caller {
                return Ok(());
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Caller is not an authorized governance agent".into()
    )))
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find stake".into()
    )))
}

/// Look up a stake by its ID using the StakeIdToStake link index.
/// Returns (CollateralStake, Record) or an error if not found.
fn find_stake_by_id(stake_id: &str) -> ExternResult<(CollateralStake, Record)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(stake_id)?, LinkTypes::StakeIdToStake)?,
        GetStrategy::default(),
    )?;
    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Stake not found".into())))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Stake record not found".into()
    )))?;
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
fn find_escrow_by_id(escrow_id: &str) -> ExternResult<(CryptoEscrow, Record)> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(escrow_id)?, LinkTypes::EscrowIdToEscrow)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Escrow not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Escrow record not found".into()
    )))?;
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Active stake not found".into()
        )));
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Complete withdrawal after unbonding period
#[hdk_extern]
pub fn withdraw_stake(stake_id: String) -> ExternResult<Record> {
    let now = sys_time()?;
    let (stake, record) = find_stake_by_id(&stake_id)?;

    if stake.status != StakeStatus::Unbonding {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unbonding stake not found".into()
        )));
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Update stake MYCEL score
#[hdk_extern]
pub fn update_stake_mycel(input: UpdateMycelInput) -> ExternResult<Record> {
    let (stake, record) = find_stake_by_id(&input.stake_id)?;

    if stake.status != StakeStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Active stake not found".into()
        )));
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
                Err(_) => Ok(0.0),
            }
        }
        _ => Ok(0.0), // Recognition unreachable → minimum weight
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

    get(event_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Reveal hash preimage to satisfy hash-lock condition
#[hdk_extern]
pub fn reveal_hash_preimage(input: RevealPreimageInput) -> ExternResult<Record> {
    let (escrow, record) = find_escrow_by_id(&input.escrow_id)?;

    if escrow.status != EscrowStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Escrow not found or not pending".into()
        )));
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

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "No hash-lock condition found".into()
    )))
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

/// Add multi-sig signature to escrow
#[hdk_extern]
pub fn add_escrow_signature(input: AddSignatureInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let (escrow, record) = find_escrow_by_id(&input.escrow_id)?;

    if escrow.status != EscrowStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Escrow not found or not pending".into()
        )));
    }

    // Verify signer is authorized
    if !escrow.multisig_signers.contains(&input.signer_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Signer not authorized".into()
        )));
    }

    // Check if already signed
    if escrow
        .collected_signatures
        .iter()
        .any(|s| s.signer_did == input.signer_did)
    {
        return Err(wasm_error!(WasmErrorInner::Guest("Already signed".into())));
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let (escrow, record) = find_escrow_by_id(&escrow_id)?;

    if escrow.status != EscrowStatus::Releasable {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Releasable escrow not found".into()
        )));
    }

    let updated = CryptoEscrow {
        status: EscrowStatus::Released,
        released_at: Some(now),
        ..escrow
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CryptoEscrow(updated),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
        if let Some(record) = get(ah, GetOptions::default())? {
            stakes.push(record);
        }
    }

    Ok(stakes)
}

/// Get all active stakes (paginated, default limit 100)
#[hdk_extern]
pub fn get_active_stakes(limit: Option<usize>) -> ExternResult<Vec<Record>> {
    let limit = limit.unwrap_or(100);
    let query = LinkQuery::try_new(anchor_hash(ACTIVE_STAKES_ANCHOR)?, LinkTypes::ActiveStakes)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut stakes = Vec::new();
    for link in links {
        let ah = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(ah.clone(), GetOptions::default())? {
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
    }

    Ok(stakes)
}

/// Get escrows for a depositor (paginated, default limit 100)
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
        if let Some(record) = get(ah, GetOptions::default())? {
            escrows.push(record);
        }
    }

    Ok(escrows)
}

/// Get escrows for a beneficiary (paginated, default limit 100)
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
        if let Some(record) = get(ah, GetOptions::default())? {
            escrows.push(record);
        }
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
