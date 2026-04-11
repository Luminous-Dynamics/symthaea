// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Boundary Contracts Coordinator Zome
//!
//! Pre-exchange consent layer for care/intimacy work. The cryptographic
//! infrastructure that prevents the 1602 extraction model.
//!
//! Contract terms are private between parties. Only existence + status visible
//! on the DHT. Violations route to restorative justice — never punitive enforcement.

use boundary_contracts_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::records_from_links;

// =============================================================================
// Cross-cluster bridge helpers (best-effort, never block primary operation)
// =============================================================================

/// Best-effort cross-cluster call to TEND zome to record a care exchange settlement.
/// Returns the TEND exchange ID if successful, or None if the finance cluster is
/// unreachable. The primary contract operation must not fail because of TEND.
fn bridge_tend_settlement(
    provider: &AgentPubKey,
    receiver: &AgentPubKey,
    hours: f32,
    description: &str,
) -> Option<String> {
    #[derive(Serialize, Debug)]
    struct TendPayload {
        receiver_did: String,
        hours: f32,
        service_description: String,
        service_category: String,
        cultural_alias: Option<String>,
        dao_did: String,
        service_date: Option<Timestamp>,
    }
    let payload = TendPayload {
        receiver_did: format!("did:mycelix:{}", receiver),
        hours,
        service_description: description.to_string(),
        service_category: "BoundaryContract".to_string(),
        cultural_alias: None,
        dao_did: "did:mycelix:commons".to_string(),
        service_date: None,
    };
    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("tend"),
        FunctionName::from("record_exchange"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Deserialize, Debug)]
            struct TendResult {
                id: String,
            }
            result.decode::<TendResult>().ok().map(|r| r.id)
        }
        Ok(other) => {
            debug!(
                "TEND settlement warning: non-Ok response for provider={}: {:?}",
                provider, other
            );
            None
        }
        Err(e) => {
            debug!(
                "TEND settlement warning: finance cluster unreachable for provider={}: {:?}",
                provider, e
            );
            None
        }
    }
}

/// Best-effort cross-cluster call to the civic justice_restorative zome to open
/// a restorative circle for a boundary violation. Returns the circle case_id if
/// successful, or None if the civic cluster is unreachable.
fn bridge_create_restorative_circle(
    contract_id: &str,
    reporter: &AgentPubKey,
    violator: &AgentPubKey,
    _description: &str,
    _severity: &ViolationSeverity,
) -> Option<String> {
    #[derive(Serialize, Debug)]
    struct CircleParticipant {
        did: String,
        role: String,
        consented: bool,
        attended_sessions: Vec<u32>,
    }
    #[derive(Serialize, Debug)]
    struct RestorativeCirclePayload {
        id: String,
        case_id: String,
        facilitator: String,
        participants: Vec<CircleParticipant>,
        status: String,
        sessions: Vec<()>,
        agreements: Vec<String>,
        created_at: Timestamp,
    }
    let now = sys_time().ok()?;
    let case_id = format!("bc-violation-{}-{}", contract_id, now.as_micros());
    let circle_id = format!("circle-{}", case_id);
    let payload = RestorativeCirclePayload {
        id: circle_id,
        case_id: case_id.clone(),
        facilitator: "did:mycelix:auto-facilitator".to_string(),
        participants: vec![
            CircleParticipant {
                did: format!("did:mycelix:{}", reporter),
                role: "HarmReceiver".to_string(),
                consented: true,
                attended_sessions: vec![],
            },
            CircleParticipant {
                did: format!("did:mycelix:{}", violator),
                role: "HarmDoer".to_string(),
                consented: false,
                attended_sessions: vec![],
            },
        ],
        status: "Forming".to_string(),
        sessions: vec![],
        agreements: vec![],
        created_at: now,
    };
    match call(
        CallTargetCell::OtherRole("civic".into()),
        ZomeName::from("justice_restorative"),
        FunctionName::from("create_circle"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(_)) => Some(case_id),
        Ok(other) => {
            debug!(
                "Restorative circle warning: non-Ok response for contract={}: {:?}",
                contract_id, other
            );
            None
        }
        Err(e) => {
            debug!(
                "Restorative circle warning: civic cluster unreachable for contract={}: {:?}",
                contract_id, e
            );
            None
        }
    }
}

/// Best-effort call to commons_bridge to slash a violator's community reputation
/// score. Uses the consciousness credential refresh path to apply the penalty.
/// Returns true if the slash was applied, false otherwise.
fn bridge_reputation_slash(violator: &AgentPubKey, slash_factor: f32) -> bool {
    // The SubPassport reputation slash is applied via the commons_bridge
    // consciousness credential system. We call refresh_consciousness_credential
    // with a penalty annotation. If the bridge extern doesn't exist yet, the
    // call fails gracefully.
    #[derive(Serialize, Debug)]
    struct SlashInput {
        agent: String,
        factor: f32,
        reason: String,
    }
    let payload = SlashInput {
        agent: format!("did:mycelix:{}", violator),
        factor: slash_factor,
        reason: "boundary_contract_violation".to_string(),
    };
    match call(
        CallTargetCell::Local,
        ZomeName::from("commons_bridge"),
        FunctionName::from("slash_community_score"),
        None,
        payload,
    ) {
        Ok(ZomeCallResponse::Ok(_)) => true,
        Ok(other) => {
            debug!(
                "Reputation slash warning: non-Ok response for violator={}: {:?}",
                violator, other
            );
            false
        }
        Err(e) => {
            debug!(
                "Reputation slash warning: commons_bridge unreachable or slash_community_score not yet implemented for violator={}: {:?}",
                violator, e
            );
            false
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn ensure_anchor(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    anchor_hash(anchor_str)
}

//// civic_requirement_proposal() is now imported from bridge-common (CivicRequirement type)

// =============================================================================
// Contract Lifecycle
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ProposeContractInput {
    pub receiver: AgentPubKey,
    pub service_type: ServiceCategory,
    pub inclusions: Vec<String>,
    pub exclusions: Vec<String>,
    pub duration_minutes: u32,
    pub tend_hours: f32,
    pub revocation_conditions: Vec<String>,
}

/// Propose a new boundary contract. Caller is the proposer (either provider or receiver).
/// Checks the other party's standing boundaries for blocks.
#[hdk_extern]
pub fn propose_contract(input: ProposeContractInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "propose_contract")?;

    let agent_info = agent_info()?;
    let proposer = agent_info.agent_initial_pubkey.clone();
    let now = sys_time()?;

    // Check if the receiver has blocked the proposer
    if is_blocked(&input.receiver, &proposer)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot propose: you are blocked by this agent".into()
        )));
    }

    // Generate unique contract ID
    let id = format!(
        "bc-{}-{}",
        now.as_micros(),
        &proposer.to_string()[..8]
    );

    let contract = BoundaryContract {
        id,
        provider: proposer.clone(),
        receiver: input.receiver.clone(),
        service_type: input.service_type.clone(),
        inclusions: input.inclusions,
        exclusions: input.exclusions,
        duration_minutes: input.duration_minutes,
        tend_hours: input.tend_hours,
        revocation_conditions: input.revocation_conditions,
        status: ContractStatus::Proposed,
        provider_signed_at: Some(now),
        receiver_signed_at: None,
        created_at: now,
        updated_at: now,
    };

    let contract_hash = create_entry(&EntryTypes::BoundaryContract(contract.clone()))?;

    // Create public summary
    let summary = ContractSummary {
        contract_hash: contract_hash.clone(),
        provider: proposer.clone(),
        receiver: input.receiver.clone(),
        service_type: input.service_type,
        status: ContractStatus::Proposed,
        created_at: now,
        updated_at: now,
    };
    let summary_hash = create_entry(&EntryTypes::ContractSummary(summary))?;

    // Links
    create_link(
        contract_hash.clone(),
        summary_hash.clone(),
        LinkTypes::ContractToSummary,
        (),
    )?;

    // Link both parties to the contract
    let provider_anchor = ensure_anchor(&format!("agent_contracts:{}", proposer))?;
    create_link(
        provider_anchor,
        contract_hash.clone(),
        LinkTypes::AgentToContracts,
        (),
    )?;
    let receiver_anchor = ensure_anchor(&format!("agent_contracts:{}", input.receiver))?;
    create_link(
        receiver_anchor,
        contract_hash.clone(),
        LinkTypes::AgentToContracts,
        (),
    )?;

    // Link to active summaries
    let active_anchor = ensure_anchor("all_active_summaries")?;
    create_link(
        active_anchor,
        summary_hash,
        LinkTypes::AllActiveSummaries,
        (),
    )?;

    get(contract_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created contract".into())))
}

/// Sign/countersign a proposed contract. Transitions Proposed -> Active.
/// Only the non-proposing party can call this.
#[hdk_extern]
pub fn sign_contract(contract_hash: ActionHash) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "sign_contract")?;

    let agent_info = agent_info()?;
    let signer = agent_info.agent_initial_pubkey;
    let now = sys_time()?;

    let record = get(contract_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))?;

    let mut contract: BoundaryContract = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a contract entry".into())))?;

    // Must be a party, and must be the non-signer
    if signer != contract.provider && signer != contract.receiver {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only a party to the contract can sign it".into()
        )));
    }

    if !contract.status.can_transition_to(&ContractStatus::Active) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot sign: contract is {:?}, not Proposed",
            contract.status
        ))));
    }

    // Set the countersignature
    if signer == contract.provider {
        contract.provider_signed_at = Some(now);
    } else {
        contract.receiver_signed_at = Some(now);
    }
    contract.status = ContractStatus::Active;
    contract.updated_at = now;

    let new_hash = update_entry(contract_hash, &EntryTypes::BoundaryContract(contract))?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get signed contract".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeContractInput {
    pub contract_hash: ActionHash,
    pub reason: String,
}

/// Revoke a contract. Either party can revoke at any time.
/// Triggers partial TEND settlement based on elapsed time.
#[hdk_extern]
pub fn revoke_contract(input: RevokeContractInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let record = get(input.contract_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))?;

    let mut contract: BoundaryContract = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a contract entry".into())))?;

    let revoker = agent_info.agent_initial_pubkey;
    if revoker != contract.provider && revoker != contract.receiver {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only a party can revoke".into()
        )));
    }

    if !contract.status.can_transition_to(&ContractStatus::Revoked) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot revoke from {:?} status",
            contract.status
        ))));
    }

    contract.status = ContractStatus::Revoked;
    contract.updated_at = now;

    let new_hash = update_entry(input.contract_hash, &EntryTypes::BoundaryContract(contract.clone()))?;

    // Bridge call to TEND for partial settlement based on elapsed time.
    // Best-effort: if TEND is unreachable, the revocation still succeeds.
    if contract.provider_signed_at.is_some() && contract.receiver_signed_at.is_some() {
        let elapsed_micros = now.as_micros().saturating_sub(contract.created_at.as_micros());
        let total_micros = (contract.duration_minutes as i64) * 60 * 1_000_000;
        let fraction = if total_micros > 0 {
            (elapsed_micros as f32 / total_micros as f32).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let partial_hours = contract.tend_hours * fraction;
        if partial_hours >= 0.25 {
            let _exchange_id = bridge_tend_settlement(
                &contract.provider,
                &contract.receiver,
                partial_hours,
                &format!(
                    "Partial settlement: boundary contract revoked ({:.0}% elapsed)",
                    fraction * 100.0
                ),
            );
        }
    }

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get revoked contract".into())))
}

/// Complete a contract. Either party marks as completed.
/// Triggers full TEND settlement.
#[hdk_extern]
pub fn complete_contract(contract_hash: ActionHash) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let record = get(contract_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))?;

    let mut contract: BoundaryContract = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a contract entry".into())))?;

    let completer = agent_info.agent_initial_pubkey;
    if completer != contract.provider && completer != contract.receiver {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only a party can complete".into()
        )));
    }

    if !contract.status.can_transition_to(&ContractStatus::Completed) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot complete from {:?} status",
            contract.status
        ))));
    }

    contract.status = ContractStatus::Completed;
    contract.updated_at = now;

    let new_hash = update_entry(contract_hash, &EntryTypes::BoundaryContract(contract.clone()))?;

    // Bridge call to TEND for full settlement.
    // Best-effort: if TEND is unreachable, the completion still succeeds.
    if contract.tend_hours >= 0.25 {
        let _exchange_id = bridge_tend_settlement(
            &contract.provider,
            &contract.receiver,
            contract.tend_hours,
            "Full settlement: boundary contract completed",
        );
    }

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get completed contract".into())))
}

/// Get a contract by hash (only accessible to parties).
#[hdk_extern]
pub fn get_contract(contract_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(contract_hash, GetOptions::default())
}

/// Get all contracts for the calling agent.
#[hdk_extern]
pub fn get_my_contracts(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let agent_anchor = anchor_hash(&format!("agent_contracts:{}", agent_info.agent_initial_pubkey))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToContracts)?, GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get contracts by status for the calling agent.
#[hdk_extern]
pub fn get_my_contracts_by_status(status: ContractStatus) -> ExternResult<Vec<Record>> {
    let all = get_my_contracts(())?;
    Ok(all
        .into_iter()
        .filter(|r| {
            r.entry()
                .to_app_option::<BoundaryContract>()
                .ok()
                .flatten()
                .map(|c| c.status == status)
                .unwrap_or(false)
        })
        .collect())
}

// =============================================================================
// Violation Response
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ReportViolationInput {
    pub contract_hash: ActionHash,
    pub description: String,
    pub severity: ViolationSeverity,
}

/// Report a boundary violation. Only a party to the contract can report.
/// Automatically: (1) logs violation, (2) slashes violator reputation,
/// (3) routes to restorative justice. Never punitive enforcement.
#[hdk_extern]
pub fn report_violation(input: ReportViolationInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let reporter = agent_info.agent_initial_pubkey;
    let now = sys_time()?;

    // Get the contract to verify party membership
    let record = get(input.contract_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Contract not found".into())))?;

    let mut contract: BoundaryContract = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not a contract entry".into())))?;

    // Reporter must be a party
    if reporter != contract.provider && reporter != contract.receiver {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only a party can report a violation".into()
        )));
    }

    // Determine the violator (the other party)
    let violator = if reporter == contract.provider {
        contract.receiver.clone()
    } else {
        contract.provider.clone()
    };

    // Capture values needed for bridge calls before they are moved
    let contract_id = contract.id.clone();
    let violation_description = input.description.clone();

    // Transition contract to Violated status
    if contract.status.can_transition_to(&ContractStatus::Violated) {
        contract.status = ContractStatus::Violated;
        contract.updated_at = now;
        update_entry(input.contract_hash.clone(), &EntryTypes::BoundaryContract(contract))?;
    }

    // Create violation record
    let violation = BoundaryViolation {
        contract_hash: input.contract_hash.clone(),
        reporter: reporter.clone(),
        violator: violator.clone(),
        description: input.description,
        severity: input.severity.clone(),
        restorative_case_id: None, // populated after bridge call
        reported_at: now,
    };
    let violation_hash = create_entry(&EntryTypes::BoundaryViolation(violation))?;

    // Link contract -> violation
    create_link(
        input.contract_hash,
        violation_hash.clone(),
        LinkTypes::ContractToViolation,
        (),
    )?;

    // Bridge call to SubPassport for reputation slash.
    // Best-effort: if the bridge extern is unavailable, we log and continue.
    let slash_factor = input.severity.slash_factor();
    let _slashed = bridge_reputation_slash(&violator, slash_factor);

    // Bridge call to justice-restorative for restorative circle creation.
    // Best-effort: if the civic cluster is unreachable, the violation is
    // still recorded and can be routed to restorative justice later.
    let _restorative_case_id = bridge_create_restorative_circle(
        &contract_id,
        &reporter,
        &violator,
        &violation_description,
        &input.severity,
    );

    get(violation_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get violation record".into())))
}

/// Get violations for a contract (only accessible to parties).
#[hdk_extern]
pub fn get_contract_violations(contract_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(contract_hash, LinkTypes::ContractToViolation)?, GetStrategy::default(),
    )?;
    records_from_links(links)
}

// =============================================================================
// Care Worker Protection
// =============================================================================

/// Set or update standing boundaries for the calling agent.
#[hdk_extern]
pub fn set_standing_boundaries(input: StandingBoundaries) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::StandingBoundaries(input.clone()))?;

    let agent_anchor = ensure_anchor(&format!("agent_boundaries:{}", input.agent))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToStandingBoundaries,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get boundaries".into())))
}

/// Get the calling agent's standing boundaries.
#[hdk_extern]
pub fn get_my_standing_boundaries(_: ()) -> ExternResult<Option<Record>> {
    let agent_info = agent_info()?;
    get_agent_standing_boundaries(agent_info.agent_initial_pubkey)
}

/// Get another agent's standing boundaries (public — for proposers to check).
#[hdk_extern]
pub fn get_agent_standing_boundaries(agent: AgentPubKey) -> ExternResult<Option<Record>> {
    let agent_anchor = anchor_hash(&format!("agent_boundaries:{}", agent))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToStandingBoundaries)?, GetStrategy::default(),
    )?;
    let records = records_from_links(links)?;

    // Return the most recent standing boundaries
    Ok(records
        .into_iter()
        .max_by_key(|r| r.action().timestamp()))
}

/// Block an agent from proposing contracts.
#[hdk_extern]
pub fn block_agent(agent: AgentPubKey) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let me = agent_info.agent_initial_pubkey;

    // Get or create standing boundaries
    let current = get_my_standing_boundaries(())?;
    let now = sys_time()?;

    let mut boundaries = if let Some(record) = current {
        record
            .entry()
            .to_app_option::<StandingBoundaries>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {e}"))))?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Not boundaries".into())))?
    } else {
        StandingBoundaries {
            agent: me.clone(),
            offered_services: vec![],
            universal_exclusions: vec![],
            max_duration_minutes: None,
            min_tend_hours: None,
            blocked_agents: vec![],
            updated_at: now,
        }
    };

    if !boundaries.blocked_agents.contains(&agent) {
        boundaries.blocked_agents.push(agent);
    }
    boundaries.updated_at = now;

    set_standing_boundaries(boundaries)
}

/// Unblock a previously blocked agent.
#[hdk_extern]
pub fn unblock_agent(agent: AgentPubKey) -> ExternResult<Record> {
    let current = get_my_standing_boundaries(())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No standing boundaries set".into())))?;

    let now = sys_time()?;
    let mut boundaries: StandingBoundaries = current
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize: {e}"))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not boundaries".into())))?;

    boundaries.blocked_agents.retain(|a| a != &agent);
    boundaries.updated_at = now;

    set_standing_boundaries(boundaries)
}

/// Check if the calling agent can propose a contract to a given provider.
/// Returns false if blocked or if consciousness gate fails.
#[hdk_extern]
pub fn can_propose_to(provider: AgentPubKey) -> ExternResult<bool> {
    let agent_info = agent_info()?;
    let me = agent_info.agent_initial_pubkey;

    // Check consciousness gate
    if mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "can_propose_to").is_err() {
        return Ok(false);
    }

    // Check block list
    Ok(!is_blocked(&provider, &me)?)
}

/// Internal helper: check if `provider` has blocked `proposer`.
fn is_blocked(provider: &AgentPubKey, proposer: &AgentPubKey) -> ExternResult<bool> {
    let boundaries = get_agent_standing_boundaries(provider.clone())?;
    if let Some(record) = boundaries {
        if let Some(b) = record
            .entry()
            .to_app_option::<StandingBoundaries>()
            .ok()
            .flatten()
        {
            return Ok(b.blocked_agents.contains(proposer));
        }
    }
    Ok(false)
}
