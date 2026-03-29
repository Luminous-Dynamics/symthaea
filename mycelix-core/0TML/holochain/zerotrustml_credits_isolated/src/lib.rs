// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
 * ZeroTrustML Credits - Holochain Currency DNA
 *
 * Rewards quality contributions to federated learning:
 * - Quality gradients (0-100 credits based on PoGQ score)
 * - Byzantine detection (50 credits)
 * - Peer validation (10 credits)
 * - Network contribution (1 credit/hour uptime)
 *
 * Features:
 * - Zero transaction costs
 * - Immutable audit trail
 * - Peer-validated credit issuance
 * - Transfer functionality
 * - Bridge escrow for inter-currency exchange (Phase 7)
 */

use hdk::prelude::*;

/// Credit entry - represents earned or transferred credits
#[hdk_entry_helper]
#[derive(Clone)]
pub struct Credit {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: EarnReason,
    pub timestamp: Timestamp,
    pub verifiers: Vec<AgentPubKey>,
}

/// Reason for earning credits
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EarnReason {
    /// Quality gradient contribution
    QualityGradient {
        pogq_score: f64,
        gradient_hash: String,
    },
    /// Detected Byzantine node
    ByzantineDetection {
        caught_node_id: u32,
        evidence_hash: String,
    },
    /// Validated peer's gradient
    PeerValidation {
        validated_node_id: u32,
        gradient_hash: String,
    },
    /// Network uptime contribution
    NetworkContribution {
        uptime_hours: u64,
    },
    /// Transfer between peers
    Transfer {
        from: Option<AgentPubKey>,
        to: Option<AgentPubKey>,
    },
}

/// Bridge escrow for inter-currency exchange (Phase 7)
#[hdk_entry_helper]
#[derive(Clone)]
pub struct BridgeEscrow {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub destination_chain: String,
    pub swap_intent: SwapIntent,
    pub lock_time: Timestamp,
    pub status: EscrowStatus,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SwapIntent {
    pub from_currency: String,
    pub to_currency: String,
    pub rate: f64,
    pub min_rate: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum EscrowStatus {
    Locked,
    Executed,
    Cancelled,
    Expired,
}

/// Credit statistics for a holder
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CreditStats {
    pub total_earned: u64,
    pub total_spent: u64,
    pub current_balance: u64,
    pub quality_gradients_count: u32,
    pub byzantine_detections_count: u32,
    pub peer_validations_count: u32,
    pub network_contributions_count: u32,
    pub average_pogq_score: f64,
}

/// Input for creating credit
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CreateCreditInput {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub earned_from: EarnReason,
    pub verifiers: Vec<AgentPubKey>,
}

/// Input for transfer
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransferInput {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub amount: u64,
}

/// Input for bridge escrow
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BridgeEscrowInput {
    pub holder: AgentPubKey,
    pub amount: u64,
    pub destination_chain: String,
    pub swap_intent: SwapIntent,
}

//==============================================================================
// Entry Definitions
//==============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Credit(Credit),
    BridgeEscrow(BridgeEscrow),
}

//==============================================================================
// Link Types
//==============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    HolderToCredit,
    EscrowToCredit,
}

//==============================================================================
// Zome Functions
//==============================================================================

/// Get credit balance for a holder
#[hdk_extern]
pub fn get_balance(holder: AgentPubKey) -> ExternResult<u64> {
    let credits = query_credits_for_holder(holder)?;

    let mut balance: i64 = 0;
    for credit in credits {
        match credit.earned_from {
            EarnReason::Transfer { from, to } => {
                // If this is a debit (from us), subtract
                if from.as_ref().map_or(false, |f| f == &credit.holder) {
                    balance -= credit.amount as i64;
                }
                // If this is a credit (to us), add
                else if to.as_ref().map_or(false, |t| t == &credit.holder) {
                    balance += credit.amount as i64;
                }
            }
            // All other reasons are earnings (positive)
            _ => {
                balance += credit.amount as i64;
            }
        }
    }

    Ok(balance.max(0) as u64)
}

/// Create credit entry (usually called by automated system)
#[hdk_extern]
pub fn create_credit(input: CreateCreditInput) -> ExternResult<ActionHash> {
    let credit = Credit {
        holder: input.holder.clone(),
        amount: input.amount,
        earned_from: input.earned_from,
        timestamp: sys_time()?,
        verifiers: input.verifiers,
    };

    let credit_hash = create_entry(EntryTypes::Credit(credit.clone()))?;

    // Create link from holder to credit
    create_link(
        input.holder,
        credit_hash.clone(),
        LinkTypes::HolderToCredit,
        (),
    )?;

    Ok(credit_hash)
}

/// Transfer credits between holders
#[hdk_extern]
pub fn transfer(input: TransferInput) -> ExternResult<ActionHash> {
    // Validate sender has sufficient balance
    let sender_balance = get_balance(input.from.clone())?;
    if sender_balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Insufficient balance: has {}, needs {}", sender_balance, input.amount)
        )));
    }

    // Create debit entry for sender
    let debit = Credit {
        holder: input.from.clone(),
        amount: input.amount,
        earned_from: EarnReason::Transfer {
            from: Some(input.from.clone()),
            to: Some(input.to.clone()),
        },
        timestamp: sys_time()?,
        verifiers: vec![],
    };

    create_entry(EntryTypes::Credit(debit.clone()))?;
    create_link(
        input.from.clone(),
        hash_entry(&debit)?,
        LinkTypes::HolderToCredit,
        (),
    )?;

    // Create credit entry for receiver
    let credit = Credit {
        holder: input.to.clone(),
        amount: input.amount,
        earned_from: EarnReason::Transfer {
            from: Some(input.from.clone()),
            to: Some(input.to.clone()),
        },
        timestamp: sys_time()?,
        verifiers: vec![],
    };

    let credit_hash = create_entry(EntryTypes::Credit(credit.clone()))?;
    create_link(
        input.to.clone(),
        credit_hash.clone(),
        LinkTypes::HolderToCredit,
        (),
    )?;

    Ok(credit_hash)
}

/// Query credits for a holder
#[hdk_extern]
pub fn query_credits(holder: AgentPubKey) -> ExternResult<Vec<Credit>> {
    query_credits_for_holder(holder)
}

/// Get audit trail (all credits with metadata)
#[hdk_extern]
pub fn get_audit_trail(holder: AgentPubKey) -> ExternResult<Vec<(Credit, ActionHash, Timestamp)>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(holder, LinkTypes::HolderToCredit)?.build(),
    )?;

    let mut audit_entries = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Ok(credit) = Credit::try_from(&record) {
                    let timestamp = record.action().timestamp();
                    audit_entries.push((credit, action_hash, timestamp));
                }
            }
        }
    }

    // Sort by timestamp (newest first)
    audit_entries.sort_by(|a, b| b.2.cmp(&a.2));

    Ok(audit_entries)
}

/// Get credit statistics for a holder
#[hdk_extern]
pub fn get_credit_statistics(holder: AgentPubKey) -> ExternResult<CreditStats> {
    let credits = query_credits_for_holder(holder.clone())?;

    let mut stats = CreditStats {
        total_earned: 0,
        total_spent: 0,
        current_balance: 0,
        quality_gradients_count: 0,
        byzantine_detections_count: 0,
        peer_validations_count: 0,
        network_contributions_count: 0,
        average_pogq_score: 0.0,
    };

    let mut pogq_scores = Vec::new();

    for credit in credits {
        match &credit.earned_from {
            EarnReason::QualityGradient { pogq_score, .. } => {
                stats.total_earned += credit.amount;
                stats.quality_gradients_count += 1;
                pogq_scores.push(*pogq_score);
            }
            EarnReason::ByzantineDetection { .. } => {
                stats.total_earned += credit.amount;
                stats.byzantine_detections_count += 1;
            }
            EarnReason::PeerValidation { .. } => {
                stats.total_earned += credit.amount;
                stats.peer_validations_count += 1;
            }
            EarnReason::NetworkContribution { .. } => {
                stats.total_earned += credit.amount;
                stats.network_contributions_count += 1;
            }
            EarnReason::Transfer { from, .. } => {
                if from.is_some() {
                    stats.total_spent += credit.amount;
                } else {
                    stats.total_earned += credit.amount;
                }
            }
        }
    }

    // Calculate average PoGQ score
    if !pogq_scores.is_empty() {
        stats.average_pogq_score = pogq_scores.iter().sum::<f64>() / pogq_scores.len() as f64;
    }

    // Calculate current balance
    stats.current_balance = get_balance(holder)?;

    Ok(stats)
}

/// Create bridge escrow for inter-currency exchange (Phase 7)
#[hdk_extern]
pub fn create_bridge_escrow(input: BridgeEscrowInput) -> ExternResult<ActionHash> {
    // Validate sufficient balance
    let balance = get_balance(input.holder.clone())?;
    if balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Insufficient balance for escrow: has {}, needs {}", balance, input.amount)
        )));
    }

    let escrow = BridgeEscrow {
        holder: input.holder.clone(),
        amount: input.amount,
        destination_chain: input.destination_chain,
        swap_intent: input.swap_intent,
        lock_time: sys_time()?,
        status: EscrowStatus::Locked,
    };

    let escrow_hash = create_entry(EntryTypes::BridgeEscrow(escrow))?;

    // Create link for query
    create_link(
        input.holder,
        escrow_hash.clone(),
        LinkTypes::EscrowToCredit,
        (),
    )?;

    // Emit signal for bridge validators
    emit_signal(&EscrowCreatedSignal {
        escrow_hash: escrow_hash.clone(),
    })?;

    Ok(escrow_hash)
}

//==============================================================================
// Validation
//==============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => {
            let entry = store_entry.entry.clone();
            // Try to deserialize as Credit first
            if let Ok(credit) = Credit::try_from(&entry) {
                return validate_credit(credit);
            }
            // Try to deserialize as BridgeEscrow
            if let Ok(escrow) = BridgeEscrow::try_from(&entry) {
                return validate_escrow(escrow);
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate credit entry
fn validate_credit(credit: Credit) -> ExternResult<ValidateCallbackResult> {
    match &credit.earned_from {
        EarnReason::QualityGradient { pogq_score, .. } => {
            // Max 100 credits per gradient
            if credit.amount > 100 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Credit amount exceeds maximum (100) for quality gradient".into()
                ));
            }

            // Must have good PoGQ score (> 0.5)
            if *pogq_score < 0.5 {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("PoGQ score {} too low for credit issuance (minimum 0.5)", pogq_score)
                ));
            }

            // Must have at least 3 verifiers
            if credit.verifiers.len() < 3 {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Insufficient verifiers: {} (minimum 3)", credit.verifiers.len())
                ));
            }
        }
        EarnReason::ByzantineDetection { .. } => {
            // Byzantine detection rewards are fixed at 50
            if credit.amount != 50 {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Invalid Byzantine detection reward: {} (must be 50)", credit.amount)
                ));
            }

            // Must have verifiers
            if credit.verifiers.len() < 2 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Insufficient verifiers for Byzantine detection".into()
                ));
            }
        }
        EarnReason::PeerValidation { .. } => {
            // Peer validation rewards are fixed at 10
            if credit.amount != 10 {
                return Ok(ValidateCallbackResult::Invalid(
                    format!("Invalid peer validation reward: {} (must be 10)", credit.amount)
                ));
            }
        }
        EarnReason::NetworkContribution { uptime_hours } => {
            // 1 credit per hour
            if credit.amount != *uptime_hours {
                return Ok(ValidateCallbackResult::Invalid(
                    "Network contribution credits must equal uptime hours".into()
                ));
            }
        }
        EarnReason::Transfer { .. } => {
            // Transfers have no verifiers
            // Balance validation happens in zome function
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate bridge escrow
fn validate_escrow(escrow: BridgeEscrow) -> ExternResult<ValidateCallbackResult> {
    // Validate amount is positive
    if escrow.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be positive".into()
        ));
    }

    // Validate rates
    if escrow.swap_intent.rate <= 0.0 || escrow.swap_intent.min_rate <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Swap rates must be positive".into()
        ));
    }

    if escrow.swap_intent.min_rate > escrow.swap_intent.rate {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum rate cannot exceed target rate".into()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

//==============================================================================
// Helper Functions
//==============================================================================

/// Query all credits for a holder
fn query_credits_for_holder(holder: AgentPubKey) -> ExternResult<Vec<Credit>> {
    let links = get_links(
        GetLinksInputBuilder::try_new(holder, LinkTypes::HolderToCredit)?.build(),
    )?;

    let mut credits = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Ok(credit) = Credit::try_from(&record) {
                    credits.push(credit);
                }
            }
        }
    }

    Ok(credits)
}

//==============================================================================
// Signals
//==============================================================================

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EscrowCreatedSignal {
    pub escrow_hash: ActionHash,
}