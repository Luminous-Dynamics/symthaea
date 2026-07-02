// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use hdk::prelude::*;
use mycelix_common::{bridge, error_handling, link_queries, remote_calls, time};
use transactions_integrity::*;

/// Application ID for Bridge reputation reporting
const APP_ID: &str = "mycelix-marketplace";

/// Create a new transaction (buyer initiates purchase)
///
/// This starts the transaction lifecycle. The buyer creates the transaction
/// in Pending state, and the seller must confirm it.
#[hdk_extern]
pub fn create_transaction(input: CreateTransactionInput) -> ExternResult<TransactionOutput> {
    let agent_info = agent_info()?;

    // Create transaction entry
    let transaction = Transaction {
        buyer: agent_info.agent_initial_pubkey.clone(),
        seller: input.seller.clone(),
        listing_hash: input.listing_hash.clone(),
        quantity: input.quantity,
        total_price_cents: input.total_price_cents,
        status: TransactionStatus::Pending,
        created_at: time::now()?,
        updated_at: time::now()?,
        tracking_info: None,
        epistemic: EpistemicClassification {
            // Transaction starts as testimonial (E1)
            empirical: EmpiricalLevel::E1Testimonial,
            // Communal agreement between buyer-seller (N1)
            normative: NormativeLevel::N1Communal,
            // Temporal during transaction (M1)
            materiality: MaterialityLevel::M1Temporal,
        },
    };

    let action_hash = create_entry(&EntryTypes::Transaction(transaction.clone()))?;

    // Create links for discovery
    create_link(
        transaction.buyer.clone(),
        action_hash.clone(),
        LinkTypes::BuyerToTransactions,
        (),
    )?;

    create_link(
        transaction.seller.clone(),
        action_hash.clone(),
        LinkTypes::SellerToTransactions,
        (),
    )?;

    create_link(
        transaction.listing_hash.clone(),
        action_hash.clone(),
        LinkTypes::ListingToTransactions,
        (),
    )?;

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::TransactionCreated,
        transaction.total_price_cents as f64,
        Some(transaction.buyer.clone()),
        Some(format!(
            "seller:{:?},quantity:{}",
            transaction.seller, transaction.quantity
        )),
    )?;

    Ok(TransactionOutput {
        transaction_hash: action_hash,
        transaction,
    })
}

/// Get a transaction by hash
#[hdk_extern]
pub fn get_transaction(transaction_hash: ActionHash) -> ExternResult<Option<TransactionOutput>> {
    let record = get(transaction_hash.clone(), GetOptions::default())?;

    match record {
        Some(record) => {
            // Use shared utility for deserialization
            let transaction: Transaction = error_handling::deserialize_entry(&record)?;

            Ok(Some(TransactionOutput {
                transaction_hash,
                transaction,
            }))
        }
        None => Ok(None),
    }
}

/// Get all transactions for the current user (as buyer or seller)
#[hdk_extern]
pub fn get_my_transactions(_: ()) -> ExternResult<TransactionsResponse> {
    let agent_info = agent_info()?;
    let agent = agent_info.agent_initial_pubkey;

    let mut transactions = Vec::new();

    // Get transactions as buyer
    // Use shared utility for get_links
    let buyer_links = link_queries::get_links_local(agent.clone(), LinkTypes::BuyerToTransactions)?;

    for link in buyer_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(output) = get_transaction(action_hash)? {
                transactions.push(output);
            }
        }
    }

    // Get transactions as seller
    // Use shared utility for get_links
    let seller_links = link_queries::get_links_local(agent, LinkTypes::SellerToTransactions)?;

    for link in seller_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(output) = get_transaction(action_hash)? {
                // Avoid duplicates (in case same agent is buyer and seller)
                if !transactions
                    .iter()
                    .any(|t| t.transaction_hash == output.transaction_hash)
                {
                    transactions.push(output);
                }
            }
        }
    }

    Ok(TransactionsResponse { transactions })
}

/// Seller confirms the transaction
///
/// State transition: Pending → Confirmed
#[hdk_extern]
pub fn confirm_transaction(transaction_hash: ActionHash) -> ExternResult<TransactionOutput> {
    update_transaction_status(
        transaction_hash,
        TransactionStatus::Confirmed,
        None,
        vec![TransactionStatus::Pending],
        RequiredParty::Seller,
    )
}

/// Seller marks transaction as shipped
///
/// State transition: Confirmed → Shipped
#[hdk_extern]
pub fn mark_shipped(input: MarkShippedInput) -> ExternResult<TransactionOutput> {
    update_transaction_status(
        input.transaction_hash,
        TransactionStatus::Shipped,
        input.tracking_info,
        vec![TransactionStatus::Confirmed],
        RequiredParty::Seller,
    )
}

/// Buyer confirms delivery
///
/// State transition: Shipped → Delivered
#[hdk_extern]
pub fn confirm_delivery(transaction_hash: ActionHash) -> ExternResult<TransactionOutput> {
    update_transaction_status(
        transaction_hash,
        TransactionStatus::Delivered,
        None,
        vec![TransactionStatus::Shipped],
        RequiredParty::Buyer,
    )
}

/// Complete the transaction
///
/// State transition: Delivered → Completed
/// This triggers MATL score updates for both buyer and seller
/// and reports the transaction to Bridge for cross-app reputation sharing.
#[hdk_extern]
pub fn complete_transaction(transaction_hash: ActionHash) -> ExternResult<TransactionOutput> {
    // Get current transaction
    let current = get_transaction(transaction_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Transaction not found".into())
    ))?;

    // Verify state transition is valid
    if current.transaction.status != TransactionStatus::Delivered {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot complete transaction from status {:?}",
            current.transaction.status
        ))));
    }

    // Settle in the finance cluster BEFORE transitioning to Completed.
    // settle_transaction_in_finance previously existed but was never called
    // from anywhere in the transaction lifecycle — marketplace transactions
    // completed purely on local state with zero SAP ever moving, success or
    // failure notwithstanding. This makes real settlement a precondition
    // for completion: the transaction stays in Delivered (unchanged,
    // retriable) if settlement fails, rather than silently completing
    // unpaid. See MYCELIX_REVIEW.md P1 #4.
    //
    // NOTE: this will currently fail with a role-not-found error in any
    // deployment where marketplace's happ.yaml doesn't declare a finance
    // role (the case as of 2026-04-17 per
    // mycelix-workspace/happs/happ.yaml's "deliberately excluded" note) —
    // that's a separate, ops-level deployment-bundle decision, not
    // something this code fix can resolve on its own. The fix here makes
    // completion CORRECT once marketplace and finance are actually
    // deployed together; it does not by itself change what's deployed.
    let settlement = settle_transaction_in_finance(transaction_hash.clone())?;
    if !settlement.settled {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot complete transaction: finance settlement failed ({}). Transaction remains in \
             Delivered status and can be retried.",
            settlement
                .error
                .unwrap_or_else(|| "unknown error".to_string())
        ))));
    }

    // Update transaction status
    let mut updated_transaction = current.transaction.clone();
    updated_transaction.status = TransactionStatus::Completed;
    updated_transaction.updated_at = time::now()?;
    updated_transaction.epistemic.materiality = MaterialityLevel::M2Persistent;

    let new_action_hash = update_entry(transaction_hash, &updated_transaction)?;

    // Call reputation zome to update MATL scores
    // This is where the 45% Byzantine tolerance magic happens!
    // Use shared utility for remote calls
    remote_calls::call_zome_void(
        "reputation",
        "update_matl_score",
        UpdateMatlInput {
            agent: updated_transaction.seller.clone(),
            successful: true,
            transaction_value_cents: updated_transaction.total_price_cents,
        },
    )?;

    // Report to Bridge for cross-app reputation sharing
    // This allows the seller's reputation to be visible in other apps
    let seller_did = bridge::agent_to_did(&updated_transaction.seller);
    let bridge_report = bridge::ReportReputationInput {
        did: seller_did,
        positive: true,
        value_cents: updated_transaction.total_price_cents,
        app_id: APP_ID.to_string(),
        context: Some(format!("marketplace_transaction:{}", new_action_hash)),
    };

    // Report seller reputation (gracefully handle Bridge unavailability)
    match bridge::report_reputation(bridge_report) {
        bridge::BridgeResult::Success(_) => {
            // Successfully reported to Bridge
        }
        bridge::BridgeResult::Unavailable => {
            // Bridge not available - this is fine, local reputation still works
            // Could emit a debug metric here if desired
        }
        bridge::BridgeResult::Error(e) => {
            // Log error but don't fail the transaction
            // The local reputation update already succeeded
            monitoring::emit_metric(
                monitoring::MetricType::BridgeError,
                0.0,
                Some(updated_transaction.seller.clone()),
                Some(format!("bridge_report_error:{}", e)),
            )?;
        }
    }

    // Also report buyer reputation (positive for completing payment)
    let buyer_did = bridge::agent_to_did(&updated_transaction.buyer);
    let buyer_report = bridge::ReportReputationInput {
        did: buyer_did,
        positive: true,
        value_cents: updated_transaction.total_price_cents,
        app_id: APP_ID.to_string(),
        context: Some("buyer_completed_transaction".to_string()),
    };

    // Report buyer reputation (gracefully handle Bridge unavailability)
    let _ = bridge::report_reputation(buyer_report); // Ignore result for buyer

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::TransactionCompleted,
        updated_transaction.total_price_cents as f64,
        Some(updated_transaction.buyer.clone()),
        Some(format!("seller:{:?}", updated_transaction.seller)),
    )?;

    Ok(TransactionOutput {
        transaction_hash: new_action_hash,
        transaction: updated_transaction,
    })
}

/// Dispute a transaction
///
/// State transition: Any (except Completed/Cancelled) → Disputed
/// Reports negative reputation signal to Bridge (pending arbitration outcome).
#[hdk_extern]
pub fn dispute_transaction(input: DisputeTransactionInput) -> ExternResult<TransactionOutput> {
    // Get current transaction
    let current = get_transaction(input.transaction_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Transaction not found".into())
    ))?;

    let agent_info = agent_info()?;
    let caller = agent_info.agent_initial_pubkey;

    // Verify caller is buyer or seller
    if caller != current.transaction.buyer && caller != current.transaction.seller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only buyer or seller can dispute transaction".into()
        )));
    }

    // Cannot dispute completed or cancelled transactions
    if current.transaction.status == TransactionStatus::Completed
        || current.transaction.status == TransactionStatus::Cancelled
    {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot dispute transaction with status {:?}",
            current.transaction.status
        ))));
    }

    // Update transaction status
    let mut updated_transaction = current.transaction.clone();
    updated_transaction.status = TransactionStatus::Disputed;
    updated_transaction.updated_at = time::now()?;

    let new_action_hash = update_entry(input.transaction_hash, &updated_transaction)?;

    // Store dispute reason (linked to transaction)
    // This will be used by the arbitration zome
    create_link(
        new_action_hash.clone(),
        new_action_hash.clone(),
        LinkTypes::ListingToTransactions, // Reusing link type for simplicity
        (),
    )?;

    // Report dispute to Bridge (negative signal, pending arbitration)
    // The other party (not the caller) gets a potential negative mark
    // This will be resolved when arbitration completes
    let other_party = if caller == current.transaction.buyer {
        &current.transaction.seller
    } else {
        &current.transaction.buyer
    };

    let other_did = bridge::agent_to_did(other_party);
    let dispute_report = bridge::ReportReputationInput {
        did: other_did,
        positive: false, // Negative signal for dispute
        value_cents: updated_transaction.total_price_cents,
        app_id: APP_ID.to_string(),
        context: Some(format!(
            "dispute_filed:{}:reason:{}",
            new_action_hash, input.reason
        )),
    };

    // Report to Bridge (gracefully handle unavailability)
    let _ = bridge::report_reputation(dispute_report);

    // Emit monitoring metric
    monitoring::emit_metric(
        monitoring::MetricType::TransactionDisputed,
        updated_transaction.total_price_cents as f64,
        Some(caller),
        Some(format!(
            "buyer:{:?},seller:{:?}",
            updated_transaction.buyer, updated_transaction.seller
        )),
    )?;

    Ok(TransactionOutput {
        transaction_hash: new_action_hash,
        transaction: updated_transaction,
    })
}

/// Cancel a transaction
///
/// State transition: Pending/Confirmed → Cancelled
#[hdk_extern]
pub fn cancel_transaction(transaction_hash: ActionHash) -> ExternResult<TransactionOutput> {
    // Get current transaction
    let current = get_transaction(transaction_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Transaction not found".into())
    ))?;

    let agent_info = agent_info()?;
    let caller = agent_info.agent_initial_pubkey;

    // Verify caller is buyer or seller
    if caller != current.transaction.buyer && caller != current.transaction.seller {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only buyer or seller can cancel transaction".into()
        )));
    }

    // Can only cancel from Pending or Confirmed states
    match current.transaction.status {
        TransactionStatus::Pending | TransactionStatus::Confirmed => {}
        _ => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cannot cancel transaction from status {:?}",
                current.transaction.status
            ))));
        }
    }

    // Update transaction status
    let mut updated_transaction = current.transaction.clone();
    updated_transaction.status = TransactionStatus::Cancelled;
    updated_transaction.updated_at = time::now()?;

    let new_action_hash = update_entry(transaction_hash, &updated_transaction)?;

    Ok(TransactionOutput {
        transaction_hash: new_action_hash,
        transaction: updated_transaction,
    })
}

/// Get transactions for a specific listing
#[hdk_extern]
pub fn get_listing_transactions(listing_hash: ActionHash) -> ExternResult<TransactionsResponse> {
    // Use shared utility for get_links
    let links = link_queries::get_links_local(listing_hash, LinkTypes::ListingToTransactions)?;

    let mut transactions = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(output) = get_transaction(action_hash)? {
                transactions.push(output);
            }
        }
    }

    Ok(TransactionsResponse { transactions })
}

// ===== Gap 2: Finance Settlement Bridge =====

/// Result of attempting to settle a marketplace transaction in the finance cluster.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransactionSettlementResult {
    pub settled: bool,
    pub finance_reference: Option<String>,
    pub error: Option<String>,
}

/// Settle a completed marketplace transaction in the finance cluster.
///
/// Looks up the local transaction by `transaction_hash`, builds a cross-hApp
/// payment payload, and calls the finance cluster via `CallTargetCell::OtherRole`.
/// If the finance cluster is not installed the call will return `Err`, which is
/// caught and surfaced as a non-fatal result.
pub fn settle_transaction_in_finance(
    transaction_hash: ActionHash,
) -> ExternResult<TransactionSettlementResult> {
    let output = match get_transaction(transaction_hash.clone())? {
        Some(o) => o,
        None => {
            return Ok(TransactionSettlementResult {
                settled: false,
                finance_reference: None,
                error: Some(format!("Transaction {} not found", transaction_hash)),
            });
        }
    };

    let tx = &output.transaction;

    // Convert AgentPubKey to a DID string. MUST be "did:mycelix:{agent}" —
    // finance's verify_caller_is_did compares this exact string against
    // format!("did:mycelix:{}", agent_info()?.agent_initial_pubkey)
    // (mycelix-finance/zomes/shared/src/lib.rs). Neither "did:key:{:?}"
    // (Debug format, not even the same shape as Display) nor
    // bridge::agent_to_did's "did:holo:{agent}" match this — every
    // settlement attempt was guaranteed to fail auth regardless of Phase 1's
    // process_payment fix. See MYCELIX_REVIEW.md P1 #4.
    let buyer_did = format!("did:mycelix:{}", tx.buyer);
    let seller_did = format!("did:mycelix:{}", tx.seller);

    let payload = serde_json::json!({
        "source_happ": "mycelix-marketplace",
        "from_did": buyer_did,
        "to_did": seller_did,
        "amount": tx.total_price_cents,
        "currency": "SAP",
        "reference": format!("marketplace_tx:{}", transaction_hash),
    });

    let encoded =
        ExternIO::encode(payload).map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("finance_bridge"),
        FunctionName::from("process_payment"),
        None,
        encoded,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            let value: serde_json::Value = data.decode().unwrap_or(serde_json::Value::Null);
            let finance_reference = value
                .get("entry")
                .and_then(|e| e.get("id"))
                .and_then(|id| id.as_str())
                .map(|s| s.to_string());
            Ok(TransactionSettlementResult {
                settled: true,
                finance_reference,
                error: None,
            })
        }
        Ok(other) => Ok(TransactionSettlementResult {
            settled: false,
            finance_reference: None,
            error: Some(format!("Finance cluster rejected settlement: {:?}", other)),
        }),
        Err(_) => Ok(TransactionSettlementResult {
            settled: false,
            finance_reference: None,
            error: Some("Finance cluster not available".to_string()),
        }),
    }
}

// ===== Helper Functions =====

/// Update transaction status with validation
/// Which party in the transaction is required to be the caller for a
/// given state transition.
enum RequiredParty {
    Buyer,
    Seller,
}

fn update_transaction_status(
    transaction_hash: ActionHash,
    new_status: TransactionStatus,
    tracking_info: Option<String>,
    allowed_from_states: Vec<TransactionStatus>,
    required_caller: RequiredParty,
) -> ExternResult<TransactionOutput> {
    // Get current transaction
    let current = get_transaction(transaction_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Transaction not found".into())
    ))?;

    // Verify caller is the party authorized for this transition — mirrors
    // the buyer/seller check already used correctly by dispute_transaction
    // and cancel_transaction.
    let caller = agent_info()?.agent_initial_pubkey;
    let expected = match required_caller {
        RequiredParty::Buyer => &current.transaction.buyer,
        RequiredParty::Seller => &current.transaction.seller,
    };
    if caller != *expected {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Only the {} may perform this transition",
            match required_caller {
                RequiredParty::Buyer => "buyer",
                RequiredParty::Seller => "seller",
            }
        ))));
    }

    // Verify state transition is valid
    if !allowed_from_states.contains(&current.transaction.status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid state transition from {:?} to {:?}",
            current.transaction.status, new_status
        ))));
    }

    // Create updated transaction
    let mut updated_transaction = current.transaction.clone();
    updated_transaction.status = new_status;
    updated_transaction.updated_at = time::now()?;

    if let Some(info) = tracking_info {
        updated_transaction.tracking_info = Some(info);
    }

    // Update entry
    let new_action_hash = update_entry(transaction_hash, &updated_transaction)?;

    Ok(TransactionOutput {
        transaction_hash: new_action_hash,
        transaction: updated_transaction,
    })
}

// ===== Input/Output Types =====

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateTransactionInput {
    pub seller: AgentPubKey,
    pub listing_hash: ActionHash,
    pub quantity: u32,
    pub total_price_cents: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransactionOutput {
    pub transaction_hash: ActionHash,
    pub transaction: Transaction,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TransactionsResponse {
    pub transactions: Vec<TransactionOutput>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MarkShippedInput {
    pub transaction_hash: ActionHash,
    pub tracking_info: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DisputeTransactionInput {
    pub transaction_hash: ActionHash,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateMatlInput {
    pub agent: AgentPubKey,
    pub successful: bool,
    pub transaction_value_cents: u64,
}

// ===== Tests =====
#[cfg(test)]
mod tests;
