// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Payment distribution integration for contributor rewards.
//!
//! Bridges FL round completion to PaymentRouter via Shapley value computation.

use hdk::prelude::*;
use federated_learning_integrity::*;

use crate::auth::*;
use crate::bridge::call_bridge_broadcast_event;
use crate::gradients::{get_round_gradients, get_round_byzantine_records};
use super::ensure_path;

/// Event type for payment distribution
pub(crate) const EVENT_PAYMENT_DISTRIBUTED: &str = "fl_payment_distributed";

/// Input for triggering payment distribution after round completion
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TriggerPaymentDistributionInput {
    /// Round number
    pub round: u32,
    /// Model ID for payment routing
    pub model_id: String,
    /// Total payment amount in wei (as string for large numbers)
    pub total_amount_wei: String,
    /// Optional: Override Shapley computation with pre-computed values
    pub shapley_overrides: Option<Vec<(String, f64)>>,
}

/// Result of payment distribution trigger
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaymentDistributionResult {
    /// Round number
    pub round: u32,
    /// Model ID
    pub model_id: String,
    /// Number of recipients
    pub recipient_count: usize,
    /// Payment splits (node_id -> basis points)
    pub payment_splits: Vec<PaymentSplit>,
    /// Total basis points distributed (should sum to 10000 minus platform fee)
    pub total_basis_points: u64,
    /// Platform fee in basis points
    pub platform_fee_bps: u64,
    /// Ethereum transaction hash (if submitted)
    pub tx_hash: Option<String>,
    /// Status of the distribution
    pub status: PaymentDistributionStatus,
    /// Bridge event hash (if broadcast)
    pub bridge_event_hash: Option<ActionHash>,
}

/// Individual payment split for a contributor
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaymentSplit {
    /// Node/agent identifier
    pub node_id: String,
    /// Ethereum address for payment
    pub eth_address: Option<String>,
    /// Shapley value (0.0 to 1.0)
    pub shapley_value: f64,
    /// Payment amount in basis points (0-10000)
    pub basis_points: u64,
    /// Was this contributor flagged as Byzantine?
    pub is_byzantine: bool,
}

/// Status of payment distribution
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentDistributionStatus {
    /// Shapley values computed, ready to submit
    Computed,
    /// Transaction submitted to Ethereum
    Submitted,
    /// Transaction confirmed on Ethereum
    Confirmed,
    /// Distribution failed
    Failed { reason: String },
    /// Skipped (no valid contributors)
    Skipped { reason: String },
}

/// Trigger payment distribution for a completed round
///
/// This function:
/// 1. Retrieves all gradients for the round
/// 2. Computes Shapley values for fair contribution attribution
/// 3. Excludes Byzantine contributors (0 allocation)
/// 4. Converts values to basis points for PaymentRouter
/// 5. Broadcasts the distribution event via Bridge
///
/// SECURITY (F-03): Requires coordinator role
#[hdk_extern]
pub fn trigger_payment_distribution(input: TriggerPaymentDistributionInput) -> ExternResult<PaymentDistributionResult> {
    // F-03: Authorization check - only coordinators can trigger payments
    require_coordinator_role()?;

    let round = input.round;
    let model_id = input.model_id.clone();

    // Get gradients for this round
    let gradients = get_round_gradients(round)?;

    if gradients.is_empty() {
        return Ok(PaymentDistributionResult {
            round,
            model_id,
            recipient_count: 0,
            payment_splits: vec![],
            total_basis_points: 0,
            platform_fee_bps: 0,
            tx_hash: None,
            status: PaymentDistributionStatus::Skipped {
                reason: "No gradients submitted for round".to_string()
            },
            bridge_event_hash: None,
        });
    }

    // Get Byzantine records for this round
    let byzantine_records = get_round_byzantine_records(round)?;
    let byzantine_nodes: std::collections::HashSet<String> = byzantine_records
        .iter()
        .map(|r| r.node_id.clone())
        .collect();

    // Compute Shapley values
    let shapley_values = if let Some(overrides) = input.shapley_overrides {
        // Use provided overrides
        overrides.into_iter().collect::<std::collections::HashMap<_, _>>()
    } else {
        // Compute from gradients
        compute_shapley_values_from_gradients(&gradients, &byzantine_nodes)?
    };

    // Platform fee: 5% (500 basis points)
    let platform_fee_bps: u64 = 500;
    let available_bps: u64 = 10000 - platform_fee_bps;

    // Convert Shapley values to payment splits
    let mut payment_splits: Vec<PaymentSplit> = Vec::new();
    let mut total_valid_shapley = 0.0;

    // First pass: sum valid Shapley values (excluding Byzantine)
    for (node_id, shapley) in &shapley_values {
        if !byzantine_nodes.contains(node_id) && *shapley > 0.0 {
            total_valid_shapley += shapley;
        }
    }

    // Second pass: compute basis points
    for (node_id, shapley) in &shapley_values {
        let is_byzantine = byzantine_nodes.contains(node_id);

        let basis_points = if is_byzantine || *shapley <= 0.0 || total_valid_shapley == 0.0 {
            0
        } else {
            // Normalize and convert to basis points
            ((shapley / total_valid_shapley) * available_bps as f64).round() as u64
        };

        // Try to get Ethereum address from node registration
        let eth_address = get_node_eth_address(node_id);

        payment_splits.push(PaymentSplit {
            node_id: node_id.clone(),
            eth_address,
            shapley_value: *shapley,
            basis_points,
            is_byzantine,
        });
    }

    // Calculate actual distributed basis points
    let total_basis_points: u64 = payment_splits.iter().map(|s| s.basis_points).sum();

    // Filter to only those with allocations
    let recipients: Vec<_> = payment_splits.iter()
        .filter(|s| s.basis_points > 0)
        .collect();

    // Broadcast payment distribution event via Bridge
    let event_payload = serde_json::json!({
        "round": round,
        "model_id": model_id,
        "total_amount_wei": input.total_amount_wei,
        "recipient_count": recipients.len(),
        "total_basis_points": total_basis_points,
        "platform_fee_bps": platform_fee_bps,
        "splits": payment_splits.iter()
            .filter(|s| s.basis_points > 0)
            .map(|s| serde_json::json!({
                "node_id": s.node_id,
                "eth_address": s.eth_address,
                "basis_points": s.basis_points,
            }))
            .collect::<Vec<_>>(),
    });

    let bridge_event_hash = call_bridge_broadcast_event(
        EVENT_PAYMENT_DISTRIBUTED,
        &event_payload,
        1, // High priority
    )?;

    Ok(PaymentDistributionResult {
        round,
        model_id,
        recipient_count: recipients.len(),
        payment_splits,
        total_basis_points,
        platform_fee_bps,
        tx_hash: None, // Set by off-chain service after submission
        status: PaymentDistributionStatus::Computed,
        bridge_event_hash,
    })
}

/// Maximum participants for Shapley computation in WASM.
/// Beyond this, MonteCarlo Shapley is too expensive for the instruction budget.
const MAX_SHAPLEY_PARTICIPANTS: usize = 20;

/// Compute Shapley values for round contributors using real ShapleyCalculator.
///
/// Uses MonteCarlo sampling (100 samples) for n <= 20 participants.
/// Falls back to equal-share for n > 20 or if HV data is unavailable.
/// Byzantine contributors always receive 0 value.
pub(crate) fn compute_shapley_values_from_gradients(
    gradients: &[(ActionHash, ModelGradient)],
    byzantine_nodes: &std::collections::HashSet<String>,
) -> ExternResult<std::collections::HashMap<String, f64>> {
    use mycelix_fl::fl_core::{ShapleyCalculator, ShapleyConfig};

    let mut shapley_values = std::collections::HashMap::new();

    if gradients.is_empty() {
        return Ok(shapley_values);
    }

    // Filter to valid (non-Byzantine) participants
    let valid_gradients: Vec<_> = gradients.iter()
        .filter(|(_, g)| !byzantine_nodes.contains(&g.node_id))
        .collect();
    let valid_count = valid_gradients.len();

    if valid_count == 0 {
        // All participants are Byzantine — assign 0 to everyone
        for (_, gradient) in gradients {
            shapley_values.insert(gradient.node_id.clone(), 0.0);
        }
        return Ok(shapley_values);
    }

    // Attempt real Shapley computation if we can fetch HV data and n is manageable
    let round = gradients.first().map(|(_, g)| g.round).unwrap_or(0);
    let hv_data = crate::hyperfeel::get_round_hypervectors(round).ok();

    let use_real_shapley = valid_count <= MAX_SHAPLEY_PARTICIPANTS
        && hv_data.as_ref().map_or(false, |hvs| !hvs.is_empty());

    if use_real_shapley {
        let hvs = hv_data.as_ref().expect("use_real_shapley guard ensures hv_data is Some");

        // Build gradient_map: convert HV16 bytes to bipolar f32 for Shapley
        let mut gradient_map: std::collections::HashMap<String, Vec<f32>> =
            std::collections::HashMap::new();

        for (node_id, hv_bytes) in hvs.iter() {
            if byzantine_nodes.contains(node_id.as_str()) {
                continue;
            }
            gradient_map.insert(node_id.clone(), crate::pipeline::hv16_to_bipolar(hv_bytes));
        }

        if gradient_map.len() >= 2 {
            // Compute aggregated reference (reputation-weighted mean of valid HVs)
            let dim = gradient_map.values().next().map_or(0, |v| v.len());
            let mut aggregated = vec![0.0f32; dim];

            // Look up reputations for weighting
            let mut weight_sum = 0.0f32;
            for (node_id, vals) in &gradient_map {
                let rep = match crate::pipeline::get_or_create_reputation(node_id) {
                    Ok(r) => (r.reputation_score as f32).max(0.1), // floor at 0.1 to avoid zero-weight
                    Err(_) => 0.5, // default for unknown nodes
                };
                weight_sum += rep;
                for (i, v) in vals.iter().enumerate() {
                    if i < dim {
                        aggregated[i] += v * rep;
                    }
                }
            }
            // Normalize by total weight
            if weight_sum > 0.0 {
                for v in &mut aggregated {
                    *v /= weight_sum;
                }
            }

            // Run Shapley: MonteCarlo with 100 samples + normalization for WASM budget
            let config = ShapleyConfig::monte_carlo(100).with_normalization();
            let mut calculator = ShapleyCalculator::new(config);
            let result = calculator.calculate_values(&gradient_map, &aggregated);

            // Populate values: real Shapley for valid, 0 for Byzantine
            for (_, gradient) in gradients {
                if byzantine_nodes.contains(&gradient.node_id) {
                    shapley_values.insert(gradient.node_id.clone(), 0.0);
                } else {
                    let sv = result.get(&gradient.node_id).unwrap_or(0.0) as f64;
                    shapley_values.insert(gradient.node_id.clone(), sv);
                }
            }
            return Ok(shapley_values);
        }
    }

    // Fallback for large rounds or missing HV data:
    // Use marginal-contribution sampling (lightweight Shapley approximation).
    // For each participant, estimate their contribution by comparing the aggregated
    // result with and without their gradient, using cosine similarity as the utility.
    if let Some(ref hvs) = hv_data {
        if !hvs.is_empty() {
            let mut gradient_map: std::collections::HashMap<String, Vec<f32>> =
                std::collections::HashMap::new();
            for (node_id, hv_bytes) in hvs.iter() {
                if !byzantine_nodes.contains(node_id.as_str()) {
                    gradient_map.insert(node_id.clone(), crate::pipeline::hv16_to_bipolar(hv_bytes));
                }
            }

            if gradient_map.len() >= 2 {
                let dim = gradient_map.values().next().map_or(0, |v| v.len());
                // Compute full aggregation (mean of all)
                let mut full_agg = vec![0.0f32; dim];
                let n = gradient_map.len() as f32;
                for vals in gradient_map.values() {
                    for (i, v) in vals.iter().enumerate() {
                        if i < dim { full_agg[i] += v / n; }
                    }
                }

                // For each participant, compute leave-one-out aggregation
                // Marginal contribution = cosine_sim(full_agg) - cosine_sim(leave_one_out_agg)
                // This is O(n * d) — feasible even for n > 20
                for (_, gradient) in gradients {
                    if byzantine_nodes.contains(&gradient.node_id) {
                        shapley_values.insert(gradient.node_id.clone(), 0.0);
                        continue;
                    }
                    if let Some(participant_hv) = gradient_map.get(&gradient.node_id) {
                        // Leave-one-out: remove this participant's contribution from the mean
                        let n_minus_1 = n - 1.0;
                        if n_minus_1 > 0.0 {
                            let mut loo_agg = vec![0.0f32; dim];
                            for (i, v) in full_agg.iter().enumerate() {
                                loo_agg[i] = (v * n - participant_hv[i]) / n_minus_1;
                            }
                            // Utility = cosine similarity between aggregation and ideal (all 1s as proxy)
                            let norm_full: f32 = full_agg.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let norm_loo: f32 = loo_agg.iter().map(|x| x * x).sum::<f32>().sqrt();
                            let dot_full: f32 = full_agg.iter().sum();
                            let dot_loo: f32 = loo_agg.iter().sum();
                            let sim_full = if norm_full > 0.0 { dot_full / (norm_full * (dim as f32).sqrt()) } else { 0.0 };
                            let sim_loo = if norm_loo > 0.0 { dot_loo / (norm_loo * (dim as f32).sqrt()) } else { 0.0 };
                            let marginal = (sim_full - sim_loo).max(0.0) as f64;
                            shapley_values.insert(gradient.node_id.clone(), marginal);
                        } else {
                            // Only one participant left — they get full credit
                            shapley_values.insert(gradient.node_id.clone(), 1.0);
                        }
                    } else {
                        shapley_values.insert(gradient.node_id.clone(), 0.0);
                    }
                }
                return Ok(shapley_values);
            }
        }
    }

    // Final fallback: equal-share when no HV data is available at all
    let equal_share = 1.0 / valid_count as f64;
    for (_, gradient) in gradients {
        if byzantine_nodes.contains(&gradient.node_id) {
            shapley_values.insert(gradient.node_id.clone(), 0.0);
        } else {
            let quality_multiplier = gradient.trust_score.unwrap_or(1.0) as f64;
            shapley_values.insert(gradient.node_id.clone(), equal_share * quality_multiplier);
        }
    }

    Ok(shapley_values)
}

/// Get Ethereum address for a node (if registered)
pub(crate) fn get_node_eth_address(node_id: &str) -> Option<String> {
    // Look up from node registration
    // For now, return None - actual lookup requires additional storage
    let _ = node_id;
    None
}

/// Input for getting payment distribution history
#[derive(Serialize, Deserialize, Debug)]
pub struct GetPaymentHistoryInput {
    pub round: Option<u32>,
    pub node_id: Option<String>,
    pub limit: Option<usize>,
}

/// Get payment distribution history for audit
#[hdk_extern]
pub fn get_payment_distribution_history(input: GetPaymentHistoryInput) -> ExternResult<Vec<PaymentDistributionResult>> {
    let audit_path = Path::from("audit_log_chain");
    let audit_hash = match audit_path.clone().typed(LinkTypes::AuditLogChain) {
        Ok(typed) => {
            if !typed.exists()? {
                return Ok(vec![]);
            }
            typed.path_entry_hash()?
        }
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::new(
            audit_hash,
            LinkTypeFilter::single_type(0.into(), (LinkTypes::AuditLogChain as u8).into()),
        ),
        GetStrategy::default(),
    )?;

    let limit = input.limit.unwrap_or(100);
    let mut results = Vec::new();

    for link in links.iter().rev() {
        if results.len() >= limit {
            break;
        }

        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(Entry::App(bytes)) = record.entry().as_option() {
                    if let Ok(log_entry) = CoordinatorAuditLog::try_from(
                        SerializedBytes::from(UnsafeBytes::from(bytes.bytes().to_vec()))
                    ) {
                        // Filter for payment distribution actions only
                        if log_entry.action_type != "trigger_payment_distribution" {
                            continue;
                        }

                        // Parse the context JSON to reconstruct PaymentDistributionResult
                        if let Ok(context) = serde_json::from_str::<serde_json::Value>(&log_entry.context_json) {
                            let round = context.get("round")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as u32;
                            let model_id = context.get("model_id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();

                            // Apply round filter
                            if let Some(filter_round) = input.round {
                                if round != filter_round {
                                    continue;
                                }
                            }

                            // Parse payment splits from context
                            let payment_splits: Vec<PaymentSplit> = context.get("splits")
                                .and_then(|v| serde_json::from_value(v.clone()).ok())
                                .unwrap_or_default();

                            // Apply node_id filter
                            if let Some(ref filter_node) = input.node_id {
                                if !payment_splits.iter().any(|s| &s.node_id == filter_node) {
                                    continue;
                                }
                            }

                            let recipient_count = context.get("recipient_count")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0) as usize;
                            let total_basis_points = context.get("total_basis_points")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            let platform_fee_bps = context.get("platform_fee_bps")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(500);
                            let tx_hash = context.get("tx_hash")
                                .and_then(|v| v.as_str())
                                .map(|s| s.to_string());
                            let status_str = context.get("status")
                                .and_then(|v| v.as_str())
                                .unwrap_or("Computed");

                            let status = match status_str {
                                "Submitted" => PaymentDistributionStatus::Submitted,
                                "Confirmed" => PaymentDistributionStatus::Confirmed,
                                _ => PaymentDistributionStatus::Computed,
                            };

                            results.push(PaymentDistributionResult {
                                round,
                                model_id,
                                recipient_count,
                                payment_splits,
                                total_basis_points,
                                platform_fee_bps,
                                tx_hash,
                                status,
                                bridge_event_hash: None,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}

/// Input for registering Ethereum address for a node
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterEthAddressInput {
    pub node_id: String,
    pub eth_address: String,
    /// Signature proving ownership of the Ethereum address
    pub signature: String,
}

/// Register Ethereum address for payment distribution
#[hdk_extern]
pub fn register_eth_address(input: RegisterEthAddressInput) -> ExternResult<ActionHash> {
    // F-01: Validate node_id
    validate_node_id(&input.node_id)?;

    // Validate Ethereum address format (0x + 40 hex chars)
    if !input.eth_address.starts_with("0x") || input.eth_address.len() != 42 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid Ethereum address format".to_string()
        )));
    }

    // Verify Ed25519 signature over the Ethereum address bytes proves caller ownership
    let agent = agent_info()?.agent_initial_pubkey;

    // Decode the hex-encoded signature string into raw bytes
    let sig_bytes: Vec<u8> = (0..input.signature.len())
        .step_by(2)
        .filter_map(|i| {
            let end = (i + 2).min(input.signature.len());
            u8::from_str_radix(&input.signature[i..end], 16).ok()
        })
        .collect();

    if sig_bytes.len() != 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Invalid signature length: expected 64 bytes (Ed25519), got {} bytes from hex string",
                sig_bytes.len()
            )
        )));
    }

    let signature = Signature::try_from(sig_bytes).map_err(|_| {
        wasm_error!(WasmErrorInner::Guest(
            "Invalid signature format: could not parse as Ed25519 signature".to_string()
        ))
    })?;

    // Verify signature over the Ethereum address bytes using the agent's public key
    let valid = verify_signature(agent.clone(), signature, input.eth_address.as_bytes().to_vec())?;
    if !valid {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Signature verification failed: Ed25519 signature over Ethereum address does not match agent public key".to_string()
        )));
    }

    // Store as link from node_id path to a serialized address entry
    // Using NodeToReputation link type for eth address storage
    let node_path = Path::from(format!("node_eth_address.{}", input.node_id));
    let node_hash = ensure_path(node_path.clone(), LinkTypes::NodeToReputation)?;

    // Create a link with eth address in tag
    create_link(
        node_hash.clone(),
        node_hash, // Self-link with tag containing address
        LinkTypes::NodeToReputation,
        input.eth_address.as_bytes().to_vec(),
    )?;

    // Return a hash (reusing the path hash since we don't have a dedicated entry)
    Ok(ActionHash::from_raw_39(vec![0u8; 39]))
}
