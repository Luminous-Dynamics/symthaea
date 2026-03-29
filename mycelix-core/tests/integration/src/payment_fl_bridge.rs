// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PaymentRouter ↔ FL Aggregator Integration Tests
//!
//! This module tests the end-to-end flow from federated learning contributions
//! to fair payment distribution using Shapley values.
//!
//! ## Flow
//!
//! ```text
//! FL Gradient Contribution → Shapley Value Attribution → Payment Split (basis points)
//! ```
//!
//! ## Shapley → PaymentRouter Mapping
//!
//! The Shapley value (normalized 0-1) maps to PaymentRouter's basis points (0-10000):
//!
//! ```text
//! shareBps = (shapley_value / total_shapley) × 10000
//! ```
//!
//! This ensures:
//! - Fair reward distribution proportional to contribution quality
//! - Byzantine contributors with negative Shapley get 0 (excluded from rewards)
//! - Sum of all shares equals 10000 (100%)

use fl_aggregator::{
    Aggregator, AggregatorConfig, Defense, Gradient, NodeId,
    ShapleyCalculator, ShapleyConfig, ShapleyResult, SamplingMethod,
    Baseline, ValueFunction,
};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Basis points denominator (matches PaymentRouter.sol)
pub const BASIS_POINTS: u64 = 10_000;

/// Maximum platform fee (matches PaymentRouter.sol MAX_PLATFORM_FEE = 500 = 5%)
pub const MAX_PLATFORM_FEE_BPS: u64 = 500;

/// Minimum payment share to be included (dust threshold)
pub const MIN_SHARE_BPS: u64 = 10; // 0.1%

/// Payment split for a single recipient (mirrors Solidity struct)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PaymentSplit {
    /// Recipient address (Ethereum address as hex string)
    pub recipient: String,
    /// Share in basis points (0-10000)
    pub share_bps: u64,
}

/// Result of converting FL contributions to payment splits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FLPaymentDistribution {
    /// List of payment splits
    pub splits: Vec<PaymentSplit>,
    /// Total basis points (should equal 10000 after normalization)
    pub total_bps: u64,
    /// Model ID this distribution is for
    pub model_id: String,
    /// Round number
    pub round: u64,
    /// Nodes excluded due to negative Shapley values
    pub excluded_nodes: Vec<NodeId>,
    /// Shapley computation metadata
    pub shapley_result: ShapleyResult,
}

impl FLPaymentDistribution {
    /// Check if the distribution is valid for PaymentRouter
    pub fn is_valid(&self) -> bool {
        // Empty distribution is valid (no contributors)
        if self.splits.is_empty() {
            return true;
        }

        // All shares must be positive
        if self.splits.iter().any(|s| s.share_bps == 0) {
            return false;
        }

        // Recipients must be unique
        let mut seen = std::collections::HashSet::new();
        for split in &self.splits {
            if !seen.insert(&split.recipient) {
                return false;
            }
        }

        // Contributor shares should sum to total_bps
        let contributor_total: u64 = self.splits.iter().map(|s| s.share_bps).sum();
        // With platform fee, total_bps = contributor_total + fee
        // So contributor_total should equal total_bps - fee, or be less (if fee not added to total_bps)
        contributor_total <= self.total_bps && contributor_total > 0
    }

    /// Check if distribution is ready for PaymentRouter (shares sum to exactly 10000)
    pub fn is_ready_for_router(&self) -> bool {
        if self.splits.is_empty() {
            return false;
        }

        let total: u64 = self.splits.iter().map(|s| s.share_bps).sum();
        total == BASIS_POINTS
    }
}

/// Bridge between FL Aggregator and PaymentRouter
pub struct FLPaymentBridge {
    /// Shapley calculator for contribution attribution
    shapley_calculator: ShapleyCalculator,
    /// Mapping from node IDs to Ethereum addresses
    pub node_to_address: HashMap<NodeId, String>,
    /// Platform fee in basis points (deducted before distribution)
    platform_fee_bps: u64,
}

impl FLPaymentBridge {
    /// Create a new FL-Payment bridge
    pub fn new(platform_fee_bps: u64) -> Self {
        assert!(platform_fee_bps <= MAX_PLATFORM_FEE_BPS,
            "Platform fee exceeds maximum: {} > {}", platform_fee_bps, MAX_PLATFORM_FEE_BPS);

        // Use CosineSimilarity - measures alignment with target gradient
        // Byzantine gradients pointing in opposite direction get low/negative values
        let config = ShapleyConfig::default()
            .with_value_function(ValueFunction::CosineSimilarity)
            .with_normalization();

        Self {
            shapley_calculator: ShapleyCalculator::new(config),
            node_to_address: HashMap::new(),
            platform_fee_bps,
        }
    }

    /// Create with exact Shapley computation (for small number of contributors)
    pub fn new_exact(platform_fee_bps: u64) -> Self {
        assert!(platform_fee_bps <= MAX_PLATFORM_FEE_BPS);

        // Use CosineSimilarity - measures alignment with target gradient
        // Byzantine gradients pointing in opposite direction get low/negative values
        let config = ShapleyConfig::exact()
            .with_value_function(ValueFunction::CosineSimilarity)
            .with_normalization()
            .with_seed(42);

        Self {
            shapley_calculator: ShapleyCalculator::new(config),
            node_to_address: HashMap::new(),
            platform_fee_bps,
        }
    }

    /// Register a node's Ethereum address
    pub fn register_node(&mut self, node_id: NodeId, eth_address: String) {
        // Normalize address to lowercase
        let normalized = if eth_address.starts_with("0x") {
            eth_address.to_lowercase()
        } else {
            format!("0x{}", eth_address.to_lowercase())
        };
        self.node_to_address.insert(node_id, normalized);
    }

    /// Compute payment distribution from FL round contributions
    pub fn compute_distribution(
        &mut self,
        gradients: &HashMap<NodeId, Gradient>,
        aggregated: &Gradient,
        model_id: &str,
        round: u64,
    ) -> FLPaymentDistribution {
        // Compute Shapley values
        let shapley_result = self.shapley_calculator.calculate_values(gradients, aggregated);

        // Separate positive and negative contributors
        let mut positive_contributors: Vec<(&String, f32)> = Vec::new();
        let mut excluded_nodes: Vec<NodeId> = Vec::new();

        for (node_id, value) in &shapley_result.values {
            if *value > 0.0 {
                positive_contributors.push((node_id, *value));
            } else {
                excluded_nodes.push(node_id.clone());
            }
        }

        // If no positive contributors, return empty distribution
        if positive_contributors.is_empty() {
            return FLPaymentDistribution {
                splits: Vec::new(),
                total_bps: 0,
                model_id: model_id.to_string(),
                round,
                excluded_nodes,
                shapley_result,
            };
        }

        // Calculate distributable basis points (after platform fee)
        let distributable_bps = BASIS_POINTS - self.platform_fee_bps;

        // Normalize positive Shapley values to basis points
        let total_positive: f32 = positive_contributors.iter().map(|(_, v)| v).sum();

        let mut splits: Vec<PaymentSplit> = Vec::new();
        let mut allocated_bps: u64 = 0;

        for (node_id, value) in &positive_contributors {
            // Calculate share as proportion of total positive contributions
            let share_ratio = *value / total_positive;
            let share_bps = ((share_ratio as f64) * (distributable_bps as f64)).round() as u64;

            // Skip dust amounts
            if share_bps < MIN_SHARE_BPS {
                excluded_nodes.push((*node_id).clone());
                continue;
            }

            // Get Ethereum address (use node ID as fallback)
            let eth_address = self.node_to_address
                .get(*node_id)
                .cloned()
                .unwrap_or_else(|| format!("0x{:0>40}", node_id.replace("-", "")));

            splits.push(PaymentSplit {
                recipient: eth_address,
                share_bps,
            });
            allocated_bps += share_bps;
        }

        // Handle rounding remainder - give to largest contributor
        if allocated_bps < distributable_bps && !splits.is_empty() {
            let remainder = distributable_bps - allocated_bps;
            // Find the split with the largest share
            if let Some(largest) = splits.iter_mut().max_by_key(|s| s.share_bps) {
                largest.share_bps += remainder;
                allocated_bps += remainder;
            }
        }

        // Add platform fee recipient if fee > 0
        // In production, this would be the platform fee address
        let total_bps = if self.platform_fee_bps > 0 {
            // For testing, we just verify the math works out
            allocated_bps + self.platform_fee_bps
        } else {
            allocated_bps
        };

        FLPaymentDistribution {
            splits,
            total_bps,
            model_id: model_id.to_string(),
            round,
            excluded_nodes,
            shapley_result,
        }
    }

    /// Convert distribution to PaymentRouter.routePayment() calldata format
    pub fn to_calldata(&self, distribution: &FLPaymentDistribution) -> PaymentRouterCalldata {
        PaymentRouterCalldata {
            payment_id: format!("{}:{}", distribution.model_id, distribution.round),
            recipients: distribution.splits.clone(),
            token: "0x0000000000000000000000000000000000000000".to_string(), // Native ETH
        }
    }
}

/// Calldata format for PaymentRouter.routePayment()
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentRouterCalldata {
    /// Unique payment identifier
    pub payment_id: String,
    /// Recipients with their shares
    pub recipients: Vec<PaymentSplit>,
    /// Token address (address(0) for native)
    pub token: String,
}

impl PaymentRouterCalldata {
    /// Validate the calldata is ready for on-chain submission
    pub fn validate(&self) -> Result<(), String> {
        if self.recipients.is_empty() {
            return Err("No recipients".to_string());
        }

        if self.recipients.len() > 100 {
            return Err("Too many recipients (max 100)".to_string());
        }

        let total: u64 = self.recipients.iter().map(|r| r.share_bps).sum();
        if total != BASIS_POINTS {
            return Err(format!("Shares must sum to {} (got {})", BASIS_POINTS, total));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn create_honest_gradients() -> HashMap<NodeId, Gradient> {
        let mut gradients = HashMap::new();
        // Honest nodes cluster around similar gradients
        gradients.insert("node1".to_string(), array![1.0, 2.0, 3.0, 4.0]);
        gradients.insert("node2".to_string(), array![1.1, 2.1, 3.1, 4.1]);
        gradients.insert("node3".to_string(), array![0.9, 1.9, 2.9, 3.9]);
        gradients.insert("node4".to_string(), array![1.05, 2.05, 3.05, 4.05]);
        gradients
    }

    fn create_mixed_gradients() -> HashMap<NodeId, Gradient> {
        let mut gradients = create_honest_gradients();
        // Add a Byzantine node with divergent gradient
        gradients.insert("byzantine".to_string(), array![-10.0, -20.0, -30.0, -40.0]);
        gradients
    }

    #[test]
    fn test_basic_payment_distribution() {
        let mut bridge = FLPaymentBridge::new_exact(0); // No platform fee

        // Register Ethereum addresses
        bridge.register_node("node1".to_string(), "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string());
        bridge.register_node("node2".to_string(), "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_string());
        bridge.register_node("node3".to_string(), "0xcccccccccccccccccccccccccccccccccccccccc".to_string());
        bridge.register_node("node4".to_string(), "0xdddddddddddddddddddddddddddddddddddddddd".to_string());

        let gradients = create_honest_gradients();
        let aggregated = array![1.0, 2.0, 3.0, 4.0]; // Target gradient

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregated,
            "model_001",
            1,
        );

        // All honest nodes should receive positive shares
        assert_eq!(distribution.splits.len(), 4, "All 4 honest nodes should get shares");
        assert!(distribution.excluded_nodes.is_empty(), "No nodes should be excluded");
        assert_eq!(distribution.total_bps, BASIS_POINTS, "Total should equal 10000");
        assert!(distribution.is_valid(), "Distribution should be valid for PaymentRouter");

        // Verify shares sum correctly
        let total_shares: u64 = distribution.splits.iter().map(|s| s.share_bps).sum();
        assert_eq!(total_shares, BASIS_POINTS, "Shares should sum to 10000");

        println!("Distribution for round {}: {:?}", distribution.round, distribution.splits);
    }

    #[test]
    fn test_byzantine_exclusion() {
        let mut bridge = FLPaymentBridge::new_exact(0);

        let gradients = create_mixed_gradients();
        let aggregated = array![1.0, 2.0, 3.0, 4.0];

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregated,
            "model_001",
            1,
        );

        // Get Byzantine Shapley value
        let byzantine_shapley = distribution.shapley_result.values.get("byzantine").unwrap();

        // Byzantine node should have lower Shapley value than honest nodes
        let honest_min: f32 = ["node1", "node2", "node3", "node4"]
            .iter()
            .filter_map(|n| distribution.shapley_result.values.get(*n))
            .copied()
            .fold(f32::INFINITY, f32::min);

        assert!(
            *byzantine_shapley < honest_min,
            "Byzantine Shapley ({}) should be less than honest min ({})",
            byzantine_shapley,
            honest_min
        );

        // If Byzantine is excluded, verify only honest nodes get shares
        if distribution.excluded_nodes.contains(&"byzantine".to_string()) {
            assert_eq!(distribution.splits.len(), 4, "Only 4 honest nodes should get shares");
        } else {
            // If not excluded, Byzantine should at least have smallest share
            let byzantine_share = distribution.splits.iter()
                .find(|s| s.recipient.contains("byzantine"))
                .map(|s| s.share_bps)
                .unwrap_or(0);
            let min_honest_share = distribution.splits.iter()
                .filter(|s| !s.recipient.contains("byzantine"))
                .map(|s| s.share_bps)
                .min()
                .unwrap_or(BASIS_POINTS);
            assert!(
                byzantine_share <= min_honest_share,
                "Byzantine share ({}) should be <= honest min ({})",
                byzantine_share,
                min_honest_share
            );
        }

        println!("Excluded nodes: {:?}", distribution.excluded_nodes);
        println!("Byzantine Shapley value: {}", byzantine_shapley);
        println!("Honest min Shapley value: {}", honest_min);
    }

    #[test]
    fn test_platform_fee_deduction() {
        let platform_fee_bps = 250; // 2.5%
        let mut bridge = FLPaymentBridge::new_exact(platform_fee_bps);

        let gradients = create_honest_gradients();
        let aggregated = array![1.0, 2.0, 3.0, 4.0];

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregated,
            "model_001",
            1,
        );

        // Total should include platform fee
        assert_eq!(distribution.total_bps, BASIS_POINTS, "Total should be 10000");

        // Contributors' shares should sum to 10000 - platform_fee
        let contributors_total: u64 = distribution.splits.iter().map(|s| s.share_bps).sum();
        assert_eq!(
            contributors_total,
            BASIS_POINTS - platform_fee_bps,
            "Contributors should receive 10000 - platform_fee"
        );

        println!("Platform fee: {} bps", platform_fee_bps);
        println!("Contributors receive: {} bps", contributors_total);
    }

    #[test]
    fn test_calldata_generation() {
        let mut bridge = FLPaymentBridge::new_exact(0);

        bridge.register_node("node1".to_string(), "0x1111111111111111111111111111111111111111".to_string());
        bridge.register_node("node2".to_string(), "0x2222222222222222222222222222222222222222".to_string());

        let mut gradients = HashMap::new();
        gradients.insert("node1".to_string(), array![1.0, 2.0]);
        gradients.insert("node2".to_string(), array![1.0, 2.0]); // Identical = equal shares

        let aggregated = array![1.0, 2.0];

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregated,
            "model_002",
            5,
        );

        let calldata = bridge.to_calldata(&distribution);

        // Validate calldata
        assert!(calldata.validate().is_ok(), "Calldata should be valid");
        assert_eq!(calldata.payment_id, "model_002:5");
        assert_eq!(calldata.token, "0x0000000000000000000000000000000000000000");

        // Symmetric gradients should get equal shares
        let shares: Vec<u64> = calldata.recipients.iter().map(|r| r.share_bps).collect();
        assert_eq!(shares[0], shares[1], "Symmetric contributors should get equal shares");

        println!("Calldata: {:?}", calldata);
    }

    #[test]
    fn test_aggregator_to_payment_full_flow() {
        // Full E2E: FL Aggregation → Shapley → PaymentRouter

        // 1. Create FL Aggregator
        let config = AggregatorConfig::default()
            .with_defense(Defense::Krum { f: 1 })
            .with_expected_nodes(5);
        let mut aggregator = Aggregator::new(config);

        // 2. Submit gradients (simulating FL round)
        let gradients = create_mixed_gradients();
        for (node_id, gradient) in &gradients {
            aggregator.submit(node_id, gradient.clone()).unwrap();
        }

        // 3. Finalize aggregation
        let aggregation_result = aggregator.finalize_round().unwrap();

        // 4. Create payment bridge and compute distribution
        let mut bridge = FLPaymentBridge::new_exact(250); // 2.5% fee

        // Register addresses for all nodes
        for (i, node_id) in gradients.keys().enumerate() {
            let addr = format!("0x{:0>40x}", i + 1);
            bridge.register_node(node_id.clone(), addr);
        }

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregation_result,
            "fl_model_v1",
            1,
        );

        // 5. Validate the flow
        assert!(distribution.is_valid(),
            "Distribution should be valid. Splits: {:?}, Total: {}",
            distribution.splits, distribution.total_bps);

        // Get Byzantine Shapley value
        let byzantine_shapley = *distribution.shapley_result.values.get("byzantine").unwrap();

        // Byzantine node should have lower Shapley value than honest nodes
        let honest_values: Vec<f32> = ["node1", "node2", "node3", "node4"]
            .iter()
            .filter_map(|n| distribution.shapley_result.values.get(*n))
            .copied()
            .collect();

        let honest_min = honest_values.iter().copied().fold(f32::INFINITY, f32::min);

        assert!(
            byzantine_shapley < honest_min,
            "Byzantine Shapley ({}) should be less than honest min ({})",
            byzantine_shapley,
            honest_min
        );

        // All honest nodes should have positive contribution
        assert!(
            honest_values.iter().all(|v| *v > 0.0),
            "All honest nodes should have positive Shapley values"
        );

        // Byzantine should get less payment than any honest node
        if !distribution.excluded_nodes.contains(&"byzantine".to_string()) {
            // Find Byzantine's payment split
            let byz_addr = bridge.node_to_address.get("byzantine").cloned()
                .unwrap_or_else(|| "byzantine".to_string());
            let byzantine_share = distribution.splits.iter()
                .find(|s| s.recipient.to_lowercase().contains(&byz_addr.to_lowercase())
                    || s.recipient.to_lowercase().contains("byzantine"))
                .map(|s| s.share_bps)
                .unwrap_or(0);
            let min_honest_share = distribution.splits.iter()
                .filter(|s| !s.recipient.to_lowercase().contains(&byz_addr.to_lowercase())
                    && !s.recipient.to_lowercase().contains("byzantine"))
                .map(|s| s.share_bps)
                .min()
                .unwrap_or(BASIS_POINTS);

            assert!(
                byzantine_share <= min_honest_share,
                "Byzantine share ({}) should be <= honest min ({})",
                byzantine_share,
                min_honest_share
            );
        }

        println!("\n=== Full FL → Payment Flow ===");
        println!("Aggregation result norm: {:.4}",
            aggregation_result.iter().map(|x| x * x).sum::<f32>().sqrt());
        println!("Payment splits: {:?}", distribution.splits);
        println!("Excluded (Byzantine): {:?}", distribution.excluded_nodes);
        println!("Byzantine Shapley: {}", byzantine_shapley);
        println!("Honest min Shapley: {}", honest_min);
        println!("Total distributed: {} bps", distribution.total_bps);
    }

    #[test]
    fn test_multi_round_payments() {
        let mut bridge = FLPaymentBridge::new_exact(100); // 1% fee

        // Register nodes
        for i in 1..=4 {
            bridge.register_node(
                format!("node{}", i),
                format!("0x{:0>40}", i),
            );
        }

        let mut total_paid_per_node: HashMap<String, u64> = HashMap::new();

        // Simulate 5 rounds
        for round in 1..=5 {
            let mut gradients = HashMap::new();

            // Node performance varies by round
            for i in 1..=4 {
                let quality = if i == round % 4 + 1 { 1.0 } else { 0.8 + (i as f32) * 0.05 };
                gradients.insert(
                    format!("node{}", i),
                    array![1.0 * quality, 2.0 * quality, 3.0 * quality],
                );
            }

            let aggregated = array![1.0, 2.0, 3.0];

            let distribution = bridge.compute_distribution(
                &gradients,
                &aggregated,
                "model_continuous",
                round,
            );

            // Accumulate payments
            for split in &distribution.splits {
                *total_paid_per_node.entry(split.recipient.clone()).or_insert(0) += split.share_bps;
            }
        }

        // All nodes should have received some payment
        assert_eq!(total_paid_per_node.len(), 4, "All 4 nodes should have been paid");

        // Total payments should equal 5 rounds × (10000 - 100 fee) = 49500
        let total_all_nodes: u64 = total_paid_per_node.values().sum();
        assert_eq!(total_all_nodes, 5 * (BASIS_POINTS - 100), "Total payments should match expected");

        println!("\n=== Multi-Round Payment Summary ===");
        for (addr, total) in &total_paid_per_node {
            println!("{}: {} bps total ({:.2}%)", addr, total, (*total as f64) / 500.0);
        }
    }

    #[test]
    fn test_edge_case_single_contributor() {
        let mut bridge = FLPaymentBridge::new_exact(0);

        let mut gradients = HashMap::new();
        gradients.insert("only_node".to_string(), array![1.0, 2.0, 3.0]);

        let aggregated = array![1.0, 2.0, 3.0];

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregated,
            "single_model",
            1,
        );

        // Single contributor gets everything
        assert_eq!(distribution.splits.len(), 1);
        assert_eq!(distribution.splits[0].share_bps, BASIS_POINTS);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut bridge = FLPaymentBridge::new_exact(0);

        let gradients = create_honest_gradients();
        let aggregated = array![1.0, 2.0, 3.0, 4.0];

        let distribution = bridge.compute_distribution(
            &gradients,
            &aggregated,
            "serialize_test",
            42,
        );

        // Serialize to JSON
        let json = serde_json::to_string(&distribution).unwrap();

        // Deserialize
        let restored: FLPaymentDistribution = serde_json::from_str(&json).unwrap();

        assert_eq!(distribution.model_id, restored.model_id);
        assert_eq!(distribution.round, restored.round);
        assert_eq!(distribution.total_bps, restored.total_bps);
        assert_eq!(distribution.splits.len(), restored.splits.len());

        println!("JSON size: {} bytes", json.len());
    }
}
