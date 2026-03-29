// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated Learning Zome Integration Tests
//!
//! Tests the federated_learning coordinator zome functionality:
//! - Gradient submission and storage
//! - Round management
//! - Byzantine-resistant aggregation
//! - Multi-agent coordination

use crate::mock_conductor::{MockConductor, MockAgentPubKey};
use serde::{Deserialize, Serialize};

/// Input for submitting a gradient
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitGradientInput {
    pub round: u64,
    pub gradient: Vec<f32>,
    pub commitment: Option<String>,
}

/// Output from gradient submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitGradientOutput {
    pub action_hash: String,
    pub round: u64,
}

/// Aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    pub round: u64,
    pub aggregated: Vec<f32>,
    pub contributors: u64,
    pub status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_single_gradient_submission() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let alice = conductor.add_agent();

        let result: serde_json::Value = conductor.call(
            &alice,
            "federated_learning",
            "submit_gradient",
            serde_json::json!({
                "round": 1,
                "gradient": [0.1, -0.2, 0.3, -0.4],
            }),
        ).await.unwrap();

        assert!(result.get("action_hash").is_some());
        println!("Gradient submitted: {:?}", result);
    }

    #[tokio::test]
    async fn test_multi_agent_gradient_submission() {
        let mut conductor = MockConductor::with_mycelix_zomes();

        // Create 5 agents
        let agents: Vec<MockAgentPubKey> = (0..5)
            .map(|_| conductor.add_agent())
            .collect();

        // Each agent submits a gradient for round 1
        for (i, agent) in agents.iter().enumerate() {
            let gradient: Vec<f32> = (0..10)
                .map(|j| ((i + j) as f32) * 0.1)
                .collect();

            let result: serde_json::Value = conductor.call(
                agent,
                "federated_learning",
                "submit_gradient",
                serde_json::json!({
                    "round": 1,
                    "gradient": gradient,
                }),
            ).await.unwrap();

            assert!(result.get("action_hash").is_some());
        }

        // Wait for DHT consistency
        conductor.consistency(5).await;

        // Query round gradients
        let gradients: serde_json::Value = conductor.call(
            &agents[0],
            "federated_learning",
            "get_round_gradients",
            serde_json::json!({ "round": 1 }),
        ).await.unwrap();

        let gradient_list = gradients.get("gradients").unwrap().as_array().unwrap();
        assert_eq!(gradient_list.len(), 5, "Should have 5 gradients from 5 agents");
    }

    #[tokio::test]
    async fn test_round_aggregation() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let coordinator = conductor.add_agent();

        // Submit some gradients first
        for i in 0..3 {
            let agent = conductor.add_agent();
            let _: serde_json::Value = conductor.call(
                &agent,
                "federated_learning",
                "submit_gradient",
                serde_json::json!({
                    "round": 1,
                    "gradient": [0.1 * (i as f32), 0.2 * (i as f32), 0.3 * (i as f32)],
                }),
            ).await.unwrap();
        }

        // Aggregate round
        let result: AggregationResult = conductor.call(
            &coordinator,
            "federated_learning",
            "aggregate_round",
            serde_json::json!({ "round": 1 }),
        ).await.unwrap();

        assert_eq!(result.round, 1);
        assert_eq!(result.status, "completed");
        assert!(result.contributors >= 3);
        println!("Aggregation result: {:?}", result);
    }

    #[tokio::test]
    async fn test_multiple_rounds() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // Submit gradients for multiple rounds
        for round in 1..=3 {
            let _: serde_json::Value = conductor.call(
                &agent,
                "federated_learning",
                "submit_gradient",
                serde_json::json!({
                    "round": round,
                    "gradient": vec![round as f32; 10],
                }),
            ).await.unwrap();
        }

        // Query each round's gradients
        for round in 1..=3 {
            let gradients: serde_json::Value = conductor.call(
                &agent,
                "federated_learning",
                "get_round_gradients",
                serde_json::json!({ "round": round }),
            ).await.unwrap();

            let gradient_list = gradients.get("gradients").unwrap().as_array().unwrap();
            assert!(!gradient_list.is_empty(), "Round {} should have gradients", round);
        }
    }

    #[tokio::test]
    async fn test_gradient_with_commitment() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // Submit gradient with commitment hash
        let commitment = "sha256:abc123def456";
        let result: serde_json::Value = conductor.call(
            &agent,
            "federated_learning",
            "submit_gradient",
            serde_json::json!({
                "round": 1,
                "gradient": [0.5, 0.5, 0.5],
                "commitment": commitment,
            }),
        ).await.unwrap();

        assert!(result.get("action_hash").is_some());
    }

    #[tokio::test]
    async fn test_large_gradient() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // Create a large gradient (1000 elements)
        let large_gradient: Vec<f32> = (0..1000)
            .map(|i| (i as f32) / 1000.0)
            .collect();

        let result: serde_json::Value = conductor.call(
            &agent,
            "federated_learning",
            "submit_gradient",
            serde_json::json!({
                "round": 1,
                "gradient": large_gradient,
            }),
        ).await.unwrap();

        assert!(result.get("action_hash").is_some());
        println!("Large gradient (1000 elements) submitted successfully");
    }

    /// Simulates Byzantine behavior for testing
    #[tokio::test]
    async fn test_byzantine_gradient_detection() {
        let mut conductor = MockConductor::with_mycelix_zomes();

        // Honest agents
        let honest: Vec<MockAgentPubKey> = (0..4)
            .map(|_| conductor.add_agent())
            .collect();

        // Byzantine agent
        let byzantine = conductor.add_agent();

        // Honest agents submit similar gradients
        for agent in &honest {
            let _: serde_json::Value = conductor.call(
                agent,
                "federated_learning",
                "submit_gradient",
                serde_json::json!({
                    "round": 1,
                    "gradient": [1.0, 2.0, 3.0, 4.0],
                }),
            ).await.unwrap();
        }

        // Byzantine agent submits divergent gradient
        let _: serde_json::Value = conductor.call(
            &byzantine,
            "federated_learning",
            "submit_gradient",
            serde_json::json!({
                "round": 1,
                "gradient": [-100.0, -200.0, -300.0, -400.0],
            }),
        ).await.unwrap();

        // In a real implementation, the aggregation would detect and exclude
        // the Byzantine gradient. Here we just verify submission works.
        let gradients: serde_json::Value = conductor.call(
            &honest[0],
            "federated_learning",
            "get_round_gradients",
            serde_json::json!({ "round": 1 }),
        ).await.unwrap();

        let gradient_list = gradients.get("gradients").unwrap().as_array().unwrap();
        assert_eq!(gradient_list.len(), 5, "Should have 5 gradients (4 honest + 1 byzantine)");
    }
}
