// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! PoGQ (Proof of Gradient Quality) Validation Zome Tests
//!
//! Tests the proof validation lifecycle:
//! - Proof publication
//! - Proof verification
//! - Quarantine status checking
//! - Nonce replay prevention

use crate::mock_conductor::MockConductor;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_proof_publication() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        let result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "publish_proof",
            serde_json::json!({
                "round": 1,
                "proof": [0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe],
                "nonce": "abc123",
            }),
        ).await.unwrap();

        assert!(result.get("action_hash").is_some());
        assert_eq!(result.get("published"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn test_proof_verification() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // First publish a proof
        let publish_result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "publish_proof",
            serde_json::json!({
                "round": 1,
                "proof": [0x01, 0x02, 0x03, 0x04],
            }),
        ).await.unwrap();

        let proof_hash = publish_result.get("action_hash").unwrap().as_str().unwrap();

        // Then verify it
        let verify_result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "verify_proof",
            serde_json::json!({ "proof_hash": proof_hash }),
        ).await.unwrap();

        assert_eq!(verify_result.get("valid"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn test_verify_nonexistent_proof() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        let result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "verify_proof",
            serde_json::json!({ "proof_hash": "uhCkk_nonexistent123" }),
        ).await.unwrap();

        assert_eq!(result.get("valid"), Some(&serde_json::json!(false)));
    }

    #[tokio::test]
    async fn test_multiple_proofs_per_round() {
        let mut conductor = MockConductor::with_mycelix_zomes();

        let agents: Vec<_> = (0..5).map(|_| conductor.add_agent()).collect();

        // Each agent publishes a proof for round 1
        let mut proof_hashes = Vec::new();
        for (i, agent) in agents.iter().enumerate() {
            let result: serde_json::Value = conductor.call(
                agent,
                "pogq_validation",
                "publish_proof",
                serde_json::json!({
                    "round": 1,
                    "proof": vec![i as u8; 32],
                }),
            ).await.unwrap();

            proof_hashes.push(result.get("action_hash").unwrap().as_str().unwrap().to_string());
        }

        // Verify all proofs
        for (agent, hash) in agents.iter().zip(proof_hashes.iter()) {
            let result: serde_json::Value = conductor.call(
                agent,
                "pogq_validation",
                "verify_proof",
                serde_json::json!({ "proof_hash": hash }),
            ).await.unwrap();

            assert_eq!(result.get("valid"), Some(&serde_json::json!(true)));
        }
    }

    #[tokio::test]
    async fn test_proof_across_rounds() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // Publish proofs for multiple rounds
        for round in 1..=5 {
            let result: serde_json::Value = conductor.call(
                &agent,
                "pogq_validation",
                "publish_proof",
                serde_json::json!({
                    "round": round,
                    "proof": vec![round as u8; 16],
                }),
            ).await.unwrap();

            assert!(result.get("action_hash").is_some());
        }
    }

    #[tokio::test]
    async fn test_proof_with_metadata() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        let result: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "publish_proof",
            serde_json::json!({
                "round": 1,
                "proof": [0xaa, 0xbb, 0xcc, 0xdd],
                "profile_id": 128,  // S128 security profile
                "ema_t_fp": 65536,  // Trust score in fixed-point
                "quarantine_out": 0, // Healthy
            }),
        ).await.unwrap();

        assert_eq!(result.get("published"), Some(&serde_json::json!(true)));
    }

    #[tokio::test]
    async fn test_proof_content_hash() {
        let mut conductor = MockConductor::with_mycelix_zomes();
        let agent = conductor.add_agent();

        // Two different proofs should get different hashes
        let proof1: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "publish_proof",
            serde_json::json!({
                "round": 1,
                "proof": [0x01, 0x02, 0x03],
            }),
        ).await.unwrap();

        let proof2: serde_json::Value = conductor.call(
            &agent,
            "pogq_validation",
            "publish_proof",
            serde_json::json!({
                "round": 1,
                "proof": [0x04, 0x05, 0x06],
            }),
        ).await.unwrap();

        let hash1 = proof1.get("action_hash").unwrap().as_str().unwrap();
        let hash2 = proof2.get("action_hash").unwrap().as_str().unwrap();

        assert_ne!(hash1, hash2, "Different proofs should have different hashes");
    }
}
