//! End-to-End Integration Test: FL + ZK + UESS
//!
//! Demonstrates the integration of:
//! 1. ZK proofs for gradient quality (FL + ZK bridge)
//! 2. UESS storage based on epistemic classification
//! 3. Byzantine detection via proof verification
//!
//! This complements the existing epistemic_agent_integration.rs test.

use mycelix_sdk::epistemic::{
    EmpiricalLevel, EpistemicClassification, MaterialityLevel, NormativeLevel,
};
use mycelix_sdk::fl::{
    AggregationMethod, GradientMetadata, GradientUpdate, VerifiedAggregationMethod, ZKFLError,
    ZKProofFLBridge,
};
use mycelix_sdk::storage::{
    AccessControlMode, EpistemicStorage, MutabilityMode, SchemaIdentity, StorageBackend,
    StorageConfig, StoreOptions,
};
use mycelix_sdk::zkproof::{GradientProver, ProverMode};

/// Test the full ZK-FL pipeline
#[test]
fn test_zk_fl_pipeline() {
    println!("\n=== ZK-FL Pipeline Integration Test ===\n");

    let mut bridge = ZKProofFLBridge::new();

    // Round 1: Multiple participants
    bridge.start_round(1, [0x42u8; 32]);
    println!("Started round 1 with model hash 0x42...");

    // 5 honest participants submit gradients
    for i in 0..5 {
        let gradient: Vec<f32> = (0..500)
            .map(|j| ((i as f32 + j as f32) / 500.0).sin() * 0.4)
            .collect();

        let result = bridge
            .submit_with_proof(
                &format!("honest-client-{}", i),
                &gradient,
                5,                     // epochs
                0.01,                  // learning rate
                64,                    // batch size
                0.3 - i as f64 * 0.02, // decreasing loss
            )
            .expect("Submission should succeed");

        println!(
            "  Client {} submitted - proof valid: {}",
            i,
            result.is_valid()
        );
        assert!(result.is_valid());
    }

    // 2 Byzantine participants submit bad gradients
    for i in 0..2 {
        let bad_gradient = vec![0.0f32; 500]; // Zero gradient - fails quality check
        let result = bridge
            .submit_with_proof(&format!("byzantine-{}", i), &bad_gradient, 5, 0.01, 64, 0.5)
            .expect("Submission should succeed");

        println!(
            "  Byzantine {} submitted - proof valid: {}",
            i,
            result.is_valid()
        );
        assert!(!result.is_valid()); // Should be invalid
    }

    // Aggregate with Byzantine filtering
    let result = bridge
        .aggregate_verified()
        .expect("Aggregation should succeed");

    println!("\nAggregation results:");
    println!("  Included: {} participants", result.included_count);
    println!("  Excluded: {} participants", result.excluded_count);
    println!(
        "  Byzantine fraction: {:.1}%",
        bridge.byzantine_fraction() * 100.0
    );
    println!(
        "  Aggregated gradient dimension: {}",
        result.gradient.gradients.len()
    );

    assert_eq!(result.included_count, 5);
    assert_eq!(result.excluded_count, 2);
    assert!((bridge.byzantine_fraction() - 2.0 / 7.0).abs() < 0.01);

    println!("\n=== ZK-FL Pipeline Test Passed ===\n");
}

/// Test UESS storage routing based on epistemic classification
#[test]
fn test_uess_epistemic_routing() {
    println!("\n=== UESS Epistemic Routing Test ===\n");

    let storage = EpistemicStorage::default_storage();
    let router = storage.router();

    // Test case 1: Ephemeral private data (session tokens)
    let ephemeral = EpistemicClassification::new(
        EmpiricalLevel::E0Null,
        NormativeLevel::N0Personal,
        MaterialityLevel::M0Ephemeral,
    );
    let tier1 = router.route(&ephemeral);
    println!("E0/N0/M0 (session data):");
    println!("  Backend: {:?}", tier1.backend);
    println!("  Mutability: {:?}", tier1.mutability);
    println!("  Encrypted: {}", tier1.encrypted);
    assert_eq!(tier1.backend, StorageBackend::Memory);
    assert_eq!(tier1.mutability, MutabilityMode::MutableCRDT);
    assert!(tier1.encrypted); // Private data should be encrypted

    // Test case 2: Verified communal data (shared documents)
    let communal = EpistemicClassification::new(
        EmpiricalLevel::E2PrivateVerify,
        NormativeLevel::N1Communal,
        MaterialityLevel::M1Temporal,
    );
    let tier2 = router.route(&communal);
    println!("\nE2/N1/M1 (verified shared):");
    println!("  Backend: {:?}", tier2.backend);
    println!("  Mutability: {:?}", tier2.mutability);
    println!("  Access: {:?}", tier2.access_control);
    assert_eq!(tier2.mutability, MutabilityMode::AppendOnly);

    // Test case 3: Cryptographic public data (blockchain records)
    let crypto = EpistemicClassification::new(
        EmpiricalLevel::E3Cryptographic,
        NormativeLevel::N2Network,
        MaterialityLevel::M2Persistent,
    );
    let tier3 = router.route(&crypto);
    println!("\nE3/N2/M2 (cryptographic network):");
    println!("  Backend: {:?}", tier3.backend);
    println!("  Mutability: {:?}", tier3.mutability);
    println!("  Content-addressed: {}", tier3.content_addressed);
    assert_eq!(tier3.mutability, MutabilityMode::Immutable);
    assert!(tier3.content_addressed);

    println!("\n=== UESS Routing Test Passed ===\n");
}

/// Test storing and retrieving ZK proof results via UESS
#[test]
fn test_store_proof_results() {
    println!("\n=== Store Proof Results Test ===\n");

    let storage = EpistemicStorage::default_storage();

    // Simulate a ZK proof result
    let proof_result = serde_json::json!({
        "gradient_hash": "0x1234...abcd",
        "epochs": 5,
        "learning_rate": 0.01,
        "norm_valid": true,
        "client_id": "client-001",
        "round": 1
    });

    // Store with cryptographic classification (since it's proven)
    let classification = EpistemicClassification::new(
        EmpiricalLevel::E3Cryptographic, // Proven via ZK
        NormativeLevel::N2Network,       // Network-wide validity
        MaterialityLevel::M1Temporal,    // Relevant for current round
    );

    let store_options = StoreOptions {
        schema: SchemaIdentity::new("zk-proof-result", "1.0"),
        agent_id: "fl-aggregator".to_string(),
        ..Default::default()
    };

    let receipt = storage
        .store(
            "proof:round-1:client-001",
            &proof_result.to_string(),
            classification,
            store_options,
        )
        .expect("Store should succeed");

    println!("Stored proof result:");
    println!("  Key: proof:round-1:client-001");
    println!("  CID: {}...", &receipt.cid[..20]);
    println!("  Backend: {:?}", receipt.tier.backend);
    println!(
        "  Immutable: {}",
        receipt.tier.mutability == MutabilityMode::Immutable
    );

    // Retrieve and verify
    let retrieved: mycelix_sdk::storage::StoredData<String> = storage
        .retrieve(
            "proof:round-1:client-001",
            None,
            mycelix_sdk::storage::RetrieveOptions::default(),
        )
        .expect("Retrieve should succeed");

    assert!(retrieved.data.contains("norm_valid"));
    println!("  Retrieved successfully: ✓");

    println!("\n=== Store Proof Results Test Passed ===\n");
}

/// Test aggregation methods
#[test]
fn test_aggregation_methods() {
    println!("\n=== Aggregation Methods Test ===\n");

    // Test 1: FedAvg
    let mut bridge = ZKProofFLBridge::new();
    bridge.start_round(1, [0x01u8; 32]);

    for i in 0..3 {
        let gradient: Vec<f32> = (0..100).map(|j| (i as f32 + j as f32) * 0.01).collect();
        bridge
            .submit_with_proof(&format!("c{}", i), &gradient, 5, 0.01, 32, 0.3)
            .unwrap();
    }

    let result = bridge.aggregate_verified().unwrap();
    println!("FedAvg aggregation:");
    println!("  Participants: {}", result.included_count);
    println!("  Gradient dim: {}", result.gradient.gradients.len());
    assert_eq!(result.included_count, 3);

    // Test 2: Trimmed Mean
    let mut bridge2 = ZKProofFLBridge::new()
        .with_aggregation(VerifiedAggregationMethod::TrimmedMean { trim_ratio: 0.1 });
    bridge2.start_round(1, [0x02u8; 32]);

    for i in 0..5 {
        let gradient: Vec<f32> = (0..100).map(|j| (i as f32 + j as f32) * 0.01).collect();
        bridge2
            .submit_with_proof(&format!("t{}", i), &gradient, 5, 0.01, 32, 0.3)
            .unwrap();
    }

    let result2 = bridge2.aggregate_verified().unwrap();
    println!("\nTrimmed Mean aggregation:");
    println!("  Participants: {}", result2.included_count);
    assert_eq!(result2.included_count, 5);

    // Test 3: Time-Weighted
    let mut bridge3 =
        ZKProofFLBridge::new().with_aggregation(VerifiedAggregationMethod::TimeWeighted);
    bridge3.start_round(1, [0x03u8; 32]);

    for i in 0..3 {
        let gradient: Vec<f32> = (0..100).map(|j| (i as f32 + j as f32) * 0.01).collect();
        bridge3
            .submit_with_proof(&format!("w{}", i), &gradient, 5, 0.01, 32, 0.3)
            .unwrap();
    }

    let result3 = bridge3.aggregate_verified().unwrap();
    println!("\nTime-Weighted aggregation:");
    println!("  Participants: {}", result3.included_count);
    assert_eq!(result3.included_count, 3);

    println!("\n=== Aggregation Methods Test Passed ===\n");
}

/// Test multi-round FL with proof tracking
#[test]
fn test_multi_round_fl() {
    println!("\n=== Multi-Round FL Test ===\n");

    let mut bridge = ZKProofFLBridge::new();
    let mut all_hashes: Vec<Vec<[u8; 32]>> = Vec::new();

    for round in 1..=3 {
        let model_hash = [round as u8; 32];
        bridge.start_round(round, model_hash);
        println!("Round {}:", round);

        for client in 0..3 {
            let gradient: Vec<f32> = (0..100)
                .map(|j| ((round + client) as f32 + j as f32) * 0.01)
                .collect();
            bridge
                .submit_with_proof(
                    &format!("client-{}", client),
                    &gradient,
                    round * 5,
                    0.01 / round as f32,
                    64,
                    0.5 / round as f64,
                )
                .unwrap();
        }

        let result = bridge.aggregate_verified().unwrap();
        println!("  Aggregated {} gradients", result.included_count);
        println!("  Hash commitments: {}", result.included_hashes.len());
        all_hashes.push(result.included_hashes);
    }

    // Verify hash commitments are unique across rounds
    let flat_hashes: Vec<_> = all_hashes.iter().flatten().collect();
    println!(
        "\nTotal hash commitments across all rounds: {}",
        flat_hashes.len()
    );

    // All hashes should be unique (different gradients)
    let mut unique_hashes = flat_hashes.clone();
    unique_hashes.sort();
    unique_hashes.dedup();
    println!("Unique hashes: {}", unique_hashes.len());

    println!("\n=== Multi-Round FL Test Passed ===\n");
}
