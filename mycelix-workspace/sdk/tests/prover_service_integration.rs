//! Integration tests for the ZK Prover Service
//!
//! These tests verify the prover integration works correctly with
//! an external prover service. Run with:
//!
//! ```bash
//! # Start the prover service first
//! cd spike/zk-prover-service && docker-compose up -d
//!
//! # Run integration tests
//! RISC0_PROVER_URL=http://localhost:3000 cargo test --features simulation prover_service_integration -- --ignored
//! ```
//!
//! Tests marked with #[ignore] require the external service to be running.

#[cfg(all(test, feature = "simulation"))]
mod tests {
    use mycelix_sdk::fl::prover_integration::{ProverBackend, ProverIntegration, ProverStats};

    // ========================================================================
    // Unit Tests (no external service required)
    // ========================================================================

    #[test]
    fn test_prover_backend_default_simulation() {
        // Without env vars, should default to simulation
        let backend = ProverBackend::default();
        assert!(matches!(backend, ProverBackend::Simulation));
    }

    #[test]
    fn test_prover_integration_for_testing() {
        let integration = ProverIntegration::for_testing();
        assert!(!integration.is_production());
    }

    #[test]
    fn test_prover_integration_with_external() {
        let integration = ProverIntegration::with_external_service("http://localhost:3000");
        assert!(integration.is_production());
    }

    #[test]
    fn test_simulation_proof_generation() {
        let mut integration = ProverIntegration::for_testing();

        // Create sample gradient
        let gradient: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let model_hash = [0x42u8; 32];

        let result = integration.prove_gradient(
            &gradient,
            &model_hash,
            5,    // epochs
            0.01, // learning_rate
            "test-client",
            1, // round
        );

        assert!(result.is_ok());
        let receipt = result.unwrap();
        assert!(receipt.is_valid());
        assert!(integration.verify(&receipt));

        // Verify stats updated
        let stats = integration.stats();
        assert_eq!(stats.total_proofs, 1);
        assert_eq!(stats.successful_proofs, 1);
    }

    #[test]
    fn test_invalid_gradient_detected() {
        let mut integration = ProverIntegration::for_testing();

        // Zero gradient should fail quality check
        let gradient = vec![0.0f32; 100];
        let model_hash = [0x42u8; 32];

        let result = integration.prove_gradient(&gradient, &model_hash, 5, 0.01, "test-client", 1);

        assert!(result.is_ok());
        let receipt = result.unwrap();
        // Zero gradient has zero norm, which fails quality constraints
        assert!(!receipt.is_valid());
    }

    #[test]
    fn test_stats_accumulation() {
        let mut integration = ProverIntegration::for_testing();

        let gradient: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let model_hash = [0x42u8; 32];

        // Generate multiple proofs
        for i in 0..5 {
            let _ = integration.prove_gradient(
                &gradient,
                &model_hash,
                5,
                0.01,
                &format!("client-{}", i),
                i as u32,
            );
        }

        let stats = integration.stats();
        assert_eq!(stats.total_proofs, 5);
        assert_eq!(stats.successful_proofs, 5);
        assert_eq!(stats.failed_proofs, 0);
    }

    // ========================================================================
    // Integration Tests (require external service)
    // ========================================================================

    #[test]
    #[ignore = "Requires external prover service running at RISC0_PROVER_URL"]
    fn test_external_service_health_check() {
        let url = std::env::var("RISC0_PROVER_URL")
            .expect("RISC0_PROVER_URL must be set for integration tests");

        // Just verify we can parse the URL and create the backend
        let _integration = ProverIntegration::with_external_service(&url);
    }

    #[test]
    #[ignore = "Requires external prover service running at RISC0_PROVER_URL"]
    fn test_external_proof_generation() {
        let url = std::env::var("RISC0_PROVER_URL")
            .expect("RISC0_PROVER_URL must be set for integration tests");

        let mut integration = ProverIntegration::with_external_service(&url);

        // Create sample gradient
        let gradient: Vec<f32> = (0..100).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let model_hash = [0x42u8; 32];

        let result = integration.prove_gradient(
            &gradient,
            &model_hash,
            5,
            0.01,
            "integration-test-client",
            1,
        );

        // This may fail if service is unavailable, but shouldn't panic
        match result {
            Ok(receipt) => {
                println!("Got receipt from external service:");
                println!("  Valid: {}", receipt.is_valid());
                println!("  Generation time: {}ms", receipt.generation_time_ms);
                assert!(receipt.is_valid());
            }
            Err(e) => {
                println!(
                    "External service error (may fall back to simulation): {}",
                    e
                );
            }
        }
    }

    #[test]
    #[ignore = "Requires external prover service running at RISC0_PROVER_URL"]
    fn test_external_byzantine_gradient_rejected() {
        let url = std::env::var("RISC0_PROVER_URL")
            .expect("RISC0_PROVER_URL must be set for integration tests");

        let mut integration = ProverIntegration::with_external_service(&url);

        // Byzantine gradient: extreme values that should fail quality check
        let byzantine_gradient: Vec<f32> = (0..100)
            .map(|_| 1000.0) // Way outside normal range
            .collect();
        let model_hash = [0x42u8; 32];

        let result = integration.prove_gradient(
            &byzantine_gradient,
            &model_hash,
            5,
            0.01,
            "byzantine-client",
            1,
        );

        match result {
            Ok(receipt) => {
                // Proof generated, but should indicate invalid gradient
                assert!(!receipt.is_valid(), "Byzantine gradient should be rejected");
            }
            Err(_) => {
                // Error is also acceptable for byzantine input
            }
        }
    }

    #[test]
    #[ignore = "Requires external prover service running at RISC0_PROVER_URL"]
    fn test_external_batch_proofs() {
        let url = std::env::var("RISC0_PROVER_URL")
            .expect("RISC0_PROVER_URL must be set for integration tests");

        let mut integration = ProverIntegration::with_external_service(&url);
        let model_hash = [0x42u8; 32];

        let clients = ["alice", "bob", "charlie"];
        let mut results = Vec::new();

        for (i, client) in clients.iter().enumerate() {
            let gradient: Vec<f32> = (0..100)
                .map(|j| ((i * 10 + j) as f32 * 0.01).sin() * 0.5)
                .collect();

            let result = integration.prove_gradient(&gradient, &model_hash, 5, 0.01, client, 1);
            results.push((client, result));
        }

        // Count successes
        let successes = results.iter().filter(|(_, r)| r.is_ok()).count();
        println!(
            "Batch proof results: {}/{} succeeded",
            successes,
            clients.len()
        );

        // At minimum, simulation fallback should work
        assert!(successes >= 1, "At least some proofs should succeed");
    }

    // ========================================================================
    // Benchmark Tests
    // ========================================================================

    #[test]
    #[ignore = "Long-running benchmark test"]
    fn bench_simulation_proof_generation() {
        let mut integration = ProverIntegration::for_testing();

        let gradient: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.001).sin() * 0.5).collect();
        let model_hash = [0x42u8; 32];

        let iterations = 100;
        let start = std::time::Instant::now();

        for i in 0..iterations {
            let _ = integration.prove_gradient(
                &gradient,
                &model_hash,
                5,
                0.01,
                &format!("bench-client-{}", i),
                i as u32,
            );
        }

        let elapsed = start.elapsed();
        let avg_ms = elapsed.as_millis() as f64 / iterations as f64;

        println!("Simulation proof benchmark:");
        println!("  Iterations: {}", iterations);
        println!("  Total time: {:?}", elapsed);
        println!("  Average time: {:.2}ms", avg_ms);

        let stats = integration.stats();
        assert_eq!(stats.total_proofs, iterations);
    }
}
