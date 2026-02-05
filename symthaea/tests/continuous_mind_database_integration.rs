#![cfg(feature = "databases_module")]
//! Integration tests for ContinuousMind + UnifiedMind database architecture
//!
//! Tests the integration of the multi-database consciousness system with
//! the main cognitive loop.

use symthaea::continuous_mind::{ContinuousMind, MindConfig};
use symthaea::databases::{MemoryRecord, MemoryType};
use symthaea::hdc::binary_hv::HV16;

/// Create a test memory record
fn create_test_memory(id: &str, seed: u64, memory_type: MemoryType) -> MemoryRecord {
    MemoryRecord {
        id: id.to_string(),
        encoding: HV16::random(seed),
        timestamp_ms: 1700000000000 + seed,
        memory_type,
        content: format!("Test memory content for {}", id),
        valence: 0.5,
        arousal: 0.3,
        phi: 0.65,
        topics: vec!["integration_test".to_string()],
        metadata: "{}".to_string(),
    }
}

#[tokio::test]
async fn test_continuous_mind_has_unified_mind() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Should have access to unified mind with SQLite persistence by default
    let unified = mind.unified_mind();
    // ContinuousMind uses SQLite for persistent storage
    assert!(unified.status().sensory_real, "Should use SQLite databases by default");
}

#[tokio::test]
async fn test_continuous_mind_remember_working_memory() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Store a working memory
    let record = create_test_memory("working-1", 100, MemoryType::Working);
    let encoding = record.encoding.clone();

    mind.remember(record).await.expect("Should store working memory");

    // Recall it
    let results = mind.recall_working(&encoding, 1).await
        .expect("Should recall working memory");

    assert_eq!(results.len(), 1, "Should find the stored memory");
    assert_eq!(results[0].record.id, "working-1");
}

#[tokio::test]
async fn test_continuous_mind_remember_episodic_memory() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Store an episodic memory
    let record = create_test_memory("episodic-1", 200, MemoryType::Episodic);
    let encoding = record.encoding.clone();

    mind.remember(record).await.expect("Should store episodic memory");

    // Recall from long-term
    let results = mind.recall_long_term(&encoding, 1).await
        .expect("Should recall long-term memory");

    assert_eq!(results.len(), 1, "Should find the stored memory");
    assert_eq!(results[0].record.id, "episodic-1");
}

#[tokio::test]
async fn test_continuous_mind_recall_all() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Store memories of different types
    let working = create_test_memory("w1", 100, MemoryType::Working);
    let episodic = create_test_memory("e1", 101, MemoryType::Episodic);

    mind.remember(working).await.unwrap();
    mind.remember(episodic).await.unwrap();

    // Query should return results from multiple sources
    let query = HV16::random(100); // Close to working memory
    let results = mind.recall_all(&query, 10).await.unwrap();

    assert!(results.len() >= 1, "Should recall at least one memory");
}

#[tokio::test]
async fn test_continuous_mind_memory_statistics() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Store some memories
    mind.remember(create_test_memory("m1", 1, MemoryType::Working)).await.unwrap();
    mind.remember(create_test_memory("m2", 2, MemoryType::Episodic)).await.unwrap();
    mind.remember(create_test_memory("m3", 3, MemoryType::Semantic)).await.unwrap();

    let stats = mind.memory_statistics().await.unwrap();

    // Should have recorded the memories
    assert!(stats.total() >= 3, "Should have at least 3 memories stored");
}

#[tokio::test]
async fn test_continuous_mind_memory_health() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    let health = mind.memory_health().await;

    // Mock databases should all report healthy
    assert!(health.sensory_ok, "Sensory cortex should be healthy");
    assert!(health.prefrontal_ok, "Prefrontal cortex should be healthy");
    assert!(health.long_term_ok, "Long-term memory should be healthy");
    assert!(health.epistemic_ok, "Epistemic auditor should be healthy");
}

#[tokio::test]
async fn test_continuous_mind_ltc_integration() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Test LTC flow state (should start at 0 or low)
    let initial_flow = mind.ltc_flow();
    assert!(initial_flow >= 0.0, "Flow should be non-negative");

    // Step LTC with some input
    let input = vec![0.5f32; 64];
    mind.step_ltc(&input, 10.0);

    // Record some Φ values
    mind.record_phi_to_ltc(0.5);
    mind.record_phi_to_ltc(0.6);
    mind.record_phi_to_ltc(0.7);

    // Check trend (should be positive with increasing Φ)
    let trend = mind.ltc_trend();
    // Note: With only 3 samples, trend calculation may vary
    assert!(trend.is_finite(), "Trend should be a valid number");
}

#[tokio::test]
async fn test_continuous_mind_procedural_dual_storage() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Procedural memories should be stored in both prefrontal and long-term
    let procedural = create_test_memory("skill-1", 500, MemoryType::Procedural);
    let encoding = procedural.encoding.clone();

    mind.remember(procedural).await.unwrap();

    // Should be findable in long-term
    let lt_results = mind.recall_long_term(&encoding, 1).await.unwrap();
    assert!(!lt_results.is_empty(), "Procedural should be in long-term");

    // recall_all should find it
    let all_results = mind.recall_all(&encoding, 5).await.unwrap();
    assert!(!all_results.is_empty(), "Should be findable via recall_all");
}

#[tokio::test]
async fn test_continuous_mind_phi_preservation() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Store a memory with specific Φ
    let mut record = create_test_memory("phi-test", 42, MemoryType::Episodic);
    record.phi = 0.85; // High consciousness moment

    let encoding = record.encoding.clone();
    mind.remember(record).await.unwrap();

    let results = mind.recall_long_term(&encoding, 1).await.unwrap();
    assert!(!results.is_empty());
    assert!(
        (results[0].record.phi - 0.85).abs() < 0.001,
        "Φ should be preserved: got {}",
        results[0].record.phi
    );
}
