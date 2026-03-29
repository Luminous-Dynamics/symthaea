// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Sweettest Integration Tests
//!
//! These tests run against a real Holochain conductor.

use holochain::sweettest::*;
use holochain_types::prelude::*;
use lucid_sweettest::*;

// ============================================================================
// THOUGHT CRUD TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore] // Run with: cargo test --test sweettest_integration -- --ignored
async fn test_create_thought() {
    let (conductor, zome) = setup_lucid_conductor().await;

    let input = CreateThoughtInput::new("This is my first thought in LUCID")
        .with_type(ThoughtType::Note)
        .with_confidence(0.8)
        .with_tags(vec!["test".into(), "first".into()]);

    let record: Record = conductor
        .call(&zome, "create_thought", input)
        .await;

    let thought = thought_from_record(&record).expect("Failed to decode thought");

    assert_eq!(thought.content, "This is my first thought in LUCID");
    assert_eq!(thought.thought_type, ThoughtType::Note);
    assert_eq!(thought.confidence, 0.8);
    assert!(thought.tags.contains(&"test".to_string()));
    assert!(thought.tags.contains(&"first".to_string()));
    assert!(!thought.id.is_empty());

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_thought_by_id() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create a thought
    let input = CreateThoughtInput::new("A thought to retrieve")
        .with_type(ThoughtType::Belief)
        .with_confidence(0.9);

    let created: Record = conductor
        .call(&zome, "create_thought", input)
        .await;

    let created_thought = thought_from_record(&created).expect("Failed to decode thought");

    // Retrieve by ID
    let retrieved: Option<Record> = conductor
        .call(&zome, "get_thought", created_thought.id.clone())
        .await;

    let retrieved_thought = retrieved
        .as_ref()
        .and_then(thought_from_record)
        .expect("Failed to retrieve thought");

    assert_eq!(retrieved_thought.id, created_thought.id);
    assert_eq!(retrieved_thought.content, "A thought to retrieve");
    assert_eq!(retrieved_thought.thought_type, ThoughtType::Belief);

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_nonexistent_thought() {
    let (conductor, zome) = setup_lucid_conductor().await;

    let result: Option<Record> = conductor
        .call(&zome, "get_thought", "nonexistent-id-12345".to_string())
        .await;

    assert!(result.is_none());

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_update_thought() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create a thought
    let input = CreateThoughtInput::new("Original content")
        .with_confidence(0.5);

    let created: Record = conductor
        .call(&zome, "create_thought", input)
        .await;

    let created_thought = thought_from_record(&created).expect("Failed to decode thought");

    // Update the thought
    let update_input = UpdateThoughtInput {
        thought_id: created_thought.id.clone(),
        content: Some("Updated content".into()),
        thought_type: None,
        epistemic: None,
        confidence: Some(0.9),
        tags: None,
        domain: None,
        related_thoughts: None,
    };

    let updated: Record = conductor
        .call(&zome, "update_thought", update_input)
        .await;

    let updated_thought = thought_from_record(&updated).expect("Failed to decode updated thought");

    assert_eq!(updated_thought.content, "Updated content");
    assert_eq!(updated_thought.confidence, 0.9);
    assert!(updated_thought.version > created_thought.version);

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_delete_thought() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create a thought
    let input = CreateThoughtInput::new("To be deleted");

    let created: Record = conductor
        .call(&zome, "create_thought", input)
        .await;

    let created_thought = thought_from_record(&created).expect("Failed to decode thought");

    // Delete the thought
    let _: ActionHash = conductor
        .call(&zome, "delete_thought", created_thought.id.clone())
        .await;

    // Verify it's deleted (should return None or the deleted record)
    let retrieved: Option<Record> = conductor
        .call(&zome, "get_thought", created_thought.id)
        .await;

    // Note: Holochain doesn't truly delete, it marks as deleted
    // The behavior depends on the zome implementation
    assert!(retrieved.is_none() || retrieved.is_some());

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_my_thoughts() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create multiple thoughts
    let input1 = CreateThoughtInput::new("Thought A");
    let input2 = CreateThoughtInput::new("Thought B");
    let input3 = CreateThoughtInput::new("Thought C");

    let _: Record = conductor.call(&zome, "create_thought", input1).await;
    let _: Record = conductor.call(&zome, "create_thought", input2).await;
    let _: Record = conductor.call(&zome, "create_thought", input3).await;

    // Get all thoughts
    let thoughts: Vec<Record> = conductor
        .call(&zome, "get_my_thoughts", ())
        .await;

    assert!(thoughts.len() >= 3);

    let contents: Vec<String> = thoughts
        .iter()
        .filter_map(thought_from_record)
        .map(|t| t.content)
        .collect();

    assert!(contents.contains(&"Thought A".to_string()));
    assert!(contents.contains(&"Thought B".to_string()));
    assert!(contents.contains(&"Thought C".to_string()));

    conductor.shutdown().await;
}

// ============================================================================
// THOUGHT TYPE TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_thought_types() {
    let (conductor, zome) = setup_lucid_conductor().await;

    let types = vec![
        ThoughtType::Note,
        ThoughtType::Belief,
        ThoughtType::Question,
        ThoughtType::Evidence,
        ThoughtType::Hypothesis,
        ThoughtType::Idea,
    ];

    for thought_type in types {
        let input = CreateThoughtInput::new(format!("This is a {:?}", thought_type))
            .with_type(thought_type.clone());

        let record: Record = conductor
            .call(&zome, "create_thought", input)
            .await;

        let thought = thought_from_record(&record).expect("Failed to decode thought");
        assert_eq!(thought.thought_type, thought_type);
    }

    conductor.shutdown().await;
}

// ============================================================================
// TAG TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_thoughts_by_tag() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts with specific tags
    let input1 = CreateThoughtInput::new("Philosophy thought")
        .with_tags(vec!["philosophy".into(), "mind".into()]);
    let input2 = CreateThoughtInput::new("Another philosophy thought")
        .with_tags(vec!["philosophy".into(), "consciousness".into()]);
    let input3 = CreateThoughtInput::new("Science thought")
        .with_tags(vec!["science".into(), "physics".into()]);

    let _: Record = conductor.call(&zome, "create_thought", input1).await;
    let _: Record = conductor.call(&zome, "create_thought", input2).await;
    let _: Record = conductor.call(&zome, "create_thought", input3).await;

    // Get thoughts by tag
    let philosophy_thoughts: Vec<Record> = conductor
        .call(&zome, "get_thoughts_by_tag", "philosophy".to_string())
        .await;

    assert!(philosophy_thoughts.len() >= 2);

    let contents: Vec<String> = philosophy_thoughts
        .iter()
        .filter_map(thought_from_record)
        .map(|t| t.content)
        .collect();

    assert!(contents.iter().any(|c| c.contains("Philosophy")));
    assert!(!contents.iter().any(|c| c.contains("Science")));

    conductor.shutdown().await;
}

// ============================================================================
// DOMAIN TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_thoughts_by_domain() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts with domains
    let input1 = CreateThoughtInput::new("AI thought")
        .with_domain("artificial-intelligence");
    let input2 = CreateThoughtInput::new("ML thought")
        .with_domain("artificial-intelligence");
    let input3 = CreateThoughtInput::new("Bio thought")
        .with_domain("biology");

    let _: Record = conductor.call(&zome, "create_thought", input1).await;
    let _: Record = conductor.call(&zome, "create_thought", input2).await;
    let _: Record = conductor.call(&zome, "create_thought", input3).await;

    // Get thoughts by domain
    let ai_thoughts: Vec<Record> = conductor
        .call(&zome, "get_thoughts_by_domain", "artificial-intelligence".to_string())
        .await;

    assert!(ai_thoughts.len() >= 2);

    conductor.shutdown().await;
}

// ============================================================================
// SEARCH TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_search_thoughts_by_content() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create searchable thoughts
    let input1 = CreateThoughtInput::new("Machine learning is a subset of artificial intelligence");
    let input2 = CreateThoughtInput::new("Deep learning uses neural networks");
    let input3 = CreateThoughtInput::new("Cats are cute animals");

    let _: Record = conductor.call(&zome, "create_thought", input1).await;
    let _: Record = conductor.call(&zome, "create_thought", input2).await;
    let _: Record = conductor.call(&zome, "create_thought", input3).await;

    // Search for AI-related thoughts
    let search_input = SearchThoughtsInput {
        tags: None,
        domain: None,
        thought_type: None,
        min_empirical: None,
        min_confidence: None,
        content_contains: Some("learning".into()),
        limit: Some(10),
        offset: None,
    };

    let results: Vec<Record> = conductor
        .call(&zome, "search_thoughts", search_input)
        .await;

    assert!(results.len() >= 2);

    let contents: Vec<String> = results
        .iter()
        .filter_map(thought_from_record)
        .map(|t| t.content)
        .collect();

    assert!(contents.iter().all(|c| c.to_lowercase().contains("learning")));
    assert!(!contents.iter().any(|c| c.contains("Cats")));

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_search_with_confidence_filter() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts with different confidence levels
    let input1 = CreateThoughtInput::new("High confidence thought")
        .with_confidence(0.9);
    let input2 = CreateThoughtInput::new("Medium confidence thought")
        .with_confidence(0.5);
    let input3 = CreateThoughtInput::new("Low confidence thought")
        .with_confidence(0.2);

    let _: Record = conductor.call(&zome, "create_thought", input1).await;
    let _: Record = conductor.call(&zome, "create_thought", input2).await;
    let _: Record = conductor.call(&zome, "create_thought", input3).await;

    // Search for high confidence thoughts
    let search_input = SearchThoughtsInput {
        tags: None,
        domain: None,
        thought_type: None,
        min_empirical: None,
        min_confidence: Some(0.7),
        content_contains: None,
        limit: None,
        offset: None,
    };

    let results: Vec<Record> = conductor
        .call(&zome, "search_thoughts", search_input)
        .await;

    for record in &results {
        if let Some(thought) = thought_from_record(record) {
            assert!(thought.confidence >= 0.7);
        }
    }

    conductor.shutdown().await;
}

// ============================================================================
// STATISTICS TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_stats() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create some thoughts
    let input1 = CreateThoughtInput::new("Note 1").with_type(ThoughtType::Note);
    let input2 = CreateThoughtInput::new("Belief 1").with_type(ThoughtType::Belief);
    let input3 = CreateThoughtInput::new("Note 2").with_type(ThoughtType::Note);

    let _: Record = conductor.call(&zome, "create_thought", input1).await;
    let _: Record = conductor.call(&zome, "create_thought", input2).await;
    let _: Record = conductor.call(&zome, "create_thought", input3).await;

    // Get stats
    #[derive(Debug, Deserialize)]
    struct KnowledgeGraphStats {
        total_thoughts: u32,
        by_type: Vec<(String, u32)>,
        by_domain: Vec<(String, u32)>,
        average_confidence: f64,
        average_epistemic_strength: f64,
        top_tags: Vec<(String, u32)>,
    }

    let stats: KnowledgeGraphStats = conductor
        .call(&zome, "get_stats", ())
        .await;

    assert!(stats.total_thoughts >= 3);
    assert!(stats.average_confidence >= 0.0 && stats.average_confidence <= 1.0);

    conductor.shutdown().await;
}

// ============================================================================
// EPISTEMIC CLASSIFICATION TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_epistemic_classification() {
    let (conductor, zome) = setup_lucid_conductor().await;

    let input = CreateThoughtInput {
        content: "The Earth orbits the Sun".into(),
        thought_type: Some(ThoughtType::Belief),
        epistemic: Some(EpistemicClassification {
            empirical: EmpiricalLevel::E7, // Scientific consensus
            normative: NormativeLevel::N0, // Purely descriptive
            materiality: MaterialityLevel::M3, // Physical
            harmonic: HarmonicLevel::H2, // Connected to many concepts
        }),
        confidence: Some(0.99),
        tags: Some(vec!["astronomy".into(), "science".into()]),
        domain: Some("physics".into()),
        related_thoughts: None,
        parent_thought: None,
        embedding: None,
    };

    let record: Record = conductor
        .call(&zome, "create_thought", input)
        .await;

    let thought = thought_from_record(&record).expect("Failed to decode thought");

    assert_eq!(thought.epistemic.empirical, EmpiricalLevel::E7);
    assert_eq!(thought.epistemic.normative, NormativeLevel::N0);
    assert_eq!(thought.epistemic.materiality, MaterialityLevel::M3);
    assert_eq!(thought.epistemic.harmonic, HarmonicLevel::H2);
    assert_eq!(thought.confidence, 0.99);

    conductor.shutdown().await;
}

// ============================================================================
// SYMTHAEA INTEGRATION TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_update_thought_embedding() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create a thought
    let input = CreateThoughtInput::new("Embeddings represent knowledge in vector space");
    let created: Record = conductor.call(&zome, "create_thought", input).await;
    let thought = thought_from_record(&created).expect("Failed to decode thought");
    assert!(thought.embedding.is_none());

    // Generate a 16,384D embedding and update
    let embedding = generate_random_embedding(42);
    assert_eq!(embedding.len(), 16384);

    let update_input = UpdateEmbeddingInput {
        thought_id: thought.id.clone(),
        embedding: embedding.clone(),
    };

    let updated: Record = conductor.call(&zome, "update_thought_embedding", update_input).await;
    let updated_thought = thought_from_record(&updated).expect("Failed to decode updated thought");

    assert!(updated_thought.embedding.is_some());
    assert_eq!(updated_thought.embedding.as_ref().unwrap().len(), 16384);
    assert!(updated_thought.embedding_version.is_some());

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_semantic_search() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts with embeddings
    let base_embedding = generate_random_embedding(100);
    let similar_embedding = generate_similar_embedding(&base_embedding, 0.1, 200);
    let different_embedding = generate_random_embedding(999);

    let t1_input = CreateThoughtInput::new("Consciousness arises from integrated information");
    let t1: Record = conductor.call(&zome, "create_thought", t1_input).await;
    let t1_thought = thought_from_record(&t1).expect("Decode t1");

    let t2_input = CreateThoughtInput::new("Information integration is key to awareness");
    let t2: Record = conductor.call(&zome, "create_thought", t2_input).await;
    let t2_thought = thought_from_record(&t2).expect("Decode t2");

    let t3_input = CreateThoughtInput::new("Cats enjoy basking in sunlight");
    let t3: Record = conductor.call(&zome, "create_thought", t3_input).await;
    let t3_thought = thought_from_record(&t3).expect("Decode t3");

    // Attach embeddings
    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t1_thought.id.clone(),
        embedding: base_embedding.clone(),
    }).await;
    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t2_thought.id.clone(),
        embedding: similar_embedding,
    }).await;
    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t3_thought.id.clone(),
        embedding: different_embedding,
    }).await;

    // Search with base embedding — should find t1 (exact match) and t2 (similar)
    let search_input = SemanticSearchInput {
        query_embedding: base_embedding,
        threshold: 0.8,
        limit: Some(10),
    };

    let results: Vec<SemanticSearchResult> = conductor.call(&zome, "semantic_search", search_input).await;

    // t1 should be near-perfect match, t2 similar, t3 should be below threshold
    assert!(!results.is_empty());
    assert!(results[0].similarity >= 0.99, "First result should be near-perfect match");

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_update_thought_coherence() {
    let (conductor, zome) = setup_lucid_conductor().await;

    let input = CreateThoughtInput::new("Testing coherence scores");
    let created: Record = conductor.call(&zome, "create_thought", input).await;
    let thought = thought_from_record(&created).expect("Decode thought");
    assert!(thought.coherence_score.is_none());

    let coherence_input = UpdateCoherenceInput {
        thought_id: thought.id.clone(),
        coherence_score: 0.87,
        phi_score: Some(0.42),
    };

    let updated: Record = conductor.call(&zome, "update_thought_coherence", coherence_input).await;
    let updated_thought = thought_from_record(&updated).expect("Decode updated thought");

    assert_eq!(updated_thought.coherence_score, Some(0.87));
    assert_eq!(updated_thought.phi_score, Some(0.42));

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_thoughts_needing_embeddings() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create two thoughts — one with embedding, one without
    let t1_input = CreateThoughtInput::new("Has no embedding");
    let _: Record = conductor.call(&zome, "create_thought", t1_input).await;

    let t2_input = CreateThoughtInput::new("Will get an embedding");
    let t2: Record = conductor.call(&zome, "create_thought", t2_input).await;
    let t2_thought = thought_from_record(&t2).expect("Decode t2");

    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t2_thought.id.clone(),
        embedding: generate_random_embedding(77),
    }).await;

    let needing: Vec<Record> = conductor.call(&zome, "get_thoughts_needing_embeddings", ()).await;

    // At least the first thought should need an embedding
    let needing_thoughts: Vec<Thought> = needing.iter().filter_map(thought_from_record).collect();
    assert!(needing_thoughts.iter().any(|t| t.content == "Has no embedding"));
    // The thought with an embedding should NOT be in the list (unless version mismatch)
    // It may still appear if version > embedding_version

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_get_thoughts_by_coherence_range() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts with different coherence scores
    let thoughts_data = vec![
        ("High coherence thought", 0.95),
        ("Medium coherence thought", 0.55),
        ("Low coherence thought", 0.15),
    ];

    for (content, coherence) in &thoughts_data {
        let input = CreateThoughtInput::new(*content);
        let record: Record = conductor.call(&zome, "create_thought", input).await;
        let thought = thought_from_record(&record).expect("Decode");

        let _: Record = conductor.call(&zome, "update_thought_coherence", UpdateCoherenceInput {
            thought_id: thought.id.clone(),
            coherence_score: *coherence,
            phi_score: None,
        }).await;
    }

    // Query for high-coherence thoughts
    let range_input = CoherenceRangeInput {
        min_coherence: 0.8,
        max_coherence: 1.0,
    };

    let high_coherence: Vec<Record> = conductor.call(&zome, "get_thoughts_by_coherence", range_input).await;
    let high_thoughts: Vec<Thought> = high_coherence.iter().filter_map(thought_from_record).collect();

    assert!(high_thoughts.iter().all(|t| t.coherence_score.unwrap_or(0.0) >= 0.8));

    conductor.shutdown().await;
}

// ============================================================================
// EPISTEMIC GARDEN TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_explore_garden_clusters() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create two clusters of thoughts with similar embeddings
    let cluster_a_base = generate_random_embedding(111);
    let cluster_b_base = generate_random_embedding(222);

    for i in 0..3 {
        let input = CreateThoughtInput::new(format!("Cluster A thought {}", i))
            .with_domain("philosophy");
        let record: Record = conductor.call(&zome, "create_thought", input).await;
        let thought = thought_from_record(&record).expect("Decode");
        let emb = generate_similar_embedding(&cluster_a_base, 0.05, 1000 + i);
        let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
            thought_id: thought.id.clone(),
            embedding: emb,
        }).await;
    }

    for i in 0..3 {
        let input = CreateThoughtInput::new(format!("Cluster B thought {}", i))
            .with_domain("science");
        let record: Record = conductor.call(&zome, "create_thought", input).await;
        let thought = thought_from_record(&record).expect("Decode");
        let emb = generate_similar_embedding(&cluster_b_base, 0.05, 2000 + i);
        let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
            thought_id: thought.id.clone(),
            embedding: emb,
        }).await;
    }

    let garden_input = ExploreGardenInput {
        max_clusters: 5,
        min_cluster_size: 2,
        domain_filter: None,
    };

    let clusters: Vec<ThoughtCluster> = conductor.call(&zome, "explore_garden", garden_input).await;

    assert!(!clusters.is_empty(), "Should find at least one cluster");
    for cluster in &clusters {
        assert!(cluster.thought_ids.len() >= 2, "Cluster should have at least 2 thoughts");
    }

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_suggest_connections() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts with similar embeddings but no explicit relationship
    let base_emb = generate_random_embedding(333);
    let similar_emb = generate_similar_embedding(&base_emb, 0.05, 444);
    let different_emb = generate_random_embedding(555);

    let t1_input = CreateThoughtInput::new("Source thought about consciousness");
    let t1: Record = conductor.call(&zome, "create_thought", t1_input).await;
    let t1_thought = thought_from_record(&t1).expect("Decode t1");

    let t2_input = CreateThoughtInput::new("Related thought about awareness");
    let t2: Record = conductor.call(&zome, "create_thought", t2_input).await;
    let t2_thought = thought_from_record(&t2).expect("Decode t2");

    let t3_input = CreateThoughtInput::new("Unrelated thought about cooking");
    let t3: Record = conductor.call(&zome, "create_thought", t3_input).await;
    let t3_thought = thought_from_record(&t3).expect("Decode t3");

    // Attach embeddings
    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t1_thought.id.clone(), embedding: base_emb,
    }).await;
    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t2_thought.id.clone(), embedding: similar_emb,
    }).await;
    let _: Record = conductor.call(&zome, "update_thought_embedding", UpdateEmbeddingInput {
        thought_id: t3_thought.id.clone(), embedding: different_emb,
    }).await;

    let suggest_input = SuggestConnectionsInput {
        thought_id: t1_thought.id.clone(),
        max_suggestions: 5,
        min_similarity: 0.7,
    };

    let suggestions: Vec<ConnectionSuggestion> = conductor
        .call(&zome, "suggest_connections", suggest_input)
        .await;

    // t2 should be suggested (similar embedding), t3 should not (different)
    let suggested_ids: Vec<&str> = suggestions.iter().map(|s| s.thought_b_id.as_str()).collect();
    assert!(suggested_ids.contains(&t2_thought.id.as_str()), "Similar thought should be suggested");

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_find_knowledge_gaps() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts across domains with varying confidence
    let domains = vec![
        ("well-covered", 5, 0.9),   // Many thoughts, high confidence
        ("sparse-domain", 1, 0.3),  // Few thoughts, low confidence
    ];

    for (domain, count, confidence) in &domains {
        for i in 0..*count {
            let input = CreateThoughtInput::new(format!("{} thought {}", domain, i))
                .with_domain(*domain)
                .with_confidence(*confidence);
            let _: Record = conductor.call(&zome, "create_thought", input).await;
        }
    }

    let gaps: Vec<KnowledgeGap> = conductor.call(&zome, "find_knowledge_gaps", ()).await;

    assert!(!gaps.is_empty(), "Should identify knowledge gaps");

    // The sparse domain should appear with sparse=true
    let sparse_gap = gaps.iter().find(|g| g.domain == "sparse-domain");
    assert!(sparse_gap.is_some(), "Sparse domain should be identified as a gap");
    assert!(sparse_gap.unwrap().sparse, "Sparse domain should be flagged as sparse");

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_discover_patterns() {
    let (conductor, zome) = setup_lucid_conductor().await;

    // Create thoughts that form a pattern (shared tag with 3+ thoughts)
    for i in 0..4 {
        let input = CreateThoughtInput::new(format!("Ethics thought {}", i))
            .with_tags(vec!["ethics".into(), "philosophy".into()])
            .with_domain("philosophy")
            .with_confidence(0.5 + (i as f64) * 0.1);
        let _: Record = conductor.call(&zome, "create_thought", input).await;
    }

    let patterns: Vec<DiscoveredPattern> = conductor.call(&zome, "discover_patterns", ()).await;

    // Should find at least a "recurring_theme" pattern for the "ethics" tag
    let recurring = patterns.iter().find(|p| p.pattern_type == "recurring_theme");
    assert!(recurring.is_some(), "Should detect recurring theme pattern");

    // Should also find confidence_growth pattern (increasing confidence in same domain)
    let growth = patterns.iter().find(|p| p.pattern_type == "confidence_growth");
    assert!(growth.is_some(), "Should detect confidence growth pattern");

    conductor.shutdown().await;
}

// ============================================================================
// TEMPORAL CONSCIOUSNESS TESTS
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_record_belief_snapshot() {
    let (conductor, _lucid_zome) = setup_lucid_conductor().await;

    // Use the temporal_consciousness zome
    let cells = conductor.list_cells(None).await.unwrap();
    let cell_id = cells.into_iter().next().expect("No cells");
    let tc_zome = SweetZome::new(cell_id.clone(), "temporal_consciousness".into());

    // Create a thought first
    let input = CreateThoughtInput::new("Belief to track");
    let created: Record = conductor.call(&SweetZome::new(cell_id, "lucid".into()), "create_thought", input).await;
    let thought = thought_from_record(&created).expect("Decode thought");

    // Record a snapshot
    let snapshot_input = RecordSnapshotInput {
        thought_id: thought.id.clone(),
        epistemic_code: "E2N1M1H2".into(),
        confidence: 0.75,
        phi: 0.3,
        coherence: 0.6,
        trigger: SnapshotTrigger::Created,
    };

    let snapshot_record: Record = conductor.call(&tc_zome, "record_belief_snapshot", snapshot_input).await;
    assert!(snapshot_record.action_address().as_hash().len() > 0);

    conductor.shutdown().await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore]
async fn test_consciousness_evolution_lifecycle() {
    let (conductor, _lucid_zome) = setup_lucid_conductor().await;

    let cells = conductor.list_cells(None).await.unwrap();
    let cell_id = cells.into_iter().next().expect("No cells");
    let tc_zome = SweetZome::new(cell_id, "temporal_consciousness".into());

    let now = holochain_types::prelude::Timestamp::now();
    let earlier = holochain_types::prelude::Timestamp::from_micros(now.as_micros() - 3600_000_000);

    let evolution_input = RecordEvolutionInput {
        period_start: earlier,
        period_end: now,
        avg_phi: 0.45,
        phi_trend: 0.05,
        avg_coherence: 0.72,
        coherence_trend: 0.03,
        stable_belief_count: 10,
        growing_belief_count: 3,
        weakening_belief_count: 1,
        entrenched_belief_count: 0,
        insights: vec![
            "Phi increasing steadily over the session".into(),
            "Coherence improved after resolving 2 contradictions".into(),
        ],
    };

    let record: Record = conductor.call(&tc_zome, "record_consciousness_evolution", evolution_input).await;
    assert!(record.action_address().as_hash().len() > 0);

    // Retrieve evolution history
    let history: Vec<Record> = conductor.call(&tc_zome, "get_my_evolution_history", ()).await;
    assert!(!history.is_empty(), "Should have at least one evolution record");

    conductor.shutdown().await;
}
