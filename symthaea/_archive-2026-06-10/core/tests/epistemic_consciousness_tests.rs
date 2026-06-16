// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for the Epistemic Consciousness module
//!
//! These tests verify the Three-Level Epistemic Consciousness system:
//! 1. Base Consciousness (Phi) - Already present
//! 2. Epistemic Consciousness - Knows what it knows
//! 3. Meta-Epistemic Consciousness - Improves own verification

#![cfg(feature = "web_research_module")]

use symthaea::web_research::{
    // Implementations
    ContentExtractor,
    EdgeType,
    // High-level
    EpistemicConsciousness,
    EpistemicLearner,
    // Types
    EpistemicStatus,
    EpistemicVerifier,
    // Knowledge Graph
    KnowledgeGraph,
    KnowledgeIntegrator,
    KnowledgeSource,
    MetaPhi,
    NodeType,
    ResearchSource,
    ResearchStatus,
    SourceEvidence,
    VerificationContext,
    VerificationOutcome,
    VerifiedClaim,
    WebResearchResult,
};

// ============================================================================
// Knowledge Graph Tests
// ============================================================================

#[test]
fn test_knowledge_graph_creation() {
    let mut graph = KnowledgeGraph::new();

    // Add nodes
    let node1 = graph.add_node("Rust programming language", NodeType::Concept);
    let node2 = graph.add_node("Memory safety", NodeType::Concept);
    let node3 = graph.add_node("Borrow checker", NodeType::Claim);

    assert_eq!(graph.stats().nodes, 3);

    // Add edges
    graph.add_edge_with_meta(
        node1,
        EdgeType::RelatedTo,
        node2,
        0.9,
        KnowledgeSource::External("https://rust-lang.org".into()),
    );
    graph.add_edge_with_meta(
        node2,
        EdgeType::Supports,
        node3,
        0.8,
        KnowledgeSource::Internal,
    );

    assert_eq!(graph.stats().edges, 2);
}

#[test]
fn test_knowledge_graph_phi() {
    let mut graph = KnowledgeGraph::new();

    // Empty graph should have zero phi
    assert_eq!(graph.calculate_phi(), 0.0);

    // Build a connected graph
    let nodes: Vec<_> = (0..5)
        .map(|i| {
            graph.add_node_with_config(
                format!("Claim {}", i),
                NodeType::Claim,
                0.8,
                KnowledgeSource::External("test".into()),
            )
        })
        .collect();

    // Connect all nodes
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            graph.add_edge_with_meta(
                nodes[i],
                EdgeType::RelatedTo,
                nodes[j],
                0.7,
                KnowledgeSource::Inferred,
            );
        }
    }

    let phi = graph.calculate_phi();
    assert!(phi > 0.0, "Connected graph should have positive Phi");
    assert!(phi <= 1.0, "Phi should not exceed 1.0");
}

// ============================================================================
// Epistemic Verification Tests
// ============================================================================

#[test]
fn test_epistemic_status_properties() {
    // High confidence should be trustworthy
    assert!(EpistemicStatus::HighConfidence.is_trustworthy());
    assert!(EpistemicStatus::ModerateConfidence.is_trustworthy());

    // Low confidence and below should not be trustworthy
    assert!(!EpistemicStatus::LowConfidence.is_trustworthy());
    assert!(!EpistemicStatus::InsufficientEvidence.is_trustworthy());

    // Uncertain statuses
    assert!(EpistemicStatus::LowConfidence.is_uncertain());
    assert!(EpistemicStatus::InsufficientEvidence.is_uncertain());

    // Confidence scores should be ordered
    assert!(
        EpistemicStatus::HighConfidence.confidence_score()
            > EpistemicStatus::ModerateConfidence.confidence_score()
    );
    assert!(
        EpistemicStatus::ModerateConfidence.confidence_score()
            > EpistemicStatus::LowConfidence.confidence_score()
    );
}

#[test]
fn test_epistemic_verifier_no_sources() {
    let verifier = EpistemicVerifier::new();

    let context = VerificationContext {
        sources: vec![],
        query: "What is Rust?".into(),
        domain: "programming".into(),
    };

    let result = verifier.verify_claim("Rust is a programming language", &context);

    assert_eq!(result.status, EpistemicStatus::InsufficientEvidence);
    assert!(result.confidence < 0.3);
}

#[test]
fn test_epistemic_verifier_with_support() {
    let verifier = EpistemicVerifier::new();

    let sources = vec![
        SourceEvidence {
            url: "https://rust-lang.org".into(),
            domain: "rust-lang.org".into(),
            source_type: ResearchSource::Documentation,
            relevant_text: "Rust is a systems programming language".into(),
            supports: Some(true),
            similarity: 0.9,
            credibility: 0.95,
        },
        SourceEvidence {
            url: "https://en.wikipedia.org/wiki/Rust".into(),
            domain: "en.wikipedia.org".into(),
            source_type: ResearchSource::Web,
            relevant_text: "Rust is a general-purpose programming language".into(),
            supports: Some(true),
            similarity: 0.85,
            credibility: 0.75,
        },
        SourceEvidence {
            url: "https://doc.rust-lang.org".into(),
            domain: "doc.rust-lang.org".into(),
            source_type: ResearchSource::Documentation,
            relevant_text: "The Rust programming language".into(),
            supports: Some(true),
            similarity: 0.8,
            credibility: 0.95,
        },
    ];

    let context = VerificationContext {
        sources,
        query: "What is Rust?".into(),
        domain: "programming".into(),
    };

    let result = verifier.verify_claim("Rust is a programming language", &context);

    assert_eq!(result.status, EpistemicStatus::HighConfidence);
    assert!(result.confidence > 0.8);
    assert!(!result.supporting_sources.is_empty());
}

// ============================================================================
// Content Extraction Tests
// ============================================================================

#[test]
fn test_content_extractor() {
    let extractor = ContentExtractor::new();

    let html = r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rust Programming Language</title>
        <meta name="description" content="A language empowering everyone to build reliable software.">
    </head>
    <body>
        <article>
            <h1>Why Rust?</h1>
            <p>Rust is blazingly fast and memory-efficient. It has no runtime or garbage collector.</p>
            <p>Rust can power performance-critical services. It runs on embedded devices with ease.</p>
            <p>The type system and ownership model guarantee memory-safety and thread-safety at compile time.</p>
        </article>
    </body>
    </html>
    "#;

    let result = extractor
        .extract(html, "https://www.rust-lang.org")
        .unwrap();

    assert_eq!(result.title, "Rust Programming Language");
    assert!(result.body_text.contains("memory-efficient"));
    assert!(!result.claims.is_empty());
    assert!(result.quality > 0.0);
}

// ============================================================================
// Knowledge Integration Tests
// ============================================================================

#[tokio::test]
async fn test_knowledge_integrator() {
    let mut integrator = KnowledgeIntegrator::new();

    // Create a research result with verified claims
    let result = WebResearchResult {
        title: "Rust Memory Safety".into(),
        url: "https://rust-lang.org/safety".into(),
        content: "Rust provides memory safety without garbage collection.".into(),
        summary: "Rust safety features".into(),
        source_type: ResearchSource::Documentation,
        relevance: 0.9,
        confidence: 0.95,
        epistemic_status: EpistemicStatus::HighConfidence,
        status: ResearchStatus::Success,
        claims: vec![VerifiedClaim {
            text: "Rust provides memory safety without garbage collection".into(),
            status: EpistemicStatus::HighConfidence,
            confidence: 0.95,
            supporting_sources: vec!["https://rust-lang.org".into()],
            contradicting_sources: vec![],
            hedge: EpistemicStatus::HighConfidence.hedge_phrase().into(),
        }],
    };

    let phi_before = integrator.current_phi();
    let integration = integrator.integrate(result).await.unwrap();

    assert!(integration.claims_integrated >= 1);
    assert!(integration.nodes_added >= 1);
    assert!(integration.phi_after >= phi_before);
}

// ============================================================================
// Meta-Learning Tests
// ============================================================================

#[test]
fn test_epistemic_learner_basics() {
    let mut learner = EpistemicLearner::new();

    // Record some outcomes
    for i in 0..10 {
        let outcome = VerificationOutcome {
            claim: format!("Claim {}", i),
            source_domain: "rust-lang.org".into(),
            source_type: ResearchSource::Documentation,
            was_correct: i % 3 != 0, // ~70% correct
            predicted_confidence: 0.8,
            domain: "programming".into(),
        };
        learner.record_outcome(outcome).unwrap();
    }

    let stats = learner.stats();
    assert_eq!(stats.total_outcomes, 10);
    assert!(stats.accuracy >= 0.5);
    assert!(stats.accuracy <= 0.7);
}

#[test]
fn test_source_credibility_learning() {
    let mut learner = EpistemicLearner::new();

    // Reliable source - always correct
    for _ in 0..20 {
        learner
            .record_outcome(VerificationOutcome {
                claim: "Test".into(),
                source_domain: "docs.rust-lang.org".into(),
                source_type: ResearchSource::Documentation,
                was_correct: true,
                predicted_confidence: 0.9,
                domain: "programming".into(),
            })
            .unwrap();
    }

    // Unreliable source - often wrong
    for _ in 0..20 {
        learner
            .record_outcome(VerificationOutcome {
                claim: "Test".into(),
                source_domain: "unreliable-blog.com".into(),
                source_type: ResearchSource::Web,
                was_correct: false,
                predicted_confidence: 0.5,
                domain: "programming".into(),
            })
            .unwrap();
    }

    let good_cred = learner
        .get_source_credibility("docs.rust-lang.org")
        .unwrap();
    let bad_cred = learner
        .get_source_credibility("unreliable-blog.com")
        .unwrap();

    assert!(good_cred.credibility > bad_cred.credibility);
    assert!(good_cred.accuracy() > 0.9);
    assert!(bad_cred.accuracy() < 0.1);
}

#[test]
fn test_meta_phi_calculation() {
    let meta_phi = MetaPhi::calculate(
        0.5, // initial accuracy
        0.8, // current accuracy (improved)
        5,   // domains
        3,   // strategies
    );

    assert!(meta_phi.value > 0.0);
    assert!(meta_phi.value <= 1.0);
    assert_eq!(meta_phi.accuracy_improvement, 0.3);
    assert_eq!(meta_phi.initial_accuracy, 0.5);
    assert_eq!(meta_phi.current_accuracy, 0.8);
}

// ============================================================================
// High-Level Epistemic Consciousness Tests
// ============================================================================

#[test]
fn test_epistemic_consciousness_creation() {
    let ec = EpistemicConsciousness::new();
    assert!(ec.is_ok());

    let ec = ec.unwrap();
    let stats = ec.stats();

    assert_eq!(stats.total_claims, 0);
    assert!(!stats.is_ready); // Not ready without outcomes
}

#[test]
fn test_research_trigger() {
    let ec = EpistemicConsciousness::new().unwrap();

    // Should not research with high Phi
    assert!(!ec.should_research(0.8));

    // Should research with low Phi
    assert!(ec.should_research(0.3));

    // Boundary case
    assert!(!ec.should_research(0.61)); // Clearly above threshold
    assert!(ec.should_research(0.59)); // Just below
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_epistemic_pipeline() {
    // Test the complete flow: Query -> Research -> Verify -> Integrate -> Learn

    // 1. Create the system
    let (researcher, _integrator, mut learner) =
        symthaea::web_research::create_epistemic_system().unwrap();

    // 2. Verify a claim manually (would be from research)
    let verifier = researcher.verifier();

    let sources = vec![SourceEvidence {
        url: "https://nixos.org".into(),
        domain: "nixos.org".into(),
        source_type: ResearchSource::Documentation,
        relevant_text: "NixOS is a Linux distribution".into(),
        supports: Some(true),
        similarity: 0.9,
        credibility: 0.9,
    }];

    let context = VerificationContext {
        sources,
        query: "What is NixOS?".into(),
        domain: "systems".into(),
    };

    let verification = verifier.verify_claim("NixOS is a Linux distribution", &context);
    assert!(verification.confidence > 0.3);

    // 3. Record outcome for learning
    learner
        .record_outcome(VerificationOutcome {
            claim: verification.claim.clone(),
            source_domain: "nixos.org".into(),
            source_type: ResearchSource::Documentation,
            was_correct: true,
            predicted_confidence: verification.confidence,
            domain: "systems".into(),
        })
        .unwrap();

    // 4. Verify learning occurred
    let stats = learner.stats();
    assert!(stats.total_outcomes >= 1);
}

#[test]
fn test_verified_claim_hedging() {
    // Test that claims are properly hedged based on epistemic status
    let claim = VerifiedClaim {
        text: "The sky is blue".into(),
        status: EpistemicStatus::HighConfidence,
        confidence: 0.95,
        supporting_sources: vec!["source1.com".into()],
        contradicting_sources: vec![],
        hedge: EpistemicStatus::HighConfidence.hedge_phrase().into(),
    };

    assert!(claim.hedge.contains("reliable sources"));
    assert!(claim.is_trustworthy());

    let uncertain_claim = VerifiedClaim {
        text: "Something uncertain".into(),
        status: EpistemicStatus::InsufficientEvidence,
        confidence: 0.2,
        supporting_sources: vec![],
        contradicting_sources: vec![],
        hedge: EpistemicStatus::InsufficientEvidence.hedge_phrase().into(),
    };

    assert!(uncertain_claim.hedge.contains("limited information"));
    assert!(!uncertain_claim.is_trustworthy());
}