#![cfg(feature = "web_research_module")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Web Research Integration Tests
//!
//! Tests for the web research system integrated with ContinuousMind.
//! These tests verify the research-consciousness connection without
//! requiring actual web requests.

use std::time::SystemTime;
use symthaea::continuous_mind::{ContinuousMind, MindConfig};
use symthaea::hdc::binary_hv::BinaryHV;
use symthaea::web_research::{
    Claim, EpistemicStatus, KnowledgeIntegrator, ResearchResult, Source, Verification,
    VerificationLevel,
};

// ============================================================================
// CONTINUOUS MIND WEB RESEARCH INTERFACE TESTS
// ============================================================================

#[tokio::test]
async fn test_continuous_mind_has_web_researcher() {
    let config = MindConfig::default();
    let mind = ContinuousMind::new(config);

    // Web researcher should be available by default
    assert!(
        mind.has_web_researcher(),
        "Web researcher should be initialized"
    );
}

#[tokio::test]
async fn test_knowledge_integrator_creation() {
    let integrator = KnowledgeIntegrator::new();
    // Default integrator should have zero accumulated state
    assert_eq!(integrator.total_claims(), 0);
}

#[tokio::test]
async fn test_knowledge_integrator_with_confidence() {
    let integrator = KnowledgeIntegrator::new().with_min_confidence(0.7);
    assert_eq!(integrator.total_claims(), 0);
}

// ============================================================================
// EPISTEMIC STATUS TESTS
// ============================================================================

#[test]
fn test_epistemic_status_variants() {
    let statuses = [
        EpistemicStatus::HighConfidence,
        EpistemicStatus::ModerateConfidence,
        EpistemicStatus::LowConfidence,
        EpistemicStatus::Contested,
        EpistemicStatus::Unverifiable,
        EpistemicStatus::False,
    ];

    // Each should be distinct
    for (i, s1) in statuses.iter().enumerate() {
        for (j, s2) in statuses.iter().enumerate() {
            if i != j {
                assert_ne!(s1, s2);
            }
        }
    }
}

#[test]
fn test_epistemic_status_debug() {
    let status = EpistemicStatus::HighConfidence;
    let debug = format!("{:?}", status);
    assert!(debug.contains("HighConfidence"));
}

// ============================================================================
// VERIFICATION LEVEL TESTS
// ============================================================================

#[test]
fn test_verification_level_variants() {
    let levels = [
        VerificationLevel::Minimal,
        VerificationLevel::Standard,
        VerificationLevel::Rigorous,
        VerificationLevel::Academic,
    ];

    assert_eq!(levels.len(), 4);
}

#[test]
fn test_verification_level_debug() {
    let level = VerificationLevel::Academic;
    let debug = format!("{:?}", level);
    assert!(debug.contains("Academic"));
}

// ============================================================================
// SOURCE TESTS
// ============================================================================

#[test]
fn test_source_creation() {
    let source = Source {
        url: "https://example.com/article".to_string(),
        title: "Example Article".to_string(),
        content: "Article content here".to_string(),
        published_date: Some(SystemTime::now()),
        author: Some("Test Author".to_string()),
        credibility: 0.85,
        encoding: BinaryHV::random(42),
        fetch_timestamp: SystemTime::now(),
    };

    assert_eq!(source.url, "https://example.com/article");
    assert!(source.credibility > 0.8);
}

#[test]
fn test_source_credibility_comparison() {
    let high_credibility = Source {
        url: "https://nature.com".to_string(),
        title: "Peer-reviewed".to_string(),
        content: "...".to_string(),
        published_date: None,
        author: None,
        credibility: 0.95,
        encoding: BinaryHV::zero(),
        fetch_timestamp: SystemTime::now(),
    };

    let low_credibility = Source {
        url: "https://random-blog.com".to_string(),
        title: "Opinion".to_string(),
        content: "...".to_string(),
        published_date: None,
        author: None,
        credibility: 0.3,
        encoding: BinaryHV::zero(),
        fetch_timestamp: SystemTime::now(),
    };

    assert!(high_credibility.credibility > low_credibility.credibility);
}

// ============================================================================
// CLAIM TESTS
// ============================================================================

#[test]
fn test_claim_creation() {
    let claim = Claim {
        text: "The sky is blue".to_string(),
        encoding: BinaryHV::random(123),
        subject: "sky".to_string(),
        predicate: "is".to_string(),
        object: Some("blue".to_string()),
        extraction_confidence: 0.95,
    };

    assert_eq!(claim.text, "The sky is blue");
    assert!(claim.extraction_confidence > 0.9);
}

// ============================================================================
// RESEARCH RESULT TESTS
// ============================================================================

#[test]
fn test_research_result_creation() {
    let result = ResearchResult {
        query: "What is consciousness?".to_string(),
        sources: vec![Source {
            url: "https://example.com/consciousness".to_string(),
            title: "Understanding Consciousness".to_string(),
            content: "Consciousness is...".to_string(),
            published_date: None,
            author: None,
            credibility: 0.85,
            encoding: BinaryHV::random(1),
            fetch_timestamp: SystemTime::now(),
        }],
        verifications: vec![],
        confidence: 0.75,
        summary: "Consciousness is a complex phenomenon".to_string(),
        new_concepts: vec!["qualia".to_string()],
        time_taken_ms: 1500,
    };

    assert_eq!(result.query, "What is consciousness?");
    assert_eq!(result.sources.len(), 1);
    assert!(result.confidence > 0.0);
}

// ============================================================================
// INTEGRATION TESTS (No actual web requests)
// ============================================================================

#[tokio::test]
async fn test_integrate_mock_research_result() {
    let mut integrator = KnowledgeIntegrator::new();

    let claim = Claim {
        text: "Test claim".to_string(),
        encoding: BinaryHV::random(100),
        subject: "test".to_string(),
        predicate: "is".to_string(),
        object: Some("verified".to_string()),
        extraction_confidence: 0.9,
    };

    let verification = Verification {
        claim,
        status: EpistemicStatus::HighConfidence,
        confidence: 0.9,
        evidence: vec![],
        contradictions: vec![],
        hedge_phrase: "".to_string(),
        sources: vec!["https://test.com".to_string()],
        sources_checked: 1,
        sources_supporting: 1,
        sources_refuting: 0,
    };

    let result = ResearchResult {
        query: "Test query".to_string(),
        sources: vec![Source {
            url: "https://test.com".to_string(),
            title: "Test Source".to_string(),
            content: "Full test content here".to_string(),
            published_date: None,
            author: None,
            credibility: 0.8,
            encoding: BinaryHV::random(200),
            fetch_timestamp: SystemTime::now(),
        }],
        verifications: vec![verification],
        confidence: 0.85,
        summary: "Test summary".to_string(),
        new_concepts: vec![],
        time_taken_ms: 100,
    };

    let integration = integrator.integrate(result).await;
    assert!(integration.is_ok(), "Integration should succeed");

    let result = integration.unwrap();
    assert!(result.success, "Integration should be marked successful");
}

// ============================================================================
// PHI INTEGRATION TESTS
// ============================================================================

#[tokio::test]
async fn test_integration_tracks_phi() {
    let mut integrator = KnowledgeIntegrator::new();

    let result = ResearchResult {
        query: "Phi tracking test".to_string(),
        sources: vec![],
        verifications: vec![],
        confidence: 0.5,
        summary: "".to_string(),
        new_concepts: vec![],
        time_taken_ms: 50,
    };

    let integration = integrator.integrate(result).await.unwrap();

    // Should track Φ before and after
    assert!(
        integration.phi_before >= 0.0,
        "Phi before should be non-negative"
    );
    assert!(
        integration.phi_after >= 0.0,
        "Phi after should be non-negative"
    );
}