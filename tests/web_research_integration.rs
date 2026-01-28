//! Integration tests for the web_research module.
//!
//! Tests type construction, the ClaimConfidence → EpistemicCube bridge,
//! SourceClassifier domain mapping, and KnowledgeIntegrator builder.
//!
//! Run with:
//!   cargo test --test web_research_integration

use symthaea::web_research::{
    Source, Claim, VerificationLevel, ResearchResult,
    ClaimConfidence, KnowledgeIntegrator, SourceClassifier,
    ContentExtractor, WebResearcher, MockSearchBackend, SearchHit,
};
use symthaea::mind::structured_thought::{
    EpistemicCube, ETier, NTier, MTier,
};
use symthaea::hdc::binary_hv::HV16;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// Type construction — ensure all expected fields exist
// ---------------------------------------------------------------------------

#[test]
fn test_source_construction() {
    let source = Source {
        url: "https://en.wikipedia.org/wiki/Rust_(programming_language)".to_string(),
        title: "Rust (programming language)".to_string(),
        content: "Rust is a multi-paradigm programming language.".to_string(),
        published_date: Some(SystemTime::now()),
        author: Some("Wikipedia contributors".to_string()),
        credibility: 0.8,
        encoding: HV16::zero(),
        fetch_timestamp: SystemTime::now(),
    };

    assert_eq!(source.credibility, 0.8);
    assert!(source.published_date.is_some());
    assert!(source.author.is_some());
}

#[test]
fn test_claim_construction() {
    let claim = Claim {
        text: "Rust prevents data races at compile time".to_string(),
        encoding: HV16::zero(),
        subject: "Rust".to_string(),
        predicate: "prevents".to_string(),
        object: Some("data races".to_string()),
        extraction_confidence: 0.9,
    };

    assert_eq!(claim.subject, "Rust");
    assert!(claim.object.is_some());
    assert!(claim.extraction_confidence > 0.5);
}

#[test]
fn test_verification_level_variants() {
    let levels = [
        VerificationLevel::Minimal,
        VerificationLevel::Standard,
        VerificationLevel::Rigorous,
        VerificationLevel::Academic,
    ];
    // All variants should be distinct
    for (i, a) in levels.iter().enumerate() {
        for (j, b) in levels.iter().enumerate() {
            if i == j {
                assert_eq!(a, b);
            } else {
                assert_ne!(a, b);
            }
        }
    }
}

#[test]
fn test_claim_confidence_contested_and_unverifiable() {
    // These variants must exist (resolved naming conflict)
    let contested = ClaimConfidence::Contested;
    let unverifiable = ClaimConfidence::Unverifiable;
    assert_ne!(contested, unverifiable);
    assert_ne!(contested, ClaimConfidence::HighConfidence);
}

#[test]
fn test_research_result_construction() {
    let result = ResearchResult {
        query: "What is Rust?".to_string(),
        sources: vec![],
        verifications: vec![],
        confidence: 0.75,
        summary: "Rust is a systems programming language.".to_string(),
        new_concepts: vec!["borrow checker".to_string()],
        time_taken_ms: 42,
    };

    assert_eq!(result.query, "What is Rust?");
    assert!(result.sources.is_empty());
    assert_eq!(result.new_concepts.len(), 1);
}

// ---------------------------------------------------------------------------
// ClaimConfidence variants
// ---------------------------------------------------------------------------

#[test]
fn test_all_claim_confidence_variants() {
    let variants = [
        ClaimConfidence::HighConfidence,
        ClaimConfidence::ModerateConfidence,
        ClaimConfidence::LowConfidence,
        ClaimConfidence::Contested,
        ClaimConfidence::Unverifiable,
        ClaimConfidence::False,
    ];
    assert_eq!(variants.len(), 6);
}

// ---------------------------------------------------------------------------
// EpistemicCube — 3D epistemic model
// ---------------------------------------------------------------------------

#[test]
fn test_epistemic_cube_credibility_scoring() {
    // Academic: (E3, N2, M2) → high credibility
    let academic = EpistemicCube::new(ETier::E3, NTier::N2, MTier::M2);
    assert!(academic.credibility_score() > 0.7);

    // Government: (E3, N3, M3) → highest
    let gov = EpistemicCube::new(ETier::E3, NTier::N3, MTier::M3);
    assert!(gov.credibility_score() > academic.credibility_score());

    // Blog: (E1, N0, M0) → low
    let blog = EpistemicCube::default();
    assert!(blog.credibility_score() < 0.3);
}

#[test]
fn test_epistemic_cube_default_is_low() {
    let cube = EpistemicCube::default();
    assert_eq!(cube.empirical, ETier::E1);
    assert_eq!(cube.normative, NTier::N0);
    assert_eq!(cube.meta, MTier::M0);
}

// ---------------------------------------------------------------------------
// SourceClassifier — domain → EpistemicCube
// ---------------------------------------------------------------------------

#[test]
fn test_source_classifier_academic_domains() {
    let classifier = SourceClassifier::new();

    let arxiv = classifier.classify("https://arxiv.org/abs/2301.00001");
    assert_eq!(arxiv.empirical, ETier::E3);

    let nature = classifier.classify("https://nature.com/articles/s41586");
    assert_eq!(nature.empirical, ETier::E3);
}

#[test]
fn test_source_classifier_gov_tld() {
    let classifier = SourceClassifier::new();
    let cube = classifier.classify("https://data.census.gov/stats");
    assert_eq!(cube.normative, NTier::N3);
}

#[test]
fn test_source_classifier_edu_tld() {
    let classifier = SourceClassifier::new();
    let cube = classifier.classify("https://cs.stanford.edu/paper.pdf");
    assert_eq!(cube.empirical, ETier::E3);
}

#[test]
fn test_source_classifier_wikipedia() {
    let classifier = SourceClassifier::new();
    let cube = classifier.classify("https://en.wikipedia.org/wiki/Consciousness");
    assert_eq!(cube.empirical, ETier::E2);
    assert_eq!(cube.meta, MTier::M2);
}

#[test]
fn test_source_classifier_unknown_falls_back() {
    let classifier = SourceClassifier::new();
    let cube = classifier.classify("https://random-obscure-site.xyz/page");
    assert_eq!(cube.empirical, ETier::E1);
    assert_eq!(cube.normative, NTier::N0);
    assert_eq!(cube.meta, MTier::M0);
}

#[test]
fn test_source_classifier_credibility_score() {
    let classifier = SourceClassifier::new();
    let cube = classifier.classify("https://arxiv.org/abs/123");
    let score = classifier.credibility_score(&cube);
    assert!(score > 0.7);
    assert!(score <= 1.0);
}

// ---------------------------------------------------------------------------
// KnowledgeIntegrator — builder pattern
// ---------------------------------------------------------------------------

#[test]
fn test_knowledge_integrator_with_min_confidence() {
    let integrator = KnowledgeIntegrator::new()
        .with_min_confidence(0.8);

    // Should compile and construct successfully
    // The integrator is ready for use with the configured threshold
    let _ = integrator;
}

#[test]
fn test_knowledge_integrator_default_confidence() {
    let integrator = KnowledgeIntegrator::new();
    // Default min_confidence is 0.4, but we verify construction works
    let _ = integrator;
}

// ---------------------------------------------------------------------------
// ContentExtractor — basic construction
// ---------------------------------------------------------------------------

#[test]
fn test_content_extractor_construction() {
    let extractor = ContentExtractor::new();
    let _ = extractor;
}

// ---------------------------------------------------------------------------
// MockSearchBackend — offline research pipeline
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mock_backend_research_pipeline() {
    // Build a researcher with a mock backend returning a Wikipedia hit
    let researcher = WebResearcher::new()
        .unwrap()
        .with_backend(Box::new(MockSearchBackend::wikipedia_stub()));

    let result = researcher.research("What is consciousness?").await.unwrap();

    assert_eq!(result.query, "What is consciousness?");
    assert_eq!(result.sources.len(), 1);

    let source = &result.sources[0];
    assert!(source.url.contains("wikipedia.org"));
    assert!(source.credibility > 0.0);

    // The classifier should have tagged Wikipedia as (E2, N1, M2)
    let cube = researcher.classifier().classify(&source.url);
    assert_eq!(cube.empirical, ETier::E2);
    assert_eq!(cube.meta, MTier::M2);
}

#[tokio::test]
async fn test_mock_backend_custom_hits() {
    let hits = vec![
        SearchHit {
            title: "ArXiv Paper".to_string(),
            url: "https://arxiv.org/abs/2301.99999".to_string(),
            snippet: "A novel approach to consciousness measurement.".to_string(),
        },
        SearchHit {
            title: "Government Data".to_string(),
            url: "https://data.census.gov/stats".to_string(),
            snippet: "Statistical data on AI adoption.".to_string(),
        },
    ];

    let researcher = WebResearcher::new()
        .unwrap()
        .with_backend(Box::new(MockSearchBackend::new(hits)));

    let result = researcher.research("AI consciousness").await.unwrap();

    assert_eq!(result.sources.len(), 2);

    // First source (arxiv) should have high credibility
    let arxiv_cred = result.sources[0].credibility;
    assert!(arxiv_cred > 0.7, "arxiv credibility was {}", arxiv_cred);

    // Second source (gov) should also have high credibility
    let gov_cred = result.sources[1].credibility;
    assert!(gov_cred > 0.7, "gov credibility was {}", gov_cred);
}

#[tokio::test]
async fn test_mock_backend_empty_results() {
    let researcher = WebResearcher::new()
        .unwrap()
        .with_backend(Box::new(MockSearchBackend::new(vec![])));

    let result = researcher.research("obscure query").await.unwrap();

    assert!(result.sources.is_empty());
    assert!(result.summary.contains("No results found"));
}

#[tokio::test]
async fn test_research_result_summary_from_best_source() {
    let researcher = WebResearcher::new()
        .unwrap()
        .with_backend(Box::new(MockSearchBackend::wikipedia_stub()));

    let result = researcher.research("test query").await.unwrap();

    // Summary should come from the best source's content
    assert!(!result.summary.is_empty());
    assert!(result.time_taken_ms < 1000); // mock should be fast
}
