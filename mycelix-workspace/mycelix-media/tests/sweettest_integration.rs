// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Media - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover publication, fact-checking, curation, attribution, and bridge operations.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists (no pre-built .dna yet - build first)
//! # hc dna pack dna/
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

// --- publication types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Publication {
    pub id: String,
    pub title: String,
    pub content_hash: String,
    pub content_type: ContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: License,
    pub encrypted: bool,
    pub published: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ContentType {
    Article,
    Report,
    Opinion,
    Research,
    News,
    Creative,
    Data,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum License {
    CreativeCommons,
    MIT,
    GPL,
    Proprietary,
    PublicDomain,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PublishInput {
    pub title: String,
    pub content_hash: String,
    pub content_type: ContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: License,
    pub encrypted: bool,
}

// --- factcheck types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FactCheck {
    pub id: String,
    pub publication_id: String,
    pub checker_did: String,
    pub verdict: FactCheckVerdict,
    pub evidence: Vec<EvidenceItem>,
    pub explanation: String,
    pub epistemic_position: EpistemicPosition,
    pub confidence: f64,
    pub checked_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum FactCheckVerdict {
    True,
    False,
    PartiallyTrue,
    Misleading,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvidenceItem {
    pub source_url: String,
    pub description: String,
    pub evidence_type: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EpistemicPosition {
    Empirical,
    Theoretical,
    Consensus,
    Disputed,
    Unknown,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitFactCheckInput {
    pub publication_id: String,
    pub checker_did: String,
    pub verdict: FactCheckVerdict,
    pub evidence: Vec<EvidenceItem>,
    pub explanation: String,
    pub epistemic_position: EpistemicPosition,
    pub confidence: f64,
}

// --- curation types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Endorsement {
    pub id: String,
    pub publication_id: String,
    pub endorser_did: String,
    pub endorsement_type: EndorsementType,
    pub comment: String,
    pub endorsed_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EndorsementType {
    Recommend,
    Approve,
    Highlight,
    CriticalReview,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EndorseInput {
    pub publication_id: String,
    pub endorser_did: String,
    pub endorsement_type: EndorsementType,
    pub comment: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Collection {
    pub id: String,
    pub name: String,
    pub description: String,
    pub curator_did: String,
    pub publication_ids: Vec<String>,
    pub visibility: Visibility,
    pub tags: Vec<String>,
    pub created: Timestamp,
    pub updated: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Unlisted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCollectionInput {
    pub name: String,
    pub description: String,
    pub curator_did: String,
    pub publication_ids: Vec<String>,
    pub visibility: Visibility,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QualityScore {
    pub publication_id: String,
    pub score: f64,
    pub factcheck_score: f64,
    pub endorsement_score: f64,
    pub consistency_score: f64,
}

// --- attribution types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Attribution {
    pub id: String,
    pub publication_id: String,
    pub contributor_did: String,
    pub role: ContributorRole,
    pub description: String,
    pub attributed_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ContributorRole {
    Author,
    Editor,
    Reviewer,
    Translator,
    Illustrator,
    DataProvider,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddAttributionInput {
    pub publication_id: String,
    pub contributor_did: String,
    pub role: ContributorRole,
    pub description: String,
}

// --- bridge types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifyContentInput {
    pub publication_id: String,
    pub source_happ: String,
    pub verification_type: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ContentVerificationResult {
    pub publication_id: String,
    pub is_verified: bool,
    pub factcheck_count: u32,
    pub average_confidence: f64,
    pub epistemic_classification: String,
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Path to the pre-built DNA bundle
/// NOTE: No pre-built .dna exists yet. Run `hc dna pack dna/` first.
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_media.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack dna/' first")
}

/// Decode an entry from a Record into a concrete type via MessagePack deserialization.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

// ============================================================================
// Publication Tests
// ============================================================================

#[cfg(test)]
mod publication_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_publish_and_retrieve() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = PublishInput {
            title: "Test Article".to_string(),
            content_hash: "QmTestHash123".to_string(),
            content_type: ContentType::Article,
            author_did: "did:mycelix:test-author".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec!["test".to_string(), "integration".to_string()],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let record: Record = conductor
            .call(&cell.zome("publication"), "publish", input)
            .await;

        let pub_entry: Publication = decode_entry(&record).expect("Failed to decode publication");

        assert_eq!(pub_entry.title, "Test Article");
        assert_eq!(pub_entry.content_type as u8, ContentType::Article as u8);
        assert_eq!(pub_entry.author_did, "did:mycelix:test-author");
        assert_eq!(pub_entry.version, 1);
        assert!(!pub_entry.encrypted);

        // Retrieve by ID
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("publication"),
                "get_publication",
                pub_entry.id.clone(),
            )
            .await;

        assert!(retrieved.is_some(), "Should retrieve published content");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_search_by_tag() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Publish with specific tag
        let input = PublishInput {
            title: "Tagged Article".to_string(),
            content_hash: "QmTaggedHash".to_string(),
            content_type: ContentType::Research,
            author_did: "did:mycelix:researcher".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec!["climate".to_string(), "energy".to_string()],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let _: Record = conductor
            .call(&cell.zome("publication"), "publish", input)
            .await;

        // Search by tag
        let results: Vec<Record> = conductor
            .call(
                &cell.zome("publication"),
                "search_by_tag",
                "climate".to_string(),
            )
            .await;

        assert!(!results.is_empty(), "Should find publications by tag");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_author_publications() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let author = "did:mycelix:prolific-author".to_string();

        for i in 0..3 {
            let input = PublishInput {
                title: format!("Article {}", i),
                content_hash: format!("QmHash{}", i),
                content_type: ContentType::Article,
                author_did: author.clone(),
                co_authors: vec![],
                language: "en".to_string(),
                tags: vec![],
                license: License::PublicDomain,
                encrypted: false,
            };
            let _: Record = conductor
                .call(&cell.zome("publication"), "publish", input)
                .await;
        }

        let results: Vec<Record> = conductor
            .call(
                &cell.zome("publication"),
                "get_author_publications",
                author.clone(),
            )
            .await;

        assert!(
            results.len() >= 3,
            "Should have at least 3 publications for author"
        );
    }
}

// ============================================================================
// Fact-Check Tests
// ============================================================================

#[cfg(test)]
mod factcheck_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_submit_and_retrieve_fact_check() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Publish first
        let pub_input = PublishInput {
            title: "Claim to Check".to_string(),
            content_hash: "QmClaimHash".to_string(),
            content_type: ContentType::News,
            author_did: "did:mycelix:news-author".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec!["claim".to_string()],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let pub_record: Record = conductor
            .call(&cell.zome("publication"), "publish", pub_input)
            .await;
        let publication: Publication = decode_entry(&pub_record).expect("decode pub");

        // Submit fact check
        let fc_input = SubmitFactCheckInput {
            publication_id: publication.id.clone(),
            checker_did: "did:mycelix:factchecker".to_string(),
            verdict: FactCheckVerdict::PartiallyTrue,
            evidence: vec![EvidenceItem {
                source_url: "https://example.com/source".to_string(),
                description: "Primary source confirms partially".to_string(),
                evidence_type: "document".to_string(),
            }],
            explanation: "The core claim is supported but context is missing".to_string(),
            epistemic_position: EpistemicPosition::Empirical,
            confidence: 0.75,
        };

        let fc_record: Record = conductor
            .call(&cell.zome("factcheck"), "submit_fact_check", fc_input)
            .await;

        let fact_check: FactCheck = decode_entry(&fc_record).expect("decode factcheck");
        assert_eq!(fact_check.publication_id, publication.id);
        assert!(fact_check.confidence >= 0.0 && fact_check.confidence <= 1.0);

        // Get fact checks for publication
        let checks: Vec<Record> = conductor
            .call(
                &cell.zome("factcheck"),
                "get_publication_fact_checks",
                publication.id.clone(),
            )
            .await;

        assert!(
            !checks.is_empty(),
            "Should have fact checks for publication"
        );
    }
}

// ============================================================================
// Curation Tests
// ============================================================================

#[cfg(test)]
mod curation_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_endorse_publication() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Publish
        let pub_input = PublishInput {
            title: "Endorsable Article".to_string(),
            content_hash: "QmEndorseHash".to_string(),
            content_type: ContentType::Research,
            author_did: "did:mycelix:researcher".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec![],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let pub_record: Record = conductor
            .call(&cell.zome("publication"), "publish", pub_input)
            .await;
        let publication: Publication = decode_entry(&pub_record).expect("decode pub");

        // Endorse
        let endorse_input = EndorseInput {
            publication_id: publication.id.clone(),
            endorser_did: "did:mycelix:endorser".to_string(),
            endorsement_type: EndorsementType::Recommend,
            comment: "Excellent research methodology".to_string(),
        };

        let endorse_record: Record = conductor
            .call(&cell.zome("curation"), "endorse", endorse_input)
            .await;

        let endorsement: Endorsement = decode_entry(&endorse_record).expect("decode endorsement");
        assert_eq!(endorsement.publication_id, publication.id);

        // Get endorsements
        let endorsements: Vec<Record> = conductor
            .call(
                &cell.zome("curation"),
                "get_publication_endorsements",
                publication.id.clone(),
            )
            .await;

        assert!(!endorsements.is_empty(), "Should have endorsements");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_collection() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let input = CreateCollectionInput {
            name: "Best of 2026".to_string(),
            description: "Top articles of the year".to_string(),
            curator_did: "did:mycelix:curator".to_string(),
            publication_ids: vec!["pub:1".to_string(), "pub:2".to_string()],
            visibility: Visibility::Public,
            tags: vec!["best-of".to_string()],
        };

        let record: Record = conductor
            .call(&cell.zome("curation"), "create_collection", input)
            .await;

        let collection: Collection = decode_entry(&record).expect("decode collection");
        assert_eq!(collection.name, "Best of 2026");
        assert_eq!(collection.publication_ids.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_quality_score_calculation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Publish
        let pub_input = PublishInput {
            title: "Quality Test Article".to_string(),
            content_hash: "QmQualityHash".to_string(),
            content_type: ContentType::Research,
            author_did: "did:mycelix:author".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec![],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let pub_record: Record = conductor
            .call(&cell.zome("publication"), "publish", pub_input)
            .await;
        let publication: Publication = decode_entry(&pub_record).expect("decode pub");

        let score: QualityScore = conductor
            .call(
                &cell.zome("curation"),
                "calculate_quality_score",
                publication.id.clone(),
            )
            .await;

        assert_eq!(score.publication_id, publication.id);
        assert!(
            score.score >= 0.0 && score.score <= 1.0,
            "Score must be 0-1"
        );
    }
}

// ============================================================================
// Attribution Tests
// ============================================================================

#[cfg(test)]
mod attribution_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_add_and_get_attribution() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Publish
        let pub_input = PublishInput {
            title: "Collaborative Work".to_string(),
            content_hash: "QmCollabHash".to_string(),
            content_type: ContentType::Research,
            author_did: "did:mycelix:lead-author".to_string(),
            co_authors: vec!["did:mycelix:co-author".to_string()],
            language: "en".to_string(),
            tags: vec![],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let pub_record: Record = conductor
            .call(&cell.zome("publication"), "publish", pub_input)
            .await;
        let publication: Publication = decode_entry(&pub_record).expect("decode pub");

        // Add attribution
        let attr_input = AddAttributionInput {
            publication_id: publication.id.clone(),
            contributor_did: "did:mycelix:data-provider".to_string(),
            role: ContributorRole::DataProvider,
            description: "Provided raw dataset for analysis".to_string(),
        };

        let attr_record: Record = conductor
            .call(&cell.zome("attribution"), "add_attribution", attr_input)
            .await;

        let attribution: Attribution = decode_entry(&attr_record).expect("decode attribution");
        assert_eq!(attribution.publication_id, publication.id);

        // Get attributions
        let attributions: Vec<Record> = conductor
            .call(
                &cell.zome("attribution"),
                "get_publication_attributions",
                publication.id.clone(),
            )
            .await;

        assert!(!attributions.is_empty(), "Should have attributions");
    }
}

// ============================================================================
// Bridge Tests
// ============================================================================

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_verify_content() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Publish first
        let pub_input = PublishInput {
            title: "Verifiable Content".to_string(),
            content_hash: "QmVerifyHash".to_string(),
            content_type: ContentType::Article,
            author_did: "did:mycelix:author".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec![],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let pub_record: Record = conductor
            .call(&cell.zome("publication"), "publish", pub_input)
            .await;
        let publication: Publication = decode_entry(&pub_record).expect("decode pub");

        // Verify via bridge
        let verify_input = VerifyContentInput {
            publication_id: publication.id.clone(),
            source_happ: "governance".to_string(),
            verification_type: "factcheck_status".to_string(),
        };

        let result: ContentVerificationResult = conductor
            .call(&cell.zome("media_bridge"), "verify_content", verify_input)
            .await;

        assert_eq!(result.publication_id, publication.id);
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_publish_factcheck_endorse_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Publish
        let pub_input = PublishInput {
            title: "Full Lifecycle Article".to_string(),
            content_hash: "QmLifecycleHash".to_string(),
            content_type: ContentType::Research,
            author_did: "did:mycelix:researcher".to_string(),
            co_authors: vec![],
            language: "en".to_string(),
            tags: vec!["lifecycle".to_string()],
            license: License::CreativeCommons,
            encrypted: false,
        };

        let pub_record: Record = conductor
            .call(&cell.zome("publication"), "publish", pub_input)
            .await;
        let publication: Publication = decode_entry(&pub_record).expect("decode pub");

        // 2. Add attribution
        let attr_input = AddAttributionInput {
            publication_id: publication.id.clone(),
            contributor_did: "did:mycelix:editor".to_string(),
            role: ContributorRole::Editor,
            description: "Edited and reviewed".to_string(),
        };

        let _: Record = conductor
            .call(&cell.zome("attribution"), "add_attribution", attr_input)
            .await;

        // 3. Fact-check
        let fc_input = SubmitFactCheckInput {
            publication_id: publication.id.clone(),
            checker_did: "did:mycelix:checker".to_string(),
            verdict: FactCheckVerdict::True,
            evidence: vec![EvidenceItem {
                source_url: "https://example.com/proof".to_string(),
                description: "Confirmed by primary source".to_string(),
                evidence_type: "document".to_string(),
            }],
            explanation: "All claims verified".to_string(),
            epistemic_position: EpistemicPosition::Empirical,
            confidence: 0.95,
        };

        let _: Record = conductor
            .call(&cell.zome("factcheck"), "submit_fact_check", fc_input)
            .await;

        // 4. Endorse
        let endorse_input = EndorseInput {
            publication_id: publication.id.clone(),
            endorser_did: "did:mycelix:expert".to_string(),
            endorsement_type: EndorsementType::Approve,
            comment: "Well-researched and verified".to_string(),
        };

        let _: Record = conductor
            .call(&cell.zome("curation"), "endorse", endorse_input)
            .await;

        // 5. Check quality score
        let score: QualityScore = conductor
            .call(
                &cell.zome("curation"),
                "calculate_quality_score",
                publication.id.clone(),
            )
            .await;

        assert!(
            score.score > 0.0,
            "Quality score should be positive after fact-check and endorsement"
        );
    }
}
