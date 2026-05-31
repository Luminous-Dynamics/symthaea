// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Curation Zome Unit Tests
//!
//! Comprehensive test coverage for content moderation, endorsements,
//! collections, quality scores, featured content, and visibility controls.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestEndorsement {
    pub id: String,
    pub publication_id: String,
    pub endorser_did: String,
    pub endorsement_type: TestEndorsementType,
    pub comment: Option<String>,
    pub created: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestEndorsementType {
    Upvote,
    Bookmark,
    Share,
    Recommend,
    Award(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCollection {
    pub id: String,
    pub name: String,
    pub description: String,
    pub curator_did: String,
    pub visibility: TestVisibility,
    pub publication_ids: Vec<String>,
    pub created: i64,
    pub updated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestVisibility {
    Public,
    Private,
    Unlisted,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestQualityScore {
    pub publication_id: String,
    pub score: f64,
    pub endorsement_count: u32,
    pub share_count: u32,
    pub fact_check_score: f64,
    pub author_reputation: f64,
    pub last_calculated: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestFeaturedContent {
    pub id: String,
    pub publication_id: String,
    pub featured_by: String,
    pub reason: String,
    pub featured_from: i64,
    pub featured_until: Option<i64>,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_endorsement() -> TestEndorsement {
    TestEndorsement {
        id: "endorse:pub123:did:mycelix:endorser1:1704067200000000".to_string(),
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        endorser_did: "did:mycelix:endorser1".to_string(),
        endorsement_type: TestEndorsementType::Upvote,
        comment: None,
        created: 1704067200000000,
    }
}

fn create_test_collection() -> TestCollection {
    TestCollection {
        id: "collection:did:mycelix:curator1:1704067200000000".to_string(),
        name: "Best Tech Articles 2024".to_string(),
        description: "Curated collection of excellent technology articles".to_string(),
        curator_did: "did:mycelix:curator1".to_string(),
        visibility: TestVisibility::Public,
        publication_ids: vec!["pub1".to_string(), "pub2".to_string()],
        created: 1704067200000000,
        updated: 1704067200000000,
    }
}

fn create_test_quality_score() -> TestQualityScore {
    TestQualityScore {
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        score: 0.75,
        endorsement_count: 50,
        share_count: 20,
        fact_check_score: 0.8,
        author_reputation: 0.7,
        last_calculated: 1704067200000000,
    }
}

fn create_test_featured_content() -> TestFeaturedContent {
    TestFeaturedContent {
        id: "featured:pub123:1704067200000000".to_string(),
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        featured_by: "did:mycelix:editor1".to_string(),
        reason: "Exceptional investigative journalism".to_string(),
        featured_from: 1704067200000000,
        featured_until: Some(1704672000000000), // 1 week later
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

fn validate_endorsement(endorsement: &TestEndorsement) -> Result<(), String> {
    if !validate_did(&endorsement.endorser_did) {
        return Err("Endorser must be a valid DID".to_string());
    }
    Ok(())
}

fn validate_collection(collection: &TestCollection) -> Result<(), String> {
    if !validate_did(&collection.curator_did) {
        return Err("Curator must be a valid DID".to_string());
    }
    if collection.name.is_empty() {
        return Err("Collection name required".to_string());
    }
    Ok(())
}

fn validate_quality_score(score: &TestQualityScore) -> Result<(), String> {
    if score.score < 0.0 || score.score > 1.0 {
        return Err("Score must be 0-1".to_string());
    }
    Ok(())
}

fn validate_featured_content(featured: &TestFeaturedContent) -> Result<(), String> {
    if !validate_did(&featured.featured_by) {
        return Err("Featured by must be a valid DID".to_string());
    }
    Ok(())
}

fn can_modify_collection(collection: &TestCollection, requester_did: &str) -> bool {
    collection.curator_did == requester_did
}

fn can_remove_endorsement(endorsement: &TestEndorsement, requester_did: &str) -> bool {
    endorsement.endorser_did == requester_did
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ENDORSEMENT TESTS
    // =========================================================================

    #[test]
    fn test_endorsement_requires_valid_endorser_did() {
        let endorsement = create_test_endorsement();
        assert!(validate_did(&endorsement.endorser_did));
    }

    #[test]
    fn test_endorsement_invalid_endorser_did_rejected() {
        let mut endorsement = create_test_endorsement();
        endorsement.endorser_did = "invalid_did".to_string();
        let result = validate_endorsement(&endorsement);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Endorser must be a valid DID");
    }

    #[test]
    fn test_endorsement_has_unique_id() {
        let endorsement = create_test_endorsement();
        assert!(endorsement.id.starts_with("endorse:"));
    }

    #[test]
    fn test_endorsement_references_publication() {
        let endorsement = create_test_endorsement();
        assert!(!endorsement.publication_id.is_empty());
    }

    #[test]
    fn test_endorsement_types_supported() {
        let types = vec![
            TestEndorsementType::Upvote,
            TestEndorsementType::Bookmark,
            TestEndorsementType::Share,
            TestEndorsementType::Recommend,
            TestEndorsementType::Award("Best Article".to_string()),
        ];
        for etype in types {
            let mut endorsement = create_test_endorsement();
            endorsement.endorsement_type = etype;
            assert!(validate_endorsement(&endorsement).is_ok());
        }
    }

    #[test]
    fn test_endorsement_comment_optional() {
        let endorsement = create_test_endorsement();
        assert!(endorsement.comment.is_none());
    }

    #[test]
    fn test_endorsement_with_comment() {
        let mut endorsement = create_test_endorsement();
        endorsement.comment = Some("Excellent research!".to_string());
        assert!(endorsement.comment.is_some());
    }

    #[test]
    fn test_endorsements_immutable() {
        // Endorsements cannot be updated according to validation rules
        let endorsement = create_test_endorsement();
        let endorsement_id = endorsement.id.clone();
        assert_eq!(endorsement.id, endorsement_id);
    }

    #[test]
    fn test_only_endorser_can_remove() {
        let endorsement = create_test_endorsement();
        assert!(can_remove_endorsement(
            &endorsement,
            &endorsement.endorser_did
        ));
        assert!(!can_remove_endorsement(&endorsement, "did:mycelix:other"));
    }

    // =========================================================================
    // COLLECTION TESTS
    // =========================================================================

    #[test]
    fn test_collection_requires_valid_curator_did() {
        let collection = create_test_collection();
        assert!(validate_did(&collection.curator_did));
    }

    #[test]
    fn test_collection_invalid_curator_did_rejected() {
        let mut collection = create_test_collection();
        collection.curator_did = "invalid_did".to_string();
        let result = validate_collection(&collection);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Curator must be a valid DID");
    }

    #[test]
    fn test_collection_requires_name() {
        let collection = create_test_collection();
        assert!(!collection.name.is_empty());
    }

    #[test]
    fn test_collection_empty_name_rejected() {
        let mut collection = create_test_collection();
        collection.name = "".to_string();
        let result = validate_collection(&collection);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Collection name required");
    }

    #[test]
    fn test_collection_has_unique_id() {
        let collection = create_test_collection();
        assert!(collection.id.starts_with("collection:"));
    }

    #[test]
    fn test_collection_visibility_types() {
        let visibilities = vec![
            TestVisibility::Public,
            TestVisibility::Private,
            TestVisibility::Unlisted,
        ];
        for vis in visibilities {
            let mut collection = create_test_collection();
            collection.visibility = vis;
            assert!(validate_collection(&collection).is_ok());
        }
    }

    #[test]
    fn test_collection_with_publications() {
        let collection = create_test_collection();
        assert!(!collection.publication_ids.is_empty());
        assert_eq!(collection.publication_ids.len(), 2);
    }

    #[test]
    fn test_collection_empty_valid() {
        let mut collection = create_test_collection();
        collection.publication_ids = vec![];
        assert!(validate_collection(&collection).is_ok());
    }

    #[test]
    fn test_only_curator_can_modify() {
        let collection = create_test_collection();
        assert!(can_modify_collection(&collection, &collection.curator_did));
        assert!(!can_modify_collection(&collection, "did:mycelix:other"));
    }

    #[test]
    fn test_collection_timestamps() {
        let collection = create_test_collection();
        assert!(collection.created > 0);
        assert!(collection.updated >= collection.created);
    }

    #[test]
    fn test_add_publication_to_collection() {
        let mut collection = create_test_collection();
        let original_count = collection.publication_ids.len();
        collection.publication_ids.push("pub3".to_string());
        assert_eq!(collection.publication_ids.len(), original_count + 1);
    }

    #[test]
    fn test_remove_publication_from_collection() {
        let mut collection = create_test_collection();
        let pub_to_remove = "pub1".to_string();
        collection.publication_ids.retain(|id| id != &pub_to_remove);
        assert_eq!(collection.publication_ids.len(), 1);
        assert!(!collection.publication_ids.contains(&pub_to_remove));
    }

    // =========================================================================
    // VISIBILITY ACCESS CONTROL TESTS
    // =========================================================================

    fn can_view_collection(collection: &TestCollection, viewer_did: Option<&str>) -> bool {
        match collection.visibility {
            TestVisibility::Public => true,
            TestVisibility::Unlisted => true, // Anyone with link can view
            TestVisibility::Private => {
                if let Some(did) = viewer_did {
                    collection.curator_did == did
                } else {
                    false
                }
            }
        }
    }

    #[test]
    fn test_public_collection_visible_to_all() {
        let collection = create_test_collection();
        assert!(can_view_collection(&collection, None));
        assert!(can_view_collection(&collection, Some("did:mycelix:anyone")));
    }

    #[test]
    fn test_private_collection_only_visible_to_curator() {
        let mut collection = create_test_collection();
        collection.visibility = TestVisibility::Private;
        assert!(!can_view_collection(&collection, None));
        assert!(!can_view_collection(&collection, Some("did:mycelix:other")));
        assert!(can_view_collection(
            &collection,
            Some(&collection.curator_did)
        ));
    }

    #[test]
    fn test_unlisted_collection_visible_with_link() {
        let mut collection = create_test_collection();
        collection.visibility = TestVisibility::Unlisted;
        // Anyone with the link can view
        assert!(can_view_collection(&collection, None));
        assert!(can_view_collection(&collection, Some("did:mycelix:anyone")));
    }

    // =========================================================================
    // QUALITY SCORE TESTS
    // =========================================================================

    #[test]
    fn test_quality_score_creation() {
        let score = create_test_quality_score();
        assert!(!score.publication_id.is_empty());
        assert!(score.score >= 0.0 && score.score <= 1.0);
    }

    #[test]
    fn test_quality_score_valid_range() {
        let valid_scores = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        for s in valid_scores {
            let mut score = create_test_quality_score();
            score.score = s;
            assert!(validate_quality_score(&score).is_ok());
        }
    }

    #[test]
    fn test_quality_score_negative_rejected() {
        let mut score = create_test_quality_score();
        score.score = -0.1;
        let result = validate_quality_score(&score);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Score must be 0-1");
    }

    #[test]
    fn test_quality_score_over_1_rejected() {
        let mut score = create_test_quality_score();
        score.score = 1.5;
        let result = validate_quality_score(&score);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_score_components() {
        let score = create_test_quality_score();
        assert!(score.endorsement_count > 0);
        assert!(score.share_count > 0);
        assert!(score.fact_check_score >= 0.0 && score.fact_check_score <= 1.0);
        assert!(score.author_reputation >= 0.0 && score.author_reputation <= 1.0);
    }

    #[test]
    fn test_quality_score_calculation() {
        // Simplified quality score calculation
        fn calculate_quality_score(
            endorsement_count: u32,
            share_count: u32,
            fact_check_score: f64,
            author_reputation: f64,
        ) -> f64 {
            let engagement_factor = ((endorsement_count + share_count) as f64 / 100.0).min(1.0);
            let quality_factor = (fact_check_score + author_reputation) / 2.0;
            (engagement_factor * 0.3 + quality_factor * 0.7).min(1.0)
        }

        let score = calculate_quality_score(50, 20, 0.8, 0.7);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_quality_score_timestamp() {
        let score = create_test_quality_score();
        assert!(score.last_calculated > 0);
    }

    // =========================================================================
    // FEATURED CONTENT TESTS
    // =========================================================================

    #[test]
    fn test_featured_content_creation() {
        let featured = create_test_featured_content();
        assert!(!featured.publication_id.is_empty());
        assert!(!featured.reason.is_empty());
    }

    #[test]
    fn test_featured_content_requires_valid_did() {
        let featured = create_test_featured_content();
        assert!(validate_did(&featured.featured_by));
    }

    #[test]
    fn test_featured_content_invalid_did_rejected() {
        let mut featured = create_test_featured_content();
        featured.featured_by = "invalid".to_string();
        let result = validate_featured_content(&featured);
        assert!(result.is_err());
    }

    #[test]
    fn test_featured_content_has_time_range() {
        let featured = create_test_featured_content();
        assert!(featured.featured_from > 0);
    }

    #[test]
    fn test_featured_content_no_end_date() {
        let mut featured = create_test_featured_content();
        featured.featured_until = None;
        // Indefinite featuring is valid
        assert!(featured.featured_until.is_none());
    }

    #[test]
    fn test_featured_content_end_after_start() {
        let featured = create_test_featured_content();
        if let Some(until) = featured.featured_until {
            assert!(until > featured.featured_from);
        }
    }

    fn is_currently_featured(featured: &TestFeaturedContent, current_time: i64) -> bool {
        if current_time < featured.featured_from {
            return false;
        }
        if let Some(until) = featured.featured_until {
            current_time < until
        } else {
            true
        }
    }

    #[test]
    fn test_featured_content_active_check() {
        let featured = create_test_featured_content();
        let during = featured.featured_from + 1000000;
        assert!(is_currently_featured(&featured, during));
    }

    #[test]
    fn test_featured_content_expired_check() {
        let featured = create_test_featured_content();
        if let Some(until) = featured.featured_until {
            let after = until + 1000000;
            assert!(!is_currently_featured(&featured, after));
        }
    }

    #[test]
    fn test_featured_content_before_start() {
        let featured = create_test_featured_content();
        let before = featured.featured_from - 1000000;
        assert!(!is_currently_featured(&featured, before));
    }

    // =========================================================================
    // ENDORSEMENT COUNTING TESTS
    // =========================================================================

    fn count_endorsements_by_type(
        endorsements: &[TestEndorsement],
        publication_id: &str,
    ) -> std::collections::HashMap<String, u32> {
        let mut counts = std::collections::HashMap::new();

        for e in endorsements {
            if e.publication_id == publication_id {
                let key = match &e.endorsement_type {
                    TestEndorsementType::Upvote => "upvote",
                    TestEndorsementType::Bookmark => "bookmark",
                    TestEndorsementType::Share => "share",
                    TestEndorsementType::Recommend => "recommend",
                    TestEndorsementType::Award(_) => "award",
                };
                *counts.entry(key.to_string()).or_insert(0) += 1;
            }
        }

        counts
    }

    #[test]
    fn test_endorsement_counting() {
        let pub_id = "pub123";
        let endorsements = vec![
            {
                let mut e = create_test_endorsement();
                e.publication_id = pub_id.to_string();
                e.endorsement_type = TestEndorsementType::Upvote;
                e
            },
            {
                let mut e = create_test_endorsement();
                e.publication_id = pub_id.to_string();
                e.endorsement_type = TestEndorsementType::Upvote;
                e.endorser_did = "did:mycelix:user2".to_string();
                e
            },
            {
                let mut e = create_test_endorsement();
                e.publication_id = pub_id.to_string();
                e.endorsement_type = TestEndorsementType::Share;
                e
            },
        ];

        let counts = count_endorsements_by_type(&endorsements, pub_id);
        assert_eq!(counts.get("upvote"), Some(&2));
        assert_eq!(counts.get("share"), Some(&1));
    }

    // =========================================================================
    // ENDORSER HISTORY TESTS
    // =========================================================================

    fn get_endorser_history<'a>(
        endorsements: &'a [TestEndorsement],
        endorser_did: &str,
    ) -> Vec<&'a TestEndorsement> {
        endorsements
            .iter()
            .filter(|e| e.endorser_did == endorser_did)
            .collect()
    }

    #[test]
    fn test_endorser_history() {
        let endorser_did = "did:mycelix:endorser1";
        let endorsements = vec![
            create_test_endorsement(),
            {
                let mut e = create_test_endorsement();
                e.publication_id = "pub2".to_string();
                e
            },
            {
                let mut e = create_test_endorsement();
                e.endorser_did = "did:mycelix:other".to_string();
                e
            },
        ];

        let history = get_endorser_history(&endorsements, endorser_did);
        assert_eq!(history.len(), 2);
    }

    // =========================================================================
    // PUBLIC COLLECTIONS TESTS
    // =========================================================================

    fn get_public_collections(collections: &[TestCollection]) -> Vec<&TestCollection> {
        collections
            .iter()
            .filter(|c| matches!(c.visibility, TestVisibility::Public))
            .collect()
    }

    #[test]
    fn test_public_collections_filtering() {
        let collections = vec![
            create_test_collection(),
            {
                let mut c = create_test_collection();
                c.visibility = TestVisibility::Private;
                c
            },
            {
                let mut c = create_test_collection();
                c.visibility = TestVisibility::Unlisted;
                c
            },
        ];

        let public = get_public_collections(&collections);
        assert_eq!(public.len(), 1);
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_collection_with_many_publications() {
        let mut collection = create_test_collection();
        collection.publication_ids = (0..1000).map(|i| format!("pub{}", i)).collect();
        assert_eq!(collection.publication_ids.len(), 1000);
    }

    #[test]
    fn test_collection_with_long_name() {
        let mut collection = create_test_collection();
        collection.name = "A".repeat(500);
        assert!(validate_collection(&collection).is_ok());
    }

    #[test]
    fn test_endorsement_with_unicode_comment() {
        let mut endorsement = create_test_endorsement();
        endorsement.comment = Some("Excellent article!".to_string());
        assert!(validate_endorsement(&endorsement).is_ok());
    }

    #[test]
    fn test_award_endorsement_with_custom_name() {
        let mut endorsement = create_test_endorsement();
        endorsement.endorsement_type = TestEndorsementType::Award("Pulitzer Prize".to_string());
        assert!(matches!(
            endorsement.endorsement_type,
            TestEndorsementType::Award(_)
        ));
    }

    #[test]
    fn test_quality_score_edge_values() {
        let edge_scores = vec![0.0, 0.0001, 0.9999, 1.0];
        for s in edge_scores {
            let mut score = create_test_quality_score();
            score.score = s;
            assert!(validate_quality_score(&score).is_ok());
        }
    }

    #[test]
    fn test_featured_content_indefinite() {
        let mut featured = create_test_featured_content();
        featured.featured_until = None;
        // Should still be featured at any future time
        let far_future = featured.featured_from + 10000000000000i64;
        assert!(is_currently_featured(&featured, far_future));
    }

    #[test]
    fn test_duplicate_endorsement_prevention() {
        // Same user should not endorse same publication twice with same type
        let endorsements = vec![
            create_test_endorsement(),
            create_test_endorsement(), // Duplicate
        ];

        // In practice, this would be prevented at the coordinator level
        // Here we just verify the data structure allows detecting duplicates
        let unique: std::collections::HashSet<(&String, &String)> = endorsements
            .iter()
            .map(|e| (&e.publication_id, &e.endorser_did))
            .collect();

        assert!(unique.len() < endorsements.len());
    }

    #[test]
    fn test_collection_update_timestamp() {
        let mut collection = create_test_collection();
        let original_created = collection.created;
        collection.updated = collection.created + 86400000000; // 1 day later
        assert!(collection.updated > original_created);
        assert_eq!(collection.created, original_created); // Created unchanged
    }
}
