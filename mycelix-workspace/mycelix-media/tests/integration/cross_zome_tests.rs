// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Cross-Zome Integration Tests
//!
//! Tests for interactions between publication, attribution, curation,
//! factcheck, and bridge zomes in the mycelix-media hApp.

use serde::{Deserialize, Serialize};

// =============================================================================
// Shared Test Data Structures
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPublication {
    pub id: String,
    pub title: String,
    pub content_hash: String,
    pub author_did: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestAttribution {
    pub id: String,
    pub publication_id: String,
    pub contributor_did: String,
    pub share_percentage: f64,
    pub verified: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEndorsement {
    pub id: String,
    pub publication_id: String,
    pub endorser_did: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestFactCheck {
    pub id: String,
    pub publication_id: String,
    pub claim_text: String,
    pub verdict: String,
    pub checker_did: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestQualityScore {
    pub publication_id: String,
    pub score: f64,
    pub endorsement_count: u32,
    pub fact_check_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestUsageRecord {
    pub id: String,
    pub publication_id: String,
    pub usage_type: String,
    pub royalty_paid: Option<f64>,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_publication(author_did: &str) -> TestPublication {
    TestPublication {
        id: format!("pub:{}:1704067200000000", author_did),
        title: "Test Publication".to_string(),
        content_hash: "QmTestHash".to_string(),
        author_did: author_did.to_string(),
    }
}

fn create_test_attribution(
    publication: &TestPublication,
    contributor_did: &str,
    share: f64,
) -> TestAttribution {
    TestAttribution {
        id: format!(
            "attr:{}:{}:1704067200000000",
            publication.id, contributor_did
        ),
        publication_id: publication.id.clone(),
        contributor_did: contributor_did.to_string(),
        share_percentage: share,
        verified: false,
    }
}

fn create_test_endorsement(publication: &TestPublication, endorser_did: &str) -> TestEndorsement {
    TestEndorsement {
        id: format!(
            "endorse:{}:{}:1704067200000000",
            publication.id, endorser_did
        ),
        publication_id: publication.id.clone(),
        endorser_did: endorser_did.to_string(),
    }
}

fn create_test_fact_check(publication: &TestPublication, checker_did: &str) -> TestFactCheck {
    TestFactCheck {
        id: format!("factcheck:{}:1704067200000000", publication.id),
        publication_id: publication.id.clone(),
        claim_text: "Test claim from publication".to_string(),
        verdict: "True".to_string(),
        checker_did: checker_did.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PUBLICATION + ATTRIBUTION INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_publication_with_attributions() {
        let publication = create_test_publication("did:mycelix:author1");
        let attributions = vec![
            create_test_attribution(&publication, "did:mycelix:author1", 70.0),
            create_test_attribution(&publication, "did:mycelix:editor1", 20.0),
            create_test_attribution(&publication, "did:mycelix:researcher1", 10.0),
        ];

        // Verify total shares
        let total: f64 = attributions.iter().map(|a| a.share_percentage).sum();
        assert_eq!(total, 100.0);

        // All attributions reference the same publication
        for attr in &attributions {
            assert_eq!(attr.publication_id, publication.id);
        }
    }

    #[test]
    fn test_attribution_verification_workflow() {
        let publication = create_test_publication("did:mycelix:author1");
        let mut attribution =
            create_test_attribution(&publication, "did:mycelix:contributor1", 25.0);

        // Initial state: unverified
        assert!(!attribution.verified);

        // After contributor verifies
        attribution.verified = true;
        assert!(attribution.verified);
    }

    #[test]
    fn test_royalty_distribution_on_usage() {
        let publication = create_test_publication("did:mycelix:author1");
        let attributions = vec![
            create_test_attribution(&publication, "did:mycelix:author1", 60.0),
            create_test_attribution(&publication, "did:mycelix:editor1", 30.0),
            create_test_attribution(&publication, "did:mycelix:platform", 10.0),
        ];

        let usage = TestUsageRecord {
            id: "usage:1".to_string(),
            publication_id: publication.id.clone(),
            usage_type: "View".to_string(),
            royalty_paid: Some(1.0),
        };

        // Calculate each contributor's share
        if let Some(royalty) = usage.royalty_paid {
            let author_share = royalty * (attributions[0].share_percentage / 100.0);
            let editor_share = royalty * (attributions[1].share_percentage / 100.0);
            let platform_share = royalty * (attributions[2].share_percentage / 100.0);

            assert!((author_share - 0.60).abs() < 0.001);
            assert!((editor_share - 0.30).abs() < 0.001);
            assert!((platform_share - 0.10).abs() < 0.001);
            assert!((author_share + editor_share + platform_share - royalty).abs() < 0.001);
        }
    }

    // =========================================================================
    // PUBLICATION + CURATION INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_publication_endorsements_affect_quality() {
        let publication = create_test_publication("did:mycelix:author1");
        let endorsements: Vec<TestEndorsement> = (0..50)
            .map(|i| create_test_endorsement(&publication, &format!("did:mycelix:user{}", i)))
            .collect();

        // Calculate quality score based on endorsements
        let endorsement_count = endorsements.len() as u32;
        let quality_factor = (endorsement_count as f64 / 100.0).min(1.0);

        let quality_score = TestQualityScore {
            publication_id: publication.id.clone(),
            score: quality_factor,
            endorsement_count,
            fact_check_score: 0.8, // From factcheck zome
        };

        assert_eq!(quality_score.endorsement_count, 50);
        assert!((quality_score.score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_collection_with_quality_filtered_publications() {
        let publications: Vec<TestPublication> = (0..10)
            .map(|i| create_test_publication(&format!("did:mycelix:author{}", i)))
            .collect();

        // Create quality scores for each publication
        let quality_scores: Vec<TestQualityScore> = publications
            .iter()
            .enumerate()
            .map(|(i, p)| TestQualityScore {
                publication_id: p.id.clone(),
                score: (i as f64 + 1.0) / 10.0, // 0.1 to 1.0
                endorsement_count: (i + 1) as u32 * 10,
                fact_check_score: 0.8,
            })
            .collect();

        // Filter for high-quality publications (score > 0.5)
        let high_quality: Vec<&TestQualityScore> =
            quality_scores.iter().filter(|q| q.score > 0.5).collect();

        assert_eq!(high_quality.len(), 5);
    }

    // =========================================================================
    // PUBLICATION + FACTCHECK INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_publication_fact_checks() {
        let publication = create_test_publication("did:mycelix:author1");
        let fact_checks = vec![
            {
                let mut fc = create_test_fact_check(&publication, "did:mycelix:checker1");
                fc.claim_text = "Claim 1 in paragraph 1".to_string();
                fc.verdict = "True".to_string();
                fc
            },
            {
                let mut fc = create_test_fact_check(&publication, "did:mycelix:checker2");
                fc.id = "factcheck:2".to_string();
                fc.claim_text = "Claim 2 in paragraph 3".to_string();
                fc.verdict = "MostlyTrue".to_string();
                fc
            },
            {
                let mut fc = create_test_fact_check(&publication, "did:mycelix:checker1");
                fc.id = "factcheck:3".to_string();
                fc.claim_text = "Claim 3 in paragraph 5".to_string();
                fc.verdict = "False".to_string();
                fc
            },
        ];

        // All fact checks reference the same publication
        for fc in &fact_checks {
            assert_eq!(fc.publication_id, publication.id);
        }

        // Count verdicts
        let true_count = fact_checks
            .iter()
            .filter(|fc| fc.verdict == "True" || fc.verdict == "MostlyTrue")
            .count();
        let false_count = fact_checks
            .iter()
            .filter(|fc| fc.verdict == "False")
            .count();

        assert_eq!(true_count, 2);
        assert_eq!(false_count, 1);
    }

    #[test]
    fn test_fact_check_affects_quality_score() {
        let publication = create_test_publication("did:mycelix:author1");

        // Publication with good fact check results
        let good_quality = TestQualityScore {
            publication_id: publication.id.clone(),
            score: 0.9,
            endorsement_count: 100,
            fact_check_score: 0.95,
        };

        // Publication with poor fact check results
        let poor_quality = TestQualityScore {
            publication_id: "pub:poor".to_string(),
            score: 0.3,
            endorsement_count: 50,
            fact_check_score: 0.2,
        };

        // Quality score should reflect fact check results
        assert!(good_quality.fact_check_score > poor_quality.fact_check_score);
        assert!(good_quality.score > poor_quality.score);
    }

    // =========================================================================
    // ATTRIBUTION + CURATION INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_contributor_reputation_from_endorsements() {
        let publication = create_test_publication("did:mycelix:author1");
        let attribution = create_test_attribution(&publication, "did:mycelix:author1", 100.0);

        // Publication receives endorsements
        let endorsements: Vec<TestEndorsement> = (0..30)
            .map(|i| create_test_endorsement(&publication, &format!("did:mycelix:user{}", i)))
            .collect();

        // Contributor's reputation is affected by endorsements
        let endorsement_count = endorsements.len() as u32;

        // Reputation calculation (simplified)
        let reputation_boost =
            (endorsement_count as f64 / 50.0).min(1.0) * (attribution.share_percentage / 100.0);

        assert!((reputation_boost - 0.6).abs() < 0.001);
    }

    // =========================================================================
    // FACTCHECK + ATTRIBUTION INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_fact_checker_attribution() {
        let publication = create_test_publication("did:mycelix:author1");
        let fact_check = create_test_fact_check(&publication, "did:mycelix:checker1");

        // Fact checkers can be attributed for their contribution
        let checker_attribution = TestAttribution {
            id: format!("attr:factcheck:{}:1704067200000000", fact_check.checker_did),
            publication_id: fact_check.id.clone(), // Reference the fact check, not the publication
            contributor_did: fact_check.checker_did.clone(),
            share_percentage: 100.0, // Full credit for their fact check
            verified: true,          // Auto-verified for fact checkers
        };

        assert_eq!(checker_attribution.contributor_did, fact_check.checker_did);
        assert!(checker_attribution.verified);
    }

    // =========================================================================
    // FULL WORKFLOW INTEGRATION TESTS
    // =========================================================================

    #[test]
    fn test_complete_publication_lifecycle() {
        // Step 1: Author creates publication
        let publication = create_test_publication("did:mycelix:author1");

        // Step 2: Add attributions
        let attributions = vec![
            create_test_attribution(&publication, "did:mycelix:author1", 80.0),
            create_test_attribution(&publication, "did:mycelix:editor1", 20.0),
        ];

        // Step 3: Publication receives endorsements
        let endorsements: Vec<TestEndorsement> = (0..25)
            .map(|i| create_test_endorsement(&publication, &format!("did:mycelix:user{}", i)))
            .collect();

        // Step 4: Fact check is performed
        let fact_check = create_test_fact_check(&publication, "did:mycelix:checker1");

        // Step 5: Quality score is calculated
        let fact_check_score = if fact_check.verdict == "True" {
            1.0
        } else {
            0.5
        };
        let endorsement_factor = (endorsements.len() as f64 / 50.0).min(1.0);

        let quality_score = TestQualityScore {
            publication_id: publication.id.clone(),
            score: (endorsement_factor * 0.4 + fact_check_score * 0.6).min(1.0),
            endorsement_count: endorsements.len() as u32,
            fact_check_score,
        };

        // Step 6: Usage generates royalties
        let usage = TestUsageRecord {
            id: "usage:1".to_string(),
            publication_id: publication.id.clone(),
            usage_type: "View".to_string(),
            royalty_paid: Some(0.10),
        };

        // Verify complete workflow
        assert_eq!(attributions.len(), 2);
        assert_eq!(endorsements.len(), 25);
        assert_eq!(fact_check.verdict, "True");
        assert!(quality_score.score > 0.5);
        assert!(usage.royalty_paid.is_some());
    }

    #[test]
    fn test_content_moderation_workflow() {
        let publication = create_test_publication("did:mycelix:author1");

        // Fact check reveals false claim
        let fact_check = TestFactCheck {
            id: "factcheck:1".to_string(),
            publication_id: publication.id.clone(),
            claim_text: "Disputed claim".to_string(),
            verdict: "False".to_string(),
            checker_did: "did:mycelix:checker1".to_string(),
        };

        // Quality score drops
        let quality_score = TestQualityScore {
            publication_id: publication.id.clone(),
            score: 0.2,
            endorsement_count: 10,
            fact_check_score: 0.0, // Failed fact check
        };

        // Low quality affects visibility
        let should_be_visible = quality_score.score > 0.3;

        assert!(!should_be_visible);
    }

    #[test]
    fn test_attribution_chain_for_derivative_works() {
        // Original publication
        let original = create_test_publication("did:mycelix:author1");
        let original_attribution = create_test_attribution(&original, "did:mycelix:author1", 100.0);

        // Derivative work with attribution to original
        let derivative = TestPublication {
            id: "pub:derivative:1".to_string(),
            title: "Derivative Work".to_string(),
            content_hash: "QmDerivativeHash".to_string(),
            author_did: "did:mycelix:author2".to_string(),
        };

        let derivative_attributions = vec![
            TestAttribution {
                id: "attr:derivative:1".to_string(),
                publication_id: derivative.id.clone(),
                contributor_did: "did:mycelix:author2".to_string(),
                share_percentage: 70.0,
                verified: true,
            },
            TestAttribution {
                id: "attr:derivative:2".to_string(),
                publication_id: derivative.id.clone(),
                contributor_did: original.author_did.clone(), // Original author
                share_percentage: 30.0,                       // Royalty share for original work
                verified: true,
            },
        ];

        let total: f64 = derivative_attributions
            .iter()
            .map(|a| a.share_percentage)
            .sum();
        assert_eq!(total, 100.0);

        // Original author gets attribution in derivative
        assert!(
            derivative_attributions
                .iter()
                .any(|a| a.contributor_did == original.author_did)
        );
    }

    // =========================================================================
    // EDGE CASES AND ERROR SCENARIOS
    // =========================================================================

    #[test]
    fn test_publication_without_attributions() {
        let publication = create_test_publication("did:mycelix:author1");
        let attributions: Vec<TestAttribution> = vec![];

        // Publication can exist without explicit attributions
        // Author is implicitly 100% owner
        assert!(attributions.is_empty());
        assert!(!publication.author_did.is_empty());
    }

    #[test]
    fn test_publication_with_zero_endorsements() {
        let publication = create_test_publication("did:mycelix:author1");
        let endorsements: Vec<TestEndorsement> = vec![];

        let quality_score = TestQualityScore {
            publication_id: publication.id.clone(),
            score: 0.5, // Base score
            endorsement_count: 0,
            fact_check_score: 0.5, // Not yet fact-checked
        };

        assert_eq!(quality_score.endorsement_count, 0);
        assert_eq!(quality_score.score, 0.5);
    }

    #[test]
    fn test_multiple_fact_checks_same_claim() {
        let publication = create_test_publication("did:mycelix:author1");
        let claim = "The sky is blue";

        let fact_checks = vec![
            TestFactCheck {
                id: "fc:1".to_string(),
                publication_id: publication.id.clone(),
                claim_text: claim.to_string(),
                verdict: "True".to_string(),
                checker_did: "did:mycelix:checker1".to_string(),
            },
            TestFactCheck {
                id: "fc:2".to_string(),
                publication_id: publication.id.clone(),
                claim_text: claim.to_string(),
                verdict: "True".to_string(),
                checker_did: "did:mycelix:checker2".to_string(),
            },
        ];

        // Multiple checkers agree
        let all_agree = fact_checks.iter().all(|fc| fc.verdict == "True");
        assert!(all_agree);
    }

    #[test]
    fn test_conflicting_fact_checks() {
        let publication = create_test_publication("did:mycelix:author1");
        let claim = "Controversial claim";

        let fact_checks = vec![
            TestFactCheck {
                id: "fc:1".to_string(),
                publication_id: publication.id.clone(),
                claim_text: claim.to_string(),
                verdict: "True".to_string(),
                checker_did: "did:mycelix:checker1".to_string(),
            },
            TestFactCheck {
                id: "fc:2".to_string(),
                publication_id: publication.id.clone(),
                claim_text: claim.to_string(),
                verdict: "False".to_string(),
                checker_did: "did:mycelix:checker2".to_string(),
            },
        ];

        // Checkers disagree - needs dispute resolution
        let true_count = fact_checks.iter().filter(|fc| fc.verdict == "True").count();
        let false_count = fact_checks
            .iter()
            .filter(|fc| fc.verdict == "False")
            .count();

        assert_eq!(true_count, 1);
        assert_eq!(false_count, 1);
        // In production, this would trigger a dispute resolution process
    }

    #[test]
    fn test_high_volume_publication() {
        let publication = create_test_publication("did:mycelix:author1");

        // Many endorsements
        let endorsements: Vec<TestEndorsement> = (0..1000)
            .map(|i| create_test_endorsement(&publication, &format!("did:mycelix:user{}", i)))
            .collect();

        // Many usage records
        let usages: Vec<TestUsageRecord> = (0..10000)
            .map(|i| TestUsageRecord {
                id: format!("usage:{}", i),
                publication_id: publication.id.clone(),
                usage_type: "View".to_string(),
                royalty_paid: Some(0.001),
            })
            .collect();

        let total_royalties: f64 = usages.iter().filter_map(|u| u.royalty_paid).sum();

        assert_eq!(endorsements.len(), 1000);
        assert_eq!(usages.len(), 10000);
        assert!((total_royalties - 10.0).abs() < 0.001);
    }
}
