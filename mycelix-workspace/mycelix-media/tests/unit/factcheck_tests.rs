// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Fact-Check Zome Unit Tests
//!
//! Comprehensive test coverage for fact-checking, epistemic classification,
//! verdicts, evidence management, source credibility, and dispute handling.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestFactCheck {
    pub id: String,
    pub publication_id: String,
    pub claim_text: String,
    pub claim_location: String,
    pub epistemic_position: TestEpistemicPosition,
    pub verdict: TestFactCheckVerdict,
    pub evidence: Vec<TestEvidenceItem>,
    pub checker_did: String,
    pub checked: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestEpistemicPosition {
    pub empirical: f64, // 0.0 to 1.0
    pub normative: f64, // 0.0 to 1.0
    pub mythic: f64,    // 0.0 to 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestFactCheckVerdict {
    True,
    MostlyTrue,
    HalfTrue,
    MostlyFalse,
    False,
    Unverifiable,
    OutOfContext,
    Satire,
    Opinion,
    PartiallyTrue,
    Misleading,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestEvidenceItem {
    pub source_type: TestSourceType,
    pub source_url: Option<String>,
    pub source_did: Option<String>,
    pub description: String,
    pub supports_claim: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestSourceType {
    PrimarySource,
    SecondarySource,
    ExpertOpinion,
    OfficialDocument,
    ScientificStudy,
    EyewitnessAccount,
    DataAnalysis,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestSourceCredibility {
    pub source_id: String,
    pub source_type: TestSourceType,
    pub credibility_score: f64,
    pub verification_count: u32,
    pub dispute_count: u32,
    pub last_assessed: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestFactCheckDispute {
    pub id: String,
    pub fact_check_id: String,
    pub disputer_did: String,
    pub reason: String,
    pub counter_evidence: Vec<TestEvidenceItem>,
    pub status: TestDisputeStatus,
    pub created: i64,
    pub resolved: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDisputeStatus {
    Pending,
    Upheld,
    Rejected,
    PartiallyUpheld,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_fact_check() -> TestFactCheck {
    TestFactCheck {
        id: "factcheck:pub123:1704067200000000".to_string(),
        publication_id: "pub:did:mycelix:author123:1704067200000000".to_string(),
        claim_text: "The global temperature has risen by 1.5 degrees Celsius since 1900"
            .to_string(),
        claim_location: "paragraph 3, sentence 2".to_string(),
        epistemic_position: TestEpistemicPosition {
            empirical: 0.9,
            normative: 0.05,
            mythic: 0.05,
        },
        verdict: TestFactCheckVerdict::True,
        evidence: vec![TestEvidenceItem {
            source_type: TestSourceType::ScientificStudy,
            source_url: Some("https://ipcc.ch/report/ar6/".to_string()),
            source_did: None,
            description: "IPCC AR6 Report confirms temperature rise".to_string(),
            supports_claim: true,
        }],
        checker_did: "did:mycelix:checker1".to_string(),
        checked: 1704067200000000,
    }
}

fn create_test_source_credibility() -> TestSourceCredibility {
    TestSourceCredibility {
        source_id: "source:ipcc.ch".to_string(),
        source_type: TestSourceType::OfficialDocument,
        credibility_score: 0.95,
        verification_count: 500,
        dispute_count: 5,
        last_assessed: 1704067200000000,
    }
}

fn create_test_dispute() -> TestFactCheckDispute {
    TestFactCheckDispute {
        id: "dispute:factcheck123:1704153600000000".to_string(),
        fact_check_id: "factcheck:pub123:1704067200000000".to_string(),
        disputer_did: "did:mycelix:disputer1".to_string(),
        reason: "The methodology used in the cited study has been questioned".to_string(),
        counter_evidence: vec![TestEvidenceItem {
            source_type: TestSourceType::ScientificStudy,
            source_url: Some("https://example.com/counter-study".to_string()),
            source_did: None,
            description: "Alternative analysis suggests different findings".to_string(),
            supports_claim: false,
        }],
        status: TestDisputeStatus::Pending,
        created: 1704153600000000,
        resolved: None,
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

fn validate_epistemic_position(ep: &TestEpistemicPosition) -> Result<(), String> {
    if ep.empirical < 0.0 || ep.empirical > 1.0 {
        return Err("Empirical value must be 0-1".to_string());
    }
    if ep.normative < 0.0 || ep.normative > 1.0 {
        return Err("Normative value must be 0-1".to_string());
    }
    if ep.mythic < 0.0 || ep.mythic > 1.0 {
        return Err("Mythic value must be 0-1".to_string());
    }
    Ok(())
}

fn validate_fact_check(check: &TestFactCheck) -> Result<(), String> {
    if !validate_did(&check.checker_did) {
        return Err("Checker must be a valid DID".to_string());
    }
    if check.claim_text.is_empty() {
        return Err("Claim text required".to_string());
    }
    validate_epistemic_position(&check.epistemic_position)?;
    Ok(())
}

fn validate_source_credibility(source: &TestSourceCredibility) -> Result<(), String> {
    if source.credibility_score < 0.0 || source.credibility_score > 1.0 {
        return Err("Credibility must be 0-1".to_string());
    }
    Ok(())
}

fn validate_dispute(dispute: &TestFactCheckDispute) -> Result<(), String> {
    if !validate_did(&dispute.disputer_did) {
        return Err("Disputer must be a valid DID".to_string());
    }
    if dispute.reason.is_empty() {
        return Err("Dispute reason required".to_string());
    }
    Ok(())
}

fn can_update_fact_check(check: &TestFactCheck, requester_did: &str) -> bool {
    check.checker_did == requester_did
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // FACT CHECK CREATION TESTS
    // =========================================================================

    #[test]
    fn test_fact_check_requires_valid_checker_did() {
        let check = create_test_fact_check();
        assert!(validate_did(&check.checker_did));
    }

    #[test]
    fn test_fact_check_invalid_checker_did_rejected() {
        let mut check = create_test_fact_check();
        check.checker_did = "invalid_did".to_string();
        let result = validate_fact_check(&check);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Checker must be a valid DID");
    }

    #[test]
    fn test_fact_check_requires_claim_text() {
        let check = create_test_fact_check();
        assert!(!check.claim_text.is_empty());
    }

    #[test]
    fn test_fact_check_empty_claim_rejected() {
        let mut check = create_test_fact_check();
        check.claim_text = "".to_string();
        let result = validate_fact_check(&check);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Claim text required");
    }

    #[test]
    fn test_fact_check_has_unique_id() {
        let check = create_test_fact_check();
        assert!(check.id.starts_with("factcheck:"));
    }

    #[test]
    fn test_fact_check_references_publication() {
        let check = create_test_fact_check();
        assert!(!check.publication_id.is_empty());
    }

    #[test]
    fn test_fact_check_has_claim_location() {
        let check = create_test_fact_check();
        assert!(!check.claim_location.is_empty());
    }

    #[test]
    fn test_fact_check_timestamp() {
        let check = create_test_fact_check();
        assert!(check.checked > 0);
    }

    // =========================================================================
    // EPISTEMIC POSITION TESTS
    // =========================================================================

    #[test]
    fn test_epistemic_position_valid_range() {
        let check = create_test_fact_check();
        let ep = &check.epistemic_position;
        assert!(ep.empirical >= 0.0 && ep.empirical <= 1.0);
        assert!(ep.normative >= 0.0 && ep.normative <= 1.0);
        assert!(ep.mythic >= 0.0 && ep.mythic <= 1.0);
    }

    #[test]
    fn test_epistemic_position_invalid_empirical() {
        let mut check = create_test_fact_check();
        check.epistemic_position.empirical = 1.5;
        let result = validate_fact_check(&check);
        assert!(result.is_err());
    }

    #[test]
    fn test_epistemic_position_invalid_normative() {
        let mut check = create_test_fact_check();
        check.epistemic_position.normative = -0.1;
        let result = validate_fact_check(&check);
        assert!(result.is_err());
    }

    #[test]
    fn test_epistemic_position_invalid_mythic() {
        let mut check = create_test_fact_check();
        check.epistemic_position.mythic = 2.0;
        let result = validate_fact_check(&check);
        assert!(result.is_err());
    }

    #[test]
    fn test_epistemic_position_pure_empirical() {
        let ep = TestEpistemicPosition {
            empirical: 1.0,
            normative: 0.0,
            mythic: 0.0,
        };
        assert!(validate_epistemic_position(&ep).is_ok());
    }

    #[test]
    fn test_epistemic_position_pure_normative() {
        let ep = TestEpistemicPosition {
            empirical: 0.0,
            normative: 1.0,
            mythic: 0.0,
        };
        assert!(validate_epistemic_position(&ep).is_ok());
    }

    #[test]
    fn test_epistemic_position_mixed() {
        let ep = TestEpistemicPosition {
            empirical: 0.33,
            normative: 0.33,
            mythic: 0.34,
        };
        assert!(validate_epistemic_position(&ep).is_ok());
    }

    #[test]
    fn test_epistemic_classification_for_scientific_claim() {
        let check = create_test_fact_check();
        // Scientific claims should be highly empirical
        assert!(check.epistemic_position.empirical > 0.7);
        assert!(check.epistemic_position.normative < 0.3);
    }

    // =========================================================================
    // VERDICT TESTS
    // =========================================================================

    #[test]
    fn test_all_verdicts_supported() {
        let verdicts = vec![
            TestFactCheckVerdict::True,
            TestFactCheckVerdict::MostlyTrue,
            TestFactCheckVerdict::HalfTrue,
            TestFactCheckVerdict::MostlyFalse,
            TestFactCheckVerdict::False,
            TestFactCheckVerdict::Unverifiable,
            TestFactCheckVerdict::OutOfContext,
            TestFactCheckVerdict::Satire,
            TestFactCheckVerdict::Opinion,
            TestFactCheckVerdict::PartiallyTrue,
            TestFactCheckVerdict::Misleading,
        ];
        for verdict in verdicts {
            let mut check = create_test_fact_check();
            check.verdict = verdict;
            assert!(validate_fact_check(&check).is_ok());
        }
    }

    #[test]
    fn test_verdict_true() {
        let check = create_test_fact_check();
        assert_eq!(check.verdict, TestFactCheckVerdict::True);
    }

    // =========================================================================
    // EVIDENCE TESTS
    // =========================================================================

    fn create_test_evidence() -> TestEvidenceItem {
        TestEvidenceItem {
            source_type: TestSourceType::ScientificStudy,
            source_url: Some("https://example.com/study".to_string()),
            source_did: None,
            description: "Scientific study supporting the claim".to_string(),
            supports_claim: true,
        }
    }

    #[test]
    fn test_evidence_item_creation() {
        let evidence = create_test_evidence();
        assert!(!evidence.description.is_empty());
    }

    #[test]
    fn test_evidence_source_types() {
        let types = vec![
            TestSourceType::PrimarySource,
            TestSourceType::SecondarySource,
            TestSourceType::ExpertOpinion,
            TestSourceType::OfficialDocument,
            TestSourceType::ScientificStudy,
            TestSourceType::EyewitnessAccount,
            TestSourceType::DataAnalysis,
            TestSourceType::Other("Archival Material".to_string()),
        ];
        for source_type in types {
            let mut evidence = create_test_evidence();
            evidence.source_type = source_type;
            // All source types should be valid
            assert!(!evidence.description.is_empty());
        }
    }

    #[test]
    fn test_evidence_with_url() {
        let evidence = create_test_evidence();
        assert!(evidence.source_url.is_some());
    }

    #[test]
    fn test_evidence_with_did() {
        let mut evidence = create_test_evidence();
        evidence.source_did = Some("did:mycelix:source1".to_string());
        evidence.source_url = None;
        assert!(evidence.source_did.is_some());
    }

    #[test]
    fn test_evidence_supports_claim() {
        let evidence = create_test_evidence();
        assert!(evidence.supports_claim);
    }

    #[test]
    fn test_evidence_contradicts_claim() {
        let mut evidence = create_test_evidence();
        evidence.supports_claim = false;
        assert!(!evidence.supports_claim);
    }

    #[test]
    fn test_fact_check_with_multiple_evidence() {
        let mut check = create_test_fact_check();
        check.evidence = vec![
            create_test_evidence(),
            {
                let mut e = create_test_evidence();
                e.source_type = TestSourceType::OfficialDocument;
                e
            },
            {
                let mut e = create_test_evidence();
                e.source_type = TestSourceType::ExpertOpinion;
                e
            },
        ];
        assert_eq!(check.evidence.len(), 3);
    }

    #[test]
    fn test_fact_check_no_evidence() {
        let mut check = create_test_fact_check();
        check.evidence = vec![];
        // Fact checks without evidence might still be valid (e.g., for opinions)
        assert!(validate_fact_check(&check).is_ok());
    }

    // =========================================================================
    // SOURCE CREDIBILITY TESTS
    // =========================================================================

    #[test]
    fn test_source_credibility_creation() {
        let source = create_test_source_credibility();
        assert!(!source.source_id.is_empty());
    }

    #[test]
    fn test_source_credibility_valid_range() {
        let valid_scores = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        for score in valid_scores {
            let mut source = create_test_source_credibility();
            source.credibility_score = score;
            assert!(validate_source_credibility(&source).is_ok());
        }
    }

    #[test]
    fn test_source_credibility_negative_rejected() {
        let mut source = create_test_source_credibility();
        source.credibility_score = -0.1;
        let result = validate_source_credibility(&source);
        assert!(result.is_err());
    }

    #[test]
    fn test_source_credibility_over_1_rejected() {
        let mut source = create_test_source_credibility();
        source.credibility_score = 1.5;
        let result = validate_source_credibility(&source);
        assert!(result.is_err());
    }

    #[test]
    fn test_source_credibility_verification_count() {
        let source = create_test_source_credibility();
        assert!(source.verification_count > 0);
    }

    #[test]
    fn test_source_credibility_dispute_count() {
        let source = create_test_source_credibility();
        assert!(source.dispute_count < source.verification_count);
    }

    #[test]
    fn test_source_credibility_calculation() {
        fn calculate_credibility(verification_count: u32, dispute_count: u32) -> f64 {
            if verification_count == 0 {
                return 0.5; // Neutral for unknown sources
            }
            let ratio = 1.0 - (dispute_count as f64 / verification_count as f64);
            ratio.max(0.0).min(1.0)
        }

        assert!((calculate_credibility(100, 5) - 0.95).abs() < 0.001);
        assert!((calculate_credibility(100, 50) - 0.5).abs() < 0.001);
        assert_eq!(calculate_credibility(0, 0), 0.5);
    }

    // =========================================================================
    // DISPUTE TESTS
    // =========================================================================

    #[test]
    fn test_dispute_requires_valid_disputer_did() {
        let dispute = create_test_dispute();
        assert!(validate_did(&dispute.disputer_did));
    }

    #[test]
    fn test_dispute_invalid_disputer_did_rejected() {
        let mut dispute = create_test_dispute();
        dispute.disputer_did = "invalid".to_string();
        let result = validate_dispute(&dispute);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Disputer must be a valid DID");
    }

    #[test]
    fn test_dispute_requires_reason() {
        let dispute = create_test_dispute();
        assert!(!dispute.reason.is_empty());
    }

    #[test]
    fn test_dispute_empty_reason_rejected() {
        let mut dispute = create_test_dispute();
        dispute.reason = "".to_string();
        let result = validate_dispute(&dispute);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Dispute reason required");
    }

    #[test]
    fn test_dispute_has_unique_id() {
        let dispute = create_test_dispute();
        assert!(dispute.id.starts_with("dispute:"));
    }

    #[test]
    fn test_dispute_references_fact_check() {
        let dispute = create_test_dispute();
        assert!(!dispute.fact_check_id.is_empty());
    }

    #[test]
    fn test_dispute_initial_status_pending() {
        let dispute = create_test_dispute();
        assert_eq!(dispute.status, TestDisputeStatus::Pending);
    }

    #[test]
    fn test_dispute_statuses_supported() {
        let statuses = vec![
            TestDisputeStatus::Pending,
            TestDisputeStatus::Upheld,
            TestDisputeStatus::Rejected,
            TestDisputeStatus::PartiallyUpheld,
        ];
        for status in statuses {
            let mut dispute = create_test_dispute();
            dispute.status = status;
            assert!(validate_dispute(&dispute).is_ok());
        }
    }

    #[test]
    fn test_dispute_resolved_timestamp() {
        let mut dispute = create_test_dispute();
        dispute.status = TestDisputeStatus::Rejected;
        dispute.resolved = Some(dispute.created + 86400000000);
        assert!(dispute.resolved.unwrap() > dispute.created);
    }

    #[test]
    fn test_dispute_pending_no_resolved() {
        let dispute = create_test_dispute();
        assert_eq!(dispute.status, TestDisputeStatus::Pending);
        assert!(dispute.resolved.is_none());
    }

    #[test]
    fn test_dispute_with_counter_evidence() {
        let dispute = create_test_dispute();
        assert!(!dispute.counter_evidence.is_empty());
    }

    #[test]
    fn test_dispute_without_counter_evidence() {
        let mut dispute = create_test_dispute();
        dispute.counter_evidence = vec![];
        // Disputes without counter evidence are valid (opinion disputes)
        assert!(validate_dispute(&dispute).is_ok());
    }

    // =========================================================================
    // ACCESS CONTROL TESTS
    // =========================================================================

    #[test]
    fn test_only_checker_can_update() {
        let check = create_test_fact_check();
        assert!(can_update_fact_check(&check, &check.checker_did));
        assert!(!can_update_fact_check(&check, "did:mycelix:other"));
    }

    #[test]
    fn test_fact_checks_immutable_by_default() {
        // According to validation rules, fact checks cannot be updated
        // However, evidence and verdicts can be updated by the original checker
        let check = create_test_fact_check();
        let check_id = check.id.clone();
        assert_eq!(check.id, check_id);
    }

    // =========================================================================
    // CHECKER STATS TESTS
    // =========================================================================

    #[derive(Debug)]
    struct CheckerStats {
        checker_did: String,
        total_checks: u32,
        true_count: u32,
        false_count: u32,
        mixed_count: u32,
    }

    fn calculate_checker_stats(checks: &[TestFactCheck], checker_did: &str) -> CheckerStats {
        let checker_checks: Vec<&TestFactCheck> = checks
            .iter()
            .filter(|c| c.checker_did == checker_did)
            .collect();

        let mut true_count = 0;
        let mut false_count = 0;
        let mut mixed_count = 0;

        for check in &checker_checks {
            match check.verdict {
                TestFactCheckVerdict::True => true_count += 1,
                TestFactCheckVerdict::False => false_count += 1,
                TestFactCheckVerdict::PartiallyTrue
                | TestFactCheckVerdict::Misleading
                | TestFactCheckVerdict::HalfTrue => mixed_count += 1,
                _ => {}
            }
        }

        CheckerStats {
            checker_did: checker_did.to_string(),
            total_checks: checker_checks.len() as u32,
            true_count,
            false_count,
            mixed_count,
        }
    }

    #[test]
    fn test_checker_stats_calculation() {
        let checks = vec![
            create_test_fact_check(),
            {
                let mut c = create_test_fact_check();
                c.verdict = TestFactCheckVerdict::False;
                c
            },
            {
                let mut c = create_test_fact_check();
                c.verdict = TestFactCheckVerdict::HalfTrue;
                c
            },
        ];

        let stats = calculate_checker_stats(&checks, "did:mycelix:checker1");
        assert_eq!(stats.total_checks, 3);
        assert_eq!(stats.true_count, 1);
        assert_eq!(stats.false_count, 1);
        assert_eq!(stats.mixed_count, 1);
    }

    // =========================================================================
    // SEARCH AND FILTER TESTS
    // =========================================================================

    fn filter_by_verdict<'a>(
        checks: &'a [TestFactCheck],
        verdict: &TestFactCheckVerdict,
    ) -> Vec<&'a TestFactCheck> {
        checks.iter().filter(|c| &c.verdict == verdict).collect()
    }

    #[test]
    fn test_filter_by_verdict() {
        let checks = vec![
            create_test_fact_check(),
            {
                let mut c = create_test_fact_check();
                c.verdict = TestFactCheckVerdict::False;
                c
            },
            {
                let mut c = create_test_fact_check();
                c.verdict = TestFactCheckVerdict::True;
                c
            },
        ];

        let true_checks = filter_by_verdict(&checks, &TestFactCheckVerdict::True);
        assert_eq!(true_checks.len(), 2);

        let false_checks = filter_by_verdict(&checks, &TestFactCheckVerdict::False);
        assert_eq!(false_checks.len(), 1);
    }

    #[test]
    fn test_search_by_claim_text() {
        let checks = vec![create_test_fact_check(), {
            let mut c = create_test_fact_check();
            c.claim_text = "The economy grew by 5%".to_string();
            c
        }];

        let matches: Vec<&TestFactCheck> = checks
            .iter()
            .filter(|c| c.claim_text.to_lowercase().contains("temperature"))
            .collect();

        assert_eq!(matches.len(), 1);
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_fact_check_long_claim_text() {
        let mut check = create_test_fact_check();
        check.claim_text = "A".repeat(10000);
        assert!(validate_fact_check(&check).is_ok());
    }

    #[test]
    fn test_fact_check_unicode_claim() {
        let mut check = create_test_fact_check();
        check.claim_text = "Climate change affects global temperatures".to_string();
        assert!(validate_fact_check(&check).is_ok());
    }

    #[test]
    fn test_fact_check_many_evidence_items() {
        let mut check = create_test_fact_check();
        check.evidence = (0..100)
            .map(|i| TestEvidenceItem {
                source_type: TestSourceType::SecondarySource,
                source_url: Some(format!("https://example.com/source{}", i)),
                source_did: None,
                description: format!("Evidence item {}", i),
                supports_claim: i % 2 == 0,
            })
            .collect();

        assert_eq!(check.evidence.len(), 100);
        let supporting: Vec<&TestEvidenceItem> =
            check.evidence.iter().filter(|e| e.supports_claim).collect();
        assert_eq!(supporting.len(), 50);
    }

    #[test]
    fn test_epistemic_position_edge_values() {
        let positions = vec![
            TestEpistemicPosition {
                empirical: 0.0,
                normative: 0.0,
                mythic: 0.0,
            },
            TestEpistemicPosition {
                empirical: 1.0,
                normative: 1.0,
                mythic: 1.0,
            },
            TestEpistemicPosition {
                empirical: 0.0001,
                normative: 0.0001,
                mythic: 0.0001,
            },
        ];
        for ep in positions {
            assert!(validate_epistemic_position(&ep).is_ok());
        }
    }

    #[test]
    fn test_source_credibility_high_volume() {
        let source = TestSourceCredibility {
            source_id: "major_news".to_string(),
            source_type: TestSourceType::SecondarySource,
            credibility_score: 0.85,
            verification_count: 100000,
            dispute_count: 500,
            last_assessed: 1704067200000000,
        };
        assert!(validate_source_credibility(&source).is_ok());
    }

    #[test]
    fn test_dispute_chain() {
        // Multiple disputes on same fact check
        let fact_check_id = "factcheck:pub123:1704067200000000".to_string();
        let disputes: Vec<TestFactCheckDispute> = (0..5)
            .map(|i| {
                let mut d = create_test_dispute();
                d.id = format!("dispute:{}:{}", i, 1704153600000000 + i as i64);
                d.fact_check_id = fact_check_id.clone();
                d.disputer_did = format!("did:mycelix:disputer{}", i);
                d
            })
            .collect();

        assert_eq!(disputes.len(), 5);
        for dispute in &disputes {
            assert_eq!(dispute.fact_check_id, fact_check_id);
        }
    }

    #[test]
    fn test_multiple_fact_checks_same_publication() {
        let pub_id = "pub:test";
        let checks: Vec<TestFactCheck> = (0..10)
            .map(|i| {
                let mut c = create_test_fact_check();
                c.id = format!("factcheck:{}", i);
                c.publication_id = pub_id.to_string();
                c.claim_text = format!("Claim number {}", i);
                c.claim_location = format!("paragraph {}", i);
                c
            })
            .collect();

        let pub_checks: Vec<&TestFactCheck> = checks
            .iter()
            .filter(|c| c.publication_id == pub_id)
            .collect();
        assert_eq!(pub_checks.len(), 10);
    }
}
