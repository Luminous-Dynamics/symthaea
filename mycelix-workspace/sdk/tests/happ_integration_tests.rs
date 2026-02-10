//! # Mycelix SDK - hApp Integration Tests
//!
//! These tests verify that the SDK components work correctly when integrated
//! with Holochain hApps. They simulate the usage patterns from:
//! - Federated Learning zome (MATL, Byzantine detection)
//! - Credential zome (Epistemic classification)
//! - Bridge zome (Cross-hApp reputation)
//! - Mail trust filter (Trust scoring)

use mycelix_sdk::{
    bridge::{CrossHappReputation, HappReputationScore},
    epistemic::{
        ClaimBuilder, EmpiricalLevel, EpistemicClaim, EpistemicClassification, MaterialityLevel,
        NormativeLevel,
    },
    matl::{AdaptiveThreshold, CartelDetector, HierarchicalDetector, ProofOfGradientQuality},
};

// ============================================================================
// Federated Learning Integration Tests
// ============================================================================

mod federated_learning_integration {
    use super::*;

    /// Simulates FL node trust scoring as used in federated_learning coordinator
    #[test]
    fn test_fl_node_trust_scoring() {
        // Create PoGQ for a node's contribution
        let pogq = ProofOfGradientQuality::new(
            0.92, // quality: high quality gradient
            0.88, // consistency: stable contributor
            0.05, // entropy: low divergence from aggregate
        );

        // Compute composite score with reputation
        let reputation = 0.75; // Node's historical reputation
        let composite = pogq.composite_score(reputation);

        // Verify score is in valid range
        assert!(composite >= 0.0 && composite <= 1.0);
        // High quality + good reputation should yield good score
        assert!(
            composite > 0.7,
            "Expected good composite score, got {}",
            composite
        );
    }

    /// Tests HierarchicalDetector for Byzantine attack detection
    #[test]
    fn test_fl_hierarchical_detection() {
        let mut detector = HierarchicalDetector::new(3, 2);

        // Add honest nodes (high scores)
        detector.assign("node1", 0.9);
        detector.assign("node2", 0.85);
        detector.assign("node3", 0.88);

        // Add Byzantine nodes (low scores)
        detector.assign("malicious1", 0.1);
        detector.assign("malicious2", 0.15);

        // Get suspected Byzantine nodes
        let suspected = detector.get_suspected_byzantine();

        // Malicious nodes should be flagged due to low scores
        assert!(
            suspected.contains(&"malicious1".to_string()),
            "Expected malicious1 to be detected, got {:?}",
            suspected
        );
        assert!(
            suspected.contains(&"malicious2".to_string()),
            "Expected malicious2 to be detected"
        );

        // Honest nodes should not be flagged
        assert!(
            !suspected.contains(&"node1".to_string()),
            "Honest node1 should not be flagged"
        );
    }

    /// Tests CartelDetector for colluding attacker detection
    #[test]
    fn test_fl_cartel_detection() {
        // threshold=0.8, min_size=2
        let mut detector = CartelDetector::new(0.8, 2);

        // Record high similarity between cartel members
        detector.record_similarity("cartel1", "cartel2", 0.95);
        detector.record_similarity("cartel2", "cartel3", 0.92);
        detector.record_similarity("cartel1", "cartel3", 0.90);

        // Record low similarity between honest nodes
        detector.record_similarity("honest1", "honest2", 0.3);
        detector.record_similarity("honest1", "cartel1", 0.1);

        // Detect cartels
        let cartels = detector.detect();

        // Should detect the colluding group
        assert!(!cartels.is_empty(), "Expected cartel to be detected");

        // Verify cartel members are identified
        assert!(
            detector.is_cartel_member("cartel1"),
            "cartel1 should be identified"
        );
        assert!(
            detector.is_cartel_member("cartel2"),
            "cartel2 should be identified"
        );
        assert!(
            !detector.is_cartel_member("honest1"),
            "honest1 should not be in cartel"
        );
    }

    /// Tests AdaptiveThreshold for dynamic threshold adjustment
    #[test]
    fn test_fl_adaptive_threshold() {
        // node_id="test_node", window_size=10
        let mut threshold = AdaptiveThreshold::new("test_node", 10);

        // Simulate observing node behaviors (good quality)
        threshold.observe(0.9);
        threshold.observe(0.85);
        threshold.observe(0.88);
        threshold.observe(0.87);
        threshold.observe(0.91);

        // Threshold should adapt based on observations
        let t = threshold.threshold();
        assert!(t >= 0.0 && t <= 1.0, "Threshold {} should be in [0,1]", t);

        // A low score should be anomalous for this high-performing node
        assert!(
            threshold.is_anomalous(0.3),
            "0.3 should be anomalous for a node averaging ~0.88"
        );
    }
}

// ============================================================================
// Epistemic Classification Integration Tests
// ============================================================================

mod epistemic_integration {
    use super::*;

    /// Simulates credential epistemic classification as used in credential_zome
    #[test]
    fn test_credential_epistemic_classification() {
        // Educational credential (cryptographically signed, community validated)
        let classification = EpistemicClassification::new(
            EmpiricalLevel::E3Cryptographic, // Ed25519 signature
            NormativeLevel::N1Communal,      // Institution-level authority
            MaterialityLevel::M2Persistent,  // Archived after completion
        );

        // Check minimum requirements for educational context
        assert!(
            classification.meets_requirements(
                EmpiricalLevel::E1Testimonial,
                NormativeLevel::N1Communal,
                MaterialityLevel::M1Temporal,
            ),
            "Educational credential should meet minimum epistemic requirements"
        );
    }

    /// Tests empirical level auto-determination from proof type
    #[test]
    fn test_proof_type_to_empirical_level() {
        // Ed25519 signature -> E3 Cryptographic
        assert_eq!(
            determine_empirical_level("Ed25519Signature2020"),
            EmpiricalLevel::E3Cryptographic
        );

        // ZK proof -> E4 Public Reproducible
        assert_eq!(
            determine_empirical_level("ZKProof2023"),
            EmpiricalLevel::E4PublicRepro
        );

        // Unknown -> E2 Private Verify
        assert_eq!(
            determine_empirical_level("CustomProof"),
            EmpiricalLevel::E2PrivateVerify
        );
    }

    /// Helper to determine empirical level (mirrors credential_zome logic)
    fn determine_empirical_level(proof_type: &str) -> EmpiricalLevel {
        match proof_type {
            "Ed25519Signature2020" | "JsonWebSignature2020" => EmpiricalLevel::E3Cryptographic,
            "RsaSignature2018" => EmpiricalLevel::E3Cryptographic,
            "ZKProof2023" | "BbsBlsSignature2020" => EmpiricalLevel::E4PublicRepro,
            _ => EmpiricalLevel::E2PrivateVerify,
        }
    }

    /// Tests EpistemicClaim builder for creating claims
    #[test]
    fn test_epistemic_claim_creation() {
        let claim = ClaimBuilder::new("degree_completion")
            .empirical(EmpiricalLevel::E3Cryptographic)
            .normative(NormativeLevel::N2Network)
            .materiality(MaterialityLevel::M3Foundational)
            .evidence("hash:abc123")
            .issuer("did:example:123")
            .build();

        assert_eq!(claim.content, "degree_completion");
        assert_eq!(claim.empirical, EmpiricalLevel::E3Cryptographic);
        assert_eq!(claim.normative, NormativeLevel::N2Network);
        assert_eq!(claim.materiality, MaterialityLevel::M3Foundational);
        assert_eq!(claim.issuer, "did:example:123");
        assert_eq!(claim.evidence.len(), 1);
    }

    /// Tests u8 representation for HDI compatibility
    #[test]
    fn test_epistemic_u8_representation() {
        // These are stored as u8 in Holochain entries
        assert_eq!(EmpiricalLevel::E0Null as u8, 0);
        assert_eq!(EmpiricalLevel::E1Testimonial as u8, 1);
        assert_eq!(EmpiricalLevel::E2PrivateVerify as u8, 2);
        assert_eq!(EmpiricalLevel::E3Cryptographic as u8, 3);
        assert_eq!(EmpiricalLevel::E4PublicRepro as u8, 4);

        assert_eq!(NormativeLevel::N0Personal as u8, 0);
        assert_eq!(NormativeLevel::N1Communal as u8, 1);
        assert_eq!(NormativeLevel::N2Network as u8, 2);
        assert_eq!(NormativeLevel::N3Axiomatic as u8, 3);

        assert_eq!(MaterialityLevel::M0Ephemeral as u8, 0);
        assert_eq!(MaterialityLevel::M1Temporal as u8, 1);
        assert_eq!(MaterialityLevel::M2Persistent as u8, 2);
        assert_eq!(MaterialityLevel::M3Foundational as u8, 3);
    }

    /// Tests EpistemicClaim meets_standard method
    #[test]
    fn test_claim_meets_standard() {
        let claim = EpistemicClaim::new(
            "Signed document",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        // Should meet lower standards
        assert!(claim.meets_standard(EmpiricalLevel::E2PrivateVerify, NormativeLevel::N1Communal));

        // Should not meet higher standards
        assert!(!claim.meets_standard(EmpiricalLevel::E4PublicRepro, NormativeLevel::N1Communal));
    }

    /// Tests classification code generation
    #[test]
    fn test_classification_code() {
        let claim = EpistemicClaim::new(
            "Test claim",
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
        );

        let code = claim.code();
        assert_eq!(code, "E3-N2-M2");
    }
}

// ============================================================================
// Bridge Integration Tests
// ============================================================================

mod bridge_integration {
    use super::*;

    /// Tests cross-hApp reputation aggregation as used in bridge coordinator
    #[test]
    fn test_cross_happ_reputation() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: 1000,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.8,
                interactions: 50,
                last_updated: 900,
            },
            HappReputationScore {
                happ_id: "edunet".to_string(),
                happ_name: "Mycelix EduNet".to_string(),
                score: 0.95,
                interactions: 25,
                last_updated: 800,
            },
        ];

        let reputation = CrossHappReputation::from_scores("agent1", scores.clone());

        // Verify aggregation
        assert_eq!(reputation.agent, "agent1");
        assert_eq!(reputation.scores.len(), 3);

        // Weighted average: (0.9*100 + 0.8*50 + 0.95*25) / 175 = 0.8786
        let expected_aggregate = (0.9 * 100.0 + 0.8 * 50.0 + 0.95 * 25.0) / 175.0;
        assert!(
            (reputation.aggregate - expected_aggregate).abs() < 0.001,
            "Expected aggregate {}, got {}",
            expected_aggregate,
            reputation.aggregate
        );
    }

    /// Tests trustworthiness check
    #[test]
    fn test_is_trustworthy() {
        let scores = vec![HappReputationScore {
            happ_id: "mail".to_string(),
            happ_name: "Mycelix Mail".to_string(),
            score: 0.85,
            interactions: 100,
            last_updated: 1000,
        }];

        let reputation = CrossHappReputation::from_scores("agent1", scores);

        // Should be trustworthy at threshold 0.5
        assert!(reputation.is_trustworthy(0.5));

        // Should not be trustworthy at very high threshold
        assert!(!reputation.is_trustworthy(0.9));
    }

    /// Tests empty reputation handling
    #[test]
    fn test_empty_reputation() {
        let reputation = CrossHappReputation::from_scores("new_agent", vec![]);

        assert_eq!(reputation.agent, "new_agent");
        assert_eq!(reputation.scores.len(), 0);
        // SDK returns 0.5 as default for unknown agents (neutral stance)
        assert_eq!(reputation.aggregate, 0.5);

        // New agent meets low thresholds but not high ones
        assert!(
            reputation.is_trustworthy(0.4),
            "Default 0.5 should meet 0.4 threshold"
        );
        assert!(
            !reputation.is_trustworthy(0.6),
            "Default 0.5 should not meet 0.6 threshold"
        );
    }
}

// ============================================================================
// Mail Trust Filter Integration Tests
// ============================================================================

mod mail_trust_integration {
    use super::*;

    /// Tests trust scoring workflow for mail filtering
    #[test]
    fn test_mail_trust_workflow() {
        // Step 1: Get sender's cross-hApp reputation
        let sender_scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.75,
                interactions: 50,
                last_updated: 1000,
            },
            HappReputationScore {
                happ_id: "edunet".to_string(),
                happ_name: "Mycelix EduNet".to_string(),
                score: 0.8,
                interactions: 20,
                last_updated: 900,
            },
        ];

        let reputation = CrossHappReputation::from_scores("sender", sender_scores);

        // Step 2: Check against trust level thresholds
        let priority_threshold = 0.8; // High bar for priority inbox
        let spam_threshold = 0.3; // Low bar to avoid spam filter

        let is_priority = reputation.is_trustworthy(priority_threshold);
        let is_spam = !reputation.is_trustworthy(spam_threshold);

        // Verify classification
        assert!(
            !is_priority,
            "Reputation {} shouldn't meet priority threshold {}",
            reputation.aggregate, priority_threshold
        );
        assert!(
            !is_spam,
            "Reputation {} should avoid spam threshold {}",
            reputation.aggregate, spam_threshold
        );
    }

    /// Tests combining PoGQ with reputation for mail scoring
    #[test]
    fn test_mail_composite_trust() {
        // Use MATL for computing sender trust
        let pogq = ProofOfGradientQuality::new(
            0.85, // Message quality (e.g., no spam indicators)
            0.9,  // Consistency (regular sender, not burst)
            0.1,  // Low entropy (expected content type)
        );

        let sender_reputation = 0.7;
        let trust_score = pogq.composite_score(sender_reputation);

        // Trust score determines inbox placement
        assert!(trust_score > 0.5, "Expected moderate trust score");
    }
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

mod e2e_workflows {
    use super::*;

    /// Tests complete FL round workflow
    #[test]
    fn test_complete_fl_round() {
        // Simulate a federated learning round with 5 nodes

        // Step 1: Collect node contributions with scores
        let contributions = vec![
            ("node1", 0.92), // (id, composite_score)
            ("node2", 0.89),
            ("node3", 0.91),
            ("node4", 0.25), // Byzantine node - low score
            ("node5", 0.88),
        ];

        // Step 2: Use HierarchicalDetector to classify nodes
        let mut detector = HierarchicalDetector::new(2, 2);
        for (id, score) in &contributions {
            detector.assign(id, *score);
        }

        // Step 3: Get suspected Byzantine nodes
        let suspected = detector.get_suspected_byzantine();

        // Step 4: Verify Byzantine node is detected
        assert!(
            suspected.contains(&"node4".to_string()),
            "Byzantine node should be detected as anomaly, got {:?}",
            suspected
        );

        // Step 5: Verify honest nodes are not flagged
        assert!(!suspected.contains(&"node1".to_string()));
        assert!(!suspected.contains(&"node2".to_string()));
    }

    /// Tests credential issuance with epistemic validation
    #[test]
    fn test_credential_issuance_workflow() {
        // Step 1: Define credential epistemic requirements
        let min_empirical = EmpiricalLevel::E2PrivateVerify;
        let min_normative = NormativeLevel::N1Communal;

        // Step 2: Create credential with classification
        let credential_classification = EpistemicClassification::new(
            EmpiricalLevel::E3Cryptographic,
            NormativeLevel::N1Communal,
            MaterialityLevel::M2Persistent,
        );

        // Step 3: Validate meets requirements
        assert!(
            credential_classification.meets_requirements(
                min_empirical,
                min_normative,
                MaterialityLevel::M0Ephemeral, // Any materiality is OK
            ),
            "Credential should meet epistemic requirements"
        );

        // Step 4: Create claim for the credential
        let claim = ClaimBuilder::new("course_completion")
            .empirical(credential_classification.empirical)
            .normative(credential_classification.normative)
            .materiality(credential_classification.materiality)
            .evidence("signature:xyz789")
            .issuer("did:institution:123")
            .build();

        assert_eq!(claim.content, "course_completion");
        assert!(claim.meets_standard(min_empirical, min_normative));
    }

    /// Tests cross-hApp trust flow
    #[test]
    fn test_cross_happ_trust_flow() {
        // Scenario: A mail hApp checking sender reputation from multiple sources

        // Step 1: Collect reputation from bridge
        let scores = vec![
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Marketplace".to_string(),
                score: 0.85,
                interactions: 30,
                last_updated: 1000,
            },
            HappReputationScore {
                happ_id: "edunet".to_string(),
                happ_name: "EduNet".to_string(),
                score: 0.9,
                interactions: 50,
                last_updated: 950,
            },
        ];

        let reputation = CrossHappReputation::from_scores("sender_agent", scores);

        // Step 2: Use MATL to compute message trust
        let pogq = ProofOfGradientQuality::new(
            0.8,  // Message quality
            0.85, // Sender consistency
            0.1,  // Content entropy
        );

        // Step 3: Combine reputation with message quality
        let combined_trust = pogq.composite_score(reputation.aggregate);

        // Step 4: Make routing decision
        let priority_threshold = 0.75;
        let should_prioritize = combined_trust >= priority_threshold;

        // Verify the flow produces a valid decision
        assert!(combined_trust >= 0.0 && combined_trust <= 1.0);
        // Good reputation + good message quality should result in prioritization
        assert!(
            should_prioritize,
            "Expected prioritization with trust score {}",
            combined_trust
        );
    }
}
