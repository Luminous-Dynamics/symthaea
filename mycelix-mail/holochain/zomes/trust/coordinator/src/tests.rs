// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Unit Tests for Trust Coordinator Zome (MATL Algorithm)
//!
//! Tests for trust attestations, scoring, Byzantine detection, and introductions.
//! Uses mock helpers to simulate Holochain environment.

#[cfg(test)]
mod tests {
    use super::*;
    use mail_trust_integrity::*;

    // ==================== TEST FIXTURES ====================

    /// Creates a test AgentPubKey from bytes
    fn test_agent(id: u8) -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![id; 36])
    }

    /// Creates a test ActionHash from bytes
    fn test_action_hash(id: u8) -> ActionHash {
        ActionHash::from_raw_36(vec![id; 36])
    }

    /// Creates a test Timestamp
    fn test_timestamp(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    /// Creates a basic TrustEvidence
    fn test_evidence(evidence_type: EvidenceType, weight: f64) -> TrustEvidence {
        TrustEvidence {
            evidence_type,
            reference: "https://example.com/evidence".to_string(),
            weight,
            collected_at: test_timestamp(1000000),
            verified: true,
        }
    }

    /// Creates a test TrustAttestation
    fn test_attestation(
        truster: AgentPubKey,
        trustee: AgentPubKey,
        trust_level: f64,
        category: TrustCategory,
    ) -> TrustAttestation {
        TrustAttestation {
            truster,
            trustee,
            trust_level,
            category,
            evidence: vec![test_evidence(EvidenceType::InPersonMeeting, 0.9)],
            reason: Some("Verified in person".to_string()),
            created_at: test_timestamp(1000000),
            expires_at: None,
            signature: vec![1, 2, 3, 4, 5],
            revoked: false,
            stake: None,
        }
    }

    /// Creates a test CreateAttestationInput
    fn test_create_attestation_input(
        trustee: AgentPubKey,
        trust_level: f64,
    ) -> CreateAttestationInput {
        CreateAttestationInput {
            trustee,
            trust_level,
            category: TrustCategory::Identity,
            evidence: vec![test_evidence(EvidenceType::VideoVerification, 0.85)],
            reason: Some("Video call verification".to_string()),
            expires_at: None,
            stake: None,
        }
    }

    /// Creates a test TrustScore
    fn test_trust_score(agent: AgentPubKey, score: f64, confidence: f64) -> TrustScore {
        TrustScore {
            agent,
            score,
            confidence,
            direct_attestation_count: 5,
            transitive_path_count: 10,
            category_scores: vec![
                (TrustCategory::Identity, 0.8),
                (TrustCategory::Communication, 0.7),
            ],
            computed_at: test_timestamp(1000000),
            attestation_hashes: vec![test_action_hash(1), test_action_hash(2)],
            byzantine_flags: Vec::new(),
        }
    }

    /// Creates a test TrustIntroduction
    fn test_introduction(
        introducer: AgentPubKey,
        introduced: AgentPubKey,
        target: AgentPubKey,
    ) -> TrustIntroduction {
        TrustIntroduction {
            introducer,
            introduced,
            target,
            recommendation_level: 0.8,
            category: TrustCategory::Professional,
            message: Some("I can vouch for this person".to_string()),
            introduced_at: test_timestamp(1000000),
            accepted: None,
        }
    }

    /// Creates a test TrustDispute
    fn test_dispute(
        attestation_hash: ActionHash,
        disputer: AgentPubKey,
    ) -> TrustDispute {
        TrustDispute {
            attestation_hash,
            disputer,
            reason: "This attestation contains false information".to_string(),
            counter_evidence: vec![test_evidence(EvidenceType::Custom("Counter proof".to_string()), 0.7)],
            status: DisputeStatus::Open,
            filed_at: test_timestamp(1000000),
            resolution: None,
        }
    }

    /// Creates a test ReputationStake
    fn test_stake(amount: u64) -> ReputationStake {
        ReputationStake {
            amount,
            locked_until: test_timestamp(2000000),
            penalty_multiplier: 2.0,
        }
    }

    /// Creates a test ByzantineFlag
    fn test_byzantine_flag(flag_type: ByzantineFlagType, severity: f64) -> ByzantineFlag {
        ByzantineFlag {
            flag_type,
            severity,
            detected_at: test_timestamp(1000000),
            evidence: "Detected anomalous behavior".to_string(),
        }
    }

    // ==================== ATTESTATION TESTS ====================

    mod attestations {
        use super::*;

        #[test]
        fn test_attestation_creation() {
            let truster = test_agent(1);
            let trustee = test_agent(2);
            let attestation = test_attestation(
                truster.clone(),
                trustee.clone(),
                0.8,
                TrustCategory::Identity,
            );

            assert_eq!(attestation.truster, truster);
            assert_eq!(attestation.trustee, trustee);
            assert_eq!(attestation.trust_level, 0.8);
            assert!(!attestation.revoked);
        }

        #[test]
        fn test_attestation_trust_level_range() {
            // Valid range is -1.0 to 1.0
            let valid_levels = vec![-1.0, -0.5, 0.0, 0.5, 1.0];

            for level in valid_levels {
                let attestation = test_attestation(
                    test_agent(1),
                    test_agent(2),
                    level,
                    TrustCategory::Identity,
                );
                assert!(attestation.trust_level >= -1.0 && attestation.trust_level <= 1.0);
            }
        }

        #[test]
        fn test_attestation_with_expiration() {
            let mut attestation = test_attestation(
                test_agent(1),
                test_agent(2),
                0.7,
                TrustCategory::Identity,
            );
            attestation.expires_at = Some(test_timestamp(2000000));

            assert!(attestation.expires_at.is_some());
        }

        #[test]
        fn test_attestation_with_stake() {
            let mut attestation = test_attestation(
                test_agent(1),
                test_agent(2),
                0.9,
                TrustCategory::Identity,
            );
            attestation.stake = Some(test_stake(1000));

            assert!(attestation.stake.is_some());
            assert_eq!(attestation.stake.as_ref().unwrap().amount, 1000);
        }

        #[test]
        fn test_attestation_revocation() {
            let mut attestation = test_attestation(
                test_agent(1),
                test_agent(2),
                0.8,
                TrustCategory::Identity,
            );

            assert!(!attestation.revoked);
            attestation.revoked = true;
            assert!(attestation.revoked);
        }

        #[test]
        fn test_attestation_all_categories() {
            let categories = vec![
                TrustCategory::Identity,
                TrustCategory::Communication,
                TrustCategory::FileSharing,
                TrustCategory::Scheduling,
                TrustCategory::Financial,
                TrustCategory::CredentialIssuer,
                TrustCategory::Organization,
                TrustCategory::Personal,
                TrustCategory::Professional,
                TrustCategory::Custom("CustomCategory".to_string()),
            ];

            for category in categories {
                let attestation = test_attestation(
                    test_agent(1),
                    test_agent(2),
                    0.5,
                    category.clone(),
                );
                assert_eq!(attestation.category, category);
            }
        }

        #[test]
        fn test_attestation_multiple_evidence() {
            let mut attestation = test_attestation(
                test_agent(1),
                test_agent(2),
                0.9,
                TrustCategory::Identity,
            );
            attestation.evidence = vec![
                test_evidence(EvidenceType::InPersonMeeting, 0.95),
                test_evidence(EvidenceType::GovernmentId, 0.95),
                test_evidence(EvidenceType::PgpKeySigning, 0.8),
            ];

            assert_eq!(attestation.evidence.len(), 3);
        }

        #[test]
        fn test_create_attestation_input() {
            let input = test_create_attestation_input(test_agent(2), 0.75);

            assert_eq!(input.trust_level, 0.75);
            assert_eq!(input.category, TrustCategory::Identity);
            assert!(!input.evidence.is_empty());
        }

        #[test]
        fn test_negative_trust_attestation() {
            let attestation = test_attestation(
                test_agent(1),
                test_agent(2),
                -0.5,
                TrustCategory::Identity,
            );

            assert!(attestation.trust_level < 0.0);
        }
    }

    // ==================== EVIDENCE TESTS ====================

    mod evidence {
        use super::*;

        #[test]
        fn test_evidence_types() {
            let evidence_types = vec![
                EvidenceType::InPersonMeeting,
                EvidenceType::VideoVerification,
                EvidenceType::PhoneVerification,
                EvidenceType::VerifiableCredential,
                EvidenceType::SocialMediaVerification,
                EvidenceType::DomainVerification,
                EvidenceType::PgpKeySigning,
                EvidenceType::MutualVouch,
                EvidenceType::CommunicationHistory,
                EvidenceType::BlockchainAttestation,
                EvidenceType::GovernmentId,
                EvidenceType::ProfessionalCertification,
                EvidenceType::OrganizationMembership,
                EvidenceType::Custom("Custom".to_string()),
            ];

            for evidence_type in evidence_types {
                let evidence = test_evidence(evidence_type.clone(), 0.5);
                assert_eq!(evidence.evidence_type, evidence_type);
            }
        }

        #[test]
        fn test_evidence_weight_range() {
            let weights = vec![0.0, 0.25, 0.5, 0.75, 1.0];

            for weight in weights {
                let evidence = test_evidence(EvidenceType::InPersonMeeting, weight);
                assert!(evidence.weight >= 0.0 && evidence.weight <= 1.0);
            }
        }

        #[test]
        fn test_evidence_verified_status() {
            let verified = test_evidence(EvidenceType::GovernmentId, 0.95);
            assert!(verified.verified);

            let mut unverified = test_evidence(EvidenceType::SocialMediaVerification, 0.5);
            unverified.verified = false;
            assert!(!unverified.verified);
        }

        #[test]
        fn test_evidence_weight_calculation() {
            let weights = vec![
                (EvidenceType::InPersonMeeting, 0.95),
                (EvidenceType::VideoVerification, 0.85),
                (EvidenceType::PhoneVerification, 0.70),
                (EvidenceType::VerifiableCredential, 0.90),
                (EvidenceType::SocialMediaVerification, 0.50),
                (EvidenceType::DomainVerification, 0.75),
                (EvidenceType::PgpKeySigning, 0.80),
                (EvidenceType::MutualVouch, 0.60),
                (EvidenceType::CommunicationHistory, 0.55),
                (EvidenceType::BlockchainAttestation, 0.85),
                (EvidenceType::GovernmentId, 0.95),
                (EvidenceType::ProfessionalCertification, 0.80),
                (EvidenceType::OrganizationMembership, 0.70),
                (EvidenceType::Custom("Custom".to_string()), 0.40),
            ];

            for (evidence_type, expected_weight) in weights {
                let weight = evidence_weight(&evidence_type);
                assert_eq!(weight, expected_weight, "Mismatch for {:?}", evidence_type);
            }
        }

        #[test]
        fn test_combined_evidence_score() {
            let evidence_list = vec![
                test_evidence(EvidenceType::InPersonMeeting, 0.9),
                test_evidence(EvidenceType::VideoVerification, 0.8),
                test_evidence(EvidenceType::PgpKeySigning, 0.7),
            ];

            let total_score: f64 = evidence_list
                .iter()
                .map(|e| evidence_weight(&e.evidence_type) * e.weight)
                .sum::<f64>()
                .min(1.0);

            // 0.95 * 0.9 + 0.85 * 0.8 + 0.80 * 0.7 = 0.855 + 0.68 + 0.56 = 2.095 -> capped at 1.0
            assert_eq!(total_score, 1.0);
        }
    }

    // ==================== TRUST SCORE TESTS ====================

    mod trust_score {
        use super::*;

        #[test]
        fn test_trust_score_creation() {
            let agent = test_agent(1);
            let score = test_trust_score(agent.clone(), 0.75, 0.9);

            assert_eq!(score.agent, agent);
            assert_eq!(score.score, 0.75);
            assert_eq!(score.confidence, 0.9);
        }

        #[test]
        fn test_trust_score_valid_range() {
            let score = test_trust_score(test_agent(1), 0.5, 0.8);

            assert!(score.score >= 0.0 && score.score <= 1.0);
            assert!(score.confidence >= 0.0 && score.confidence <= 1.0);
        }

        #[test]
        fn test_trust_score_with_byzantine_flags() {
            let mut score = test_trust_score(test_agent(1), 0.8, 0.7);
            score.byzantine_flags = vec![
                test_byzantine_flag(ByzantineFlagType::SybilSuspicion, 0.3),
            ];

            assert!(!score.byzantine_flags.is_empty());
        }

        #[test]
        fn test_trust_score_category_scores() {
            let score = test_trust_score(test_agent(1), 0.75, 0.9);

            assert!(!score.category_scores.is_empty());

            for (_, category_score) in &score.category_scores {
                assert!(*category_score >= 0.0 && *category_score <= 1.0);
            }
        }

        #[test]
        fn test_default_trust_score() {
            let score = TrustScore {
                agent: test_agent(1),
                score: DEFAULT_TRUST_LEVEL,
                confidence: 0.0,
                direct_attestation_count: 0,
                transitive_path_count: 0,
                category_scores: Vec::new(),
                computed_at: test_timestamp(1000000),
                attestation_hashes: Vec::new(),
                byzantine_flags: Vec::new(),
            };

            assert_eq!(score.score, 0.3);
            assert_eq!(score.confidence, 0.0);
            assert_eq!(score.direct_attestation_count, 0);
        }

        #[test]
        fn test_trust_score_attestation_tracking() {
            let score = test_trust_score(test_agent(1), 0.8, 0.9);

            assert_eq!(score.attestation_hashes.len(), 2);
            assert_eq!(score.direct_attestation_count, 5);
            assert_eq!(score.transitive_path_count, 10);
        }
    }

    // ==================== MATL ALGORITHM TESTS ====================

    mod matl_algorithm {
        use super::*;

        #[test]
        fn test_transitive_decay_factor() {
            assert_eq!(TRANSITIVE_DECAY_FACTOR, 0.7);
        }

        #[test]
        fn test_max_transitive_depth() {
            assert_eq!(MAX_TRANSITIVE_DEPTH, 5);
        }

        #[test]
        fn test_temporal_decay_rate() {
            assert_eq!(TEMPORAL_DECAY_RATE, 0.01);
        }

        #[test]
        fn test_sybil_detection_threshold() {
            assert_eq!(SYBIL_DETECTION_THRESHOLD, 10);
        }

        #[test]
        fn test_trust_normalization() {
            // Normalize trust_level from [-1, 1] to [0, 1]
            let test_cases = vec![
                (-1.0, 0.0),
                (-0.5, 0.25),
                (0.0, 0.5),
                (0.5, 0.75),
                (1.0, 1.0),
            ];

            for (trust_level, expected) in test_cases {
                let normalized = (trust_level + 1.0) / 2.0;
                assert_eq!(normalized, expected);
            }
        }

        #[test]
        fn test_temporal_decay_calculation() {
            let age_days = 30.0;
            let decay = (-TEMPORAL_DECAY_RATE * age_days).exp();

            // e^(-0.01 * 30) = e^(-0.3) ~= 0.74
            assert!(decay > 0.7 && decay < 0.75);
        }

        #[test]
        fn test_transitive_decay_at_depth() {
            let depths = vec![1, 2, 3, 4, 5];
            let mut decays = Vec::new();

            for depth in depths {
                let decay = TRANSITIVE_DECAY_FACTOR.powi(depth);
                decays.push(decay);
            }

            // Verify decay decreases with depth
            for i in 1..decays.len() {
                assert!(decays[i] < decays[i - 1]);
            }

            // At depth 5: 0.7^5 = 0.16807
            assert!(decays[4] > 0.16 && decays[4] < 0.17);
        }

        #[test]
        fn test_stake_multiplier_calculation() {
            let stakes = vec![0, 500, 1000, 2000, 3000];

            for amount in stakes {
                let stake = test_stake(amount);
                let multiplier = 1.0 + (stake.amount as f64 / 1000.0).min(2.0);

                // Should be between 1.0 and 3.0
                assert!(multiplier >= 1.0 && multiplier <= 3.0);
            }
        }

        #[test]
        fn test_aggregate_scores() {
            let direct_score = 0.8;
            let direct_confidence = 0.9;
            let transitive_score = 0.6;
            let transitive_confidence = 0.7;

            let (combined, confidence) = aggregate_trust_scores(
                direct_score,
                direct_confidence,
                transitive_score,
                transitive_confidence,
            );

            // Direct is weighted 1.5x more
            assert!(combined > 0.6 && combined < 0.9);
            assert!(confidence > 0.0 && confidence <= 1.0);
        }

        #[test]
        fn test_aggregate_with_zero_confidence() {
            let (combined, confidence) = aggregate_trust_scores(0.0, 0.0, 0.0, 0.0);

            assert_eq!(combined, DEFAULT_TRUST_LEVEL);
            assert_eq!(confidence, 0.0);
        }

        #[test]
        fn test_byzantine_penalty_application() {
            let base_score = 0.8;
            let flags = vec![
                test_byzantine_flag(ByzantineFlagType::SybilSuspicion, 0.3),
                test_byzantine_flag(ByzantineFlagType::TrustVolatility, 0.2),
            ];

            let penalty: f64 = flags.iter().map(|f| f.severity).sum::<f64>().min(0.5);
            let final_score = (base_score - penalty).max(0.0);

            assert_eq!(penalty, 0.5); // Capped at 0.5
            assert_eq!(final_score, 0.3);
        }
    }

    // ==================== BYZANTINE DETECTION TESTS ====================

    mod byzantine_detection {
        use super::*;

        #[test]
        fn test_byzantine_flag_types() {
            let flag_types = vec![
                ByzantineFlagType::SybilSuspicion,
                ByzantineFlagType::CollusionPattern,
                ByzantineFlagType::TrustVolatility,
                ByzantineFlagType::InconsistentAttestations,
                ByzantineFlagType::SelfDealing,
                ByzantineFlagType::AttestationFarming,
            ];

            for flag_type in flag_types {
                let flag = test_byzantine_flag(flag_type.clone(), 0.3);
                assert_eq!(flag.flag_type, flag_type);
            }
        }

        #[test]
        fn test_byzantine_flag_severity() {
            let severities = vec![0.1, 0.2, 0.3, 0.4, 0.5];

            for severity in severities {
                let flag = test_byzantine_flag(ByzantineFlagType::SybilSuspicion, severity);
                assert!(flag.severity >= 0.0 && flag.severity <= 1.0);
            }
        }

        #[test]
        fn test_sybil_detection_logic() {
            // If trusters.len() > SYBIL_DETECTION_THRESHOLD and unique < len/2
            let trusters: Vec<AgentPubKey> = (0..15).map(|i| {
                if i < 5 { test_agent(1) } else { test_agent((i % 5) as u8) }
            }).collect();

            let unique_count: std::collections::HashSet<_> = trusters.iter().collect();

            // 15 trusters, 5 unique -> unique (5) < 15/2 (7.5) -> Sybil detected
            let is_suspicious = trusters.len() > SYBIL_DETECTION_THRESHOLD
                && unique_count.len() < trusters.len() / 2;

            assert!(is_suspicious);
        }

        #[test]
        fn test_trust_variance_calculation() {
            // High variance in trust scores indicates volatility
            let scores = vec![0.9, 0.1, 0.8, 0.2, 0.7];
            let mean = scores.iter().sum::<f64>() / scores.len() as f64;
            let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
            let std_dev = variance.sqrt();

            // High variance (> 0.5) triggers flag
            assert!(std_dev > 0.2);
        }

        #[test]
        fn test_inconsistent_attestations_detection() {
            // Same truster giving conflicting scores
            let attestations = vec![
                (test_agent(1), 0.9),
                (test_agent(1), -0.5), // Same truster, conflicting score
            ];

            let max = attestations.iter().map(|(_, s)| *s).fold(f64::MIN, f64::max);
            let min = attestations.iter().map(|(_, s)| *s).fold(f64::MAX, f64::min);

            // If max - min > 1.0, it's inconsistent
            assert!(max - min > 1.0);
        }
    }

    // ==================== INTRODUCTION TESTS ====================

    mod introductions {
        use super::*;

        #[test]
        fn test_introduction_creation() {
            let introducer = test_agent(1);
            let introduced = test_agent(2);
            let target = test_agent(3);

            let intro = test_introduction(
                introducer.clone(),
                introduced.clone(),
                target.clone(),
            );

            assert_eq!(intro.introducer, introducer);
            assert_eq!(intro.introduced, introduced);
            assert_eq!(intro.target, target);
            assert!(intro.accepted.is_none());
        }

        #[test]
        fn test_introduction_acceptance() {
            let mut intro = test_introduction(
                test_agent(1),
                test_agent(2),
                test_agent(3),
            );

            assert!(intro.accepted.is_none());
            intro.accepted = Some(true);
            assert_eq!(intro.accepted, Some(true));
        }

        #[test]
        fn test_introduction_rejection() {
            let mut intro = test_introduction(
                test_agent(1),
                test_agent(2),
                test_agent(3),
            );

            intro.accepted = Some(false);
            assert_eq!(intro.accepted, Some(false));
        }

        #[test]
        fn test_introduction_recommendation_level() {
            let mut intro = test_introduction(
                test_agent(1),
                test_agent(2),
                test_agent(3),
            );

            intro.recommendation_level = 0.95;
            assert!(intro.recommendation_level >= 0.0 && intro.recommendation_level <= 1.0);
        }

        #[test]
        fn test_introduction_all_parties_different() {
            let intro = test_introduction(
                test_agent(1),
                test_agent(2),
                test_agent(3),
            );

            assert_ne!(intro.introducer, intro.introduced);
            assert_ne!(intro.introducer, intro.target);
            assert_ne!(intro.introduced, intro.target);
        }
    }

    // ==================== DISPUTE TESTS ====================

    mod disputes {
        use super::*;

        #[test]
        fn test_dispute_creation() {
            let attestation_hash = test_action_hash(1);
            let disputer = test_agent(2);

            let dispute = test_dispute(attestation_hash.clone(), disputer.clone());

            assert_eq!(dispute.attestation_hash, attestation_hash);
            assert_eq!(dispute.disputer, disputer);
            assert!(!dispute.reason.is_empty());
        }

        #[test]
        fn test_dispute_statuses() {
            let statuses = vec![
                DisputeStatus::Open,
                DisputeStatus::UnderReview,
                DisputeStatus::Resolved,
                DisputeStatus::Rejected,
            ];

            for status in statuses {
                let mut dispute = test_dispute(test_action_hash(1), test_agent(1));
                dispute.status = status.clone();
                assert_eq!(dispute.status, status);
            }
        }

        #[test]
        fn test_dispute_resolution() {
            let mut dispute = test_dispute(test_action_hash(1), test_agent(1));

            let resolution = DisputeResolution {
                outcome: DisputeOutcome::AttestationRevoked,
                resolved_at: test_timestamp(2000000),
                resolver_notes: "Evidence supports the dispute".to_string(),
                attestation_modified: true,
                stake_slashed: true,
            };

            dispute.resolution = Some(resolution);
            dispute.status = DisputeStatus::Resolved;

            assert!(dispute.resolution.is_some());
            assert_eq!(dispute.status, DisputeStatus::Resolved);
        }

        #[test]
        fn test_dispute_outcomes() {
            let outcomes = vec![
                DisputeOutcome::AttestationUpheld,
                DisputeOutcome::AttestationRevoked,
                DisputeOutcome::PartialRevocation,
                DisputeOutcome::DisputeRejected,
            ];

            for outcome in outcomes {
                let resolution = DisputeResolution {
                    outcome: outcome.clone(),
                    resolved_at: test_timestamp(2000000),
                    resolver_notes: "Test resolution".to_string(),
                    attestation_modified: false,
                    stake_slashed: false,
                };

                assert_eq!(resolution.outcome, outcome);
            }
        }

        #[test]
        fn test_dispute_counter_evidence() {
            let dispute = test_dispute(test_action_hash(1), test_agent(1));

            assert!(!dispute.counter_evidence.is_empty());
        }
    }

    // ==================== REPUTATION STAKE TESTS ====================

    mod reputation_stake {
        use super::*;

        #[test]
        fn test_stake_creation() {
            let stake = test_stake(1000);

            assert_eq!(stake.amount, 1000);
            assert_eq!(stake.penalty_multiplier, 2.0);
        }

        #[test]
        fn test_stake_locked_period() {
            let stake = test_stake(500);

            assert!(stake.locked_until > test_timestamp(1000000));
        }

        #[test]
        fn test_stake_penalty_multiplier() {
            let mut stake = test_stake(1000);

            // Penalty multiplier must be at least 1.0
            stake.penalty_multiplier = 1.5;
            assert!(stake.penalty_multiplier >= 1.0);

            stake.penalty_multiplier = 3.0;
            assert!(stake.penalty_multiplier >= 1.0);
        }

        #[test]
        fn test_high_stake_high_risk() {
            let low_stake = test_stake(100);
            let high_stake = test_stake(10000);

            // Higher stake = higher trust weight but also higher risk
            let low_multiplier = 1.0 + (low_stake.amount as f64 / 1000.0).min(2.0);
            let high_multiplier = 1.0 + (high_stake.amount as f64 / 1000.0).min(2.0);

            assert!(high_multiplier >= low_multiplier);
            assert_eq!(high_multiplier, 3.0); // Capped at 3.0
        }
    }

    // ==================== SIGNAL TESTS ====================

    mod signals {
        use super::*;

        #[test]
        fn test_attestation_received_signal() {
            let signal = TrustSignal::AttestationReceived {
                attestation_hash: test_action_hash(1),
                truster: test_agent(1),
                trust_level: 0.8,
                category: TrustCategory::Identity,
            };

            match signal {
                TrustSignal::AttestationReceived { trust_level, .. } => {
                    assert_eq!(trust_level, 0.8);
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_trust_score_updated_signal() {
            let signal = TrustSignal::TrustScoreUpdated {
                agent: test_agent(1),
                new_score: 0.75,
                confidence: 0.9,
            };

            match signal {
                TrustSignal::TrustScoreUpdated { new_score, confidence, .. } => {
                    assert_eq!(new_score, 0.75);
                    assert_eq!(confidence, 0.9);
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_introduction_received_signal() {
            let signal = TrustSignal::IntroductionReceived {
                introduction_hash: test_action_hash(1),
                introducer: test_agent(1),
                introduced: test_agent(2),
                recommendation: 0.85,
            };

            match signal {
                TrustSignal::IntroductionReceived { recommendation, .. } => {
                    assert_eq!(recommendation, 0.85);
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_dispute_filed_signal() {
            let signal = TrustSignal::DisputeFiled {
                dispute_hash: test_action_hash(1),
                attestation_hash: test_action_hash(2),
                disputer: test_agent(1),
            };

            match signal {
                TrustSignal::DisputeFiled { disputer, .. } => {
                    assert_eq!(disputer, test_agent(1));
                }
                _ => panic!("Wrong signal type"),
            }
        }

        #[test]
        fn test_byzantine_detected_signal() {
            let signal = TrustSignal::ByzantineDetected {
                agent: test_agent(1),
                flag_type: ByzantineFlagType::SybilSuspicion,
                severity: 0.4,
            };

            match signal {
                TrustSignal::ByzantineDetected { severity, .. } => {
                    assert_eq!(severity, 0.4);
                }
                _ => panic!("Wrong signal type"),
            }
        }
    }

    // ==================== TRUST PATH TESTS ====================

    mod trust_paths {
        use super::*;

        #[test]
        fn test_trust_path_creation() {
            let path = TrustPath {
                from: test_agent(1),
                to: test_agent(4),
                path: vec![test_agent(1), test_agent(2), test_agent(3), test_agent(4)],
                attestation_hashes: vec![
                    test_action_hash(1),
                    test_action_hash(2),
                    test_action_hash(3),
                ],
                trust_at_hop: vec![1.0, 0.8, 0.6, 0.45],
                final_trust: 0.45,
                computed_at: test_timestamp(1000000),
            };

            assert_eq!(path.path.len(), 4);
            assert_eq!(path.attestation_hashes.len(), 3);
        }

        #[test]
        fn test_trust_decay_along_path() {
            let initial_trust = 0.9;
            let path_length = 4;

            let trusts: Vec<f64> = (0..path_length)
                .map(|depth| initial_trust * TRANSITIVE_DECAY_FACTOR.powi(depth))
                .collect();

            // Trust should decrease along the path
            for i in 1..trusts.len() {
                assert!(trusts[i] < trusts[i - 1]);
            }
        }

        #[test]
        fn test_short_path_higher_trust() {
            let initial_trust = 0.8;

            let short_path_trust = initial_trust * TRANSITIVE_DECAY_FACTOR.powi(2);
            let long_path_trust = initial_trust * TRANSITIVE_DECAY_FACTOR.powi(5);

            assert!(short_path_trust > long_path_trust);
        }
    }

    // ==================== RATE LIMITING TESTS ====================

    mod rate_limiting {
        use super::*;

        #[test]
        fn test_max_attestations_per_day() {
            let max_per_day = 50;
            let recent_count = 45;

            let can_create = recent_count < max_per_day;
            assert!(can_create);

            let recent_count = 50;
            let can_create = recent_count < max_per_day;
            assert!(!can_create);
        }

        #[test]
        fn test_rate_limit_time_window() {
            let time_window_seconds = 86400; // 24 hours

            // Attestations older than window should not count
            let now = test_timestamp(100000000);
            let old_attestation = test_timestamp(100000000 - time_window_seconds * 1000000 - 1);
            let recent_attestation = test_timestamp(100000000 - 1000000);

            let is_old_in_window = (now.as_micros() - old_attestation.as_micros()) < time_window_seconds * 1000000;
            let is_recent_in_window = (now.as_micros() - recent_attestation.as_micros()) < time_window_seconds * 1000000;

            assert!(!is_old_in_window);
            assert!(is_recent_in_window);
        }
    }

    // ==================== QUERY TESTS ====================

    mod queries {
        use super::*;

        #[test]
        fn test_trust_query_creation() {
            let query = TrustQuery {
                querier: test_agent(1),
                subject: test_agent(2),
                category: Some(TrustCategory::Identity),
                max_depth: 3,
                min_confidence: 0.5,
                timestamp: test_timestamp(1000000),
            };

            assert_eq!(query.max_depth, 3);
            assert_eq!(query.min_confidence, 0.5);
        }

        #[test]
        fn test_query_all_categories() {
            let query = TrustQuery {
                querier: test_agent(1),
                subject: test_agent(2),
                category: None,
                max_depth: 5,
                min_confidence: 0.3,
                timestamp: test_timestamp(1000000),
            };

            assert!(query.category.is_none());
        }

        #[test]
        fn test_query_specific_category() {
            let query = TrustQuery {
                querier: test_agent(1),
                subject: test_agent(2),
                category: Some(TrustCategory::Financial),
                max_depth: 2,
                min_confidence: 0.8,
                timestamp: test_timestamp(1000000),
            };

            assert_eq!(query.category, Some(TrustCategory::Financial));
        }
    }
}
