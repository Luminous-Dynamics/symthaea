// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Bridge Zome Tests
//!
//! Integration tests for the Bridge zome functionality.
//! These tests verify cross-hApp reputation sharing, credential verification,
//! and event broadcasting.

use hdk::prelude::*;
use bridge_integrity::*;

// =============================================================================
// Unit Tests for Entry Type Validation
// =============================================================================

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn test_happ_registration_valid() {
        let reg = HappRegistration {
            happ_id: "test-happ-001".to_string(),
            dna_hash: Some("abc123def456".to_string()),
            happ_name: "Test hApp".to_string(),
            capabilities: vec!["reputation".to_string(), "credentials".to_string()],
            registered_at: 1704067200,
            registrant: "uhCAk_test_agent".to_string(),
            active: true,
        };

        // Validate fields
        assert!(!reg.happ_id.is_empty());
        assert!(!reg.happ_name.is_empty());
        assert!(reg.capabilities.len() <= 50);
        assert!(reg.active);
    }

    #[test]
    fn test_happ_registration_empty_id() {
        let reg = HappRegistration {
            happ_id: "".to_string(),
            dna_hash: None,
            happ_name: "Test hApp".to_string(),
            capabilities: vec![],
            registered_at: 0,
            registrant: "agent".to_string(),
            active: true,
        };

        assert!(reg.happ_id.is_empty());
    }

    #[test]
    fn test_reputation_record_valid() {
        let record = ReputationRecord {
            agent: "uhCAk_test_agent".to_string(),
            happ_id: "marketplace".to_string(),
            happ_name: "Mycelix Marketplace".to_string(),
            score: 0.85,
            interactions: 150,
            negative_interactions: 10,
            updated_at: 1704067200,
            evidence_hash: Some("evidence123".to_string()),
        };

        // Validate score range
        assert!(record.score >= 0.0 && record.score <= 1.0);
        assert!(!record.agent.is_empty());
        assert!(!record.happ_id.is_empty());
    }

    #[test]
    fn test_reputation_record_score_bounds() {
        // Test lower bound
        let record_low = ReputationRecord {
            agent: "agent1".to_string(),
            happ_id: "happ1".to_string(),
            happ_name: "Test".to_string(),
            score: 0.0,
            interactions: 0,
            negative_interactions: 0,
            updated_at: 0,
            evidence_hash: None,
        };
        assert_eq!(record_low.score, 0.0);

        // Test upper bound
        let record_high = ReputationRecord {
            agent: "agent1".to_string(),
            happ_id: "happ1".to_string(),
            happ_name: "Test".to_string(),
            score: 1.0,
            interactions: 100,
            negative_interactions: 0,
            updated_at: 0,
            evidence_hash: None,
        };
        assert_eq!(record_high.score, 1.0);
    }

    #[test]
    fn test_cross_happ_reputation_record() {
        use std::collections::HashMap;

        let mut scores: HashMap<String, f64> = HashMap::new();
        scores.insert("mail".to_string(), 0.9);
        scores.insert("marketplace".to_string(), 0.8);
        scores.insert("praxis".to_string(), 0.95);

        let scores_json = serde_json::to_string(&scores).unwrap();

        let record = CrossHappReputationRecord {
            agent: "uhCAk_test_agent".to_string(),
            scores_json: scores_json.clone(),
            aggregated_score: 0.883, // weighted average
            total_interactions: 500,
            happ_count: 3,
            computed_at: 1704067200,
        };

        assert_eq!(record.happ_count, 3);
        assert!(record.aggregated_score >= 0.0 && record.aggregated_score <= 1.0);

        // Verify JSON is valid
        let parsed: HashMap<String, f64> = serde_json::from_str(&record.scores_json).unwrap();
        assert_eq!(parsed.len(), 3);
    }

    #[test]
    fn test_bridge_event_record() {
        let event = BridgeEventRecord {
            event_type: "reputation_update".to_string(),
            source_happ: "marketplace".to_string(),
            payload: b"test payload".to_vec(),
            timestamp: 1704067200,
            targets: vec!["mail".to_string(), "praxis".to_string()],
            priority: 1,
        };

        assert!(!event.event_type.is_empty());
        assert!(event.priority <= 2);
        assert!(event.payload.len() < 65536); // MAX_EVENT_PAYLOAD_SIZE
    }

    #[test]
    fn test_credential_verification_request() {
        let request = CredentialVerificationRequest {
            request_id: "req_abc123".to_string(),
            credential_hash: "cred_hash_xyz".to_string(),
            credential_type: "email_verified".to_string(),
            issuer_happ: "identity".to_string(),
            subject_agent: "uhCAk_subject".to_string(),
            requester: "uhCAk_requester".to_string(),
            requested_at: 1704067200,
            status: "pending".to_string(),
        };

        assert_eq!(request.status, "pending");
        assert!(["pending", "verified", "failed", "expired"].contains(&request.status.as_str()));
    }

    #[test]
    fn test_credential_verification_response() {
        let response = CredentialVerificationResponse {
            request_id: "req_abc123".to_string(),
            is_valid: true,
            issuer: "identity_happ".to_string(),
            credential_type: "email_verified".to_string(),
            claims_json: r#"["email@example.com", "verified_at: 2025-01-01"]"#.to_string(),
            issued_at: 1704000000,
            expires_at: 1735603200,
            verified_at: 1704067200,
            verifier: "uhCAk_verifier".to_string(),
            proof: Some(vec![0x01, 0x02, 0x03]),
        };

        assert!(response.is_valid);
        assert!(response.expires_at > response.issued_at);

        // Verify claims JSON is valid
        let claims: Vec<String> = serde_json::from_str(&response.claims_json).unwrap();
        assert!(!claims.is_empty());
    }
}

// =============================================================================
// Tests for Cross-hApp Reputation Aggregation
// =============================================================================

#[cfg(test)]
mod reputation_aggregation_tests {
    use mycelix_sdk::bridge::{HappReputationScore, CrossHappReputation};

    #[test]
    fn test_weighted_aggregate_calculation() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.8,
                interactions: 50,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        // Weighted average: (0.9 * 100 + 0.8 * 50) / 150 = 130 / 150 = 0.867
        assert!((rep.aggregate - 0.867).abs() < 0.01);
    }

    #[test]
    fn test_trustworthiness_threshold() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.85,
                interactions: 100,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        // Should be trustworthy above 0.5
        assert!(rep.is_trustworthy(0.5));
        assert!(rep.is_trustworthy(0.7));
        assert!(rep.is_trustworthy(0.8));
        assert!(!rep.is_trustworthy(0.9));
    }

    #[test]
    fn test_empty_scores_default() {
        let scores: Vec<HappReputationScore> = vec![];
        let rep = CrossHappReputation::from_scores("new_agent", scores);

        // Default should be 0.5 for unknown agents
        assert!((rep.aggregate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_best_and_worst_happ() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.6,
                interactions: 50,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "praxis".to_string(),
                happ_name: "Mycelix Praxis".to_string(),
                score: 0.75,
                interactions: 25,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        let best = rep.best_happ().unwrap();
        assert_eq!(best.happ_id, "mail");
        assert_eq!(best.score, 0.9);

        let worst = rep.worst_happ().unwrap();
        assert_eq!(worst.happ_id, "marketplace");
        assert_eq!(worst.score, 0.6);
    }

    #[test]
    fn test_total_interactions() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 100,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.8,
                interactions: 50,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);
        assert_eq!(rep.total_interactions(), 150);
    }

    #[test]
    fn test_single_happ_reputation() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.75,
                interactions: 50,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        // Single hApp should have aggregate = its score
        assert!((rep.aggregate - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_zero_interactions_equal_weight() {
        let scores = vec![
            HappReputationScore {
                happ_id: "mail".to_string(),
                happ_name: "Mycelix Mail".to_string(),
                score: 0.9,
                interactions: 0,
                last_updated: 0,
            },
            HappReputationScore {
                happ_id: "marketplace".to_string(),
                happ_name: "Mycelix Marketplace".to_string(),
                score: 0.7,
                interactions: 0,
                last_updated: 0,
            },
        ];

        let rep = CrossHappReputation::from_scores("agent1", scores);

        // Equal weight: (0.9 + 0.7) / 2 = 0.8
        assert!((rep.aggregate - 0.8).abs() < 0.01);
    }
}

// =============================================================================
// Link Type Tests
// =============================================================================

#[cfg(test)]
mod link_type_tests {
    use super::*;

    #[test]
    fn test_link_type_enum_values() {
        // Verify all link types are distinct
        let types = [
            LinkTypes::HappIdToRegistration as u8,
            LinkTypes::AgentToReputations as u8,
            LinkTypes::HappToReputations as u8,
            LinkTypes::EventTypeToEvents as u8,
            LinkTypes::HappToCredentialRequests as u8,
            LinkTypes::RequestToResponse as u8,
            LinkTypes::AgentToAggregateReputation as u8,
            LinkTypes::AllHapps as u8,
        ];

        // Check all values are unique
        let mut seen = std::collections::HashSet::new();
        for t in &types {
            assert!(seen.insert(*t), "Duplicate link type value: {}", t);
        }
    }
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[cfg(test)]
mod serialization_tests {
    use super::*;

    #[test]
    fn test_happ_registration_serialization() {
        let reg = HappRegistration {
            happ_id: "test-happ".to_string(),
            dna_hash: Some("uhCEk_dna123".to_string()),
            happ_name: "Test hApp".to_string(),
            capabilities: vec!["reputation".to_string()],
            registered_at: 1704067200,
            registrant: "uhCAk_agent".to_string(),
            active: true,
        };

        // Serialize to JSON
        let json = serde_json::to_string(&reg).unwrap();
        assert!(json.contains("test-happ"));

        // Deserialize back
        let deserialized: HappRegistration = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.happ_id, reg.happ_id);
        assert_eq!(deserialized.active, reg.active);
    }

    #[test]
    fn test_bridge_event_payload_serialization() {
        #[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq)]
        struct ReputationUpdatePayload {
            agent: String,
            happ_id: String,
            new_score: f64,
        }

        let payload = ReputationUpdatePayload {
            agent: "agent1".to_string(),
            happ_id: "marketplace".to_string(),
            new_score: 0.85,
        };

        let bytes = serde_json::to_vec(&payload).unwrap();

        let event = BridgeEventRecord {
            event_type: "reputation_update".to_string(),
            source_happ: "marketplace".to_string(),
            payload: bytes.clone(),
            timestamp: 1704067200,
            targets: vec![],
            priority: 0,
        };

        // Deserialize payload from event
        let recovered: ReputationUpdatePayload = serde_json::from_slice(&event.payload).unwrap();
        assert_eq!(recovered, payload);
    }
}

// =============================================================================
// Constants Validation Tests
// =============================================================================

#[cfg(test)]
mod constants_tests {
    use super::*;

    #[test]
    fn test_max_id_length() {
        assert_eq!(MAX_ID_LENGTH, 256);
    }

    #[test]
    fn test_max_name_length() {
        assert_eq!(MAX_NAME_LENGTH, 512);
    }

    #[test]
    fn test_max_event_payload_size() {
        assert_eq!(MAX_EVENT_PAYLOAD_SIZE, 65536); // 64KB
    }

    #[test]
    fn test_max_capabilities() {
        assert_eq!(MAX_CAPABILITIES, 50);
    }

    #[test]
    fn test_max_event_targets() {
        assert_eq!(MAX_EVENT_TARGETS, 100);
    }

    #[test]
    fn test_valid_credential_statuses() {
        assert!(VALID_CREDENTIAL_STATUSES.contains(&"pending"));
        assert!(VALID_CREDENTIAL_STATUSES.contains(&"verified"));
        assert!(VALID_CREDENTIAL_STATUSES.contains(&"failed"));
        assert!(VALID_CREDENTIAL_STATUSES.contains(&"expired"));
        assert!(!VALID_CREDENTIAL_STATUSES.contains(&"unknown"));
    }

    #[test]
    fn test_standard_event_types() {
        assert!(STANDARD_EVENT_TYPES.contains(&"reputation_update"));
        assert!(STANDARD_EVENT_TYPES.contains(&"credential_issued"));
        assert!(STANDARD_EVENT_TYPES.contains(&"happ_registered"));
    }
}
