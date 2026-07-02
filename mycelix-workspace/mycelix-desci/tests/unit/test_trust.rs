// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Comprehensive tests for the trust module

use mycelix_desci_core::trust::{TrustManager, TrustScore};

#[test]
fn test_default_trust_score() {
    let score = TrustScore::default();

    assert_eq!(score.score, 0.5);  // Neutral start
    assert_eq!(score.confidence, 0.1);
    assert_eq!(score.interaction_count, 0);
}

#[test]
fn test_create_trust_manager() {
    let manager = TrustManager::new();

    // Check default participant has default score
    let score = manager.get_score("new_participant");
    assert_eq!(score.score, 0.5);
    assert_eq!(score.interaction_count, 0);
}

#[test]
fn test_update_score_positive() {
    let mut manager = TrustManager::new();

    let result = manager.update_score("participant_1", true, 1.0);
    assert!(result.is_ok());

    let score = manager.get_score("participant_1");
    assert!(score.score > 0.5);  // Should increase from neutral
    assert_eq!(score.interaction_count, 1);
}

#[test]
fn test_update_score_negative() {
    let mut manager = TrustManager::new();

    let result = manager.update_score("participant_1", false, 1.0);
    assert!(result.is_ok());

    let score = manager.get_score("participant_1");
    assert!(score.score < 0.5);  // Should decrease from neutral
    assert_eq!(score.interaction_count, 1);
}

#[test]
fn test_multiple_positive_interactions() {
    let mut manager = TrustManager::new();

    for _ in 0..5 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    let score = manager.get_score("participant_1");
    assert!(score.score > 0.5);
    assert_eq!(score.interaction_count, 5);
    assert!(score.confidence > 0.1);  // Confidence should increase
}

#[test]
fn test_mixed_interactions() {
    let mut manager = TrustManager::new();

    // 3 positive
    for _ in 0..3 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    // 2 negative
    for _ in 0..2 {
        manager.update_score("participant_1", false, 1.0).unwrap();
    }

    let score = manager.get_score("participant_1");
    assert!(score.score > 0.5);  // Should still be net positive
    assert_eq!(score.interaction_count, 5);
}

#[test]
fn test_score_bounds() {
    let mut manager = TrustManager::new();

    // Try to push score above 1.0
    for _ in 0..100 {
        manager.update_score("participant_1", true, 10.0).unwrap();
    }

    let score = manager.get_score("participant_1");
    assert!(score.score <= 1.0);  // Should be clamped
    assert!(score.score >= 0.0);

    // Try to push score below 0.0
    for _ in 0..200 {
        manager.update_score("participant_1", false, 10.0).unwrap();
    }

    let score = manager.get_score("participant_1");
    assert!(score.score <= 1.0);
    assert!(score.score >= 0.0);  // Should be clamped
}

#[test]
fn test_confidence_increases_with_interactions() {
    let mut manager = TrustManager::new();

    let initial_confidence = manager.get_score("participant_1").confidence;

    for _ in 0..50 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    let final_confidence = manager.get_score("participant_1").confidence;

    assert!(final_confidence > initial_confidence);
}

#[test]
fn test_confidence_capped_at_one() {
    let mut manager = TrustManager::new();

    // Lots of interactions
    for _ in 0..200 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    let score = manager.get_score("participant_1");
    assert!(score.confidence <= 1.0);
}

#[test]
fn test_is_trusted_threshold() {
    let mut manager = TrustManager::new();
    manager.min_trust = 0.6;

    // Initially not trusted (score = 0.5)
    assert!(!manager.is_trusted("participant_1"));

    // Increase score above threshold
    for _ in 0..5 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    assert!(manager.is_trusted("participant_1"));
}

#[test]
fn test_is_trusted_boundary() {
    let mut manager = TrustManager::new();
    manager.min_trust = 0.5;

    // At exactly threshold
    let score = manager.get_score("new_participant");
    assert_eq!(score.score, 0.5);
    assert!(manager.is_trusted("new_participant"));  // >= threshold
}

#[test]
fn test_decay_toward_neutral() {
    let mut manager = TrustManager::new();

    // Build up high trust
    for _ in 0..10 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    let high_score = manager.get_score("participant_1").score;
    assert!(high_score > 0.5);

    // Apply decay
    manager.apply_decay();

    let decayed_score = manager.get_score("participant_1").score;

    // Should move toward 0.5 (neutral)
    assert!(decayed_score < high_score);
    assert!(decayed_score > 0.5);
}

#[test]
fn test_decay_multiple_times() {
    let mut manager = TrustManager::new();

    // Build high trust
    for _ in 0..10 {
        manager.update_score("participant_1", true, 1.0).unwrap();
    }

    let initial_score = manager.get_score("participant_1").score;

    // Apply decay multiple times
    for _ in 0..10 {
        manager.apply_decay();
    }

    let final_score = manager.get_score("participant_1").score;

    // Should be closer to neutral
    assert!(final_score < initial_score);
    assert!((final_score - 0.5).abs() < (initial_score - 0.5).abs());
}

#[test]
fn test_multiple_participants() {
    let mut manager = TrustManager::new();

    manager.update_score("p1", true, 1.0).unwrap();
    manager.update_score("p2", false, 1.0).unwrap();
    manager.update_score("p3", true, 2.0).unwrap();

    let score1 = manager.get_score("p1");
    let score2 = manager.get_score("p2");
    let score3 = manager.get_score("p3");

    assert!(score1.score > 0.5);
    assert!(score2.score < 0.5);
    assert!(score3.score > score1.score);  // Higher weight
}

#[test]
fn test_weight_affects_score_change() {
    let mut manager = TrustManager::new();

    manager.update_score("p1", true, 0.5).unwrap();
    manager.update_score("p2", true, 2.0).unwrap();

    let score1 = manager.get_score("p1");
    let score2 = manager.get_score("p2");

    // Higher weight should have more impact
    assert!(score2.score > score1.score);
}

#[test]
fn test_decay_affects_all_participants() {
    let mut manager = TrustManager::new();

    // Create several participants with different scores
    manager.update_score("p1", true, 5.0).unwrap();
    manager.update_score("p2", true, 3.0).unwrap();
    manager.update_score("p3", false, 3.0).unwrap();

    let before1 = manager.get_score("p1").score;
    let before2 = manager.get_score("p2").score;
    let before3 = manager.get_score("p3").score;

    manager.apply_decay();

    let after1 = manager.get_score("p1").score;
    let after2 = manager.get_score("p2").score;
    let after3 = manager.get_score("p3").score;

    // All should move toward neutral
    assert!(after1 < before1);
    assert!(after2 < before2);
    assert!(after3 > before3);  // Below neutral, so increases
}

#[test]
fn test_interaction_count_persists() {
    let mut manager = TrustManager::new();

    for i in 0..5 {
        manager.update_score("participant_1", true, 1.0).unwrap();
        let score = manager.get_score("participant_1");
        assert_eq!(score.interaction_count, i + 1);
    }
}

#[test]
fn test_zero_weight_update() {
    let mut manager = TrustManager::new();

    manager.update_score("participant_1", true, 0.0).unwrap();

    let score = manager.get_score("participant_1");
    // Score should not change much with zero weight
    assert!((score.score - 0.5).abs() < 0.01);
    assert_eq!(score.interaction_count, 1);  // But count still increments
}

#[test]
fn test_trust_manager_default() {
    let manager = TrustManager::default();

    assert!(manager.get_score("anyone").score == 0.5);
}

#[test]
fn test_alternating_positive_negative() {
    let mut manager = TrustManager::new();

    for i in 0..10 {
        let positive = i % 2 == 0;
        manager.update_score("participant_1", positive, 1.0).unwrap();
    }

    let score = manager.get_score("participant_1");
    // Should end up close to neutral
    assert!((score.score - 0.5).abs() < 0.2);
    assert_eq!(score.interaction_count, 10);
}

#[test]
fn test_recovery_from_low_trust() {
    let mut manager = TrustManager::new();

    // Tank the trust
    for _ in 0..5 {
        manager.update_score("participant_1", false, 2.0).unwrap();
    }

    let low_score = manager.get_score("participant_1").score;
    assert!(low_score < 0.5);

    // Recover with good behavior
    for _ in 0..10 {
        manager.update_score("participant_1", true, 2.0).unwrap();
    }

    let recovered_score = manager.get_score("participant_1").score;
    assert!(recovered_score > low_score);
    assert!(recovered_score > 0.5);
}
