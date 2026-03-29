// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Recognition System
//!
//! Replaces CGC (gift circulation credits) with weighted recognition.
//! Each member can recognize up to 10 others per monthly cycle.
//! Recognition weight = recognizer's MYCEL score × base_weight.
//!
//! Constitutional: Recognition weighted by recognizer's MYCEL (prevents Sybil attacks).

pub use mycelix_finance_types::ContributionType;
use serde::{Deserialize, Serialize};

/// A recognition event from one member to another
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionEvent {
    /// DID of the recognizer
    pub recognizer_did: String,
    /// DID of the recipient
    pub recipient_did: String,
    /// Computed weight: recognizer's MYCEL × base_weight
    pub weight: f64,
    /// Type of contribution being recognized
    pub contribution_type: ContributionType,
    /// Monthly cycle ID (e.g., "2026-02")
    pub cycle_id: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Configuration for the recognition system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionConfig {
    /// Maximum recognitions per member per cycle
    pub max_per_cycle: u32,
    /// Base weight multiplied by recognizer's MYCEL score
    pub base_weight: f64,
    /// Minimum MYCEL to give recognition (apprentices cannot)
    pub min_mycel_to_give: f64,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            max_per_cycle: 10,
            base_weight: 1.0,
            min_mycel_to_give: 0.3, // Apprentices (< 0.3) cannot give recognition
        }
    }
}

/// Calculate the recognition score for a member from received recognitions.
///
/// Returns a score from 0.0 to 1.0.
/// Higher-MYCEL recognizers produce more valuable recognition.
pub fn calculate_recognition_score(
    recognitions: &[RecognitionEvent],
    config: &RecognitionConfig,
) -> f64 {
    if recognitions.is_empty() {
        return 0.0;
    }

    let total_weight: f64 = recognitions.iter().map(|r| r.weight).sum();

    // Normalize: 10 recognitions from max-MYCEL (1.0) members = perfect score
    let max_possible = config.max_per_cycle as f64 * config.base_weight * 1.0;
    let normalized = (total_weight / max_possible).min(1.0);

    // Apply diminishing returns for diversity bonus
    let unique_recognizers: std::collections::HashSet<&str> = recognitions
        .iter()
        .map(|r| r.recognizer_did.as_str())
        .collect();
    let diversity_factor = (unique_recognizers.len() as f64 / config.max_per_cycle as f64).min(1.0);

    // Blend raw weight with diversity (70% weight, 30% diversity)
    (normalized * 0.7 + diversity_factor * 0.3).min(1.0)
}

/// Calculate recognition weight for a specific recognizer
pub fn recognition_weight(recognizer_mycel: f64, config: &RecognitionConfig) -> f64 {
    recognizer_mycel * config.base_weight
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_recognition(
        recognizer: &str,
        recipient: &str,
        mycel: f64,
        cycle: &str,
    ) -> RecognitionEvent {
        let config = RecognitionConfig::default();
        RecognitionEvent {
            recognizer_did: recognizer.to_string(),
            recipient_did: recipient.to_string(),
            weight: recognition_weight(mycel, &config),
            contribution_type: ContributionType::General,
            cycle_id: cycle.to_string(),
            timestamp: 1000,
        }
    }

    #[test]
    fn test_steward_recognition_worth_more() {
        let config = RecognitionConfig::default();
        let steward_weight = recognition_weight(0.7, &config);
        let apprentice_weight = recognition_weight(0.1, &config);

        // Steward's recognition is 7x an apprentice's
        assert!((steward_weight / apprentice_weight - 7.0).abs() < 0.001);
    }

    #[test]
    fn test_recognition_score_empty() {
        let config = RecognitionConfig::default();
        assert_eq!(calculate_recognition_score(&[], &config), 0.0);
    }

    #[test]
    fn test_recognition_score_diverse_recognizers() {
        let config = RecognitionConfig::default();
        let recognitions: Vec<RecognitionEvent> = (0..5)
            .map(|i| {
                make_recognition(
                    &format!("did:test:recognizer_{}", i),
                    "did:test:recipient",
                    0.5,
                    "2026-02",
                )
            })
            .collect();

        let score = calculate_recognition_score(&recognitions, &config);
        // 5 recognitions from 0.5 MYCEL members = moderate score
        assert!(score > 0.1);
        assert!(score < 0.8);
    }

    #[test]
    fn test_recognition_score_caps_at_one() {
        let config = RecognitionConfig::default();
        // 20 recognitions from max-MYCEL members — should cap at 1.0
        let recognitions: Vec<RecognitionEvent> = (0..20)
            .map(|i| {
                make_recognition(
                    &format!("did:test:recognizer_{}", i),
                    "did:test:recipient",
                    1.0,
                    "2026-02",
                )
            })
            .collect();

        let score = calculate_recognition_score(&recognitions, &config);
        assert!((score - 1.0).abs() < 0.001);
    }
}
