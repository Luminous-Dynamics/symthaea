// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducibility Tracking System
//!
//! Tracks replication attempts for scientific claims.
//! Implements reproducibility scoring per Epistemic Charter §4.
//! Enables the scientific community to collectively verify claims.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Status of a replication attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationStatus {
    /// Attempt is planned/registered
    Registered,
    /// Currently in progress
    InProgress,
    /// Completed with results
    Completed,
    /// Abandoned (with reason)
    Abandoned,
    /// Failed due to methodology issues
    MethodologyFailed,
}

/// Outcome of a completed replication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplicationOutcome {
    /// Full replication - same results within error margin
    FullReplication,
    /// Partial replication - some findings confirmed
    PartialReplication,
    /// Failure to replicate - contradictory results
    FailureToReplicate,
    /// Inconclusive - unable to determine
    Inconclusive,
    /// Not applicable - claim type doesn't support replication
    NotApplicable,
}

impl ReplicationOutcome {
    /// Weight factor for reproducibility scoring
    pub fn weight(&self) -> f64 {
        match self {
            Self::FullReplication => 1.0,
            Self::PartialReplication => 0.5,
            Self::FailureToReplicate => -0.5,
            Self::Inconclusive => 0.0,
            Self::NotApplicable => 0.0,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::FullReplication => "Results fully replicated",
            Self::PartialReplication => "Some findings confirmed",
            Self::FailureToReplicate => "Unable to reproduce results",
            Self::Inconclusive => "Results inconclusive",
            Self::NotApplicable => "Replication not applicable",
        }
    }
}

/// Methodology match level between original and replication
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MethodologyMatch {
    /// Exact replication (same methods, materials, procedures)
    Exact,
    /// Close replication (minor variations)
    Close,
    /// Conceptual replication (same concept, different methods)
    Conceptual,
    /// Extension (replication with additions)
    Extension,
}

impl MethodologyMatch {
    /// Relevance multiplier for scoring
    pub fn relevance(&self) -> f64 {
        match self {
            Self::Exact => 1.0,
            Self::Close => 0.9,
            Self::Conceptual => 0.7,
            Self::Extension => 0.8,
        }
    }
}

/// A replication attempt record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationAttempt {
    /// Unique ID
    pub id: Uuid,
    /// ID of the claim being replicated
    pub claim_id: Uuid,
    /// Who is conducting the replication
    pub replicator_id: String,
    /// Institution or organization
    pub institution: Option<String>,
    /// Status of the attempt
    pub status: ReplicationStatus,
    /// Outcome (if completed)
    pub outcome: Option<ReplicationOutcome>,
    /// How closely methodology matches original
    pub methodology_match: MethodologyMatch,
    /// Sample size used
    pub sample_size: Option<usize>,
    /// Statistical power achieved
    pub statistical_power: Option<f64>,
    /// Effect size observed (vs original)
    pub effect_size_ratio: Option<f64>,
    /// Detailed notes
    pub notes: String,
    /// Reference to published preregistration
    pub preregistration_ref: Option<String>,
    /// Reference to published results
    pub publication_ref: Option<String>,
    /// Data/code availability reference
    pub data_ref: Option<String>,
    /// When registered
    pub registered_at: DateTime<Utc>,
    /// When completed
    pub completed_at: Option<DateTime<Utc>>,
}

impl ReplicationAttempt {
    /// Create a new registered replication attempt
    pub fn register(
        claim_id: Uuid,
        replicator_id: String,
        methodology_match: MethodologyMatch,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            claim_id,
            replicator_id,
            institution: None,
            status: ReplicationStatus::Registered,
            outcome: None,
            methodology_match,
            sample_size: None,
            statistical_power: None,
            effect_size_ratio: None,
            notes: String::new(),
            preregistration_ref: None,
            publication_ref: None,
            data_ref: None,
            registered_at: Utc::now(),
            completed_at: None,
        }
    }

    /// Mark as in progress
    pub fn start(&mut self) {
        self.status = ReplicationStatus::InProgress;
    }

    /// Complete the replication with outcome
    pub fn complete(&mut self, outcome: ReplicationOutcome, effect_size_ratio: Option<f64>) {
        self.status = ReplicationStatus::Completed;
        self.outcome = Some(outcome);
        self.effect_size_ratio = effect_size_ratio;
        self.completed_at = Some(Utc::now());
    }

    /// Abandon the attempt
    pub fn abandon(&mut self, reason: &str) {
        self.status = ReplicationStatus::Abandoned;
        self.notes = format!("Abandoned: {}", reason);
        self.completed_at = Some(Utc::now());
    }

    /// Calculate quality score for this attempt
    pub fn quality_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // Preregistration bonus
        if self.preregistration_ref.is_some() {
            score += 1.0;
            factors += 1;
        }

        // Data availability bonus
        if self.data_ref.is_some() {
            score += 1.0;
            factors += 1;
        }

        // Publication bonus
        if self.publication_ref.is_some() {
            score += 0.8;
            factors += 1;
        }

        // Sample size adequacy (assuming 100 is adequate)
        if let Some(n) = self.sample_size {
            score += (n as f64 / 100.0).min(1.0);
            factors += 1;
        }

        // Statistical power adequacy (0.8 is standard threshold)
        if let Some(power) = self.statistical_power {
            score += (power / 0.8).min(1.0);
            factors += 1;
        }

        if factors == 0 {
            0.5 // Default moderate quality
        } else {
            score / factors as f64
        }
    }
}

/// Aggregated reproducibility statistics for a claim
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityStats {
    /// ID of the claim
    pub claim_id: Uuid,
    /// Total replication attempts
    pub total_attempts: usize,
    /// Completed attempts
    pub completed_attempts: usize,
    /// Full replications
    pub full_replications: usize,
    /// Partial replications
    pub partial_replications: usize,
    /// Failed replications
    pub failed_replications: usize,
    /// Inconclusive attempts
    pub inconclusive_attempts: usize,
    /// Overall reproducibility score (0.0-1.0)
    pub reproducibility_score: f64,
    /// Confidence in the score (based on sample size)
    pub score_confidence: f64,
    /// Average effect size ratio across replications
    pub average_effect_size_ratio: Option<f64>,
    /// Last updated
    pub updated_at: DateTime<Utc>,
}

impl ReproducibilityStats {
    pub fn new(claim_id: Uuid) -> Self {
        Self {
            claim_id,
            total_attempts: 0,
            completed_attempts: 0,
            full_replications: 0,
            partial_replications: 0,
            failed_replications: 0,
            inconclusive_attempts: 0,
            reproducibility_score: 0.5, // Prior: uncertain
            score_confidence: 0.0,
            average_effect_size_ratio: None,
            updated_at: Utc::now(),
        }
    }

    /// Recalculate from attempts
    pub fn recalculate(&mut self, attempts: &[ReplicationAttempt]) {
        self.total_attempts = attempts.len();
        self.completed_attempts = attempts
            .iter()
            .filter(|a| a.status == ReplicationStatus::Completed)
            .count();

        self.full_replications = 0;
        self.partial_replications = 0;
        self.failed_replications = 0;
        self.inconclusive_attempts = 0;

        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        let mut effect_sizes = Vec::new();

        for attempt in attempts {
            if let Some(outcome) = attempt.outcome {
                let quality = attempt.quality_score();
                let relevance = attempt.methodology_match.relevance();
                let weight = quality * relevance;

                weighted_sum += outcome.weight() * weight;
                weight_total += weight;

                match outcome {
                    ReplicationOutcome::FullReplication => self.full_replications += 1,
                    ReplicationOutcome::PartialReplication => self.partial_replications += 1,
                    ReplicationOutcome::FailureToReplicate => self.failed_replications += 1,
                    ReplicationOutcome::Inconclusive => self.inconclusive_attempts += 1,
                    ReplicationOutcome::NotApplicable => {}
                }

                if let Some(ratio) = attempt.effect_size_ratio {
                    effect_sizes.push(ratio);
                }
            }
        }

        // Calculate reproducibility score with Bayesian updating
        if weight_total > 0.0 {
            // Normalize to 0-1 range (raw is -0.5 to 1.0)
            let raw_score = weighted_sum / weight_total;
            self.reproducibility_score = (raw_score + 0.5) / 1.5;
        }

        // Confidence based on number of completed attempts
        self.score_confidence = 1.0 - (1.0 / (self.completed_attempts as f64 + 1.0));

        // Average effect size ratio
        if !effect_sizes.is_empty() {
            self.average_effect_size_ratio =
                Some(effect_sizes.iter().sum::<f64>() / effect_sizes.len() as f64);
        }

        self.updated_at = Utc::now();
    }

    /// Get human-readable reproducibility status
    pub fn status_summary(&self) -> &'static str {
        if self.completed_attempts == 0 {
            return "Not yet replicated";
        }

        if self.reproducibility_score >= 0.8 {
            "Highly reproducible"
        } else if self.reproducibility_score >= 0.6 {
            "Generally reproducible"
        } else if self.reproducibility_score >= 0.4 {
            "Mixed reproducibility"
        } else if self.reproducibility_score >= 0.2 {
            "Low reproducibility"
        } else {
            "Reproducibility crisis"
        }
    }
}

/// Reproducibility registry for tracking all attempts
#[derive(Debug, Clone, Default)]
pub struct ReproducibilityRegistry {
    /// All registered attempts
    attempts: Vec<ReplicationAttempt>,
    /// Cached stats per claim
    stats_cache: std::collections::HashMap<Uuid, ReproducibilityStats>,
}

impl ReproducibilityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new replication attempt
    pub fn register_attempt(&mut self, attempt: ReplicationAttempt) {
        let claim_id = attempt.claim_id;
        self.attempts.push(attempt);
        self.invalidate_cache(claim_id);
    }

    /// Get all attempts for a claim
    pub fn get_attempts(&self, claim_id: Uuid) -> Vec<&ReplicationAttempt> {
        self.attempts
            .iter()
            .filter(|a| a.claim_id == claim_id)
            .collect()
    }

    /// Get mutable reference to an attempt
    pub fn get_attempt_mut(&mut self, attempt_id: Uuid) -> Option<&mut ReplicationAttempt> {
        self.attempts.iter_mut().find(|a| a.id == attempt_id)
    }

    /// Get stats for a claim
    pub fn get_stats(&mut self, claim_id: Uuid) -> &ReproducibilityStats {
        if !self.stats_cache.contains_key(&claim_id) {
            let mut stats = ReproducibilityStats::new(claim_id);
            let attempts: Vec<_> = self
                .attempts
                .iter()
                .filter(|a| a.claim_id == claim_id)
                .cloned()
                .collect();
            stats.recalculate(&attempts);
            self.stats_cache.insert(claim_id, stats);
        }
        // Safe: we just inserted if not present, so get() will always succeed
        self.stats_cache.get(&claim_id).expect("Stats must exist after insertion")
    }

    /// Invalidate cached stats for a claim
    fn invalidate_cache(&mut self, claim_id: Uuid) {
        self.stats_cache.remove(&claim_id);
    }

    /// Complete an attempt
    pub fn complete_attempt(
        &mut self,
        attempt_id: Uuid,
        outcome: ReplicationOutcome,
        effect_size_ratio: Option<f64>,
    ) -> bool {
        if let Some(attempt) = self.get_attempt_mut(attempt_id) {
            let claim_id = attempt.claim_id;
            attempt.complete(outcome, effect_size_ratio);
            self.invalidate_cache(claim_id);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_attempt() {
        let claim_id = Uuid::new_v4();
        let attempt = ReplicationAttempt::register(
            claim_id,
            "researcher@lab.edu".to_string(),
            MethodologyMatch::Exact,
        );

        assert_eq!(attempt.claim_id, claim_id);
        assert_eq!(attempt.status, ReplicationStatus::Registered);
        assert!(attempt.outcome.is_none());
    }

    #[test]
    fn test_complete_attempt() {
        let mut attempt = ReplicationAttempt::register(
            Uuid::new_v4(),
            "researcher@lab.edu".to_string(),
            MethodologyMatch::Close,
        );

        attempt.start();
        assert_eq!(attempt.status, ReplicationStatus::InProgress);

        attempt.complete(ReplicationOutcome::FullReplication, Some(0.95));
        assert_eq!(attempt.status, ReplicationStatus::Completed);
        assert_eq!(attempt.outcome, Some(ReplicationOutcome::FullReplication));
        assert_eq!(attempt.effect_size_ratio, Some(0.95));
    }

    #[test]
    fn test_quality_score() {
        let mut attempt = ReplicationAttempt::register(
            Uuid::new_v4(),
            "researcher@lab.edu".to_string(),
            MethodologyMatch::Exact,
        );

        // Low quality (no metadata)
        let low_score = attempt.quality_score();
        assert_eq!(low_score, 0.5);

        // Add quality indicators
        attempt.preregistration_ref = Some("osf.io/12345".to_string());
        attempt.data_ref = Some("zenodo.org/123".to_string());
        attempt.sample_size = Some(200);
        attempt.statistical_power = Some(0.9);

        let high_score = attempt.quality_score();
        assert!(high_score > 0.9);
    }

    #[test]
    fn test_reproducibility_stats() {
        let claim_id = Uuid::new_v4();

        let mut attempt1 = ReplicationAttempt::register(
            claim_id,
            "lab1@uni.edu".to_string(),
            MethodologyMatch::Exact,
        );
        attempt1.complete(ReplicationOutcome::FullReplication, Some(0.98));

        let mut attempt2 = ReplicationAttempt::register(
            claim_id,
            "lab2@uni.edu".to_string(),
            MethodologyMatch::Close,
        );
        attempt2.complete(ReplicationOutcome::PartialReplication, Some(0.75));

        let mut stats = ReproducibilityStats::new(claim_id);
        stats.recalculate(&[attempt1, attempt2]);

        assert_eq!(stats.completed_attempts, 2);
        assert_eq!(stats.full_replications, 1);
        assert_eq!(stats.partial_replications, 1);
        assert!(stats.reproducibility_score > 0.5);
        assert!(stats.score_confidence > 0.0);
    }

    #[test]
    fn test_registry() {
        let mut registry = ReproducibilityRegistry::new();
        let claim_id = Uuid::new_v4();

        let attempt = ReplicationAttempt::register(
            claim_id,
            "researcher@lab.edu".to_string(),
            MethodologyMatch::Exact,
        );
        let attempt_id = attempt.id;

        registry.register_attempt(attempt);

        // Complete the attempt
        registry.complete_attempt(attempt_id, ReplicationOutcome::FullReplication, Some(1.0));

        // Check stats
        let stats = registry.get_stats(claim_id);
        assert_eq!(stats.full_replications, 1);
        assert!(stats.reproducibility_score > 0.8);
    }

    #[test]
    fn test_outcome_weights() {
        assert_eq!(ReplicationOutcome::FullReplication.weight(), 1.0);
        assert_eq!(ReplicationOutcome::PartialReplication.weight(), 0.5);
        assert_eq!(ReplicationOutcome::FailureToReplicate.weight(), -0.5);
        assert_eq!(ReplicationOutcome::Inconclusive.weight(), 0.0);
    }
}
