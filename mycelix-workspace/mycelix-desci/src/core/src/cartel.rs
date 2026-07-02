// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cartel Detection System
//!
//! Implements anti-collusion mechanisms for the MATL trust system.
//! Detects coordinated verification behavior that could game
//! reputation scores through statistical analysis.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Represents a verification event for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationEvent {
    /// ID of the verifier
    pub verifier_id: String,
    /// ID of the claim verified
    pub claim_id: Uuid,
    /// ID of the claim author
    pub author_id: String,
    /// Timestamp of verification
    pub timestamp: DateTime<Utc>,
    /// Verification outcome (true = confirmed, false = rejected)
    pub confirmed: bool,
}

/// Detected pattern of suspicious behavior
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CartelPattern {
    /// Mutual verification ring (A verifies B, B verifies A)
    MutualVerification,
    /// Synchronized timing (multiple verifications within short window)
    SynchronizedTiming,
    /// Exclusive clique (group only verifies each other)
    ExclusiveClique,
    /// Coordinated voting (same vote patterns across claims)
    CoordinatedVoting,
    /// Velocity anomaly (abnormal verification frequency)
    VelocityAnomaly,
    /// Geographic clustering (should be distributed)
    GeographicClustering,
}

impl CartelPattern {
    pub fn severity(&self) -> f64 {
        match self {
            Self::MutualVerification => 0.4,
            Self::SynchronizedTiming => 0.5,
            Self::VelocityAnomaly => 0.5,
            Self::GeographicClustering => 0.3,
            Self::CoordinatedVoting => 0.7,
            Self::ExclusiveClique => 0.9,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::MutualVerification => "Reciprocal verification pattern detected",
            Self::SynchronizedTiming => "Verifications clustered in suspicious time window",
            Self::ExclusiveClique => "Closed group only verifying each other",
            Self::CoordinatedVoting => "Identical voting patterns across multiple claims",
            Self::VelocityAnomaly => "Abnormal verification frequency",
            Self::GeographicClustering => "Insufficient geographic distribution",
        }
    }
}

/// Result of cartel detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartelDetectionResult {
    /// Detected patterns
    pub patterns: Vec<(CartelPattern, f64)>,
    /// Agents involved
    pub involved_agents: HashSet<String>,
    /// Overall suspicion score (0.0 = clean, 1.0 = definite cartel)
    pub suspicion_score: f64,
    /// Recommended action
    pub recommendation: CartelRecommendation,
    /// Analysis timestamp
    pub analyzed_at: DateTime<Utc>,
}

/// Recommended action based on detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CartelRecommendation {
    /// No action needed
    NoAction,
    /// Increase monitoring of these agents
    IncreaseMonitoring,
    /// Discount verifications from this group
    DiscountVerifications,
    /// Flag for manual review
    ManualReview,
    /// Apply reputation penalty
    ApplyPenalty,
    /// Quarantine pending investigation
    Quarantine,
}

/// Configuration for cartel detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartelDetectionConfig {
    /// Time window for synchronized timing detection (seconds)
    pub sync_window_seconds: i64,
    /// Minimum mutual verification ratio to flag
    pub mutual_threshold: f64,
    /// Minimum clique exclusivity ratio to flag
    pub clique_threshold: f64,
    /// Maximum verifications per hour before velocity flag
    pub max_velocity_per_hour: usize,
    /// Minimum sample size for analysis
    pub min_sample_size: usize,
}

impl Default for CartelDetectionConfig {
    fn default() -> Self {
        Self {
            sync_window_seconds: 300, // 5 minutes
            mutual_threshold: 0.7,
            clique_threshold: 0.8,
            max_velocity_per_hour: 20,
            min_sample_size: 10,
        }
    }
}

/// Cartel detection engine
#[derive(Debug, Clone)]
pub struct CartelDetector {
    /// Configuration
    pub config: CartelDetectionConfig,
    /// Verification history
    events: Vec<VerificationEvent>,
}

impl CartelDetector {
    /// Create a new detector with default config
    pub fn new() -> Self {
        Self {
            config: CartelDetectionConfig::default(),
            events: Vec::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: CartelDetectionConfig) -> Self {
        Self {
            config,
            events: Vec::new(),
        }
    }

    /// Record a verification event
    pub fn record_event(&mut self, event: VerificationEvent) {
        self.events.push(event);
    }

    /// Record multiple events
    pub fn record_events(&mut self, events: Vec<VerificationEvent>) {
        self.events.extend(events);
    }

    /// Analyze a specific set of agents for cartel behavior
    pub fn analyze_agents(&self, agents: &[String]) -> CartelDetectionResult {
        let agent_set: HashSet<String> = agents.iter().cloned().collect();
        let relevant_events: Vec<_> = self
            .events
            .iter()
            .filter(|e| agent_set.contains(&e.verifier_id) || agent_set.contains(&e.author_id))
            .cloned()
            .collect();

        if relevant_events.len() < self.config.min_sample_size {
            return CartelDetectionResult {
                patterns: vec![],
                involved_agents: HashSet::new(),
                suspicion_score: 0.0,
                recommendation: CartelRecommendation::NoAction,
                analyzed_at: Utc::now(),
            };
        }

        let mut patterns = Vec::new();
        let mut involved = HashSet::new();

        // Check mutual verification
        if let Some((score, agents_involved)) = self.detect_mutual_verification(&relevant_events) {
            if score > self.config.mutual_threshold {
                patterns.push((CartelPattern::MutualVerification, score));
                involved.extend(agents_involved);
            }
        }

        // Check synchronized timing
        if let Some((score, agents_involved)) = self.detect_synchronized_timing(&relevant_events) {
            patterns.push((CartelPattern::SynchronizedTiming, score));
            involved.extend(agents_involved);
        }

        // Check exclusive clique
        if let Some((score, agents_involved)) =
            self.detect_exclusive_clique(&relevant_events, &agent_set)
        {
            if score > self.config.clique_threshold {
                patterns.push((CartelPattern::ExclusiveClique, score));
                involved.extend(agents_involved);
            }
        }

        // Check velocity anomaly
        if let Some((score, agents_involved)) = self.detect_velocity_anomaly(&relevant_events) {
            patterns.push((CartelPattern::VelocityAnomaly, score));
            involved.extend(agents_involved);
        }

        // Check coordinated voting
        if let Some((score, agents_involved)) = self.detect_coordinated_voting(&relevant_events) {
            if score > 0.8 {
                patterns.push((CartelPattern::CoordinatedVoting, score));
                involved.extend(agents_involved);
            }
        }

        // Calculate overall suspicion score
        let suspicion_score = if patterns.is_empty() {
            0.0
        } else {
            let weighted_sum: f64 = patterns
                .iter()
                .map(|(p, confidence)| p.severity() * confidence)
                .sum();
            let max_possible: f64 = patterns.iter().map(|(p, _)| p.severity()).sum();
            (weighted_sum / max_possible).min(1.0)
        };

        let recommendation = self.determine_recommendation(suspicion_score, &patterns);

        CartelDetectionResult {
            patterns,
            involved_agents: involved,
            suspicion_score,
            recommendation,
            analyzed_at: Utc::now(),
        }
    }

    /// Detect mutual verification patterns
    fn detect_mutual_verification(
        &self,
        events: &[VerificationEvent],
    ) -> Option<(f64, Vec<String>)> {
        let mut verification_pairs: HashMap<(String, String), usize> = HashMap::new();

        for event in events {
            let pair = (event.verifier_id.clone(), event.author_id.clone());
            *verification_pairs.entry(pair).or_insert(0) += 1;
        }

        let mut mutual_pairs = Vec::new();
        let mut total_pairs = 0;

        for ((a, b), count_ab) in &verification_pairs {
            if a != b {
                total_pairs += 1;
                if let Some(&count_ba) = verification_pairs.get(&(b.clone(), a.clone())) {
                    if *count_ab > 0 && count_ba > 0 {
                        mutual_pairs.push((a.clone(), b.clone()));
                    }
                }
            }
        }

        if total_pairs == 0 {
            return None;
        }

        let mutual_ratio = mutual_pairs.len() as f64 / total_pairs as f64;
        let involved: Vec<String> = mutual_pairs
            .into_iter()
            .flat_map(|(a, b)| vec![a, b])
            .collect();

        Some((mutual_ratio, involved))
    }

    /// Detect synchronized timing patterns
    fn detect_synchronized_timing(
        &self,
        events: &[VerificationEvent],
    ) -> Option<(f64, Vec<String>)> {
        let window = Duration::seconds(self.config.sync_window_seconds);
        let mut clusters: Vec<Vec<&VerificationEvent>> = Vec::new();

        // Group events by time clusters
        let mut sorted_events: Vec<_> = events.iter().collect();
        sorted_events.sort_by_key(|e| e.timestamp);

        let mut current_cluster: Vec<&VerificationEvent> = Vec::new();

        for event in sorted_events {
            if current_cluster.is_empty() {
                current_cluster.push(event);
            } else if event.timestamp - current_cluster.last().unwrap().timestamp <= window {
                current_cluster.push(event);
            } else {
                if current_cluster.len() >= 3 {
                    clusters.push(current_cluster.clone());
                }
                current_cluster = vec![event];
            }
        }

        if current_cluster.len() >= 3 {
            clusters.push(current_cluster);
        }

        if clusters.is_empty() {
            return None;
        }

        let total_events = events.len();
        let clustered_events: usize = clusters.iter().map(|c| c.len()).sum();
        let sync_ratio = clustered_events as f64 / total_events as f64;

        let involved: Vec<String> = clusters
            .into_iter()
            .flat_map(|c| c.into_iter().map(|e| e.verifier_id.clone()))
            .collect();

        Some((sync_ratio, involved))
    }

    /// Detect exclusive clique patterns
    fn detect_exclusive_clique(
        &self,
        events: &[VerificationEvent],
        group: &HashSet<String>,
    ) -> Option<(f64, Vec<String>)> {
        let mut internal_verifications = 0;
        let mut external_verifications = 0;

        for event in events {
            let verifier_in_group = group.contains(&event.verifier_id);
            let author_in_group = group.contains(&event.author_id);

            if verifier_in_group && author_in_group {
                internal_verifications += 1;
            } else if verifier_in_group || author_in_group {
                external_verifications += 1;
            }
        }

        let total = internal_verifications + external_verifications;
        if total == 0 {
            return None;
        }

        let exclusivity = internal_verifications as f64 / total as f64;
        let involved: Vec<String> = group.iter().cloned().collect();

        Some((exclusivity, involved))
    }

    /// Detect velocity anomalies
    fn detect_velocity_anomaly(&self, events: &[VerificationEvent]) -> Option<(f64, Vec<String>)> {
        let mut hourly_counts: HashMap<String, HashMap<i64, usize>> = HashMap::new();

        for event in events {
            let hour = event.timestamp.timestamp() / 3600;
            *hourly_counts
                .entry(event.verifier_id.clone())
                .or_default()
                .entry(hour)
                .or_insert(0) += 1;
        }

        let mut anomalous_agents = Vec::new();
        let mut max_ratio = 0.0_f64;

        for (agent, hours) in &hourly_counts {
            for (_, count) in hours {
                if *count > self.config.max_velocity_per_hour {
                    let ratio = *count as f64 / self.config.max_velocity_per_hour as f64;
                    max_ratio = max_ratio.max(ratio);
                    anomalous_agents.push(agent.clone());
                    break;
                }
            }
        }

        if anomalous_agents.is_empty() {
            return None;
        }

        let anomaly_score = (max_ratio - 1.0).min(1.0);
        Some((anomaly_score, anomalous_agents))
    }

    /// Detect coordinated voting patterns
    fn detect_coordinated_voting(
        &self,
        events: &[VerificationEvent],
    ) -> Option<(f64, Vec<String>)> {
        // Build vote matrix: claim_id -> verifier_id -> vote
        let mut vote_matrix: HashMap<Uuid, HashMap<String, bool>> = HashMap::new();

        for event in events {
            vote_matrix
                .entry(event.claim_id)
                .or_default()
                .insert(event.verifier_id.clone(), event.confirmed);
        }

        // Find agents who voted on multiple claims
        let mut agent_votes: HashMap<String, Vec<bool>> = HashMap::new();
        for votes in vote_matrix.values() {
            for (agent, vote) in votes {
                agent_votes.entry(agent.clone()).or_default().push(*vote);
            }
        }

        // Check for identical voting patterns
        let mut pattern_groups: HashMap<Vec<bool>, Vec<String>> = HashMap::new();
        for (agent, votes) in &agent_votes {
            if votes.len() >= 3 {
                let mut sorted_votes = votes.clone();
                sorted_votes.sort();
                pattern_groups
                    .entry(sorted_votes)
                    .or_default()
                    .push(agent.clone());
            }
        }

        // Find largest group with identical pattern
        let largest_group = pattern_groups.values().max_by_key(|g| g.len())?;

        if largest_group.len() < 3 {
            return None;
        }

        let coordination_score = largest_group.len() as f64 / agent_votes.len() as f64;
        Some((coordination_score, largest_group.clone()))
    }

    /// Determine recommendation based on analysis
    fn determine_recommendation(
        &self,
        score: f64,
        patterns: &[(CartelPattern, f64)],
    ) -> CartelRecommendation {
        if score < 0.2 {
            return CartelRecommendation::NoAction;
        }

        if score < 0.4 {
            return CartelRecommendation::IncreaseMonitoring;
        }

        if score < 0.6 {
            return CartelRecommendation::DiscountVerifications;
        }

        // Check for severe patterns
        let has_severe = patterns
            .iter()
            .any(|(p, c)| p.severity() > 0.7 && *c > 0.8);

        if has_severe || score > 0.8 {
            return CartelRecommendation::Quarantine;
        }

        if score > 0.7 {
            return CartelRecommendation::ApplyPenalty;
        }

        CartelRecommendation::ManualReview
    }

    /// Apply trust penalty based on cartel detection
    pub fn calculate_trust_penalty(&self, result: &CartelDetectionResult) -> f64 {
        match result.recommendation {
            CartelRecommendation::NoAction => 1.0, // No penalty
            CartelRecommendation::IncreaseMonitoring => 0.95,
            CartelRecommendation::DiscountVerifications => 0.7,
            CartelRecommendation::ManualReview => 0.5,
            CartelRecommendation::ApplyPenalty => 0.3,
            CartelRecommendation::Quarantine => 0.0, // Full penalty
        }
    }
}

impl Default for CartelDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_event(
        verifier: &str,
        author: &str,
        claim_id: Uuid,
        offset_minutes: i64,
    ) -> VerificationEvent {
        VerificationEvent {
            verifier_id: verifier.to_string(),
            author_id: author.to_string(),
            claim_id,
            timestamp: Utc::now() + Duration::minutes(offset_minutes),
            confirmed: true,
        }
    }

    #[test]
    fn test_mutual_verification_detection() {
        let mut detector = CartelDetector::with_config(CartelDetectionConfig {
            min_sample_size: 4,
            mutual_threshold: 0.3,
            ..Default::default()
        });

        // Create strong mutual verification pattern between alice and bob
        // Alice verifies multiple Bob claims
        for i in 0..5 {
            detector.record_event(create_event("alice", "bob", Uuid::new_v4(), i * 10));
        }
        // Bob verifies multiple Alice claims (mutual!)
        for i in 0..5 {
            detector.record_event(create_event("bob", "alice", Uuid::new_v4(), i * 10 + 50));
        }

        let result = detector.analyze_agents(&["alice".to_string(), "bob".to_string()]);

        // Should detect mutual pattern
        assert!(result
            .patterns
            .iter()
            .any(|(p, _)| *p == CartelPattern::MutualVerification));
    }

    #[test]
    fn test_synchronized_timing_detection() {
        let mut detector = CartelDetector::with_config(CartelDetectionConfig {
            min_sample_size: 5,
            sync_window_seconds: 300, // 5 minutes
            ..Default::default()
        });

        // Multiple verifications within 5 minutes (synchronized)
        // All within a 4-minute window
        detector.record_event(create_event("alice", "target", Uuid::new_v4(), 0));
        detector.record_event(create_event("bob", "target", Uuid::new_v4(), 1));
        detector.record_event(create_event("charlie", "target", Uuid::new_v4(), 2));
        detector.record_event(create_event("dave", "target", Uuid::new_v4(), 3));
        detector.record_event(create_event("eve", "target", Uuid::new_v4(), 4));

        // Add more synchronized events in another cluster
        detector.record_event(create_event("alice", "other", Uuid::new_v4(), 1000));
        detector.record_event(create_event("bob", "other", Uuid::new_v4(), 1001));
        detector.record_event(create_event("charlie", "other", Uuid::new_v4(), 1002));

        let result = detector.analyze_agents(&[
            "alice".to_string(),
            "bob".to_string(),
            "charlie".to_string(),
            "dave".to_string(),
            "eve".to_string(),
            "target".to_string(),
        ]);

        assert!(result
            .patterns
            .iter()
            .any(|(p, _)| *p == CartelPattern::SynchronizedTiming));
    }

    #[test]
    fn test_velocity_anomaly_detection() {
        let mut detector = CartelDetector::with_config(CartelDetectionConfig {
            max_velocity_per_hour: 5,
            min_sample_size: 5,
            ..Default::default()
        });

        // Alice makes 10 verifications in one hour (anomalous)
        for i in 0..10 {
            detector.record_event(create_event(
                "alice",
                "target",
                Uuid::new_v4(),
                i, // All within same hour
            ));
        }

        let result = detector.analyze_agents(&["alice".to_string()]);

        assert!(result
            .patterns
            .iter()
            .any(|(p, _)| *p == CartelPattern::VelocityAnomaly));
        assert!(result.involved_agents.contains("alice"));
    }

    #[test]
    fn test_clean_behavior() {
        let mut detector = CartelDetector::new();

        // Normal, distributed verification patterns
        let agents = vec!["alice", "bob", "charlie", "dave", "eve"];
        for (i, verifier) in agents.iter().enumerate() {
            let author = agents[(i + 1) % agents.len()];
            detector.record_event(create_event(verifier, author, Uuid::new_v4(), (i * 60) as i64));
        }

        // Add more spread events
        for i in 0..10 {
            detector.record_event(create_event("frank", "george", Uuid::new_v4(), i * 120));
        }

        let result = detector.analyze_agents(&agents.iter().map(|s| s.to_string()).collect::<Vec<_>>());

        assert!(result.suspicion_score < 0.5);
        assert!(result.recommendation == CartelRecommendation::NoAction
            || result.recommendation == CartelRecommendation::IncreaseMonitoring);
    }

    #[test]
    fn test_trust_penalty_calculation() {
        let detector = CartelDetector::new();

        let clean_result = CartelDetectionResult {
            patterns: vec![],
            involved_agents: HashSet::new(),
            suspicion_score: 0.0,
            recommendation: CartelRecommendation::NoAction,
            analyzed_at: Utc::now(),
        };

        assert_eq!(detector.calculate_trust_penalty(&clean_result), 1.0);

        let suspicious_result = CartelDetectionResult {
            patterns: vec![(CartelPattern::ExclusiveClique, 0.9)],
            involved_agents: HashSet::from(["alice".to_string(), "bob".to_string()]),
            suspicion_score: 0.9,
            recommendation: CartelRecommendation::Quarantine,
            analyzed_at: Utc::now(),
        };

        assert_eq!(detector.calculate_trust_penalty(&suspicious_result), 0.0);
    }
}
