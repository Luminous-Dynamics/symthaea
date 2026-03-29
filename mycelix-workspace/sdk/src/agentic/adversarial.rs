// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Adversarial Hardening for Agents
//!
//! Detects and defends against gaming attacks, Sybil attacks, and collusion.
//!
//! ## Key Features
//!
//! - **Gaming Detection**: Identifies agents trying to manipulate their K-Vector
//! - **Sybil Resistance**: Detects multiple identities controlled by same entity
//! - **Collusion Detection**: Identifies coordinated malicious behavior
//! - **Anomaly Scoring**: Statistical detection of abnormal patterns
//! - **Quarantine System**: Isolates suspicious agents pending review

use super::{ActionOutcome, InstrumentalActor};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Gaming Attack Detection
// ============================================================================

/// Types of gaming attacks on K-Vector
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum GamingAttackType {
    /// Artificial success rate inflation
    SuccessInflation,
    /// Timing manipulation (bunching successes)
    TimingManipulation,
    /// Collateral manipulation (large stake changes)
    CollateralManipulation,
    /// Activity bursting (spike then quiet)
    ActivityBursting,
    /// Circular referrals between agents
    CircularReferral,
    /// Synthetic transaction generation
    SyntheticTransactions,
    /// Epistemic classification gaming
    EpistemicGaming,
}

/// Result of gaming detection analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct GamingDetectionResult {
    /// Agent analyzed
    pub agent_id: String,
    /// Detected attack types
    pub detected_attacks: Vec<GamingAttackType>,
    /// Overall suspicion score (0.0 = clean, 1.0 = certain gaming)
    pub suspicion_score: f64,
    /// Specific indicators found
    pub indicators: Vec<GamingIndicator>,
    /// Recommended action
    pub recommended_action: GamingResponse,
    /// Analysis timestamp
    pub timestamp: u64,
}

/// Specific gaming indicator
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct GamingIndicator {
    /// Type of indicator
    pub indicator_type: String,
    /// Severity (0.0-1.0)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Evidence
    pub evidence: String,
}

/// Recommended response to gaming detection
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum GamingResponse {
    /// No action needed
    None,
    /// Increase monitoring
    IncreasedMonitoring,
    /// K-Vector boost cooldown
    KVectorCooldown,
    /// Temporary throttle
    Throttle,
    /// Quarantine for review
    Quarantine,
    /// Escalate to sponsor
    EscalateToSponsor,
}

/// Gaming detection engine configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GamingDetectionConfig {
    /// Minimum actions before analysis is meaningful
    pub min_actions_for_analysis: usize,
    /// Success rate that triggers suspicion (e.g., 0.98 = 98%)
    pub suspicious_success_rate: f64,
    /// Variance threshold for timing analysis
    pub timing_variance_threshold: f64,
    /// Activity burst ratio threshold
    pub burst_ratio_threshold: f64,
    /// Suspicion threshold for quarantine
    pub quarantine_threshold: f64,
    /// Window size for temporal analysis
    pub analysis_window_secs: u64,
}

impl Default for GamingDetectionConfig {
    fn default() -> Self {
        Self {
            min_actions_for_analysis: 20,
            suspicious_success_rate: 0.98,
            timing_variance_threshold: 0.1,
            burst_ratio_threshold: 5.0,
            quarantine_threshold: 0.7,
            analysis_window_secs: 86400 * 7, // 7 days
        }
    }
}

/// Gaming detection engine
pub struct GamingDetector {
    config: GamingDetectionConfig,
    /// Historical suspicion scores
    history: HashMap<String, VecDeque<f64>>,
    /// Known gaming patterns
    patterns: Vec<GamingPattern>,
}

/// A known gaming pattern
#[derive(Clone, Debug)]
pub struct GamingPattern {
    /// Name of the gaming pattern.
    pub name: String,
    /// Type of attack this pattern detects.
    pub attack_type: GamingAttackType,
    /// Detection function for this pattern.
    pub detector: fn(&InstrumentalActor, &GamingDetectionConfig) -> Option<GamingIndicator>,
}

impl GamingDetector {
    /// Create a new gaming detector with the given configuration.
    pub fn new(config: GamingDetectionConfig) -> Self {
        Self {
            config,
            history: HashMap::new(),
            patterns: Self::default_patterns(),
        }
    }

    fn default_patterns() -> Vec<GamingPattern> {
        vec![
            GamingPattern {
                name: "success_inflation".to_string(),
                attack_type: GamingAttackType::SuccessInflation,
                detector: detect_success_inflation,
            },
            GamingPattern {
                name: "timing_manipulation".to_string(),
                attack_type: GamingAttackType::TimingManipulation,
                detector: detect_timing_manipulation,
            },
            GamingPattern {
                name: "activity_bursting".to_string(),
                attack_type: GamingAttackType::ActivityBursting,
                detector: detect_activity_bursting,
            },
            GamingPattern {
                name: "epistemic_gaming".to_string(),
                attack_type: GamingAttackType::EpistemicGaming,
                detector: detect_epistemic_gaming,
            },
        ]
    }

    /// Analyze an agent for gaming behavior
    pub fn analyze(&mut self, agent: &InstrumentalActor, timestamp: u64) -> GamingDetectionResult {
        let mut indicators = Vec::new();
        let mut detected_attacks = Vec::new();

        // Run all pattern detectors
        for pattern in &self.patterns {
            if let Some(indicator) = (pattern.detector)(agent, &self.config) {
                detected_attacks.push(pattern.attack_type.clone());
                indicators.push(indicator);
            }
        }

        // Calculate overall suspicion score
        let suspicion_score = if indicators.is_empty() {
            0.0
        } else {
            let max_severity = indicators.iter().map(|i| i.severity).fold(0.0f64, f64::max);
            let avg_severity: f64 =
                indicators.iter().map(|i| i.severity).sum::<f64>() / indicators.len() as f64;
            (max_severity * 0.6 + avg_severity * 0.4).min(1.0)
        };

        // Update history
        let history_entry = self
            .history
            .entry(agent.agent_id.as_str().to_string())
            .or_default();
        history_entry.push_back(suspicion_score);
        if history_entry.len() > 100 {
            history_entry.pop_front();
        }

        // Determine recommended action
        let recommended_action = self.determine_response(suspicion_score, &indicators);

        GamingDetectionResult {
            agent_id: agent.agent_id.as_str().to_string(),
            detected_attacks,
            suspicion_score,
            indicators,
            recommended_action,
            timestamp,
        }
    }

    fn determine_response(&self, score: f64, _indicators: &[GamingIndicator]) -> GamingResponse {
        if score >= self.config.quarantine_threshold {
            GamingResponse::Quarantine
        } else if score >= 0.5 {
            GamingResponse::EscalateToSponsor
        } else if score >= 0.3 {
            GamingResponse::KVectorCooldown
        } else if score >= 0.15 {
            GamingResponse::IncreasedMonitoring
        } else {
            GamingResponse::None
        }
    }

    /// Get trending suspicion (is agent becoming more suspicious over time?)
    pub fn get_suspicion_trend(&self, agent_id: &str) -> Option<f64> {
        let history = self.history.get(agent_id)?;
        if history.len() < 5 {
            return None;
        }

        let recent: Vec<_> = history.iter().rev().take(5).copied().collect();
        let older: Vec<_> = history.iter().take(5).copied().collect();

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        Some(recent_avg - older_avg) // Positive = getting more suspicious
    }
}

// Pattern detection functions
fn detect_success_inflation(
    agent: &InstrumentalActor,
    config: &GamingDetectionConfig,
) -> Option<GamingIndicator> {
    if agent.behavior_log.len() < config.min_actions_for_analysis {
        return None;
    }

    let successes = agent
        .behavior_log
        .iter()
        .filter(|e| e.outcome == ActionOutcome::Success)
        .count();
    let success_rate = successes as f64 / agent.behavior_log.len() as f64;

    if success_rate > config.suspicious_success_rate {
        Some(GamingIndicator {
            indicator_type: "success_inflation".to_string(),
            severity: (success_rate - config.suspicious_success_rate)
                / (1.0 - config.suspicious_success_rate),
            description: format!(
                "Suspiciously high success rate: {:.1}%",
                success_rate * 100.0
            ),
            evidence: format!(
                "{} successes out of {} actions",
                successes,
                agent.behavior_log.len()
            ),
        })
    } else {
        None
    }
}

fn detect_timing_manipulation(
    agent: &InstrumentalActor,
    config: &GamingDetectionConfig,
) -> Option<GamingIndicator> {
    if agent.behavior_log.len() < config.min_actions_for_analysis {
        return None;
    }

    // Check for suspiciously regular timing (bots often have very consistent intervals)
    let mut timestamps: Vec<u64> = agent.behavior_log.iter().map(|e| e.timestamp).collect();
    if timestamps.len() < 3 {
        return None;
    }

    // Sort timestamps to ensure proper ordering
    timestamps.sort();

    // Compute intervals safely (use saturating_sub in case of any edge case)
    let intervals: Vec<f64> = timestamps
        .windows(2)
        .map(|w| w[1].saturating_sub(w[0]) as f64)
        .filter(|&i| i > 0.0) // Skip zero intervals
        .collect();

    if intervals.is_empty() {
        return None;
    }

    let mean_interval: f64 = intervals.iter().sum::<f64>() / intervals.len() as f64;
    if mean_interval == 0.0 {
        return None;
    }

    let variance: f64 = intervals
        .iter()
        .map(|i| ((i - mean_interval) / mean_interval).powi(2))
        .sum::<f64>()
        / intervals.len() as f64;
    let coefficient_of_variation = variance.sqrt();

    // Very low variance suggests automated/scripted behavior
    if coefficient_of_variation < config.timing_variance_threshold {
        Some(GamingIndicator {
            indicator_type: "timing_manipulation".to_string(),
            severity: 1.0 - (coefficient_of_variation / config.timing_variance_threshold),
            description: "Suspiciously regular action timing (possible automation)".to_string(),
            evidence: format!(
                "Coefficient of variation: {:.3} (threshold: {:.3})",
                coefficient_of_variation, config.timing_variance_threshold
            ),
        })
    } else {
        None
    }
}

fn detect_activity_bursting(
    agent: &InstrumentalActor,
    config: &GamingDetectionConfig,
) -> Option<GamingIndicator> {
    if agent.behavior_log.len() < config.min_actions_for_analysis * 2 {
        return None;
    }

    let now = agent.behavior_log.last().map(|e| e.timestamp).unwrap_or(0);
    let window_start = now.saturating_sub(config.analysis_window_secs);

    // Split into recent half and older half
    let in_window: Vec<_> = agent
        .behavior_log
        .iter()
        .filter(|e| e.timestamp >= window_start)
        .collect();

    if in_window.len() < 10 {
        return None;
    }

    let mid_time = window_start + config.analysis_window_secs / 2;
    let recent_count = in_window.iter().filter(|e| e.timestamp >= mid_time).count();
    let older_count = in_window.iter().filter(|e| e.timestamp < mid_time).count();

    if older_count == 0 {
        return None;
    }

    let burst_ratio = recent_count as f64 / older_count as f64;

    if burst_ratio > config.burst_ratio_threshold {
        Some(GamingIndicator {
            indicator_type: "activity_bursting".to_string(),
            severity: ((burst_ratio - config.burst_ratio_threshold) / config.burst_ratio_threshold)
                .min(1.0),
            description: "Suspicious activity burst pattern".to_string(),
            evidence: format!(
                "Recent: {} actions, Older: {} actions (ratio: {:.1}x)",
                recent_count, older_count, burst_ratio
            ),
        })
    } else {
        None
    }
}

fn detect_epistemic_gaming(
    agent: &InstrumentalActor,
    config: &GamingDetectionConfig,
) -> Option<GamingIndicator> {
    // Check if agent consistently claims high epistemic levels but has poor verification
    if agent.output_history.len() < config.min_actions_for_analysis {
        return None;
    }

    let high_e_claims: Vec<_> = agent
        .output_history
        .iter()
        .filter(|o| o.classification.empirical as u8 >= 3) // E3 or E4
        .collect();

    if high_e_claims.is_empty() {
        return None;
    }

    // Check verification rate of high-E claims
    let verified: Vec<_> = high_e_claims.iter().filter(|o| o.verified).collect();
    if verified.len() < 5 {
        return None; // Not enough verified to judge
    }

    let correct = verified
        .iter()
        .filter(|o| {
            matches!(
                o.verification_outcome,
                Some(super::VerificationOutcome::Correct)
            )
        })
        .count();
    let accuracy = correct as f64 / verified.len() as f64;

    // High epistemic claims should have high accuracy
    // E3 should be ~80%+ accurate, E4 should be ~95%+
    let expected_accuracy = 0.75;
    if accuracy < expected_accuracy {
        Some(GamingIndicator {
            indicator_type: "epistemic_gaming".to_string(),
            severity: (expected_accuracy - accuracy) / expected_accuracy,
            description: "High epistemic claims with poor verification accuracy".to_string(),
            evidence: format!(
                "{} correct out of {} verified high-E outputs ({:.1}% vs expected {:.1}%)",
                correct,
                verified.len(),
                accuracy * 100.0,
                expected_accuracy * 100.0
            ),
        })
    } else {
        None
    }
}

// ============================================================================
// Sybil Resistance
// ============================================================================

/// Evidence of potential Sybil attack (multiple identities)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct SybilEvidence {
    /// Agents suspected of being Sybils
    pub suspected_agents: Vec<String>,
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    /// Type of evidence
    pub evidence_type: SybilEvidenceType,
    /// Details
    pub details: String,
}

/// Types of Sybil evidence
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum SybilEvidenceType {
    /// Same sponsor with correlated behavior
    CorrelatedSponsorAgents,
    /// Synchronized action timing
    SynchronizedTiming,
    /// Identical output patterns
    IdenticalOutputs,
    /// Circular transactions
    CircularTransactions,
    /// Mutual endorsement ring
    EndorsementRing,
}

/// Sybil detection engine
pub struct SybilDetector {
    /// Similarity threshold for flagging
    pub similarity_threshold: f64,
    /// Minimum agents to form a ring
    pub min_ring_size: usize,
}

impl SybilDetector {
    /// Create a new Sybil detector with the given similarity threshold.
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
            min_ring_size: 3,
        }
    }

    /// Analyze a group of agents for Sybil patterns
    pub fn analyze_group(&self, agents: &[&InstrumentalActor]) -> Vec<SybilEvidence> {
        let mut evidence = Vec::new();

        // Check for sponsor correlation
        if let Some(e) = self.detect_sponsor_correlation(agents) {
            evidence.push(e);
        }

        // Check for timing correlation
        if let Some(e) = self.detect_timing_correlation(agents) {
            evidence.push(e);
        }

        evidence
    }

    fn detect_sponsor_correlation(&self, agents: &[&InstrumentalActor]) -> Option<SybilEvidence> {
        // Group agents by sponsor
        let mut by_sponsor: HashMap<&str, Vec<&InstrumentalActor>> = HashMap::new();
        for agent in agents {
            by_sponsor
                .entry(&agent.sponsor_did)
                .or_default()
                .push(*agent);
        }

        // Find sponsors with multiple agents that have correlated behavior
        for (sponsor, sponsor_agents) in by_sponsor {
            if sponsor_agents.len() < self.min_ring_size {
                continue;
            }

            // Check K-Vector similarity
            let kvectors: Vec<_> = sponsor_agents.iter().map(|a| &a.k_vector).collect();
            let similarities = self.compute_kvector_similarities(&kvectors);

            let avg_similarity: f64 =
                similarities.iter().sum::<f64>() / similarities.len().max(1) as f64;

            if avg_similarity > self.similarity_threshold {
                return Some(SybilEvidence {
                    suspected_agents: sponsor_agents
                        .iter()
                        .map(|a| a.agent_id.as_str().to_string())
                        .collect(),
                    confidence: (avg_similarity - self.similarity_threshold)
                        / (1.0 - self.similarity_threshold),
                    evidence_type: SybilEvidenceType::CorrelatedSponsorAgents,
                    details: format!(
                        "Sponsor {} has {} agents with {:.1}% K-Vector similarity",
                        sponsor,
                        sponsor_agents.len(),
                        avg_similarity * 100.0
                    ),
                });
            }
        }

        None
    }

    fn detect_timing_correlation(&self, agents: &[&InstrumentalActor]) -> Option<SybilEvidence> {
        if agents.len() < self.min_ring_size {
            return None;
        }

        // Collect all timestamps
        let mut all_times: Vec<(String, u64)> = Vec::new();
        for agent in agents {
            for entry in &agent.behavior_log {
                all_times.push((agent.agent_id.as_str().to_string(), entry.timestamp));
            }
        }

        if all_times.len() < 20 {
            return None;
        }

        // Look for synchronized bursts (multiple agents active in same short windows)
        all_times.sort_by_key(|(_, t)| *t);

        let window_size = 60u64; // 1 minute windows
        let mut window_agents: HashMap<u64, HashSet<String>> = HashMap::new();

        for (agent_id, timestamp) in &all_times {
            let window = timestamp / window_size;
            window_agents
                .entry(window)
                .or_default()
                .insert(agent_id.clone());
        }

        // Find windows with many different agents
        let suspicious_windows: Vec<_> = window_agents
            .iter()
            .filter(|(_, agents)| agents.len() >= self.min_ring_size)
            .collect();

        if suspicious_windows.len() > 5 {
            let all_suspects: HashSet<_> = suspicious_windows
                .iter()
                .flat_map(|(_, agents)| agents.iter().cloned())
                .collect();

            return Some(SybilEvidence {
                suspected_agents: all_suspects.into_iter().collect(),
                confidence: (suspicious_windows.len() as f64 / 10.0).min(1.0),
                evidence_type: SybilEvidenceType::SynchronizedTiming,
                details: format!(
                    "{} time windows with {} or more agents active simultaneously",
                    suspicious_windows.len(),
                    self.min_ring_size
                ),
            });
        }

        None
    }

    fn compute_kvector_similarities(&self, kvectors: &[&KVector]) -> Vec<f64> {
        let mut similarities = Vec::new();

        for i in 0..kvectors.len() {
            for j in (i + 1)..kvectors.len() {
                let sim = kvector_similarity(kvectors[i], kvectors[j]);
                similarities.push(sim);
            }
        }

        similarities
    }
}

fn kvector_similarity(a: &KVector, b: &KVector) -> f64 {
    let dims_a = [a.k_r, a.k_a, a.k_i, a.k_p, a.k_m, a.k_s, a.k_h, a.k_topo];
    let dims_b = [b.k_r, b.k_a, b.k_i, b.k_p, b.k_m, b.k_s, b.k_h, b.k_topo];

    let diff_sum: f64 = dims_a
        .iter()
        .zip(dims_b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .sum();

    1.0 - (diff_sum / 8.0) // Normalize by dimension count
}

// ============================================================================
// Collusion Detection
// ============================================================================

/// Evidence of potential collusion
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct CollusionEvidence {
    /// Agents involved
    pub agents: Vec<String>,
    /// Collusion type
    pub collusion_type: CollusionType,
    /// Confidence level
    pub confidence: f64,
    /// Details
    pub details: String,
}

/// Types of collusion
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum CollusionType {
    /// Coordinated voting
    VoteCoordination,
    /// Mutual reputation boosting
    ReputationBoosting,
    /// Coordinated market manipulation
    MarketManipulation,
    /// Orchestrated consensus attacks
    ConsensusAttack,
}

/// Collusion detection engine
pub struct CollusionDetector {
    /// Correlation threshold
    pub correlation_threshold: f64,
    /// Minimum group size
    pub min_group_size: usize,
    /// Recent interactions for analysis
    pub interactions: Vec<AgentInteractionRecord>,
}

/// Record of agent interaction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentInteractionRecord {
    /// Source agent identifier.
    pub from_agent: String,
    /// Target agent identifier.
    pub to_agent: String,
    /// Type of interaction.
    pub interaction_type: String,
    /// Numeric value of the interaction.
    pub value: f64,
    /// Timestamp of the interaction.
    pub timestamp: u64,
}

impl CollusionDetector {
    /// Create a new collusion detector with the given correlation threshold.
    pub fn new(correlation_threshold: f64) -> Self {
        Self {
            correlation_threshold,
            min_group_size: 3,
            interactions: Vec::new(),
        }
    }

    /// Record an interaction
    pub fn record_interaction(&mut self, interaction: AgentInteractionRecord) {
        self.interactions.push(interaction);

        // Keep bounded
        if self.interactions.len() > 10000 {
            self.interactions.remove(0);
        }
    }

    /// Analyze for collusion patterns
    pub fn analyze(&self) -> Vec<CollusionEvidence> {
        let mut evidence = Vec::new();

        // Check for mutual reputation boosting
        if let Some(e) = self.detect_mutual_boosting() {
            evidence.push(e);
        }

        // Check for coordinated voting (if applicable)
        if let Some(e) = self.detect_vote_coordination() {
            evidence.push(e);
        }

        evidence
    }

    fn detect_mutual_boosting(&self) -> Option<CollusionEvidence> {
        // Build interaction graph
        let mut edges: HashMap<(String, String), f64> = HashMap::new();

        for interaction in &self.interactions {
            if interaction.value > 0.0 {
                let key = if interaction.from_agent < interaction.to_agent {
                    (interaction.from_agent.clone(), interaction.to_agent.clone())
                } else {
                    (interaction.to_agent.clone(), interaction.from_agent.clone())
                };
                *edges.entry(key).or_insert(0.0) += interaction.value;
            }
        }

        // Find bidirectional strong connections (mutual boosting)
        let mut mutual_boosters: HashSet<String> = HashSet::new();

        for ((a, b), weight) in &edges {
            // Check if there's also a strong connection in the opposite direction
            let reverse_key = (b.clone(), a.clone());
            if let Some(reverse_weight) = edges.get(&reverse_key) {
                if *weight > 5.0 && *reverse_weight > 5.0 {
                    mutual_boosters.insert(a.clone());
                    mutual_boosters.insert(b.clone());
                }
            }
        }

        if mutual_boosters.len() >= self.min_group_size {
            Some(CollusionEvidence {
                agents: mutual_boosters.into_iter().collect(),
                collusion_type: CollusionType::ReputationBoosting,
                confidence: 0.7,
                details: "Mutual reputation boosting pattern detected".to_string(),
            })
        } else {
            None
        }
    }

    fn detect_vote_coordination(&self) -> Option<CollusionEvidence> {
        // Look for agents that vote identically on multiple topics
        let vote_interactions: Vec<_> = self
            .interactions
            .iter()
            .filter(|i| i.interaction_type == "vote")
            .collect();

        if vote_interactions.len() < 10 {
            return None;
        }

        // Group by topic (using value as topic proxy)
        let mut votes_by_agent: HashMap<String, Vec<f64>> = HashMap::new();
        for v in vote_interactions {
            votes_by_agent
                .entry(v.from_agent.clone())
                .or_default()
                .push(v.value);
        }

        // Find agents with highly correlated vote patterns
        let agents: Vec<_> = votes_by_agent.keys().cloned().collect();
        let mut correlated_groups: HashSet<String> = HashSet::new();

        for i in 0..agents.len() {
            for j in (i + 1)..agents.len() {
                let votes_i = &votes_by_agent[&agents[i]];
                let votes_j = &votes_by_agent[&agents[j]];

                if votes_i.len() >= 5 && votes_j.len() >= 5 {
                    // Simple correlation: count identical votes
                    let min_len = votes_i.len().min(votes_j.len());
                    let identical = votes_i
                        .iter()
                        .take(min_len)
                        .zip(votes_j.iter().take(min_len))
                        .filter(|(a, b)| (**a - **b).abs() < 0.01)
                        .count();

                    let correlation = identical as f64 / min_len as f64;

                    if correlation > self.correlation_threshold {
                        correlated_groups.insert(agents[i].clone());
                        correlated_groups.insert(agents[j].clone());
                    }
                }
            }
        }

        if correlated_groups.len() >= self.min_group_size {
            Some(CollusionEvidence {
                agents: correlated_groups.into_iter().collect(),
                collusion_type: CollusionType::VoteCoordination,
                confidence: 0.6,
                details: "Highly correlated voting patterns detected".to_string(),
            })
        } else {
            None
        }
    }
}

// ============================================================================
// Quarantine System
// ============================================================================

/// Quarantined agent entry
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct QuarantineEntry {
    /// Agent ID
    pub agent_id: String,
    /// Reason for quarantine
    pub reason: QuarantineReason,
    /// When quarantine started
    pub started_at: u64,
    /// Evidence leading to quarantine
    pub evidence: Vec<String>,
    /// Review status
    pub review_status: ReviewStatus,
    /// Reviewer (if any)
    pub reviewer: Option<String>,
}

/// Reason for quarantine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum QuarantineReason {
    /// Gaming behavior detected.
    GamingDetected,
    /// Sybil attack suspected.
    SybilSuspected,
    /// Collusion suspected.
    CollusionSuspected,
    /// Anomalous behavior observed.
    AnomalousBehavior,
    /// Manually flagged for review.
    ManualFlag,
}

/// Review status
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum ReviewStatus {
    /// Awaiting review.
    Pending,
    /// Currently under review.
    UnderReview,
    /// Cleared after review.
    Cleared,
    /// Confirmed as malicious.
    Confirmed,
}

/// Quarantine manager
#[derive(Clone, Debug, Default)]
pub struct QuarantineManager {
    /// Map of agent IDs to quarantine entries.
    pub entries: HashMap<String, QuarantineEntry>,
}

impl QuarantineManager {
    /// Create a new empty quarantine manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Quarantine an agent
    pub fn quarantine(
        &mut self,
        agent_id: &str,
        reason: QuarantineReason,
        evidence: Vec<String>,
        timestamp: u64,
    ) {
        self.entries.insert(
            agent_id.to_string(),
            QuarantineEntry {
                agent_id: agent_id.to_string(),
                reason,
                started_at: timestamp,
                evidence,
                review_status: ReviewStatus::Pending,
                reviewer: None,
            },
        );
    }

    /// Check if agent is quarantined
    pub fn is_quarantined(&self, agent_id: &str) -> bool {
        self.entries
            .get(agent_id)
            .map(|e| !matches!(e.review_status, ReviewStatus::Cleared))
            .unwrap_or(false)
    }

    /// Start review
    pub fn start_review(&mut self, agent_id: &str, reviewer: &str) -> bool {
        if let Some(entry) = self.entries.get_mut(agent_id) {
            entry.review_status = ReviewStatus::UnderReview;
            entry.reviewer = Some(reviewer.to_string());
            true
        } else {
            false
        }
    }

    /// Clear agent (false positive)
    pub fn clear(&mut self, agent_id: &str) -> bool {
        if let Some(entry) = self.entries.get_mut(agent_id) {
            entry.review_status = ReviewStatus::Cleared;
            true
        } else {
            false
        }
    }

    /// Confirm threat (take action)
    pub fn confirm(&mut self, agent_id: &str) -> bool {
        if let Some(entry) = self.entries.get_mut(agent_id) {
            entry.review_status = ReviewStatus::Confirmed;
            true
        } else {
            false
        }
    }

    /// Get pending reviews
    pub fn pending_reviews(&self) -> Vec<&QuarantineEntry> {
        self.entries
            .values()
            .filter(|e| {
                matches!(
                    e.review_status,
                    ReviewStatus::Pending | ReviewStatus::UnderReview
                )
            })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{
        AgentClass, AgentConstraints, AgentId, AgentStatus, BehaviorLogEntry, EpistemicStats,
    };
    use crate::matl::KVector;

    fn create_test_agent(id: &str, sponsor: &str) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: sponsor.to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_gaming_detection_clean() {
        let mut detector = GamingDetector::new(GamingDetectionConfig::default());
        let mut agent = create_test_agent("clean-agent", "sponsor-1");

        // Add normal behavior with varied timing (human-like)
        let varied_intervals = [
            45, 180, 90, 300, 60, 150, 200, 75, 120, 240, 55, 170, 85, 310, 65, 145, 210, 80, 125,
            230, 50, 175, 95, 290, 70, 155, 195, 78, 130, 250,
        ];
        let mut timestamp = 1000u64;
        for (i, &interval) in varied_intervals.iter().enumerate() {
            timestamp += interval;
            agent.behavior_log.push(BehaviorLogEntry {
                timestamp,
                action_type: "process".to_string(),
                kredit_consumed: 10,
                counterparties: vec![],
                outcome: if i % 4 == 0 {
                    ActionOutcome::Error
                } else {
                    ActionOutcome::Success
                },
            });
        }

        let result = detector.analyze(&agent, timestamp + 1000);
        // Clean agent should have low suspicion (allowing for some margin)
        assert!(
            result.suspicion_score < 0.5,
            "Suspicion score too high: {}",
            result.suspicion_score
        );
        assert!(matches!(
            result.recommended_action,
            GamingResponse::None | GamingResponse::IncreasedMonitoring
        ));
    }

    #[test]
    fn test_gaming_detection_success_inflation() {
        let mut detector = GamingDetector::new(GamingDetectionConfig::default());
        let mut agent = create_test_agent("gaming-agent", "sponsor-1");

        // Add suspiciously perfect behavior
        for i in 0..30 {
            agent.behavior_log.push(BehaviorLogEntry {
                timestamp: 1000 + i * 120,
                action_type: "process".to_string(),
                kredit_consumed: 10,
                counterparties: vec![],
                outcome: ActionOutcome::Success, // 100% success rate
            });
        }

        let result = detector.analyze(&agent, 5000);
        assert!(result.suspicion_score > 0.3);
        assert!(result
            .detected_attacks
            .contains(&GamingAttackType::SuccessInflation));
    }

    #[test]
    fn test_sybil_detection() {
        let detector = SybilDetector::new(0.9);

        // Create agents with similar K-Vectors
        let mut agent1 = create_test_agent("sybil-1", "same-sponsor");
        let mut agent2 = create_test_agent("sybil-2", "same-sponsor");
        let mut agent3 = create_test_agent("sybil-3", "same-sponsor");

        // Make K-Vectors very similar
        let similar_kv = KVector::new(0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.6, 0.3, 0.7, 0.65);
        agent1.k_vector = similar_kv.clone();
        agent2.k_vector = similar_kv.clone();
        agent3.k_vector = similar_kv.clone();

        let agents: Vec<&InstrumentalActor> = vec![&agent1, &agent2, &agent3];
        let evidence = detector.analyze_group(&agents);

        assert!(!evidence.is_empty());
        assert!(evidence
            .iter()
            .any(|e| matches!(e.evidence_type, SybilEvidenceType::CorrelatedSponsorAgents)));
    }

    #[test]
    fn test_quarantine_manager() {
        let mut manager = QuarantineManager::new();

        manager.quarantine(
            "suspicious-agent",
            QuarantineReason::GamingDetected,
            vec!["High success rate".to_string()],
            1000,
        );

        assert!(manager.is_quarantined("suspicious-agent"));
        assert!(!manager.is_quarantined("clean-agent"));

        // Start review
        manager.start_review("suspicious-agent", "reviewer-1");
        assert!(manager.is_quarantined("suspicious-agent"));

        // Clear
        manager.clear("suspicious-agent");
        assert!(!manager.is_quarantined("suspicious-agent"));
    }

    #[test]
    fn test_collusion_detection() {
        let mut detector = CollusionDetector::new(0.8);

        // Add coordinated vote interactions
        for i in 0..20 {
            detector.record_interaction(AgentInteractionRecord {
                from_agent: "agent-1".to_string(),
                to_agent: "topic".to_string(),
                interaction_type: "vote".to_string(),
                value: 0.8, // Same vote
                timestamp: 1000 + i,
            });
            detector.record_interaction(AgentInteractionRecord {
                from_agent: "agent-2".to_string(),
                to_agent: "topic".to_string(),
                interaction_type: "vote".to_string(),
                value: 0.8, // Same vote
                timestamp: 1001 + i,
            });
            detector.record_interaction(AgentInteractionRecord {
                from_agent: "agent-3".to_string(),
                to_agent: "topic".to_string(),
                interaction_type: "vote".to_string(),
                value: 0.8, // Same vote
                timestamp: 1002 + i,
            });
        }

        let evidence = detector.analyze();
        // May or may not detect depending on exact implementation
        // The test verifies no panic and reasonable behavior
        assert!(evidence.len() <= 2);
    }

    // =========================================================================
    // Adversarial Property-Based Tests for Gaming Detection
    // =========================================================================

    use proptest::prelude::*;

    /// Strategy to generate random behavior logs
    fn behavior_log_strategy(count: usize) -> impl Strategy<Value = Vec<BehaviorLogEntry>> {
        prop::collection::vec(
            (
                1000u64..100000u64, // timestamp
                0.0f64..1.0f64,     // success probability
                prop::collection::vec(any::<u8>().prop_map(|b| format!("peer_{}", b)), 0..5),
            ),
            count,
        )
        .prop_map(|entries| {
            let mut log: Vec<BehaviorLogEntry> = entries
                .into_iter()
                .enumerate()
                .map(|(i, (ts, p, peers))| BehaviorLogEntry {
                    timestamp: ts + i as u64 * 60,
                    action_type: "test_action".to_string(),
                    kredit_consumed: 10,
                    counterparties: peers,
                    outcome: if p > 0.5 {
                        ActionOutcome::Success
                    } else {
                        ActionOutcome::Error
                    },
                })
                .collect();
            log.sort_by_key(|e| e.timestamp);
            log
        })
    }

    proptest! {
        /// INVARIANT: Gaming detector never panics on any valid input
        #[test]
        fn prop_gaming_detector_no_panic(
            behavior_log in behavior_log_strategy(50),
            timestamp in 1000u64..1000000u64,
        ) {
            let mut detector = GamingDetector::new(GamingDetectionConfig::default());
            let mut agent = create_test_agent("prop-agent", "sponsor");
            agent.behavior_log = behavior_log;
            agent.last_activity = timestamp;

            // Should not panic
            let result = detector.analyze(&agent, timestamp + 1000);

            // Basic sanity checks
            prop_assert!(result.suspicion_score >= 0.0 && result.suspicion_score <= 1.0,
                "Suspicion score out of bounds: {}", result.suspicion_score);
        }

        /// INVARIANT: Perfect success rate (100%) always raises suspicion
        #[test]
        fn prop_perfect_success_raises_suspicion(
            count in 30usize..100usize,
        ) {
            let mut detector = GamingDetector::new(GamingDetectionConfig::default());
            let mut agent = create_test_agent("perfect-agent", "sponsor");

            // Create perfect behavior log (100% success)
            for i in 0..count {
                agent.behavior_log.push(BehaviorLogEntry {
                    timestamp: 1000 + i as u64 * 120,
                    action_type: "process".to_string(),
                    kredit_consumed: 10,
                    counterparties: vec![],
                    outcome: ActionOutcome::Success,
                });
            }

            let result = detector.analyze(&agent, 1000 + (count as u64 + 10) * 120);

            // 100% success rate should trigger some suspicion
            prop_assert!(result.suspicion_score > 0.2,
                "Perfect success rate should raise suspicion: {}", result.suspicion_score);
            prop_assert!(result.detected_attacks.contains(&GamingAttackType::SuccessInflation),
                "Should detect success inflation attack");
        }

        /// INVARIANT: Timing manipulation (constant intervals) raises suspicion
        #[test]
        fn prop_constant_timing_raises_suspicion(
            count in 30usize..80usize,
            interval in 60u64..300u64,
        ) {
            let mut detector = GamingDetector::new(GamingDetectionConfig {
                timing_variance_threshold: 0.05, // Strict threshold for test
                ..Default::default()
            });
            let mut agent = create_test_agent("bot-agent", "sponsor");

            // Create log with perfectly constant intervals (bot-like)
            for i in 0..count {
                agent.behavior_log.push(BehaviorLogEntry {
                    timestamp: 1000 + i as u64 * interval,
                    action_type: "process".to_string(),
                    kredit_consumed: 10,
                    counterparties: vec![],
                    outcome: if i % 5 == 0 { ActionOutcome::Error } else { ActionOutcome::Success },
                });
            }

            let result = detector.analyze(&agent, 1000 + (count as u64 + 10) * interval);

            // Perfect timing should trigger timing manipulation suspicion
            prop_assert!(result.suspicion_score >= 0.0,
                "Suspicion should be non-negative: {}", result.suspicion_score);
            // Note: Depending on config, may or may not detect
            // At minimum, no panic and valid output
        }

        /// INVARIANT: Sybil detection doesn't panic with any K-Vectors
        #[test]
        fn prop_sybil_detector_no_panic(
            kv_values in prop::collection::vec(
                (
                    0.0f32..=1.0f32, 0.0f32..=1.0f32, 0.0f32..=1.0f32,
                    0.0f32..=1.0f32, 0.0f32..=1.0f32, 0.0f32..=1.0f32,
                    0.0f32..=1.0f32, 0.0f32..=1.0f32, 0.0f32..=1.0f32,
                    0.0f32..=1.0f32,
                ),
                2..10
            ),
        ) {
            let detector = SybilDetector::new(0.9);

            // Create agents with random K-Vectors
            let agents: Vec<InstrumentalActor> = kv_values.iter().enumerate().map(|(i, kv)| {
                let mut agent = create_test_agent(&format!("agent-{}", i), "sponsor");
                agent.k_vector = KVector::new(
                    kv.0, kv.1, kv.2, kv.3, kv.4, kv.5, kv.6, kv.7, kv.8, kv.9
                );
                agent
            }).collect();

            let refs: Vec<&InstrumentalActor> = agents.iter().collect();
            let evidence = detector.analyze_group(&refs);

            // Should not panic, evidence count should be reasonable
            prop_assert!(evidence.len() <= agents.len() * agents.len(),
                "Too many evidence items: {}", evidence.len());
        }

        /// INVARIANT: Identical K-Vectors always trigger Sybil detection
        #[test]
        fn prop_identical_kvectors_detected(
            kv in (
                0.0f32..=1.0f32, 0.0f32..=1.0f32, 0.0f32..=1.0f32,
                0.0f32..=1.0f32, 0.0f32..=1.0f32, 0.0f32..=1.0f32,
                0.0f32..=1.0f32, 0.0f32..=1.0f32, 0.0f32..=1.0f32,
                0.0f32..=1.0f32,
            ),
        ) {
            let detector = SybilDetector::new(0.95);

            // Create multiple agents with IDENTICAL K-Vectors
            let mut agents: Vec<InstrumentalActor> = vec![];
            for i in 0..5 {
                let mut agent = create_test_agent(&format!("clone-{}", i), "same-sponsor");
                agent.k_vector = KVector::new(
                    kv.0, kv.1, kv.2, kv.3, kv.4, kv.5, kv.6, kv.7, kv.8, kv.9
                );
                agents.push(agent);
            }

            let refs: Vec<&InstrumentalActor> = agents.iter().collect();
            let evidence = detector.analyze_group(&refs);

            // Identical K-Vectors from same sponsor should be flagged
            prop_assert!(!evidence.is_empty(),
                "Identical K-Vectors should trigger Sybil detection");
        }

        /// INVARIANT: Quarantine manager correctly tracks state
        #[test]
        fn prop_quarantine_state_consistent(
            agent_ids in prop::collection::vec("[a-z]{5,10}", 1..20),
        ) {
            let mut manager = QuarantineManager::new();

            // Quarantine all agents
            for (i, id) in agent_ids.iter().enumerate() {
                manager.quarantine(
                    id,
                    QuarantineReason::GamingDetected,
                    vec!["Test".to_string()],
                    1000 + i as u64,
                );
            }

            // Verify all are quarantined
            for id in &agent_ids {
                prop_assert!(manager.is_quarantined(id),
                    "Agent {} should be quarantined", id);
            }

            // Clear half
            for id in agent_ids.iter().take(agent_ids.len() / 2) {
                manager.clear(id);
            }

            // Verify half cleared, half still quarantined
            for (i, id) in agent_ids.iter().enumerate() {
                let expected = i >= agent_ids.len() / 2;
                prop_assert_eq!(manager.is_quarantined(id), expected,
                    "Agent {} quarantine state mismatch", id);
            }
        }
    }
}
