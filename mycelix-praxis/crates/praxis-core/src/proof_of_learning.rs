// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Proof of Learning (PoL)
//!
//! A novel mechanism for verifying genuine learning, extending the Proof of Gradient Quality (PoGQ)
//! concept from Mycelix-Core's Byzantine-resistant federated learning.
//!
//! ## What is Proof of Learning?
//!
//! Traditional education verification relies on:
//! - Time spent (seat time)
//! - Test scores (easily gamed)
//! - Certificates (can be faked)
//!
//! Proof of Learning instead measures:
//! - **Learning Trajectory**: How knowledge builds over time
//! - **Error Patterns**: Genuine learners make specific types of mistakes
//! - **Retention Curves**: Real learning follows forgetting/relearning patterns
//! - **Application Transfer**: Can apply knowledge to novel problems
//! - **Contribution Quality**: Teaching others demonstrates mastery
//!
//! ## Integration with MATL
//!
//! PoL integrates with the Mycelix Adaptive Trust Layer to:
//! - Weight learner contributions to federated learning
//! - Verify credential authenticity
//! - Detect cheating and gaming attempts
//! - Build reputation across the ecosystem

use serde::{Deserialize, Serialize};

/// Proof of Learning score for a learner
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofOfLearning {
    /// Learner identifier (agent pubkey as string)
    pub learner_id: String,
    /// Subject/domain being learned
    pub domain: String,
    /// Overall PoL score (0.0 - 1.0)
    pub score: f64,
    /// Component scores
    pub components: PoLComponents,
    /// Evidence supporting the score
    pub evidence: Vec<LearningEvidence>,
    /// When this proof was generated
    pub generated_at: i64,
    /// Confidence in this assessment (0.0 - 1.0)
    pub confidence: f64,
    /// Model/algorithm version used
    pub algorithm_version: String,
}

/// Component scores that make up the PoL
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PoLComponents {
    /// Learning trajectory quality (builds knowledge correctly)
    pub trajectory: f64,
    /// Error pattern authenticity (makes "real learner" mistakes)
    pub error_authenticity: f64,
    /// Retention curve alignment (follows natural forgetting/relearning)
    pub retention: f64,
    /// Transfer ability (applies knowledge to new contexts)
    pub transfer: f64,
    /// Contribution quality (helps others, explains concepts)
    pub contribution: f64,
    /// Consistency over time (not just cramming)
    pub consistency: f64,
}

impl PoLComponents {
    /// Compute weighted composite score
    pub fn composite_score(&self) -> f64 {
        // Weights based on importance and gaming resistance
        const W_TRAJECTORY: f64 = 0.20;
        const W_ERROR: f64 = 0.15;
        const W_RETENTION: f64 = 0.20;
        const W_TRANSFER: f64 = 0.25;  // Highest weight - hardest to fake
        const W_CONTRIBUTION: f64 = 0.15;
        const W_CONSISTENCY: f64 = 0.05;

        W_TRAJECTORY * self.trajectory
            + W_ERROR * self.error_authenticity
            + W_RETENTION * self.retention
            + W_TRANSFER * self.transfer
            + W_CONTRIBUTION * self.contribution
            + W_CONSISTENCY * self.consistency
    }

    /// Check if all components meet minimum threshold
    pub fn meets_minimum(&self, threshold: f64) -> bool {
        self.trajectory >= threshold
            && self.error_authenticity >= threshold
            && self.retention >= threshold
            && self.transfer >= threshold
    }
}

/// Evidence supporting a PoL claim
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LearningEvidence {
    /// Completed assessment with score
    Assessment {
        assessment_id: String,
        score: f64,
        timestamp: i64,
        time_taken_seconds: u32,
    },
    /// Learning activity record
    Activity {
        activity_type: String,
        duration_minutes: u32,
        timestamp: i64,
    },
    /// Helped another learner
    PeerHelp {
        helpee_id: String,
        topic: String,
        rating: f64,
        timestamp: i64,
    },
    /// Contributed to discussion
    Discussion {
        discussion_id: String,
        helpful_votes: u32,
        timestamp: i64,
    },
    /// Submitted FL update
    FederatedUpdate {
        round_id: String,
        gradient_quality: f64,
        timestamp: i64,
    },
    /// Earned credential
    Credential {
        credential_hash: String,
        issuer: String,
        timestamp: i64,
    },
    /// Retention test result
    RetentionTest {
        original_topic: String,
        days_since_learning: u32,
        retention_score: f64,
        timestamp: i64,
    },
    /// Transfer task completion
    TransferTask {
        source_topic: String,
        target_context: String,
        success_score: f64,
        timestamp: i64,
    },
}

/// Learner activity record for PoL analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnerActivity {
    pub learner_id: String,
    pub timestamp: i64,
    pub activity_type: ActivityType,
    pub topic: String,
    pub duration_minutes: u32,
    pub outcome: ActivityOutcome,
}

/// Types of learning activities
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivityType {
    Reading,
    Video,
    Exercise,
    Assessment,
    Project,
    Discussion,
    PeerHelp,
    Revision,
    RetentionTest,
}

/// Outcome of a learning activity
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActivityOutcome {
    /// Completed successfully
    Completed { score: Option<f64> },
    /// Partially completed
    Partial { progress: f64 },
    /// Failed/incomplete
    Failed { errors: Vec<String> },
    /// Skipped
    Skipped,
}

/// Proof of Learning Analyzer
pub struct PoLAnalyzer {
    /// Minimum activities required for analysis
    min_activities: usize,
    /// Time window for analysis (days)
    _analysis_window_days: u32,
    /// Weights for component scoring
    _weights: PoLWeights,
}

/// Configurable weights for PoL calculation
#[derive(Clone, Debug)]
pub struct PoLWeights {
    pub trajectory: f64,
    pub error_authenticity: f64,
    pub retention: f64,
    pub transfer: f64,
    pub contribution: f64,
    pub consistency: f64,
}

impl Default for PoLWeights {
    fn default() -> Self {
        Self {
            trajectory: 0.20,
            error_authenticity: 0.15,
            retention: 0.20,
            transfer: 0.25,
            contribution: 0.15,
            consistency: 0.05,
        }
    }
}

impl PoLAnalyzer {
    /// Create a new PoL analyzer with default settings
    pub fn new() -> Self {
        Self {
            min_activities: 10,
            _analysis_window_days: 90,
            _weights: PoLWeights::default(),
        }
    }

    /// Create with custom settings
    pub fn with_settings(min_activities: usize, window_days: u32, weights: PoLWeights) -> Self {
        Self {
            min_activities,
            _analysis_window_days: window_days,
            _weights: weights,
        }
    }

    /// Analyze a learner's activities and generate PoL
    /// For WASM/Holochain, use `analyze_at()` with explicit timestamp
    #[cfg(not(target_arch = "wasm32"))]
    pub fn analyze(&self, learner_id: &str, domain: &str, activities: &[LearnerActivity]) -> Option<ProofOfLearning> {
        self.analyze_at(learner_id, domain, activities, chrono::Utc::now().timestamp())
    }

    /// Analyze a learner's activities with explicit timestamp (works in WASM)
    pub fn analyze_at(&self, learner_id: &str, domain: &str, activities: &[LearnerActivity], generated_at: i64) -> Option<ProofOfLearning> {
        // Filter to relevant activities
        let relevant: Vec<_> = activities
            .iter()
            .filter(|a| a.topic.contains(domain) || domain.contains(&a.topic))
            .collect();

        if relevant.len() < self.min_activities {
            return None; // Insufficient data
        }

        // Calculate component scores
        let trajectory = self.analyze_trajectory(&relevant);
        let error_authenticity = self.analyze_error_patterns(&relevant);
        let retention = self.analyze_retention(&relevant);
        let transfer = self.analyze_transfer(&relevant);
        let contribution = self.analyze_contribution(&relevant);
        let consistency = self.analyze_consistency(&relevant);

        let components = PoLComponents {
            trajectory,
            error_authenticity,
            retention,
            transfer,
            contribution,
            consistency,
        };

        // Calculate confidence based on data quality
        let confidence = self.calculate_confidence(&relevant);

        // Collect evidence
        let evidence = self.collect_evidence(&relevant);

        Some(ProofOfLearning {
            learner_id: learner_id.to_string(),
            domain: domain.to_string(),
            score: components.composite_score(),
            components,
            evidence,
            generated_at,
            confidence,
            algorithm_version: "pol-v1.0".to_string(),
        })
    }

    /// Analyze learning trajectory - does knowledge build correctly?
    fn analyze_trajectory(&self, activities: &[&LearnerActivity]) -> f64 {
        if activities.len() < 2 {
            return 0.5; // Neutral for insufficient data
        }

        // Sort by timestamp
        let mut sorted: Vec<_> = activities.iter().collect();
        sorted.sort_by_key(|a| a.timestamp);

        // Check for progressive improvement
        let mut improvements = 0;
        let mut comparisons = 0;

        for window in sorted.windows(2) {
            if let (Some(a1), Some(a2)) = (window.get(0), window.get(1)) {
                if let (ActivityOutcome::Completed { score: Some(s1) },
                        ActivityOutcome::Completed { score: Some(s2) }) =
                    (&a1.outcome, &a2.outcome)
                {
                    comparisons += 1;
                    if s2 >= s1 {
                        improvements += 1;
                    }
                }
            }
        }

        if comparisons == 0 {
            return 0.5;
        }

        (improvements as f64 / comparisons as f64).min(1.0)
    }

    /// Analyze error patterns - genuine learners make specific types of mistakes
    fn analyze_error_patterns(&self, activities: &[&LearnerActivity]) -> f64 {
        // Count failed activities and their distribution
        let failed: Vec<_> = activities
            .iter()
            .filter(|a| matches!(a.outcome, ActivityOutcome::Failed { .. }))
            .collect();

        if failed.is_empty() {
            // No failures could indicate gaming (always skipping to easy content)
            // or genuine mastery - give benefit of doubt but not full score
            return 0.7;
        }

        // Check failure distribution - should be spread across learning, not clustered
        let failure_rate = failed.len() as f64 / activities.len() as f64;

        // Healthy failure rate is 10-30% - indicates pushing boundaries
        let optimal_failure_rate = 0.20;
        let rate_score = 1.0 - (failure_rate - optimal_failure_rate).abs() * 2.0;

        rate_score.max(0.0).min(1.0)
    }

    /// Analyze retention - follows natural forgetting/relearning curves
    fn analyze_retention(&self, activities: &[&LearnerActivity]) -> f64 {
        // Look for revision activities
        let revisions: Vec<_> = activities
            .iter()
            .filter(|a| matches!(a.activity_type, ActivityType::Revision | ActivityType::RetentionTest))
            .collect();

        if revisions.is_empty() {
            return 0.5; // No retention data
        }

        // Score based on spaced repetition pattern
        // Good learners revisit material at increasing intervals
        let revision_scores: Vec<f64> = revisions
            .iter()
            .filter_map(|a| {
                if let ActivityOutcome::Completed { score: Some(s) } = &a.outcome {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        if revision_scores.is_empty() {
            return 0.5;
        }

        // Average retention score
        revision_scores.iter().sum::<f64>() / revision_scores.len() as f64
    }

    /// Analyze transfer - can apply knowledge to new contexts
    fn analyze_transfer(&self, activities: &[&LearnerActivity]) -> f64 {
        // This is the hardest to fake - requires actual understanding
        // Look for project completions and cross-topic applications

        let projects: Vec<_> = activities
            .iter()
            .filter(|a| matches!(a.activity_type, ActivityType::Project))
            .collect();

        if projects.is_empty() {
            return 0.3; // Penalize for no practical application
        }

        let project_scores: Vec<f64> = projects
            .iter()
            .filter_map(|a| {
                if let ActivityOutcome::Completed { score: Some(s) } = &a.outcome {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();

        if project_scores.is_empty() {
            return 0.4;
        }

        project_scores.iter().sum::<f64>() / project_scores.len() as f64
    }

    /// Analyze contribution - helping others demonstrates mastery
    fn analyze_contribution(&self, activities: &[&LearnerActivity]) -> f64 {
        let contributions: Vec<_> = activities
            .iter()
            .filter(|a| matches!(a.activity_type, ActivityType::PeerHelp | ActivityType::Discussion))
            .collect();

        if contributions.is_empty() {
            return 0.5; // Neutral - not everyone is social
        }

        // More contributions = better, up to a point
        let contribution_count = contributions.len() as f64;
        let count_score = (contribution_count / 10.0).min(1.0);

        count_score
    }

    /// Analyze consistency - learning over time vs cramming
    fn analyze_consistency(&self, activities: &[&LearnerActivity]) -> f64 {
        if activities.len() < 3 {
            return 0.5;
        }

        // Sort by timestamp
        let mut sorted: Vec<_> = activities.iter().collect();
        sorted.sort_by_key(|a| a.timestamp);

        // Calculate time gaps between activities
        let gaps: Vec<i64> = sorted
            .windows(2)
            .map(|w| w[1].timestamp - w[0].timestamp)
            .collect();

        if gaps.is_empty() {
            return 0.5;
        }

        // Calculate coefficient of variation of gaps
        // Lower CV = more consistent spacing
        let mean_gap = gaps.iter().sum::<i64>() as f64 / gaps.len() as f64;
        let variance = gaps.iter().map(|&g| (g as f64 - mean_gap).powi(2)).sum::<f64>() / gaps.len() as f64;
        let std_dev = variance.sqrt();

        if mean_gap == 0.0 {
            return 0.5;
        }

        let cv = std_dev / mean_gap;

        // CV < 0.5 = very consistent, CV > 2.0 = very inconsistent
        let consistency_score = 1.0 - (cv / 2.0).min(1.0);

        consistency_score.max(0.0)
    }

    /// Calculate confidence in the PoL assessment
    fn calculate_confidence(&self, activities: &[&LearnerActivity]) -> f64 {
        // More data = higher confidence
        let data_score = (activities.len() as f64 / 50.0).min(1.0);

        // More activity types = higher confidence
        let types: std::collections::HashSet<_> = activities
            .iter()
            .map(|a| std::mem::discriminant(&a.activity_type))
            .collect();
        let diversity_score = (types.len() as f64 / 5.0).min(1.0);

        (data_score * 0.6 + diversity_score * 0.4).min(0.95) // Cap at 0.95
    }

    /// Collect evidence from activities
    fn collect_evidence(&self, activities: &[&LearnerActivity]) -> Vec<LearningEvidence> {
        let mut evidence = Vec::new();

        for activity in activities.iter().take(20) {
            // Limit evidence
            match &activity.outcome {
                ActivityOutcome::Completed { score } => {
                    if matches!(activity.activity_type, ActivityType::Assessment) {
                        evidence.push(LearningEvidence::Assessment {
                            assessment_id: format!("{}_{}", activity.topic, activity.timestamp),
                            score: score.unwrap_or(0.0),
                            timestamp: activity.timestamp,
                            time_taken_seconds: activity.duration_minutes * 60,
                        });
                    } else {
                        evidence.push(LearningEvidence::Activity {
                            activity_type: format!("{:?}", activity.activity_type),
                            duration_minutes: activity.duration_minutes,
                            timestamp: activity.timestamp,
                        });
                    }
                }
                _ => {}
            }
        }

        evidence
    }
}

impl Default for PoLAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration with MATL trust scoring
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoLMATLScore {
    /// Base MATL composite score
    pub matl_base: f64,
    /// PoL score adjustment
    pub pol_adjustment: f64,
    /// Final combined score
    pub combined: f64,
    /// Weight given to PoL in combination
    pub pol_weight: f64,
}

impl PoLMATLScore {
    /// Combine PoL with existing MATL score
    pub fn combine(matl_base: f64, pol: &ProofOfLearning, pol_weight: f64) -> Self {
        let pol_adjustment = (pol.score - 0.5) * pol_weight;
        let combined = (matl_base + pol_adjustment).clamp(0.0, 1.0);

        Self {
            matl_base,
            pol_adjustment,
            combined,
            pol_weight,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pol_components_composite() {
        let components = PoLComponents {
            trajectory: 0.8,
            error_authenticity: 0.7,
            retention: 0.9,
            transfer: 0.85,
            contribution: 0.6,
            consistency: 0.75,
        };

        let score = components.composite_score();
        assert!(score > 0.7 && score < 0.9);
    }

    #[test]
    fn test_pol_analyzer_creation() {
        let analyzer = PoLAnalyzer::new();
        assert_eq!(analyzer.min_activities, 10);
    }

    #[test]
    fn test_pol_matl_combination() {
        let pol = ProofOfLearning {
            learner_id: "test".to_string(),
            domain: "programming".to_string(),
            score: 0.8,
            components: PoLComponents::default(),
            evidence: vec![],
            generated_at: 0,
            confidence: 0.9,
            algorithm_version: "test".to_string(),
        };

        let combined = PoLMATLScore::combine(0.7, &pol, 0.3);
        assert!(combined.combined > 0.7); // Should increase due to high PoL
    }
}
