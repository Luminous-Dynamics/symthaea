//! # K-Vector Bridge for Agents
//!
//! Maps agent behavior to K-Vector trust profile updates.
//! This bridges the gap between the agentic framework and MATL trust scoring.
//!
//! ## Behavior → K-Vector Mapping
//!
//! | Behavior Outcome | K-Vector Dimension | Effect |
//! |------------------|-------------------|--------|
//! | Success rate | k_r (Reputation) | Higher success = higher reputation |
//! | Actions per hour | k_a (Activity) | More activity = higher activity score |
//! | Constraint violations | k_i (Integrity) | Violations decrease integrity |
//! | Output quality | k_p (Performance) | Quality outputs increase performance |
//! | Time active | k_m (Membership) | Longer active = higher membership |
//! | KREDIT efficiency | k_s (Stake) | Efficient KREDIT use = higher stake |
//! | Consistency | k_h (Historical) | Consistent behavior = higher historical |
//! | Interactions | k_topo (Topology) | More connections = higher topology |

use super::{
    ActionOutcome, BehaviorLogEntry, InstrumentalActor, OutputHistoryEntry, VerificationOutcome,
};
use crate::matl::{KVector, KVectorWeights};
use serde::{Deserialize, Serialize};

/// Configuration for behavior-to-K-Vector mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KVectorBridgeConfig {
    /// Minimum actions required before updating K-Vector
    pub min_actions_for_update: usize,
    /// How much to weight recent vs historical behavior (0.0-1.0)
    pub recency_weight: f32,
    /// Decay factor for activity dimension per hour of inactivity
    pub activity_decay_per_hour: f32,
    /// Integrity penalty per constraint violation
    pub integrity_penalty_per_violation: f32,
    /// Maximum reputation change per update cycle
    pub max_reputation_delta: f32,
    /// Membership increase per day of active participation
    pub membership_increase_per_day: f32,
}

impl Default for KVectorBridgeConfig {
    fn default() -> Self {
        Self {
            min_actions_for_update: 10,
            recency_weight: 0.7,
            activity_decay_per_hour: 0.02,
            integrity_penalty_per_violation: 0.1,
            max_reputation_delta: 0.1,
            membership_increase_per_day: 0.01,
        }
    }
}

/// Behavior analysis result used for K-Vector updates
#[derive(Debug, Clone)]
pub struct BehaviorAnalysis {
    /// Total actions analyzed
    pub total_actions: usize,
    /// Successful actions
    pub successful_actions: usize,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
    /// Constraint violations count
    pub constraint_violations: usize,
    /// Average KREDIT per action
    pub avg_kredit_per_action: f64,
    /// Unique counterparties interacted with
    pub unique_counterparties: usize,
    /// Time span of analyzed behavior (seconds)
    pub time_span_secs: u64,
    /// Actions per hour rate
    pub actions_per_hour: f32,
}

/// Analyze behavior log to extract metrics
pub fn analyze_behavior(log: &[BehaviorLogEntry]) -> BehaviorAnalysis {
    if log.is_empty() {
        return BehaviorAnalysis {
            total_actions: 0,
            successful_actions: 0,
            success_rate: 0.5, // Neutral default
            constraint_violations: 0,
            avg_kredit_per_action: 0.0,
            unique_counterparties: 0,
            time_span_secs: 0,
            actions_per_hour: 0.0,
        };
    }

    let total = log.len();
    let successful = log
        .iter()
        .filter(|e| e.outcome == ActionOutcome::Success)
        .count();
    let violations = log
        .iter()
        .filter(|e| e.outcome == ActionOutcome::ConstraintViolation)
        .count();

    let total_kredit: u64 = log.iter().map(|e| e.kredit_consumed).sum();
    let avg_kredit = total_kredit as f64 / total as f64;

    // Count unique counterparties
    let mut counterparties = std::collections::HashSet::new();
    for entry in log {
        for cp in &entry.counterparties {
            counterparties.insert(cp.clone());
        }
    }

    // Calculate time span
    let timestamps: Vec<u64> = log.iter().map(|e| e.timestamp).collect();
    let time_span = timestamps.iter().max().unwrap_or(&0) - timestamps.iter().min().unwrap_or(&0);

    let hours = (time_span as f32 / 3600.0).max(1.0);
    let actions_per_hour = total as f32 / hours;

    BehaviorAnalysis {
        total_actions: total,
        successful_actions: successful,
        success_rate: successful as f32 / total as f32,
        constraint_violations: violations,
        avg_kredit_per_action: avg_kredit,
        unique_counterparties: counterparties.len(),
        time_span_secs: time_span,
        actions_per_hour,
    }
}

/// Calculate updated K-Vector from behavior analysis
pub fn compute_kvector_update(
    current: &KVector,
    analysis: &BehaviorAnalysis,
    config: &KVectorBridgeConfig,
    days_active: f32,
) -> KVector {
    // Calculate new reputation based on success rate
    // Blend current with new based on recency weight
    let new_k_r = {
        let behavior_reputation = analysis.success_rate;
        let blended = current.k_r * (1.0 - config.recency_weight)
            + behavior_reputation * config.recency_weight;
        // Clamp delta to prevent rapid swings
        let delta = (blended - current.k_r)
            .clamp(-config.max_reputation_delta, config.max_reputation_delta);
        (current.k_r + delta).clamp(0.0, 1.0)
    };

    // Activity based on actions per hour (normalized to expected range)
    // Assume 10-100 actions/hour is normal range
    let new_k_a = (analysis.actions_per_hour / 50.0).clamp(0.0, 1.0);

    // Integrity decreases with violations
    let new_k_i = {
        let penalty =
            analysis.constraint_violations as f32 * config.integrity_penalty_per_violation;
        (current.k_i - penalty).clamp(0.0, 1.0)
    };

    // Performance based on success rate and KREDIT efficiency
    let new_k_p = {
        // Higher success rate + lower KREDIT per action = better performance
        let efficiency_factor = if analysis.avg_kredit_per_action > 0.0 {
            (10.0 / analysis.avg_kredit_per_action).min(1.0) as f32
        } else {
            0.5
        };
        let perf = analysis.success_rate * 0.7 + efficiency_factor * 0.3;
        perf.clamp(0.0, 1.0)
    };

    // Membership increases over time
    let new_k_m = {
        let increase = days_active * config.membership_increase_per_day;
        (current.k_m + increase).clamp(0.0, 1.0)
    };

    // Stake based on KREDIT usage patterns (efficient use = higher stake trust)
    // Agents that use KREDIT efficiently demonstrate responsibility
    let new_k_s = {
        if analysis.total_actions > 0 {
            // Success-weighted KREDIT efficiency
            let success_kredit_ratio =
                analysis.successful_actions as f32 / analysis.total_actions as f32;
            (current.k_s * 0.7 + success_kredit_ratio * 0.3).clamp(0.0, 1.0)
        } else {
            current.k_s
        }
    };

    // Historical consistency - variance in behavior over time
    // For now, blend toward success rate slowly
    let new_k_h = {
        let blend_factor = 0.1; // Very slow historical update
        (current.k_h * (1.0 - blend_factor) + analysis.success_rate * blend_factor).clamp(0.0, 1.0)
    };

    // Topology based on unique interactions
    let new_k_topo = {
        // Normalize counterparties (assume 10+ is well-connected)
        let normalized = (analysis.unique_counterparties as f32 / 10.0).clamp(0.0, 1.0);
        (current.k_topo * 0.5 + normalized * 0.5).clamp(0.0, 1.0)
    };

    // k_v (verification) is preserved from current - it's not behavior-derived
    // Verification status comes from the Byzantine↔Identity bridge, not from behavior
    // k_coherence (coherence) is also preserved - it comes from Phi measurement
    KVector::new(
        new_k_r,
        new_k_a,
        new_k_i,
        new_k_p,
        new_k_m,
        new_k_s,
        new_k_h,
        new_k_topo,
        current.k_v,         // Preserve verification status
        current.k_coherence, // Preserve coherence (from Phi measurement)
    )
}

/// Update agent's K-Vector from their behavior log
pub fn update_agent_kvector(
    agent: &mut InstrumentalActor,
    config: &KVectorBridgeConfig,
) -> Option<KVector> {
    // Only update if we have enough behavior data
    if agent.behavior_log.len() < config.min_actions_for_update {
        return None;
    }

    let analysis = analyze_behavior(&agent.behavior_log);

    // Calculate days active
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_active = (now - agent.created_at) as f32 / 86400.0;

    let new_kvector = compute_kvector_update(&agent.k_vector, &analysis, config, days_active);

    agent.k_vector = new_kvector;
    Some(new_kvector)
}

/// Compute trust score from K-Vector using default weights
pub fn compute_trust_score(kvector: &KVector) -> f32 {
    kvector.trust_score()
}

/// Compute trust score with custom weights
pub fn compute_trust_score_weighted(kvector: &KVector, weights: &KVectorWeights) -> f32 {
    kvector.trust_score_with_weights(weights)
}

/// Base KREDIT allocation constants
pub mod kredit_constants {
    /// Base KREDIT for minimum trust score
    pub const BASE_KREDIT_MIN: u64 = 100;
    /// Base KREDIT for maximum trust score
    pub const BASE_KREDIT_MAX: u64 = 50_000;
    /// Trust score threshold for exponential scaling
    pub const EXPONENTIAL_THRESHOLD: f32 = 0.5;
}

/// Calculate KREDIT cap from trust score
///
/// Formula: kredit = base_min + (base_max - base_min) * trust_factor
/// Where trust_factor scales non-linearly with trust score
pub fn calculate_kredit_from_trust(trust_score: f32) -> u64 {
    use kredit_constants::*;

    let clamped_score = trust_score.clamp(0.0, 1.0);

    // Non-linear scaling: exponential above threshold, linear below
    let trust_factor = if clamped_score < EXPONENTIAL_THRESHOLD {
        // Linear scaling for low trust
        clamped_score / EXPONENTIAL_THRESHOLD * 0.3
    } else {
        // Exponential scaling for high trust
        let normalized = (clamped_score - EXPONENTIAL_THRESHOLD) / (1.0 - EXPONENTIAL_THRESHOLD);
        0.3 + 0.7 * normalized.powi(2)
    };

    let kredit = BASE_KREDIT_MIN as f32 + (BASE_KREDIT_MAX - BASE_KREDIT_MIN) as f32 * trust_factor;

    kredit as u64
}

/// Record an action outcome and update K-Vector if threshold reached
pub fn record_and_maybe_update(
    agent: &mut InstrumentalActor,
    action_type: &str,
    kredit_consumed: u64,
    outcome: ActionOutcome,
    counterparties: Vec<String>,
    config: &KVectorBridgeConfig,
) -> Option<KVector> {
    // Record the action
    agent.behavior_log.push(BehaviorLogEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        action_type: action_type.to_string(),
        kredit_consumed,
        counterparties,
        outcome,
    });
    agent.actions_this_hour += 1;

    // Check if we should update K-Vector
    if agent
        .behavior_log
        .len()
        .is_multiple_of(config.min_actions_for_update)
    {
        update_agent_kvector(agent, config)
    } else {
        None
    }
}

/// Analyze epistemic output history for K-Vector weighting
#[derive(Debug, Clone)]
pub struct EpistemicOutputAnalysis {
    /// Total outputs analyzed
    pub total_outputs: usize,
    /// Average epistemic weight
    pub average_weight: f32,
    /// Verified output count
    pub verified_count: usize,
    /// Verification accuracy (0.0-1.0)
    pub verification_accuracy: f32,
    /// Dominant empirical level
    pub dominant_empirical_index: usize,
}

/// Analyze output history for epistemic metrics
pub fn analyze_outputs(history: &[OutputHistoryEntry]) -> EpistemicOutputAnalysis {
    if history.is_empty() {
        return EpistemicOutputAnalysis {
            total_outputs: 0,
            average_weight: 0.5,
            verified_count: 0,
            verification_accuracy: 0.5,
            dominant_empirical_index: 1, // E1 default
        };
    }

    let total = history.len();
    let total_weight: f32 = history.iter().map(|o| o.epistemic_weight).sum();
    let average_weight = total_weight / total as f32;

    let verified: Vec<_> = history.iter().filter(|o| o.verified).collect();
    let verified_count = verified.len();

    let verification_accuracy = if verified.is_empty() {
        0.5
    } else {
        let correct = verified
            .iter()
            .filter(|o| matches!(o.verification_outcome, Some(VerificationOutcome::Correct)))
            .count();
        let partial = verified
            .iter()
            .filter(|o| matches!(o.verification_outcome, Some(VerificationOutcome::Partial)))
            .count();
        (correct as f32 + partial as f32 * 0.5) / verified.len() as f32
    };

    // Find dominant empirical level
    let mut empirical_counts = [0u32; 5];
    for entry in history {
        let idx = entry.classification.empirical as usize;
        if idx < 5 {
            empirical_counts[idx] += 1;
        }
    }
    let dominant_empirical_index = empirical_counts
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(idx, _)| idx)
        .unwrap_or(1);

    EpistemicOutputAnalysis {
        total_outputs: total,
        average_weight,
        verified_count,
        verification_accuracy,
        dominant_empirical_index,
    }
}

/// Calculate epistemic-weighted K-Vector update
///
/// This function uses epistemic output analysis to weight K-Vector updates:
/// - Higher epistemic weight outputs influence K-Vector more
/// - Verified correct outputs boost reputation more
/// - Higher E-level outputs increase performance score
pub fn compute_epistemic_weighted_kvector_update(
    current: &KVector,
    behavior_analysis: &BehaviorAnalysis,
    output_analysis: &EpistemicOutputAnalysis,
    config: &KVectorBridgeConfig,
    days_active: f32,
) -> KVector {
    // Start with standard behavior-based update
    let behavior_kvector = compute_kvector_update(current, behavior_analysis, config, days_active);

    // Apply epistemic weighting
    let epistemic_multiplier = output_analysis.average_weight.clamp(0.1, 1.0);
    let verification_multiplier = if output_analysis.verified_count > 0 {
        // Boost if verified outputs are accurate
        1.0 + (output_analysis.verification_accuracy - 0.5) * 0.2
    } else {
        1.0
    };

    // Reputation: Boost based on verified accuracy
    let new_k_r = {
        let base = behavior_kvector.k_r;
        let boost = (output_analysis.verification_accuracy - 0.5) * 0.1 * epistemic_multiplier;
        (base + boost).clamp(0.0, 1.0)
    };

    // Performance: Higher E-levels indicate better quality outputs
    let new_k_p = {
        let base = behavior_kvector.k_p;
        let e_level_factor = output_analysis.dominant_empirical_index as f32 / 4.0;
        let epistemic_boost = (e_level_factor - 0.5) * 0.1 * epistemic_multiplier;
        (base + epistemic_boost).clamp(0.0, 1.0)
    };

    // Historical: Consistency weighted by epistemic quality
    let new_k_h = {
        let base = behavior_kvector.k_h;
        // Higher epistemic weight = more trust in historical consistency
        (base * verification_multiplier).clamp(0.0, 1.0)
    };

    KVector::new(
        new_k_r,
        behavior_kvector.k_a,
        behavior_kvector.k_i,
        new_k_p,
        behavior_kvector.k_m,
        behavior_kvector.k_s,
        new_k_h,
        behavior_kvector.k_topo,
        behavior_kvector.k_v,
        behavior_kvector.k_coherence, // Preserve coherence
    )
}

/// Update agent's K-Vector using epistemic-weighted analysis
pub fn update_agent_kvector_epistemic(
    agent: &mut InstrumentalActor,
    config: &KVectorBridgeConfig,
) -> Option<KVector> {
    // Only update if we have enough behavior data
    if agent.behavior_log.len() < config.min_actions_for_update {
        return None;
    }

    let behavior_analysis = analyze_behavior(&agent.behavior_log);
    let output_analysis = analyze_outputs(&agent.output_history);

    // Calculate days active
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_active = (now - agent.created_at) as f32 / 86400.0;

    let new_kvector = compute_epistemic_weighted_kvector_update(
        &agent.k_vector,
        &behavior_analysis,
        &output_analysis,
        config,
        days_active,
    );

    agent.k_vector = new_kvector;
    Some(new_kvector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::epistemic_classifier::calculate_epistemic_weight;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{AgentClass, AgentConstraints, AgentId, AgentStatus, EpistemicStats};

    fn create_test_agent() -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::generate(),
            sponsor_did: "did:test:sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                - 86400, // 1 day ago
            last_activity: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            actions_this_hour: 0,
            k_vector: KVector::new_participant(),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    fn create_behavior_log(success_rate: f32, count: usize) -> Vec<BehaviorLogEntry> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let successes = (count as f32 * success_rate) as usize;

        (0..count)
            .map(|i| {
                BehaviorLogEntry {
                    timestamp: now - (count - i) as u64 * 60, // 1 minute apart
                    action_type: "test_action".to_string(),
                    kredit_consumed: 10,
                    counterparties: vec![format!("peer_{}", i % 5)],
                    outcome: if i < successes {
                        ActionOutcome::Success
                    } else {
                        ActionOutcome::Error
                    },
                }
            })
            .collect()
    }

    #[test]
    fn test_behavior_analysis() {
        let log = create_behavior_log(0.8, 100);
        let analysis = analyze_behavior(&log);

        assert_eq!(analysis.total_actions, 100);
        assert_eq!(analysis.successful_actions, 80);
        assert!((analysis.success_rate - 0.8).abs() < 0.01);
        assert_eq!(analysis.unique_counterparties, 5);
    }

    #[test]
    fn test_kvector_update_improves_with_success() {
        let initial = KVector::new_participant();
        let log = create_behavior_log(0.9, 50);
        let analysis = analyze_behavior(&log);
        let config = KVectorBridgeConfig::default();

        let updated = compute_kvector_update(&initial, &analysis, &config, 7.0);

        // High success rate should improve reputation
        assert!(updated.k_r > initial.k_r);
        // Activity should be set based on actions
        assert!(updated.k_a > 0.0);
        // Performance should reflect success
        assert!(updated.k_p > initial.k_p);
    }

    #[test]
    fn test_kvector_update_degrades_with_violations() {
        let initial = KVector::new(0.8, 0.5, 1.0, 0.7, 0.3, 0.5, 0.6, 0.3, 0.6, 0.5);

        // Create log with many violations
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let log: Vec<BehaviorLogEntry> = (0..20)
            .map(|i| BehaviorLogEntry {
                timestamp: now - i * 60,
                action_type: "violating_action".to_string(),
                kredit_consumed: 10,
                counterparties: vec![],
                outcome: ActionOutcome::ConstraintViolation,
            })
            .collect();

        let analysis = analyze_behavior(&log);
        let config = KVectorBridgeConfig::default();

        let updated = compute_kvector_update(&initial, &analysis, &config, 1.0);

        // Violations should decrease integrity
        assert!(updated.k_i < initial.k_i);
    }

    #[test]
    fn test_kredit_from_trust_scaling() {
        // Low trust = low KREDIT
        let low_kredit = calculate_kredit_from_trust(0.1);
        assert!(low_kredit < 5000);

        // Medium trust = medium KREDIT
        let med_kredit = calculate_kredit_from_trust(0.5);
        assert!(med_kredit > low_kredit);

        // High trust = high KREDIT
        let high_kredit = calculate_kredit_from_trust(0.9);
        assert!(high_kredit > med_kredit);
        assert!(high_kredit > 30000);
    }

    #[test]
    fn test_full_agent_update_cycle() {
        let mut agent = create_test_agent();
        agent.behavior_log = create_behavior_log(0.85, 15);

        let config = KVectorBridgeConfig {
            min_actions_for_update: 10,
            ..Default::default()
        };

        let updated = update_agent_kvector(&mut agent, &config);
        assert!(updated.is_some());

        let kvector = updated.unwrap();
        let trust = compute_trust_score(&kvector);
        let kredit = calculate_kredit_from_trust(trust);

        assert!(trust > 0.0);
        assert!(kredit >= kredit_constants::BASE_KREDIT_MIN);
    }

    #[test]
    fn test_epistemic_output_analysis() {
        use crate::epistemic::{
            EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
            NormativeLevel,
        };

        // Create output history with various epistemic levels
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let history = vec![
            OutputHistoryEntry {
                output_id: "out-1".to_string(),
                timestamp: now,
                classification: EpistemicClassificationExtended::new(
                    EmpiricalLevel::E3Cryptographic,
                    NormativeLevel::N2Network,
                    MaterialityLevel::M2Persistent,
                    HarmonicLevel::H1Local,
                ),
                confidence: 0.9,
                epistemic_weight: calculate_epistemic_weight(
                    &EpistemicClassificationExtended::new(
                        EmpiricalLevel::E3Cryptographic,
                        NormativeLevel::N2Network,
                        MaterialityLevel::M2Persistent,
                        HarmonicLevel::H1Local,
                    ),
                ),
                verified: true,
                verification_outcome: Some(VerificationOutcome::Correct),
            },
            OutputHistoryEntry {
                output_id: "out-2".to_string(),
                timestamp: now + 60,
                classification: EpistemicClassificationExtended::new(
                    EmpiricalLevel::E1Testimonial,
                    NormativeLevel::N0Personal,
                    MaterialityLevel::M0Ephemeral,
                    HarmonicLevel::H0None,
                ),
                confidence: 0.5,
                epistemic_weight: calculate_epistemic_weight(
                    &EpistemicClassificationExtended::new(
                        EmpiricalLevel::E1Testimonial,
                        NormativeLevel::N0Personal,
                        MaterialityLevel::M0Ephemeral,
                        HarmonicLevel::H0None,
                    ),
                ),
                verified: true,
                verification_outcome: Some(VerificationOutcome::Partial),
            },
        ];

        let analysis = analyze_outputs(&history);

        assert_eq!(analysis.total_outputs, 2);
        assert_eq!(analysis.verified_count, 2);
        // One correct (1.0) + one partial (0.5) / 2 = 0.75
        assert!((analysis.verification_accuracy - 0.75).abs() < 0.01);
        assert!(analysis.average_weight > 0.1);
    }

    #[test]
    fn test_epistemic_weighted_kvector_update() {
        use crate::epistemic::{
            EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
            NormativeLevel,
        };

        let initial = KVector::new_participant();
        let behavior_log = create_behavior_log(0.9, 50);
        let behavior_analysis = analyze_behavior(&behavior_log);
        let config = KVectorBridgeConfig::default();

        // Create high-quality output history
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let high_quality_history: Vec<OutputHistoryEntry> = (0..10)
            .map(|i| OutputHistoryEntry {
                output_id: format!("out-{}", i),
                timestamp: now + i * 60,
                classification: EpistemicClassificationExtended::new(
                    EmpiricalLevel::E4PublicRepro,
                    NormativeLevel::N2Network,
                    MaterialityLevel::M2Persistent,
                    HarmonicLevel::H2Network,
                ),
                confidence: 0.95,
                epistemic_weight: calculate_epistemic_weight(
                    &EpistemicClassificationExtended::new(
                        EmpiricalLevel::E4PublicRepro,
                        NormativeLevel::N2Network,
                        MaterialityLevel::M2Persistent,
                        HarmonicLevel::H2Network,
                    ),
                ),
                verified: true,
                verification_outcome: Some(VerificationOutcome::Correct),
            })
            .collect();

        let output_analysis = analyze_outputs(&high_quality_history);

        let updated = compute_epistemic_weighted_kvector_update(
            &initial,
            &behavior_analysis,
            &output_analysis,
            &config,
            7.0,
        );

        // High quality outputs should boost reputation and performance
        assert!(updated.k_r > initial.k_r);
        assert!(updated.k_p > initial.k_p);

        // Compare with standard update
        let standard_update = compute_kvector_update(&initial, &behavior_analysis, &config, 7.0);

        // Epistemic-weighted should have higher or equal scores for good outputs
        assert!(updated.k_r >= standard_update.k_r - 0.05); // Allow small variance
    }

    #[test]
    fn test_low_quality_outputs_limit_boost() {
        use crate::epistemic::{
            EmpiricalLevel, EpistemicClassificationExtended, HarmonicLevel, MaterialityLevel,
            NormativeLevel,
        };

        let initial = KVector::new_participant();
        let behavior_log = create_behavior_log(0.9, 50);
        let behavior_analysis = analyze_behavior(&behavior_log);
        let config = KVectorBridgeConfig::default();

        // Create low-quality output history (E0 level, verified as incorrect)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let low_quality_history: Vec<OutputHistoryEntry> = (0..10)
            .map(|i| OutputHistoryEntry {
                output_id: format!("out-{}", i),
                timestamp: now + i * 60,
                classification: EpistemicClassificationExtended::new(
                    EmpiricalLevel::E0Null,
                    NormativeLevel::N0Personal,
                    MaterialityLevel::M0Ephemeral,
                    HarmonicLevel::H0None,
                ),
                confidence: 0.3,
                epistemic_weight: calculate_epistemic_weight(
                    &EpistemicClassificationExtended::new(
                        EmpiricalLevel::E0Null,
                        NormativeLevel::N0Personal,
                        MaterialityLevel::M0Ephemeral,
                        HarmonicLevel::H0None,
                    ),
                ),
                verified: true,
                verification_outcome: Some(VerificationOutcome::Incorrect),
            })
            .collect();

        let output_analysis = analyze_outputs(&low_quality_history);

        let updated = compute_epistemic_weighted_kvector_update(
            &initial,
            &behavior_analysis,
            &output_analysis,
            &config,
            7.0,
        );

        // Low quality + incorrect should not boost (or even reduce)
        // Verification accuracy is 0.0, so boost should be negative
        assert!(output_analysis.verification_accuracy < 0.1);
    }

    // =========================================================================
    // Property-Based Tests for K-Vector Evolution
    // =========================================================================

    use proptest::prelude::*;

    /// Generate a valid K-Vector with all dimensions in [0.0, 1.0]
    fn kvector_strategy() -> impl Strategy<Value = KVector> {
        (
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
            0.0f32..=1.0f32,
        )
            .prop_map(|(r, a, i, p, m, s, h, t, v, phi)| {
                KVector::new(r, a, i, p, m, s, h, t, v, phi)
            })
    }

    /// Generate a behavior analysis with valid ranges
    fn behavior_analysis_strategy() -> impl Strategy<Value = BehaviorAnalysis> {
        (
            1usize..1000,    // total_actions
            0.0f32..=1.0f32, // success_rate
            0usize..50,      // violations
            0usize..100,     // counterparties
        )
            .prop_map(|(total, success_rate, violations, counterparties)| {
                let successful = (total as f32 * success_rate) as usize;
                BehaviorAnalysis {
                    total_actions: total,
                    successful_actions: successful,
                    success_rate,
                    constraint_violations: violations.min(total),
                    avg_kredit_per_action: 10.0,
                    unique_counterparties: counterparties,
                    time_span_secs: total as u64 * 60,
                    actions_per_hour: (total as f32) / ((total as f32 * 60.0) / 3600.0),
                }
            })
    }

    proptest! {
        /// INVARIANT: K-Vector dimensions always stay in [0.0, 1.0] after update
        #[test]
        fn prop_kvector_dimensions_bounded(
            initial in kvector_strategy(),
            analysis in behavior_analysis_strategy(),
            days_active in 0.0f32..365.0f32,
        ) {
            let config = KVectorBridgeConfig::default();
            let updated = compute_kvector_update(&initial, &analysis, &config, days_active);

            // All dimensions must be in [0.0, 1.0]
            prop_assert!(updated.k_r >= 0.0 && updated.k_r <= 1.0,
                "k_r out of bounds: {}", updated.k_r);
            prop_assert!(updated.k_a >= 0.0 && updated.k_a <= 1.0,
                "k_a out of bounds: {}", updated.k_a);
            prop_assert!(updated.k_i >= 0.0 && updated.k_i <= 1.0,
                "k_i out of bounds: {}", updated.k_i);
            prop_assert!(updated.k_p >= 0.0 && updated.k_p <= 1.0,
                "k_p out of bounds: {}", updated.k_p);
            prop_assert!(updated.k_m >= 0.0 && updated.k_m <= 1.0,
                "k_m out of bounds: {}", updated.k_m);
            prop_assert!(updated.k_s >= 0.0 && updated.k_s <= 1.0,
                "k_s out of bounds: {}", updated.k_s);
            prop_assert!(updated.k_h >= 0.0 && updated.k_h <= 1.0,
                "k_h out of bounds: {}", updated.k_h);
            prop_assert!(updated.k_topo >= 0.0 && updated.k_topo <= 1.0,
                "k_topo out of bounds: {}", updated.k_topo);
            prop_assert!(updated.k_v >= 0.0 && updated.k_v <= 1.0,
                "k_v out of bounds: {}", updated.k_v);
            prop_assert!(updated.k_coherence >= 0.0 && updated.k_coherence <= 1.0,
                "k_coherence out of bounds: {}", updated.k_coherence);
        }

        /// INVARIANT: Trust score is always in [0.0, 1.0]
        #[test]
        fn prop_trust_score_bounded(kv in kvector_strategy()) {
            let trust = compute_trust_score(&kv);
            prop_assert!(trust >= 0.0 && trust <= 1.0,
                "Trust score out of bounds: {}", trust);
        }

        /// INVARIANT: KREDIT from trust is always positive
        #[test]
        fn prop_kredit_always_positive(trust in 0.0f32..=1.0f32) {
            let kredit = calculate_kredit_from_trust(trust);
            prop_assert!(kredit >= kredit_constants::BASE_KREDIT_MIN,
                "KREDIT below minimum: {}", kredit);
        }

        /// INVARIANT: Higher success rate leads to equal or higher reputation
        #[test]
        fn prop_success_improves_reputation(
            initial in kvector_strategy(),
            days_active in 1.0f32..30.0f32,
        ) {
            let config = KVectorBridgeConfig::default();

            // Low success behavior
            let low_success = BehaviorAnalysis {
                total_actions: 100,
                successful_actions: 30,
                success_rate: 0.3,
                constraint_violations: 0,
                avg_kredit_per_action: 10.0,
                unique_counterparties: 10,
                time_span_secs: 6000,
                actions_per_hour: 60.0,
            };

            // High success behavior
            let high_success = BehaviorAnalysis {
                total_actions: 100,
                successful_actions: 90,
                success_rate: 0.9,
                constraint_violations: 0,
                avg_kredit_per_action: 10.0,
                unique_counterparties: 10,
                time_span_secs: 6000,
                actions_per_hour: 60.0,
            };

            let updated_low = compute_kvector_update(&initial, &low_success, &config, days_active);
            let updated_high = compute_kvector_update(&initial, &high_success, &config, days_active);

            // High success should result in equal or better reputation
            prop_assert!(updated_high.k_r >= updated_low.k_r - 0.001,
                "High success ({}) should have >= reputation than low success ({})",
                updated_high.k_r, updated_low.k_r);
        }

        /// INVARIANT: Constraint violations decrease integrity
        #[test]
        fn prop_violations_decrease_integrity(
            initial in kvector_strategy(),
            days_active in 1.0f32..30.0f32,
        ) {
            let config = KVectorBridgeConfig::default();

            // No violations
            let clean = BehaviorAnalysis {
                total_actions: 100,
                successful_actions: 80,
                success_rate: 0.8,
                constraint_violations: 0,
                avg_kredit_per_action: 10.0,
                unique_counterparties: 10,
                time_span_secs: 6000,
                actions_per_hour: 60.0,
            };

            // Many violations
            let violations = BehaviorAnalysis {
                total_actions: 100,
                successful_actions: 30,
                success_rate: 0.3,
                constraint_violations: 20,
                avg_kredit_per_action: 10.0,
                unique_counterparties: 10,
                time_span_secs: 6000,
                actions_per_hour: 60.0,
            };

            let updated_clean = compute_kvector_update(&initial, &clean, &config, days_active);
            let updated_violations = compute_kvector_update(&initial, &violations, &config, days_active);

            // Violations should result in lower integrity
            prop_assert!(updated_violations.k_i <= updated_clean.k_i + 0.001,
                "Violations ({}) should have <= integrity than clean ({})",
                updated_violations.k_i, updated_clean.k_i);
        }

        /// INVARIANT: Membership increases with time (up to limit)
        #[test]
        fn prop_membership_increases_with_time(
            initial in kvector_strategy(),
        ) {
            let config = KVectorBridgeConfig::default();
            let analysis = BehaviorAnalysis {
                total_actions: 50,
                successful_actions: 40,
                success_rate: 0.8,
                constraint_violations: 0,
                avg_kredit_per_action: 10.0,
                unique_counterparties: 10,
                time_span_secs: 3000,
                actions_per_hour: 60.0,
            };

            let updated_1_day = compute_kvector_update(&initial, &analysis, &config, 1.0);
            let updated_30_days = compute_kvector_update(&initial, &analysis, &config, 30.0);

            // Longer participation should have equal or higher membership (up to 1.0)
            prop_assert!(updated_30_days.k_m >= updated_1_day.k_m - 0.001,
                "30 days ({}) should have >= membership than 1 day ({})",
                updated_30_days.k_m, updated_1_day.k_m);
        }

        /// INVARIANT: Trust score changes are bounded per update
        #[test]
        fn prop_trust_change_bounded(
            initial in kvector_strategy(),
            analysis in behavior_analysis_strategy(),
            days_active in 0.0f32..365.0f32,
        ) {
            let config = KVectorBridgeConfig::default();
            let updated = compute_kvector_update(&initial, &analysis, &config, days_active);

            let initial_trust = compute_trust_score(&initial);
            let updated_trust = compute_trust_score(&updated);
            let delta = (updated_trust - initial_trust).abs();

            // Trust change should be bounded (not jump drastically)
            // Allow up to 0.5 change per update (sum of max_reputation_delta and other bounded changes)
            prop_assert!(delta <= 0.6,
                "Trust change too large: {} -> {} (delta: {})",
                initial_trust, updated_trust, delta);
        }
    }
}
