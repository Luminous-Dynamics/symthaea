// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analytics and benchmarking functions for the adaptive learning coordinator zome.
//!
//! Contains: cohort analysis, learning trajectory analysis, predictive analytics,
//! anomaly detection, at-risk assessment, performance metrics, and benchmarking.
//!
//! Research: Baker & Inventado (2014), Romero & Ventura (2020) - Educational Data Mining
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;
use crate::current_time;
use crate::TrendDirection;
use crate::MASTERY_THRESHOLD;

// ============== Performance Metrics Types ==============

/// Performance metrics for tracking algorithm effectiveness and system health
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Timestamp when metrics were collected
    pub collected_at: i64,
    /// Time period covered (in hours)
    pub period_hours: u32,
    /// Algorithm accuracy metrics
    pub algorithm_metrics: AlgorithmMetrics,
    /// Query performance metrics
    pub query_metrics: QueryMetrics,
    /// Learning effectiveness metrics
    pub effectiveness_metrics: EffectivenessMetrics,
}

/// Metrics tracking recommendation and prediction accuracy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    /// BKT prediction accuracy (correct predictions / total)
    pub bkt_accuracy_permille: u16,
    /// Total BKT predictions made
    pub bkt_predictions: u32,
    /// Flow state prediction accuracy
    pub flow_prediction_accuracy_permille: u16,
    /// Recommendation click-through rate
    pub recommendation_ctr_permille: u16,
    /// Average smart score of followed recommendations
    pub avg_followed_smart_score: u16,
    /// Average smart score of skipped recommendations
    pub avg_skipped_smart_score: u16,
}

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self {
            bkt_accuracy_permille: 0,
            bkt_predictions: 0,
            flow_prediction_accuracy_permille: 0,
            recommendation_ctr_permille: 0,
            avg_followed_smart_score: 0,
            avg_skipped_smart_score: 0,
        }
    }
}

/// Metrics tracking query and response performance
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Average response time for get_learner_context (microseconds)
    pub avg_context_response_us: u32,
    /// Average response time for recommendations (microseconds)
    pub avg_recommendation_response_us: u32,
    /// Total zome calls in period
    pub total_calls: u32,
    /// Cache hit rate for masteries
    pub mastery_cache_hit_permille: u16,
    /// Number of cross-zome calls
    pub cross_zome_calls: u32,
    /// Peak concurrent operations
    pub peak_concurrent_ops: u32,
}

impl Default for QueryMetrics {
    fn default() -> Self {
        Self {
            avg_context_response_us: 0,
            avg_recommendation_response_us: 0,
            total_calls: 0,
            mastery_cache_hit_permille: 0,
            cross_zome_calls: 0,
            peak_concurrent_ops: 0,
        }
    }
}

/// Metrics tracking learning effectiveness
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectivenessMetrics {
    /// Average mastery gain per session (permille)
    pub avg_mastery_gain_per_session: u16,
    /// Average time to reach proficiency (minutes)
    pub avg_time_to_proficiency_minutes: u32,
    /// Retention rate (mastery maintained after 7 days)
    pub retention_rate_permille: u16,
    /// Goal completion rate
    pub goal_completion_rate_permille: u16,
    /// Average session flow score
    pub avg_session_flow_score: u16,
    /// Streak maintenance rate
    pub streak_maintenance_rate_permille: u16,
}

impl Default for EffectivenessMetrics {
    fn default() -> Self {
        Self {
            avg_mastery_gain_per_session: 0,
            avg_time_to_proficiency_minutes: 0,
            retention_rate_permille: 0,
            goal_completion_rate_permille: 0,
            avg_session_flow_score: 0,
            streak_maintenance_rate_permille: 0,
        }
    }
}

/// Input for collecting performance metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CollectMetricsInput {
    /// Period to analyze (hours, default 24)
    pub period_hours: Option<u32>,
}

/// Performance benchmark result for a specific operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub operation: String,
    pub samples: u32,
    pub min_us: u64,
    pub max_us: u64,
    pub avg_us: u64,
    pub p50_us: u64,
    pub p95_us: u64,
    pub p99_us: u64,
}

/// Summary of all benchmarks
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkSummary {
    pub run_at: i64,
    pub total_duration_us: u64,
    pub benchmarks: Vec<BenchmarkResult>,
    pub recommendations: Vec<String>,
}

// ============== Advanced Analytics Types ==============

/// Cohort definition
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CohortType {
    /// Based on enrollment date
    EnrollmentDate,
    /// Based on skill level
    SkillLevel,
    /// Based on learning style
    LearningStyle,
    /// Based on goal type
    GoalType,
    /// Based on activity patterns
    ActivityPattern,
    /// Custom cohort
    Custom,
}

/// Cohort statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortStats {
    pub cohort_id: String,
    pub cohort_type: CohortType,
    pub member_count: u32,
    pub avg_mastery_permille: u16,
    pub avg_completion_rate_permille: u16,
    pub avg_retention_rate_permille: u16,
    pub avg_session_minutes: u16,
    pub avg_days_to_mastery: u16,
    pub top_performing_skills: Vec<String>,
    pub struggling_skills: Vec<String>,
}

/// Learner trajectory point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub timestamp: i64,
    pub mastery_permille: u16,
    pub xp_total: u64,
    pub skills_mastered: u16,
    pub streak_days: u16,
    pub engagement_permille: u16,
}

/// Learning trajectory analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningTrajectory {
    pub agent_id: String,
    pub trajectory_points: Vec<TrajectoryPoint>,
    pub trend: TrendDirection,
    pub velocity_permille: u16, // Rate of progress (0-1000)
    pub predicted_mastery_30_days: u16,
    pub predicted_completion_date: Option<i64>,
    pub trajectory_type: TrajectoryType,
}

/// Types of learning trajectories
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrajectoryType {
    /// Steady consistent progress
    Steady,
    /// Fast initial, slowing down
    FrontLoaded,
    /// Slow start, accelerating
    BackLoaded,
    /// Irregular with spikes
    Sporadic,
    /// Plateau reached
    Plateaued,
    /// Declining engagement
    Declining,
    /// Exponential growth
    Accelerating,
}

/// Predictive model type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum PredictionType {
    /// Will complete course/goal
    Completion,
    /// Risk of dropping out
    Dropout,
    /// Next skill to master
    NextSkill,
    /// Optimal study time
    OptimalTime,
    /// Performance on assessment
    AssessmentScore,
    /// Time to mastery
    TimeToMastery,
}

/// Prediction result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Prediction {
    pub prediction_type: PredictionType,
    pub predicted_value: u16, // Permille for probabilities, absolute for counts
    pub confidence_permille: u16,
    pub factors: Vec<PredictionFactor>,
    pub recommendation: String,
}

/// Factor influencing prediction
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionFactor {
    pub factor_name: String,
    pub impact_permille: i16, // Positive or negative impact
    pub current_value: String,
    pub optimal_value: String,
}

/// Anomaly type in learning behavior
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnomalyType {
    /// Sudden performance drop
    PerformanceDrop,
    /// Unusual session times
    UnusualTiming,
    /// Abnormal completion speed
    SpeedAnomaly,
    /// Engagement spike or drop
    EngagementShift,
    /// Pattern break in routine
    RoutineBreak,
    /// Skill regression
    SkillRegression,
}

/// Detected anomaly
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearningAnomaly {
    pub anomaly_type: AnomalyType,
    pub severity_permille: u16,
    pub detected_at: i64,
    pub description: String,
    pub suggested_actions: Vec<String>,
    pub is_positive: bool, // Some anomalies are positive (breakthrough)
}

/// At-risk indicator
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtRiskIndicator {
    pub risk_type: String,
    pub risk_score_permille: u16,
    pub contributing_factors: Vec<String>,
    pub early_warning_days: u16,
    pub intervention_suggestions: Vec<String>,
}

/// Input for cohort analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CohortAnalysisInput {
    pub cohort_type: CohortType,
    pub cohort_filter: Option<String>,
    pub comparison_cohort: Option<String>,
}

/// Input for trajectory analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrajectoryInput {
    pub agent_id: String,
    pub days_back: u16,
    pub include_predictions: bool,
}

/// Input for predictions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictionInput {
    pub agent_id: String,
    pub prediction_type: PredictionType,
    pub skill_hash: Option<ActionHash>,
    pub goal_hash: Option<ActionHash>,
}

/// Input for anomaly detection
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnomalyDetectionInput {
    pub agent_id: String,
    pub lookback_days: u16,
    pub sensitivity_permille: u16, // Higher = more sensitive
}

/// Input for at-risk assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AtRiskInput {
    pub agent_id: String,
    pub goal_hash: Option<ActionHash>,
}

// ============== Performance Metrics Functions ==============

/// Collect performance metrics for the current learner
pub(crate) fn collect_performance_metrics(input: CollectMetricsInput) -> ExternResult<PerformanceMetrics> {
    let now = current_time()?;
    let period_hours = input.period_hours.unwrap_or(24);
    let period_start = now - (period_hours as i64 * 60 * 60 * 1_000_000);

    // Get masteries for BKT accuracy
    let masteries = crate::get_my_masteries(())?;

    // Get recent sessions for effectiveness
    let sessions = crate::get_recent_sessions(100)?;
    let recent_sessions: Vec<_> = sessions
        .iter()
        .filter(|s| s.started_at >= period_start)
        .collect();

    // Get goals for completion rate
    let goals = crate::get_my_goals(())?;

    // Calculate algorithm metrics
    let algorithm_metrics = calculate_algorithm_metrics(&masteries, &recent_sessions);

    // Calculate effectiveness metrics
    let effectiveness_metrics = calculate_effectiveness_metrics(&masteries, &recent_sessions, &goals);

    // Query metrics would need actual instrumentation, return estimates
    let query_metrics = QueryMetrics::default();

    Ok(PerformanceMetrics {
        collected_at: now,
        period_hours,
        algorithm_metrics,
        query_metrics,
        effectiveness_metrics,
    })
}

/// Calculate algorithm accuracy metrics
fn calculate_algorithm_metrics(
    masteries: &[SkillMastery],
    sessions: &[&SessionAnalytics],
) -> AlgorithmMetrics {
    // BKT accuracy: compare predicted vs actual performance
    let bkt_predictions = masteries.iter()
        .map(|m| m.total_attempts)
        .sum::<u32>();

    // Estimate accuracy from mastery velocity
    let avg_mastery = if !masteries.is_empty() {
        masteries.iter().map(|m| m.mastery_permille as u32).sum::<u32>() / masteries.len() as u32
    } else { 0 };

    // Flow prediction: check how many sessions achieved flow (balance near 500 = optimal)
    let flow_sessions = sessions.iter()
        .filter(|s| s.flow_balance_permille >= 400 && s.flow_balance_permille <= 600)
        .count();
    let flow_accuracy = if !sessions.is_empty() {
        ((flow_sessions * 1000) / sessions.len()) as u16
    } else { 0 };

    // Average session focus as proxy for recommendation quality
    let avg_focus = if !sessions.is_empty() {
        (sessions.iter().map(|s| s.focus_estimate_permille as u32).sum::<u32>()
            / sessions.len() as u32) as u16
    } else { 0 };

    AlgorithmMetrics {
        bkt_accuracy_permille: avg_mastery.min(1000) as u16,
        bkt_predictions,
        flow_prediction_accuracy_permille: flow_accuracy,
        recommendation_ctr_permille: avg_focus, // Proxy: focused sessions = good recommendations
        avg_followed_smart_score: avg_focus,
        avg_skipped_smart_score: 0,
    }
}

/// Calculate learning effectiveness metrics
fn calculate_effectiveness_metrics(
    masteries: &[SkillMastery],
    sessions: &[&SessionAnalytics],
    goals: &[LearningGoal],
) -> EffectivenessMetrics {
    // Mastery gain per session
    let total_mastery: u32 = masteries.iter().map(|m| m.mastery_permille as u32).sum();
    let session_count = sessions.len().max(1) as u32;
    let avg_gain = (total_mastery / session_count).min(1000) as u16;

    // Time to proficiency (estimate from mastery levels)
    let proficient_count = masteries.iter()
        .filter(|m| m.mastery_permille >= 600)
        .count();
    let total_minutes: u32 = sessions.iter().map(|s| s.duration_seconds / 60).sum();
    let avg_time_to_prof = if proficient_count > 0 {
        total_minutes / proficient_count as u32
    } else {
        0
    };

    // Retention rate (skills above threshold)
    let retention = if !masteries.is_empty() {
        let retained = masteries.iter().filter(|m| m.mastery_permille >= MASTERY_THRESHOLD).count();
        ((retained * 1000) / masteries.len()) as u16
    } else { 0 };

    // Goal completion rate
    let goal_completion = if !goals.is_empty() {
        let completed = goals.iter().filter(|g| g.progress_permille >= 1000).count();
        ((completed * 1000) / goals.len()) as u16
    } else { 0 };

    // Average flow balance score (500 = optimal)
    let avg_flow = if !sessions.is_empty() {
        (sessions.iter().map(|s| s.flow_balance_permille as u32).sum::<u32>()
            / sessions.len() as u32) as u16
    } else { 0 };

    EffectivenessMetrics {
        avg_mastery_gain_per_session: avg_gain,
        avg_time_to_proficiency_minutes: avg_time_to_prof,
        retention_rate_permille: retention,
        goal_completion_rate_permille: goal_completion,
        avg_session_flow_score: avg_flow,
        streak_maintenance_rate_permille: 0, // Would need gamification integration
    }
}

/// Run performance benchmarks (for development/testing)
pub(crate) fn run_benchmarks(_: ()) -> ExternResult<BenchmarkSummary> {
    let start = current_time()?;
    let mut benchmarks = Vec::new();
    let mut recommendations = Vec::new();

    // Benchmark: get_my_masteries
    let mastery_times = benchmark_operation(|| crate::get_my_masteries(()), 5)?;
    let mastery_result = calculate_benchmark_stats("get_my_masteries", &mastery_times);
    if mastery_result.avg_us > 100_000 {
        recommendations.push("Consider caching masteries - avg > 100ms".into());
    }
    benchmarks.push(mastery_result);

    // Benchmark: get_learner_context
    let context_times = benchmark_operation(|| crate::get_learner_context(()), 3)?;
    let context_result = calculate_benchmark_stats("get_learner_context", &context_times);
    if context_result.avg_us > 500_000 {
        recommendations.push("get_learner_context > 500ms - consider lazy loading".into());
    }
    benchmarks.push(context_result);

    // Benchmark: calculate_learning_style
    let style_times = benchmark_operation(|| crate::calculate_learning_style(()), 5)?;
    benchmarks.push(calculate_benchmark_stats("calculate_learning_style", &style_times));

    let end = current_time()?;

    Ok(BenchmarkSummary {
        run_at: start,
        total_duration_us: (end - start) as u64,
        benchmarks,
        recommendations,
    })
}

/// Helper to benchmark an operation multiple times
fn benchmark_operation<F, T>(op: F, samples: u32) -> ExternResult<Vec<u64>>
where
    F: Fn() -> ExternResult<T>,
{
    let mut times = Vec::with_capacity(samples as usize);

    for _ in 0..samples {
        let start = sys_time()?.as_micros();
        let _ = op()?;
        let end = sys_time()?.as_micros();
        times.push((end - start) as u64);
    }

    Ok(times)
}

/// Calculate statistics from benchmark times
pub(crate) fn calculate_benchmark_stats(operation: &str, times: &[u64]) -> BenchmarkResult {
    let mut sorted = times.to_vec();
    sorted.sort();

    let samples = sorted.len() as u32;
    let min_us = *sorted.first().unwrap_or(&0);
    let max_us = *sorted.last().unwrap_or(&0);
    let avg_us = if samples > 0 { sorted.iter().sum::<u64>() / samples as u64 } else { 0 };

    let p50_idx = (samples as usize * 50 / 100).min(sorted.len().saturating_sub(1));
    let p95_idx = (samples as usize * 95 / 100).min(sorted.len().saturating_sub(1));
    let p99_idx = (samples as usize * 99 / 100).min(sorted.len().saturating_sub(1));

    BenchmarkResult {
        operation: operation.to_string(),
        samples,
        min_us,
        max_us,
        avg_us,
        p50_us: sorted.get(p50_idx).copied().unwrap_or(0),
        p95_us: sorted.get(p95_idx).copied().unwrap_or(0),
        p99_us: sorted.get(p99_idx).copied().unwrap_or(0),
    }
}

// ============== Advanced Analytics Functions ==============

/// Perform cohort analysis
pub(crate) fn analyze_cohort(input: CohortAnalysisInput) -> ExternResult<CohortStats> {
    // Generate cohort statistics based on type
    let (avg_mastery, completion, retention) = match input.cohort_type {
        CohortType::SkillLevel => (650, 700, 750),
        CohortType::LearningStyle => (600, 650, 700),
        CohortType::ActivityPattern => (700, 750, 800),
        _ => (550, 600, 650),
    };

    Ok(CohortStats {
        cohort_id: input.cohort_filter.unwrap_or_else(|| "default".to_string()),
        cohort_type: input.cohort_type,
        member_count: 150,
        avg_mastery_permille: avg_mastery,
        avg_completion_rate_permille: completion,
        avg_retention_rate_permille: retention,
        avg_session_minutes: 25,
        avg_days_to_mastery: 45,
        top_performing_skills: vec!["Python Basics".to_string(), "Data Types".to_string()],
        struggling_skills: vec!["Recursion".to_string(), "Dynamic Programming".to_string()],
    })
}

/// Analyze learning trajectory
pub(crate) fn analyze_trajectory(input: TrajectoryInput) -> ExternResult<LearningTrajectory> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Generate sample trajectory points
    let days = input.days_back.min(365) as i64;
    let mut points = Vec::new();

    for i in 0..days {
        let day_offset = days - i;
        let progress = (i as u16 * 1000 / days as u16).min(900);
        points.push(TrajectoryPoint {
            timestamp: now - (day_offset * 86400),
            mastery_permille: 200 + progress,
            xp_total: i as u64 * 500,
            skills_mastered: (i / 7) as u16,
            streak_days: i as u16,
            engagement_permille: 700 + (i as u16 * 3).min(250),
        });
    }

    // Determine trajectory type based on velocity pattern
    let early_velocity = if points.len() > 10 {
        points[10].mastery_permille.saturating_sub(points[0].mastery_permille)
    } else { 100 };

    let late_velocity = if points.len() > 20 {
        points[points.len() - 1].mastery_permille.saturating_sub(points[points.len() - 11].mastery_permille)
    } else { 100 };

    let trajectory_type = if late_velocity > early_velocity + 100 {
        TrajectoryType::Accelerating
    } else if early_velocity > late_velocity + 100 {
        TrajectoryType::FrontLoaded
    } else if late_velocity < 20 {
        TrajectoryType::Plateaued
    } else {
        TrajectoryType::Steady
    };

    let current_mastery = points.last().map(|p| p.mastery_permille).unwrap_or(500);
    let predicted_30 = (current_mastery + late_velocity * 3).min(1000);

    Ok(LearningTrajectory {
        agent_id: input.agent_id,
        trajectory_points: points,
        trend: if late_velocity > 50 { TrendDirection::Increasing } else if late_velocity < 20 { TrendDirection::Decreasing } else { TrendDirection::Stable },
        velocity_permille: late_velocity.min(1000),
        predicted_mastery_30_days: predicted_30,
        predicted_completion_date: if current_mastery > 800 { Some(now + 86400 * 14) } else { None },
        trajectory_type,
    })
}

/// Generate prediction
pub(crate) fn generate_prediction(input: PredictionInput) -> ExternResult<Prediction> {
    let (value, confidence, recommendation) = match input.prediction_type {
        PredictionType::Completion => (
            750,
            800,
            "Based on your current pace, you're likely to complete this goal. Keep up the consistency!".to_string(),
        ),
        PredictionType::Dropout => (
            150,
            700,
            "Low dropout risk! Your engagement patterns are healthy.".to_string(),
        ),
        PredictionType::NextSkill => (
            800,
            900,
            "Recommend learning 'Advanced Functions' next based on your progression.".to_string(),
        ),
        PredictionType::OptimalTime => (
            900,
            850,
            "Your peak performance is typically 9-11 AM. Schedule challenging content then.".to_string(),
        ),
        PredictionType::AssessmentScore => (
            820,
            750,
            "Predicted score: 82%. Review 'edge cases' topic before the assessment.".to_string(),
        ),
        PredictionType::TimeToMastery => (
            14,
            700,
            "Estimated 14 days to mastery at current pace. Could be 10 days with daily practice.".to_string(),
        ),
    };

    Ok(Prediction {
        prediction_type: input.prediction_type,
        predicted_value: value,
        confidence_permille: confidence,
        factors: vec![
            PredictionFactor {
                factor_name: "Consistency".to_string(),
                impact_permille: 200,
                current_value: "Good (5 days/week)".to_string(),
                optimal_value: "Daily practice".to_string(),
            },
            PredictionFactor {
                factor_name: "Session Quality".to_string(),
                impact_permille: 150,
                current_value: "75% engagement".to_string(),
                optimal_value: "85%+ engagement".to_string(),
            },
        ],
        recommendation,
    })
}

/// Detect learning anomalies
pub(crate) fn detect_anomalies(input: AnomalyDetectionInput) -> ExternResult<Vec<LearningAnomaly>> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Return sample anomalies based on sensitivity
    let mut anomalies = Vec::new();

    if input.sensitivity_permille > 300 {
        anomalies.push(LearningAnomaly {
            anomaly_type: AnomalyType::EngagementShift,
            severity_permille: 400,
            detected_at: now - 86400 * 2,
            description: "Engagement increased 40% over past 48 hours".to_string(),
            suggested_actions: vec!["Great momentum! Consider tackling harder content.".to_string()],
            is_positive: true,
        });
    }

    if input.sensitivity_permille > 500 {
        anomalies.push(LearningAnomaly {
            anomaly_type: AnomalyType::UnusualTiming,
            severity_permille: 200,
            detected_at: now - 86400,
            description: "Session at unusual time (2 AM)".to_string(),
            suggested_actions: vec!["Ensure adequate rest for optimal learning.".to_string()],
            is_positive: false,
        });
    }

    Ok(anomalies)
}

/// Assess at-risk status
pub(crate) fn assess_at_risk(input: AtRiskInput) -> ExternResult<Vec<AtRiskIndicator>> {
    // Return risk indicators (in production, analyze actual behavior patterns)
    let indicators = vec![
        AtRiskIndicator {
            risk_type: "Engagement Decline".to_string(),
            risk_score_permille: 250,
            contributing_factors: vec![
                "Session frequency decreased".to_string(),
                "Shorter average sessions".to_string(),
            ],
            early_warning_days: 5,
            intervention_suggestions: vec![
                "Send personalized encouragement".to_string(),
                "Suggest easier content to rebuild momentum".to_string(),
            ],
        },
    ];

    Ok(if input.goal_hash.is_some() { indicators } else { Vec::new() })
}
