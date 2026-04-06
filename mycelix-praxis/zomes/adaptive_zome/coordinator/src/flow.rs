// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flow state analysis and optimization for adaptive learning.
//!
//! Contains: flow state detection, optimal learning window calculation,
//! flow assessment, and flow-based recommendations.
//!
//! Research: Csikszentmihalyi (1990) - Flow Theory, Optimal Experience
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
use adaptive_integrity::*;
use crate::get_recent_sessions;

// ============== Types from lib.rs (flow state analysis) ==============

/// Detailed flow state analysis based on Csikszentmihalyi's flow theory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowStateAnalysis {
    /// Current flow state classification
    pub state: FlowState,
    /// Confidence in this classification (0-1000)
    pub confidence_permille: u16,
    /// Detailed metrics
    pub metrics: FlowMetrics,
    /// Recommended adjustments
    pub adjustments: Vec<FlowAdjustment>,
    /// Estimated time in current state (seconds)
    pub state_duration_seconds: u32,
    /// Trend direction
    pub trend: FlowTrend,
}

/// Flow state classifications based on challenge/skill balance
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FlowState {
    /// High skill, low challenge - needs harder content
    Boredom,
    /// Skill slightly above challenge - relaxed engagement
    Relaxation,
    /// Perfect balance - optimal learning state
    Flow,
    /// Challenge slightly above skill - engaged but stretched
    Arousal,
    /// High challenge, moderate skill - risk of frustration
    Anxiety,
    /// Too hard - needs easier content
    Overwhelm,
    /// Beginning of session - not enough data
    Warming,
    /// Fatigue setting in - needs break
    Fatigue,
}

impl Default for FlowState {
    fn default() -> Self {
        FlowState::Warming
    }
}

/// Detailed flow metrics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowMetrics {
    /// Challenge level (0-1000) - based on content difficulty vs mastery
    pub challenge_permille: u16,
    /// Skill level (0-1000) - based on recent accuracy
    pub skill_permille: u16,
    /// Focus level (0-1000) - based on response patterns
    pub focus_permille: u16,
    /// Engagement level (0-1000) - based on time-on-task
    pub engagement_permille: u16,
    /// Streak factor (0-1000) - consecutive correct answers
    pub streak_permille: u16,
    /// Time pressure (0-1000) - response time trends
    pub time_pressure_permille: u16,
}

/// Recommended adjustment to maintain/achieve flow
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FlowAdjustment {
    /// Increase difficulty
    IncreaseDifficulty { amount_permille: u16 },
    /// Decrease difficulty
    DecreaseDifficulty { amount_permille: u16 },
    /// Take a break
    SuggestBreak { reason: String },
    /// Change content type
    ChangeContentType { suggested: ContentType },
    /// Slow down
    SlowPace { reason: String },
    /// Speed up (user seems bored)
    IncreasePace { reason: String },
    /// Switch topics (interleaving)
    SwitchTopic { reason: String },
    /// Provide encouragement
    Encourage { message: String },
    /// Maintain current approach
    Maintain,
}

/// Flow trend direction
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum FlowTrend {
    Improving,
    Stable,
    Declining,
}

/// Input for flow state analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalyzeFlowInput {
    /// Recent session analytics (last 1-3 sessions)
    pub recent_sessions: Vec<SessionAnalytics>,
    /// Current session in-progress metrics
    pub current_items_attempted: u32,
    pub current_correct_count: u32,
    pub current_avg_response_time_ms: u32,
    pub current_hints_used: u32,
    pub current_skips: u32,
    pub session_elapsed_seconds: u32,
}

/// Optimal learning window based on historical flow analysis
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimalLearningWindow {
    /// Recommended session duration
    pub optimal_duration_minutes: u16,
    /// Recommended items per session
    pub optimal_items_per_session: u16,
    /// Recommended break frequency
    pub optimal_break_frequency_minutes: u16,
    /// Best content types for this learner
    pub recommended_content_types: Vec<ContentType>,
    /// Peak productive hours (0-23)
    pub peak_hours: Vec<u8>,
    /// Confidence in these recommendations
    pub confidence_permille: u16,
}

// ============== Types (flow assessment) ==============

/// Enhanced flow state assessment result for adaptive learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowAssessment {
    /// Current flow state
    pub state: FlowState,
    /// Challenge-skill balance (-1000 to +1000, 0 = perfect)
    pub balance: i16,
    /// Recommended difficulty adjustment (-500 to +500)
    pub difficulty_adjustment: i16,
    /// Time in current state (minutes)
    pub time_in_state: u16,
    /// Engagement score (0-1000)
    pub engagement_score: u16,
    /// Recommendation for next activity
    pub recommendation: FlowRecommendation,
}

/// Flow-based recommendations
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowRecommendation {
    /// Increase challenge
    IncreaseDifficulty,
    /// Perfect, keep going
    MaintainCurrent,
    /// Decrease challenge
    DecreaseDifficulty,
    /// Take a break, reset
    TakeBreak,
    /// Try a different skill/topic
    SwitchTopic,
    /// Review fundamentals first
    ReviewBasics,
}

impl Default for FlowRecommendation {
    fn default() -> Self {
        FlowRecommendation::MaintainCurrent
    }
}

/// Input for flow assessment
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowAssessmentInput {
    /// Current skill mastery (0-1000)
    pub skill_level: u16,
    /// Current challenge difficulty (0-1000)
    pub challenge_level: u16,
    /// Recent accuracy (0-1000)
    pub recent_accuracy: u16,
    /// Response time trend (1000 = normal, >1000 = slower)
    pub response_time_factor: u16,
    /// Minutes in current session
    pub session_minutes: u16,
    /// Errors in last 5 minutes
    pub recent_errors: u8,
}

// ============== Functions ==============

/// Analyze current flow state and provide optimization recommendations
pub(crate) fn analyze_flow_state_impl(input: AnalyzeFlowInput) -> ExternResult<FlowStateAnalysis> {
    // Calculate current accuracy
    let accuracy = if input.current_items_attempted > 0 {
        (input.current_correct_count * 1000 / input.current_items_attempted) as u16
    } else {
        500 // Default neutral
    };

    // Calculate skill level from accuracy
    let skill_permille = accuracy;

    // Estimate challenge from hints/skips/response time
    let hint_factor = (input.current_hints_used * 50).min(200) as u16;
    let skip_factor = (input.current_skips * 100).min(300) as u16;
    let time_factor = if input.current_avg_response_time_ms > 5000 { 150 }
        else if input.current_avg_response_time_ms > 3000 { 100 }
        else { 0 };
    let challenge_permille = (500u16 + hint_factor + skip_factor + time_factor).min(1000);

    // Calculate focus from time on task vs interruptions
    let interruptions = input.current_hints_used + input.current_skips;
    let focus_permille = if input.current_items_attempted > 0 {
        let base_focus = 800u16;
        base_focus.saturating_sub((interruptions * 100) as u16).max(200)
    } else {
        500
    };

    // Calculate engagement from response patterns
    let engagement_permille = if input.session_elapsed_seconds > 0 && input.current_items_attempted > 0 {
        let items_per_minute = (input.current_items_attempted * 60) / input.session_elapsed_seconds.max(1);
        (items_per_minute * 100).min(1000) as u16
    } else {
        500
    };

    // Streak factor (would need consecutive correct tracking - estimate from accuracy)
    let streak_permille = if accuracy >= 900 { 900 }
        else if accuracy >= 800 { 700 }
        else if accuracy >= 700 { 500 }
        else { 300 };

    // Time pressure from response times
    let time_pressure_permille = if input.current_avg_response_time_ms < 2000 { 200 }
        else if input.current_avg_response_time_ms < 3000 { 400 }
        else if input.current_avg_response_time_ms < 5000 { 600 }
        else { 800 };

    let metrics = FlowMetrics {
        challenge_permille,
        skill_permille,
        focus_permille,
        engagement_permille,
        streak_permille,
        time_pressure_permille,
    };

    // Determine flow state from challenge/skill balance
    let balance = skill_permille as i32 - challenge_permille as i32;
    let state = determine_flow_state(balance, &metrics, &input);

    // Calculate confidence based on data available
    let confidence_permille = if input.current_items_attempted >= 10 { 900 }
        else if input.current_items_attempted >= 5 { 700 }
        else if input.current_items_attempted >= 2 { 500 }
        else { 300 };

    // Generate adjustments
    let adjustments = generate_flow_adjustments(&state, &metrics, &input);

    // Determine trend from recent sessions
    let trend = determine_flow_trend(&input.recent_sessions);

    Ok(FlowStateAnalysis {
        state,
        confidence_permille,
        metrics,
        adjustments,
        state_duration_seconds: input.session_elapsed_seconds,
        trend,
    })
}

/// Determine flow state from challenge/skill balance
fn determine_flow_state(
    balance: i32,
    metrics: &FlowMetrics,
    input: &AnalyzeFlowInput,
) -> FlowState {
    // Check for warming up (not enough data)
    if input.current_items_attempted < 3 {
        return FlowState::Warming;
    }

    // Check for fatigue (long session, declining focus)
    if input.session_elapsed_seconds > 1800 && metrics.focus_permille < 500 {
        return FlowState::Fatigue;
    }

    // Check for overwhelm (very low accuracy + many hints)
    if metrics.skill_permille < 300 && input.current_hints_used > 5 {
        return FlowState::Overwhelm;
    }

    // Flow state based on challenge/skill balance
    match balance {
        b if b > 200 => FlowState::Boredom,      // Skill >> Challenge
        b if b > 100 => FlowState::Relaxation,   // Skill > Challenge
        b if b >= -100 => FlowState::Flow,       // Balanced (±100)
        b if b >= -200 => FlowState::Arousal,    // Challenge > Skill
        b if b >= -400 => FlowState::Anxiety,    // Challenge >> Skill
        _ => FlowState::Overwhelm,               // Challenge >>> Skill
    }
}

/// Generate adjustments to improve flow
fn generate_flow_adjustments(
    state: &FlowState,
    metrics: &FlowMetrics,
    input: &AnalyzeFlowInput,
) -> Vec<FlowAdjustment> {
    let mut adjustments = Vec::new();

    match state {
        FlowState::Boredom => {
            adjustments.push(FlowAdjustment::IncreaseDifficulty {
                amount_permille: 200,
            });
            if metrics.engagement_permille < 500 {
                adjustments.push(FlowAdjustment::SwitchTopic {
                    reason: "Content may be too familiar".to_string(),
                });
            }
            adjustments.push(FlowAdjustment::IncreasePace {
                reason: "You're ready for more challenging content".to_string(),
            });
        }
        FlowState::Relaxation => {
            adjustments.push(FlowAdjustment::IncreaseDifficulty {
                amount_permille: 100,
            });
        }
        FlowState::Flow => {
            adjustments.push(FlowAdjustment::Maintain);
            if input.session_elapsed_seconds > 1500 {
                adjustments.push(FlowAdjustment::Encourage {
                    message: "Great flow! Consider a short break to maintain focus".to_string(),
                });
            }
        }
        FlowState::Arousal => {
            // Good challenge level - encourage persistence
            adjustments.push(FlowAdjustment::Encourage {
                message: "You're being stretched - this is optimal learning!".to_string(),
            });
        }
        FlowState::Anxiety => {
            adjustments.push(FlowAdjustment::DecreaseDifficulty {
                amount_permille: 100,
            });
            adjustments.push(FlowAdjustment::SlowPace {
                reason: "Take your time with these concepts".to_string(),
            });
        }
        FlowState::Overwhelm => {
            adjustments.push(FlowAdjustment::DecreaseDifficulty {
                amount_permille: 200,
            });
            adjustments.push(FlowAdjustment::ChangeContentType {
                suggested: ContentType::Lesson,
            });
            if input.session_elapsed_seconds > 600 {
                adjustments.push(FlowAdjustment::SuggestBreak {
                    reason: "Let's step back and review the fundamentals".to_string(),
                });
            }
        }
        FlowState::Fatigue => {
            adjustments.push(FlowAdjustment::SuggestBreak {
                reason: "You've been working hard! A 5-10 minute break will help retention".to_string(),
            });
            adjustments.push(FlowAdjustment::DecreaseDifficulty {
                amount_permille: 100,
            });
        }
        FlowState::Warming => {
            adjustments.push(FlowAdjustment::Maintain);
        }
    }

    adjustments
}

/// Determine trend from recent sessions
fn determine_flow_trend(sessions: &[SessionAnalytics]) -> FlowTrend {
    if sessions.len() < 2 {
        return FlowTrend::Stable;
    }

    // Compare most recent to previous sessions
    let recent_flow = sessions.first().map(|s| s.flow_balance_permille).unwrap_or(500);
    let older_flow: u16 = sessions.iter()
        .skip(1)
        .take(3)
        .map(|s| s.flow_balance_permille as u32)
        .sum::<u32>()
        .checked_div(sessions.len().min(4) as u32 - 1)
        .unwrap_or(500) as u16;

    match recent_flow as i32 - older_flow as i32 {
        d if d > 50 => FlowTrend::Improving,
        d if d < -50 => FlowTrend::Declining,
        _ => FlowTrend::Stable,
    }
}

/// Get optimal learning window based on flow history
pub(crate) fn get_optimal_learning_window_impl(_: ()) -> ExternResult<OptimalLearningWindow> {
    let sessions = get_recent_sessions(20)?;

    if sessions.is_empty() {
        // Default recommendations without data
        return Ok(OptimalLearningWindow {
            optimal_duration_minutes: 25, // Pomodoro default
            optimal_items_per_session: 20,
            optimal_break_frequency_minutes: 25,
            recommended_content_types: vec![ContentType::Lesson, ContentType::Quiz],
            peak_hours: vec![10, 14, 16], // Common productive hours
            confidence_permille: 200, // Low confidence without data
        });
    }

    // Find sessions with best flow balance
    let flow_sessions: Vec<&SessionAnalytics> = sessions.iter()
        .filter(|s| s.flow_balance_permille >= 600 && s.flow_balance_permille <= 800)
        .collect();

    // Calculate optimal duration from high-flow sessions
    let optimal_duration = if !flow_sessions.is_empty() {
        let avg_duration = flow_sessions.iter()
            .map(|s| s.duration_seconds as u32)
            .sum::<u32>() / flow_sessions.len() as u32;
        (avg_duration / 60).max(15).min(60) as u16
    } else {
        25 // Default Pomodoro
    };

    // Calculate optimal items
    let optimal_items = if !flow_sessions.is_empty() {
        let avg_items = flow_sessions.iter()
            .map(|s| s.items_completed as u32)
            .sum::<u32>() / flow_sessions.len() as u32;
        avg_items.max(10).min(50) as u16
    } else {
        20
    };

    // Find peak hours from flow sessions
    let _peak_hours: Vec<u8> = vec![10, 14, 16]; // Would need timestamp analysis

    let confidence = if flow_sessions.len() >= 5 { 800 }
        else if flow_sessions.len() >= 2 { 600 }
        else { 400 };

    Ok(OptimalLearningWindow {
        optimal_duration_minutes: optimal_duration,
        optimal_items_per_session: optimal_items,
        optimal_break_frequency_minutes: optimal_duration,
        recommended_content_types: vec![ContentType::Lesson, ContentType::Exercise, ContentType::Quiz],
        peak_hours: vec![10, 14, 16],
        confidence_permille: confidence,
    })
}

/// Assess current flow state and provide recommendations
pub(crate) fn assess_flow_state_impl(input: FlowAssessmentInput) -> ExternResult<FlowAssessment> {
    // Calculate challenge-skill balance
    // Positive = challenge > skill (potentially anxious)
    // Negative = skill > challenge (potentially bored)
    let balance = input.challenge_level as i16 - input.skill_level as i16;

    // Determine flow state based on balance and performance
    // Using existing FlowState enum: Boredom, Relaxation, Flow, Arousal, Anxiety, Overwhelm, Warming, Fatigue
    let state = if input.session_minutes < 3 {
        FlowState::Warming // Still warming up
    } else if balance > 300 || input.recent_accuracy < 400 {
        if balance > 500 || input.recent_accuracy < 200 {
            FlowState::Overwhelm
        } else {
            FlowState::Anxiety
        }
    } else if balance > 100 {
        FlowState::Arousal // Good for learning - slightly stretched
    } else if balance < -300 || input.recent_accuracy > 950 {
        FlowState::Boredom
    } else if balance < -100 {
        FlowState::Relaxation // Easy but not boring
    } else if input.response_time_factor > 1400 && input.session_minutes > 30 {
        FlowState::Fatigue // Slowing down, needs break
    } else {
        FlowState::Flow
    };

    // Calculate difficulty adjustment needed
    let difficulty_adjustment = match state {
        FlowState::Overwhelm => -400,
        FlowState::Anxiety => -200,
        FlowState::Arousal => 0, // Perfect for learning
        FlowState::Flow => 50, // Slight increase to maintain engagement
        FlowState::Relaxation => 100, // Slightly increase challenge
        FlowState::Boredom => 200,
        FlowState::Warming => 0, // Don't adjust yet
        FlowState::Fatigue => -100, // Reduce load
    };

    // Calculate engagement based on accuracy and response time
    let accuracy_factor = input.recent_accuracy as u32;
    let time_factor = if input.response_time_factor > 1500 {
        500 // Slow = disengaged
    } else if input.response_time_factor < 800 {
        900 // Fast = engaged
    } else {
        750
    };
    let engagement_score = ((accuracy_factor + time_factor as u32) / 2).min(1000) as u16;

    // Determine recommendation for all FlowState variants
    let recommendation = match state {
        FlowState::Overwhelm => {
            if input.session_minutes > 30 {
                FlowRecommendation::TakeBreak
            } else {
                FlowRecommendation::ReviewBasics
            }
        }
        FlowState::Anxiety => FlowRecommendation::DecreaseDifficulty,
        FlowState::Arousal => FlowRecommendation::MaintainCurrent, // Optimal learning zone
        FlowState::Flow => {
            if input.session_minutes > 45 {
                FlowRecommendation::TakeBreak
            } else {
                FlowRecommendation::MaintainCurrent
            }
        }
        FlowState::Relaxation => {
            if input.recent_accuracy > 900 {
                FlowRecommendation::IncreaseDifficulty
            } else {
                FlowRecommendation::MaintainCurrent
            }
        }
        FlowState::Boredom => {
            if input.recent_errors == 0 && input.recent_accuracy > 900 {
                FlowRecommendation::IncreaseDifficulty
            } else {
                FlowRecommendation::SwitchTopic
            }
        }
        FlowState::Warming => FlowRecommendation::MaintainCurrent, // Let them warm up
        FlowState::Fatigue => FlowRecommendation::TakeBreak,
    };

    Ok(FlowAssessment {
        state,
        balance,
        difficulty_adjustment,
        time_in_state: 0, // Would need state tracking over time
        engagement_score,
        recommendation,
    })
}
