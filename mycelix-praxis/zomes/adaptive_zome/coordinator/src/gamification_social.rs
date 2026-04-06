// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gamification and social features for the adaptive learning coordinator zome.
//!
//! Contains: leaderboards, achievements, social comparison, active challenges.
//!
//! Research: Deterding (2011), Hamari (2014) - Gamification effectiveness
//! Social comparison: Festinger (1954) - Social Comparison Theory
//!
//! Extracted from lib.rs as a pure structural refactor — no logic changes.

use hdk::prelude::*;
#[allow(unused_imports)]
use adaptive_integrity::*;

// ============== Types ==============

/// Leaderboard type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum LeaderboardType {
    /// Global ranking across all learners
    Global,
    /// Within a specific course or skill
    Course,
    /// Friends/connections only
    Friends,
    /// Weekly reset
    Weekly,
    /// Monthly reset
    Monthly,
    /// All-time
    AllTime,
}

/// Leaderboard entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub rank: u32,
    pub agent_id: String,
    pub display_name: String,
    pub score: u64,
    pub xp_total: u64,
    pub streak_days: u16,
    pub badges_earned: u16,
    pub mastery_skills: u16,
}

/// Achievement category
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AchievementCategory {
    Learning,      // Complete lessons, master skills
    Consistency,   // Streaks, daily goals
    Social,        // Help others, collaborate
    Mastery,       // Deep expertise achievements
    Exploration,   // Try new areas
    Speed,         // Fast completion
    Accuracy,      // High precision
    Creative,      // Unique solutions
}

/// Achievement tier
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AchievementTier {
    Bronze,
    Silver,
    Gold,
    Platinum,
    Diamond,
    Legendary,
}

impl AchievementTier {
    pub fn xp_multiplier(&self) -> u16 {
        match self {
            AchievementTier::Bronze => 100,     // 1x
            AchievementTier::Silver => 150,     // 1.5x
            AchievementTier::Gold => 200,       // 2x
            AchievementTier::Platinum => 300,   // 3x
            AchievementTier::Diamond => 500,    // 5x
            AchievementTier::Legendary => 1000, // 10x
        }
    }
}

/// Achievement definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Achievement {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: AchievementCategory,
    pub tier: AchievementTier,
    pub xp_reward: u32,
    pub badge_icon: String,
    pub requirements: Vec<AchievementRequirement>,
    pub is_hidden: bool, // Secret achievements
    pub is_stackable: bool, // Can earn multiple times
}

/// Achievement requirement
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AchievementRequirement {
    pub metric: String,
    pub threshold: u32,
    pub comparison: String, // "gte", "lte", "eq"
}

/// Challenge type
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChallengeType {
    /// Daily challenge
    Daily,
    /// Weekly challenge
    Weekly,
    /// Limited-time event
    Event,
    /// Skill-specific challenge
    Skill,
    /// Community goal
    Community,
    /// 1v1 or small group
    Versus,
}

/// Challenge definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Challenge {
    pub id: String,
    pub challenge_type: ChallengeType,
    pub title: String,
    pub description: String,
    pub objectives: Vec<ChallengeObjective>,
    pub start_timestamp: i64,
    pub end_timestamp: i64,
    pub xp_reward: u32,
    pub bonus_reward_permille: u16, // Extra XP for early/perfect completion
    pub participants: u32,
    pub completers: u32,
}

/// Challenge objective
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChallengeObjective {
    pub description: String,
    pub target_value: u32,
    pub current_value: u32,
    pub is_completed: bool,
}

/// Social comparison type (based on Festinger's Social Comparison Theory)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SocialComparisonType {
    /// Compare with those slightly better (motivation)
    UpwardClose,
    /// Compare with top performers (inspiration)
    UpwardFar,
    /// Compare with those similar (validation)
    Lateral,
    /// Compare with those slightly behind (confidence)
    DownwardClose,
    /// No comparison (self-referenced growth)
    SelfReferenced,
}

/// Social comparison result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocialComparison {
    pub comparison_type: SocialComparisonType,
    pub reference_group: String,
    pub your_percentile: u16,
    pub relative_performance_permille: u16,
    pub motivational_message: String,
    pub suggested_goals: Vec<String>,
}

/// Input for leaderboard generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LeaderboardInput {
    pub leaderboard_type: LeaderboardType,
    pub skill_hash: Option<ActionHash>,
    pub limit: u8,
    pub requesting_agent: String,
}

/// Input for achievement check
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AchievementCheckInput {
    pub agent_id: String,
    pub category: Option<AchievementCategory>,
    pub include_hidden: bool,
}

/// Input for social comparison
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SocialComparisonInput {
    pub agent_id: String,
    pub metric: String, // "xp", "mastery", "streak", "badges"
    pub preferred_comparison: SocialComparisonType,
    pub current_value: u64,
}

/// Active challenge for a learner
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveChallenge {
    pub challenge: Challenge,
    pub progress_permille: u16,
    pub time_remaining_seconds: i64,
    pub is_completed: bool,
    pub bonus_eligible: bool,
}

/// Input for getting active challenges
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActiveChallengesInput {
    pub agent_id: String,
    pub include_completed: bool,
}

// ============== Functions ==============

/// Generate leaderboard
pub(crate) fn get_leaderboard(input: LeaderboardInput) -> ExternResult<Vec<LeaderboardEntry>> {
    // In production, this would query the DHT for learner profiles
    // For now, return sample structure
    let mut entries = Vec::new();

    // Generate sample entries based on type
    let count = input.limit.min(100) as usize;
    for i in 0..count {
        entries.push(LeaderboardEntry {
            rank: (i + 1) as u32,
            agent_id: format!("agent_{}", i),
            display_name: format!("Learner {}", i + 1),
            score: 10000u64.saturating_sub(i as u64 * 500),
            xp_total: 50000u64.saturating_sub(i as u64 * 2000),
            streak_days: (30u16).saturating_sub(i as u16),
            badges_earned: (20u16).saturating_sub(i as u16),
            mastery_skills: (15u16).saturating_sub(i as u16 / 2),
        });
    }

    Ok(entries)
}

/// Check available achievements
pub(crate) fn check_achievements(_input: AchievementCheckInput) -> ExternResult<Vec<Achievement>> {
    // Define standard achievements
    let achievements = vec![
        Achievement {
            id: "first_lesson".to_string(),
            name: "First Steps".to_string(),
            description: "Complete your first lesson".to_string(),
            category: AchievementCategory::Learning,
            tier: AchievementTier::Bronze,
            xp_reward: 100,
            badge_icon: "🎓".to_string(),
            requirements: vec![AchievementRequirement {
                metric: "lessons_completed".to_string(),
                threshold: 1,
                comparison: "gte".to_string(),
            }],
            is_hidden: false,
            is_stackable: false,
        },
        Achievement {
            id: "week_streak".to_string(),
            name: "Week Warrior".to_string(),
            description: "Maintain a 7-day learning streak".to_string(),
            category: AchievementCategory::Consistency,
            tier: AchievementTier::Silver,
            xp_reward: 500,
            badge_icon: "🔥".to_string(),
            requirements: vec![AchievementRequirement {
                metric: "streak_days".to_string(),
                threshold: 7,
                comparison: "gte".to_string(),
            }],
            is_hidden: false,
            is_stackable: false,
        },
        Achievement {
            id: "master_skill".to_string(),
            name: "Skill Master".to_string(),
            description: "Achieve 90%+ mastery in any skill".to_string(),
            category: AchievementCategory::Mastery,
            tier: AchievementTier::Gold,
            xp_reward: 1000,
            badge_icon: "⭐".to_string(),
            requirements: vec![AchievementRequirement {
                metric: "max_mastery".to_string(),
                threshold: 900,
                comparison: "gte".to_string(),
            }],
            is_hidden: false,
            is_stackable: true,
        },
        Achievement {
            id: "helper".to_string(),
            name: "Community Helper".to_string(),
            description: "Help 10 other learners".to_string(),
            category: AchievementCategory::Social,
            tier: AchievementTier::Gold,
            xp_reward: 750,
            badge_icon: "🤝".to_string(),
            requirements: vec![AchievementRequirement {
                metric: "peers_helped".to_string(),
                threshold: 10,
                comparison: "gte".to_string(),
            }],
            is_hidden: false,
            is_stackable: false,
        },
        Achievement {
            id: "speed_demon".to_string(),
            name: "Speed Demon".to_string(),
            description: "Complete a lesson 50% faster than average".to_string(),
            category: AchievementCategory::Speed,
            tier: AchievementTier::Silver,
            xp_reward: 300,
            badge_icon: "⚡".to_string(),
            requirements: vec![AchievementRequirement {
                metric: "speed_ratio".to_string(),
                threshold: 500,
                comparison: "lte".to_string(),
            }],
            is_hidden: false,
            is_stackable: true,
        },
        Achievement {
            id: "perfectionist".to_string(),
            name: "Perfectionist".to_string(),
            description: "Score 100% on 5 assessments in a row".to_string(),
            category: AchievementCategory::Accuracy,
            tier: AchievementTier::Platinum,
            xp_reward: 2000,
            badge_icon: "💎".to_string(),
            requirements: vec![AchievementRequirement {
                metric: "perfect_streak".to_string(),
                threshold: 5,
                comparison: "gte".to_string(),
            }],
            is_hidden: false,
            is_stackable: true,
        },
    ];

    Ok(achievements)
}

/// Get social comparison
pub(crate) fn get_social_comparison(input: SocialComparisonInput) -> ExternResult<SocialComparison> {
    // Simulate percentile calculation (in production, compare against DHT data)
    let percentile = ((input.current_value as f64 / 100000.0) * 1000.0).min(990.0) as u16;

    let (comparison_type, message, goals) = match percentile {
        0..=200 => (
            SocialComparisonType::UpwardClose,
            "You're building momentum! Many learners started here.".to_string(),
            vec!["Complete 3 more lessons to reach the next tier".to_string()],
        ),
        201..=500 => (
            SocialComparisonType::Lateral,
            "You're in the middle of the pack - keep pushing!".to_string(),
            vec!["Increase your streak to move up".to_string()],
        ),
        501..=800 => (
            SocialComparisonType::UpwardClose,
            "Great progress! You're in the top half of learners.".to_string(),
            vec!["Master 2 more skills to reach the top quartile".to_string()],
        ),
        801..=950 => (
            SocialComparisonType::SelfReferenced,
            "Excellent! You're outperforming most learners.".to_string(),
            vec!["Focus on depth - achieve 95%+ in your top skills".to_string()],
        ),
        _ => (
            SocialComparisonType::SelfReferenced,
            "Outstanding! You're among the top performers.".to_string(),
            vec!["Consider mentoring others".to_string(), "Explore new skill areas".to_string()],
        ),
    };

    Ok(SocialComparison {
        comparison_type,
        reference_group: format!("learners on {}", input.metric),
        your_percentile: percentile,
        relative_performance_permille: percentile,
        motivational_message: message,
        suggested_goals: goals,
    })
}

/// Get active challenges
pub(crate) fn get_active_challenges(_input: ActiveChallengesInput) -> ExternResult<Vec<ActiveChallenge>> {
    // Return sample challenges
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    let challenges = vec![
        ActiveChallenge {
            challenge: Challenge {
                id: "daily_1".to_string(),
                challenge_type: ChallengeType::Daily,
                title: "Daily Dedication".to_string(),
                description: "Complete 3 lessons today".to_string(),
                objectives: vec![ChallengeObjective {
                    description: "Complete lessons".to_string(),
                    target_value: 3,
                    current_value: 1,
                    is_completed: false,
                }],
                start_timestamp: now - 3600,
                end_timestamp: now + 72000,
                xp_reward: 200,
                bonus_reward_permille: 500, // 50% bonus for early completion
                participants: 1234,
                completers: 456,
            },
            progress_permille: 333, // 1/3 complete
            time_remaining_seconds: 72000,
            is_completed: false,
            bonus_eligible: true,
        },
        ActiveChallenge {
            challenge: Challenge {
                id: "weekly_mastery".to_string(),
                challenge_type: ChallengeType::Weekly,
                title: "Weekly Mastery Push".to_string(),
                description: "Improve mastery in 5 skills this week".to_string(),
                objectives: vec![ChallengeObjective {
                    description: "Improve skill mastery".to_string(),
                    target_value: 5,
                    current_value: 2,
                    is_completed: false,
                }],
                start_timestamp: now - 86400 * 3,
                end_timestamp: now + 86400 * 4,
                xp_reward: 1000,
                bonus_reward_permille: 250,
                participants: 5678,
                completers: 1234,
            },
            progress_permille: 400, // 2/5 complete
            time_remaining_seconds: 86400 * 4,
            is_completed: false,
            bonus_eligible: false,
        },
    ];

    Ok(challenges)
}
