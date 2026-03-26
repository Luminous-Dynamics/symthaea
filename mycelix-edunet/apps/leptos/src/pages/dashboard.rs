// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Unified learner dashboard — the most valuable page in EduNet.
//!
//! Shows XP/level, streak, due reviews, skill mastery, recommendations,
//! and recent activity. Each section is a reusable component that fetches
//! data from the conductor (falling back to mocks).

use leptos::prelude::*;

use crate::consciousness::ConsciousnessCard;
use crate::holochain::use_holochain;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LearnerStats {
    pub xp_total: u64,
    pub xp_today: u64,
    pub xp_this_week: u64,
    pub level: u32,
    pub xp_to_next_level: u64,
    pub xp_in_current_level: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StreakInfo {
    pub current_days: u32,
    pub freeze_count: u32,
    pub bonus_multiplier: f32,
    pub longest_streak: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DueReviews {
    pub total_due: u32,
    pub overdue: u32,
    pub new_available: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkillMastery {
    pub name: String,
    pub level: f32,
    pub domain: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Recommendation {
    pub title: String,
    pub reason: String,
    pub course_domain: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ActivityEvent {
    pub description: String,
    pub timestamp: String,
    pub kind: ActivityKind,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ActivityKind {
    CourseProgress,
    ReviewCompleted,
    BadgeEarned,
    LevelUp,
    StreakMilestone,
}

// ---------------------------------------------------------------------------
// Mock data generators
// ---------------------------------------------------------------------------

fn mock_stats() -> LearnerStats {
    LearnerStats {
        xp_total: 4_280,
        xp_today: 120,
        xp_this_week: 680,
        level: 7,
        xp_to_next_level: 500,
        xp_in_current_level: 280,
    }
}

fn mock_streak() -> StreakInfo {
    StreakInfo {
        current_days: 12,
        freeze_count: 1,
        bonus_multiplier: 1.5,
        longest_streak: 21,
    }
}

fn mock_due_reviews() -> DueReviews {
    DueReviews {
        total_due: 18,
        overdue: 3,
        new_available: 5,
    }
}

fn mock_skills() -> Vec<SkillMastery> {
    vec![
        SkillMastery { name: "Rust Ownership".into(), level: 0.85, domain: "Programming".into() },
        SkillMastery { name: "Consensus Algorithms".into(), level: 0.72, domain: "Distributed Systems".into() },
        SkillMastery { name: "Soil Chemistry".into(), level: 0.63, domain: "Agriculture".into() },
        SkillMastery { name: "DHT Fundamentals".into(), level: 0.58, domain: "Networking".into() },
        SkillMastery { name: "Cooperative Governance".into(), level: 0.45, domain: "Economics".into() },
    ]
}

fn mock_recommendations() -> Vec<Recommendation> {
    vec![
        Recommendation {
            title: "Async Rust Patterns".into(),
            reason: "Your Rust Ownership mastery is high — ready for async".into(),
            course_domain: "Programming".into(),
        },
        Recommendation {
            title: "Soil Microbiome Lab".into(),
            reason: "Complements your Soil Chemistry progress".into(),
            course_domain: "Agriculture".into(),
        },
        Recommendation {
            title: "Byzantine Fault Tolerance".into(),
            reason: "Builds on Consensus Algorithms knowledge".into(),
            course_domain: "Distributed Systems".into(),
        },
    ]
}

fn mock_activity() -> Vec<ActivityEvent> {
    vec![
        ActivityEvent {
            description: "Completed 'Ownership & Borrowing' module".into(),
            timestamp: "2 hours ago".into(),
            kind: ActivityKind::CourseProgress,
        },
        ActivityEvent {
            description: "Reviewed 12 SRS cards (92% correct)".into(),
            timestamp: "5 hours ago".into(),
            kind: ActivityKind::ReviewCompleted,
        },
        ActivityEvent {
            description: "Earned 'Soil Scholar' badge".into(),
            timestamp: "Yesterday".into(),
            kind: ActivityKind::BadgeEarned,
        },
        ActivityEvent {
            description: "Reached Level 7".into(),
            timestamp: "2 days ago".into(),
            kind: ActivityKind::LevelUp,
        },
        ActivityEvent {
            description: "10-day learning streak!".into(),
            timestamp: "4 days ago".into(),
            kind: ActivityKind::StreakMilestone,
        },
    ]
}

// ---------------------------------------------------------------------------
// Dashboard page (layout)
// ---------------------------------------------------------------------------

#[component]
pub fn DashboardPage() -> impl IntoView {
    view! {
        <div class="dashboard">
            <h2>"Learner Dashboard"</h2>
            <div class="dashboard-grid">
                <ConsciousnessCard />
                <XpLevelCard />
                <StreakCard />
                <DueReviewsCard />
                <SkillsCard />
            </div>
            <RecommendationsSection />
            <RecentActivitySection />
        </div>
    }
}

// ---------------------------------------------------------------------------
// XP & Level card
// ---------------------------------------------------------------------------

#[component]
fn XpLevelCard() -> impl IntoView {
    let hc = use_holochain();

    let stats = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc.call_zome::<(), LearnerStats>("gamification", "get_learner_stats", &()).await {
                Ok(s) => s,
                Err(_) => mock_stats(),
            }
        }
    });

    view! {
        <div class="dash-card xp-card">
            <h3>"XP & Level"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    stats.get().map(|s| {
                        let s: LearnerStats = (*s).clone();
                        let progress_pct = if s.xp_to_next_level > 0 {
                            (s.xp_in_current_level as f64 / s.xp_to_next_level as f64 * 100.0).min(100.0)
                        } else {
                            100.0
                        };
                        view! {
                            <div class="stat-big">
                                <span class="level-badge">"Lv. " {s.level}</span>
                                <span class="xp-total">{format!("{} XP", s.xp_total)}</span>
                            </div>
                            <div class="progress-bar-container">
                                <div class="progress-bar"
                                    style=format!("width: {}%", progress_pct)>
                                </div>
                            </div>
                            <div class="xp-details">
                                <span>"Today: " <strong>{format!("+{}", s.xp_today)}</strong></span>
                                <span>"This week: " <strong>{format!("+{}", s.xp_this_week)}</strong></span>
                            </div>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Streak card
// ---------------------------------------------------------------------------

#[component]
fn StreakCard() -> impl IntoView {
    let hc = use_holochain();

    let streak = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc.call_zome::<(), StreakInfo>("gamification", "get_streak", &()).await {
                Ok(s) => s,
                Err(_) => mock_streak(),
            }
        }
    });

    view! {
        <div class="dash-card streak-card">
            <h3>"Streak"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    streak.get().map(|s| {
                        let s: StreakInfo = (*s).clone();
                        view! {
                            <div class="stat-big">
                                <span class="streak-count">{s.current_days} " days"</span>
                            </div>
                            <div class="streak-details">
                                <div class="streak-row">
                                    <span class="label">"Bonus"</span>
                                    <span class="value">{format!("{:.1}x", s.bonus_multiplier)}</span>
                                </div>
                                <div class="streak-row">
                                    <span class="label">"Freezes left"</span>
                                    <span class="value">{s.freeze_count}</span>
                                </div>
                                <div class="streak-row">
                                    <span class="label">"Best"</span>
                                    <span class="value">{s.longest_streak} " days"</span>
                                </div>
                            </div>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Due reviews card
// ---------------------------------------------------------------------------

#[component]
fn DueReviewsCard() -> impl IntoView {
    let hc = use_holochain();

    let reviews = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc.call_zome::<(), DueReviews>("srs", "get_due_summary", &()).await {
                Ok(r) => r,
                Err(_) => mock_due_reviews(),
            }
        }
    });

    view! {
        <div class="dash-card reviews-card">
            <h3>"Due Reviews"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    reviews.get().map(|r| {
                        let r: DueReviews = (*r).clone();
                        view! {
                            <div class="stat-big">
                                <span class="due-count">{r.total_due}</span>
                                <span class="due-label">" cards due"</span>
                            </div>
                            <div class="review-breakdown">
                                <span class="overdue">{r.overdue} " overdue"</span>
                                <span class="new-cards">{r.new_available} " new"</span>
                            </div>
                            <a href="/review" class="btn-primary">"Start Review"</a>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Skills card
// ---------------------------------------------------------------------------

#[component]
fn SkillsCard() -> impl IntoView {
    let hc = use_holochain();

    let skills = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc.call_zome::<(), Vec<SkillMastery>>("adaptive", "get_top_skills", &()).await {
                Ok(s) => s,
                Err(_) => mock_skills(),
            }
        }
    });

    view! {
        <div class="dash-card skills-card">
            <h3>"Top Skills"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    skills.get().map(|data| {
                        let data: Vec<SkillMastery> = (*data).clone();
                        view! {
                            <div class="skills-list">
                                {data.into_iter().map(|skill| {
                                    let pct = (skill.level * 100.0) as u32;
                                    view! {
                                        <div class="skill-row">
                                            <div class="skill-info">
                                                <span class="skill-name">{skill.name}</span>
                                                <span class="skill-domain">{skill.domain}</span>
                                            </div>
                                            <div class="skill-bar-container">
                                                <div class="skill-bar"
                                                    style=format!("width: {}%", pct)>
                                                </div>
                                            </div>
                                            <span class="skill-pct">{pct} "%"</span>
                                        </div>
                                    }
                                }).collect_view()}
                            </div>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Recommendations section
// ---------------------------------------------------------------------------

#[component]
fn RecommendationsSection() -> impl IntoView {
    let hc = use_holochain();

    let recs = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc
                .call_zome::<(), Vec<Recommendation>>(
                    "adaptive",
                    "get_recommendations",
                    &(),
                )
                .await
            {
                Ok(r) => r,
                Err(_) => mock_recommendations(),
            }
        }
    });

    view! {
        <div class="dash-section recommendations">
            <h3>"Recommendations"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    recs.get().map(|data| {
                        let data: Vec<Recommendation> = (*data).clone();
                        view! {
                            <div class="rec-grid">
                                {data.into_iter().map(|rec| {
                                    view! {
                                        <div class="rec-card">
                                            <h4>{rec.title}</h4>
                                            <p>{rec.reason}</p>
                                            <span class="domain-tag">{rec.course_domain}</span>
                                        </div>
                                    }
                                }).collect_view()}
                            </div>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Recent activity section
// ---------------------------------------------------------------------------

#[component]
fn RecentActivitySection() -> impl IntoView {
    let hc = use_holochain();

    let activity = LocalResource::new(move || {
        let hc = hc.clone();
        async move {
            match hc
                .call_zome::<(), Vec<ActivityEvent>>(
                    "gamification",
                    "get_recent_activity",
                    &(),
                )
                .await
            {
                Ok(a) => a,
                Err(_) => mock_activity(),
            }
        }
    });

    view! {
        <div class="dash-section activity">
            <h3>"Recent Activity"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    activity.get().map(|data| {
                        let data: Vec<ActivityEvent> = (*data).clone();
                        view! {
                            <ul class="activity-feed">
                                {data.into_iter().map(|event| {
                                    let icon = match event.kind {
                                        ActivityKind::CourseProgress => "^",
                                        ActivityKind::ReviewCompleted => "*",
                                        ActivityKind::BadgeEarned => "#",
                                        ActivityKind::LevelUp => "+",
                                        ActivityKind::StreakMilestone => "~",
                                    };
                                    view! {
                                        <li class="activity-item">
                                            <span class="activity-icon">{icon}</span>
                                            <div class="activity-content">
                                                <span class="activity-desc">{event.description}</span>
                                                <span class="activity-time">{event.timestamp}</span>
                                            </div>
                                        </li>
                                    }
                                }).collect_view()}
                            </ul>
                        }
                    })
                }}
            </Suspense>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Shared loading skeleton
// ---------------------------------------------------------------------------

#[component]
fn CardLoading() -> impl IntoView {
    view! {
        <div class="card-loading">
            <div class="skeleton-line wide"></div>
            <div class="skeleton-line medium"></div>
            <div class="skeleton-line narrow"></div>
        </div>
    }
}
