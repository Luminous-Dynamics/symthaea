// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Unified learner dashboard -- the most valuable page in Praxis.
//!
//! Shows XP/level, streak, due reviews, skill mastery, recommendations,
//! and recent activity. Integrated with the adaptivity engine to show
//! sovereignty level, cognitive state, and active adaptations.

use leptos::prelude::*;
use mycelix_leptos_core::{SovereignRadar, SovereignRadarSize};

use crate::adaptivity_provider::use_adaptivity;
use crate::cognitive_adaptivity::*;
use crate::components::suggestion_overlay::{SuggestionOverlay, CognitiveStateMirror};
use crate::curriculum::{curriculum_graph, use_progress, ProgressStatus};
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
// Real data generators — computed from localStorage, no mocks
// ---------------------------------------------------------------------------

fn real_stats(progress: &crate::curriculum::ProgressStore, tracker: &crate::study_tracker::StudyTracker) -> LearnerStats {
    // XP: 10 per problem answered + 50 per mastered topic + 25 per pomodoro
    let problems_xp = progress.bkt_states.values().map(|b| b.attempts as u64 * 10).sum::<u64>();
    let mastery_xp = progress.mastered_count() as u64 * 50;
    let pomodoro_xp = (tracker.total_minutes as u64 / 25) * 25;
    let total = problems_xp + mastery_xp + pomodoro_xp;

    // Level = sqrt(total_xp / 100), XP to next = 100 * (level+1)^2
    let level = ((total as f64 / 100.0).sqrt()) as u32;
    let xp_for_current = 100 * level as u64 * level as u64;
    let xp_for_next = 100 * (level as u64 + 1) * (level as u64 + 1);

    LearnerStats {
        xp_total: total,
        xp_today: pomodoro_xp.min(200), // approximate
        xp_this_week: total.min(2000) / 4, // approximate
        level,
        xp_to_next_level: (xp_for_next - xp_for_current) as u64,
        xp_in_current_level: (total - xp_for_current) as u64,
    }
}

fn real_streak(tracker: &crate::study_tracker::StudyTracker) -> StreakInfo {
    let current = tracker.current_streak();
    let longest = tracker.longest_streak();
    let bonus = match current {
        0..=6 => 1.0,
        7..=13 => 1.1,
        14..=29 => 1.25,
        30..=99 => 1.5,
        _ => 2.0,
    };
    StreakInfo {
        current_days: current,
        freeze_count: 0,
        bonus_multiplier: bonus,
        longest_streak: longest,
    }
}

fn real_due_reviews(progress: &crate::curriculum::ProgressStore) -> DueReviews {
    let now = js_sys::Date::now();
    let due: Vec<_> = progress.srs_cards.values().filter(|c| now >= c.next_review_ms).collect();
    let overdue = due.iter().filter(|c| now - c.next_review_ms > 24.0 * 60.0 * 60.0 * 1000.0).count() as u32;
    DueReviews {
        total_due: due.len() as u32,
        overdue,
        new_available: progress.srs_cards.values().filter(|c| c.repetitions == 0).count() as u32,
    }
}

fn real_skills(progress: &crate::curriculum::ProgressStore) -> Vec<SkillMastery> {
    let graph = curriculum_graph();
    let mut subjects: std::collections::HashMap<String, (f32, usize)> = std::collections::HashMap::new();
    for n in &graph.nodes {
        let bkt = progress.bkt(&n.id);
        if bkt.attempts > 0 {
            let entry = subjects.entry(n.subject_area.clone()).or_insert((0.0, 0));
            entry.0 += bkt.p_mastery;
            entry.1 += 1;
        }
    }
    let mut skills: Vec<SkillMastery> = subjects.into_iter()
        .filter(|(_, (_, count))| *count > 0)
        .map(|(name, (total_mastery, count))| SkillMastery {
            name: name.clone(),
            level: total_mastery / count as f32,
            domain: name,
        })
        .collect();
    skills.sort_by(|a, b| b.level.partial_cmp(&a.level).unwrap_or(std::cmp::Ordering::Equal));
    skills.truncate(5);
    if skills.is_empty() {
        // Show top subjects by node count for new users
        let mut top: Vec<_> = graph.subjects().iter().take(5).map(|s| SkillMastery {
            name: s.to_string(), level: 0.0, domain: s.to_string()
        }).collect();
        top
    } else {
        skills
    }
}

fn real_recommendations(progress: &crate::curriculum::ProgressStore) -> Vec<Recommendation> {
    let weakest = progress.weakest_topics(3);
    let graph = curriculum_graph();
    if weakest.is_empty() {
        // New user — suggest starting points
        vec![
            Recommendation { title: "Start with your grade's top topic".into(), reason: "Begin your learning journey on the constellation".into(), course_domain: "Getting Started".into() },
            Recommendation { title: "Try a flashcard session".into(), reason: "5 minutes of spaced repetition goes a long way".into(), course_domain: "Review".into() },
        ]
    } else {
        weakest.iter().map(|(id, pct)| {
            let title = graph.node(id).map(|n| n.title.clone()).unwrap_or_default();
            Recommendation {
                title,
                reason: format!("{}% mastery — practice to strengthen", (pct * 100.0) as u32),
                course_domain: graph.node(id).map(|n| n.subject_area.clone()).unwrap_or_default(),
            }
        }).collect()
    }
}

fn real_activity(tracker: &crate::study_tracker::StudyTracker, progress: &crate::curriculum::ProgressStore) -> Vec<ActivityEvent> {
    let mut events = Vec::new();
    let streak = tracker.current_streak();
    let mastered = progress.mastered_count();
    let hours = tracker.hours_studied();

    if hours > 0.0 {
        events.push(ActivityEvent {
            description: format!("{:.0} hours of focused study", hours),
            timestamp: "total".into(),
            kind: ActivityKind::CourseProgress,
        });
    }
    if mastered > 0 {
        events.push(ActivityEvent {
            description: format!("{} topics mastered", mastered),
            timestamp: "cumulative".into(),
            kind: ActivityKind::CourseProgress,
        });
    }
    if streak >= 3 {
        events.push(ActivityEvent {
            description: format!("{}-day study streak!", streak),
            timestamp: "current".into(),
            kind: ActivityKind::StreakMilestone,
        });
    }
    let problems_done: u32 = progress.bkt_states.values().map(|b| b.attempts).sum();
    if problems_done > 0 {
        events.push(ActivityEvent {
            description: format!("{} practice problems completed", problems_done),
            timestamp: "total".into(),
            kind: ActivityKind::CourseProgress,
        });
    }
    if events.is_empty() {
        events.push(ActivityEvent {
            description: "Start studying to see your activity here".into(),
            timestamp: "now".into(),
            kind: ActivityKind::CourseProgress,
        });
    }
    events
}

// ---------------------------------------------------------------------------
// Dashboard page (layout)
// ---------------------------------------------------------------------------

#[component]
pub fn DashboardPage() -> impl IntoView {
    let progress = use_progress();

    view! {
        <div class="dashboard">
            <h2>"Dashboard"</h2>

            // Exam countdown + streak (real data)
            <crate::study_tracker::ExamCountdown />

            // CAPS Progress Overview
            <ProgressCard />

            // Pending TEND — economic value from learning
            <PendingTendCard />

            // Solo cooperation prompt
            <div class="cooperation-prompt">
                "Learning solo? Connect to the mesh to find peers studying the same topics."
                <br />
                <strong>"Cooperation multiplier: 1.2x TEND for group study sessions."</strong>
            </div>

            // Suggestion overlay
            <SuggestionOverlay />

            // Mirror mode cognitive state display
            <CognitiveStateMirror />

            // Start Session — links to highest priority topic
            {move || {
                let p = progress.get();
                let graph = curriculum_graph();

                // Find best topic: highest exam weight, not mastered, prerequisites met
                let mut best: Option<(String, String, u16)> = None;
                for n in &graph.nodes {
                    if p.get(&n.id).status == ProgressStatus::Mastered { continue; }
                    let weight = n.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
                    let prereqs_met = graph.prereqs_for(&n.id).iter().all(|pid| {
                        p.get(pid).status == ProgressStatus::Mastered
                    });
                    if prereqs_met {
                        if best.is_none() || weight > best.as_ref().unwrap().2 {
                            best = Some((n.id.clone(), n.title.clone(), weight));
                        }
                    }
                }

                if let Some((id, title, marks)) = best {
                    let href = format!("/study/{}", id);
                    view! {
                        <a href=href style="display: block; padding: 1.25rem; background: linear-gradient(135deg, var(--primary), var(--info)); border-radius: 12px; text-decoration: none; color: var(--text-on-primary); margin: 1rem 0; transition: transform 0.15s">
                            <div style="display: flex; justify-content: space-between; align-items: center">
                                <div>
                                    <div style="font-size: 0.8rem; opacity: 0.8">"Recommended next"</div>
                                    <div style="font-size: 1.1rem; font-weight: 700; margin-top: 0.25rem">{title}</div>
                                </div>
                                <div style="text-align: right">
                                    {if marks > 0 {
                                        view! { <div style="font-size: 0.9rem; font-weight: 600">{marks}"m"</div> }.into_any()
                                    } else {
                                        view! { <span></span> }.into_any()
                                    }}
                                    <div style="font-size: 1.2rem; margin-top: 0.25rem">"Start \u{2192}"</div>
                                </div>
                            </div>
                        </a>
                    }.into_any()
                } else {
                    view! {
                        <div style="padding: 1.25rem; background: var(--surface); border-radius: 12px; text-align: center; margin: 1rem 0; color: var(--success); font-weight: 600">
                            "All available topics mastered!"
                        </div>
                    }.into_any()
                }
            }}

            // Quick actions
            <div style="display: flex; gap: 0.75rem; margin: 0.5rem 0 1rem; flex-wrap: wrap">
                <a href="/skill-map" class="feature-card" style="flex: 1; min-width: 140px; text-align: center; padding: 0.75rem">
                    <div style="font-weight: 600; font-size: 0.85rem">"Constellation"</div>
                </a>
                <a href="/review" class="feature-card" style="flex: 1; min-width: 140px; text-align: center; padding: 0.75rem">
                    <div style="font-weight: 600; font-size: 0.85rem">"Review Cards"</div>
                </a>
                <a href="/exam-prep" class="feature-card" style="flex: 1; min-width: 140px; text-align: center; padding: 0.75rem">
                    <div style="font-weight: 600; font-size: 0.85rem">"Exam Prep"</div>
                </a>
            </div>

            // Learning velocity + due cards
            <div class="dashboard-grid">
                <crate::study_tracker::LearningVelocity />
                <crate::study_tracker::DueCardsQueue />
            </div>

            // Achievement milestones
            <crate::achievements::AchievementBadges />

            // Share + Export/Import progress
            <ShareProgress />
            <ProgressPortability />

            // Mastery heat map
            <MasteryHeatMap />

            // Exam score prediction
            <crate::study_tracker::ExamPrediction />

            // Subject mastery breakdown
            <SubjectMasteryBreakdown />

            <div class="dashboard-grid">
                <SovereigntyCard />
                <LearningReadinessCard />
                <div class="dash-card">
                    <h3>"Civic Profile"</h3>
                    <SovereignRadar size=SovereignRadarSize::Small />
                </div>
            </div>
            <CurriculumRecommendationsSection />
        </div>
    }
}

// ---------------------------------------------------------------------------
// Sovereignty card (NEW -- shows independence level and growth)
// ---------------------------------------------------------------------------

#[component]
fn SovereigntyCard() -> impl IntoView {
    let adaptivity = use_adaptivity();

    view! {
        <div class="dash-card sovereignty-card">
            <h3>"Your Learning Independence"</h3>
            {move || {
                let sov = adaptivity.sovereignty.get();
                let mode = sov.mode();
                let (mode_label, mode_icon, mode_description) = match mode {
                    InteractionMode::Guardian => (
                        "Helper Mode",
                        "\u{1f6e1}\u{fe0f}",
                        "I help guide your learning and explain what I'm doing",
                    ),
                    InteractionMode::Guide => (
                        "Guide Mode",
                        "\u{1f331}",
                        "I suggest options and you choose what works for you",
                    ),
                    InteractionMode::Mirror => (
                        "Mirror Mode",
                        "\u{1fa9e}",
                        "I show you how you're doing -- you decide what to do",
                    ),
                    InteractionMode::Autonomous => (
                        "Independent Mode",
                        "\u{2b50}",
                        "You're in charge! I'm here if you need me",
                    ),
                };

                let progress_pct = ((sov.level as f64 / 1000.0) * 100.0) as u32;

                // Last few growth events (most recent first)
                let recent_events: Vec<_> = sov.growth_events.iter().rev().take(3).cloned().collect();

                view! {
                    <div class="sovereignty-display">
                        <div class="sovereignty-mode">
                            <span class="sovereignty-icon">{mode_icon}</span>
                            <span class="sovereignty-mode-label">{mode_label}</span>
                        </div>
                        <p class="sovereignty-description">{mode_description}</p>

                        <div class="sovereignty-progress">
                            <div class="sovereignty-bar-container">
                                <div class="sovereignty-bar"
                                    style=format!("width: {}%", progress_pct)>
                                </div>
                            </div>
                            <span class="sovereignty-level-text">
                                {format!("{}/1000", sov.level)}
                            </span>
                        </div>
                    </div>

                    // Recent sovereignty growth events
                    {if !recent_events.is_empty() {
                        view! {
                            <div class="sovereignty-history">
                                <p class="sovereignty-history-label">"How you earned independence:"</p>
                                <ul class="sovereignty-events">
                                    {recent_events.into_iter().map(|event| {
                                        let icon = match event.event_type {
                                            SovereigntyGrowthType::SelfRegulatedBreak => "\u{1f9d8}",
                                            SovereigntyGrowthType::PerseveranceSuccess => "\u{1f4aa}",
                                            SovereigntyGrowthType::AskedForHelp => "\u{1f91d}",
                                            SovereigntyGrowthType::PlannedAndExecuted => "\u{1f4cb}",
                                            SovereigntyGrowthType::ReflectiveResponse => "\u{1f4ad}",
                                            SovereigntyGrowthType::DifficultyCalibration => "\u{1f3af}",
                                            SovereigntyGrowthType::PeerTeaching => "\u{1f9d1}\u{200d}\u{1f3eb}",
                                            SovereigntyGrowthType::IndependentSuccess => "\u{1f680}",
                                            SovereigntyGrowthType::FulfilledPledge => "\u{1f91d}",
                                            SovereigntyGrowthType::CommunityContribution => "\u{1f331}",
                                        };
                                        view! {
                                            <li class="sovereignty-event">
                                                <span class="event-icon">{icon}</span>
                                                <span class="event-desc">{event.description.clone()}</span>
                                                <span class="event-delta">{format!("+{}", event.delta)}</span>
                                            </li>
                                        }
                                    }).collect_view()}
                                </ul>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <p class="sovereignty-hint">
                                <small>"Keep learning and you'll earn more independence!"</small>
                            </p>
                        }.into_any()
                    }}
                }
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Active Adaptation section (NEW -- shows what the system is doing)
// ---------------------------------------------------------------------------

#[component]
fn ActiveAdaptationSection() -> impl IntoView {
    let adaptivity = use_adaptivity();

    view! {
        <div class="dash-section active-adaptation">
            {move || {
                let adapt = adaptivity.adaptation.get();
                let sov = adaptivity.sovereignty.get();
                let mode = sov.mode();

                // Only show in Guardian or Guide mode
                if matches!(mode, InteractionMode::Mirror | InteractionMode::Autonomous) {
                    return view! { <div class="adaptation-hidden"></div> }.into_any();
                }

                let complexity_text = match &adapt.text_complexity {
                    TextComplexity::Standard => "Standard difficulty",
                    TextComplexity::Simplified => "Simplified wording",
                    TextComplexity::Minimal => "Numbers only",
                    TextComplexity::Personalized { interest_topic } => {
                        // Can't return a &str from a match arm with borrowed data easily,
                        // so we'll handle this in the view
                        return view! {
                            <div class="adaptation-status">
                                <h3>"How I'm Helping"</h3>
                                <div class="adaptation-items">
                                    <div class="adaptation-item">
                                        <span class="adaptation-label">"Content"</span>
                                        <span class="adaptation-value">
                                            {format!("Personalized with {}", interest_topic)}
                                        </span>
                                    </div>
                                </div>
                            </div>
                        }.into_any();
                    }
                };

                let diff_text = if adapt.difficulty_delta > 0.1 {
                    "Making problems a bit harder"
                } else if adapt.difficulty_delta < -0.1 {
                    "Making problems a bit easier"
                } else {
                    "Just right"
                };

                let accuracy = adaptivity.recent_accuracy.get();
                let accuracy_pct = (accuracy * 100.0) as u32;

                view! {
                    <div class="adaptation-status">
                        <h3>"How I'm Helping"</h3>
                        <div class="adaptation-items">
                            <div class="adaptation-item">
                                <span class="adaptation-label">"Content"</span>
                                <span class="adaptation-value">{complexity_text}</span>
                            </div>
                            <div class="adaptation-item">
                                <span class="adaptation-label">"Difficulty"</span>
                                <span class="adaptation-value">{diff_text}</span>
                            </div>
                            <div class="adaptation-item">
                                <span class="adaptation-label">"Your accuracy"</span>
                                <span class="adaptation-value">{format!("{}%", accuracy_pct)}</span>
                            </div>
                        </div>
                    </div>
                }.into_any()
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Learning Readiness card (kid-friendly consciousness wrapper)
// ---------------------------------------------------------------------------

#[component]
fn LearningReadinessCard() -> impl IntoView {
    let ctx = crate::consciousness::use_consciousness();
    let (show_details, set_show_details) = signal(false);

    view! {
        <div class="dash-card readiness-card">
            <h3>"Learning Readiness"</h3>
            <div class="readiness-display">
                {move || {
                    let s = ctx.state.get();
                    // Map consciousness state to kid-friendly readiness
                    let (indicator, status_text, hint, css_class) = if s.phi > 0.5 && s.neuromod_norepinephrine < 0.7 {
                        ("\u{1f7e2}", "Ready to learn!", "Your brain is warmed up and ready for new challenges", "readiness-green")
                    } else if s.phi > 0.3 {
                        ("\u{1f7e1}", "Warming up...", "Almost there! Try a quick review to get going", "readiness-yellow")
                    } else if s.neuromod_norepinephrine > 0.7 || s.neuromod_dopamine < 0.3 {
                        ("\u{1f7e0}", "Time for a break", "Maybe stretch, get some water, or take a walk", "readiness-orange")
                    } else {
                        ("\u{1f534}", "Let's rest", "It's okay to take a break and come back later", "readiness-red")
                    };

                    // Focus bar: map phi (0-1) to a simple percentage
                    let focus_pct = ((s.phi * 100.0) as u32).min(100);
                    let focus_label = if focus_pct > 70 { "Great" }
                        else if focus_pct > 50 { "Good" }
                        else if focus_pct > 30 { "Okay" }
                        else { "Low" };

                    view! {
                        <div class=format!("readiness-status {}", css_class)>
                            <span class="readiness-indicator">{indicator}</span>
                            <span class="readiness-text">{status_text}</span>
                        </div>
                        <div class="readiness-focus">
                            <div class="readiness-focus-bar-container">
                                <div class="readiness-focus-bar"
                                    style=format!("width: {}%", focus_pct)>
                                </div>
                            </div>
                            <span class="readiness-focus-label">"Focus: " {focus_label}</span>
                        </div>
                        <p class="readiness-hint">{hint}</p>
                    }
                }}
            </div>

            // Toggle for teacher/advanced view
            <button
                class="details-toggle"
                on:click=move |_| set_show_details.update(|v| *v = !*v)
            >
                {move || if show_details.get() { "Hide details" } else { "Show details" }}
            </button>

            // Advanced panel (hidden by default)
            {move || {
                if show_details.get() {
                    let s = ctx.state.get();
                    view! {
                        <div class="readiness-details">
                            <div class="detail-metric">
                                <span class="detail-metric-label">"Phi"</span>
                                <span class="detail-metric-value">{format!("{:.3}", s.phi)}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"Coherence"</span>
                                <span class="detail-metric-value">{format!("{:.3}", s.coherence)}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"Free Energy"</span>
                                <span class="detail-metric-value">{format!("{:.3}", s.free_energy)}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"DA"</span>
                                <span class="detail-metric-value">{format!("{:.2}", s.neuromod_dopamine)}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"NE"</span>
                                <span class="detail-metric-value">{format!("{:.2}", s.neuromod_norepinephrine)}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"5-HT"</span>
                                <span class="detail-metric-value">{format!("{:.2}", s.neuromod_serotonin)}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"Workspace"</span>
                                <span class="detail-metric-value">
                                    {if s.workspace_ignited { "Ignited" } else { "Sub-threshold" }}
                                </span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"Harmony"</span>
                                <span class="detail-metric-value">{s.dominant_harmony.clone()}</span>
                            </div>
                            <div class="detail-metric">
                                <span class="detail-metric-label">"Cycle"</span>
                                <span class="detail-metric-value">{s.cycle_count}</span>
                            </div>
                            <div class="consciousness-disclaimer">
                                <small>"Simulated \u{2014} real Spore WASM integration pending"</small>
                            </div>
                        </div>
                    }.into_any()
                } else {
                    view! { <div></div> }.into_any()
                }
            }}
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
            match hc.call_zome_default::<(), LearnerStats>("gamification", "get_learner_stats", &()).await {
                Ok(s) => s,
                Err(_) => { let p = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress").unwrap_or_default(); let t = crate::persistence::load::<crate::study_tracker::StudyTracker>("praxis_study_tracker").unwrap_or_default(); real_stats(&p, &t) },
            }
        }
    });

    view! {
        <div class="dash-card xp-card">
            <h3>"XP & Level"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    stats.get().map(|s| {
                        let s: LearnerStats = s.clone();
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
            match hc.call_zome_default::<(), StreakInfo>("gamification", "get_streak", &()).await {
                Ok(s) => s,
                Err(_) => { let t = crate::persistence::load::<crate::study_tracker::StudyTracker>("praxis_study_tracker").unwrap_or_default(); real_streak(&t) },
            }
        }
    });

    view! {
        <div class="dash-card streak-card">
            <h3>"Streak"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    streak.get().map(|s| {
                        let s: StreakInfo = s.clone();
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
            match hc.call_zome_default::<(), DueReviews>("srs", "get_due_summary", &()).await {
                Ok(r) => r,
                Err(_) => { let p = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress").unwrap_or_default(); real_due_reviews(&p) },
            }
        }
    });

    view! {
        <div class="dash-card reviews-card">
            <h3>"Due Reviews"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    reviews.get().map(|r| {
                        let r: DueReviews = r.clone();
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
            match hc.call_zome_default::<(), Vec<SkillMastery>>("adaptive", "get_top_skills", &()).await {
                Ok(s) => s,
                Err(_) => { let p = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress").unwrap_or_default(); real_skills(&p) },
            }
        }
    });

    view! {
        <div class="dash-card skills-card">
            <h3>"What I'm Learning"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    skills.get().map(|data| {
                        let data: Vec<SkillMastery> = data.clone();
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
                .call_zome_default::<(), Vec<Recommendation>>(
                    "adaptive",
                    "get_recommendations",
                    &(),
                )
                .await
            {
                Ok(r) => r,
                Err(_) => { let p = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress").unwrap_or_default(); real_recommendations(&p) },
            }
        }
    });

    view! {
        <div class="dash-section recommendations">
            <h3>"What's Next"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    recs.get().map(|data| {
                        let data: Vec<Recommendation> = data.clone();
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
                .call_zome_default::<(), Vec<ActivityEvent>>(
                    "gamification",
                    "get_recent_activity",
                    &(),
                )
                .await
            {
                Ok(a) => a,
                Err(_) => { let t = crate::persistence::load::<crate::study_tracker::StudyTracker>("praxis_study_tracker").unwrap_or_default(); let p = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress").unwrap_or_default(); real_activity(&t, &p) },
            }
        }
    });

    view! {
        <div class="dash-section activity">
            <h3>"What I Did Today"</h3>
            <Suspense fallback=move || view! { <CardLoading /> }>
                {move || {
                    activity.get().map(|data| {
                        let data: Vec<ActivityEvent> = data.clone();
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

// ---------------------------------------------------------------------------
// CAPS Progress Card — real curriculum data
// ---------------------------------------------------------------------------

#[component]
fn ProgressCard() -> impl IntoView {
    let progress = use_progress();
    let graph = curriculum_graph();

    let math_mastery = Memo::new(move |_| {
        let p = progress.get();
        p.subject_mastery(graph, "Mathematics")
    });

    let physics_mastery = Memo::new(move |_| {
        let p = progress.get();
        p.subject_mastery(graph, "Physical Sciences")
    });

    let counts = Memo::new(move |_| {
        let p = progress.get();
        let total = graph.nodes.len();
        let mastered = p.mastered_count();
        let studying = p.studying_count();
        (total, mastered, studying)
    });

    // Find highest-weight unmastered Gr12 topics
    let priority_topics = Memo::new(move |_| {
        let p = progress.get();
        let mut topics: Vec<_> = graph.nodes.iter()
            .filter(|n| {
                n.grade_levels.first().map(|g| g == "Grade12").unwrap_or(false)
                    && p.get(&n.id).status != ProgressStatus::Mastered
                    && n.exam_weight.is_some()
            })
            .collect();
        topics.sort_by(|a, b| {
            let wa = a.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
            let wb = b.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
            wb.cmp(&wa)
        });
        topics.into_iter().take(3).map(|n| (n.id.clone(), n.title.clone(), n.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0))).collect::<Vec<_>>()
    });

    view! {
        <div class="dash-card" style="grid-column: 1 / -1">
            <h3>"Your Learning Progress"</h3>
            <div class="praxis-progress-grid">
                // Overall
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.25rem">"Overall"</div>
                    {move || {
                        let (total, mastered, studying) = counts.get();
                        let pct = if total > 0 { mastered * 100 / total } else { 0 };
                        view! {
                            <div style="font-size: 1.5rem; font-weight: 700">{mastered}"/" {total}</div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill success" style=format!("width: {}%", pct)></div>
                            </div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem">
                                {studying}" studying, "{total - mastered - studying}" remaining"
                            </div>
                        }
                    }}
                </div>
                // Mathematics
                <div>
                    <div style="font-size: 0.8rem; color: var(--subject-math); margin-bottom: 0.25rem">"Mathematics"</div>
                    {move || {
                        let m = math_mastery.get();
                        view! {
                            <div style="font-size: 1.5rem; font-weight: 700">{m / 10}"%"</div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill primary" style=format!("width: {}%", m / 10)></div>
                            </div>
                        }
                    }}
                </div>
                // Physical Sciences
                <div>
                    <div style="font-size: 0.8rem; color: var(--subject-physics); margin-bottom: 0.25rem">"Physical Sciences"</div>
                    {move || {
                        let m = physics_mastery.get();
                        view! {
                            <div style="font-size: 1.5rem; font-weight: 700">{m / 10}"%"</div>
                            <div class="progress-bar">
                                <div class="progress-bar-fill success" style=format!("width: {}%", m / 10)></div>
                            </div>
                        }
                    }}
                </div>
            </div>

            // Priority topics
            {move || {
                let topics = priority_topics.get();
                if topics.is_empty() {
                    view! { <p style="color: var(--success); font-weight: 600">"All Grade 12 topics mastered!"</p> }.into_any()
                } else {
                    view! {
                        <div style="margin-top: 1rem">
                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem">"Priority — highest exam weight, not yet mastered"</div>
                            {topics.iter().map(|(id, title, marks)| {
                                let href = format!("/study/{}", id);
                                let title = title.clone();
                                let marks = *marks;
                                view! {
                                    <a href=href style="display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid var(--border); text-decoration: none; color: var(--text)">
                                        <span style="font-size: 0.9rem">{title}</span>
                                        <span class="praxis-badge praxis-badge-exam">{marks}"m"</span>
                                    </a>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// CAPS Recommendations — real data from curriculum graph
// ---------------------------------------------------------------------------

#[component]
fn CurriculumRecommendationsSection() -> impl IntoView {
    let progress = use_progress();
    let graph = curriculum_graph();

    let recommendations = Memo::new(move |_| {
        let p = progress.get();

        // Find topics where all prerequisites are mastered but the topic itself isn't
        let mut recs = Vec::new();
        for node in &graph.nodes {
            let status = p.get(&node.id).status;
            if status == ProgressStatus::Mastered { continue; }

            let prereqs = graph.prereqs_for(&node.id);
            let all_prereqs_met = prereqs.is_empty() || prereqs.iter().all(|pid| {
                p.get(pid).status == ProgressStatus::Mastered
            });

            if all_prereqs_met {
                let weight = node.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
                recs.push((node.id.clone(), node.title.clone(), node.subdomain.clone(), weight, status));
            }
        }

        // Sort: studying first, then by exam weight
        recs.sort_by(|a, b| {
            let studying_a = a.4 == ProgressStatus::Studying;
            let studying_b = b.4 == ProgressStatus::Studying;
            studying_b.cmp(&studying_a).then(b.3.cmp(&a.3))
        });
        recs.into_iter().take(5).collect::<Vec<_>>()
    });

    view! {
        <div class="dash-section">
            <h3 class="dash-section-title">"Recommended Next Steps"</h3>
            <div class="recommendations-list">
                {move || {
                    recommendations.get().iter().map(|(id, title, subdomain, marks, status)| {
                        let href = format!("/study/{}", id);
                        let title = title.clone();
                        let subdomain = subdomain.clone();
                        let marks = *marks;
                        let label = if *status == ProgressStatus::Studying { "Continue" } else { "Start" };
                        view! {
                            <a href=href class="feature-card" style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem">
                                <div style="flex: 1">
                                    <div style="font-weight: 600; font-size: 0.9rem">{title}</div>
                                    <div style="font-size: 0.8rem; color: var(--text-secondary)">{subdomain}</div>
                                </div>
                                {if marks > 0 {
                                    view! { <span class="praxis-badge praxis-badge-exam">{marks}"m"</span> }.into_any()
                                } else {
                                    view! { <span></span> }.into_any()
                                }}
                                <span style="color: var(--primary); font-size: 0.8rem; font-weight: 600">{label}" \u{2192}"</span>
                            </a>
                        }
                    }).collect::<Vec<_>>()
                }}
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Mastery Heat Map — visual grid of Gr12 topics colored by mastery
// ---------------------------------------------------------------------------

#[component]
fn MasteryHeatMap() -> impl IntoView {
    let progress = use_progress();

    view! {
        <div class="dash-section">
            <h3>"Mastery Map"</h3>
            {move || {
                let p = progress.get();
                let graph = curriculum_graph();

                let gr12: Vec<_> = graph.nodes.iter()
                    .filter(|n| n.grade_levels.first().map(|g| g == "Grade12").unwrap_or(false))
                    .collect();

                view! {
                    <div class="mastery-heatmap">
                        {gr12.iter().map(|n| {
                            let np = p.get(&n.id);
                            let (bg, border) = match np.status {
                                ProgressStatus::Mastered => ("var(--mastery-green)", "var(--mastery-green)"),
                                ProgressStatus::Studying => ("var(--mastery-yellow)", "var(--mastery-yellow)"),
                                ProgressStatus::NotStarted => ("var(--border)", "transparent"),
                            };
                            let title = n.title.clone();
                            let href = format!("/study/{}", n.id);
                            let short: String = if title.chars().count() > 12 {
                                title.chars().take(10).collect::<String>() + ".."
                            } else {
                                title.clone()
                            };
                            view! {
                                <a href=href class="heatmap-cell" title=title
                                   style=format!("background: {}; border-color: {}", bg, border)>
                                    {short}
                                </a>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                }
            }}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Subject Mastery Breakdown — real data replacing mock
// ---------------------------------------------------------------------------

#[component]
fn ShareProgress() -> impl IntoView {
    let progress = use_progress();
    let tracker = crate::study_tracker::use_tracker();
    let profile = crate::student_profile::use_profile();
    let (copied, set_copied) = signal(false);

    let share_text = Memo::new(move |_| {
        let p = progress.get();
        let t = tracker.get();
        let prof = profile.get();
        let graph = curriculum_graph();
        let mastered = p.mastered_count();
        let total = graph.nodes.len();
        let pct = if total > 0 { mastered * 100 / total } else { 0 };
        let streak = t.current_streak();
        let hours = t.hours_studied();
        let days_left = t.days_until_exam().unwrap_or(0);

        let name = if prof.name.is_empty() { "A student".to_string() } else { prof.name.clone() };
        format!(
            "{} on Praxis:\n\u{1F331} {} topics mastered ({}% of curriculum)\n\u{1F525} {}-day study streak\n\u{23F0} {:.0} hours studied\n\u{1F3AF} {} days until exams\n\nhttps://praxis.mycelix.net",
            name, mastered, pct, streak, hours, days_left
        )
    });

    view! {
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem">
            <button
                style="flex: 1; padding: 0.5rem; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--text-secondary); font-size: 0.8rem; cursor: pointer; font-family: inherit; transition: border-color 0.2s"
                on:click=move |_| {
                    let text = share_text.get();
                    if let Some(window) = web_sys::window() {
                        let clipboard = window.navigator().clipboard();
                        let _ = clipboard.write_text(&text);
                        set_copied.set(true);
                        wasm_bindgen_futures::spawn_local(async move {
                            gloo_timers::future::sleep(std::time::Duration::from_millis(2000)).await;
                            set_copied.set(false);
                        });
                    }
                }
            >
                {move || if copied.get() { "\u{2714} Copied!" } else { "\u{1F4CB} Share My Progress" }}
            </button>
        </div>
    }
}

#[component]
fn ProgressPortability() -> impl IntoView {
    let (import_status, set_import_status) = signal::<Option<String>>(None);

    view! {
        <div style="display: flex; gap: 0.5rem; margin-bottom: 0.75rem; flex-wrap: wrap">
            <button
                style="flex: 1; min-width: 120px; padding: 0.5rem; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--text-secondary); font-size: 0.8rem; cursor: pointer; font-family: inherit"
                on:click=move |_| {
                    let json = crate::persistence::export_all();
                    crate::persistence::trigger_download("praxis-progress.json", &json);
                }
            >"\u{1F4E5} Export Progress"</button>

            <label style="flex: 1; min-width: 120px; padding: 0.5rem; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; color: var(--text-secondary); font-size: 0.8rem; cursor: pointer; font-family: inherit; text-align: center">
                "\u{1F4E4} Import Progress"
                <input
                    type="file"
                    accept=".json"
                    style="display: none"
                    on:change=move |ev| {
                        use wasm_bindgen::JsCast;
                        let Some(target) = ev.target() else { return };
                        let input: web_sys::HtmlInputElement = target.unchecked_into();
                        if let Some(files) = input.files() {
                            if let Some(file) = files.get(0) {
                                let Ok(reader) = web_sys::FileReader::new() else { return };
                                let reader_clone = reader.clone();
                                let onload = wasm_bindgen::closure::Closure::wrap(Box::new(move |_: web_sys::Event| {
                                    if let Ok(result) = reader_clone.result() {
                                        if let Some(text) = result.as_string() {
                                            match crate::persistence::import_all(&text) {
                                                Ok(summary) => {
                                                    set_import_status.set(Some(format!("Imported: {}", summary)));
                                                    if let Some(w) = web_sys::window() { let _ = w.location().reload(); }
                                                }
                                                Err(e) => set_import_status.set(Some(format!("Error: {}", e))),
                                            }
                                        }
                                    }
                                }) as Box<dyn FnMut(_)>);
                                reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                                onload.forget();
                                let _ = reader.read_as_text(&file);
                            }
                        }
                    }
                />
            </label>
        </div>
        {move || import_status.get().map(|msg| {
            let color = if msg.starts_with("Error") { "var(--error)" } else { "var(--mastery-green)" };
            view! { <div style=format!("font-size: 0.75rem; color: {}; margin-bottom: 0.5rem; text-align: center", color)>{msg}</div> }
        })}
    }
}

#[component]
fn SubjectMasteryBreakdown() -> impl IntoView {
    let progress = use_progress();

    view! {
        <div class="dash-section">
            <h3>"Subject Mastery"</h3>
            {move || {
                let p = progress.get();
                let graph = curriculum_graph();

                let mut subjects: std::collections::HashMap<String, (usize, usize)> = std::collections::HashMap::new();
                for n in &graph.nodes {
                    let entry = subjects.entry(n.subject_area.clone()).or_insert((0, 0));
                    entry.0 += 1; // total
                    if p.get(&n.id).status == ProgressStatus::Mastered {
                        entry.1 += 1; // mastered
                    }
                }

                let mut subject_list: Vec<_> = subjects.into_iter().collect();
                subject_list.sort_by(|a, b| b.1.0.cmp(&a.1.0)); // sort by total nodes
                subject_list.truncate(6); // top 6 subjects

                view! {
                    <div class="skills-list">
                        {subject_list.iter().map(|(subject, (total, mastered))| {
                            let pct = if *total > 0 { *mastered * 100 / *total } else { 0 };
                            let subject = subject.clone();
                            let total = *total;
                            let mastered = *mastered;
                            view! {
                                <div class="skill-row">
                                    <div class="skill-info">
                                        <span class="skill-name">{subject}</span>
                                        <span class="skill-domain">{mastered}"/"{total}" topics"</span>
                                    </div>
                                    <div class="skill-bar-container">
                                        <div class="skill-bar" style=format!("width: {}%", pct)></div>
                                    </div>
                                    <span class="skill-pct">{pct}"%"</span>
                                </div>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                }
            }}
        </div>
    }
}

/// Pending TEND card — shows economic value earned from learning.
#[component]
fn PendingTendCard() -> impl IntoView {
    let ledger = crate::persistence::PendingTendLedger::load();
    let total = ledger.total_earned;
    let pending = ledger.total_pending;
    let event_count = ledger.events.len();
    let should_prompt = ledger.should_prompt_connection();

    view! {
        <div class="dash-card tend-card">
            <h3>"Learning Economy"</h3>
            <div style="display: flex; align-items: baseline; gap: 0.5rem">
                <span style="font-size: 2rem; font-weight: 700; color: var(--primary)">{format!("{:.1}", total)}</span>
                <span style="font-size: 0.9rem; color: var(--text-secondary)">"TEND earned"</span>
            </div>
            {if pending > 0.0 {
                view! {
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem">
                        {format!("{:.1} pending", pending)}" \u{2014} connect to claim"
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem">
                {event_count}" sessions \u{2014} 1 TEND = 1 hour community service"
            </div>
            {if should_prompt {
                view! {
                    <div role="status" style="margin-top: 0.75rem; padding: 0.75rem; background: var(--primary, #7c3aed); color: #fff; border-radius: 0.5rem; text-align: center">
                        <strong>"Your knowledge has value."</strong>
                        <div style="font-size: 0.85rem; margin-top: 0.25rem">
                            "Connect to the network to secure "{format!("{:.1}", pending)}" TEND and join the mesh."
                        </div>
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
        </div>
    }
}
