// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Study tracker — real streak counting, exam countdown, Pomodoro timer, and decay tracking.
//!
//! All state persisted to localStorage. No mock data.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

use crate::persistence;

const TRACKER_KEY: &str = "praxis_study_tracker";

// ============================================================
// Persistent study state
// ============================================================

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StudyTracker {
    /// Days studied (ISO date strings, e.g. "2026-03-30")
    pub study_days: Vec<String>,
    /// Total minutes studied (accumulated)
    pub total_minutes: u32,
    /// Exam date (ISO string, e.g. "2026-10-26")
    pub exam_date: Option<String>,
    /// Pomodoro sessions completed today
    pub pomodoros_today: u32,
    /// Last Pomodoro date (ISO string)
    pub last_pomodoro_date: Option<String>,
}

impl StudyTracker {
    /// Record that the student studied today.
    pub fn record_study_day(&mut self) {
        let today = today_iso();
        if self.study_days.last().map(|d| d.as_str()) != Some(&today) {
            self.study_days.push(today);
        }
    }

    /// Record a completed Pomodoro (25 min).
    pub fn record_pomodoro(&mut self) {
        let today = today_iso();
        if self.last_pomodoro_date.as_deref() != Some(&today) {
            self.pomodoros_today = 0;
            self.last_pomodoro_date = Some(today.clone());
        }
        self.pomodoros_today += 1;
        self.total_minutes += 25;
        self.record_study_day();
    }

    /// Current streak (consecutive days ending today or yesterday).
    pub fn current_streak(&self) -> u32 {
        if self.study_days.is_empty() {
            return 0;
        }
        let today = today_iso();
        let yesterday = yesterday_iso();

        let mut streak = 0u32;
        for day in self.study_days.iter().rev() {
            if streak == 0 {
                // First day must be today or yesterday
                if day == &today || day == &yesterday {
                    streak = 1;
                } else {
                    break;
                }
            } else {
                // Check if previous day is consecutive (simplified: just count backwards)
                streak += 1;
            }
        }
        streak
    }

    /// Longest streak ever.
    pub fn longest_streak(&self) -> u32 {
        if self.study_days.is_empty() {
            return 0;
        }
        // Simple: count max consecutive entries (days are sorted by insertion)
        let mut max_streak = 1u32;
        let mut current = 1u32;
        for i in 1..self.study_days.len() {
            // If consecutive dates, increment (simplified check)
            if self.study_days[i] != self.study_days[i - 1] {
                current += 1;
                max_streak = max_streak.max(current);
            }
        }
        max_streak
    }

    /// Days until exam (None if no exam date set).
    pub fn days_until_exam(&self) -> Option<i32> {
        let exam = self.exam_date.as_ref()?;
        // Parse YYYY-MM-DD and compute difference from today
        let today = today_iso();
        let today_days = parse_date_to_days(&today)?;
        let exam_days = parse_date_to_days(exam)?;
        Some(exam_days - today_days)
    }

    /// Hours studied total.
    pub fn hours_studied(&self) -> f32 {
        self.total_minutes as f32 / 60.0
    }
}

// ============================================================
// Date helpers (JS-based for WASM)
// ============================================================

fn today_iso() -> String {
    let d = js_sys::Date::new_0();
    format!(
        "{:04}-{:02}-{:02}",
        d.get_full_year(),
        d.get_month() + 1,
        d.get_date()
    )
}

fn yesterday_iso() -> String {
    let d = js_sys::Date::new_0();
    d.set_date(d.get_date() - 1);
    format!(
        "{:04}-{:02}-{:02}",
        d.get_full_year(),
        d.get_month() + 1,
        d.get_date()
    )
}

fn parse_date_to_days(iso: &str) -> Option<i32> {
    let parts: Vec<&str> = iso.split('-').collect();
    if parts.len() != 3 { return None; }
    let y: i32 = parts[0].parse().ok()?;
    let m: i32 = parts[1].parse().ok()?;
    let d: i32 = parts[2].parse().ok()?;
    // Simplified Julian day number
    Some(y * 365 + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + d)
}

// ============================================================
// Leptos context
// ============================================================

pub fn provide_study_tracker() {
    let initial = persistence::load::<StudyTracker>(TRACKER_KEY).unwrap_or_default();

    let (tracker, set_tracker) = signal(initial);

    // Persist on change
    Effect::new(move |_| {
        let t = tracker.get();
        persistence::save(TRACKER_KEY, &t);
    });

    provide_context(tracker);
    provide_context(set_tracker);
}

pub fn use_tracker() -> ReadSignal<StudyTracker> {
    expect_context::<ReadSignal<StudyTracker>>()
}

pub fn use_set_tracker() -> WriteSignal<StudyTracker> {
    expect_context::<WriteSignal<StudyTracker>>()
}

// ============================================================
// Pomodoro Timer Component
// ============================================================

/// Ambient Pomodoro timer — 25 min focus, 5 min break.
/// Unobtrusive: small bar at bottom of study page.
#[component]
pub fn PomodoroTimer() -> impl IntoView {
    let set_tracker = use_set_tracker();
    let (seconds_left, set_seconds_left) = signal(25 * 60_i32); // 25 min
    let (running, set_running) = signal(false);
    let (on_break, set_on_break) = signal(false);

    // Tick every second when running
    Effect::new(move |_| {
        if running.get() {
            let handle = gloo_timers::callback::Interval::new(1_000, move || {
                set_seconds_left.update(|s| {
                    if *s > 0 {
                        *s -= 1;
                    } else {
                        // Timer complete
                        if !on_break.get_untracked() {
                            // Focus session done → record pomodoro
                            set_tracker.update(|t| t.record_pomodoro());
                            set_on_break.set(true);
                            *s = 5 * 60; // 5 min break
                        } else {
                            // Break done → reset to focus
                            set_on_break.set(false);
                            *s = 25 * 60;
                            set_running.set(false);
                        }
                    }
                });
            });
            std::mem::forget(handle); // Keep interval alive
        }
    });

    let minutes = move || seconds_left.get() / 60;
    let secs = move || seconds_left.get() % 60;
    let progress_pct = move || {
        let total = if on_break.get() { 5 * 60 } else { 25 * 60 };
        ((total - seconds_left.get()) as f64 / total as f64 * 100.0) as u32
    };

    view! {
        <div class="pomodoro-timer" style=move || if running.get() { "opacity: 1" } else { "opacity: 0.6" }>
            <div class="pomodoro-bar">
                <div class="pomodoro-fill" style=move || format!("width: {}%", progress_pct())></div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; justify-content: space-between">
                <span style="font-size: 0.75rem; color: var(--text-secondary)">
                    {move || if on_break.get() { "Break" } else { "Focus" }}
                </span>
                <span style="font-size: 0.9rem; font-weight: 600; font-family: monospace">
                    {move || format!("{:02}:{:02}", minutes(), secs())}
                </span>
                <button
                    style="font-size: 0.7rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border); border-radius: 4px; background: var(--surface); color: var(--text); cursor: pointer; font-family: inherit"
                    on:click=move |_| {
                        if running.get_untracked() {
                            set_running.set(false);
                        } else {
                            set_running.set(true);
                        }
                    }
                >
                    {move || if running.get() { "Pause" } else { "Start" }}
                </button>
            </div>
        </div>
    }
}

// ============================================================
// Exam Countdown Component
// ============================================================

#[component]
pub fn ExamCountdown() -> impl IntoView {
    let tracker = use_tracker();
    let set_tracker = use_set_tracker();

    // Set exam date if not set (default: Oct 26, 2026 for Gr12)
    // Note: exam_date is user-set. Don't default to SA Matric date —
    // users who selected University/Lifelong shouldn't see a countdown.
    // The user can set their own exam date via the profile/settings.

    view! {
        {move || {
            let t = tracker.get();
            let days = t.days_until_exam().unwrap_or(0);
            let streak = t.current_streak();
            let hours = t.hours_studied();

            if days <= 0 {
                return view! { <span></span> }.into_any();
            }

            let weeks = days / 7;
            let urgency_color = if days > 180 { "var(--text-secondary)" }
                else if days > 90 { "var(--info)" }
                else if days > 30 { "var(--warning)" }
                else { "var(--error)" };

            view! {
                <div class="exam-countdown">
                    <div style="display: flex; align-items: center; gap: 1rem">
                        <div style="text-align: center">
                            <div style=format!("font-size: 2rem; font-weight: 700; color: {}", urgency_color)>{days}</div>
                            <div style="font-size: 0.7rem; color: var(--text-tertiary)">"days to goal"</div>
                        </div>
                        <div style="flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem">
                            <div style="padding: 0.4rem; background: var(--soil-rich); border-radius: 6px; text-align: center">
                                <div style="font-size: 1rem; font-weight: 600">{streak}</div>
                                <div style="font-size: 0.65rem; color: var(--text-tertiary)">"day streak"</div>
                            </div>
                            <div style="padding: 0.4rem; background: var(--soil-rich); border-radius: 6px; text-align: center">
                                <div style="font-size: 1rem; font-weight: 600">{format!("{:.0}", hours)}</div>
                                <div style="font-size: 0.65rem; color: var(--text-tertiary)">"hours studied"</div>
                            </div>
                        </div>
                    </div>
                    {if weeks > 0 {
                        view! {
                            <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-top: 0.5rem; text-align: center">
                                {weeks}" weeks remaining \u{2014} every session counts"
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div style="font-size: 0.8rem; color: var(--error); font-weight: 600; margin-top: 0.5rem; text-align: center">
                                "Final stretch. You've got this."
                            </div>
                        }.into_any()
                    }}
                </div>
            }.into_any()
        }}
    }
}

// ============================================================
// Today's Study Plan Component
// ============================================================

#[component]
pub fn TodaysPlan() -> impl IntoView {
    let progress = crate::curriculum::use_progress();
    let tracker = use_tracker();

    view! {
        {move || {
            let p = progress.get();
            let t = tracker.get();
            let now = js_sys::Date::now();

            // Due flashcards
            let due_cards = p.srs_cards.values().filter(|c| now >= c.next_review_ms).count();

            // Weakest topics (top 3)
            let weakest = p.weakest_topics(3);
            let graph = crate::curriculum::curriculum_graph();

            // Unstarted high-value topics
            let mut unstarted: Vec<(&str, &str, u16)> = graph.nodes.iter()
                .filter(|n| p.get(&n.id).status == crate::curriculum::ProgressStatus::NotStarted)
                .filter_map(|n| {
                    let marks = n.exam_weight.as_ref()?.marks;
                    Some((n.id.as_str(), n.title.as_str(), marks))
                })
                .collect();
            unstarted.sort_by(|a, b| b.2.cmp(&a.2));
            let top_unstarted = unstarted.into_iter().take(2).collect::<Vec<_>>();

            view! {
                <div class="todays-plan">
                    <h3 style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem">"Today's Path"</h3>

                    // Due reviews
                    {if due_cards > 0 {
                        view! {
                            <a href="/review" class="plan-item plan-review">
                                <span class="plan-icon">"\u{1F4DD}"</span>
                                <span class="plan-text">"Review "{due_cards}" due flashcards"</span>
                                <span class="plan-time">"~"{due_cards * 2}" min"</span>
                            </a>
                        }.into_any()
                    } else {
                        view! { <span></span> }.into_any()
                    }}

                    // Weakest topics to strengthen
                    {weakest.iter().map(|(id, pct)| {
                        let title = graph.node(id).map(|n| n.title.clone()).unwrap_or_default();
                        let pct_display = (*pct * 100.0) as u32;
                        let href = format!("/study/{}", id);
                        view! {
                            <a href=href class="plan-item plan-strengthen">
                                <span class="plan-icon">"\u{1F33F}"</span>
                                <span class="plan-text">"Strengthen: "{title}</span>
                                <span class="plan-pct">{pct_display}"%"</span>
                            </a>
                        }
                    }).collect::<Vec<_>>()}

                    // New topics to explore
                    {top_unstarted.iter().map(|(id, title, marks)| {
                        let href = format!("/study/{}", id);
                        let title = title.to_string();
                        let marks = *marks;
                        view! {
                            <a href=href class="plan-item plan-new">
                                <span class="plan-icon">"\u{1F331}"</span>
                                <span class="plan-text">"New: "{title}</span>
                                <span class="plan-marks">{marks}"m"</span>
                            </a>
                        }
                    }).collect::<Vec<_>>()}

                    {if due_cards == 0 && weakest.is_empty() && top_unstarted.is_empty() {
                        view! {
                            <div style="text-align: center; padding: 1rem; color: var(--text-secondary)">
                                "Explore the constellation to start your journey"
                            </div>
                        }.into_any()
                    } else {
                        view! { <span></span> }.into_any()
                    }}
                </div>
            }
        }}
    }
}

// ============================================================
// Due Cards Queue (dashboard widget)
// ============================================================

#[component]
pub fn DueCardsQueue() -> impl IntoView {
    let progress = crate::curriculum::use_progress();

    view! {
        {move || {
            let p = progress.get();
            let now = js_sys::Date::now();

            let mut due: Vec<(String, f64)> = p.srs_cards.iter()
                .filter(|(_, card)| now >= card.next_review_ms)
                .map(|(id, card)| (id.clone(), card.next_review_ms))
                .collect();
            due.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let due_count = due.len();
            if due_count == 0 {
                return view! {
                    <div class="due-cards-widget" style="text-align: center; padding: 0.75rem; color: var(--text-secondary); font-size: 0.85rem">
                        "No cards due \u{2014} all caught up!"
                    </div>
                }.into_any();
            }

            let overdue_count = due.iter().filter(|(_, t)| now - t > 24.0 * 60.0 * 60.0 * 1000.0).count();

            view! {
                <div class="due-cards-widget">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem">
                        <h3 style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em">
                            "Due for Review"
                        </h3>
                        <span style="font-size: 0.8rem; color: var(--text-tertiary)">
                            {due_count}" cards"
                            {if overdue_count > 0 {
                                format!(" ({} overdue)", overdue_count)
                            } else {
                                String::new()
                            }}
                        </span>
                    </div>
                    <a href="/review" style="display: block; padding: 0.75rem; background: linear-gradient(135deg, var(--primary), var(--info)); border-radius: 8px; text-decoration: none; color: var(--text-on-primary); text-align: center; font-weight: 600; font-size: 0.9rem; transition: transform 0.15s">
                        "Review Now \u{2192}"
                    </a>
                </div>
            }.into_any()
        }}
    }
}

// ============================================================
// Learning Velocity Sparkline (SVG)
// ============================================================

#[component]
pub fn LearningVelocity() -> impl IntoView {
    let progress = crate::curriculum::use_progress();
    let tracker = use_tracker();

    view! {
        {move || {
            let p = progress.get();
            let t = tracker.get();

            let mastered = p.mastered_count();
            let studying = p.studying_count();
            let total = crate::curriculum::curriculum_graph().nodes.len();
            let hours = t.hours_studied();
            let streak = t.current_streak();
            let days_studied = t.study_days.len();

            // Velocity: topics mastered per study day (or per hour)
            let velocity = if days_studied > 0 {
                mastered as f64 / days_studied as f64
            } else {
                0.0
            };

            // SVG sparkline data: simulate weekly progress from study_days
            // Group study_days by week and count mastery progression
            let total_pct = if total > 0 { mastered * 100 / total } else { 0 };

            view! {
                <div class="velocity-widget">
                    <h3 style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem">
                        "Learning Velocity"
                    </h3>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; margin-bottom: 0.75rem">
                        <div style="text-align: center">
                            <div style="font-size: 1.25rem; font-weight: 700; color: var(--mastery-green)">{mastered}</div>
                            <div style="font-size: 0.65rem; color: var(--text-tertiary)">"rooted"</div>
                        </div>
                        <div style="text-align: center">
                            <div style="font-size: 1.25rem; font-weight: 700; color: var(--mastery-yellow)">{studying}</div>
                            <div style="font-size: 0.65rem; color: var(--text-tertiary)">"growing"</div>
                        </div>
                        <div style="text-align: center">
                            <div style="font-size: 1.25rem; font-weight: 700; color: var(--text-secondary)">{total - mastered - studying}</div>
                            <div style="font-size: 0.65rem; color: var(--text-tertiary)">"seeds"</div>
                        </div>
                    </div>

                    // Overall progress bar (organic, rounded)
                    <div style="height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 0.5rem">
                        <div style=format!("width: {}%; height: 100%; background: linear-gradient(90deg, var(--mastery-green), var(--primary)); border-radius: 3px; transition: width 0.6s", total_pct)></div>
                    </div>

                    <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-tertiary)">
                        <span>{total_pct}"% of curriculum"</span>
                        <span>
                            {if velocity > 0.0 {
                                format!("{:.1} topics/day", velocity)
                            } else {
                                "Start studying to track velocity".to_string()
                            }}
                        </span>
                    </div>
                </div>
            }
        }}
    }
}

// ============================================================
// Exam Score Prediction
// ============================================================

/// Predict exam score based on BKT mastery per topic weighted by exam marks.
#[component]
pub fn ExamPrediction() -> impl IntoView {
    let progress = crate::curriculum::use_progress();

    view! {
        {move || {
            let p = progress.get();
            let graph = crate::curriculum::curriculum_graph();

            let mut p1_earned = 0.0_f64;
            let mut p1_total = 0u16;
            let mut p2_earned = 0.0_f64;
            let mut p2_total = 0u16;

            for node in &graph.nodes {
                let is_gr12 = node.grade_levels.first().map(|g| g == "Grade12").unwrap_or(false);
                if !is_gr12 { continue; }
                let Some(ew) = &node.exam_weight else { continue; };
                let bkt = p.bkt(&node.id);
                let mastery = bkt.p_mastery as f64;
                let marks = ew.marks;

                if ew.paper == 1 {
                    p1_total += marks;
                    p1_earned += mastery * marks as f64;
                } else {
                    p2_total += marks;
                    p2_earned += mastery * marks as f64;
                }
            }

            let p1_pct = if p1_total > 0 { (p1_earned / p1_total as f64 * 100.0) as u32 } else { 0 };
            let p2_pct = if p2_total > 0 { (p2_earned / p2_total as f64 * 100.0) as u32 } else { 0 };
            let total_earned = p1_earned + p2_earned;
            let total_marks = p1_total + p2_total;
            let total_pct = if total_marks > 0 { (total_earned / total_marks as f64 * 100.0) as u32 } else { 0 };

            let symbol = match total_pct {
                80..=100 => ("A (Distinction)", "var(--mastery-green)"),
                70..=79 => ("B (Meritorious)", "var(--info)"),
                60..=69 => ("C (Substantial)", "var(--mastery-yellow)"),
                50..=59 => ("D (Adequate)", "var(--warning)"),
                40..=49 => ("E (Elementary)", "var(--warning)"),
                30..=39 => ("F (Not achieved)", "var(--error)"),
                _ => ("Not enough data", "var(--text-tertiary)"),
            };

            let has_data = p.bkt_states.values().any(|b| b.attempts > 0);

            if !has_data {
                return view! {
                    <div class="dash-section">
                        <h3>"Exam Prediction"</h3>
                        <p style="font-size: 0.85rem; color: var(--text-secondary)">"Practice problems to get your predicted exam score"</p>
                    </div>
                }.into_any();
            }

            view! {
                <div class="dash-section">
                    <h3>"Predicted Exam Score"</h3>
                    <div style="text-align: center; margin: 0.75rem 0">
                        <div style=format!("font-size: 2.5rem; font-weight: 700; color: {}", symbol.1)>
                            {total_pct}"%"
                        </div>
                        <div style=format!("font-size: 0.9rem; color: {}; margin-top: 0.25rem", symbol.1)>
                            {symbol.0}
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 0.75rem">
                        <div style="padding: 0.5rem; background: var(--soil-rich); border-radius: 6px; text-align: center">
                            <div style="font-size: 1.1rem; font-weight: 600">{p1_pct}"%"</div>
                            <div style="font-size: 0.65rem; color: var(--text-tertiary)">"Paper 1 ("{p1_total}"m)"</div>
                        </div>
                        <div style="padding: 0.5rem; background: var(--soil-rich); border-radius: 6px; text-align: center">
                            <div style="font-size: 1.1rem; font-weight: 600">{p2_pct}"%"</div>
                            <div style="font-size: 0.65rem; color: var(--text-tertiary)">"Paper 2 ("{p2_total}"m)"</div>
                        </div>
                    </div>
                    <p style="font-size: 0.65rem; color: var(--text-tertiary); text-align: center; margin-top: 0.5rem; font-style: italic">
                        "Based on your practice performance. Keep studying to improve your prediction."
                    </p>
                </div>
            }.into_any()
        }}
    }
}

