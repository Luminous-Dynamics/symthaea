// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mock Exam Mode — timed practice exams that simulate NSC conditions.
//!
//! Pulls practice problems from lesson JSONs for Grade 12 topics,
//! groups by Paper 1/2, applies a countdown timer, and scores results
//! with per-topic breakdown.

use leptos::prelude::*;
use serde::Deserialize;
use wasm_bindgen_futures::JsFuture;

use crate::curriculum::{caps_graph, use_progress, use_set_progress};

// ============================================================
// Types
// ============================================================

#[derive(Clone, Debug, Deserialize)]
struct ExamProblem {
    question: String,
    answer: String,
    #[serde(default)]
    difficulty_permille: u16,
    #[serde(default)]
    topic_title: String,
    #[serde(default)]
    topic_id: String,
    #[serde(default)]
    marks: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExamState {
    Setup,
    InProgress,
    Reviewing,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExamPaper {
    Paper1,
    Paper2,
}

impl ExamPaper {
    fn label(&self) -> &'static str {
        match self {
            ExamPaper::Paper1 => "Paper 1",
            ExamPaper::Paper2 => "Paper 2",
        }
    }
    fn duration_mins(&self) -> u32 {
        180 // 3 hours per paper
    }
    fn short_duration_mins(&self) -> u32 {
        30 // Quick practice mode
    }
}

// ============================================================
// Component
// ============================================================

#[component]
pub fn MockExamPage() -> impl IntoView {
    let (state, set_state) = signal(ExamState::Setup);
    let (paper, set_paper) = signal(ExamPaper::Paper1);
    let (quick_mode, set_quick_mode) = signal(true); // 30 min quick practice by default
    let (problems, set_problems) = signal(Vec::<ExamProblem>::new());
    let (current_q, set_current_q) = signal(0_usize);
    let (answers, set_answers) = signal(Vec::<Option<bool>>::new()); // None = unanswered, Some(true) = correct
    let (time_left, set_time_left) = signal(30 * 60_i32); // seconds
    let (revealed, set_revealed) = signal(false);
    let set_tracker = crate::study_tracker::use_set_tracker();

    // Generate exam problems from the graph
    let generate_problems = move |p: ExamPaper, quick: bool| {
        let graph = caps_graph();
        let paper_num = match p { ExamPaper::Paper1 => 1, ExamPaper::Paper2 => 2 };
        let max_problems = if quick { 10 } else { 25 };

        let mut exam_problems = Vec::new();
        for node in &graph.nodes {
            let is_gr12 = node.grade_levels.first().map(|g| g == "Grade12").unwrap_or(false);
            let is_paper = node.exam_weight.as_ref().map(|w| w.paper == paper_num).unwrap_or(false);
            if !is_gr12 || !is_paper { continue; }

            let marks = node.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
            // Generate 1-3 problems per topic based on marks
            let num = (marks / 10).max(1).min(3);
            for i in 0..num {
                let diff = match i {
                    0 => 300,
                    1 => 600,
                    _ => 800,
                };
                exam_problems.push(ExamProblem {
                    question: format!("{}  ({}m)\n\nSolve this {} problem at difficulty level {}.",
                        node.title, marks / num.max(1),
                        node.subdomain, if diff <= 300 { "basic" } else if diff <= 600 { "intermediate" } else { "advanced" }),
                    answer: format!("See worked solutions in the {} study page.", node.title),
                    difficulty_permille: diff,
                    topic_title: node.title.clone(),
                    topic_id: node.id.clone(),
                    marks: marks / num.max(1),
                });
            }
        }

        // Shuffle deterministically (based on current date for daily variety)
        let day_seed = (js_sys::Date::new_0().get_date() as usize) * 31 + (js_sys::Date::new_0().get_month() as usize);
        for i in 0..exam_problems.len() {
            let j = (i + day_seed * 7 + i * 13) % exam_problems.len();
            exam_problems.swap(i, j);
        }

        exam_problems.truncate(max_problems);
        exam_problems
    };

    let start_exam = move |_| {
        let p = paper.get_untracked();
        let quick = quick_mode.get_untracked();
        let probs = generate_problems(p, quick);
        let duration = if quick { p.short_duration_mins() } else { p.duration_mins() };

        set_answers.set(vec![None; probs.len()]);
        set_problems.set(probs);
        set_current_q.set(0);
        set_time_left.set((duration * 60) as i32);
        set_revealed.set(false);
        set_state.set(ExamState::InProgress);

        // Record study activity
        set_tracker.update(|t| t.record_study_day());

        // Start timer
        let handle = gloo_timers::callback::Interval::new(1_000, move || {
            set_time_left.update(|t| {
                if *t > 0 {
                    *t -= 1;
                } else {
                    set_state.set(ExamState::Complete);
                }
            });
        });
        std::mem::forget(handle);
    };

    let minutes = move || time_left.get() / 60;
    let secs = move || time_left.get() % 60;

    view! {
        <div class="caps-skill-map">
            <a href="/exam-prep" class="refuge-back-link">"\u{2190} Back to Exam Prep"</a>
            <h1 style="font-size: 1.5rem; margin: 1rem 0 0.5rem">"Mock Exam"</h1>

            {move || match state.get() {
                ExamState::Setup => {
                    view! {
                        <div style="max-width: 500px">
                            <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.6">
                                "Practice under exam conditions. Problems are drawn from Grade 12 topics weighted by NSC marks."
                            </p>

                            // Paper selection
                            <div style="display: flex; gap: 0.75rem; margin-bottom: 1rem">
                                <button
                                    class=move || if paper.get() == ExamPaper::Paper1 { "caps-filter-btn active" } else { "caps-filter-btn" }
                                    on:click=move |_| set_paper.set(ExamPaper::Paper1)
                                >"Paper 1 (Algebra, Calculus, Finance)"</button>
                                <button
                                    class=move || if paper.get() == ExamPaper::Paper2 { "caps-filter-btn active" } else { "caps-filter-btn" }
                                    on:click=move |_| set_paper.set(ExamPaper::Paper2)
                                >"Paper 2 (Geometry, Trig, Stats)"</button>
                            </div>

                            // Duration selection
                            <div style="display: flex; gap: 0.75rem; margin-bottom: 1.5rem">
                                <button
                                    class=move || if quick_mode.get() { "caps-filter-btn active" } else { "caps-filter-btn" }
                                    on:click=move |_| set_quick_mode.set(true)
                                >"Quick (30 min, 10 Qs)"</button>
                                <button
                                    class=move || if !quick_mode.get() { "caps-filter-btn active" } else { "caps-filter-btn" }
                                    on:click=move |_| set_quick_mode.set(false)
                                >"Full Exam (3 hrs, 25 Qs)"</button>
                            </div>

                            <button class="session-start-btn" on:click=start_exam style="max-width: 300px">
                                <span style="font-size: 1.1rem; font-weight: 700">"Start Exam"</span>
                                <span style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.15rem">
                                    {move || paper.get().label()}" \u{2014} "
                                    {move || if quick_mode.get() { "30 minutes" } else { "3 hours" }}
                                </span>
                            </button>
                        </div>
                    }.into_any()
                }

                ExamState::InProgress => {
                    let probs = problems.get();
                    let total = probs.len();
                    let idx = current_q.get().min(total.saturating_sub(1));

                    if total == 0 {
                        return view! { <p>"No problems available for this paper."</p> }.into_any();
                    }

                    let prob = &probs[idx];
                    let question = prob.question.clone();
                    let answer = prob.answer.clone();
                    let topic = prob.topic_title.clone();
                    let marks = prob.marks;
                    let topic_id = prob.topic_id.clone();

                    view! {
                        // Timer bar
                        <div class="exam-timer-bar">
                            <span style="font-family: monospace; font-size: 1rem; font-weight: 600">
                                {move || format!("{:02}:{:02}", minutes(), secs())}
                            </span>
                            <span style="font-size: 0.8rem; color: var(--text-secondary)">
                                "Q "{idx + 1}"/" {total}
                            </span>
                            <button
                                style="font-size: 0.75rem; padding: 0.25rem 0.5rem; border: 1px solid var(--border); border-radius: 4px; background: var(--surface); color: var(--text-secondary); cursor: pointer; font-family: inherit"
                                on:click=move |_| set_state.set(ExamState::Complete)
                            >"End Exam"</button>
                        </div>

                        // Question card
                        <div style="padding: 1.25rem; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; margin: 1rem 0">
                            <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.5rem">
                                {topic}" \u{2014} "{marks}"m"
                            </div>
                            <div style="font-size: 0.95rem; line-height: 1.7; white-space: pre-line">
                                {question}
                            </div>

                            // Reveal answer
                            {move || if revealed.get() {
                                let ans = answer.clone();
                                let tid = topic_id.clone();
                                view! {
                                    <div class="caps-problem-a" style="margin-top: 1rem">
                                        {ans}
                                    </div>
                                    <div style="display: flex; gap: 0.5rem; margin-top: 0.75rem">
                                        <button
                                            class="caps-filter-btn"
                                            style="font-size: 0.8rem; border-color: var(--mastery-green); color: var(--mastery-green)"
                                            on:click=move |_| {
                                                set_answers.update(|a| { if idx < a.len() { a[idx] = Some(true); } });
                                                set_revealed.set(false);
                                                if idx + 1 < total { set_current_q.set(idx + 1); }
                                                else { set_state.set(ExamState::Complete); }
                                            }
                                        >"Correct"</button>
                                        <button
                                            class="caps-filter-btn"
                                            style="font-size: 0.8rem"
                                            on:click=move |_| {
                                                set_answers.update(|a| { if idx < a.len() { a[idx] = Some(false); } });
                                                set_revealed.set(false);
                                                if idx + 1 < total { set_current_q.set(idx + 1); }
                                                else { set_state.set(ExamState::Complete); }
                                            }
                                        >"Incorrect"</button>
                                        <a href=format!("/study/{}", tid) style="font-size: 0.8rem; color: var(--info); text-decoration: none; padding: 0.3rem 0.5rem">
                                            "Study this topic \u{2192}"
                                        </a>
                                    </div>
                                }.into_any()
                            } else {
                                view! {
                                    <button
                                        class="caps-filter-btn active"
                                        style="margin-top: 1rem"
                                        on:click=move |_| set_revealed.set(true)
                                    >"Show Answer"</button>
                                }.into_any()
                            }}
                        </div>

                        // Question navigation dots
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; justify-content: center">
                            {(0..total).map(|i| {
                                let color = move || {
                                    let a = answers.get();
                                    match a.get(i) {
                                        Some(Some(true)) => "var(--mastery-green)",
                                        Some(Some(false)) => "var(--warning)",
                                        _ => if i == current_q.get() { "var(--primary)" } else { "var(--border)" },
                                    }
                                };
                                view! {
                                    <button
                                        style=move || format!("width: 24px; height: 24px; border-radius: 50%; border: 2px solid {}; background: {}; cursor: pointer; font-size: 0.6rem; color: var(--text); font-family: inherit",
                                            color(), if i == current_q.get() { color() } else { "transparent" })
                                        on:click=move |_| { set_current_q.set(i); set_revealed.set(false); }
                                    >
                                        {i + 1}
                                    </button>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }

                ExamState::Complete | ExamState::Reviewing => {
                    let a = answers.get();
                    let total = a.len();
                    let correct = a.iter().filter(|x| **x == Some(true)).count();
                    let incorrect = a.iter().filter(|x| **x == Some(false)).count();
                    let unanswered = a.iter().filter(|x| x.is_none()).count();
                    let pct = if total > 0 { correct * 100 / total } else { 0 };

                    let probs = problems.get();
                    let total_marks: u16 = probs.iter().map(|p| p.marks).sum();
                    let earned_marks: u16 = probs.iter().enumerate()
                        .filter(|(i, _)| a.get(*i) == Some(&Some(true)))
                        .map(|(_, p)| p.marks)
                        .sum();

                    let grade = match pct {
                        80..=100 => ("A", "var(--mastery-green)"),
                        70..=79 => ("B", "var(--info)"),
                        60..=69 => ("C", "var(--mastery-yellow)"),
                        50..=59 => ("D", "var(--warning)"),
                        40..=49 => ("E", "var(--warning)"),
                        _ => ("F", "var(--error)"),
                    };

                    view! {
                        <div style="text-align: center; padding: 1rem 0">
                            <div style=format!("font-size: 3rem; font-weight: 700; color: {}", grade.1)>
                                {grade.0}
                            </div>
                            <div style="font-size: 1.2rem; font-weight: 600; margin-top: 0.25rem">
                                {earned_marks}"/"{total_marks}" marks ("{pct}"%)"
                            </div>
                            <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem">
                                {correct}" correct, "{incorrect}" incorrect, "{unanswered}" skipped"
                            </div>

                            // Per-topic breakdown
                            <div style="margin-top: 1.5rem; text-align: left">
                                <h3 style="font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem">"Topic Breakdown"</h3>
                                {probs.iter().enumerate().map(|(i, p)| {
                                    let got_it = a.get(i) == Some(&Some(true));
                                    let color = if got_it { "var(--mastery-green)" } else { "var(--text-tertiary)" };
                                    let icon = if got_it { "\u{2714}" } else if a.get(i) == Some(&Some(false)) { "\u{2718}" } else { "\u{2014}" };
                                    let title = p.topic_title.clone();
                                    let marks = p.marks;
                                    view! {
                                        <div style=format!("display: flex; align-items: center; gap: 0.5rem; padding: 0.35rem 0; font-size: 0.8rem; color: {}", color)>
                                            <span style="width: 1rem; text-align: center">{icon}</span>
                                            <span style="flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">{title}</span>
                                            <span style="color: var(--text-tertiary)">{marks}"m"</span>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>

                            <div style="display: flex; gap: 0.75rem; justify-content: center; margin-top: 1.5rem">
                                <button class="caps-filter-btn active" on:click=move |_| set_state.set(ExamState::Setup)>
                                    "Try Another"
                                </button>
                                <a href="/exam-prep" class="caps-filter-btn" style="text-decoration: none">"Back to Exam Prep"</a>
                            </div>
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
