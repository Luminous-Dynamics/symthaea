// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exam Prep page — study schedule generator for NSC Matric exams.
//!
//! Shows all Grade 12 topics weighted by exam marks, highlights weak areas
//! (low mastery + high weight = highest priority), and generates a weekly
//! study schedule.

use leptos::prelude::*;

use crate::curriculum::{curriculum_graph, use_progress, use_set_progress, ProgressStatus};
use crate::student_profile::use_profile;

#[component]
pub fn ExamPrepPage() -> impl IntoView {
    let progress = use_progress();
    let set_progress = use_set_progress();
    let graph = curriculum_graph();
    let profile = use_profile();

    // Use exam date from student profile if available, otherwise default
    let initial_date = {
        let p = profile.get_untracked();
        if p.exam_date.is_empty() { "2026-10-26".to_string() } else { p.exam_date.clone() }
    };
    let (exam_date, set_exam_date) = signal(initial_date);
    let (hours_per_day, set_hours_per_day) = signal(3u8);
    let (show_schedule, set_show_schedule) = signal(false);

    // All Gr12 topics with exam weights, sorted by priority
    let exam_topics = Memo::new(move |_| {
        let p = progress.get();
        let mut topics: Vec<_> = graph.nodes.iter()
            .filter(|n| n.grade_levels.first().map(|g| g == "Grade12").unwrap_or(false))
            .map(|n| {
                let status = p.get(&n.id).status;
                let mastery = p.get(&n.id).mastery_permille;
                let weight = n.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
                let paper = n.exam_weight.as_ref().map(|w| w.paper).unwrap_or(0);
                let pct = n.exam_weight.as_ref().map(|w| w.percentage).unwrap_or(0.0);
                // Priority = exam_weight * (1 - mastery/1000)
                let priority = weight as f32 * (1.0 - mastery as f32 / 1000.0);
                (n.id.clone(), n.title.clone(), n.subject_area.clone(), paper, weight, pct, mastery, status, priority, n.estimated_hours)
            })
            .collect();
        topics.sort_by(|a, b| b.8.partial_cmp(&a.8).unwrap_or(std::cmp::Ordering::Equal));
        topics
    });

    // Math totals
    let math_stats = Memo::new(move |_| {
        let topics = exam_topics.get();
        let math: Vec<_> = topics.iter().filter(|t| t.2 == "Mathematics").collect();
        let mastered = math.iter().filter(|t| t.7 == ProgressStatus::Mastered).count();
        (math.len(), mastered)
    });

    // Physics totals
    let phys_stats = Memo::new(move |_| {
        let topics = exam_topics.get();
        let phys: Vec<_> = topics.iter().filter(|t| t.2 == "PhysicalSciences").collect();
        let mastered = phys.iter().filter(|t| t.7 == ProgressStatus::Mastered).count();
        (phys.len(), mastered)
    });

    view! {
        <div class="praxis-skill-map">
            <a href="/dashboard" style="color: var(--primary); text-decoration: none; font-size: 0.85rem">
                "\u{2190} Back to Dashboard"
            </a>

            <h1 style="font-size: 1.5rem; margin: 1rem 0 0.5rem">"Exam Preparation"</h1>
            <p style="color: var(--text-secondary); margin-bottom: 1rem">
                "Topics ranked by priority: exam weight \u{00D7} (1 \u{2212} mastery)"
            </p>

            // Mock exam link
            <a href="/mock-exam" style="display: block; padding: 0.75rem 1rem; background: linear-gradient(135deg, var(--primary), #06b6d4); border-radius: 10px; text-decoration: none; color: var(--text-on-primary); margin-bottom: 1.5rem; text-align: center; font-weight: 600; transition: transform 0.15s">
                "\u{23F1} Take a Mock Exam"
            </a>

            // Summary cards
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1.5rem">
                <div class="praxis-detail" style="text-align: center">
                    <div style="font-size: 0.8rem; color: var(--subject-math)">"Mathematics"</div>
                    {move || {
                        let (total, mastered) = math_stats.get();
                        view! {
                            <div style="font-size: 2rem; font-weight: 700">{mastered}"/" {total}</div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary)">"topics mastered"</div>
                        }
                    }}
                </div>
                <div class="praxis-detail" style="text-align: center">
                    <div style="font-size: 0.8rem; color: var(--subject-physics)">"Physical Sciences"</div>
                    {move || {
                        let (total, mastered) = phys_stats.get();
                        view! {
                            <div style="font-size: 2rem; font-weight: 700">{mastered}"/" {total}</div>
                            <div style="font-size: 0.75rem; color: var(--text-secondary)">"topics mastered"</div>
                        }
                    }}
                </div>
            </div>

            // Study schedule config
            <div class="praxis-detail" style="margin-bottom: 1.5rem">
                <h3 style="font-size: 1rem; margin-bottom: 1rem">"Study Schedule"</h3>
                <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem">
                    <div>
                        <label style="font-size: 0.8rem; color: var(--text-secondary)">"Exam date"</label><br/>
                        <input type="date"
                            style="padding: 4px 8px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 0.85rem"
                            prop:value=move || exam_date.get()
                            on:input=move |ev| {
                                let v = leptos::prelude::event_target_value(&ev);
                                set_exam_date.set(v);
                            }
                        />
                    </div>
                    <div>
                        <label style="font-size: 0.8rem; color: var(--text-secondary)">"Hours per day"</label><br/>
                        <input type="number" min="1" max="12"
                            style="padding: 4px 8px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text); font-size: 0.85rem; width: 60px"
                            prop:value=move || hours_per_day.get().to_string()
                            on:input=move |ev| {
                                let v: u8 = leptos::prelude::event_target_value(&ev).parse().unwrap_or(3);
                                set_hours_per_day.set(v);
                            }
                        />
                    </div>
                </div>
                <button
                    class="praxis-filter-btn active"
                    on:click=move |_| set_show_schedule.update(|v| *v = !*v)
                >
                    {move || if show_schedule.get() { "Hide Schedule" } else { "Generate Schedule" }}
                </button>

                // Generated schedule
                {move || {
                    if !show_schedule.get() { return view! { <span></span> }.into_any(); }

                    let p = progress.get();
                    let hours = hours_per_day.get() as u32;
                    let hours_per_week = hours * 5; // assume 5 days/week

                    // Get unmastered Gr12 topics sorted by priority
                    let mut topics: Vec<_> = graph.nodes.iter()
                        .filter(|n| n.grade_levels.first().map(|g| g == "Grade12").unwrap_or(false)
                            && p.get(&n.id).status != ProgressStatus::Mastered)
                        .map(|n| {
                            let weight = n.exam_weight.as_ref().map(|w| w.marks).unwrap_or(0);
                            let mastery = p.get(&n.id).mastery_permille;
                            let priority = weight as f32 * (1.0 - mastery as f32 / 1000.0);
                            (n.title.clone(), n.id.clone(), n.estimated_hours, weight, priority)
                        })
                        .collect();
                    topics.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));

                    if topics.is_empty() {
                        return view! { <p style="margin-top: 1rem; color: var(--success); font-weight: 600">"All Grade 12 topics mastered!"</p> }.into_any();
                    }

                    let total_hours: u32 = topics.iter().map(|t| t.2).sum();
                    let weeks_needed = (total_hours + hours_per_week - 1) / hours_per_week;

                    let mut schedule_rows = Vec::new();
                    let mut week = 1u32;
                    let mut hours_this_week = 0u32;

                    for (title, id, est_hours, marks, _) in &topics {
                        if hours_this_week + est_hours > hours_per_week && hours_this_week > 0 {
                            week += 1;
                            hours_this_week = 0;
                        }
                        schedule_rows.push((week, title.clone(), id.clone(), *est_hours, *marks));
                        hours_this_week += est_hours;
                    }

                    view! {
                        <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border)">
                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.75rem">
                                {weeks_needed}" weeks \u{00b7} "{total_hours}"h total \u{00b7} "{hours_per_week}"h/week"
                            </div>
                            {schedule_rows.iter().map(|(wk, title, id, hrs, marks)| {
                                let href = format!("/study/{}", id);
                                let title = title.clone();
                                let wk = *wk;
                                let hrs = *hrs;
                                let marks = *marks;
                                view! {
                                    <div style="display: flex; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid var(--bg); font-size: 0.85rem">
                                        <span style="width: 50px; color: var(--text-tertiary); font-weight: 600">"Wk "{wk}</span>
                                        <a href=href style="flex: 1; color: var(--text); text-decoration: none">{title}</a>
                                        <span style="width: 40px; text-align: right; color: var(--text-secondary)">{hrs}"h"</span>
                                        {if marks > 0 {
                                            view! { <span class="praxis-badge praxis-badge-exam" style="margin-left: 0.5rem">{marks}"m"</span> }.into_any()
                                        } else {
                                            view! { <span></span> }.into_any()
                                        }}
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }}
            </div>

            // Topic list by priority
            <h3 style="font-size: 1rem; margin-bottom: 0.75rem">"All Grade 12 Topics by Priority"</h3>
            <div style="font-size: 0.75rem; color: var(--text-tertiary); display: flex; padding: 0.5rem 1rem; border-bottom: 1px solid var(--border)">
                <span style="flex: 1">"Topic"</span>
                <span style="width: 80px; text-align: right">"Paper"</span>
                <span style="width: 60px; text-align: right">"Marks"</span>
                <span style="width: 80px; text-align: right">"Status"</span>
            </div>
            {move || {
                exam_topics.get().iter().map(|(id, title, subject, paper, marks, _pct, _mastery, status, priority, _hours)| {
                    let id = id.clone();
                    let title = title.clone();
                    let paper = *paper;
                    let marks = *marks;
                    let status = *status;
                    let priority = *priority;
                    let href = format!("/study/{}", id);
                    let subject_color = if subject == "Mathematics" { "var(--subject-math)" } else { "var(--subject-physics)" };
                    let status_label = status.label();
                    let status_class = status.css_class();
                    let opacity = if status == ProgressStatus::Mastered { "0.5" } else { "1" };

                    let id_for_toggle = id.clone();
                    view! {
                        <div style=format!("display: flex; align-items: center; padding: 0.6rem 1rem; border-bottom: 1px solid var(--border); opacity: {}", opacity)>
                            <a href=href style="flex: 1; display: flex; align-items: center; gap: 0.5rem; text-decoration: none; color: var(--text)">
                                <span style=format!("width: 3px; height: 20px; border-radius: 2px; background: {}", subject_color)></span>
                                <span style="font-size: 0.9rem">{title}</span>
                            </a>
                            <span style="width: 80px; text-align: right; font-size: 0.8rem; color: var(--text-secondary)">
                                "P" {paper}
                            </span>
                            <span style="width: 60px; text-align: right">
                                <span class="praxis-badge praxis-badge-exam">{marks}</span>
                            </span>
                            <button
                                class=format!("praxis-status-btn {}", if status == ProgressStatus::Mastered { "active-mastered" } else { "" })
                                style="width: 80px; font-size: 0.7rem"
                                on:click=move |_| {
                                    let new_status = if status == ProgressStatus::Mastered {
                                        ProgressStatus::NotStarted
                                    } else {
                                        ProgressStatus::Mastered
                                    };
                                    set_progress.update(|p| p.set_status(&id_for_toggle, new_status));
                                }
                            >
                                {if status == ProgressStatus::Mastered { "\u{2714} Mastered" } else { "Mark done" }}
                            </button>
                        </div>
                    }
                }).collect::<Vec<_>>()
            }}
        </div>
    }
}
