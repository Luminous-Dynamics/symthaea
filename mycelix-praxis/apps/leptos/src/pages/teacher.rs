// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Teacher Dashboard page — class overview with mastery heatmap,
//! at-risk student alerts, aggregate stats, and action buttons.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::student_profile::use_profile;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct StudentMastery {
    name: &'static str,
    skills: Vec<u16>, // mastery permille per skill (0-1000)
}

#[derive(Clone, Debug)]
struct AtRiskAlert {
    student: &'static str,
    reason: &'static str,
    severity: AlertSeverity,
}

#[derive(Clone, Debug)]
enum AlertSeverity {
    Warning,
    Critical,
}

#[derive(Clone, Debug)]
struct ClassStats {
    avg_mastery_pct: u16,
    on_track: u8,
    ahead: u8,
    behind: u8,
    total: u8,
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

fn skill_names() -> Vec<String> {
    crate::curriculum::curriculum_graph().subjects().iter().take(6).map(|s| {
        if s.len() > 8 { format!("{}...", &s[..6]) } else { s.to_string() }
    }).collect()
}

fn skill_full_names() -> Vec<String> {
    crate::curriculum::curriculum_graph().subjects().iter().take(6).map(|s| s.to_string()).collect()
}

fn real_students() -> Vec<StudentMastery> {
    let progress = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress")
        .unwrap_or_default();
    let profile = crate::persistence::load::<crate::student_profile::StudentProfile>("praxis_profile");
    let name = profile.map(|p| p.name).unwrap_or_else(|| "Student".into());
    let name_static: &'static str = Box::leak(name.into_boxed_str());

    let graph = crate::curriculum::curriculum_graph();
    // Compute mastery per top subject (matching SKILL_NAMES)
    let top_subjects: Vec<String> = graph.subjects().iter().take(6).map(|s| s.to_string()).collect();
    let skills: Vec<u16> = top_subjects.iter().map(|subj| {
        let nodes: Vec<_> = graph.nodes.iter().filter(|n| n.subject_area == *subj).collect();
        if nodes.is_empty() { return 0; }
        let total: u32 = nodes.iter().map(|n| progress.bkt(&n.id).p_mastery as u32 * 1000 / 1000).sum::<u32>();
        let avg = total * 1000 / nodes.len() as u32;
        avg.min(1000) as u16
    }).collect();

    vec![StudentMastery { name: name_static, skills }]
}

fn real_at_risk() -> Vec<AtRiskAlert> {
    let progress = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress")
        .unwrap_or_default();
    let tracker = crate::persistence::load::<crate::study_tracker::StudyTracker>("praxis_study_tracker")
        .unwrap_or_default();

    let mut alerts = Vec::new();
    let streak = tracker.current_streak();
    if streak == 0 && !tracker.study_days.is_empty() {
        alerts.push(AtRiskAlert {
            student: "Current student",
            reason: "Study streak broken — no study activity today",
            severity: AlertSeverity::Warning,
        });
    }

    // Check for very low mastery topics that have been attempted
    let weakest = progress.weakest_topics(1);
    if let Some((id, pct)) = weakest.first() {
        if *pct < 0.3 {
            let title = crate::curriculum::curriculum_graph().node(id)
                .map(|n| n.title.as_str()).unwrap_or("Unknown");
            let title_static: &'static str = Box::leak(title.to_string().into_boxed_str());
            let reason: &'static str = Box::leak(
                format!("Struggling with {} ({:.0}% mastery) — needs focused practice", title, pct * 100.0)
                    .into_boxed_str()
            );
            alerts.push(AtRiskAlert {
                student: "Current student",
                reason,
                severity: AlertSeverity::Critical,
            });
        }
    }

    if alerts.is_empty() {
        alerts.push(AtRiskAlert {
            student: "No alerts",
            reason: "All students are on track. When Holochain connects, this page will show real class data.",
            severity: AlertSeverity::Warning,
        });
    }
    alerts
}

fn real_class_stats() -> ClassStats {
    let progress = crate::persistence::load::<crate::curriculum::ProgressStore>("praxis_progress")
        .unwrap_or_default();
    let graph = crate::curriculum::curriculum_graph();
    let total_nodes = graph.nodes.len();
    let mastered = progress.mastered_count();
    let studying = progress.studying_count();
    let avg = if total_nodes > 0 { (mastered * 100 / total_nodes) as u16 } else { 0 };

    ClassStats {
        avg_mastery_pct: avg,
        on_track: if studying > 0 { 1 } else { 0 },
        ahead: if mastered > 10 { 1 } else { 0 },
        behind: if mastered == 0 && studying == 0 { 1 } else { 0 },
        total: 1, // Single student for now
    }
}

// ---------------------------------------------------------------------------
// Heatmap cell color
// ---------------------------------------------------------------------------

fn mastery_color(permille: u16) -> &'static str {
    if permille >= 900 {
        "heatmap-gold"
    } else if permille >= 700 {
        "heatmap-green"
    } else if permille >= 300 {
        "heatmap-yellow"
    } else if permille > 0 {
        "heatmap-red"
    } else {
        "heatmap-empty"
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[component]
fn ClassHeader() -> impl IntoView {
    let profile = use_profile();
    let name = move || {
        let p = profile.get();
        if p.name.is_empty() { "Teacher".to_string() } else { p.name.clone() }
    };
    let grade_label = move || {
        let p = profile.get();
        let g = p.grade;
        if g == 0 || g > 15 {
            "All Grades".to_string()
        } else if g <= 12 {
            format!("Grade {}", g)
        } else {
            "Post-Secondary".to_string()
        }
    };

    // Check Holochain status to determine solo vs connected mode
    let is_solo = move || {
        let status = js_sys::Reflect::get(
            &js_sys::global().unchecked_into::<js_sys::Object>(),
            &wasm_bindgen::JsValue::from_str("__HC_STATUS"),
        ).ok().and_then(|v| v.as_string()).unwrap_or_default();
        status != "connected"
    };

    view! {
        <div class="teacher-class-header">
            <h2>{grade_label}" Mathematics"</h2>
            <div class="class-header-meta">
                <span>{move || if is_solo() { "Solo Mode".to_string() } else { "Connected".to_string() }}</span>
                <span class="meta-sep">"|"</span>
                <span>{move || if is_solo() {
                    "Connect to the mesh to see your class".to_string()
                } else {
                    "0 students".to_string()
                }}</span>
                <span class="meta-sep">"|"</span>
                <span>{name}</span>
            </div>
        </div>
    }
}

#[component]
fn MasteryHeatmap(students: Vec<StudentMastery>) -> impl IntoView {
    view! {
        <div class="dash-section teacher-heatmap-section">
            <h3>"Class Mastery Heatmap"</h3>
            <div class="heatmap-scroll">
                <table class="heatmap-table">
                    <thead>
                        <tr>
                            <th class="heatmap-name-col">"Student"</th>
                            {skill_names().iter().map(|s| {
                                let s = s.clone();
                                view! { <th class="heatmap-skill-col">{s}</th> }
                            }).collect_view()}
                            <th class="heatmap-avg-col">"Avg"</th>
                        </tr>
                    </thead>
                    <tbody>
                        {students.into_iter().map(|student| {
                            let total: u32 = student.skills.iter().map(|s| *s as u32).sum();
                            let avg = if student.skills.is_empty() { 0 } else { (total / student.skills.len() as u32) as u16 };
                            let avg_pct = avg / 10;
                            view! {
                                <tr>
                                    <td class="heatmap-name">{student.name}</td>
                                    {student.skills.iter().map(|&m| {
                                        let pct = m / 10;
                                        let cls = mastery_color(m);
                                        view! {
                                            <td class=format!("heatmap-cell {}", cls)>
                                                {pct}"%"
                                            </td>
                                        }
                                    }).collect_view()}
                                    <td class=format!("heatmap-cell {}", mastery_color(avg))>
                                        {avg_pct}"%"
                                    </td>
                                </tr>
                            }
                        }).collect_view()}
                    </tbody>
                </table>
            </div>
            <div class="heatmap-legend">
                <span class="legend-item"><span class="legend-swatch heatmap-gold"></span>"Mastered (90%+)"</span>
                <span class="legend-item"><span class="legend-swatch heatmap-green"></span>"Proficient (70-89%)"</span>
                <span class="legend-item"><span class="legend-swatch heatmap-yellow"></span>"Developing (30-69%)"</span>
                <span class="legend-item"><span class="legend-swatch heatmap-red"></span>"Beginning (<30%)"</span>
                <span class="legend-item"><span class="legend-swatch heatmap-empty"></span>"Not Started"</span>
            </div>
        </div>
    }
}

#[component]
fn AtRiskPanel(alerts: Vec<AtRiskAlert>) -> impl IntoView {
    view! {
        <div class="dash-card at-risk-panel">
            <h3>"At-Risk Students"</h3>
            {if alerts.is_empty() {
                view! {
                    <p class="at-risk-empty">"No at-risk students detected."</p>
                }.into_any()
            } else {
                view! {
                    <div class="at-risk-list">
                        {alerts.into_iter().map(|alert| {
                            let severity_cls = match alert.severity {
                                AlertSeverity::Critical => "alert-critical",
                                AlertSeverity::Warning => "alert-warning",
                            };
                            let icon = match alert.severity {
                                AlertSeverity::Critical => "!!",
                                AlertSeverity::Warning => "!",
                            };
                            view! {
                                <div class=format!("at-risk-item {}", severity_cls)>
                                    <span class="at-risk-icon">{icon}</span>
                                    <div class="at-risk-content">
                                        <span class="at-risk-student">{alert.student}</span>
                                        <span class="at-risk-reason">{alert.reason}</span>
                                    </div>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
fn ClassStatsCard(stats: ClassStats) -> impl IntoView {
    view! {
        <div class="dash-card class-stats-card">
            <h3>"Class Stats"</h3>
            <div class="class-stats-grid">
                <div class="class-stat">
                    <span class="class-stat-value">{stats.avg_mastery_pct}"%"</span>
                    <span class="class-stat-label">"Avg Mastery"</span>
                </div>
                <div class="class-stat">
                    <span class="class-stat-value stat-on-track">{stats.on_track}"/"{ stats.total}</span>
                    <span class="class-stat-label">"On Track"</span>
                </div>
                <div class="class-stat">
                    <span class="class-stat-value stat-ahead">{stats.ahead}"/"{ stats.total}</span>
                    <span class="class-stat-label">"Ahead"</span>
                </div>
                <div class="class-stat">
                    <span class="class-stat-value stat-behind">{stats.behind}"/"{ stats.total}</span>
                    <span class="class-stat-label">"Behind"</span>
                </div>
            </div>
        </div>
    }
}

#[component]
fn SkillBreakdown(students: Vec<StudentMastery>) -> impl IntoView {
    // Compute per-skill class average
    let num = students.len() as u32;
    let skill_avgs: Vec<u16> = (0..6).map(|i| {
        let total: u32 = students.iter().map(|s| s.skills[i] as u32).sum();
        if num > 0 { (total / num) as u16 } else { 0 }
    }).collect();

    view! {
        <div class="dash-section teacher-skill-breakdown">
            <h3>"Skill Averages"</h3>
            <div class="skill-breakdown-list">
                {skill_full_names().iter().zip(skill_avgs.iter()).map(|(name, &avg)| {
                    let pct = avg / 10;
                    let cls = mastery_color(avg);
                    let name = name.clone();
                    view! {
                        <div class="skill-breakdown-row">
                            <span class="skill-breakdown-name">{name}</span>
                            <div class="skill-node-bar-container">
                                <div
                                    class=format!("skill-node-bar {}", cls)
                                    style=format!("width: {}%", pct)
                                ></div>
                            </div>
                            <span class="skill-breakdown-pct">{pct}"%"</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}

#[component]
fn TeacherActions() -> impl IntoView {
    view! {
        <div class="dash-section teacher-actions">
            <h3>"Actions"</h3>
            <div class="teacher-actions-grid">
                <a href="#" class="btn-primary">"Create Assessment"</a>
                <a href="#" class="btn-secondary">"Assign Curriculum"</a>
                <a href="#" class="btn-secondary">"Generate Report Cards"</a>
                <a href="#" class="btn-secondary">"View Attendance"</a>
            </div>
        </div>
    }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

#[component]
pub fn TeacherDashboardPage() -> impl IntoView {
    let students = real_students();
    let at_risk = real_at_risk();
    let stats = real_class_stats();

    view! {
        <div class="teacher-dashboard">
            <ClassHeader />

            <MasteryHeatmap students=students.clone() />

            <div class="teacher-bottom-grid">
                <AtRiskPanel alerts=at_risk />
                <ClassStatsCard stats=stats />
            </div>

            <SkillBreakdown students=students />

            <TeacherActions />
        </div>
    }
}
