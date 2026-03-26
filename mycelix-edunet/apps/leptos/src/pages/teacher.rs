// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Teacher Dashboard page — class overview with mastery heatmap,
//! at-risk student alerts, aggregate stats, and action buttons.

use leptos::prelude::*;

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

const SKILL_NAMES: [&str; 6] = [
    "PlcVal", "Add", "Sub", "Mult", "Div", "Frac",
];

const SKILL_FULL_NAMES: [&str; 6] = [
    "Place Value", "Addition", "Subtraction",
    "Multiplication", "Division", "Fractions",
];

fn mock_students() -> Vec<StudentMastery> {
    vec![
        StudentMastery { name: "Alice",   skills: vec![950, 920, 880, 780, 420, 310] },
        StudentMastery { name: "Bob",     skills: vec![900, 850, 280, 150,   0,   0] },
        StudentMastery { name: "Carol",   skills: vec![980, 960, 950, 890, 820, 450] },
        StudentMastery { name: "Dan",     skills: vec![750, 600, 350, 200,   0,   0] },
        StudentMastery { name: "Elena",   skills: vec![920, 900, 870, 750, 580, 280] },
        StudentMastery { name: "Frank",   skills: vec![880, 860, 810, 690, 500, 350] },
        StudentMastery { name: "Grace",   skills: vec![960, 940, 910, 850, 700, 520] },
        StudentMastery { name: "Henry",   skills: vec![840, 780, 720, 550, 300, 100] },
    ]
}

fn mock_at_risk() -> Vec<AtRiskAlert> {
    vec![
        AtRiskAlert {
            student: "Bob",
            reason: "Struggling with Subtraction (28%) — below grade-level expectations",
            severity: AlertSeverity::Critical,
        },
        AtRiskAlert {
            student: "Dan",
            reason: "Low engagement — only 2 sessions this week, declining mastery trend",
            severity: AlertSeverity::Warning,
        },
    ]
}

fn mock_class_stats() -> ClassStats {
    ClassStats {
        avg_mastery_pct: 68,
        on_track: 18,
        ahead: 4,
        behind: 2,
        total: 24,
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
    view! {
        <div class="teacher-class-header">
            <h2>"Grade 3 Mathematics"</h2>
            <div class="class-header-meta">
                <span>"24 students"</span>
                <span class="meta-sep">"|"</span>
                <span>"Fall 2025"</span>
                <span class="meta-sep">"|"</span>
                <span>"Mrs. Rodriguez"</span>
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
                            {SKILL_NAMES.iter().map(|s| {
                                view! { <th class="heatmap-skill-col">{*s}</th> }
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
                {SKILL_FULL_NAMES.iter().zip(skill_avgs.iter()).map(|(name, &avg)| {
                    let pct = avg / 10;
                    let cls = mastery_color(avg);
                    view! {
                        <div class="skill-breakdown-row">
                            <span class="skill-breakdown-name">{*name}</span>
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
    let students = mock_students();
    let at_risk = mock_at_risk();
    let stats = mock_class_stats();

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
