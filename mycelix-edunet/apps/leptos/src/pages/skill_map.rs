// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Skill Map page — CAPS curriculum graph visualization.
//!
//! Displays the student's progress through the South African CAPS
//! Mathematics and Physical Sciences curriculum as an interactive
//! SVG DAG with mastery tracking and a detail panel.

use leptos::prelude::*;

use crate::curriculum::{
    caps_graph, use_grade, use_progress, use_set_grade, use_set_progress, use_set_subject,
    use_subject, CapsNode, Grade, ProgressStatus, Subject,
};

// ============================================================
// Page component
// ============================================================

#[component]
pub fn SkillMapPage() -> impl IntoView {
    let subject = use_subject();
    let set_subject = use_set_subject();
    let grade = use_grade();
    let set_grade = use_set_grade();
    let progress = use_progress();
    let (selected_id, set_selected_id) = signal::<Option<String>>(None);

    // Derived: filtered nodes for current subject + grade
    let filtered_nodes = Memo::new(move |_| {
        let graph = caps_graph();
        let s = subject.get().as_str().to_string();
        let g = grade.get().as_str().to_string();
        graph.nodes_for(&s, &g).into_iter().cloned().collect::<Vec<_>>()
    });

    // Derived: edges within the filtered set
    let filtered_edges = Memo::new(move |_| {
        let graph = caps_graph();
        let nodes = filtered_nodes.get();
        let node_ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
        graph.edges.iter()
            .filter(|e| node_ids.contains(e.to.as_str()) || node_ids.contains(e.from.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    });

    // Progress summary
    let progress_summary = Memo::new(move |_| {
        let graph = caps_graph();
        let p = progress.get();
        let total = graph.nodes.len();
        let mastered = p.mastered_count();
        let studying = p.studying_count();
        (total, mastered, studying)
    });

    view! {
        <div class="caps-skill-map">
            // Progress summary
            <div class="caps-progress-summary">
                {move || {
                    let (total, mastered, studying) = progress_summary.get();
                    let pct = if total > 0 { mastered * 100 / total } else { 0 };
                    view! {
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem">
                            <span style="font-weight: 700; font-size: 1.1rem">"CAPS Curriculum"</span>
                            <span style="font-size: 0.85rem; color: var(--text-secondary)">
                                {mastered}"/" {total}" mastered ("{pct}"%)"
                            </span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-bar-fill success"
                                 style=format!("width: {}%", pct)>
                            </div>
                        </div>
                    }
                }}
            </div>

            // Filters
            <div class="caps-filters">
                <div class="caps-filter-group">
                    <button
                        class=move || if subject.get() == Subject::Mathematics { "caps-filter-btn active" } else { "caps-filter-btn" }
                        on:click=move |_| set_subject.set(Subject::Mathematics)
                    >"Mathematics"</button>
                    <button
                        class=move || if subject.get() == Subject::PhysicalSciences { "caps-filter-btn active" } else { "caps-filter-btn" }
                        on:click=move |_| set_subject.set(Subject::PhysicalSciences)
                    >"Physical Sciences"</button>
                </div>
                <div class="caps-filter-group">
                    <button
                        class=move || if grade.get() == Grade::Gr10 { "caps-filter-btn active" } else { "caps-filter-btn" }
                        on:click=move |_| set_grade.set(Grade::Gr10)
                    >"Grade 10"</button>
                    <button
                        class=move || if grade.get() == Grade::Gr11 { "caps-filter-btn active" } else { "caps-filter-btn" }
                        on:click=move |_| set_grade.set(Grade::Gr11)
                    >"Grade 11"</button>
                    <button
                        class=move || if grade.get() == Grade::Gr12 { "caps-filter-btn active" } else { "caps-filter-btn" }
                        on:click=move |_| set_grade.set(Grade::Gr12)
                    >"Grade 12"</button>
                </div>
            </div>

            // Node grid (simpler than SVG for initial version — cards in a responsive grid)
            <div class="feature-grid">
                <For
                    each=move || filtered_nodes.get()
                    key=|n| n.id.clone()
                    children=move |node: CapsNode| {
                        let id = node.id.clone();
                        let id_for_click = id.clone();
                        let title = node.title.clone();
                        let subdomain = node.subdomain.clone();
                        let bloom = node.bloom_level.clone();
                        let hours = node.estimated_hours;
                        let exam_weight = node.exam_weight.clone();

                        let mastery_class = {
                            let id = id.clone();
                            move || {
                                let p = progress.get();
                                let np = p.get(&id);
                                match np.status {
                                    ProgressStatus::Mastered => "mastery-gold",
                                    ProgressStatus::Studying => "mastery-yellow",
                                    ProgressStatus::NotStarted => if np.mastery_permille >= 700 { "mastery-green" } else { "" },
                                }
                            }
                        };

                        let is_selected = {
                            let id = id.clone();
                            move || selected_id.get().as_deref() == Some(&id)
                        };

                        view! {
                            <button
                                class=move || {
                                    let mut cls = format!("feature-card caps-node-card {}", mastery_class());
                                    if is_selected() { cls.push_str(" selected"); }
                                    cls
                                }
                                on:click=move |_| set_selected_id.set(Some(id_for_click.clone()))
                                style="text-align: left; cursor: pointer; font-family: inherit; width: 100%"
                            >
                                <h3 style="font-size: 0.9rem; margin-bottom: 0.25rem">{title.clone()}</h3>
                                <p style="font-size: 0.75rem; margin-bottom: 0.5rem">{subdomain.clone()}</p>
                                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap">
                                    <span class="caps-badge caps-badge-bloom">{bloom.clone()}</span>
                                    <span class="caps-badge caps-badge-hours">{hours}"h"</span>
                                    {exam_weight.map(|ew| view! {
                                        <span class="caps-badge caps-badge-exam">
                                            "P"{ew.paper}": "{ew.marks}"m"
                                        </span>
                                    })}
                                </div>
                            </button>
                        }
                    }
                />
            </div>

            // Detail panel
            {move || {
                let sel_id = selected_id.get();
                sel_id.and_then(|id| {
                    let graph = caps_graph();
                    graph.node(&id).cloned()
                }).map(|node| {
                    view! {
                        <NodeDetail
                            node=node
                            on_close=move || set_selected_id.set(None)
                        />
                    }
                })
            }}
        </div>
    }
}

// ============================================================
// Node detail panel
// ============================================================

#[component]
fn NodeDetail(
    node: CapsNode,
    on_close: impl Fn() + 'static,
) -> impl IntoView {
    let progress = use_progress();
    let set_progress = use_set_progress();
    let node_id = node.id.clone();
    let (active_tab, set_active_tab) = signal("learn");

    // Current status
    let status = {
        let id = node_id.clone();
        Memo::new(move |_| {
            progress.get().get(&id).status
        })
    };

    // Prerequisites
    let prereqs = {
        let graph = caps_graph();
        graph.prereqs_for(&node_id).iter().filter_map(|pid| {
            graph.node(pid).map(|n| (n.id.clone(), n.title.clone()))
        }).collect::<Vec<_>>()
    };

    let grade_label = node.grade_levels.first().cloned().unwrap_or_default().replace("Grade", "Grade ");
    let exam_html = node.exam_weight.as_ref().map(|ew| {
        format!("Paper {}: {}/{} marks ({:.1}%)", ew.paper, ew.marks, ew.total_paper_marks, ew.percentage)
    });

    let resources = node.supplementary_resources.clone();
    let description = node.description.clone();
    let title = node.title.clone();
    let subdomain = node.subdomain.clone();
    let bloom = node.bloom_level.clone();
    let hours = node.estimated_hours;

    view! {
        <div class="caps-detail">
            <div class="caps-detail-header">
                <div class="caps-detail-title">{title.clone()}</div>
                <button class="caps-detail-close" aria-label="Close detail panel" on:click=move |_| on_close()>"\u{00D7}"</button>
            </div>

            // Status buttons
            <div class="caps-status-btns">
                {
                    let id = node_id.clone();
                    let id2 = node_id.clone();
                    let id3 = node_id.clone();
                    view! {
                        <button
                            class=move || if status.get() == ProgressStatus::NotStarted { "caps-status-btn active-not-started" } else { "caps-status-btn" }
                            on:click={
                                let id = id.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::NotStarted))
                            }
                        >"Not Started"</button>
                        <button
                            class=move || if status.get() == ProgressStatus::Studying { "caps-status-btn active-studying" } else { "caps-status-btn" }
                            on:click={
                                let id = id2.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::Studying))
                            }
                        >"Studying"</button>
                        <button
                            class=move || if status.get() == ProgressStatus::Mastered { "caps-status-btn active-mastered" } else { "caps-status-btn" }
                            on:click={
                                let id = id3.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::Mastered))
                            }
                        >"Mastered"</button>
                    }
                }
            </div>

            // Meta badges
            <div class="caps-detail-meta">
                <span class="caps-badge caps-badge-grade">{grade_label.clone()}</span>
                {exam_html.map(|eh| view! { <span class="caps-badge caps-badge-exam">{eh}</span> })}
                <span class="caps-badge caps-badge-bloom">{bloom.clone()}</span>
                <span class="caps-badge caps-badge-hours">{hours}"h estimated"</span>
            </div>

            // Action buttons
            <div style="margin-bottom: 1rem">
                <a
                    href=format!("/study/{}", node_id)
                    style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.25rem; background: var(--primary); color: var(--text-on-primary); border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 0.85rem"
                >"Start Learning \u{2192}"</a>
            </div>

            // Tabs
            <div class="caps-tabs">
                <button
                    class=move || if active_tab.get() == "learn" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("learn")
                >"Learn"</button>
                <button
                    class=move || if active_tab.get() == "prereqs" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("prereqs")
                >"Prerequisites"</button>
                <button
                    class=move || if active_tab.get() == "resources" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("resources")
                >"Resources"</button>
            </div>

            // Tab content
            <div style=move || if active_tab.get() == "learn" { "display: block" } else { "display: none" }>
                <div class="section">
                    <h4 style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem">"Description"</h4>
                    <p style="font-size: 0.9rem; line-height: 1.7">{description.clone()}</p>
                </div>
            </div>

            <div style=move || if active_tab.get() == "prereqs" { "display: block" } else { "display: none" }>
                {if prereqs.is_empty() {
                    view! { <p style="color: var(--text-secondary); font-size: 0.9rem">"No prerequisites (entry point)"</p> }.into_any()
                } else {
                    view! {
                        <ul style="list-style: none; padding: 0">
                            {prereqs.iter().map(|(id, ptitle)| {
                                let ptitle = ptitle.clone();
                                let pid = id.clone();
                                view! {
                                    <li style="padding: 0.3rem 0; font-size: 0.9rem">
                                        <span style="color: var(--primary); margin-right: 0.5rem">"\u{2192}"</span>
                                        {ptitle}
                                        <span style="color: var(--text-tertiary); font-size: 0.75rem; margin-left: 0.5rem">"("{pid}")"</span>
                                    </li>
                                }
                            }).collect::<Vec<_>>()}
                        </ul>
                    }.into_any()
                }}
            </div>

            <div style=move || if active_tab.get() == "resources" { "display: block" } else { "display: none" }>
                <div class="caps-resources">
                    {resources.iter().map(|r| {
                        let url = r.url.clone();
                        let title = r.title.clone();
                        view! {
                            <a class="caps-resource-link" href={url} target="_blank" rel="noopener">{title}</a>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>
        </div>
    }
}
