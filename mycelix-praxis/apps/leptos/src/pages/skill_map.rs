// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Skill Map page — curriculum knowledge graph visualization.
//!
//! Displays the student's progress through the South African CAPS
//! Mathematics and Physical Sciences curriculum as an interactive
//! SVG DAG with mastery tracking and a detail panel.

use leptos::prelude::*;

use crate::curriculum::{
    curriculum_graph, display_subject, use_grade, use_progress, use_set_grade, use_set_progress,
    use_set_subject, use_subject, CurriculumNode, Grade, ProgressStatus, Subject,
};

// ============================================================
// Growth stage helpers — Indlela design: seed → sprout → sapling → tree
// ============================================================

/// Returns (base_radius, growth_class, opacity) for a node based on mastery.
fn growth_stage(status: ProgressStatus) -> (f64, &'static str, &'static str) {
    match status {
        ProgressStatus::NotStarted => (8.0, "growth-seed", "0.45"),
        ProgressStatus::Studying   => (12.0, "growth-sprout", "0.85"),
        ProgressStatus::Mastered   => (15.0, "growth-tree", "1"),
    }
}

/// Community soil richness for a topic (0.0 = sandy, 1.0 = rich).
///
/// When Holochain is connected, this will use aggregate mastery data from the
/// learning_zome across all learners. Until then, uses a stable hash of the
/// node ID for consistent visual warmth per topic. The hash-based simulation
/// approximates what real community data would look like: most topics have
/// moderate engagement (0.4-0.8), with some standouts.
fn community_soil(node_id: &str) -> f64 {
    // TODO: When Holochain conductor is live, call learning_zome::get_community_mastery(node_id)
    // and blend with local data. For now, stable hash gives consistent warmth.
    let hash: u32 = node_id.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    0.4 + (hash % 400) as f64 / 1000.0
}

/// Knowledge decay factor (0.0 = fresh, 1.0 = very stale).
/// Based on time since last review. Returns 0 for unstarted topics.
fn knowledge_decay(last_reviewed: Option<f64>) -> f64 {
    let Some(last) = last_reviewed else { return 0.0 };
    let now = js_sys::Date::now();
    let days_since = (now - last) / (24.0 * 60.0 * 60.0 * 1000.0);
    // Exponential decay: 50% forgotten after 7 days (Ebbinghaus curve approximation)
    (1.0 - (-days_since / 10.0_f64).exp()).min(1.0).max(0.0)
}

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
        let graph = curriculum_graph();
        let s = subject.get().as_str().to_string();
        let g = grade.get().as_str().to_string();
        graph.nodes_for(&s, &g).into_iter().cloned().collect::<Vec<_>>()
    });

    // Recommended next node (FEP pulse target)
    let recommended_id = Memo::new(move |_| {
        let graph = curriculum_graph();
        let p = progress.get();
        let mastered: std::collections::HashSet<String> = p
            .nodes
            .iter()
            .filter(|(_, np)| np.status == ProgressStatus::Mastered)
            .map(|(id, _)| id.clone())
            .collect();
        graph.recommended_next(&mastered).map(|s| s.to_string())
    });

    // Progress summary
    let progress_summary = Memo::new(move |_| {
        let graph = curriculum_graph();
        let p = progress.get();
        let total = graph.nodes.len();
        let mastered = p.mastered_count();
        let studying = p.studying_count();
        (total, mastered, studying)
    });

    view! {
        <div class="praxis-skill-map">
            // Progress summary — organic growth circle
            <div class="praxis-progress-summary indlela-progress">
                {move || {
                    let (total, mastered, studying) = progress_summary.get();
                    let pct = if total > 0 { mastered * 100 / total } else { 0 };
                    // SVG arc for organic progress circle (wabi-sabi: gap = growth space)
                    let r = 28.0_f64;
                    let circumference = 2.0 * std::f64::consts::PI * r;
                    let filled = circumference * (pct as f64 / 100.0);
                    let gap = circumference - filled;
                    // Slight wobble for hand-drawn feel
                    let dash = format!("{:.1} {:.1}", filled, gap);

                    view! {
                        <div style="display: flex; align-items: center; gap: 1.25rem">
                            <svg width="72" height="72" viewBox="0 0 72 72" class="growth-circle">
                                // Background ring (the growth space)
                                <circle cx="36" cy="36" r=r fill="none"
                                    stroke="var(--border)" stroke-width="4" opacity="0.2" />
                                // Progress arc (organic, slightly offset for imperfection)
                                <circle cx="36" cy="36" r=r fill="none"
                                    stroke="var(--mastery-green)" stroke-width="4.5"
                                    stroke-dasharray=dash
                                    stroke-linecap="round"
                                    transform="rotate(-90 36 36)"
                                    class="growth-arc" />
                                // Center text
                                <text x="36" y="34" text-anchor="middle" font-size="14"
                                    font-weight="700" fill="var(--text)" font-family="Inter, sans-serif">
                                    {mastered}
                                </text>
                                <text x="36" y="46" text-anchor="middle" font-size="8"
                                    fill="var(--text-secondary)" font-family="Inter, sans-serif">
                                    "growing"
                                </text>
                            </svg>
                            <div>
                                <div style="font-weight: 700; font-size: 1.1rem">"Knowledge Garden"</div>
                                <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.25rem">
                                    {mastered}" rooted, "{studying}" sprouting, "{total - mastered - studying}" seeds"
                                </div>
                            </div>
                        </div>
                    }
                }}
            </div>

            // Filters
            <div class="praxis-filters">
                // Subject filter — dynamically generated from graph
                <div class="praxis-filter-group" style="overflow-x: auto; flex-wrap: nowrap">
                    {
                        let graph = curriculum_graph();
                        // Show top subjects (those with most nodes)
                        let mut subject_counts: Vec<(String, usize)> = graph.subjects().iter()
                            .map(|s| (s.to_string(), graph.nodes.iter().filter(|n| n.subject_area == *s).count()))
                            .collect();
                        subject_counts.sort_by(|a, b| b.1.cmp(&a.1));
                        subject_counts.into_iter().take(8).map(|(s, count)| {
                            let s_for_check = Subject(s.clone());
                            let s_for_click = Subject(s.clone());
                            let display = display_subject(&s);
                            let label = if display.chars().count() > 20 { format!("{}...", display.chars().take(18).collect::<String>()) } else { display };
                            view! {
                                <button
                                    class=move || if subject.get() == s_for_check { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                    on:click=move |_| set_subject.set(s_for_click.clone())
                                    style="flex-shrink: 0"
                                >{label}" ("{count}")"</button>
                            }
                        }).collect::<Vec<_>>()
                    }
                </div>
                // K-12 grades
                <div class="praxis-filter-group">
                    {Grade::k12().iter().map(|g| {
                        let g = *g;
                        let label = g.label();
                        view! {
                            <button
                                class=move || if grade.get() == g { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                on:click=move |_| set_grade.set(g)
                            >{label}</button>
                        }
                    }).collect::<Vec<_>>()}
                </div>
                // Post-secondary
                <div class="praxis-filter-group" style="border-top: 1px solid var(--border); padding-top: 0.5rem; margin-top: 0.25rem">
                    <span style="font-size: 0.7rem; color: var(--text-tertiary); margin-right: 0.5rem">"Post-secondary:"</span>
                    {Grade::post_secondary().iter().map(|g| {
                        let g = *g;
                        let label = g.label();
                        view! {
                            <button
                                class=move || if grade.get() == g { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                                on:click=move |_| set_grade.set(g)
                            >{label}</button>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>

            // Knowledge Constellation — SVG graph visualization
            {move || {
                let graph = curriculum_graph();
                let nodes = filtered_nodes.get();
                let p = progress.get();
                let node_count = nodes.len();

                if node_count == 0 {
                    return view! { <p style="color: var(--text-secondary); padding: 2rem; text-align: center">"No topics found for this selection."</p> }.into_any();
                }

                // Compute positions: organic 2D constellation layout
                // Nodes arranged in a staggered pattern, spreading across the space
                let cols = if node_count <= 6 { 3 } else if node_count <= 12 { 4 } else { 5 };
                let spacing_x = 160.0;
                let spacing_y = 100.0;
                let margin = 80.0;

                let has_game_fn = crate::games::has_game;
                let rec_id = recommended_id.get();

                let positions: Vec<(f64, f64, String, String, ProgressStatus, Option<u16>, bool, f64, f64, bool)> = nodes.iter().enumerate().map(|(i, n)| {
                    let row = i / cols;
                    let col = i % cols;
                    let x_offset = if row % 2 == 1 { spacing_x * 0.5 } else { 0.0 };
                    let x = margin + col as f64 * spacing_x + x_offset;
                    let y = margin + row as f64 * spacing_y;
                    let np = p.get(&n.id);
                    let status = np.status;
                    let marks = n.exam_weight.as_ref().map(|w| w.marks);
                    let has_game = has_game_fn(&n.id);
                    let soil = community_soil(&n.id);
                    let decay = knowledge_decay(np.last_reviewed);
                    let is_recommended = rec_id.as_deref() == Some(n.id.as_str());
                    (x, y, n.id.clone(), n.title.clone(), status, marks, has_game, soil, decay, is_recommended)
                }).collect();

                let rows = (node_count + cols - 1) / cols;
                let svg_width = margin * 2.0 + (cols as f64) * spacing_x;
                let svg_height = margin * 2.0 + (rows as f64) * spacing_y;
                let view_box = format!("0 0 {} {}", svg_width, svg_height);

                // Edges within the view — prerequisite connections become visible root systems
                let node_ids: std::collections::HashSet<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
                let edges: Vec<(f64, f64, f64, f64, bool)> = graph.edges.iter().filter_map(|e| {
                    if !node_ids.contains(e.from.as_str()) || !node_ids.contains(e.to.as_str()) { return None; }
                    let from_pos = positions.iter().find(|pp| pp.2 == e.from)?;
                    let to_pos = positions.iter().find(|pp| pp.2 == e.to)?;
                    let both_mastered = from_pos.4 == ProgressStatus::Mastered && to_pos.4 == ProgressStatus::Mastered;
                    Some((from_pos.0, from_pos.1, to_pos.0, to_pos.1, both_mastered))
                }).collect();

                view! {
                    <div class="constellation-container">
                        <svg viewBox=view_box xmlns="http://www.w3.org/2000/svg" class="constellation-svg">
                            // Edges — root system connections (thicker when mastered)
                            {edges.iter().map(|(x1, y1, x2, y2, mastered)| {
                                let x1 = *x1; let y1 = *y1; let x2 = *x2; let y2 = *y2;
                                if *mastered {
                                    // Mastered: organic root-like connection
                                    let mx = (x1 + x2) / 2.0 + 8.0;
                                    let my = (y1 + y2) / 2.0 - 5.0;
                                    let d = format!("M {} {} Q {} {} {} {}", x1, y1, mx, my, x2, y2);
                                    view! {
                                        <path d=d fill="none" stroke="var(--mastery-green)" stroke-width="2" opacity="0.35" class="root-edge" />
                                    }.into_any()
                                } else {
                                    view! {
                                        <line x1=x1 y1=y1 x2=x2 y2=y2 stroke="var(--border)" stroke-width="1" opacity="0.12" stroke-dasharray="4 3" />
                                    }.into_any()
                                }
                            }).collect::<Vec<_>>()}

                            // Nodes — organic growth stages (seed → sprout → tree)
                            {positions.iter().map(|(x, y, id, title, status, marks, has_game, soil, decay, is_recommended)| {
                                let x = *x; let y = *y;
                                let soil = *soil;
                                let decay = *decay;
                                let is_recommended = *is_recommended;
                                let id_click = id.clone();
                                let title_short: String = if title.chars().count() > 28 {
                                    title.chars().take(25).collect::<String>() + "..."
                                } else {
                                    title.clone()
                                };

                                let (r, growth_class, opacity) = growth_stage(*status);
                                let marks_text = marks.map(|m| format!("{}m", m)).unwrap_or_default();
                                let has_game = *has_game;

                                // Soil color: rich dark for strong community, sandy for new topics
                                let soil_opacity = format!("{:.2}", soil * 0.15);

                                view! {
                                    <g class=format!("constellation-node {}", growth_class)
                                       style="cursor: pointer"
                                       on:click=move |_| set_selected_id.set(Some(id_click.clone()))>

                                        // Community soil glow (Ubuntu: collective warmth)
                                        <circle cx=x cy=y r={r + 8.0} fill="var(--mastery-green)" opacity=soil_opacity class="soil-glow" />

                                        // Knowledge decay dew (fading knowledge needs refreshing)
                                        {if decay > 0.15 && *status != ProgressStatus::NotStarted {
                                            let dew_opacity = format!("{:.2}", decay * 0.3);
                                            let dew_r = r + 5.0 + decay * 4.0;
                                            view! {
                                                <circle cx=x cy=y r=dew_r fill="none"
                                                    stroke="var(--info)" stroke-width="1.5"
                                                    opacity=dew_opacity stroke-dasharray="3 4"
                                                    class="decay-dew" />
                                            }.into_any()
                                        } else {
                                            view! { <g></g> }.into_any()
                                        }}

                                        // Root lines for mastered nodes (visible anchoring)
                                        {if *status == ProgressStatus::Mastered {
                                            view! {
                                                <line x1=x y1={y + r} x2={x - 6.0} y2={y + r + 12.0}
                                                      stroke="var(--mastery-green)" stroke-width="1.5" opacity="0.3" class="root-line" />
                                                <line x1=x y1={y + r} x2={x + 5.0} y2={y + r + 10.0}
                                                      stroke="var(--mastery-green)" stroke-width="1" opacity="0.25" class="root-line" />
                                            }.into_any()
                                        } else {
                                            view! { <g></g> }.into_any()
                                        }}

                                        // FEP pulse ring — recommended next node
                                        {if is_recommended {
                                            view! {
                                                <circle cx=x cy=y r={r + 6.0} class="fep-pulse-ring" />
                                            }.into_any()
                                        } else {
                                            view! { <g></g> }.into_any()
                                        }}
                                        // Outer ring
                                        <circle cx=x cy=y r={r + 3.0} fill="none" stroke="transparent" stroke-width="2" class="constellation-ring" />
                                        // Main node
                                        <circle cx=x cy=y r=r opacity=opacity class="node-core" />
                                        // Game indicator
                                        {if has_game {
                                            view! { <circle cx={x + r * 0.7} cy={y - r * 0.7} r="4" fill="var(--success)" stroke="var(--bg)" stroke-width="1" /> }.into_any()
                                        } else {
                                            view! { <g></g> }.into_any()
                                        }}
                                        // Title
                                        <text x=x y={y + r + 16.0} text-anchor="middle" font-size="9" fill="var(--text-secondary)" font-family="Inter, sans-serif">
                                            {title_short}
                                        </text>
                                        // Exam marks
                                        {if !marks_text.is_empty() {
                                            view! {
                                                <text x=x y={y + r + 26.0} text-anchor="middle" font-size="7" fill="var(--primary)" font-family="Inter, sans-serif">
                                                    {marks_text}
                                                </text>
                                            }.into_any()
                                        } else {
                                            view! { <g></g> }.into_any()
                                        }}
                                    </g>
                                }
                            }).collect::<Vec<_>>()}
                        </svg>
                    </div>
                }.into_any()
            }}

            // Detail panel
            {move || {
                let sel_id = selected_id.get();
                sel_id.and_then(|id| {
                    let graph = curriculum_graph();
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
    node: CurriculumNode,
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
        let graph = curriculum_graph();
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
        <div class="praxis-detail">
            <div class="praxis-detail-header">
                <div class="praxis-detail-title">{title.clone()}</div>
                <button class="praxis-detail-close" aria-label="Close detail panel" on:click=move |_| on_close()>"\u{00D7}"</button>
            </div>

            // Status buttons (with celebration on Mastered)
            <div class="praxis-status-btns">
                {
                    let id = node_id.clone();
                    let id2 = node_id.clone();
                    let id3 = node_id.clone();
                    let title_for_celebrate = title.clone();
                    let set_celebrating = expect_context::<WriteSignal<bool>>();
                    let set_celebration_topic = expect_context::<WriteSignal<String>>();
                    view! {
                        <button
                            class=move || if status.get() == ProgressStatus::NotStarted { "praxis-status-btn active-not-started" } else { "praxis-status-btn" }
                            on:click={
                                let id = id.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::NotStarted))
                            }
                        >"Not Started"</button>
                        <button
                            class=move || if status.get() == ProgressStatus::Studying { "praxis-status-btn active-studying" } else { "praxis-status-btn" }
                            on:click={
                                let id = id2.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::Studying))
                            }
                        >"Studying"</button>
                        <button
                            class=move || if status.get() == ProgressStatus::Mastered { "praxis-status-btn active-mastered" } else { "praxis-status-btn" }
                            on:click={
                                let id = id3.clone();
                                let t = title_for_celebrate.clone();
                                move |_| {
                                    set_progress.update(|p| p.set_status(&id, ProgressStatus::Mastered));
                                    set_celebrating.set(true);
                                    set_celebration_topic.set(t.clone());
                                    wasm_bindgen_futures::spawn_local(async move {
                                        gloo_timers::future::sleep(std::time::Duration::from_millis(2000)).await;
                                        set_celebrating.set(false);
                                    });
                                }
                            }
                        >"Mastered"</button>
                    }
                }
            </div>

            // Meta badges
            <div class="praxis-detail-meta">
                <span class="praxis-badge praxis-badge-grade">{grade_label.clone()}</span>
                {exam_html.map(|eh| view! { <span class="praxis-badge praxis-badge-exam">{eh}</span> })}
                <span class="praxis-badge praxis-badge-bloom">{bloom.clone()}</span>
                <span class="praxis-badge praxis-badge-hours">{hours}"h estimated"</span>
            </div>

            // Action buttons
            <div style="margin-bottom: 1rem">
                <a
                    href=format!("/study/{}", node_id)
                    style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1.25rem; background: var(--primary); color: var(--text-on-primary); border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 0.85rem"
                >"Start Learning \u{2192}"</a>
            </div>

            // Tabs
            <div class="praxis-tabs">
                <button
                    class=move || if active_tab.get() == "learn" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("learn")
                >"Learn"</button>
                <button
                    class=move || if active_tab.get() == "prereqs" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("prereqs")
                >"Prerequisites"</button>
                <button
                    class=move || if active_tab.get() == "resources" { "praxis-tab active" } else { "praxis-tab" }
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
                <div class="praxis-resources">
                    {resources.iter().map(|r| {
                        let url = r.url.clone();
                        let title = r.title.clone();
                        view! {
                            <a class="praxis-resource-link" href={url} target="_blank" rel="noopener">{title}</a>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>
        </div>
    }
}
