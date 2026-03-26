// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Skill Map page — interactive knowledge graph visualization.
//!
//! Displays the student's progress through a curriculum as a tiered
//! CSS-grid tree with color-coded mastery, locked nodes, and a detail
//! panel for the selected node.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

fn event_target_value(ev: &leptos::ev::Event) -> String {
    ev.target()
        .and_then(|t| t.dyn_into::<web_sys::HtmlSelectElement>().ok())
        .map(|el| el.value())
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct SkillNodeData {
    pub id: &'static str,
    pub name: &'static str,
    pub mastery_permille: u16,
    pub is_locked: bool,
    pub bloom_level: &'static str,
    pub standard_code: &'static str,
    pub prerequisites: Vec<&'static str>,
    pub description: &'static str,
}

impl SkillNodeData {
    fn mastery_pct(&self) -> u16 {
        self.mastery_permille / 10
    }

    fn mastery_class(&self) -> &'static str {
        if self.is_locked {
            "mastery-locked"
        } else if self.mastery_permille >= 900 {
            "mastery-gold"
        } else if self.mastery_permille >= 700 {
            "mastery-green"
        } else if self.mastery_permille >= 300 {
            "mastery-yellow"
        } else {
            "mastery-red"
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct SkillTierData {
    name: &'static str,
    nodes: Vec<SkillNodeData>,
}

// ---------------------------------------------------------------------------
// Mock data — Grade 3 Math skill tree
// ---------------------------------------------------------------------------

fn mock_grade3_math() -> Vec<SkillTierData> {
    vec![
        SkillTierData {
            name: "Foundations",
            nodes: vec![
                SkillNodeData {
                    id: "place-value",
                    name: "Place Value",
                    mastery_permille: 950,
                    is_locked: false,
                    bloom_level: "Remember",
                    standard_code: "CCSS.MATH.3.NBT.A.1",
                    prerequisites: vec![],
                    description: "Understand the value of digits in multi-digit numbers up to 1000.",
                },
                SkillNodeData {
                    id: "number-sense",
                    name: "Number Sense",
                    mastery_permille: 880,
                    is_locked: false,
                    bloom_level: "Understand",
                    standard_code: "CCSS.MATH.3.NBT.A.2",
                    prerequisites: vec![],
                    description: "Compare and order whole numbers. Develop fluency with number relationships.",
                },
            ],
        },
        SkillTierData {
            name: "Operations I",
            nodes: vec![
                SkillNodeData {
                    id: "addition",
                    name: "Addition",
                    mastery_permille: 920,
                    is_locked: false,
                    bloom_level: "Apply",
                    standard_code: "CCSS.MATH.3.NBT.A.2",
                    prerequisites: vec!["Place Value", "Number Sense"],
                    description: "Fluently add within 1000 using strategies and algorithms.",
                },
                SkillNodeData {
                    id: "subtraction",
                    name: "Subtraction",
                    mastery_permille: 850,
                    is_locked: false,
                    bloom_level: "Apply",
                    standard_code: "CCSS.MATH.3.NBT.A.2",
                    prerequisites: vec!["Place Value", "Number Sense"],
                    description: "Fluently subtract within 1000 using strategies and algorithms.",
                },
                SkillNodeData {
                    id: "rounding",
                    name: "Rounding",
                    mastery_permille: 780,
                    is_locked: false,
                    bloom_level: "Apply",
                    standard_code: "CCSS.MATH.3.NBT.A.1",
                    prerequisites: vec!["Place Value"],
                    description: "Round whole numbers to the nearest 10 or 100.",
                },
            ],
        },
        SkillTierData {
            name: "Operations II",
            nodes: vec![
                SkillNodeData {
                    id: "multiplication",
                    name: "Multiplication",
                    mastery_permille: 650,
                    is_locked: false,
                    bloom_level: "Apply",
                    standard_code: "CCSS.MATH.3.OA.A.1",
                    prerequisites: vec!["Addition"],
                    description: "Interpret products of whole numbers. Multiply within 100 using strategies.",
                },
                SkillNodeData {
                    id: "division",
                    name: "Division",
                    mastery_permille: 480,
                    is_locked: false,
                    bloom_level: "Apply",
                    standard_code: "CCSS.MATH.3.OA.A.2",
                    prerequisites: vec!["Subtraction", "Multiplication"],
                    description: "Interpret quotients of whole numbers. Divide within 100.",
                },
                SkillNodeData {
                    id: "patterns",
                    name: "Patterns",
                    mastery_permille: 560,
                    is_locked: false,
                    bloom_level: "Analyze",
                    standard_code: "CCSS.MATH.3.OA.D.9",
                    prerequisites: vec!["Addition", "Multiplication"],
                    description: "Identify arithmetic patterns and explain them using properties of operations.",
                },
            ],
        },
        SkillTierData {
            name: "Applications",
            nodes: vec![
                SkillNodeData {
                    id: "fractions",
                    name: "Fractions",
                    mastery_permille: 320,
                    is_locked: false,
                    bloom_level: "Understand",
                    standard_code: "CCSS.MATH.3.NF.A.1",
                    prerequisites: vec!["Division"],
                    description: "Understand fractions as parts of a whole. Represent on a number line.",
                },
                SkillNodeData {
                    id: "word-problems",
                    name: "Word Problems",
                    mastery_permille: 410,
                    is_locked: false,
                    bloom_level: "Analyze",
                    standard_code: "CCSS.MATH.3.OA.A.3",
                    prerequisites: vec!["Multiplication", "Division"],
                    description: "Solve two-step word problems using the four operations.",
                },
            ],
        },
        SkillTierData {
            name: "Advanced",
            nodes: vec![
                SkillNodeData {
                    id: "measurement",
                    name: "Measurement",
                    mastery_permille: 180,
                    is_locked: false,
                    bloom_level: "Apply",
                    standard_code: "CCSS.MATH.3.MD.A.1",
                    prerequisites: vec!["Multiplication", "Fractions"],
                    description: "Tell time, measure liquid volumes and masses, solve measurement problems.",
                },
                SkillNodeData {
                    id: "geometry",
                    name: "Geometry Basics",
                    mastery_permille: 0,
                    is_locked: true,
                    bloom_level: "Understand",
                    standard_code: "CCSS.MATH.3.G.A.1",
                    prerequisites: vec!["Fractions", "Measurement"],
                    description: "Understand shapes, partition shapes into parts with equal areas.",
                },
                SkillNodeData {
                    id: "data-graphs",
                    name: "Data & Graphs",
                    mastery_permille: 0,
                    is_locked: true,
                    bloom_level: "Analyze",
                    standard_code: "CCSS.MATH.3.MD.B.3",
                    prerequisites: vec!["Word Problems"],
                    description: "Draw and interpret scaled picture and bar graphs.",
                },
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

#[component]
fn FilterBar(
    grade: ReadSignal<&'static str>,
    set_grade: WriteSignal<&'static str>,
    subject: ReadSignal<&'static str>,
    set_subject: WriteSignal<&'static str>,
    view_mode: ReadSignal<&'static str>,
    set_view_mode: WriteSignal<&'static str>,
) -> impl IntoView {
    view! {
        <div class="skillmap-filters">
            <div class="filter-group">
                <label>"Grade"</label>
                <select
                    on:change=move |ev| {
                        let val = event_target_value(&ev);
                        set_grade.set(match val.as_str() {
                            "K" => "K",
                            "1" => "1",
                            "2" => "2",
                            "3" => "3",
                            "4" => "4",
                            "5" => "5",
                            _ => "3",
                        });
                    }
                    prop:value=move || grade.get()
                >
                    <option value="K">"K"</option>
                    <option value="1">"1"</option>
                    <option value="2">"2"</option>
                    <option value="3" selected>"3"</option>
                    <option value="4">"4"</option>
                    <option value="5">"5"</option>
                </select>
            </div>
            <div class="filter-group">
                <label>"Subject"</label>
                <select
                    on:change=move |ev| {
                        let val = event_target_value(&ev);
                        set_subject.set(match val.as_str() {
                            "Math" => "Math",
                            "ELA" => "ELA",
                            "Science" => "Science",
                            _ => "Math",
                        });
                    }
                    prop:value=move || subject.get()
                >
                    <option value="Math" selected>"Math"</option>
                    <option value="ELA">"ELA"</option>
                    <option value="Science">"Science"</option>
                </select>
            </div>
            <div class="filter-group">
                <label>"View"</label>
                <select
                    on:change=move |ev| {
                        let val = event_target_value(&ev);
                        set_view_mode.set(match val.as_str() {
                            "tree" => "tree",
                            "list" => "list",
                            _ => "tree",
                        });
                    }
                    prop:value=move || view_mode.get()
                >
                    <option value="tree" selected>"Tree"</option>
                    <option value="list">"List"</option>
                </select>
            </div>
        </div>
    }
}

#[component]
fn SkillNodeCard(
    node: SkillNodeData,
    selected_id: ReadSignal<Option<&'static str>>,
    set_selected: WriteSignal<Option<&'static str>>,
) -> impl IntoView {
    let node_id = node.id;
    let mastery_pct = node.mastery_pct();
    let mastery_cls = node.mastery_class();
    let is_locked = node.is_locked;
    let name = node.name;

    view! {
        <button
            class=move || {
                let base = format!("skill-node {}", mastery_cls);
                if selected_id.get() == Some(node_id) {
                    format!("{} skill-node-selected", base)
                } else {
                    base
                }
            }
            on:click=move |_| {
                if !is_locked {
                    set_selected.set(Some(node_id));
                }
            }
            disabled=is_locked
        >
            <span class="skill-node-name">
                {if is_locked { "[locked] " } else { "" }}
                {name}
            </span>
            <div class="skill-node-bar-container">
                <div
                    class="skill-node-bar"
                    style=format!("width: {}%", mastery_pct)
                ></div>
            </div>
            <span class="skill-node-pct">{mastery_pct}"%"</span>
        </button>
    }
}

#[component]
fn SkillTierRow(
    tier: SkillTierData,
    selected_id: ReadSignal<Option<&'static str>>,
    set_selected: WriteSignal<Option<&'static str>>,
) -> impl IntoView {
    let tier_name = tier.name;
    view! {
        <div class="skill-tier">
            <div class="skill-tier-label">{tier_name}</div>
            <div class="skill-tier-nodes">
                {tier.nodes.into_iter().map(|node| {
                    view! {
                        <SkillNodeCard
                            node=node
                            selected_id=selected_id
                            set_selected=set_selected
                        />
                    }
                }).collect_view()}
            </div>
            <div class="skill-tier-arrow">"-->"</div>
        </div>
    }
}

#[component]
fn NodeDetail(
    node: SkillNodeData,
) -> impl IntoView {
    let mastery_pct = node.mastery_pct();
    view! {
        <div class="node-detail">
            <div class="node-detail-header">
                <h3>{node.name}</h3>
                <span class="node-detail-meta">
                    "Grade 3"
                    " | Bloom: " {node.bloom_level}
                </span>
            </div>
            <p class="node-detail-desc">{node.description}</p>
            <div class="node-detail-mastery">
                <span class="node-detail-mastery-label">"Mastery"</span>
                <div class="progress-bar-container">
                    <div
                        class="progress-bar"
                        style=format!("width: {}%", mastery_pct)
                    ></div>
                </div>
                <span class="node-detail-mastery-pct">{mastery_pct}"%"</span>
            </div>
            <div class="node-detail-row">
                <span class="node-detail-label">"Standards"</span>
                <span class="node-detail-value">{node.standard_code}</span>
            </div>
            <div class="node-detail-row">
                <span class="node-detail-label">"Prerequisites"</span>
                <span class="node-detail-value">
                    {if node.prerequisites.is_empty() {
                        "None".to_string()
                    } else {
                        node.prerequisites.iter().map(|p| format!("{} (done)", p)).collect::<Vec<_>>().join(", ")
                    }}
                </span>
            </div>
            <div class="node-detail-actions">
                <a href="#" class="btn-primary">"Start Learning"</a>
                <a href="#" class="btn-secondary">"Review"</a>
                <a href="#" class="btn-secondary">"Take Assessment"</a>
            </div>
        </div>
    }
}

#[component]
fn SkillListView(
    tiers: Vec<SkillTierData>,
) -> impl IntoView {
    view! {
        <div class="skill-list-view">
            {tiers.into_iter().map(|tier| {
                let tier_name = tier.name;
                view! {
                    <div class="skill-list-section">
                        <h4 class="skill-list-tier-label">{tier_name}</h4>
                        {tier.nodes.into_iter().map(|node| {
                            let mastery_pct = node.mastery_pct();
                            let cls = node.mastery_class();
                            view! {
                                <div class=format!("skill-list-item {}", cls)>
                                    <span class="skill-list-name">
                                        {if node.is_locked { "[locked] " } else { "" }}
                                        {node.name}
                                    </span>
                                    <span class="skill-list-bloom">{node.bloom_level}</span>
                                    <span class="skill-list-standard">{node.standard_code}</span>
                                    <div class="skill-node-bar-container">
                                        <div
                                            class="skill-node-bar"
                                            style=format!("width: {}%", mastery_pct)
                                        ></div>
                                    </div>
                                    <span class="skill-list-pct">{mastery_pct}"%"</span>
                                </div>
                            }
                        }).collect_view()}
                    </div>
                }
            }).collect_view()}
        </div>
    }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

#[component]
pub fn SkillMapPage() -> impl IntoView {
    let (grade, set_grade) = signal("3");
    let (subject, set_subject) = signal("Math");
    let (view_mode, set_view_mode) = signal("tree");
    let (selected, set_selected) = signal::<Option<&'static str>>(None);

    let tiers = mock_grade3_math();

    // Build a flat lookup for the detail panel
    let all_nodes: Vec<SkillNodeData> = tiers
        .iter()
        .flat_map(|t| t.nodes.iter().cloned())
        .collect();

    view! {
        <div class="skill-map-page">
            <h2>"Skill Map"</h2>
            <p class="subtitle">"Grade 3 Mathematics — Track your progress through the curriculum"</p>

            <FilterBar
                grade=grade set_grade=set_grade
                subject=subject set_subject=set_subject
                view_mode=view_mode set_view_mode=set_view_mode
            />

            {move || {
                if view_mode.get() == "tree" {
                    let tiers_clone = mock_grade3_math();
                    view! {
                        <div class="skill-map-tree">
                            {tiers_clone.into_iter().map(|tier| {
                                view! {
                                    <SkillTierRow
                                        tier=tier
                                        selected_id=selected
                                        set_selected=set_selected
                                    />
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                } else {
                    let tiers_clone = mock_grade3_math();
                    view! {
                        <SkillListView tiers=tiers_clone />
                    }.into_any()
                }
            }}

            {move || {
                let sel_id = selected.get();
                sel_id.and_then(|id| {
                    all_nodes.iter().find(|n| n.id == id).cloned()
                }).map(|node| {
                    view! { <NodeDetail node=node /> }
                })
            }}
        </div>
    }
}
