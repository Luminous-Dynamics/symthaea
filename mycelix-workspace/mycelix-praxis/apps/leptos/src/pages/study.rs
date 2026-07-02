// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Study Page — renders lesson content for a specific CAPS topic.
//!
//! Fetches the generated lesson JSON on demand and displays:
//! - Explanation with key vocabulary
//! - Worked examples (step-by-step, expandable)
//! - Practice problems (with hints and distractors)
//! - Misconceptions
//! - Supplementary resource links

use leptos::prelude::*;
use serde::Deserialize;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::curriculum::{curriculum_graph, use_progress, use_set_progress, ProgressStatus, BktState};
use crate::games;
use crate::social_proof;
use crate::persistence;

// ============================================================
// Lesson data types (from generated JSON)
// ============================================================

#[derive(Clone, Debug, Deserialize)]
struct NodeContent {
    node_id: String,
    lesson: GeneratedLesson,
    #[serde(default)]
    flashcards: Vec<Flashcard>,
}

#[derive(Clone, Debug, Deserialize)]
struct GeneratedLesson {
    #[serde(default)]
    title: String,
    #[serde(default)]
    explanation: String,
    #[serde(default)]
    examples: Vec<WorkedExample>,
    #[serde(default)]
    practice_problems: Vec<PracticeProblem>,
    #[serde(default)]
    key_vocabulary: Vec<VocabTerm>,
    #[serde(default)]
    misconceptions: Vec<Misconception>,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkedExample {
    problem: String,
    #[serde(default)]
    steps: Vec<SolutionStep>,
    answer: String,
}

#[derive(Clone, Debug, Deserialize)]
struct SolutionStep {
    instruction: String,
    result: String,
}

#[derive(Clone, Debug, Deserialize)]
struct PracticeProblem {
    question: String,
    answer: String,
    #[serde(default)]
    difficulty_permille: u16,
    #[serde(default)]
    bloom_level: String,
    #[serde(default)]
    hints: Vec<String>,
    #[serde(default)]
    explanation: String,
    #[serde(default)]
    distractors: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct VocabTerm {
    term: String,
    definition: String,
    #[serde(default)]
    example_usage: String,
}

#[derive(Clone, Debug, Deserialize)]
struct Misconception {
    misconception: String,
    correction: String,
    #[serde(default)]
    why_students_think_this: String,
}

#[derive(Clone, Debug, Deserialize)]
struct Flashcard {
    front: String,
    back: String,
}

// ============================================================
// Fetch lesson JSON
// ============================================================

async fn fetch_lesson(node_id: &str) -> Result<NodeContent, String> {
    let id_lower = node_id.to_lowercase();
    // Sanitize for filename: replace / with _ for post-Gr12 IDs like "AL/BasicAnalysis"
    let filename = format!("{}.json", id_lower.replace('/', "_"));

    // Determine subject-grade dir
    let graph = curriculum_graph();
    let node_meta = graph.node(node_id);
    let grade_level = node_meta.and_then(|n| n.grade_levels.first().cloned());

    let url = match grade_level.as_deref() {
        Some("Undergraduate") | Some("Graduate") | Some("Doctoral") | Some("Adult") => {
            // Post-Gr12: use subject-based directory
            let subject_dir = node_meta
                .map(|n| n.subject_area.to_lowercase().replace(' ', "-").replace('&', "and"))
                .unwrap_or_else(|| "general".into());
            format!("/curriculum/generated/{}/{}", subject_dir, filename)
        }
        _ => {
            // K-12: existing logic
            let grade_num = (1..=12).rev().find(|g| id_lower.contains(&format!("gr{}", g))).unwrap_or(12);
            let subject_dir = if id_lower.contains("naturalsciences") {
                format!("natsci-{}", grade_num)
            } else if id_lower.contains("physicalsciences") {
                format!("physics-{}", grade_num)
            } else {
                format!("math-{}", grade_num)
            };
            format!("/curriculum/generated/{}/{}", subject_dir, filename)
        }
    };

    let window = web_sys::window().ok_or("no window")?;
    let resp = JsFuture::from(window.fetch_with_str(&url))
        .await
        .map_err(|e| format!("fetch error: {:?}", e))?;

    let resp: web_sys::Response = resp.dyn_into().map_err(|_| "not a Response")?;

    if !resp.ok() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let text = JsFuture::from(resp.text().map_err(|_| "text() failed")?)
        .await
        .map_err(|e| format!("text error: {:?}", e))?;

    let text_str = text.as_string().ok_or("not a string")?;
    serde_json::from_str(&text_str).map_err(|e| format!("JSON parse error: {}", e))
}

// ============================================================
// Study page component
// ============================================================

#[component]
pub fn StudyPage(node_id: String) -> impl IntoView {
    let progress = use_progress();
    let set_progress = use_set_progress();
    let graph = curriculum_graph();

    let node = graph.node(&node_id).cloned();
    let (active_tab, set_active_tab) = signal("explanation");

    // Fetch lesson data
    let node_id_for_fetch = node_id.clone();
    let lesson_resource = LocalResource::new(move || {
        let id = node_id_for_fetch.clone();
        async move { fetch_lesson(&id).await }
    });

    let Some(node) = node else {
        return view! {
            <div class="stub-page">
                <h2>"Topic not found"</h2>
                <p>"The topic "{node_id}" was not found in the curriculum."</p>
            </div>
        }.into_any();
    };

    let title = node.title.clone();
    let description = node.description.clone();
    let subdomain = node.subdomain.clone();
    let grade_label = node.grade_levels.first().cloned().unwrap_or_default().replace("Grade", "Grade ");
    let bloom = node.bloom_level.clone();
    let hours = node.estimated_hours;
    let exam_weight = node.exam_weight.clone();
    let node_id_for_status = node_id.clone();
    let node_id_for_status2 = node_id.clone();

    view! {
        <div class="praxis-skill-map study-refuge">
            // Pomodoro timer (ambient, at top)
            <crate::study_tracker::PomodoroTimer />

            // Back link — return to prospect
            <a href="/skill-map" class="refuge-back-link">
                "\u{2190} Back to constellation"
            </a>

            // Header
            <div style="margin: 1rem 0">
                <h1 style="font-size: 1.5rem; margin-bottom: 0.5rem">{title}</h1>
                <div class="praxis-detail-meta">
                    <span class="praxis-badge praxis-badge-grade">{grade_label}</span>
                    <span class="praxis-badge praxis-badge-bloom">{bloom}</span>
                    <span class="praxis-badge praxis-badge-hours">{hours}"h estimated"</span>
                    {exam_weight.map(|ew| view! {
                        <span class="praxis-badge praxis-badge-exam">
                            "Paper "{ew.paper}": "{ew.marks}"/" {ew.total_paper_marks}" marks ("{format!("{:.0}", ew.percentage)}"%)"
                        </span>
                    })}
                </div>

                // NSC exam insight
                {social_proof::exam_insight(&node_id).map(|insight| {
                    let pct = insight.percentage_struggled;
                    let error = insight.common_error;
                    let tip = insight.examiner_tip;
                    view! {
                        <div style="margin-top: 0.75rem; padding: 0.6rem 0.75rem; border-left: 3px solid var(--info); background: rgba(59, 130, 246, 0.05); border-radius: 0 6px 6px 0; font-size: 0.8rem; line-height: 1.5">
                            <span style="color: var(--info); font-weight: 600">"Exam insight: "</span>
                            <span style="color: var(--text-secondary)">{pct}"% of candidates found this challenging. "</span>
                            <span style="color: var(--text-tertiary); font-style: italic">{tip}</span>
                        </div>
                    }
                })}
            </div>

            // Status
            <div class="praxis-status-btns">
                {
                    let id_ns = node_id_for_status.clone();
                    let id_s = node_id_for_status.clone();
                    let id_s2 = id_s.clone();
                    let id_m = node_id_for_status2.clone();
                    let id_m2 = id_m.clone();
                    view! {
                        <button
                            class=move || {
                                let s = progress.get().get(&id_ns).status;
                                if s == ProgressStatus::NotStarted { "praxis-status-btn active-not-started" } else { "praxis-status-btn" }
                            }
                            on:click={
                                let id = node_id.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::NotStarted))
                            }
                        >"Not Started"</button>
                        <button
                            class=move || {
                                let s = progress.get().get(&id_s).status;
                                if s == ProgressStatus::Studying { "praxis-status-btn active-studying" } else { "praxis-status-btn" }
                            }
                            on:click={
                                let id = id_s2.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::Studying))
                            }
                        >"Studying"</button>
                        <button
                            class=move || {
                                let s = progress.get().get(&id_m).status;
                                if s == ProgressStatus::Mastered { "praxis-status-btn active-mastered" } else { "praxis-status-btn" }
                            }
                            on:click={
                                let id = id_m2.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::Mastered))
                            }
                        >"Mastered"</button>
                    }
                }
            </div>

            // Tabs
            <div class="praxis-tabs" style="max-width: 600px">
                <button class=move || if active_tab.get() == "explanation" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("explanation")>"Learn"</button>
                <button class=move || if active_tab.get() == "examples" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("examples")>"Examples"</button>
                <button class=move || if active_tab.get() == "practice" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("practice")>"Practice"</button>
                <button class=move || if active_tab.get() == "pitfalls" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("pitfalls")>"Pitfalls"</button>
                <button class=move || if active_tab.get() == "connections" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("connections")
                    style="color: var(--info)"
                >"Connections"</button>
                <button class=move || if active_tab.get() == "notes" { "praxis-tab active" } else { "praxis-tab" }
                    on:click=move |_| set_active_tab.set("notes")>"Notes"</button>
                {if games::has_game(&node_id_for_status) {
                    view! {
                        <button class=move || if active_tab.get() == "explore" { "praxis-tab active" } else { "praxis-tab" }
                            on:click=move |_| set_active_tab.set("explore")
                            style="color: var(--success)"
                        >"Explore"</button>
                    }.into_any()
                } else {
                    view! { <span></span> }.into_any()
                }}
            </div>

            // Content
            <Suspense fallback=move || view! {
                <div class="card-loading">
                    <div class="skeleton-line" style="width: 80%"></div>
                    <div class="skeleton-line" style="width: 60%"></div>
                    <div class="skeleton-line" style="width: 90%"></div>
                </div>
            }>
                {move || {
                    lesson_resource.get().map(|result| {
                        match &result {
                            Ok(content) => {
                                let lesson = content.lesson.clone();
                                let explanation = lesson.explanation.clone();
                                let vocab = lesson.key_vocabulary.clone();
                                let examples = lesson.examples.clone();
                                let problems = lesson.practice_problems.clone();
                                let misconceptions = lesson.misconceptions.clone();

                                view! {
                                    // Explanation tab
                                    <div style=move || if active_tab.get() == "explanation" { "display: block" } else { "display: none" }>
                                        <div class="praxis-detail" style="margin-bottom: 1rem">
                                            <p style="font-size: 0.95rem; line-height: 1.8">{explanation.clone()}</p>
                                        </div>

                                        {if !vocab.is_empty() {
                                            view! {
                                                <div class="praxis-detail">
                                                    <h3 style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.75rem; color: var(--text-secondary)">"Key Vocabulary"</h3>
                                                    {vocab.iter().map(|v| {
                                                        let term = v.term.clone();
                                                        let def = v.definition.clone();
                                                        view! {
                                                            <div style="margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border)">
                                                                <strong>{term}</strong>
                                                                <span style="color: var(--text-secondary)">" — "{def}</span>
                                                            </div>
                                                        }
                                                    }).collect::<Vec<_>>()}
                                                </div>
                                            }.into_any()
                                        } else {
                                            view! { <span></span> }.into_any()
                                        }}
                                    </div>

                                    // Examples tab
                                    <div style=move || if active_tab.get() == "examples" { "display: block" } else { "display: none" }>
                                        {examples.iter().enumerate().map(|(i, ex)| {
                                            let problem = ex.problem.clone();
                                            let answer = ex.answer.clone();
                                            let steps = ex.steps.clone();
                                            view! {
                                                <div class="praxis-example">
                                                    <div class="praxis-example-problem">"Example "{i + 1}": "{problem}</div>
                                                    {steps.iter().map(|s| {
                                                        let instr = s.instruction.clone();
                                                        let result = s.result.clone();
                                                        view! {
                                                            <div class="praxis-example-step">
                                                                <strong>{instr}</strong>" \u{2192} "{result}
                                                            </div>
                                                        }
                                                    }).collect::<Vec<_>>()}
                                                    <div class="praxis-example-answer">{answer}</div>
                                                </div>
                                            }
                                        }).collect::<Vec<_>>()}
                                    </div>

                                    // Practice tab — BKT-adaptive active recall
                                    <div style=move || if active_tab.get() == "practice" { "display: block" } else { "display: none" }>
                                        // BKT mastery indicator
                                        {
                                            let nid = node_id_for_status.clone();
                                            move || {
                                                let p = progress.get();
                                                let bkt = p.bkt(&nid);
                                                let pct = (bkt.p_mastery * 100.0) as u32;
                                                let terrain = bkt.terrain_label();
                                                view! {
                                                    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding: 0.5rem 0.75rem; background: var(--soil-rich); border-radius: 8px">
                                                        <div style="font-size: 0.8rem; color: var(--text-secondary)">
                                                            "Understanding: "
                                                            <strong style="color: var(--text)">{pct}"%"</strong>
                                                        </div>
                                                        <div style="font-size: 0.75rem; color: var(--text-tertiary); font-style: italic">
                                                            "Recommended: "{terrain}
                                                        </div>
                                                    </div>
                                                }
                                            }
                                        }
                                        <p style="font-size: 0.8rem; color: var(--text-tertiary); margin-bottom: 1rem">"Try to solve each problem before revealing the answer. Your responses shape the difficulty."</p>
                                        {problems.iter().enumerate().map(|(i, p)| {
                                            let question = p.question.clone();
                                            let answer = p.answer.clone();
                                            let diff = match p.difficulty_permille {
                                                0..=300 => "Gentle soil",
                                                301..=600 => "Steady ground",
                                                601..=800 => "Rocky terrain",
                                                _ => "Mountain path",
                                            };
                                            let bloom_level = p.bloom_level.clone();
                                            let explanation = p.explanation.clone();
                                            let hints = p.hints.clone();
                                            let (revealed, set_revealed) = signal(false);
                                            let (show_hint, set_show_hint) = signal(false);
                                            let (evaluated, set_evaluated) = signal(false);
                                            let nid_correct = node_id_for_status.clone();
                                            let nid_wrong = node_id_for_status.clone();
                                            view! {
                                                <div class="praxis-problem" style="cursor: pointer">
                                                    <div class="praxis-problem-q">{i + 1}". "{question}</div>
                                                    <div class="praxis-problem-meta">{diff}" | "{bloom_level}</div>
                                                    {move || if revealed.get() {
                                                        view! {
                                                            <div class="praxis-problem-a" style="margin-top: 0.5rem">
                                                                "Answer: "{answer.clone()}
                                                            </div>
                                                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem">
                                                                {explanation.clone()}
                                                            </div>
                                                            // Self-evaluation — feeds BKT
                                                            {
                                                                let nid_c = nid_correct.clone();
                                                                let nid_w = nid_wrong.clone();
                                                                move || if !evaluated.get() {
                                                                    let nc = nid_c.clone();
                                                                    let nw = nid_w.clone();
                                                                    view! {
                                                                        <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem">
                                                                            <button
                                                                                class="praxis-filter-btn"
                                                                                style="font-size: 0.75rem; border-color: var(--mastery-green); color: var(--mastery-green)"
                                                                                on:click=move |_| {
                                                                                    set_progress.update(|p| p.record_response(&nc, true));
                                                                                    set_evaluated.set(true);
                                                                                }
                                                                            >"I got this right"</button>
                                                                            <button
                                                                                class="praxis-filter-btn"
                                                                                style="font-size: 0.75rem; opacity: 0.7"
                                                                                on:click=move |_| {
                                                                                    set_progress.update(|p| p.record_response(&nw, false));
                                                                                    set_evaluated.set(true);
                                                                                }
                                                                            >"Still learning"</button>
                                                                        </div>
                                                                    }.into_any()
                                                                } else {
                                                                    view! {
                                                                        <div style="font-size: 0.75rem; color: var(--text-tertiary); margin-top: 0.5rem; font-style: italic">
                                                                            "Recorded — your path is adapting"
                                                                        </div>
                                                                    }.into_any()
                                                                }
                                                            }
                                                        }.into_any()
                                                    } else {
                                                        let hints_available = !hints.is_empty();
                                                        let hint_text = hints.first().cloned().unwrap_or_default();
                                                        view! {
                                                            <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem">
                                                                <button
                                                                    class="praxis-filter-btn active"
                                                                    style="font-size: 0.75rem"
                                                                    on:click=move |_| set_revealed.set(true)
                                                                >"Reveal Answer"</button>
                                                                {if hints_available {
                                                                    view! {
                                                                        <button
                                                                            class="praxis-filter-btn"
                                                                            style="font-size: 0.75rem"
                                                                            on:click=move |_| set_show_hint.set(true)
                                                                        >"Hint"</button>
                                                                    }.into_any()
                                                                } else {
                                                                    view! { <span></span> }.into_any()
                                                                }}
                                                            </div>
                                                            {move || if show_hint.get() {
                                                                view! {
                                                                    <div style="font-size: 0.8rem; color: var(--warning); margin-top: 0.25rem; font-style: italic">
                                                                        {hint_text.clone()}
                                                                    </div>
                                                                }.into_any()
                                                            } else {
                                                                view! { <span></span> }.into_any()
                                                            }}
                                                        }.into_any()
                                                    }}
                                                </div>
                                            }
                                        }).collect::<Vec<_>>()}
                                    </div>

                                    // Pitfalls tab
                                    <div style=move || if active_tab.get() == "pitfalls" { "display: block" } else { "display: none" }>
                                        {if misconceptions.is_empty() {
                                            view! { <p style="color: var(--text-secondary)">"No common misconceptions documented for this topic."</p> }.into_any()
                                        } else {
                                            view! {
                                                {misconceptions.iter().map(|m| {
                                                    let mc = m.misconception.clone();
                                                    let corr = m.correction.clone();
                                                    let why = m.why_students_think_this.clone();
                                                    view! {
                                                        <div class="praxis-misconception">
                                                            <div class="praxis-misconception-wrong">{mc}</div>
                                                            <div class="praxis-misconception-right">{corr}</div>
                                                            {if !why.is_empty() {
                                                                view! { <div class="praxis-misconception-why">{why}</div> }.into_any()
                                                            } else {
                                                                view! { <span></span> }.into_any()
                                                            }}
                                                        </div>
                                                    }
                                                }).collect::<Vec<_>>()}
                                            }.into_any()
                                        }}
                                    </div>

                                    // Explore tab (game)
                                    <div style=move || if active_tab.get() == "explore" { "display: block" } else { "display: none" }>
                                        <games::GameContainer node_id=node_id.clone() />
                                    </div>

                                    // Notes tab
                                    // Connections tab — cross-subject links
                                    <div style=move || if active_tab.get() == "connections" { "display: block" } else { "display: none" }>
                                        {
                                            let graph = curriculum_graph();
                                            let neighbors = graph.cross_subject_neighbors(&node_id);
                                            if neighbors.is_empty() {
                                                view! {
                                                    <div class="praxis-detail">
                                                        <p style="color: var(--text-secondary); font-size: 0.9rem">
                                                            "No cross-subject connections mapped for this topic yet. "
                                                            "Connections show where knowledge from one subject applies in another."
                                                        </p>
                                                    </div>
                                                }.into_any()
                                            } else {
                                                view! {
                                                    <div class="praxis-detail">
                                                        <h3 style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.75rem; color: var(--text-secondary)">"Connected Across Subjects"</h3>
                                                        <p style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 1rem">"This topic connects to knowledge in other subjects — tap to explore."</p>
                                                        {neighbors.iter().map(|(neighbor, edge)| {
                                                            let href = format!("/study/{}", neighbor.id);
                                                            let title = neighbor.title.clone();
                                                            let subject = neighbor.subject_area.clone();
                                                            let rationale = edge.rationale.clone();
                                                            let edge_type = edge.edge_type.clone();
                                                            let grade = neighbor.grade_levels.first().cloned().unwrap_or_default();
                                                            view! {
                                                                <a href=href class="connection-card">
                                                                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem">
                                                                        <span class="praxis-badge" style="background: var(--info); color: var(--text-on-primary); font-size: 0.65rem">{subject}</span>
                                                                        <span style="font-size: 0.65rem; color: var(--text-tertiary)">{grade}</span>
                                                                        <span style="font-size: 0.6rem; color: var(--text-tertiary); font-style: italic">{edge_type}</span>
                                                                    </div>
                                                                    <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 0.25rem">{title}</div>
                                                                    <div style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.5">{rationale}</div>
                                                                </a>
                                                            }
                                                        }).collect::<Vec<_>>()}
                                                    </div>
                                                }.into_any()
                                            }
                                        }
                                    </div>

                                    // Notes tab
                                    <div style=move || if active_tab.get() == "notes" { "display: block" } else { "display: none" }>
                                        {
                                            let notes_key = format!("praxis_notes_{}", node_id);
                                            let initial_notes = persistence::load::<String>(&notes_key).unwrap_or_default();
                                            let (notes, set_notes) = signal(initial_notes);
                                            let notes_key_save = notes_key.clone();

                                            view! {
                                                <div class="praxis-detail">
                                                    <h3 style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.5rem; color: var(--text-secondary)">"Your Notes"</h3>
                                                    <p style="font-size: 0.75rem; color: var(--text-tertiary); margin-bottom: 0.75rem">"Write anything that helps you remember. Saved automatically."</p>
                                                    <textarea
                                                        style="width: 100%; min-height: 150px; padding: 0.75rem; background: var(--bg); border: 1px solid var(--border); border-radius: 8px; color: var(--text); font-size: 0.9rem; font-family: inherit; resize: vertical; line-height: 1.6; outline: none"
                                                        placeholder="Write your notes here..."
                                                        prop:value=move || notes.get()
                                                        on:input=move |ev| {
                                                            let val = leptos::prelude::event_target_value(&ev);
                                                            set_notes.set(val.clone());
                                                            persistence::save(&notes_key_save, &val);
                                                        }
                                                    ></textarea>
                                                </div>
                                            }
                                        }
                                    </div>
                                }.into_any()
                            }
                            Err(_e) => {
                                let desc = description.clone();
                                let sub = subdomain.clone();
                                view! {
                                    <div class="praxis-detail" style="margin-top: 1rem">
                                        <div style="padding: 0.75rem; background: var(--soil-sandy); border-radius: 8px; margin-bottom: 1rem">
                                            <div style="font-size: 0.85rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.25rem">
                                                "This lesson is still growing"
                                            </div>
                                            <div style="font-size: 0.8rem; color: var(--text-tertiary)">
                                                "Full content isn't available yet, but here's what we know about this topic."
                                            </div>
                                        </div>
                                        <h3 style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem">"Overview"</h3>
                                        <p style="font-size: 0.9rem; line-height: 1.7; color: var(--text-secondary)">
                                            {desc}
                                        </p>
                                        {if !sub.is_empty() {
                                            view! {
                                                <p style="font-size: 0.8rem; color: var(--text-tertiary); margin-top: 0.5rem">
                                                    "Part of: "{sub}
                                                </p>
                                            }.into_any()
                                        } else {
                                            view! { <span></span> }.into_any()
                                        }}
                                        <div style="margin-top: 1rem">
                                            <a href="/skill-map" style="color: var(--primary); text-decoration: none; font-size: 0.85rem">
                                                "\u{2190} Explore related topics in the constellation"
                                            </a>
                                        </div>
                                    </div>
                                }.into_any()
                            }
                        }
                    })
                }}
            </Suspense>

            // Feedback widget — "Was this helpful?"
            {
                let feedback_key = format!("praxis_feedback_{}", node_id_for_status2);
                let initial_fb = persistence::load::<String>(&feedback_key);
                let (fb_given, set_fb_given) = signal(initial_fb.is_some());
                let (show_text, set_show_text) = signal(false);
                let fb_key_up = feedback_key.clone();
                let fb_key_down = feedback_key.clone();
                let fb_key_text = feedback_key.clone();

                view! {
                    <div style="margin-top: 2rem; padding: 1rem; border-top: 1px solid var(--border); text-align: center">
                        {move || if fb_given.get() {
                            view! {
                                <p style="font-size: 0.8rem; color: var(--text-tertiary); font-style: italic">"Thanks for your feedback"</p>
                            }.into_any()
                        } else if show_text.get() {
                            let fk = fb_key_text.clone();
                            view! {
                                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem">"What could be better?"</p>
                                <textarea
                                    style="width: 100%; max-width: 400px; min-height: 60px; padding: 0.5rem; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); font-size: 0.8rem; font-family: inherit; resize: vertical"
                                    placeholder="Tell us what was confusing or missing..."
                                    on:change=move |ev| {
                                        let val = leptos::prelude::event_target_value(&ev);
                                        persistence::save(&fk, &format!("feedback:{}", val));
                                        set_fb_given.set(true);
                                    }
                                ></textarea>
                            }.into_any()
                        } else {
                            let fk_up = fb_key_up.clone();
                            view! {
                                <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem">"Was this lesson helpful?"</p>
                                <div style="display: flex; gap: 0.75rem; justify-content: center">
                                    <button
                                        style="padding: 0.4rem 1rem; background: var(--soil-rich); border: 1px solid var(--mastery-green); border-radius: 6px; color: var(--mastery-green); cursor: pointer; font-family: inherit; font-size: 0.85rem"
                                        on:click=move |_| {
                                            persistence::save(&fk_up, &"helpful".to_string());
                                            set_fb_given.set(true);
                                        }
                                    >"\u{1F44D} Yes"</button>
                                    <button
                                        style="padding: 0.4rem 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; color: var(--text-secondary); cursor: pointer; font-family: inherit; font-size: 0.85rem"
                                        on:click=move |_| set_show_text.set(true)
                                    >"\u{1F44E} Could be better"</button>
                                </div>
                            }.into_any()
                        }}
                    </div>
                }
            }
        </div>
    }.into_any()
}
