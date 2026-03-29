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

use crate::curriculum::{caps_graph, use_progress, use_set_progress, ProgressStatus};

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
    // Convert node ID to file path: CAPS.Mathematics.Gr12.P1.CALC -> caps/generated/math-12/caps.mathematics.gr12.p1.calc.json
    let id_lower = node_id.to_lowercase();
    let filename = format!("{}.json", id_lower);

    // Determine subject-grade dir
    let subject_dir = if id_lower.contains("mathematics") {
        if id_lower.contains("gr9") { "math-9" }
        else if id_lower.contains("gr10") { "math-10" }
        else if id_lower.contains("gr11") { "math-11" }
        else { "math-12" }
    } else if id_lower.contains("naturalsciences") {
        "natsci-9"
    } else {
        if id_lower.contains("gr10") { "physics-10" }
        else if id_lower.contains("gr11") { "physics-11" }
        else { "physics-12" }
    };

    let url = format!("/caps/generated/{}/{}", subject_dir, filename);

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
    let graph = caps_graph();

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
        <div class="caps-skill-map">
            // Back link
            <a href="/skill-map" style="color: var(--primary); text-decoration: none; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 0.25rem">
                "\u{2190} Back to Skill Map"
            </a>

            // Header
            <div style="margin: 1rem 0">
                <h1 style="font-size: 1.5rem; margin-bottom: 0.5rem">{title}</h1>
                <div class="caps-detail-meta">
                    <span class="caps-badge caps-badge-grade">{grade_label}</span>
                    <span class="caps-badge caps-badge-bloom">{bloom}</span>
                    <span class="caps-badge caps-badge-hours">{hours}"h estimated"</span>
                    {exam_weight.map(|ew| view! {
                        <span class="caps-badge caps-badge-exam">
                            "Paper "{ew.paper}": "{ew.marks}"/" {ew.total_paper_marks}" marks ("{format!("{:.0}", ew.percentage)}"%)"
                        </span>
                    })}
                </div>
            </div>

            // Status
            <div class="caps-status-btns">
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
                                if s == ProgressStatus::NotStarted { "caps-status-btn active-not-started" } else { "caps-status-btn" }
                            }
                            on:click={
                                let id = node_id.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::NotStarted))
                            }
                        >"Not Started"</button>
                        <button
                            class=move || {
                                let s = progress.get().get(&id_s).status;
                                if s == ProgressStatus::Studying { "caps-status-btn active-studying" } else { "caps-status-btn" }
                            }
                            on:click={
                                let id = id_s2.clone();
                                move |_| set_progress.update(|p| p.set_status(&id, ProgressStatus::Studying))
                            }
                        >"Studying"</button>
                        <button
                            class=move || {
                                let s = progress.get().get(&id_m).status;
                                if s == ProgressStatus::Mastered { "caps-status-btn active-mastered" } else { "caps-status-btn" }
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
            <div class="caps-tabs" style="max-width: 600px">
                <button class=move || if active_tab.get() == "explanation" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("explanation")>"Learn"</button>
                <button class=move || if active_tab.get() == "examples" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("examples")>"Examples"</button>
                <button class=move || if active_tab.get() == "practice" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("practice")>"Practice"</button>
                <button class=move || if active_tab.get() == "pitfalls" { "caps-tab active" } else { "caps-tab" }
                    on:click=move |_| set_active_tab.set("pitfalls")>"Pitfalls"</button>
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
                        match &*result {
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
                                        <div class="caps-detail" style="margin-bottom: 1rem">
                                            <p style="font-size: 0.95rem; line-height: 1.8">{explanation.clone()}</p>
                                        </div>

                                        {if !vocab.is_empty() {
                                            view! {
                                                <div class="caps-detail">
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
                                                <div class="caps-example">
                                                    <div class="caps-example-problem">"Example "{i + 1}": "{problem}</div>
                                                    {steps.iter().map(|s| {
                                                        let instr = s.instruction.clone();
                                                        let result = s.result.clone();
                                                        view! {
                                                            <div class="caps-example-step">
                                                                <strong>{instr}</strong>" \u{2192} "{result}
                                                            </div>
                                                        }
                                                    }).collect::<Vec<_>>()}
                                                    <div class="caps-example-answer">{answer}</div>
                                                </div>
                                            }
                                        }).collect::<Vec<_>>()}
                                    </div>

                                    // Practice tab — active recall: click each problem to reveal
                                    <div style=move || if active_tab.get() == "practice" { "display: block" } else { "display: none" }>
                                        <p style="font-size: 0.8rem; color: var(--text-tertiary); margin-bottom: 1rem">"Try to solve each problem before revealing the answer."</p>
                                        {problems.iter().enumerate().map(|(i, p)| {
                                            let question = p.question.clone();
                                            let answer = p.answer.clone();
                                            let diff = match p.difficulty_permille {
                                                0..=300 => "Easy",
                                                301..=600 => "Medium",
                                                601..=800 => "Challenging",
                                                _ => "Advanced",
                                            };
                                            let bloom_level = p.bloom_level.clone();
                                            let explanation = p.explanation.clone();
                                            let hints = p.hints.clone();
                                            let (revealed, set_revealed) = signal(false);
                                            let (show_hint, set_show_hint) = signal(false);
                                            view! {
                                                <div class="caps-problem" style="cursor: pointer">
                                                    <div class="caps-problem-q">{i + 1}". "{question}</div>
                                                    <div class="caps-problem-meta">{diff}" | "{bloom_level}</div>
                                                    {move || if revealed.get() {
                                                        view! {
                                                            <div class="caps-problem-a" style="margin-top: 0.5rem">
                                                                "Answer: "{answer.clone()}
                                                            </div>
                                                            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem">
                                                                {explanation.clone()}
                                                            </div>
                                                        }.into_any()
                                                    } else {
                                                        let hints_available = !hints.is_empty();
                                                        let hint_text = hints.first().cloned().unwrap_or_default();
                                                        view! {
                                                            <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem">
                                                                <button
                                                                    class="caps-filter-btn active"
                                                                    style="font-size: 0.75rem"
                                                                    on:click=move |_| set_revealed.set(true)
                                                                >"Reveal Answer"</button>
                                                                {if hints_available {
                                                                    view! {
                                                                        <button
                                                                            class="caps-filter-btn"
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
                                                        <div class="caps-misconception">
                                                            <div class="caps-misconception-wrong">{mc}</div>
                                                            <div class="caps-misconception-right">{corr}</div>
                                                            {if !why.is_empty() {
                                                                view! { <div class="caps-misconception-why">{why}</div> }.into_any()
                                                            } else {
                                                                view! { <span></span> }.into_any()
                                                            }}
                                                        </div>
                                                    }
                                                }).collect::<Vec<_>>()}
                                            }.into_any()
                                        }}
                                    </div>
                                }.into_any()
                            }
                            Err(ref e) => {
                                view! {
                                    <div class="caps-detail">
                                        <p style="color: var(--text-secondary)">"Could not load lesson content: "{e.clone()}</p>
                                        <p style="color: var(--text-secondary); margin-top: 0.5rem; font-size: 0.85rem">
                                            {description.clone()}
                                        </p>
                                    </div>
                                }.into_any()
                            }
                        }
                    })
                }}
            </Suspense>
        </div>
    }.into_any()
}
