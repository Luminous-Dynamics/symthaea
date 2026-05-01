// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LUCID domain — excellent: state store, thought CRUD, graph interactivity.

use domain_lucid::types::*;
use leptos::prelude::*;
use sensorium_viz::force_graph::{GraphEdge, GraphNode};
use sensorium_viz::{line_chart::Series, ForceGraph, LineChart};

use crate::identity::{ConductorStatus, SensoriumIdentity};

// ── State Store ──

#[derive(Clone, Copy, PartialEq)]
enum LucidView {
    Thoughts,
    Graph,
    Collective,
    Reasoning,
}

#[derive(Clone)]
struct LucidState {
    thoughts: RwSignal<Vec<Thought>>,
    relationships: RwSignal<Vec<Relationship>>,
    selected_thought: RwSignal<Option<String>>,
    loading: RwSignal<bool>,
    show_create_form: RwSignal<bool>,
}

impl LucidState {
    fn new() -> Self {
        Self {
            thoughts: RwSignal::new(Vec::new()),
            relationships: RwSignal::new(Vec::new()),
            selected_thought: RwSignal::new(None),
            loading: RwSignal::new(true),
            show_create_form: RwSignal::new(false),
        }
    }
}

// ── Mock Data ──

fn mock_thoughts() -> Vec<Thought> {
    vec![
        Thought { id: "t-001".into(), content: "Consciousness is substrate-independent \u{2014} the pattern matters, not the medium.".into(), thought_type: ThoughtType::Claim, confidence: 0.85, tags: vec!["consciousness".into(), "substrate".into()], domain: Some("philosophy".into()), epistemic: EpistemicProfile { empirical: 0.3, normative: 0.2, materiality: 0.7, harmonic: 0.6 }, author_did: "did:mycelix:uhCAkAlice".into(), created_at: 1711900800, updated_at: 1711900800 },
        Thought { id: "t-002".into(), content: "IIT's Phi metric captures integration but misses temporal dynamics.".into(), thought_type: ThoughtType::Critique, confidence: 0.72, tags: vec!["IIT".into(), "phi".into()], domain: Some("neuroscience".into()), epistemic: EpistemicProfile { empirical: 0.6, normative: 0.1, materiality: 0.5, harmonic: 0.4 }, author_did: "did:mycelix:uhCAkBob".into(), created_at: 1711814400, updated_at: 1711900800 },
        Thought { id: "t-003".into(), content: "Combining HDC with liquid-time CfC creates genuinely novel temporal binding.".into(), thought_type: ThoughtType::Hypothesis, confidence: 0.68, tags: vec!["HDC".into(), "CfC".into()], domain: Some("AI".into()), epistemic: EpistemicProfile { empirical: 0.5, normative: 0.0, materiality: 0.8, harmonic: 0.7 }, author_did: "did:mycelix:uhCAkAlice".into(), created_at: 1711728000, updated_at: 1711728000 },
        Thought { id: "t-004".into(), content: "TEND currency creates alignment between individual and collective benefit.".into(), thought_type: ThoughtType::Observation, confidence: 0.91, tags: vec!["TEND".into(), "economics".into()], domain: Some("economics".into()), epistemic: EpistemicProfile { empirical: 0.7, normative: 0.5, materiality: 0.3, harmonic: 0.8 }, author_did: "did:mycelix:uhCAkCarol".into(), created_at: 1711641600, updated_at: 1711641600 },
        Thought { id: "t-005".into(), content: "What if moral reasoning emerges from autonomic-cognitive interaction?".into(), thought_type: ThoughtType::Question, confidence: 0.55, tags: vec!["morality".into(), "embodiment".into()], domain: Some("philosophy".into()), epistemic: EpistemicProfile { empirical: 0.2, normative: 0.6, materiality: 0.4, harmonic: 0.5 }, author_did: "did:mycelix:uhCAkBob".into(), created_at: 1711555200, updated_at: 1711555200 },
    ]
}

fn mock_relationships() -> Vec<Relationship> {
    vec![
        Relationship {
            id: "r-01".into(),
            source_id: "t-002".into(),
            target_id: "t-001".into(),
            relationship_type: RelationshipType::Extends,
            strength: 0.8,
            created_at: 1711900800,
        },
        Relationship {
            id: "r-02".into(),
            source_id: "t-003".into(),
            target_id: "t-001".into(),
            relationship_type: RelationshipType::Supports,
            strength: 0.7,
            created_at: 1711814400,
        },
        Relationship {
            id: "r-03".into(),
            source_id: "t-003".into(),
            target_id: "t-002".into(),
            relationship_type: RelationshipType::Refines,
            strength: 0.6,
            created_at: 1711814400,
        },
        Relationship {
            id: "r-04".into(),
            source_id: "t-004".into(),
            target_id: "t-001".into(),
            relationship_type: RelationshipType::Contextualizes,
            strength: 0.5,
            created_at: 1711728000,
        },
        Relationship {
            id: "r-05".into(),
            source_id: "t-005".into(),
            target_id: "t-001".into(),
            relationship_type: RelationshipType::QuestionsAssumptionOf,
            strength: 0.65,
            created_at: 1711641600,
        },
        Relationship {
            id: "r-06".into(),
            source_id: "t-005".into(),
            target_id: "t-004".into(),
            relationship_type: RelationshipType::Inspires,
            strength: 0.4,
            created_at: 1711641600,
        },
    ]
}

fn type_color(t: &ThoughtType) -> &'static str {
    match t {
        ThoughtType::Claim => "#22c55e",
        ThoughtType::Hypothesis => "#60a5fa",
        ThoughtType::Critique => "#f59e0b",
        ThoughtType::Observation => "#8b5cf6",
        ThoughtType::Question => "#ec4899",
        ThoughtType::Reflection => "#06b6d4",
        ThoughtType::Synthesis => "#e8c547",
    }
}
fn type_label(t: &ThoughtType) -> &'static str {
    match t {
        ThoughtType::Claim => "Claim",
        ThoughtType::Hypothesis => "Hypothesis",
        ThoughtType::Critique => "Critique",
        ThoughtType::Observation => "Observation",
        ThoughtType::Question => "Question",
        ThoughtType::Reflection => "Reflection",
        ThoughtType::Synthesis => "Synthesis",
    }
}
fn rel_color(r: &RelationshipType) -> &'static str {
    match r {
        RelationshipType::Supports => "rgba(34, 197, 94, 0.4)",
        RelationshipType::Contradicts => "rgba(239, 68, 68, 0.4)",
        RelationshipType::Extends | RelationshipType::Refines => "rgba(96, 165, 250, 0.4)",
        RelationshipType::Inspires => "rgba(232, 197, 71, 0.3)",
        RelationshipType::QuestionsAssumptionOf => "rgba(236, 72, 153, 0.4)",
        RelationshipType::Synthesizes => "rgba(139, 92, 246, 0.4)",
        RelationshipType::Contextualizes => "rgba(6, 182, 212, 0.3)",
    }
}

// ── Main Component ──

#[component]
pub fn LucidOverview() -> impl IntoView {
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");
    let (active_view, set_active_view) = signal(LucidView::Thoughts);

    let state = LucidState::new();
    provide_context(state.clone());

    // Load data
    let identity_for_load = identity.clone();
    let state_for_load = state.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let (thoughts, rels) = if identity_for_load.conductor_status.get()
            == ConductorStatus::Connected
        {
            let t = identity_for_load
                .call_zome::<(), Vec<Thought>>("lucid", "lucid", "get_my_thoughts", &())
                .await
                .unwrap_or_else(|_| mock_thoughts());
            let r = identity_for_load
                .call_zome::<(), Vec<Relationship>>("lucid", "lucid", "get_my_relationships", &())
                .await
                .unwrap_or_else(|_| mock_relationships());
            (t, r)
        } else {
            (mock_thoughts(), mock_relationships())
        };
        state_for_load.thoughts.set(thoughts);
        state_for_load.relationships.set(rels);
        state_for_load.loading.set(false);
    });

    view! {
        <div class="lucid-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == LucidView::Thoughts { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| { set_active_view.set(LucidView::Thoughts); state.selected_thought.set(None); }>"Thoughts"</button>
                <button class=move || if active_view.get() == LucidView::Graph { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(LucidView::Graph)>"Graph"</button>
                <button class=move || if active_view.get() == LucidView::Collective { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(LucidView::Collective)>"Collective"</button>
                <button class=move || if active_view.get() == LucidView::Reasoning { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(LucidView::Reasoning)>"Reasoning"</button>
            </div>

            <Show when=move || state.loading.get()>
                <div class="domain-coming-soon"><p class="coming-soon-text">"Loading knowledge graph..."</p></div>
            </Show>

            // Thoughts view with detail
            <Show when=move || !state.loading.get() && active_view.get() == LucidView::Thoughts>
                {move || {
                    if let Some(ref tid) = state.selected_thought.get() {
                        let thought = state.thoughts.get().iter().find(|t| t.id == *tid).cloned();
                        if let Some(t) = thought {
                            view! { <ThoughtDetail thought=t /> }.into_any()
                        } else { view! { <p>"Not found"</p> }.into_any() }
                    } else {
                        view! { <ThoughtList /> }.into_any()
                    }
                }}
            </Show>

            // Graph view
            <Show when=move || !state.loading.get() && active_view.get() == LucidView::Graph>
                <GraphView />
            </Show>

            // Collective
            <Show when=move || !state.loading.get() && active_view.get() == LucidView::Collective>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #8b5cf6">"THOUGHTS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.thoughts.get().len().to_string()}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22d3ee">"COHERENCE"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"0.71"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"RELATIONSHIPS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.relationships.get().len().to_string()}</p>
                    </div>
                </div>
                <h3 class="section-title">"Coherence Trend"</h3>
                <LineChart series=vec![
                    Series { label: "Confidence".into(), color: "#8b5cf6".into(), data: vec![0.62, 0.65, 0.68, 0.71, 0.69, 0.74, 0.72, 0.75] },
                    Series { label: "Coherence".into(), color: "#22d3ee".into(), data: vec![0.45, 0.52, 0.58, 0.55, 0.61, 0.67, 0.64, 0.70] },
                ] width=400.0 height=160.0 />
            </Show>

            // Reasoning
            <Show when=move || !state.loading.get() && active_view.get() == LucidView::Reasoning>
                <div class="thought-card">
                    <h3 class="section-title">"Reasoning Chains"</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary)">"Deductive, inductive, and abductive reasoning chains through the knowledge graph."</p>
                </div>
            </Show>
        </div>
    }
}

// ── Thought List ──

#[component]
fn ThoughtList() -> impl IntoView {
    let state = use_context::<LucidState>().expect("LucidState");

    view! {
        <div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 0.7rem; color: var(--text-muted)">
                    {move || format!("{} thoughts", state.thoughts.get().len())}
                </span>
                <button class="domain-nav-btn" on:click=move |_| state.show_create_form.update(|v| *v = !*v)>
                    {move || if state.show_create_form.get() { "- Close" } else { "+ New Thought" }}
                </button>
            </div>

            <div class="thought-form" style:display=move || if state.show_create_form.get() { "flex" } else { "none" }>
                <CreateThoughtInline />
            </div>

            <div class="thought-list">
                {move || { state.thoughts.get().iter().map(|t| {
                    let id = t.id.clone();
                    let content = t.content.clone();
                    let ttype = type_label(&t.thought_type);
                    let tcolor = type_color(&t.thought_type);
                    let confidence = t.confidence;
                    let tags = t.tags.clone();
                    let domain = t.domain.clone().unwrap_or_default();
                    let ep = t.epistemic.clone();

                    view! {
                        <div class="thought-card" style="cursor: pointer;"
                            on:click=move |_| state.selected_thought.set(Some(id.clone()))>
                            <div class="thought-meta">
                                <span class="thought-type" style=format!("color: {tcolor}")>{ttype}</span>
                                <span class="thought-confidence">{format!("{:.0}%", confidence * 100.0)}</span>
                                {(!domain.is_empty()).then(|| view! { <span class="thought-domain">{domain.clone()}</span> })}
                            </div>
                            <p class="thought-content">{content}</p>
                            <div class="thought-tags">
                                {tags.iter().map(|tag| { let tag = tag.clone(); view! { <span class="thought-tag">{tag}</span> } }).collect::<Vec<_>>()}
                            </div>
                            <div class="epistemic-bar">
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #22c55e;", ep.empirical * 100.0) />
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #f59e0b;", ep.normative * 100.0) />
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #60a5fa;", ep.materiality * 100.0) />
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #8b5cf6;", ep.harmonic * 100.0) />
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>() }}
            </div>
        </div>
    }
}

// ── Thought Detail ──

#[component]
fn ThoughtDetail(thought: Thought) -> impl IntoView {
    let state = use_context::<LucidState>().expect("LucidState");
    let t = thought;
    let tcolor = type_color(&t.thought_type);
    let related: Vec<_> = state
        .relationships
        .get()
        .iter()
        .filter(|r| r.source_id == t.id || r.target_id == t.id)
        .cloned()
        .collect();

    view! {
        <div class="proposal-detail">
            <button class="domain-nav-btn" style="margin-bottom: 1rem;"
                on:click=move |_| state.selected_thought.set(None)>"\u{2190} Back"</button>

            <div class="thought-card" style=format!("border-left: 3px solid {tcolor};")>
                <div class="thought-meta" style="margin-bottom: 0.5rem;">
                    <span class="thought-type" style=format!("color: {tcolor}")>{type_label(&t.thought_type)}</span>
                    <span class="thought-confidence">{format!("{:.0}% confidence", t.confidence * 100.0)}</span>
                    {t.domain.clone().map(|d| view! { <span class="thought-domain">{d}</span> })}
                </div>
                <p style="font-size: 1rem; line-height: 1.6; margin-bottom: 1rem;">{t.content.clone()}</p>

                // Epistemic profile detail
                <div style="margin-bottom: 1rem;">
                    <h4 class="section-title">"Epistemic Profile"</h4>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; text-align: center;">
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #22c55e">{format!("{:.0}%", t.epistemic.empirical * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Empirical"</div></div>
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #f59e0b">{format!("{:.0}%", t.epistemic.normative * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Normative"</div></div>
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #60a5fa">{format!("{:.0}%", t.epistemic.materiality * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Materiality"</div></div>
                        <div><div style="font-size: 1.2rem; font-weight: 700; color: #8b5cf6">{format!("{:.0}%", t.epistemic.harmonic * 100.0)}</div><div style="font-size: 0.6rem; color: var(--text-muted)">"Harmonic"</div></div>
                    </div>
                </div>

                // Related thoughts
                {(!related.is_empty()).then(|| {
                    let rels = related.clone();
                    view! {
                        <div>
                            <h4 class="section-title">"Relationships"</h4>
                            <div style="display: flex; flex-direction: column; gap: 0.3rem;">
                                {rels.iter().map(|r| {
                                    let other_id = if r.source_id == t.id { r.target_id.clone() } else { r.source_id.clone() };
                                    let rel_type = format!("{:?}", r.relationship_type);
                                    let strength = r.strength;
                                    view! {
                                        <div style="font-size: 0.75rem; color: var(--text-secondary)">
                                            {format!("{rel_type} \u{2194} {other_id} (strength: {strength:.1})")}
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                    }
                })}

                // Tags + metadata
                <div class="thought-tags" style="margin-top: 0.75rem;">
                    {t.tags.iter().map(|tag| { let tag = tag.clone(); view! { <span class="thought-tag">{tag}</span> } }).collect::<Vec<_>>()}
                </div>
                <div style="font-size: 0.65rem; color: var(--text-muted); margin-top: 0.5rem;">
                    {format!("Author: {}...  \u{00b7}  ID: {}", &t.author_did[..20.min(t.author_did.len())], t.id)}
                </div>
            </div>
        </div>
    }
}

// ── Graph View ──

#[component]
fn GraphView() -> impl IntoView {
    let state = use_context::<LucidState>().expect("LucidState");

    let graph_nodes = move || {
        state
            .thoughts
            .get()
            .iter()
            .map(|t| GraphNode {
                id: t.id.clone(),
                label: if t.content.len() > 25 {
                    format!("{}...", &t.content[..25])
                } else {
                    t.content.clone()
                },
                color: type_color(&t.thought_type).into(),
                size: 0.8 + t.confidence * 0.6,
                group: t.domain.clone(),
            })
            .collect::<Vec<_>>()
    };

    let graph_edges = move || {
        state
            .relationships
            .get()
            .iter()
            .map(|r| GraphEdge {
                source: r.source_id.clone(),
                target: r.target_id.clone(),
                weight: r.strength,
                color: Some(rel_color(&r.relationship_type).into()),
            })
            .collect::<Vec<_>>()
    };

    // ForceGraph needs static data (not reactive closures) — snapshot once
    let nodes = graph_nodes();
    let edges = graph_edges();

    view! {
        <div class="lucid-graph-container">
            <h3 class="section-title">{move || format!("Thought Network ({} nodes, {} edges)", state.thoughts.get().len(), state.relationships.get().len())}</h3>
            <ForceGraph nodes=nodes edges=edges width=500.0 height=350.0 repulsion=4000.0 attraction=0.003 />
            <div style="margin-top: 0.5rem; font-size: 0.65rem; color: var(--text-muted);">
                "Node size = confidence. Color = thought type. Edge opacity = relationship strength."
            </div>
        </div>
    }
}

// ── Create Thought (inline form) ──

#[component]
fn CreateThoughtInline() -> impl IntoView {
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");
    let state = use_context::<LucidState>().expect("LucidState");
    let (content, set_content) = signal(String::new());
    let (thought_type, set_thought_type) = signal("Observation".to_string());
    let (confidence, set_confidence) = signal(0.7_f64);
    let (tags_input, set_tags) = signal(String::new());
    let (submitting, set_submitting) = signal(false);

    let identity_for_submit = identity.clone();
    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        let c = content.get();
        if c.trim().is_empty() {
            return;
        }
        set_submitting.set(true);

        let identity = identity_for_submit.clone();
        let tt = thought_type.get();
        let conf = confidence.get();
        let tags: Vec<String> = tags_input
            .get()
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        let content_val = c.trim().to_string();

        wasm_bindgen_futures::spawn_local(async move {
            let input = CreateThoughtInput {
                content: content_val.clone(),
                thought_type: match tt.as_str() {
                    "Claim" => ThoughtType::Claim,
                    "Hypothesis" => ThoughtType::Hypothesis,
                    "Critique" => ThoughtType::Critique,
                    "Question" => ThoughtType::Question,
                    "Reflection" => ThoughtType::Reflection,
                    "Synthesis" => ThoughtType::Synthesis,
                    _ => ThoughtType::Observation,
                },
                confidence: conf,
                tags: tags.clone(),
                domain: None,
                epistemic: EpistemicProfile {
                    empirical: 0.5,
                    normative: 0.3,
                    materiality: 0.5,
                    harmonic: 0.5,
                },
            };
            let _ = identity
                .call_zome::<CreateThoughtInput, serde_json::Value>(
                    "lucid",
                    "lucid",
                    "create_thought",
                    &input,
                )
                .await;

            // Optimistic: add to state
            let new_thought = Thought {
                id: format!("t-{:03}", state.thoughts.get().len() + 6),
                content: content_val,
                thought_type: input.thought_type,
                confidence: conf,
                tags,
                domain: None,
                epistemic: input.epistemic,
                author_did: "did:mycelix:uhCAkYou".into(),
                created_at: 0,
                updated_at: 0,
            };
            state.thoughts.update(|v| v.insert(0, new_thought));
            set_content.set(String::new());
            set_tags.set(String::new());
            set_submitting.set(false);
            state.show_create_form.set(false);
        });
    };

    view! {
        <form on:submit=on_submit style="display: flex; flex-direction: column; gap: 0.5rem; width: 100%;">
            <textarea class="form-textarea" placeholder="What are you thinking?..." rows="3"
                prop:value=move || content.get()
                on:input=move |ev| set_content.set(event_target_value(&ev)) />
            <div class="form-row">
                <select class="form-select" on:change=move |ev| set_thought_type.set(event_target_value(&ev))>
                    <option value="Observation">"Observation"</option>
                    <option value="Claim">"Claim"</option>
                    <option value="Hypothesis">"Hypothesis"</option>
                    <option value="Question">"Question"</option>
                    <option value="Critique">"Critique"</option>
                    <option value="Synthesis">"Synthesis"</option>
                </select>
                <div class="confidence-slider">
                    <span class="form-label">"Confidence: " {move || format!("{:.0}%", confidence.get() * 100.0)}</span>
                    <input type="range" min="0" max="1" step="0.05"
                        prop:value=move || confidence.get().to_string()
                        on:input=move |ev| { if let Ok(v) = event_target_value(&ev).parse::<f64>() { set_confidence.set(v); } } />
                </div>
            </div>
            <input type="text" class="form-input" placeholder="Tags (comma-separated)..."
                prop:value=move || tags_input.get()
                on:input=move |ev| set_tags.set(event_target_value(&ev)) />
            <button type="submit" class="form-submit" disabled=move || submitting.get()>
                {move || if submitting.get() { "Adding..." } else { "Add Thought" }}
            </button>
        </form>
    }
}
