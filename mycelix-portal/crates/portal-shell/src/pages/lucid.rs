// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LUCID domain pages — thoughts, relationship graph, collective sensemaking.

use leptos::prelude::*;
use domain_lucid::types::*;
use portal_viz::{ForceGraph, LineChart, line_chart::Series};
use portal_viz::force_graph::{GraphNode, GraphEdge};

use crate::identity::{ConductorStatus, PortalIdentity};

/// Mock thoughts for demo mode.
fn mock_thoughts() -> Vec<Thought> {
    vec![
        Thought {
            id: "t-001".into(),
            content: "Consciousness is substrate-independent — the pattern matters, not the medium.".into(),
            thought_type: ThoughtType::Claim,
            confidence: 0.85,
            tags: vec!["consciousness".into(), "substrate".into()],
            domain: Some("philosophy".into()),
            epistemic: EpistemicProfile { empirical: 0.3, normative: 0.2, materiality: 0.7, harmonic: 0.6 },
            author_did: "did:mycelix:uhCAkAlice".into(),
            created_at: 1711900800,
            updated_at: 1711900800,
        },
        Thought {
            id: "t-002".into(),
            content: "IIT's Phi metric captures integration but misses temporal dynamics.".into(),
            thought_type: ThoughtType::Critique,
            confidence: 0.72,
            tags: vec!["IIT".into(), "phi".into(), "temporal".into()],
            domain: Some("neuroscience".into()),
            epistemic: EpistemicProfile { empirical: 0.6, normative: 0.1, materiality: 0.5, harmonic: 0.4 },
            author_did: "did:mycelix:uhCAkBob".into(),
            created_at: 1711814400,
            updated_at: 1711900800,
        },
        Thought {
            id: "t-003".into(),
            content: "Combining HDC with liquid-time CfC creates genuinely novel temporal binding.".into(),
            thought_type: ThoughtType::Hypothesis,
            confidence: 0.68,
            tags: vec!["HDC".into(), "CfC".into(), "binding".into()],
            domain: Some("AI".into()),
            epistemic: EpistemicProfile { empirical: 0.5, normative: 0.0, materiality: 0.8, harmonic: 0.7 },
            author_did: "did:mycelix:uhCAkAlice".into(),
            created_at: 1711728000,
            updated_at: 1711728000,
        },
        Thought {
            id: "t-004".into(),
            content: "Mycelix's TEND currency creates alignment between individual and collective benefit.".into(),
            thought_type: ThoughtType::Observation,
            confidence: 0.91,
            tags: vec!["TEND".into(), "economics".into(), "alignment".into()],
            domain: Some("economics".into()),
            epistemic: EpistemicProfile { empirical: 0.7, normative: 0.5, materiality: 0.3, harmonic: 0.8 },
            author_did: "did:mycelix:uhCAkCarol".into(),
            created_at: 1711641600,
            updated_at: 1711641600,
        },
        Thought {
            id: "t-005".into(),
            content: "What if moral reasoning emerges from the interaction of autonomic and cognitive systems?".into(),
            thought_type: ThoughtType::Question,
            confidence: 0.55,
            tags: vec!["morality".into(), "embodiment".into()],
            domain: Some("philosophy".into()),
            epistemic: EpistemicProfile { empirical: 0.2, normative: 0.6, materiality: 0.4, harmonic: 0.5 },
            author_did: "did:mycelix:uhCAkBob".into(),
            created_at: 1711555200,
            updated_at: 1711555200,
        },
    ]
}

fn mock_relationships() -> Vec<Relationship> {
    vec![
        Relationship { id: "r-01".into(), source_id: "t-002".into(), target_id: "t-001".into(), relationship_type: RelationshipType::Extends, strength: 0.8, created_at: 1711900800 },
        Relationship { id: "r-02".into(), source_id: "t-003".into(), target_id: "t-001".into(), relationship_type: RelationshipType::Supports, strength: 0.7, created_at: 1711814400 },
        Relationship { id: "r-03".into(), source_id: "t-003".into(), target_id: "t-002".into(), relationship_type: RelationshipType::Refines, strength: 0.6, created_at: 1711814400 },
        Relationship { id: "r-04".into(), source_id: "t-004".into(), target_id: "t-001".into(), relationship_type: RelationshipType::Contextualizes, strength: 0.5, created_at: 1711728000 },
        Relationship { id: "r-05".into(), source_id: "t-005".into(), target_id: "t-001".into(), relationship_type: RelationshipType::QuestionsAssumptionOf, strength: 0.65, created_at: 1711641600 },
        Relationship { id: "r-06".into(), source_id: "t-005".into(), target_id: "t-004".into(), relationship_type: RelationshipType::Inspires, strength: 0.4, created_at: 1711641600 },
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

/// LUCID overview — thoughts list + relationship force graph.
#[component]
pub fn LucidOverview() -> impl IntoView {
    let identity = use_context::<PortalIdentity>().expect("PortalIdentity");

    // Try conductor, fall back to mock
    let data_resource = LocalResource::new(move || {
        let identity = identity.clone();
        async move {
            if identity.conductor_status.get() == ConductorStatus::Connected {
                let thoughts_result = identity.call_zome::<(), Vec<Thought>>(
                    "lucid", "lucid", "get_my_thoughts", &()
                ).await;
                let rels_result = identity.call_zome::<(), Vec<Relationship>>(
                    "lucid", "lucid", "get_my_relationships", &()
                ).await;
                if let (Ok(t), Ok(r)) = (thoughts_result, rels_result) {
                    return (t, r);
                }
            }
            (mock_thoughts(), mock_relationships())
        }
    });

    let thoughts = move || data_resource.get().map(|(t, _)| t).unwrap_or_else(mock_thoughts);
    let relationships = move || data_resource.get().map(|(_, r)| r).unwrap_or_else(mock_relationships);

    // Build graph data for ForceGraph (from initial load)
    let initial_thoughts = thoughts();
    let initial_rels = relationships();

    let graph_nodes: Vec<GraphNode> = initial_thoughts.iter().map(|t| {
        GraphNode {
            id: t.id.clone(),
            label: if t.content.len() > 30 { format!("{}...", &t.content[..30]) } else { t.content.clone() },
            color: type_color(&t.thought_type).into(),
            size: 0.8 + t.confidence * 0.6,
            group: t.domain.clone(),
        }
    }).collect();

    let graph_edges: Vec<GraphEdge> = initial_rels.iter().map(|r| {
        GraphEdge {
            source: r.source_id.clone(),
            target: r.target_id.clone(),
            weight: r.strength,
            color: Some(rel_color(&r.relationship_type).into()),
        }
    }).collect();

    // Confidence trend (mock time series)
    let confidence_series = vec![
        Series {
            label: "Avg Confidence".into(),
            color: "#8b5cf6".into(),
            data: vec![0.62, 0.65, 0.68, 0.71, 0.69, 0.74, 0.72, 0.75],
        },
        Series {
            label: "Coherence".into(),
            color: "#22d3ee".into(),
            data: vec![0.45, 0.52, 0.58, 0.55, 0.61, 0.67, 0.64, 0.70],
        },
    ];

    view! {
        <div class="lucid-content">
            <div class="lucid-nav">
                <button class="domain-nav-btn active">"Thoughts"</button>
                <button class="domain-nav-btn">"Relationships"</button>
                <button class="domain-nav-btn">"Collective"</button>
                <button class="domain-nav-btn">"Reasoning"</button>
            </div>

            // Relationship graph
            <div class="lucid-graph-container">
                <h3 class="section-title">"Thought Network"</h3>
                <ForceGraph
                    nodes=graph_nodes
                    edges=graph_edges
                    width=500.0
                    height=350.0
                    repulsion=4000.0
                    attraction=0.003
                />
            </div>

            // Thought list
            <div class="thought-list">
                <h3 class="section-title">"Recent Thoughts"</h3>
                {move || { thoughts().iter().map(|t| {
                    let content = t.content.clone();
                    let ttype = type_label(&t.thought_type);
                    let tcolor = type_color(&t.thought_type);
                    let confidence = t.confidence;
                    let tags = t.tags.clone();
                    let domain = t.domain.clone().unwrap_or_default();
                    let ep = t.epistemic.clone();

                    view! {
                        <div class="thought-card">
                            <div class="thought-meta">
                                <span class="thought-type" style=format!("color: {tcolor}")>{ttype}</span>
                                <span class="thought-confidence">{format!("{:.0}%", confidence * 100.0)}</span>
                                {(!domain.is_empty()).then(|| view! {
                                    <span class="thought-domain">{domain}</span>
                                })}
                            </div>
                            <p class="thought-content">{content}</p>
                            <div class="thought-tags">
                                {tags.iter().map(|tag| {
                                    let tag = tag.clone();
                                    view! { <span class="thought-tag">{tag}</span> }
                                }).collect::<Vec<_>>()}
                            </div>
                            <div class="epistemic-bar">
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #22c55e;", ep.empirical * 100.0) title="Empirical" />
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #f59e0b;", ep.normative * 100.0) title="Normative" />
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #60a5fa;", ep.materiality * 100.0) title="Materiality" />
                                <span class="ep-segment" style=format!("width: {:.0}%; background: #8b5cf6;", ep.harmonic * 100.0) title="Harmonic" />
                            </div>
                        </div>
                    }
                }).collect::<Vec<_>>() }}
            </div>

            // Confidence trend chart
            <div class="lucid-chart">
                <h3 class="section-title">"Knowledge Coherence Trend"</h3>
                <LineChart
                    series=confidence_series
                    width=400.0
                    height=160.0
                />
            </div>
        </div>
    }
}
