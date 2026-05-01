// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Knowledge domain — excellent: claims, fact-check, prediction markets.

use domain_knowledge::types::*;
use leptos::prelude::*;
use sensorium_viz::{bar_chart::Bar, BarChart};

#[derive(Clone, Copy, PartialEq)]
enum KnowView {
    Claims,
    FactCheck,
    Markets,
}

#[component]
pub fn KnowledgeOverview() -> impl IntoView {
    let (active_view, set_active_view) = signal(KnowView::Claims);
    let claims: RwSignal<Vec<Claim>> = RwSignal::new(vec![
        Claim { id: "cl-1".into(), author_did: "did:alice".into(), statement: "Decentralized governance produces more equitable outcomes than centralized decision-making.".into(), domain: "political science".into(), evidence: vec![Evidence { source: "Ostrom 1990".into(), evidence_type: EvidenceType::Empirical, strength: 0.8, url: None }], credibility: 0.82, status: ClaimStatus::Verified, created_at: 1711900800 },
        Claim { id: "cl-2".into(), author_did: "did:bob".into(), statement: "TEND-based mutual credit eliminates the need for fractional reserve banking.".into(), domain: "economics".into(), evidence: vec![], credibility: 0.55, status: ClaimStatus::UnderReview, created_at: 1711814400 },
        Claim { id: "cl-3".into(), author_did: "did:carol".into(), statement: "HDC vectors can encode semantic meaning with 16,384 dimensions.".into(), domain: "computer science".into(), evidence: vec![Evidence { source: "Kanerva 2009".into(), evidence_type: EvidenceType::Empirical, strength: 0.9, url: None }], credibility: 0.91, status: ClaimStatus::Verified, created_at: 1711728000 },
        Claim { id: "cl-4".into(), author_did: "did:dave".into(), statement: "Community solar gardens reduce energy costs by 30% within 5 years.".into(), domain: "energy".into(), evidence: vec![], credibility: 0.45, status: ClaimStatus::Disputed, created_at: 1711641600 },
    ]);

    let markets: RwSignal<Vec<PredictionMarket>> = RwSignal::new(vec![
        PredictionMarket {
            id: "pm-1".into(),
            question: "Will the water stewardship protocol reduce waste by 20% in Q2?".into(),
            resolution_date: 1719792000,
            yes_probability: 0.72,
            volume: 4500,
            resolved: None,
        },
        PredictionMarket {
            id: "pm-2".into(),
            question: "Will all Block 7 solar panels be installed by June?".into(),
            resolution_date: 1717200000,
            yes_probability: 0.85,
            volume: 8200,
            resolved: None,
        },
        PredictionMarket {
            id: "pm-3".into(),
            question: "Will community garden yield exceed 500kg this season?".into(),
            resolution_date: 1725148800,
            yes_probability: 0.61,
            volume: 2100,
            resolved: None,
        },
    ]);

    fn status_color(s: &ClaimStatus) -> &'static str {
        match s {
            ClaimStatus::Verified => "#22c55e",
            ClaimStatus::UnderReview => "#f59e0b",
            ClaimStatus::Disputed => "#ef4444",
            ClaimStatus::Proposed => "#60a5fa",
            ClaimStatus::Retracted => "#6b7280",
        }
    }

    view! {
        <div class="knowledge-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == KnowView::Claims { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(KnowView::Claims)>"Claims"</button>
                <button class=move || if active_view.get() == KnowView::FactCheck { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(KnowView::FactCheck)>"Fact Check"</button>
                <button class=move || if active_view.get() == KnowView::Markets { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(KnowView::Markets)>"Prediction Markets"</button>
            </div>

            <Show when=move || active_view.get() == KnowView::Claims>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #0891B2">"CLAIMS"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || claims.get().len().to_string()}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"VERIFIED"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || claims.get().iter().filter(|c| c.status == ClaimStatus::Verified).count().to_string()}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22D3EE">"AVG CREDIBILITY"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || { let cs = claims.get(); let avg = cs.iter().map(|c| c.credibility).sum::<f64>() / cs.len().max(1) as f64; format!("{:.0}%", avg * 100.0) }}</p>
                    </div>
                </div>
                <div class="thought-list">
                    {move || { claims.get().iter().map(|c| {
                        let statement = c.statement.clone();
                        let domain = c.domain.clone();
                        let cred = c.credibility;
                        let scolor = status_color(&c.status);
                        let status = format!("{:?}", c.status);
                        let evidence_count = c.evidence.len();
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {scolor};")>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {scolor}")>{status}</span>
                                    <span class="thought-confidence">{format!("{:.0}% credibility", cred * 100.0)}</span>
                                    <span class="thought-domain">{domain}</span>
                                </div>
                                <p class="thought-content">{statement}</p>
                                <p style="font-size: 0.65rem; color: var(--text-muted)">{format!("{evidence_count} evidence source(s)")}</p>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>

            <Show when=move || active_view.get() == KnowView::FactCheck>
                <h3 class="section-title">"Verdicts by Domain"</h3>
                <BarChart data=vec![
                    Bar { label: "Science".into(), value: 89.0, color: "#0891B2".into() },
                    Bar { label: "Policy".into(), value: 62.0, color: "#22D3EE".into() },
                    Bar { label: "Health".into(), value: 78.0, color: "#0D7377".into() },
                    Bar { label: "Economics".into(), value: 54.0, color: "#D97706".into() },
                    Bar { label: "Tech".into(), value: 71.0, color: "#2563EB".into() },
                ] width=350.0 height=180.0 />
                <div class="thought-card" style="margin-top: 1rem;">
                    <p style="font-size: 0.8rem; color: var(--text-secondary)">"Fact-checking uses Broca\u{2192}Mycelix knowledge verification. Claims are assessed by community consensus + evidence quality."</p>
                </div>
            </Show>

            <Show when=move || active_view.get() == KnowView::Markets>
                <div class="thought-list">
                    {move || { markets.get().iter().map(|m| {
                        let question = m.question.clone();
                        let prob = m.yes_probability;
                        let volume = m.volume;
                        let color = if prob > 0.7 { "#22c55e" } else if prob > 0.5 { "#f59e0b" } else { "#ef4444" };
                        view! {
                            <div class="thought-card" style=format!("border-left: 3px solid {color};")>
                                <div class="thought-meta">
                                    <span class="thought-type" style=format!("color: {color}")>{format!("{:.0}% YES", prob * 100.0)}</span>
                                    <span class="thought-confidence">{format!("{volume} SAP volume")}</span>
                                </div>
                                <p class="thought-content">{question}</p>
                            </div>
                        }
                    }).collect::<Vec<_>>() }}
                </div>
            </Show>
        </div>
    }
}
