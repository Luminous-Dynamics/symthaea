// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Single claim deep-dive page.

use leptos::prelude::*;
use leptos_router::hooks::use_params_map;
use crate::types::*;
use crate::components::lem_cube::LemCube;
use crate::components::lem_badge::LemBadge;

fn demo_claim() -> ClaimResponse {
    ClaimResponse {
        id: "island-stability-z120".into(),
        tier: EpistemicTier::E2,
        content: ClaimContent {
            dataset_hash: "symthaea-nuclear-sweep".into(),
            description: "Superheavy stability island centered at Z=115-120, N=180. Element 120 (A=294) has shell correction -22.81 MeV.".into(),
            category: "Nuclear Physics".into(),
            keywords: vec!["island-of-stability".into(), "superheavy".into(), "shell-model".into()],
            storage_ref: Some("symthaea-nuclear::island_stability".into()),
            reproducibility_score: Some(0.85),
            license: Some("AGPL-3.0".into()),
        },
        creator: "Symthaea Nuclear Sweep".into(),
        created_at: "2026-04-04T00:00:00Z".into(),
        verifications_count: 0,
        provenance_count: 1,
    }
}

fn demo_verifications() -> Vec<Verification> {
    vec![
        Verification {
            verifier: "nuclear-reviewer-1".into(),
            timestamp: "2026-04-04T12:00:00Z".into(),
            evidence: "SEMF binding energies match Möller FRDM(2012) within 3 MeV for Z>100".into(),
            methodology_match: "Close".into(),
            outcome: "Partial Replication".into(),
        },
    ]
}

#[component]
pub fn ClaimDetailPage() -> impl IntoView {
    let params = use_params_map();
    let claim_id = move || params.read().get("id").unwrap_or_default();

    let claim = demo_claim();
    let verifications = demo_verifications();

    view! {
        <div class="page-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h1 class="page-title" style="margin-bottom: 0;">"Claim Detail"</h1>
                <LemBadge tier=claim.tier />
            </div>

            <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem;">
                // Left column — claim content
                <div>
                    <div class="glass-panel" style="margin-bottom: 1rem;">
                        <h3 style="font-size: 1rem; color: var(--accent-indigo); margin-bottom: 0.5rem;">{claim.content.category.clone()}</h3>
                        <p style="font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6;">{claim.content.description.clone()}</p>
                        <div style="margin-top: 0.75rem; display: flex; gap: 0.25rem; flex-wrap: wrap;">
                            {claim.content.keywords.iter().map(|k| view! {
                                <span style="font-size: 0.7rem; padding: 0.125rem 0.375rem; background: rgba(99,102,241,0.1); border-radius: 9999px; color: var(--accent-indigo);">{k.clone()}</span>
                            }).collect::<Vec<_>>()}
                        </div>
                    </div>

                    // Verification history
                    <div class="glass-panel" style="margin-bottom: 1rem;">
                        <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">"Verification History"</h3>
                        {verifications.iter().map(|v| view! {
                            <div style="padding: 0.75rem; background: var(--bg-secondary); border-radius: 0.5rem; margin-bottom: 0.5rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.25rem;">
                                    <span style="font-weight: 600;">{v.verifier.clone()}</span>
                                    <span style="color: var(--text-secondary);">{v.timestamp.chars().take(10).collect::<String>()}</span>
                                </div>
                                <p style="font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4;">{v.evidence.clone()}</p>
                                <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.25rem;">
                                    {format!("Methodology: {} | Outcome: {}", v.methodology_match, v.outcome)}
                                </div>
                            </div>
                        }).collect::<Vec<_>>()}
                        <a href=format!("/claims/{}/verify", claim.id) class="btn btn-primary" style="display: inline-block; margin-top: 0.5rem; text-decoration: none;">"Add Verification"</a>
                    </div>

                    // Provenance
                    <div class="glass-panel">
                        <h3 style="font-size: 1rem; margin-bottom: 0.5rem;">"Provenance"</h3>
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            <div>"Dataset: " <code style="color: var(--accent-emerald);">{claim.content.dataset_hash.clone()}</code></div>
                            {claim.content.storage_ref.as_ref().map(|s| view! { <div style="margin-top: 0.25rem;">"Source: " <code>{s.clone()}</code></div> })}
                            {claim.content.license.as_ref().map(|l| view! { <div style="margin-top: 0.25rem;">"License: " {l.clone()}</div> })}
                        </div>
                    </div>
                </div>

                // Right column — LEM + metadata
                <div>
                    <LemCube empirical=2u8 normative=2u8 materiality=2u8 />

                    <div class="glass-panel" style="margin-top: 1rem;">
                        <h4 style="font-size: 0.875rem; margin-bottom: 0.5rem;">"Metadata"</h4>
                        <div style="font-size: 0.8rem; color: var(--text-secondary); display: flex; flex-direction: column; gap: 0.25rem;">
                            <div>"Creator: " <span style="color: var(--text-primary);">{claim.creator.clone()}</span></div>
                            <div>"Created: " {claim.created_at.chars().take(10).collect::<String>()}</div>
                            <div>"Verifications: " <span style="color: var(--accent-emerald);">{claim.verifications_count}</span></div>
                            {claim.content.reproducibility_score.map(|s| view! {
                                <div>"Reproducibility: " <span style="color: var(--accent-emerald);">{format!("{:.0}%", s * 100.0)}</span></div>
                            })}
                        </div>
                    // Fact-check from knowledge graph
                    {
                        let fc = crate::holochain::mock_fact_check(&claim.content.description);
                        view! {
                            <div class="glass-panel" style="margin-top: 1rem;">
                                <h4 style="font-size: 0.875rem; margin-bottom: 0.5rem;">"Knowledge Graph Fact-Check"</h4>
                                <div style=format!("font-size: 1.25rem; font-weight: 700; color: {};", fc.verdict.css_color())>
                                    {fc.verdict.label()}
                                </div>
                                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem;">
                                    {format!("Confidence: {:.0}% | {} supporting, {} contradicting",
                                        fc.confidence * 100.0, fc.supporting_claims, fc.contradicting_claims)}
                                </div>
                                <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.375rem;">
                                    {
                                        let status = crate::holochain::check_conductor_status();
                                        format!("Source: {}", status.label())
                                    }
                                </div>
                            </div>
                        }
                    }
                    </div>
                </div>
            </div>
        </div>
    }
}
