// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Citation explorer — knowledge graph visualization.

use leptos::prelude::*;
use crate::types::*;

fn demo_citations() -> Vec<CitationMetrics> {
    vec![
        CitationMetrics { claim_id: "newton-gravitation".into(), pagerank: 0.92, h_index: 15, citation_count: 450, is_foundational: true },
        CitationMetrics { claim_id: "einstein-field-equations".into(), pagerank: 0.88, h_index: 12, citation_count: 380, is_foundational: true },
        CitationMetrics { claim_id: "schrodinger-equation".into(), pagerank: 0.85, h_index: 11, citation_count: 340, is_foundational: true },
        CitationMetrics { claim_id: "shannon-entropy".into(), pagerank: 0.78, h_index: 9, citation_count: 280, is_foundational: true },
        CitationMetrics { claim_id: "island-stability-z120".into(), pagerank: 0.15, h_index: 0, citation_count: 3, is_foundational: false },
        CitationMetrics { claim_id: "lazar-gravity-a".into(), pagerank: 0.02, h_index: 0, citation_count: 0, is_foundational: false },
        CitationMetrics { claim_id: "arts-parts-waveguide".into(), pagerank: 0.05, h_index: 0, citation_count: 1, is_foundational: false },
    ]
}

fn demo_links() -> Vec<Citation> {
    vec![
        Citation { from_claim: "island-stability-z120".into(), to_claim: "nuclear-radius-formula".into(), citation_type: "Methodological".into(), strength: 0.9 },
        Citation { from_claim: "island-stability-z120".into(), to_claim: "geiger-nuttall-law".into(), citation_type: "Supporting".into(), strength: 0.8 },
        Citation { from_claim: "arts-parts-waveguide".into(), to_claim: "waveguide-dispersion".into(), citation_type: "Supporting".into(), strength: 0.95 },
        Citation { from_claim: "lazar-gravity-a".into(), to_claim: "yukawa-potential".into(), citation_type: "Extending".into(), strength: 0.6 },
    ]
}

#[component]
pub fn CitationsPage() -> impl IntoView {
    let metrics = demo_citations();
    let links = demo_links();
    let foundational: Vec<_> = metrics.iter().filter(|m| m.is_foundational).collect();
    let recent: Vec<_> = metrics.iter().filter(|m| !m.is_foundational).collect();

    view! {
        <div class="page-container">
            <h1 class="page-title">"Citation Explorer"</h1>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem;">
                // Foundational claims
                <div class="glass-panel">
                    <h3 style="font-size: 1rem; margin-bottom: 0.75rem; color: var(--accent-indigo);">"Foundational Claims"</h3>
                    {foundational.iter().map(|m| {
                        let pr_width = format!("{}%", (m.pagerank * 100.0) as u32);
                        view! {
                            <div style="margin-bottom: 0.75rem;">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.25rem;">
                                    <span style="font-weight: 600;">{m.claim_id.clone()}</span>
                                    <span style="color: var(--accent-emerald);">{format!("PR: {:.2}", m.pagerank)}</span>
                                </div>
                                <div style="height: 4px; background: var(--bg-secondary); border-radius: 2px;">
                                    <div style=format!("width: {}; height: 100%; background: var(--accent-indigo); border-radius: 2px;", pr_width)></div>
                                </div>
                                <div style="font-size: 0.7rem; color: var(--text-secondary); margin-top: 0.125rem;">
                                    {format!("h-index: {} | {} citations", m.h_index, m.citation_count)}
                                </div>
                            </div>
                        }
                    }).collect::<Vec<_>>()}
                </div>

                // Recent claims
                <div class="glass-panel">
                    <h3 style="font-size: 1rem; margin-bottom: 0.75rem; color: var(--accent-emerald);">"Recent Claims"</h3>
                    {recent.iter().map(|m| view! {
                        <div style="padding: 0.5rem; background: var(--bg-secondary); border-radius: 0.375rem; margin-bottom: 0.5rem;">
                            <span style="font-size: 0.8rem; font-weight: 600;">{m.claim_id.clone()}</span>
                            <span style="font-size: 0.7rem; color: var(--text-secondary); margin-left: 0.5rem;">
                                {format!("PR: {:.2} | {} citations", m.pagerank, m.citation_count)}
                            </span>
                        </div>
                    }).collect::<Vec<_>>()}
                </div>
            </div>

            // Citation links
            <div class="glass-panel">
                <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">"Citation Links"</h3>
                {links.iter().map(|l| view! {
                    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0; border-bottom: 1px solid var(--border-glass); font-size: 0.8rem;">
                        <span style="font-weight: 600; color: var(--accent-indigo);">{l.from_claim.clone()}</span>
                        <span style="color: var(--text-secondary);">"→"</span>
                        <span>{l.to_claim.clone()}</span>
                        <span style="margin-left: auto; font-size: 0.7rem; padding: 0.125rem 0.375rem; background: var(--bg-secondary); border-radius: 9999px;">{l.citation_type.clone()}</span>
                        <span style="font-size: 0.75rem; color: var(--accent-emerald);">{format!("{:.0}%", l.strength * 100.0)}</span>
                    </div>
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}
