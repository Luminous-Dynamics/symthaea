// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Data Sovereignty Dashboard — see, understand, and control your data flows.
//!
//! Shows what data each installed cluster holds, how data flows between
//! clusters, and lets users configure sharing preferences.

use leptos::prelude::*;
use sensorium_domain_trait::EntryTypeInfo;

/// Data sovereignty dashboard.
#[component]
pub fn SovereigntyDashboard() -> impl IntoView {
    let registry = crate::build_registry();

    // Collect data flow information from domain dependencies and bridge connections
    let mut flows: Vec<FlowInfo> = Vec::new();
    let domain_ids: Vec<&str> = registry.domains.iter().map(|d| d.id()).collect();

    for domain in &registry.domains {
        for dep in domain.dependencies() {
            if domain_ids.contains(&dep.cluster_id) {
                flows.push(FlowInfo {
                    source: domain.id(),
                    target: dep.cluster_id,
                    reason: dep.reason,
                    required: dep.required,
                });
            }
        }
    }

    // Collect entry type info for data inventory
    let inventory: Vec<InventoryEntry> = registry
        .domains
        .iter()
        .map(|d| InventoryEntry {
            domain_name: d.name(),
            bio_name: d.bio_name(),
            color: d.color_family().primary,
            zome_count: d.zomes().len(),
            entry_types: d.entry_types().to_vec(),
        })
        .collect();

    let flow_count = flows.len();
    let domain_count = registry.domains.len();

    view! {
        <div class="sovereignty-page">
            <div class="sovereignty-header">
                <h1 class="sovereignty-title">"Data Sovereignty"</h1>
                <p class="sovereignty-subtitle">
                    "Your data, your rules. See what each domain holds and how data flows between them."
                </p>
            </div>

            // Summary stats
            <div class="sovereignty-stats">
                <div class="sov-stat">
                    <span class="sov-stat-value">{domain_count}</span>
                    <span class="sov-stat-label">"Active Domains"</span>
                </div>
                <div class="sov-stat">
                    <span class="sov-stat-value">{flow_count}</span>
                    <span class="sov-stat-label">"Data Flows"</span>
                </div>
                <div class="sov-stat">
                    <span class="sov-stat-value">"0"</span>
                    <span class="sov-stat-label">"Blocked Flows"</span>
                </div>
            </div>

            // Data Flow Graph
            <div class="sovereignty-section">
                <h2 class="section-title">"Data Flow Map"</h2>
                <p class="section-desc">
                    "Lines show which domains exchange data. "
                    "Solid lines are required dependencies, dashed are optional."
                </p>
                <FlowGraph domains=registry.domains.iter().map(|d| FlowNode {
                    id: d.id(),
                    name: d.bio_name(),
                    color: d.color_family().primary,
                    glow: d.color_family().glow,
                }).collect() flows=flows />
            </div>

            // Data Inventory
            <div class="sovereignty-section">
                <h2 class="section-title">"Data Inventory"</h2>
                <p class="section-desc">
                    "Entry types managed by each domain with their sensitivity classification."
                </p>
                <div class="inventory-grid">
                    {inventory.iter().map(|entry| {
                        view! { <InventoryCard entry=entry.clone() /> }
                    }).collect::<Vec<_>>()}
                </div>
            </div>

            // Sharing Controls
            <div class="sovereignty-section">
                <h2 class="section-title">"Sharing Controls"</h2>
                <p class="section-desc">
                    "Control which domains can share data with each other. "
                    "Blocked flows are enforced at the bridge layer — all frontends respect your choices."
                </p>
                <div class="sharing-placeholder">
                    <p>"All registered data flows are currently allowed. "</p>
                    <p class="sharing-note">"Per-flow toggle controls will be available when the data preferences zome is installed."</p>
                </div>
            </div>
        </div>
    }
}

#[derive(Clone, Debug)]
struct FlowInfo {
    source: &'static str,
    target: &'static str,
    reason: &'static str,
    required: bool,
}

#[derive(Clone, Debug)]
struct FlowNode {
    id: &'static str,
    name: &'static str,
    color: &'static str,
    glow: &'static str,
}

#[derive(Clone, Debug)]
struct InventoryEntry {
    domain_name: &'static str,
    bio_name: &'static str,
    color: &'static str,
    zome_count: usize,
    entry_types: Vec<EntryTypeInfo>,
}

#[component]
fn FlowGraph(domains: Vec<FlowNode>, flows: Vec<FlowInfo>) -> impl IntoView {
    let count = domains.len().max(1) as f64;
    let radius = 130.0;

    let positions: Vec<(f64, f64)> = (0..domains.len())
        .map(|i| {
            let angle = (i as f64) * std::f64::consts::TAU / count;
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();

    view! {
        <svg class="flow-graph" viewBox="-220 -220 440 440" aria-label="Data flow graph">
            // Flow edges
            {flows.iter().filter_map(|flow| {
                let src_idx = domains.iter().position(|d| d.id == flow.source)?;
                let tgt_idx = domains.iter().position(|d| d.id == flow.target)?;
                let (x1, y1) = positions[src_idx];
                let (x2, y2) = positions[tgt_idx];
                let color = if flow.required { "#ef4444" } else { "#64748b" };
                let dash = if flow.required { "" } else { "4 6" };
                Some(view! {
                    <line
                        x1=x1.to_string() y1=y1.to_string()
                        x2=x2.to_string() y2=y2.to_string()
                        stroke=color stroke-width="2" opacity="0.5"
                        stroke-dasharray=dash
                    >
                        <title>{format!("{} → {}: {}", flow.source, flow.target, flow.reason)}</title>
                    </line>
                })
            }).collect::<Vec<_>>()}

            // Domain nodes
            {domains.iter().enumerate().map(|(i, d)| {
                let (cx, cy) = positions[i];
                view! {
                    <g>
                        <circle cx=cx.to_string() cy=cy.to_string() r="22"
                            fill=d.color opacity="0.85" />
                        <circle cx=cx.to_string() cy=cy.to_string() r="22"
                            fill="none" stroke=d.glow stroke-width="1.5" opacity="0.6" />
                        <text x=cx.to_string() y=(cy + 4.0).to_string()
                            text-anchor="middle" fill="white" font-size="8" font-weight="600">
                            {d.name}
                        </text>
                    </g>
                }
            }).collect::<Vec<_>>()}
        </svg>
    }
}

#[component]
fn InventoryCard(entry: InventoryEntry) -> impl IntoView {
    let has_entry_types = !entry.entry_types.is_empty();

    view! {
        <div class="inventory-card" style=format!("--inv-color: {};", entry.color)>
            <div class="inv-header">
                <span class="inv-dot" />
                <div>
                    <span class="inv-name">{entry.domain_name}</span>
                    <span class="inv-bio">{entry.bio_name}</span>
                </div>
            </div>
            <div class="inv-stats">
                <span>{format!("{} zomes", entry.zome_count)}</span>
                <span>{format!("{} entry types declared", entry.entry_types.len())}</span>
            </div>
            {has_entry_types.then(|| view! {
                <div class="inv-entry-types">
                    {entry.entry_types.iter().map(|et| {
                        let sensitivity_class = et.sensitivity.css_class();
                        view! {
                            <div class="inv-entry-type">
                                <span class=format!("sensitivity-badge {}", sensitivity_class)>
                                    {et.sensitivity.label()}
                                </span>
                                <span class="et-label">{et.label}</span>
                                <span class="et-zome">{et.zome}</span>
                            </div>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            })}
        </div>
    }
}
