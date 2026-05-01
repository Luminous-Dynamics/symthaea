// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cluster catalog — browse, understand, and manage installed domains.
//!
//! Shows all known clusters with install status, dependencies, bridge
//! connections, and entry type sensitivity. The living marketplace
//! where communities choose which aspects of sovereignty to activate.

use leptos::prelude::*;
use sensorium_domain_trait::CivicTier;

/// Cluster catalog overview — rendered when user navigates to the catalog.
#[component]
pub fn CatalogOverview() -> impl IntoView {
    let registry = crate::build_registry();
    let identity = expect_context::<crate::identity::SensoriumIdentity>();
    let tier = identity.tier;

    // Build cluster cards from registry
    let clusters: Vec<ClusterCardData> = registry
        .domains
        .iter()
        .map(|d| {
            let deps: Vec<DepInfo> = d
                .dependencies()
                .iter()
                .map(|dep| {
                    let installed = registry.get(dep.cluster_id).is_some();
                    DepInfo {
                        id: dep.cluster_id,
                        required: dep.required,
                        available: installed,
                    }
                })
                .collect();

            ClusterCardData {
                id: d.id(),
                name: d.name(),
                bio_name: d.bio_name(),
                description: d.description(),
                color: d.color_family().primary,
                glow: d.color_family().glow,
                min_tier: d.min_tier(),
                zome_count: d.zomes().len(),
                entry_type_count: d.entry_types().len(),
                dependencies: deps,
            }
        })
        .collect();

    let cluster_count = clusters.len();

    view! {
        <div class="catalog-page">
            <div class="catalog-header">
                <h1 class="catalog-title">"Cluster Catalog"</h1>
                <p class="catalog-subtitle">
                    {format!("{} domains available — choose what aspects of sovereignty to activate", cluster_count)}
                </p>
            </div>

            // Dependency graph (SVG)
            <div class="catalog-graph">
                <h2 class="section-title">"Mycelial Connections"</h2>
                <DependencyGraph clusters=clusters.clone() />
            </div>

            // Cluster grid
            <div class="catalog-grid">
                {clusters.iter().map(|c| {
                    let accessible = c.min_tier <= tier.get_untracked();
                    view! { <ClusterCard data=c.clone() accessible=accessible /> }
                }).collect::<Vec<_>>()}
            </div>
        </div>
    }
}

#[derive(Clone, Debug)]
struct ClusterCardData {
    id: &'static str,
    name: &'static str,
    bio_name: &'static str,
    description: &'static str,
    color: &'static str,
    glow: &'static str,
    min_tier: CivicTier,
    zome_count: usize,
    entry_type_count: usize,
    dependencies: Vec<DepInfo>,
}

#[derive(Clone, Debug)]
struct DepInfo {
    id: &'static str,
    required: bool,
    available: bool,
}

#[component]
fn ClusterCard(data: ClusterCardData, accessible: bool) -> impl IntoView {
    let active_domain = expect_context::<RwSignal<Option<String>>>();

    let card_class = if accessible {
        "cluster-card accessible"
    } else {
        "cluster-card locked"
    };

    let dep_summary = if data.dependencies.is_empty() {
        "No dependencies".to_string()
    } else {
        let deps: Vec<String> = data
            .dependencies
            .iter()
            .map(|d| {
                let status = if d.available { "\u{2713}" } else { "\u{2717}" };
                format!("{} {}", status, d.id)
            })
            .collect();
        deps.join(", ")
    };

    let id = data.id.to_string();

    view! {
        <div
            class=card_class
            style=format!("--card-color: {}; --card-glow: {};", data.color, data.glow)
        >
            <div class="card-header">
                <span class="card-dot" />
                <div>
                    <h3 class="card-name">{data.name}</h3>
                    <span class="card-bio">{data.bio_name}</span>
                </div>
                <span class="card-tier">{data.min_tier.label()}</span>
            </div>

            {(!data.description.is_empty()).then(|| view! {
                <p class="card-description">{data.description}</p>
            })}

            <div class="card-stats">
                <div class="card-stat">
                    <span class="stat-value">{data.zome_count}</span>
                    <span class="stat-label">"zomes"</span>
                </div>
                <div class="card-stat">
                    <span class="stat-value">{data.entry_type_count}</span>
                    <span class="stat-label">"entry types"</span>
                </div>
                <div class="card-stat">
                    <span class="stat-value">{data.dependencies.len()}</span>
                    <span class="stat-label">"deps"</span>
                </div>
            </div>

            <div class="card-deps">
                <span class="deps-label">{dep_summary}</span>
            </div>

            {accessible.then(|| {
                let id = id.clone();
                view! {
                    <button class="card-enter" on:click=move |_| {
                        active_domain.set(Some(id.clone()));
                    }>"Enter"</button>
                }
            })}
        </div>
    }
}

/// SVG dependency graph showing connections between clusters.
#[component]
fn DependencyGraph(clusters: Vec<ClusterCardData>) -> impl IntoView {
    let count = clusters.len().max(1) as f64;
    let radius = 120.0;

    // Pre-compute everything before the view! macro to avoid ownership issues
    let positions: Vec<(f64, f64)> = (0..clusters.len())
        .map(|i| {
            let angle = (i as f64) * std::f64::consts::TAU / count;
            (radius * angle.cos(), radius * angle.sin())
        })
        .collect();

    // Pre-compute edge SVG data
    struct EdgeData {
        x1: f64,
        y1: f64,
        x2: f64,
        y2: f64,
        required: bool,
    }
    let mut edges: Vec<EdgeData> = Vec::new();
    for (i, cluster) in clusters.iter().enumerate() {
        for dep in &cluster.dependencies {
            if let Some(target_idx) = clusters.iter().position(|c| c.id == dep.id) {
                let (x1, y1) = positions[i];
                let (x2, y2) = positions[target_idx];
                edges.push(EdgeData {
                    x1,
                    y1,
                    x2,
                    y2,
                    required: dep.required,
                });
            }
        }
    }

    // Pre-compute node SVG data
    struct NodeData {
        cx: f64,
        cy: f64,
        color: &'static str,
        glow: &'static str,
        bio_name: &'static str,
    }
    let nodes: Vec<NodeData> = clusters
        .iter()
        .enumerate()
        .map(|(i, cluster)| {
            let (cx, cy) = positions[i];
            NodeData {
                cx,
                cy,
                color: cluster.color,
                glow: cluster.glow,
                bio_name: cluster.bio_name,
            }
        })
        .collect();

    view! {
        <svg class="dep-graph" viewBox="-200 -200 400 400" aria-label="Cluster dependency graph">
            {edges.iter().map(|e| {
                let color = if e.required { "#ef4444" } else { "#94a3b8" };
                let dash = if e.required { "" } else { "4 4" };
                view! {
                    <line
                        x1=e.x1.to_string() y1=e.y1.to_string()
                        x2=e.x2.to_string() y2=e.y2.to_string()
                        stroke=color stroke-width="1.5" opacity="0.4"
                        stroke-dasharray=dash
                    />
                }
            }).collect::<Vec<_>>()}

            {nodes.iter().map(|n| {
                view! {
                    <g>
                        <circle cx=n.cx.to_string() cy=n.cy.to_string() r="18"
                            fill=n.color opacity="0.8" />
                        <circle cx=n.cx.to_string() cy=n.cy.to_string() r="18"
                            fill="none" stroke=n.glow stroke-width="1" opacity="0.5" />
                        <text x=n.cx.to_string() y=(n.cy + 30.0).to_string()
                            text-anchor="middle" fill="currentColor" font-size="9" opacity="0.7">
                            {n.bio_name}
                        </text>
                    </g>
                }
            }).collect::<Vec<_>>()}
        </svg>
    }
}
