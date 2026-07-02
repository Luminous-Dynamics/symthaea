// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Council tree: holonic visualization of governance structure.
//! Each council is a living node. Non-recursive rendering (flat with indent).
//! AI-interactable via data-council-id, data-type, data-parent attributes.

use leptos::prelude::*;
use crate::contexts::governance_context::use_governance;
use governance_leptos_types::*;

#[component]
pub fn CouncilsPage() -> impl IntoView {
    let gov = use_governance();

    view! {
        <div class="councils-page" data-page="councils" role="main">
            <h1 class="page-title">"Council Tree"</h1>
            <p class="page-subtitle">"Holonic governance — nested circles of care and stewardship"</p>

            <div
                class="council-tree"
                role="tree"
                aria-label="governance council hierarchy"
                data-component="council-tree"
            >
                {move || {
                    let all = gov.councils.get();
                    if all.is_empty() {
                        view! {
                            <p class="empty-state">"no councils have formed yet"</p>
                        }.into_any()
                    } else {
                        // Render root councils, then children indented
                        let mut nodes = Vec::new();
                        let roots: Vec<_> = all.iter()
                            .filter(|c| c.parent_council_id.is_none())
                            .collect();

                        for root in &roots {
                            nodes.push(render_council_node(root, 0));
                            for child in all.iter().filter(|c| c.parent_council_id.as_deref() == Some(&root.id)) {
                                nodes.push(render_council_node(child, 1));
                            }
                        }

                        nodes.collect_view().into_any()
                    }
                }}
            </div>
        </div>
    }
}

fn render_council_node(council: &CouncilView, depth: u32) -> impl IntoView {
    let council_id = council.id.clone();
    let name = council.name.clone();
    let purpose = council.purpose.clone();
    let type_label = council.council_type.label();
    let type_label2 = type_label.clone();
    let member_count = council.member_count;
    let phi_threshold = council.phi_threshold;
    let parent = council.parent_council_id.clone().unwrap_or_default();
    let indent = format!("padding-left: {}rem", depth as f64 * 1.5);

    view! {
        <div
            class="council-node"
            role="treeitem"
            data-council-id=council_id
            data-council-type=type_label
            data-member-count=member_count.to_string()
            data-phi-threshold=format!("{phi_threshold:.2}")
            data-parent=parent
            data-depth=depth.to_string()
            style=indent
        >
            <div class="council-header">
                <div class="council-identity">
                    <h3 class="council-name">{name}</h3>
                    <span class="council-type-badge">{type_label2}</span>
                </div>
                <div class="council-vitals">
                    <span class="council-members" data-metric="members">
                        {format!("{member_count} members")}
                    </span>
                    <span class="council-phi" data-metric="phi-threshold">
                        {format!("φ ≥ {phi_threshold:.2}")}
                    </span>
                </div>
            </div>
            <p class="council-purpose">{purpose}</p>
        </div>
    }
}
