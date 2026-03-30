// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};

#[component]
pub fn ResourcesPage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page resources-page">
            <h1 class="page-title">"shared resources"</h1>
            <p class="page-subtitle">"inventory, loans, and budgets"</p>

            {move || {
                let members = hearth.members.get();
                let resources = hearth.resources.get();

                if resources.is_empty() {
                    view! { <div class="empty-state">"what do you share? tools, books, a kitchen — the commons begins with one offer."</div> }.into_any()
                } else {
                    view! {
                        <div class="resource-grid">
                            {resources.iter().map(|r| {
                                let name = r.name.clone();
                                let desc = r.description.clone();
                                let rtype = format!("{:?}", r.resource_type);
                                let holder = r.current_holder.as_ref()
                                    .map(|h| member_name(&members, h))
                                    .unwrap_or_else(|| "Available".into());
                                let location = r.location.clone();
                                view! {
                                    <div class="resource-card">
                                        <div class="resource-header">
                                            <h3>{name}</h3>
                                            <span class="resource-type">{rtype}</span>
                                        </div>
                                        <p class="resource-desc">{desc}</p>
                                        <div class="resource-footer">
                                            <span class="resource-holder">{holder}</span>
                                            <span class="resource-location">{location}</span>
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
