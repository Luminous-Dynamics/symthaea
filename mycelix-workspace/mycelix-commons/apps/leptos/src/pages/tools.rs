// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::contexts::commons_context::use_commons;

#[component]
pub fn ToolsPage() -> impl IntoView {
    let commons = use_commons();
    let available_count = Memo::new(move |_| commons.tools.get().iter().filter(|t| t.available).count());

    view! {
        <div class="tools-page" data-page="tools" role="main">
            <h1 class="page-title">"Tool Library"</h1>
            <p class="page-subtitle">{move || format!("{} tools available — why own when you can share?", available_count.get())}</p>
            <div class="tools-grid" data-section="tools" role="list">
                {move || commons.tools.get().into_iter().map(|t| {
                    let cat = t.category.label().to_string();
                    let cat2 = cat.clone();
                    let cond = t.condition.label().to_string();
                    let owner = t.owner_did.split(':').last().unwrap_or(&t.owner_did).to_string();
                    let avail_class = if t.available { "tool-available" } else { "tool-borrowed" };
                    view! {
                        <div class=format!("tool-card {avail_class}") data-tool-hash=t.hash.clone() data-category=cat data-available=t.available.to_string() role="listitem">
                            <div class="tool-header">
                                <h3 class="tool-name">{t.name}</h3>
                                <span class="tool-status">{if t.available { "available" } else { "borrowed" }}</span>
                            </div>
                            <p class="tool-desc">{t.description}</p>
                            <div class="tool-meta">
                                <span>{cat2}</span>
                                <span>{cond}</span>
                                <span>{format!("owner: {owner}")}</span>
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
