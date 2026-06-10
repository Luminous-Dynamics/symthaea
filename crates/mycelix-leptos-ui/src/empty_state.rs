// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Empty state placeholder component.

use leptos::prelude::*;

/// Empty state for lists/pages with no data.
#[component]
pub fn EmptyState(
    #[prop(into)] icon: String,
    #[prop(into)] title: String,
    #[prop(optional, into)] description: Option<String>,
    #[prop(optional, into)] action_label: Option<String>,
    #[prop(optional)] on_action: Option<Callback<()>>,
) -> impl IntoView {
    view! {
        <div class="empty-state">
            <div class="empty-state-icon">{icon}</div>
            <h3 class="empty-state-title">{title}</h3>
            {description.map(|d| view! { <p class="empty-state-desc">{d}</p> })}
            {action_label.zip(on_action).map(|(label, cb)| view! {
                <button class="btn btn-primary" on:click=move |_| cb.run(())>
                    {label}
                </button>
            })}
        </div>
    }
}
