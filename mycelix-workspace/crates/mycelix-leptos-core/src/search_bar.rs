// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Search bar with optional debounce.

use leptos::prelude::*;

/// Search bar input with debounced callback.
#[component]
pub fn SearchBar(
    query: ReadSignal<String>,
    on_change: Callback<String>,
    #[prop(optional, default = "Search...")] placeholder: &'static str,
) -> impl IntoView {
    view! {
        <div class="search-bar">
            <input
                class="form-input search-input"
                type="search"
                placeholder=placeholder
                prop:value=move || query.get()
                on:input=move |ev| {
                    on_change.run(event_target_value(&ev));
                }
            />
        </div>
    }
}
