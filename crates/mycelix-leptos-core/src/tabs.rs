// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tab navigation component.

use leptos::prelude::*;

/// Tab bar with labeled buttons.
#[component]
pub fn Tabs(
    active: ReadSignal<usize>,
    on_change: Callback<usize>,
    #[prop(into)] labels: Vec<String>,
    children: Children,
) -> impl IntoView {
    view! {
        <div class="tabs">
            <div class="tab-bar" role="tablist">
                {labels.into_iter().enumerate().map(|(i, label)| {
                    view! {
                        <button
                            class=move || if active.get() == i { "tab-button tab-active" } else { "tab-button" }
                            role="tab"
                            aria-selected=move || (active.get() == i).to_string()
                            on:click=move |_| on_change.run(i)
                        >
                            {label}
                        </button>
                    }
                }).collect_view()}
            </div>
            <div class="tab-content">
                {children()}
            </div>
        </div>
    }
}

/// A tab panel that shows only when its index matches the active tab.
#[component]
pub fn TabPanel(
    index: usize,
    active: ReadSignal<usize>,
    children: Children,
) -> impl IntoView {
    view! {
        <div
            class="tab-panel"
            role="tabpanel"
            style=move || if active.get() == index { "display: block" } else { "display: none" }
        >
            {children()}
        </div>
    }
}
