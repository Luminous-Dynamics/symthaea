// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Tab navigation component with keyboard support.

use leptos::prelude::*;

/// Tab bar with labeled buttons. Supports arrow key navigation.
#[component]
pub fn Tabs(
    active: ReadSignal<usize>,
    on_change: Callback<usize>,
    #[prop(into)] labels: Vec<String>,
    children: Children,
) -> impl IntoView {
    let count = labels.len();

    view! {
        <div class="tabs">
            <div
                class="tab-bar"
                role="tablist"
                on:keydown=move |ev| {
                    let current = active.get();
                    match ev.key().as_str() {
                        "ArrowRight" | "ArrowDown" => {
                            ev.prevent_default();
                            on_change.run((current + 1) % count);
                        }
                        "ArrowLeft" | "ArrowUp" => {
                            ev.prevent_default();
                            on_change.run(if current == 0 { count - 1 } else { current - 1 });
                        }
                        "Home" => { ev.prevent_default(); on_change.run(0); }
                        "End" => { ev.prevent_default(); on_change.run(count - 1); }
                        _ => {}
                    }
                }
            >
                {labels.into_iter().enumerate().map(|(i, label)| {
                    let panel_id = format!("tabpanel-{i}");
                    view! {
                        <button
                            class=move || if active.get() == i { "tab-button tab-active" } else { "tab-button" }
                            role="tab"
                            aria-selected=move || (active.get() == i).to_string()
                            aria-controls=panel_id
                            tabindex=move || if active.get() == i { "0" } else { "-1" }
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
pub fn TabPanel(index: usize, active: ReadSignal<usize>, children: Children) -> impl IntoView {
    let id = format!("tabpanel-{index}");
    view! {
        <div
            class="tab-panel"
            role="tabpanel"
            id=id
            tabindex="0"
            style=move || if active.get() == index { "display: block" } else { "display: none" }
        >
            {children()}
        </div>
    }
}
