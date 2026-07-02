// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Search / address bar with vector visualizer, search history, and keyboard nav.

use leptos::prelude::*;
use prism_common::ContentZone;
use prism_reflex::ReflexArc;
use prism_search::SearchEngine;

use crate::components::vector_visualizer::VectorSpectrum;
use crate::engine;
use crate::state::BrowserState;

const MAX_HISTORY: usize = 20;
const HISTORY_KEY: &str = "prism-search-history";

/// Load search history from localStorage.
fn load_history() -> Vec<String> {
    crate::persistence::load::<Vec<String>>("search-history").unwrap_or_default()
}

/// Save a query to search history.
fn save_to_history(query: &str) {
    let mut history = load_history();
    // Remove duplicate if exists
    history.retain(|h| h != query);
    // Add to front
    history.insert(0, query.to_string());
    // Cap at MAX_HISTORY
    history.truncate(MAX_HISTORY);
    crate::persistence::save("search-history", &history);
}

#[component]
pub fn SearchBar() -> impl IntoView {
    let state = expect_context::<BrowserState>();
    let engine_cell = expect_context::<StoredValue<Option<SearchEngine>>>();
    let reflex = expect_context::<StoredValue<ReflexArc>>();

    let (input, set_input) = signal(String::new());
    let (compare_mode, set_compare_mode) = signal(false);
    let (show_history, set_show_history) = signal(false);
    let (history, set_history) = signal(load_history());
    let input_ref = NodeRef::<leptos::html::Input>::new();

    // Sync URL → input after navigation
    let state_url = state.clone();
    Effect::new(move |_| {
        set_input.set(state_url.current_url.get());
    });

    let state_submit = state.clone();
    let on_submit = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        let value = input.get();
        if value.is_empty() {
            return;
        }

        // Save to history (only for search queries, not URLs)
        if !engine::is_url(&value) {
            save_to_history(&value);
            set_history.set(load_history());
        }

        if compare_mode.get() && !engine::is_url(&value) {
            use crate::state::PageView;
            use leptos::prelude::Set;
            state_submit
                .set_current_url
                .set(format!("prism://compare?q={}", value));
            state_submit
                .set_page_title
                .set(format!("Compare: {}", value));
            state_submit
                .set_view
                .set(PageView::Compare { query: value });
        } else {
            engine_cell.with_value(|opt| {
                if let Some(search) = opt {
                    reflex.with_value(|r| {
                        engine::process_input(&value, &state_submit, search, r);
                    });
                }
            });
        }

        set_show_history.set(false);
        if let Some(el) = input_ref.get() {
            let _ = el.blur();
        }
    };

    let on_focus = move |_: leptos::ev::FocusEvent| {
        // Clear internal prism:// URLs on focus so the user sees the placeholder
        let current = input.get_untracked();
        if current.starts_with("prism://") {
            set_input.set(String::new());
        }
        if let Some(el) = input_ref.get() {
            let _ = el.select();
        }
        let h = load_history();
        if !h.is_empty() {
            set_history.set(h);
            set_show_history.set(true);
        }
    };

    let on_blur = move |_: leptos::ev::FocusEvent| {
        // Delay hiding so clicks on history items register
        gloo_timers::callback::Timeout::new(200, move || {
            set_show_history.set(false);
        })
        .forget();
    };

    // Keyboard navigation (#4)
    let state_key = state.clone();
    let on_keydown = move |ev: leptos::ev::KeyboardEvent| {
        match ev.key().as_str() {
            "Escape" => {
                set_show_history.set(false);
                if let Some(el) = input_ref.get() {
                    let _ = el.blur();
                }
                // Restore current URL
                set_input.set(state_key.current_url.get());
            }
            _ => {}
        }
    };

    let zone_class = move || match state.zone.get() {
        ContentZone::Private => "zone-badge private",
        ContentZone::Local => "zone-badge local",
        ContentZone::Public => "zone-badge public",
    };
    let zone_text = move || match state.zone.get() {
        ContentZone::Private => "Private",
        ContentZone::Local => "Local",
        ContentZone::Public => "Public",
    };

    // Filter history based on current input
    let filtered_history = move || {
        let q = input.get().to_lowercase();
        let h = history.get();
        if q.is_empty() {
            h
        } else {
            h.into_iter()
                .filter(|item| item.to_lowercase().contains(&q))
                .collect()
        }
    };

    view! {
        <span class=zone_class>{zone_text}</span>
        <form on:submit=on_submit style="flex:1; display:flex; position:relative;">
            <div class="search-wrapper">
                <input
                    node_ref=input_ref
                    class="search-input"
                    type="text"
                    placeholder="Search anything or enter a URL\u{2026}"
                    prop:value=input
                    on:input:target=move |ev| {
                        set_input.set(ev.target().value());
                        set_show_history.set(true);
                    }
                    on:focus=on_focus
                    on:blur=on_blur
                    on:keydown=on_keydown
                />
                <VectorSpectrum input=input />
            </div>

            // Search history dropdown
            <Show when=move || show_history.get() && !filtered_history().is_empty()>
                <div class="search-history">
                    {move || filtered_history().into_iter().map(|item| {
                        let item_click = item.clone();
                        let item_display = item.clone();
                        view! {
                            <div class="history-item"
                                on:mousedown=move |_| {
                                    set_input.set(item_click.clone());
                                    set_show_history.set(false);
                                }
                            >
                                <span class="history-icon">"↻"</span>
                                {item_display}
                            </div>
                        }
                    }).collect_view()}
                </div>
            </Show>
        </form>
        <button
            class="mode-toggle"
            attr:aria-label=move || format!("Search mode: {}", state.search_mode.get().label())
            on:click=move |_| {
                use crate::state::SearchMode;
                let current = state.search_mode.get();
                let next = match current {
                    SearchMode::Basic => SearchMode::Advanced,
                    SearchMode::Advanced => SearchMode::Paradigm,
                    SearchMode::Paradigm => SearchMode::Basic,
                };
                state.set_search_mode.set(next);
                crate::persistence::save("search-mode", &next.label().to_string());
            }
            title=move || state.search_mode.get().description()
        >
            {move || state.search_mode.get().label()}
        </button>
        <button
            class=move || if compare_mode.get() { "compare-toggle active" } else { "compare-toggle" }
            on:click=move |_| set_compare_mode.set(!compare_mode.get())
            title="Compare across multiple search engines"
            attr:aria-label="Toggle compare mode"
        >
            {move || if compare_mode.get() { "Compare \u{2713}" } else { "Compare" }}
        </button>
    }
}
