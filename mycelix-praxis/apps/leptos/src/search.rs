// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Offline Semantic Search — Local-WASM Vector Intent Matching.

use leptos::prelude::*;
use crate::curriculum::curriculum_graph;

#[component]
pub fn SearchBar() -> impl IntoView {
    let (query, set_query) = signal("".to_string());
    let (is_thinking, set_is_thinking) = signal(false);

    let perform_semantic_search = move |_| {
        set_is_thinking.set(true);
        // SIMULATED: WASM-based HDC Vector Similarity Match
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;
            set_is_thinking.set(false);
            // Result would be a set of IDs ranked by cosine similarity
        });
    };

    view! {
        <div class="search-container" style="position: relative; width: 100%; max-width: 400px">
            <input 
                type="text" 
                placeholder="Search by intent (e.g. 'growing food')..."
                style="width: 100%; padding: 0.5rem 1rem; border-radius: 20px; border: 1px solid var(--border); background: var(--surface)"
                prop:value=query
                on:input=move |ev| set_query.set(event_target_value(&ev))
                on:keydown=move |ev| if ev.key() == "Enter" { perform_semantic_search(ev); }
            />
            {move || if is_thinking.get() {
                view! { <div class="search-spinner" style="position: absolute; right: 10px; top: 10px">"\u{231B}"</div> }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
        </div>
    }
}
