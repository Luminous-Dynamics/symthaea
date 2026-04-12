// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Curriculum search — find topics across 2,002 nodes instantly.

use leptos::prelude::*;

use crate::curriculum::curriculum_graph;

/// Search bar component — renders in the navbar or as a standalone widget.
#[component]
pub fn SearchBar() -> impl IntoView {
    let (query, set_query) = signal(String::new());
    let (focused, set_focused) = signal(false);

    let results = Memo::new(move |_| {
        let q = query.get();
        if q.len() < 2 {
            return vec![];
        }
        let q_lower = q.to_lowercase();
        let graph = curriculum_graph();
        let mut matches: Vec<(String, String, String, String)> = graph
            .nodes
            .iter()
            .filter(|n| {
                n.title.to_lowercase().contains(&q_lower)
                    || n.subject_area.to_lowercase().contains(&q_lower)
                    || n.subdomain.to_lowercase().contains(&q_lower)
                    || n.id.to_lowercase().contains(&q_lower)
            })
            .take(8)
            .map(|n| {
                let grade = n.grade_levels.first().cloned().unwrap_or_default();
                (n.id.clone(), n.title.clone(), n.subject_area.clone(), grade)
            })
            .collect();
        matches.sort_by(|a, b| a.1.len().cmp(&b.1.len())); // shorter titles first (more relevant)
        matches
    });

    view! {
        <div class="search-container" style=move || if focused.get() { "flex: 1; max-width: 400px" } else { "flex: 0; max-width: 200px" }>
            <input
                type="search"
                placeholder="Search topics..."
                class="search-input"
                autocomplete="off"
                prop:value=move || query.get()
                on:input=move |ev| set_query.set(leptos::prelude::event_target_value(&ev))
                on:focus=move |_| set_focused.set(true)
                on:blur=move |_| {
                    // Delay to allow click on results
                    wasm_bindgen_futures::spawn_local(async move {
                        gloo_timers::future::sleep(std::time::Duration::from_millis(200)).await;
                        set_focused.set(false);
                    });
                }
            />

            // Results dropdown
            {move || {
                let r = results.get();
                if r.is_empty() || !focused.get() {
                    return view! { <span></span> }.into_any();
                }
                view! {
                    <div class="search-results">
                        {r.iter().map(|(id, title, subject, grade)| {
                            let href = format!("/study/{}", id);
                            let title = title.clone();
                            let subject = subject.clone();
                            let grade = grade.clone();
                            view! {
                                <a href=href class="search-result-item">
                                    <div class="search-result-title">{title}</div>
                                    <div class="search-result-meta">{subject}" \u{00B7} "{grade}</div>
                                </a>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                }.into_any()
            }}
        </div>
    }
}
