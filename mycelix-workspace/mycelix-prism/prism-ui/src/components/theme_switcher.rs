// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Theme switcher: 8 visual themes aligned with the Harmonies.

use leptos::prelude::*;

const THEMES: &[(&str, &str, &str)] = &[
    ("abyss", "Abyssal Bioluminescence", "#4aecd8"),
    ("parchment", "Mycelial Parchment", "#8fbc8f"),
    ("prism", "Hyperdimensional Prism", "#a78bfa"),
    ("solar", "Solar Resonance", "#f59e0b"),
    ("verdant", "Verdant Canopy", "#22c55e"),
    ("arctic", "Arctic Aurora", "#38bdf8"),
    ("obsidian", "Obsidian Forge", "#ef4444"),
    ("lavender", "Lavender Twilight", "#c084fc"),
];

#[component]
pub fn ThemeSwitcher() -> impl IntoView {
    let (active, set_active) = signal("abyss".to_string());

    let switch = move |theme: &str| {
        let theme = theme.to_string();
        set_active.set(theme.clone());
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(html) = doc.document_element() {
                let _ = html.set_attribute("data-theme", &theme);
            }
        }
        // Persist to localStorage
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok())
            .flatten()
        {
            let _ = storage.set_item("prism-theme", &theme);
        }
    };

    // Load persisted theme on mount
    Effect::new(move |_| {
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok())
            .flatten()
        {
            if let Ok(Some(saved)) = storage.get_item("prism-theme") {
                set_active.set(saved.clone());
                if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                    if let Some(html) = doc.document_element() {
                        let _ = html.set_attribute("data-theme", &saved);
                    }
                }
            }
        }
    });

    view! {
        <div class="theme-switcher" title="Switch theme">
            {THEMES.iter().map(|(id, name, color)| {
                let id = id.to_string();
                let name = name.to_string();
                let color = color.to_string();
                let id_click = id.clone();
                let id_class = id.clone();
                view! {
                    <button
                        class=move || if active.get() == id_class { "theme-btn active" } else { "theme-btn" }
                        style:background=color.clone()
                        on:click=move |_| switch(&id_click)
                        title=name.clone()
                        attr:aria-label=format!("Theme: {}", name)
                    ></button>
                }
            }).collect_view()}
        </div>
    }
}
