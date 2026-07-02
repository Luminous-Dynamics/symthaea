// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::hooks::use_location;

/// Top navigation bar with active page highlighting.
#[component]
pub fn NavBar() -> impl IntoView {
    let location = use_location();
    let current_path = move || location.pathname.get();

    let nav_link = move |href: &'static str, label: &'static str| {
        let is_active = move || {
            let path = current_path();
            if href == "/" { path == "/" } else { path.starts_with(href) }
        };
        view! {
            <a href=href
                style=move || if is_active() { "color: var(--text-primary); background: rgba(99,102,241,0.15);" } else { "" }
            >{label}</a>
        }
    };

    view! {
        <nav class="navbar">
            <a href="/" class="logo">"DeSci"</a>
            <div class="nav-links">
                {nav_link("/", "Dashboard")}
                {nav_link("/browse", "Claims")}
                {nav_link("/discovery", "Discovery")}
                {nav_link("/markets", "Markets")}
                {nav_link("/reproducibility", "Repro")}
                {nav_link("/citations", "Citations")}
                {nav_link("/consensus", "Consensus")}
                {nav_link("/case-studies", "Cases")}
                {nav_link("/submit", "Submit")}
            </div>
        </nav>
    }
}
