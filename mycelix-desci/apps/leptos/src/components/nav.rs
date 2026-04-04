// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

/// Top navigation bar.
#[component]
pub fn NavBar() -> impl IntoView {
    view! {
        <nav class="navbar">
            <a href="/" class="logo">"DeSci"</a>
            <div class="nav-links">
                <a href="/">"Dashboard"</a>
                <a href="/browse">"Browse"</a>
                <a href="/submit">"Submit"</a>
                <a href="/discovery">"Discovery"</a>
                <a href="/trust">"Trust"</a>
                <a href="/about">"About"</a>
            </div>
        </nav>
    }
}
