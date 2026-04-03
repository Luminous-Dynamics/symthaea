// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::components::A;

#[component]
pub fn Nav() -> impl IntoView {
    view! {
        <nav class="navbar" aria-label="Primary navigation">
            <a href="/" class="logo">"Mycelix Music"</a>
            <div class="nav-links">
                <A href="/">"Synthesize"</A>
                <A href="/">"Synthesize"</A>
                <A href="/discover">"Discover"</A>
                <A href="/gallery">"Gallery"</A>
                <A href="/dashboard">"Dashboard"</A>
                <A href="/upload">"Contribute"</A>
                <A href="/artist">"For Artists"</A>
            </div>
        </nav>
    }
}
