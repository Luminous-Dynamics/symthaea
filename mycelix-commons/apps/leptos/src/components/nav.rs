// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use leptos_router::components::A;

#[component]
pub fn Nav() -> impl IntoView {
    view! {
        <nav class="navbar">
            <a href="/" class="logo">"Mycelix Commons"</a>
            <div class="nav-links">
                <A href="/property">"Property"</A>
                <A href="/housing">"Housing"</A>
                <A href="/care">"Care"</A>
                <A href="/resources">"Resources"</A>
                <A href="/transport">"Transport"</A>
            </div>
        </nav>
    }
}
