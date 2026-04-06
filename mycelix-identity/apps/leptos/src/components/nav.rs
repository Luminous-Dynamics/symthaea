// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;

use mycelix_leptos_core::{ConnectionStatus, holochain_provider::use_holochain};

#[component]
pub fn Nav() -> impl IntoView {
    let hc = use_holochain();

    let status_label = move || match hc.status.get() {
        ConnectionStatus::Connected => "Live",
        ConnectionStatus::Connecting => "Connecting",
        ConnectionStatus::Mock => "Mock",
        ConnectionStatus::Disconnected => "Offline",
    };

    let status_class = move || match hc.status.get() {
        ConnectionStatus::Connected => "status-dot connected",
        ConnectionStatus::Connecting => "status-dot connecting",
        _ => "status-dot mock",
    };

    view! {
        <nav class="navbar" aria-label="Primary navigation">
            <a href="/" class="logo">"Identity"</a>
            <div class="nav-links">
                <A href="/">"Overview"</A>
                <A href="/did">"DID"</A>
                <A href="/mfa">"MFA"</A>
                <A href="/credentials">"Credentials"</A>
                <A href="/recovery">"Recovery"</A>
                <A href="/trust">"Trust"</A>
            </div>
            <div class="nav-status">
                <span class={status_class} />
                <span class="status-label">{status_label}</span>
            </div>
        </nav>
    }
}
