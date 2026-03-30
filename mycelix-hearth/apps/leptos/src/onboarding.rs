// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! First-visit onboarding: a gentle welcome that explains what hearth is.
//!
//! Checks localStorage for "hearth-welcomed". If absent, shows a
//! translucent overlay with a brief introduction and an invitation
//! to explore. Dismisses on click and sets the flag.

use leptos::prelude::*;

fn has_been_welcomed() -> bool {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("hearth-welcomed").ok().flatten())
        .is_some()
}

fn mark_welcomed() {
    if let Some(storage) = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
    {
        let _ = storage.set_item("hearth-welcomed", "true");
    }
}

#[component]
pub fn OnboardingOverlay() -> impl IntoView {
    let (show, set_show) = signal(!has_been_welcomed());

    let dismiss = move |_| {
        mark_welcomed();
        set_show.set(false);
    };

    let show_style = move || {
        if show.get() { "display: flex" } else { "display: none" }
    };

    view! {
        <div class="onboarding-overlay" style=show_style on:click=dismiss role="dialog" aria-label="welcome to hearth">
            <div class="onboarding-card" on:click=|ev| ev.stop_propagation()>
                <h2 class="onboarding-title">"welcome to the hearth"</h2>
                <p class="onboarding-body">
                    "this is a living space for your family. not a dashboard — a home."
                </p>
                <p class="onboarding-body">
                    "the web above shows your bonds. the brighter the thread, the stronger the connection. "
                    "when bonds dim, they need tending — click them to send warmth."
                </p>
                <p class="onboarding-body">
                    "express gratitude. share stories. make decisions together. "
                    "the hearth breathes with you."
                </p>
                <button class="onboarding-enter" on:click=dismiss>"enter the hearth"</button>
            </div>
        </div>
    }
}
