// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! First-visit welcome modal — onboarding flow for new users.

use leptos::prelude::*;

const SEEN_KEY: &str = "mycelix_pulse_welcomed";

#[component]
pub fn WelcomeModal() -> impl IntoView {
    // On production (mycelix.net), skip welcome — ProfileSetup handles onboarding
    let is_production = web_sys::window()
        .and_then(|w| w.location().hostname().ok())
        .map(|h| h.contains("mycelix.net") || h == "localhost" || h == "127.0.0.1")
        .unwrap_or(false);

    // Check if user has seen the welcome before
    let already_seen = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(SEEN_KEY).ok().flatten())
        .is_some();

    let show = RwSignal::new(!already_seen && !is_production);

    let do_dismiss = move || {
        show.set(false);
        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = storage.set_item(SEEN_KEY, "1");
        }
        // Start the onboarding tour after dismissing
        crate::components::tour::start_tour();
    };

    let on_explore = move |_: leptos::ev::MouseEvent| { do_dismiss(); };
    let on_explore_touch = move |ev: web_sys::TouchEvent| { ev.prevent_default(); do_dismiss(); };
    let on_learn = move |_: leptos::ev::MouseEvent| {
        do_dismiss();
        let nav = leptos_router::hooks::use_navigate();
        nav("/settings", Default::default());
    };
    let on_learn_touch = move |ev: web_sys::TouchEvent| {
        ev.prevent_default();
        do_dismiss();
        let nav = leptos_router::hooks::use_navigate();
        nav("/settings", Default::default());
    };

    view! {
        <div class="welcome-overlay" style=move || if show.get() { "" } else { "display:none" }>
            <div class="welcome-modal">
                <div class="welcome-header">
                    <span class="welcome-icon">"\u{2709}"</span>
                    <h1>"Welcome to Mycelix Pulse"</h1>
                    <p class="welcome-subtitle">"Communication that protects you"</p>
                </div>

                <div class="welcome-features">
                    <div class="welcome-feature">
                        <span class="wf-icon">"\u{1F512}"</span>
                        <div>
                            <strong>"End-to-end encrypted"</strong>
                            <p>"Post-quantum cryptography. Not even we can read your messages."</p>
                        </div>
                    </div>
                    <div class="welcome-feature">
                        <span class="wf-icon">"\u{1F310}"</span>
                        <div>
                            <strong>"Decentralized"</strong>
                            <p>"No central server. Your data lives on your device and the peer network."</p>
                        </div>
                    </div>
                    <div class="welcome-feature">
                        <span class="wf-icon">"\u{1F9E0}"</span>
                        <div>
                            <strong>"Semantic intelligence"</strong>
                            <p>"On-device AI understands your emails. Search by meaning, not just keywords."</p>
                        </div>
                    </div>
                </div>

                <div class="welcome-actions">
                    <button class="btn btn-primary welcome-btn"
                            on:click=on_explore on:touchend=on_explore_touch
                            style="touch-action: manipulation">
                        "\u{1F680} Explore Demo"
                    </button>
                </div>

                <p class="welcome-hint">
                    "Demo mode lets you explore all features with sample data. "
                    "Connect a Holochain conductor for real encrypted messaging."
                </p>
            </div>
        </div>
    }
}
