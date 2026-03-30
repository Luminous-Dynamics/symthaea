// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use leptos_router::components::A;
use crate::holochain::ConnectionBadge;
use crate::consciousness_provider::use_consciousness;
use crate::themes::use_theme;
use personal_leptos_types::TrustTier;

#[component]
pub fn Nav() -> impl IntoView {
    let consciousness = use_consciousness();
    let _theme_state = use_theme();

    let show_governance = move || consciousness.tier.get() >= TrustTier::Basic;
    let show_emergency = move || consciousness.tier.get() >= TrustTier::Observer;

    view! {
        <nav class="navbar" role="navigation" aria-label="main navigation">
            <A href="/" attr:class="logo" attr:aria-label="hearth home">"hearth"</A>

            <div class="nav-links">
                <A href="/kinship">"bonds"</A>
                <A href="/care">"care"</A>
                <A href="/gratitude">"gratitude"</A>
                <A href="/stories">"stories"</A>
                <A href="/rhythms">"rhythms"</A>
                {move || show_governance().then(|| view! {
                    <A href="/decisions">"decisions"</A>
                })}
                {move || show_emergency().then(|| view! {
                    <A href="/emergency">"emergency"</A>
                })}
                <span class="nav-divider">"|"</span>
                <A href="/personal/profile">"vault"</A>
            </div>

            <div class="nav-actions">
                <SoundToggle />
                <ThemeSwitcher />
                <ConnectionBadge />
            </div>
        </nav>
    }
}

/// Ambient sound toggle.
#[component]
fn SoundToggle() -> impl IntoView {
    let sound = crate::ambient_sound::use_ambient_sound();

    view! {
        <button
            class="sound-toggle"
            title=move || if sound.enabled.get() { "mute ambient" } else { "ambient sound" }
            on:click=move |_| sound.enabled.set(!sound.enabled.get())
        >
            {move || if sound.enabled.get() { "\u{266b}" } else { "\u{266a}" }}
        </button>
    }
}

/// Single button that cycles through themes on click.
#[component]
fn ThemeSwitcher() -> impl IntoView {
    let theme_state = use_theme();

    view! {
        <button
            class="theme-cycle-btn"
            title=move || {
                let t = theme_state.current.get();
                format!("{} \u{2014} click to change", t.description())
            }
            on:click=move |_| {
                let next = theme_state.current.get().next();
                theme_state.current.set(next);
            }
        >
            <span class="theme-cycle-dot"></span>
            <span class="theme-cycle-label">{move || theme_state.current.get().label()}</span>
        </button>
    }
}
