// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shared inline SVG icon set — replaces the ad hoc `♡`/`♥` glyphs with a
//! consistent visual language: 2px stroke, rounded caps/joins, 24x24
//! viewBox, matching the "Line 2px · Rounded" icon style called out in
//! `UI_mocks/ChatGPT Image Jul 13, 2026, 03_21_05 PM.png`'s own design
//! system footer. Kept as plain inline SVG (no icon font/sprite sheet) so
//! there's no extra asset to bundle or fail to load.

use leptos::prelude::*;

const STROKE: &str = "currentColor";

#[component]
pub fn PlayIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="20" height="20" fill="none">
            <path
                d="M7 5.5v13a1 1 0 0 0 1.53.85l10.2-6.5a1 1 0 0 0 0-1.7l-10.2-6.5A1 1 0 0 0 7 5.5Z"
                fill=STROKE
            />
        </svg>
    }
}

#[component]
pub fn PauseIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="20" height="20" fill="none">
            <rect x="6.5" y="5" width="4" height="14" rx="1.5" fill=STROKE />
            <rect x="13.5" y="5" width="4" height="14" rx="1.5" fill=STROKE />
        </svg>
    }
}

#[component]
pub fn NextIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path
                d="M5 6v12l9-6-9-6Z"
                fill=STROKE
            />
            <rect x="16" y="6" width="2.2" height="12" rx="1" fill=STROKE />
        </svg>
    }
}

/// Restarts the current piece from the beginning — the honest behavior for
/// a "previous" control given Muse has no real play-history/queue-back
/// concept (each piece is composed forward, not stored in a navigable
/// list). Labeled/titled as "Restart" wherever it's used, not "Previous".
#[component]
pub fn RestartIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path
                d="M19 6v12l-9-6 9-6Z"
                fill=STROKE
            />
            <rect x="5.8" y="6" width="2.2" height="12" rx="1" fill=STROKE />
        </svg>
    }
}

#[component]
pub fn HeartIcon(#[prop(default = false)] filled: bool) -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path
                d="M12 20.2s-7.2-4.6-9.6-9.3C.8 7.9 2.3 4.6 5.6 3.8c2-.5 4 .3 5.2 2 .3.4.9.4 1.2 0 1.2-1.7 3.2-2.5 5.2-2 3.3.8 4.8 4.1 3.2 7.1-2.4 4.7-9.6 9.3-9.6 9.3Z"
                stroke=STROKE
                stroke-width="1.8"
                stroke-linejoin="round"
                fill=move || if filled { STROKE } else { "none" }
            />
        </svg>
    }
}

/// The Radial ("Journey") visualization mode.
#[component]
pub fn RadialIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none">
            <circle cx="12" cy="12" r="3" fill=STROKE />
            <circle
                cx="12"
                cy="12"
                r="8.5"
                stroke=STROKE
                stroke-width="1.6"
                stroke-dasharray="2.4 3.2"
            />
        </svg>
    }
}

/// The Bars ("Activity") visualization mode.
#[component]
pub fn BarsIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none">
            <rect x="4" y="12" width="3" height="8" rx="1.2" fill=STROKE />
            <rect x="10.5" y="6" width="3" height="14" rx="1.2" fill=STROKE />
            <rect x="17" y="9" width="3" height="11" rx="1.2" fill=STROKE />
        </svg>
    }
}

/// The Waves ("Resonance") visualization mode.
#[component]
pub fn WavesIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none">
            <path
                d="M3 12c1.5-3 3-3 4.5 0s3 3 4.5 0 3-3 4.5 0 3 3 4.5 0"
                stroke=STROKE
                stroke-width="1.8"
                stroke-linecap="round"
                stroke-linejoin="round"
            />
        </svg>
    }
}

/// The Still visualization mode — a frozen, non-animated scene for anyone
/// who finds continuous motion distracting rather than immersive.
#[component]
pub fn StillIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none">
            <circle cx="12" cy="12" r="7" stroke=STROKE stroke-width="1.8" />
            <circle cx="12" cy="12" r="1.6" fill=STROKE />
        </svg>
    }
}

#[component]
pub fn VolumeIcon() -> impl IntoView {
    view! {
        <svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none">
            <path
                d="M4 9.5v5h3.2L12 18V6L7.2 9.5H4Z"
                fill=STROKE
            />
            <path
                d="M15.8 8.6a5 5 0 0 1 0 6.8"
                stroke=STROKE
                stroke-width="1.8"
                stroke-linecap="round"
            />
            <path
                d="M18.2 6.4a8.5 8.5 0 0 1 0 11.2"
                stroke=STROKE
                stroke-width="1.8"
                stroke-linecap="round"
                opacity="0.6"
            />
        </svg>
    }
}
