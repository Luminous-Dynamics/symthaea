// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Member avatar: colored circle with initial letter.
//! Colors match the Canvas2D node colors (gold=guardian, purple=minor, amber=other).

use leptos::prelude::*;
use hearth_leptos_types::MemberRole;

/// Role -> CSS color for the avatar circle.
fn role_color(role: &MemberRole) -> &'static str {
    if role.is_guardian() {
        "rgba(250, 191, 38, 0.85)" // gold
    } else if role.is_minor() {
        "rgba(166, 140, 250, 0.85)" // purple
    } else {
        "rgba(212, 165, 116, 0.85)" // amber
    }
}

#[component]
pub fn MemberAvatar(
    name: String,
    role: MemberRole,
    #[prop(default = 32)]
    size: u32,
) -> impl IntoView {
    let initial = name.chars().next().unwrap_or('?').to_uppercase().to_string();
    let bg = role_color(&role);
    let font_size = (size as f64 * 0.45) as u32;

    view! {
        <span
            class="member-avatar"
            style=format!(
                "width:{}px;height:{}px;background:{};font-size:{}px",
                size, size, bg, font_size
            )
        >
            {initial}
        </span>
    }
}
