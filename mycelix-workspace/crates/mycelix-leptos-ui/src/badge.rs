// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Badge and status dot components.

use leptos::prelude::*;

/// Badge variant for semantic coloring.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BadgeVariant {
    #[default]
    Default,
    Success,
    Warning,
    Error,
    Info,
}

impl BadgeVariant {
    fn css_class(&self) -> &'static str {
        match self {
            BadgeVariant::Default => "badge",
            BadgeVariant::Success => "badge badge-success",
            BadgeVariant::Warning => "badge badge-warning",
            BadgeVariant::Error => "badge badge-error",
            BadgeVariant::Info => "badge badge-info",
        }
    }
}

/// Inline badge with optional semantic color.
#[component]
pub fn Badge(
    #[prop(into)] label: String,
    #[prop(optional)] variant: BadgeVariant,
) -> impl IntoView {
    view! {
        <span class=variant.css_class()>{label}</span>
    }
}

/// Pulsing status dot indicator.
#[component]
pub fn StatusDot(#[prop(into)] status: String, #[prop(optional)] pulse: bool) -> impl IntoView {
    view! {
        <span class=format!(
            "status-dot status-{}{}",
            status,
            if pulse { " status-pulse" } else { "" }
        )></span>
    }
}
