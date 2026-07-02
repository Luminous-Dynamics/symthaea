// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Freshness and staleness indicators for live-backed UI surfaces.

use leptos::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshnessLevel {
    Fresh,
    Aging,
    Stale,
    Unknown,
}

impl FreshnessLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Fresh => "Fresh",
            Self::Aging => "Aging",
            Self::Stale => "Stale",
            Self::Unknown => "Unknown",
        }
    }

    pub fn css_class(self) -> &'static str {
        match self {
            Self::Fresh => "freshness-fresh",
            Self::Aging => "freshness-aging",
            Self::Stale => "freshness-stale",
            Self::Unknown => "freshness-unknown",
        }
    }
}

#[component]
pub fn FreshnessBadge(
    level: FreshnessLevel,
    #[prop(optional, into)] detail: Option<String>,
) -> impl IntoView {
    let label = detail.unwrap_or_else(|| level.label().into());
    view! {
        <span class=format!("badge freshness-badge {}", level.css_class())>
            {label}
        </span>
    }
}
