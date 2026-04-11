// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! DSID assurance level badge — shows MFA verification tier on emails and profiles.

use leptos::prelude::*;
use crate::dsid::AssuranceLevel;

/// Compact assurance badge showing E0-E4 level.
#[component]
pub fn AssuranceBadge(
    /// Assurance level as u8 (0-4). None means unknown/loading.
    #[prop(optional)]
    level: Option<u8>,
) -> impl IntoView {
    let assurance = level.map(AssuranceLevel::from_u8).unwrap_or(AssuranceLevel::Anonymous);
    let class = format!("assurance-badge {}", assurance.css_class());
    let title = format!("DSID Assurance: {} — {}", assurance.short_label(), assurance.label());

    view! {
        <span class=class title=title>
            {assurance.short_label()}
        </span>
    }
}
