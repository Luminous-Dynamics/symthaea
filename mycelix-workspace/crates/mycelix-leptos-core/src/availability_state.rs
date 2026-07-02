// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic availability-state rendering helpers for Mycelix frontends.

use leptos::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AvailabilityStateKind {
    Live,
    Mock,
    Empty,
    Locked,
    Degraded,
    Unavailable,
}

impl AvailabilityStateKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Live => "Live",
            Self::Mock => "Mock",
            Self::Empty => "Empty",
            Self::Locked => "Locked",
            Self::Degraded => "Degraded",
            Self::Unavailable => "Unavailable",
        }
    }

    pub fn css_class(self) -> &'static str {
        match self {
            Self::Live => "availability-live",
            Self::Mock => "availability-mock",
            Self::Empty => "availability-empty",
            Self::Locked => "availability-locked",
            Self::Degraded => "availability-degraded",
            Self::Unavailable => "availability-unavailable",
        }
    }

    pub fn icon(self) -> &'static str {
        match self {
            Self::Live => "●",
            Self::Mock => "◌",
            Self::Empty => "○",
            Self::Locked => "◈",
            Self::Degraded => "△",
            Self::Unavailable => "×",
        }
    }
}

#[component]
pub fn AvailabilityState(
    kind: AvailabilityStateKind,
    #[prop(into)] title: String,
    #[prop(into)] description: String,
    action: Option<AnyView>,
) -> impl IntoView {
    view! {
        <div class=format!("availability-state {}", kind.css_class())>
            <div class="availability-state-header">
                <span class="availability-state-icon">{kind.icon()}</span>
                <div class="availability-state-copy">
                    <div class="availability-state-meta">
                        <span class="availability-state-title">{title}</span>
                        <span class=format!("status-pill {}", kind.css_class())>{kind.label()}</span>
                    </div>
                    <p class="availability-state-description">{description}</p>
                </div>
            </div>
            {action}
        </div>
    }
}
