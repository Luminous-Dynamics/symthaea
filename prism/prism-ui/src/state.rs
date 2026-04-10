// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browser state management via Leptos signals.

use leptos::prelude::*;
use prism_common::{ContentZone, SafetyLevel, SearchResult};

/// Search depth mode — controls how far Prism looks for answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    /// Local claims only (offline, instant, 208+ curated claims)
    Basic,
    /// Local + Wikipedia + DuckDuckGo (web-augmented, ~2s)
    Advanced,
    /// Local + web + DHT + AI reasoning (full epistemic pipeline)
    Paradigm,
}

impl SearchMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Basic => "Basic",
            Self::Advanced => "Advanced",
            Self::Paradigm => "Paradigm",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            Self::Basic => "Local claims only",
            Self::Advanced => "Local + web sources",
            Self::Paradigm => "Full epistemic pipeline",
        }
    }
}

/// Represents the current page being displayed.
#[derive(Clone, Debug)]
pub enum PageView {
    Welcome,
    Search { query: String, results: Vec<SearchResult> },
    Compare { query: String },
    Page { html: String },
    Settings,
    SubmitClaim,
    Loading,
    Error { message: String },
}

/// Global browser state, provided at the App root.
/// Uses ReadSignal/WriteSignal pairs (standard Leptos 0.8 CSR pattern).
#[derive(Clone)]
pub struct BrowserState {
    pub current_url: ReadSignal<String>,
    pub set_current_url: WriteSignal<String>,
    pub page_title: ReadSignal<String>,
    pub set_page_title: WriteSignal<String>,
    pub view: ReadSignal<PageView>,
    pub set_view: WriteSignal<PageView>,
    pub zone: ReadSignal<ContentZone>,
    pub set_zone: WriteSignal<ContentZone>,
    pub safety: ReadSignal<SafetyLevel>,
    pub set_safety: WriteSignal<SafetyLevel>,
    pub threat_count: ReadSignal<usize>,
    pub set_threat_count: WriteSignal<usize>,
    pub loading: ReadSignal<bool>,
    pub set_loading: WriteSignal<bool>,
    pub search_mode: ReadSignal<SearchMode>,
    pub set_search_mode: WriteSignal<SearchMode>,
}

impl BrowserState {
    pub fn new() -> Self {
        let (current_url, set_current_url) = signal("prism://welcome".to_string());
        let (page_title, set_page_title) = signal("Prism".to_string());
        let (view, set_view) = signal(PageView::Welcome);
        let (zone, set_zone) = signal(ContentZone::Local);
        let (safety, set_safety) = signal(SafetyLevel::Green);
        let (threat_count, set_threat_count) = signal(0usize);
        let (loading, set_loading) = signal(false);
        let (search_mode, set_search_mode) = signal(SearchMode::Advanced);

        Self {
            current_url, set_current_url,
            page_title, set_page_title,
            view, set_view,
            zone, set_zone,
            safety, set_safety,
            threat_count, set_threat_count,
            loading, set_loading,
            search_mode, set_search_mode,
        }
    }
}
