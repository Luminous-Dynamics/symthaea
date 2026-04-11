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

/// A snapshot of a visited page for the history stack.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub url: String,
    pub title: String,
    pub view: PageView,
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
    /// History stack and cursor for back/forward navigation.
    history: StoredValue<Vec<HistoryEntry>>,
    history_cursor: ReadSignal<usize>,
    set_history_cursor: WriteSignal<usize>,
    /// True while navigating via back/forward (suppresses push).
    navigating_history: StoredValue<bool>,
    /// Generation counter to discard stale external search results.
    pub search_generation: ReadSignal<u64>,
    pub set_search_generation: WriteSignal<u64>,
    /// Consciousness level (Psi) from the embedded Spore kernel.
    pub consciousness: ReadSignal<f32>,
    pub set_consciousness: WriteSignal<f32>,
    /// Epistemic honest confidence (0.0-0.95).
    pub epistemic_confidence: ReadSignal<f32>,
    pub set_epistemic_confidence: WriteSignal<f32>,
    /// Prediction error from last Spore cycle (surprise signal).
    pub prediction_error: ReadSignal<f32>,
    pub set_prediction_error: WriteSignal<f32>,
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

        let initial_mode = crate::persistence::load::<String>("search-mode")
            .and_then(|s| match s.as_str() {
                "Basic" => Some(SearchMode::Basic),
                "Advanced" => Some(SearchMode::Advanced),
                "Paradigm" => Some(SearchMode::Paradigm),
                _ => None,
            })
            .unwrap_or(SearchMode::Advanced);
        let (search_mode, set_search_mode) = signal(initial_mode);

        let initial_entry = HistoryEntry {
            url: "prism://welcome".to_string(),
            title: "Prism".to_string(),
            view: PageView::Welcome,
        };
        let history = StoredValue::new(vec![initial_entry]);
        let (history_cursor, set_history_cursor) = signal(0usize);
        let navigating_history = StoredValue::new(false);
        let (search_generation, set_search_generation) = signal(0u64);
        let (consciousness, set_consciousness) = signal(0.0f32);
        let (epistemic_confidence, set_epistemic_confidence) = signal(0.0f32);
        let (prediction_error, set_prediction_error) = signal(0.0f32);

        Self {
            current_url, set_current_url,
            page_title, set_page_title,
            view, set_view,
            zone, set_zone,
            safety, set_safety,
            threat_count, set_threat_count,
            loading, set_loading,
            search_mode, set_search_mode,
            history, history_cursor, set_history_cursor,
            navigating_history,
            search_generation, set_search_generation,
            consciousness, set_consciousness,
            epistemic_confidence, set_epistemic_confidence,
            prediction_error, set_prediction_error,
        }
    }

    /// Push a new page onto the history stack (called on every navigation).
    /// Truncates forward history if we navigated back then went somewhere new.
    pub fn push_history(&self, url: &str, title: &str, view: &PageView) {
        if self.navigating_history.get_value() {
            return;
        }
        self.history.update_value(|h| {
            let cursor = self.history_cursor.get_untracked();
            // Truncate any forward entries
            h.truncate(cursor + 1);
            h.push(HistoryEntry {
                url: url.to_string(),
                title: title.to_string(),
                view: view.clone(),
            });
            // Cap at 50 entries
            if h.len() > 50 {
                h.remove(0);
            }
            let new_cursor = h.len() - 1;
            self.set_history_cursor.set(new_cursor);
        });
    }

    /// Navigate back in history. Returns true if navigation happened.
    pub fn go_back(&self) -> bool {
        let cursor = self.history_cursor.get_untracked();
        if cursor == 0 {
            return false;
        }
        let new_cursor = cursor - 1;
        self.navigate_to_history(new_cursor);
        true
    }

    /// Navigate forward in history. Returns true if navigation happened.
    pub fn go_forward(&self) -> bool {
        let cursor = self.history_cursor.get_untracked();
        let can = self.history.with_value(|h| cursor + 1 < h.len());
        if !can {
            return false;
        }
        self.navigate_to_history(cursor + 1);
        true
    }

    fn navigate_to_history(&self, idx: usize) {
        self.navigating_history.set_value(true);
        if let Some(entry) = self.history.with_value(|h| h.get(idx).cloned()) {
            self.set_history_cursor.set(idx);
            self.set_current_url.set(entry.url);
            self.set_page_title.set(entry.title);
            self.set_view.set(entry.view);
        }
        self.navigating_history.set_value(false);
    }

    pub fn can_go_back(&self) -> bool {
        self.history_cursor.get() > 0
    }

    pub fn can_go_forward(&self) -> bool {
        let cursor = self.history_cursor.get();
        self.history.with_value(|h| cursor + 1 < h.len())
    }
}
