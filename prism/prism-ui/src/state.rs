// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Browser state management via Leptos signals.

use leptos::prelude::*;
use prism_common::{ContentZone, SafetyLevel, SearchResult};
use serde::{Deserialize, Serialize};

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
            Self::Paradigm => "Local + web + DHT + AI (requires Ollama)",
        }
    }
}

/// Page rendering mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderMode {
    /// Sanitized HTML, no JS/CSS. Overlay works. Safe.
    Reader,
    /// Full site in sandboxed iframe. Real JS/CSS. Security badge only.
    FullPage,
}

/// Represents the current page being displayed.
#[derive(Clone, Debug)]
pub enum PageView {
    Welcome,
    Search {
        query: String,
        results: Vec<SearchResult>,
    },
    Compare {
        query: String,
    },
    /// Reader mode: sanitized HTML rendered as innerHTML.
    Page {
        html: String,
    },
    /// Full page mode: raw URL loaded in sandboxed iframe.
    FullPageIframe {
        url: String,
    },
    Settings,
    SubmitClaim,
    Bookmarks,
    Loading,
    Error {
        message: String,
    },
}

/// A lightweight snapshot for the history stack — stores URL + title only.
/// Full page content is NOT stored to avoid memory explosion.
/// Back/forward navigation re-fetches or re-searches as needed.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub url: String,
    pub title: String,
}

/// A saved bookmark.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bookmark {
    pub url: String,
    pub title: String,
    pub added: u64,
}

/// A browser tab with its own view state.
#[derive(Clone, Debug)]
pub struct Tab {
    pub id: u32,
    pub title: String,
    pub url: String,
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
    /// Spore-generated epistemic summary for the current search.
    pub spore_summary: ReadSignal<String>,
    pub set_spore_summary: WriteSignal<String>,
    /// Page rendering mode (Reader vs Full Page).
    pub render_mode: ReadSignal<RenderMode>,
    pub set_render_mode: WriteSignal<RenderMode>,
    /// Open tabs.
    pub tabs: ReadSignal<Vec<Tab>>,
    pub set_tabs: WriteSignal<Vec<Tab>>,
    /// Active tab ID.
    pub active_tab: ReadSignal<u32>,
    pub set_active_tab: WriteSignal<u32>,
    /// Next tab ID counter.
    next_tab_id: StoredValue<u32>,
    /// Bookmarks.
    pub bookmarks: ReadSignal<Vec<Bookmark>>,
    pub set_bookmarks: WriteSignal<Vec<Bookmark>>,
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
        };
        let history = StoredValue::new(vec![initial_entry]);
        let (history_cursor, set_history_cursor) = signal(0usize);
        let navigating_history = StoredValue::new(false);
        let (search_generation, set_search_generation) = signal(0u64);
        let (consciousness, set_consciousness) = signal(0.0f32);
        let (epistemic_confidence, set_epistemic_confidence) = signal(0.0f32);
        let (prediction_error, set_prediction_error) = signal(0.0f32);
        let (spore_summary, set_spore_summary) = signal(String::new());
        let saved_mode =
            crate::persistence::load::<RenderMode>("render-mode").unwrap_or(RenderMode::Reader);
        let (render_mode, set_render_mode) = signal(saved_mode);

        let initial_tab = Tab {
            id: 1,
            title: "Prism".to_string(),
            url: "prism://welcome".to_string(),
            view: PageView::Welcome,
        };
        let (tabs, set_tabs) = signal(vec![initial_tab]);
        let (active_tab, set_active_tab) = signal(1u32);
        let next_tab_id = StoredValue::new(2u32);

        let saved_bookmarks =
            crate::persistence::load::<Vec<Bookmark>>("bookmarks").unwrap_or_default();
        let (bookmarks, set_bookmarks) = signal(saved_bookmarks);

        Self {
            current_url,
            set_current_url,
            page_title,
            set_page_title,
            view,
            set_view,
            zone,
            set_zone,
            safety,
            set_safety,
            threat_count,
            set_threat_count,
            loading,
            set_loading,
            search_mode,
            set_search_mode,
            history,
            history_cursor,
            set_history_cursor,
            navigating_history,
            search_generation,
            set_search_generation,
            consciousness,
            set_consciousness,
            epistemic_confidence,
            set_epistemic_confidence,
            prediction_error,
            set_prediction_error,
            spore_summary,
            set_spore_summary,
            render_mode,
            set_render_mode,
            tabs,
            set_tabs,
            active_tab,
            set_active_tab,
            next_tab_id,
            bookmarks,
            set_bookmarks,
        }
    }

    /// Push a new page onto the history stack (called on every navigation).
    /// Only stores URL + title (not full page content) to avoid memory explosion.
    pub fn push_history(&self, url: &str, title: &str, _view: &PageView) {
        if self.navigating_history.get_value() {
            return;
        }
        self.history.update_value(|h| {
            let cursor = self.history_cursor.get_untracked();
            h.truncate(cursor + 1);
            h.push(HistoryEntry {
                url: url.to_string(),
                title: title.to_string(),
            });
            if h.len() > 50 {
                h.remove(0);
            }
            let new_cursor = h.len() - 1;
            self.set_history_cursor.set(new_cursor);
        });
    }

    /// Navigate back in history. Returns the URL to re-navigate to.
    pub fn go_back(&self) -> Option<String> {
        let cursor = self.history_cursor.get_untracked();
        if cursor == 0 {
            return None;
        }
        self.navigate_to_history(cursor - 1)
    }

    /// Navigate forward in history. Returns the URL to re-navigate to.
    pub fn go_forward(&self) -> Option<String> {
        let cursor = self.history_cursor.get_untracked();
        let can = self.history.with_value(|h| cursor + 1 < h.len());
        if !can {
            return None;
        }
        self.navigate_to_history(cursor + 1)
    }

    fn navigate_to_history(&self, idx: usize) -> Option<String> {
        self.navigating_history.set_value(true);
        let url = self.history.with_value(|h| {
            h.get(idx).map(|entry| {
                self.set_history_cursor.set(idx);
                self.set_current_url.set(entry.url.clone());
                self.set_page_title.set(entry.title.clone());
                entry.url.clone()
            })
        });
        self.navigating_history.set_value(false);
        url
    }

    pub fn can_go_back(&self) -> bool {
        self.history_cursor.get() > 0
    }

    pub fn can_go_forward(&self) -> bool {
        let cursor = self.history_cursor.get();
        self.history.with_value(|h| cursor + 1 < h.len())
    }

    /// Open a new tab with the welcome page. Returns the new tab ID.
    pub fn new_tab(&self) -> u32 {
        let id = self.next_tab_id.get_value();
        self.next_tab_id.set_value(id + 1);
        let tab = Tab {
            id,
            title: "New Tab".to_string(),
            url: "prism://welcome".to_string(),
            view: PageView::Welcome,
        };
        self.set_tabs.update(|tabs| tabs.push(tab));
        self.set_active_tab.set(id);
        self.set_view.set(PageView::Welcome);
        self.set_current_url.set("prism://welcome".to_string());
        self.set_page_title.set("New Tab".to_string());
        id
    }

    /// Switch to a tab by ID.
    pub fn switch_tab(&self, id: u32) {
        let tab = self.tabs.get_untracked().into_iter().find(|t| t.id == id);
        if let Some(tab) = tab {
            self.set_active_tab.set(id);
            self.set_view.set(tab.view);
            self.set_current_url.set(tab.url);
            self.set_page_title.set(tab.title);
        }
    }

    /// Close a tab by ID. If it's the active tab, switch to the previous one.
    pub fn close_tab(&self, id: u32) {
        let tabs = self.tabs.get_untracked();
        if tabs.len() <= 1 {
            return;
        } // can't close the last tab
        let was_active = self.active_tab.get_untracked() == id;
        self.set_tabs.update(|tabs| tabs.retain(|t| t.id != id));
        if was_active {
            let remaining = self.tabs.get_untracked();
            if let Some(tab) = remaining.last() {
                self.switch_tab(tab.id);
            }
        }
    }

    /// Update the current tab's state (called on navigation).
    pub fn sync_active_tab(&self) {
        let id = self.active_tab.get_untracked();
        let url = self.current_url.get_untracked();
        let title = self.page_title.get_untracked();
        let view = self.view.get_untracked();
        self.set_tabs.update(|tabs| {
            if let Some(tab) = tabs.iter_mut().find(|t| t.id == id) {
                tab.url = url;
                tab.title = title;
                tab.view = view;
            }
        });
    }

    /// Add a bookmark for the current page.
    pub fn add_bookmark(&self) {
        let url = self.current_url.get_untracked();
        let title = self.page_title.get_untracked();
        // Don't bookmark internal pages
        if url.starts_with("prism://") {
            return;
        }
        self.set_bookmarks.update(|bm| {
            // Avoid duplicates
            if bm.iter().any(|b| b.url == url) {
                return;
            }
            bm.push(Bookmark {
                url,
                title,
                added: js_sys::Date::now() as u64,
            });
        });
        crate::persistence::save("bookmarks", &self.bookmarks.get_untracked());
    }

    /// Remove a bookmark by URL.
    pub fn remove_bookmark(&self, url: &str) {
        self.set_bookmarks.update(|bm| bm.retain(|b| b.url != url));
        crate::persistence::save("bookmarks", &self.bookmarks.get_untracked());
    }

    /// Check if the current page is bookmarked.
    pub fn is_bookmarked(&self) -> bool {
        let url = self.current_url.get();
        self.bookmarks.get().iter().any(|b| b.url == url)
    }
}
