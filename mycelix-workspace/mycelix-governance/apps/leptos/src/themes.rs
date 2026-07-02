// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Theme system: civic spaces have gravity and history.
//!
//! Named after civic gathering spaces, not aesthetics.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CivicTheme {
    /// Deep indigo-violet with copper accents. Gravity of deliberation.
    Assembly,
    /// Warmer bronze and earth tones. The commons marketplace.
    Forum,
    /// Cool grey-blue. Maritime trade routes. Finance-forward.
    Harbor,
    /// Night sky. Minimal, vast. For deep constitutional reading.
    Cosmos,
    /// Marble white. High-contrast light theme. Accessibility-first.
    Agora,
}

impl CivicTheme {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Assembly => "assembly",
            Self::Forum => "forum",
            Self::Harbor => "harbor",
            Self::Cosmos => "cosmos",
            Self::Agora => "agora",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Assembly => "indigo deliberation",
            Self::Forum => "bronze commons",
            Self::Harbor => "maritime trade",
            Self::Cosmos => "night sky",
            Self::Agora => "marble clarity",
        }
    }

    pub fn all() -> &'static [CivicTheme] {
        &[Self::Assembly, Self::Forum, Self::Harbor, Self::Cosmos, Self::Agora]
    }

    pub fn next(&self) -> Self {
        let all = Self::all();
        let idx = all.iter().position(|t| t == self).unwrap_or(0);
        all[(idx + 1) % all.len()]
    }

    pub fn is_light(&self) -> bool {
        matches!(self, Self::Agora)
    }
}

#[derive(Clone)]
pub struct ThemeState {
    pub current: RwSignal<CivicTheme>,
}

pub fn provide_theme_context() -> ThemeState {
    let saved = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("civic-theme").ok().flatten())
        .and_then(|s| serde_json::from_str::<CivicTheme>(&format!("\"{s}\"")).ok());

    let initial = saved.unwrap_or(CivicTheme::Assembly);
    let current = RwSignal::new(initial);

    Effect::new(move |_| {
        let theme = current.get();
        apply_theme(theme);

        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = storage.set_item("civic-theme", theme.label());
        }
    });

    let state = ThemeState { current };
    provide_context(state.clone());
    state
}

#[allow(dead_code)]
pub fn use_theme() -> ThemeState {
    expect_context::<ThemeState>()
}

fn apply_theme(theme: CivicTheme) {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else { return };
    let Some(document) = window.document() else { return };
    let Some(root) = document.document_element() else { return };
    let Some(el) = root.dyn_ref::<web_sys::HtmlElement>() else { return };

    let _ = el.set_attribute("data-theme", theme.label());
}
