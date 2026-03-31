// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommonsTheme { Grove, Meadow, Stone, Dusk, Frost }

impl CommonsTheme {
    pub fn label(&self) -> &'static str {
        match self { Self::Grove => "grove", Self::Meadow => "meadow", Self::Stone => "stone", Self::Dusk => "dusk", Self::Frost => "frost" }
    }
}

#[derive(Clone)]
pub struct ThemeState { pub current: RwSignal<CommonsTheme> }

pub fn provide_theme_context() -> ThemeState {
    let saved = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("commons-theme").ok().flatten())
        .and_then(|s| serde_json::from_str::<CommonsTheme>(&format!("\"{s}\"")).ok());
    let current = RwSignal::new(saved.unwrap_or(CommonsTheme::Grove));
    Effect::new(move |_| {
        let theme = current.get();
        if let Some(window) = web_sys::window() {
            if let Some(doc) = window.document() {
                if let Some(root) = doc.document_element() {
                    let _ = root.set_attribute("data-theme", theme.label());
                }
            }
        }
        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = storage.set_item("commons-theme", theme.label());
        }
    });
    let state = ThemeState { current };
    provide_context(state.clone());
    state
}
