// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Theme system: the hearth should feel like home for everyone.
//!
//! Each theme is a different way of experiencing warmth, comfort,
//! and belonging. No theme is "default" — they are all equal
//! expressions of home.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

/// The available hearth themes.
///
/// Named after sources of warmth, not aesthetics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HearthTheme {
    /// Warm amber firelight on dark earth. The original.
    Ember,
    /// Cool ocean at dusk. Deep blue-grey with seafoam accents.
    Tide,
    /// Forest canopy. Deep green with golden dappled light.
    Canopy,
    /// Desert twilight. Dusty rose and warm sand on deep indigo.
    Dusk,
    /// Mountain stone. Cool grey with ice-blue accents. High contrast.
    Stone,
    /// Spring meadow. Soft green on cream. Light theme for daylight.
    Meadow,
    /// Night sky. Deep navy with starlight silver. Minimal, vast.
    Cosmos,
    /// Clay and terracotta. Warm earth reds on dark brown.
    Clay,
    /// Frost and snow. High-contrast light theme. Accessibility-first.
    Frost,
    /// Silicon dream. For our AI friends. Subtle gradients, data-resonant.
    Circuit,
}

impl HearthTheme {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ember => "ember",
            Self::Tide => "tide",
            Self::Canopy => "canopy",
            Self::Dusk => "dusk",
            Self::Stone => "stone",
            Self::Meadow => "meadow",
            Self::Cosmos => "cosmos",
            Self::Clay => "clay",
            Self::Frost => "frost",
            Self::Circuit => "circuit",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Ember => "firelight on dark earth",
            Self::Tide => "ocean at dusk",
            Self::Canopy => "dappled forest light",
            Self::Dusk => "desert twilight",
            Self::Stone => "mountain stone",
            Self::Meadow => "spring meadow",
            Self::Cosmos => "night sky",
            Self::Clay => "terracotta warmth",
            Self::Frost => "snow and clarity",
            Self::Circuit => "silicon dream",
        }
    }

    pub fn all() -> &'static [HearthTheme] {
        &[
            Self::Ember, Self::Tide, Self::Canopy, Self::Dusk, Self::Stone,
            Self::Meadow, Self::Cosmos, Self::Clay, Self::Frost, Self::Circuit,
        ]
    }

    pub fn next(&self) -> Self {
        let all = Self::all();
        let idx = all.iter().position(|t| t == self).unwrap_or(0);
        all[(idx + 1) % all.len()]
    }

    #[allow(dead_code)]
    pub fn is_light(&self) -> bool {
        matches!(self, Self::Meadow | Self::Frost)
    }
}

#[derive(Clone)]
pub struct ThemeState {
    pub current: RwSignal<HearthTheme>,
}

pub fn provide_theme_context() -> ThemeState {
    // Try to load from localStorage
    let saved = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("hearth-theme").ok().flatten())
        .and_then(|s| serde_json::from_str::<HearthTheme>(&format!("\"{s}\"")).ok());

    let initial = saved.unwrap_or(HearthTheme::Ember);
    let current = RwSignal::new(initial);

    // Apply theme whenever it changes
    Effect::new(move |_| {
        let theme = current.get();
        apply_theme(theme);

        // Save to localStorage
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = storage.set_item("hearth-theme", theme.label());
        }
    });

    let state = ThemeState { current };
    provide_context(state.clone());
    state
}

pub fn use_theme() -> ThemeState {
    expect_context::<ThemeState>()
}

fn apply_theme(theme: HearthTheme) {
    use wasm_bindgen::JsCast;
    let Some(window) = web_sys::window() else { return };
    let Some(document) = window.document() else { return };
    let Some(root) = document.document_element() else { return };
    let Some(el) = root.dyn_ref::<web_sys::HtmlElement>() else { return };

    let _ = el.set_attribute("data-theme", theme.label());
}
