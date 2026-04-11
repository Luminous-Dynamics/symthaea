// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dark/light theme toggle (#9).

use leptos::prelude::*;
use mail_leptos_types::{Theme, Density};

const STORAGE_KEY: &str = "mycelix_mail_theme";
const DENSITY_KEY: &str = "mycelix_mail_density";

#[derive(Clone, Copy)]
pub struct ThemeState {
    pub current: RwSignal<Theme>,
    pub density: RwSignal<Density>,
}

pub fn provide_theme_context() {
    // Load from localStorage
    let saved = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(STORAGE_KEY).ok().flatten())
        .and_then(|v| match v.as_str() {
            "light" => Some(Theme::Light),
            _ => None,
        })
        .unwrap_or(Theme::Dark);

    let saved_density = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(DENSITY_KEY).ok().flatten())
        .and_then(|v| match v.as_str() {
            "compact" => Some(Density::Compact),
            "comfortable" => Some(Density::Comfortable),
            _ => None,
        })
        .unwrap_or(Density::Default);

    let state = ThemeState {
        current: RwSignal::new(saved),
        density: RwSignal::new(saved_density),
    };
    provide_context(state);

    apply_theme(saved);
    apply_density(saved_density);
    check_auto_theme_schedule(state);

    Effect::new(move |_| {
        let theme = state.current.get();
        apply_theme(theme);
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = storage.set_item(STORAGE_KEY, theme.data_attr());
        }
    });

    Effect::new(move |_| {
        let density = state.density.get();
        apply_density(density);
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
        {
            let _ = storage.set_item(DENSITY_KEY, density.data_attr());
        }
    });
}

fn apply_theme(theme: Theme) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(el) = doc.document_element() {
            let _ = el.set_attribute("data-theme", theme.data_attr());
        }
    }
}

fn apply_density(density: Density) {
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        if let Some(el) = doc.document_element() {
            let _ = el.set_attribute("data-density", density.data_attr());
        }
    }
}

/// Check if auto-schedule theme is enabled and apply. Called on init.
fn check_auto_theme_schedule(state: ThemeState) {
    let auto_enabled = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("mycelix_auto_theme").ok().flatten())
        .map(|v| v == "true")
        .unwrap_or(false);

    if auto_enabled {
        let hour = js_sys::Date::new_0().get_hours();
        // Dark after 19:00, light during day (7:00-19:00)
        let target = if hour >= 19 || hour < 7 { Theme::Dark } else { Theme::Light };
        if state.current.get_untracked() != target {
            state.current.set(target);
        }
    }
}

pub fn use_theme() -> ThemeState {
    expect_context::<ThemeState>()
}
