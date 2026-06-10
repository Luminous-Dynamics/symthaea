// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic theme framework for Mycelix Leptos apps.
//!
//! Each app defines its own theme enum and implements [`AppTheme`].
//! The shared infrastructure handles localStorage persistence,
//! `data-theme` attribute application, and reactive context.

use crate::util::set_root_attribute;
use leptos::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

/// Trait for app-specific theme enums.
///
/// Implement this for your theme enum to get automatic localStorage
/// persistence and `data-theme` attribute management.
pub trait AppTheme:
    Clone + Copy + PartialEq + Eq + Serialize + DeserializeOwned + Send + Sync + 'static
{
    /// CSS-safe label used as the `data-theme` attribute value.
    fn label(&self) -> &'static str;

    /// All available theme variants.
    fn all() -> &'static [Self];

    /// Next theme in the cycle.
    fn next(&self) -> Self;

    /// Whether this is a light theme (for system-level adjustments).
    fn is_light(&self) -> bool {
        false
    }
}

/// Reactive theme state.
#[derive(Clone)]
pub struct ThemeState<T: AppTheme> {
    pub current: RwSignal<T>,
}

/// Initialize the theme context with localStorage persistence.
///
/// `storage_key` is the localStorage key (e.g. "hearth-theme", "civic-theme").
/// `default` is used when no saved theme exists.
pub fn provide_theme_context<T: AppTheme>(storage_key: &'static str, default: T) -> ThemeState<T> {
    let saved = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(storage_key).ok().flatten())
        .and_then(|s| serde_json::from_str::<T>(&format!("\"{s}\"")).ok());

    let initial = saved.unwrap_or(default);
    let current = RwSignal::new(initial);

    // Apply theme and save whenever it changes
    Effect::new(move |_| {
        let theme = current.get();
        set_root_attribute("data-theme", theme.label());

        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let _ = storage.set_item(storage_key, theme.label());
        }
    });

    let state = ThemeState { current };
    provide_context(state.clone());
    state
}

/// Retrieve the theme state from context.
pub fn use_theme_state<T: AppTheme>() -> ThemeState<T> {
    expect_context::<ThemeState<T>>()
}
