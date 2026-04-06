// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Theme system — dark/light/high-contrast via CSS custom properties.
//!
//! Sets `data-theme` attribute on `<html>` reactively. Persists choice
//! to localStorage.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use crate::persistence;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    Dark,
    Light,
    HighContrast,
}

impl Theme {
    pub fn as_str(&self) -> &'static str {
        match self {
            Theme::Dark => "dark",
            Theme::Light => "light",
            Theme::HighContrast => "high-contrast",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Theme::Dark => "Dark",
            Theme::Light => "Light",
            Theme::HighContrast => "High Contrast",
        }
    }

    pub fn next(&self) -> Theme {
        match self {
            Theme::Dark => Theme::Light,
            Theme::Light => Theme::HighContrast,
            Theme::HighContrast => Theme::Dark,
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Theme::Dark => "\u{263E}",      // moon
            Theme::Light => "\u{2600}",     // sun
            Theme::HighContrast => "\u{25D1}", // circle half
        }
    }
}

const THEME_KEY: &str = "praxis_theme";

/// Detect system color scheme preference.
fn system_theme() -> Theme {
    web_sys::window()
        .and_then(|w| w.match_media("(prefers-color-scheme: light)").ok().flatten())
        .map(|mql| if mql.matches() { Theme::Light } else { Theme::Dark })
        .unwrap_or(Theme::Dark)
}

/// Provide theme context. Call once at app root.
/// Respects system preference on first visit, then uses localStorage.
pub fn provide_theme_context() -> (ReadSignal<Theme>, WriteSignal<Theme>) {
    let initial = persistence::load::<Theme>(THEME_KEY).unwrap_or_else(|| system_theme());
    let (theme, set_theme) = signal(initial);

    // Apply theme + time-of-day attributes to <html> reactively
    Effect::new(move |_| {
        let t = theme.get();
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(el) = doc.document_element() {
                let _ = el.set_attribute("data-theme", t.as_str());
                // Time-of-day color modulation
                let hour = js_sys::Date::new_0().get_hours() as u8;
                let time_period = match hour {
                    5..=11 => "morning",
                    12..=17 => "afternoon",
                    _ => "evening",
                };
                let _ = el.set_attribute("data-time", time_period);
            }
        }
        persistence::save(THEME_KEY, &t);
    });

    provide_context(theme);
    provide_context(set_theme);
    (theme, set_theme)
}

pub fn use_theme() -> ReadSignal<Theme> {
    expect_context::<ReadSignal<Theme>>()
}

pub fn use_set_theme() -> WriteSignal<Theme> {
    expect_context::<WriteSignal<Theme>>()
}
