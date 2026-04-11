// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified user preferences — single localStorage key, loaded on init.
//! Replaces scattered individual preference storage with one coherent struct.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use mail_leptos_types::*;

const PREFS_KEY: &str = "mycelix_pulse_preferences";
pub const MIN_FONT_SCALE: u32 = 75;
pub const MAX_FONT_SCALE: u32 = 150;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserPreferences {
    // Appearance
    pub theme: Theme,
    pub density: Density,
    pub accent_color: String,
    pub theme_name: String,
    pub font_family: String,
    pub font_scale: u32,
    pub high_contrast: bool,

    // Layout
    pub reading_pane: ReadingPanePosition,

    // Compose
    pub default_pqc: bool,
    pub default_signature_id: Option<String>,
    pub reply_style: ReplyStyle,
    pub undo_send_seconds: u32,

    // Privacy
    pub send_read_receipts: ReadReceiptPolicy,
    pub show_typing_indicators: bool,
    pub share_availability: AvailabilityPolicy,

    // Notifications
    pub sound_enabled: bool,
    pub desktop_notifications: bool,
    pub quiet_hours_start: Option<u8>,
    pub quiet_hours_end: Option<u8>,

    // Mobile
    pub swipe_left: SwipeAction,
    pub swipe_right: SwipeAction,

    // Keyboard
    pub palette_shortcut: String,

    // Attention
    pub attention_budget_daily: u32,
    pub focus_time_enabled: bool,

    // Data
    pub auto_archive_days: Option<u32>,

    // Locale
    pub date_format: DateFormat,
    pub language: Language,

    // Vacation
    pub vacation_enabled: bool,
    pub vacation_subject: String,
    pub vacation_body: String,
    pub vacation_contacts_only: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReplyStyle { InlineQuote, TopPost }
impl Default for ReplyStyle { fn default() -> Self { Self::InlineQuote } }

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReadReceiptPolicy { Always, Ask, Never }
impl Default for ReadReceiptPolicy { fn default() -> Self { Self::Ask } }

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityPolicy { Everyone, ContactsOnly, Nobody }
impl Default for AvailabilityPolicy { fn default() -> Self { Self::ContactsOnly } }

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DateFormat { Auto, DmySlash, MdySlash, YmdDash }
impl DateFormat {
    pub fn label(&self) -> &'static str {
        match self { Self::Auto => "Auto (locale)", Self::DmySlash => "DD/MM/YYYY", Self::MdySlash => "MM/DD/YYYY", Self::YmdDash => "YYYY-MM-DD" }
    }
}
impl Default for DateFormat { fn default() -> Self { Self::Auto } }

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language { English, Afrikaans, Zulu, French, Spanish }
impl Language {
    pub fn label(&self) -> &'static str {
        match self { Self::English => "English", Self::Afrikaans => "Afrikaans", Self::Zulu => "isiZulu", Self::French => "Fran\u{00E7}ais", Self::Spanish => "Espa\u{00F1}ol" }
    }
    pub const ALL: &[Language] = &[Self::English, Self::Afrikaans, Self::Zulu, Self::French, Self::Spanish];
}
impl Default for Language { fn default() -> Self { Self::English } }

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            density: Density::Default,
            accent_color: "#06D6C8".into(),
            theme_name: "Circuit".into(),
            font_family: "Inter".into(),
            font_scale: 100,
            high_contrast: false,
            reading_pane: ReadingPanePosition::Off,
            default_pqc: true,
            default_signature_id: None,
            reply_style: ReplyStyle::InlineQuote,
            undo_send_seconds: 5,
            send_read_receipts: ReadReceiptPolicy::Ask,
            show_typing_indicators: true,
            share_availability: AvailabilityPolicy::ContactsOnly,
            sound_enabled: true,
            desktop_notifications: true,
            quiet_hours_start: None,
            quiet_hours_end: None,
            swipe_left: SwipeAction::Archive,
            swipe_right: SwipeAction::Delete,
            palette_shortcut: "k".into(),
            attention_budget_daily: 100,
            focus_time_enabled: false,
            auto_archive_days: None,
            date_format: DateFormat::Auto,
            language: Language::English,
            vacation_enabled: false,
            vacation_subject: "Out of office".into(),
            vacation_body: "I'm currently away and will respond when I return.".into(),
            vacation_contacts_only: true,
        }
    }
}

pub fn clamp_font_scale(scale: u32) -> u32 {
    scale.clamp(MIN_FONT_SCALE, MAX_FONT_SCALE)
}

fn sanitize_preferences(mut prefs: UserPreferences) -> UserPreferences {
    prefs.font_scale = clamp_font_scale(prefs.font_scale);
    prefs
}

/// Load preferences from localStorage.
pub fn load_preferences() -> UserPreferences {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(PREFS_KEY).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
        .map(sanitize_preferences)
        .unwrap_or_default()
}

/// Save preferences to localStorage.
pub fn save_preferences(prefs: &UserPreferences) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        if let Ok(json) = serde_json::to_string(&sanitize_preferences(prefs.clone())) {
            let _ = storage.set_item(PREFS_KEY, &json);
        }
    }
}

/// Provide preferences as a reactive signal with auto-persistence.
pub fn provide_preferences_context() {
    let prefs = RwSignal::new(load_preferences());
    provide_context(prefs);

    // Auto-save on every change
    Effect::new(move |_| {
        let p = prefs.get();
        save_preferences(&p);
    });

    // Apply visual preferences
    Effect::new(move |_| {
        let p = sanitize_preferences(prefs.get());

        // Font family
        let _ = js_sys::eval(&format!(
            "document.documentElement.style.setProperty('--font-sans','\"{}\",-apple-system,system-ui,sans-serif')",
            p.font_family
        ));

        // Font scale
        let _ = js_sys::eval(&format!(
            "document.documentElement.style.fontSize='{}%'", p.font_scale
        ));

        // High contrast
        if p.high_contrast {
            let _ = js_sys::eval("document.documentElement.setAttribute('data-contrast','high')");
        } else {
            let _ = js_sys::eval("document.documentElement.removeAttribute('data-contrast')");
        }
    });
}

pub fn use_preferences() -> RwSignal<UserPreferences> {
    expect_context::<RwSignal<UserPreferences>>()
}
