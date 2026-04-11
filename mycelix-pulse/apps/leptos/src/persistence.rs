// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! LocalStorage persistence for demo mode.
//!
//! Saves inbox state (star, archive, pin, read, labels) to localStorage
//! so demo interactions survive page refresh.

use mail_leptos_types::EmailListItem;

const INBOX_KEY: &str = "mycelix_pulse_demo_inbox";
const CONTACTS_KEY: &str = "mycelix_pulse_demo_contacts";

/// Save the current inbox to localStorage.
pub fn save_inbox(emails: &[EmailListItem]) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        if let Ok(json) = serde_json::to_string(emails) {
            let _ = storage.set_item(INBOX_KEY, &json);
        }
    }
}

/// Load saved inbox from localStorage, if available.
pub fn load_inbox() -> Option<Vec<EmailListItem>> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(INBOX_KEY).ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
}

/// Clear saved demo data.
pub fn clear_demo_data() {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = storage.remove_item(INBOX_KEY);
        let _ = storage.remove_item(CONTACTS_KEY);
    }
}

/// Check if we have saved demo data.
pub fn has_saved_data() -> bool {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(INBOX_KEY).ok().flatten())
        .is_some()
}
