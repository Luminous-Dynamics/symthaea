// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! localStorage persistence for Prism settings.

use serde::{Serialize, de::DeserializeOwned};

const STORAGE_PREFIX: &str = "prism-";

/// Save a value to localStorage.
pub fn save<T: Serialize>(key: &str, value: &T) {
    let full_key = format!("{}{}", STORAGE_PREFIX, key);
    if let Ok(json) = serde_json::to_string(value) {
        if let Some(storage) = web_sys::window()
            .and_then(|w| w.local_storage().ok())
            .flatten()
        {
            let _ = storage.set_item(&full_key, &json);
        }
    }
}

/// Load a value from localStorage.
pub fn load<T: DeserializeOwned>(key: &str) -> Option<T> {
    let full_key = format!("{}{}", STORAGE_PREFIX, key);
    web_sys::window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
        .and_then(|s| s.get_item(&full_key).ok())
        .flatten()
        .and_then(|json| serde_json::from_str(&json).ok())
}
