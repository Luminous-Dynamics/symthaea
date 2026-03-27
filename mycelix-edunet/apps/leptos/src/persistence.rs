// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Simple localStorage persistence layer.
//!
//! No conductor needed -- saves and loads JSON to/from the browser's
//! localStorage API. Used to persist role selection, sovereignty level,
//! and other lightweight state across page reloads.

use serde::{de::DeserializeOwned, Serialize};
use web_sys::window;

/// Save a value to localStorage.
pub fn save<T: Serialize>(key: &str, value: &T) {
    if let Some(storage) = window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
    {
        if let Ok(json) = serde_json::to_string(value) {
            let _ = storage.set_item(key, &json);
        }
    }
}

/// Load a value from localStorage.
pub fn load<T: DeserializeOwned>(key: &str) -> Option<T> {
    window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
        .and_then(|s| s.get_item(key).ok())
        .flatten()
        .and_then(|json| serde_json::from_str(&json).ok())
}

/// Remove a value from localStorage.
pub fn remove(key: &str) {
    if let Some(storage) = window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
    {
        let _ = storage.remove_item(key);
    }
}
