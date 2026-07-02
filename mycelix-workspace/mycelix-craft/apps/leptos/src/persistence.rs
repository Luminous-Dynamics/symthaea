// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! localStorage persistence helpers.

use serde::{de::DeserializeOwned, Serialize};

pub fn save<T: Serialize>(key: &str, value: &T) {
    if let Some(storage) = web_sys::window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
    {
        if let Ok(json) = serde_json::to_string(value) {
            let _ = storage.set_item(key, &json);
        }
    }
}

pub fn load<T: DeserializeOwned>(key: &str) -> Option<T> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
        .and_then(|s| s.get_item(key).ok())
        .flatten()
        .and_then(|json| serde_json::from_str(&json).ok())
}
