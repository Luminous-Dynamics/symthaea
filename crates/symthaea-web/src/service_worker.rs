// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Register the service worker for asset caching.
pub fn register() {
    if let Some(window) = web_sys::window() {
        let sw = window.navigator().service_worker();
        let _ = sw.register("./assets/sw.js");
        log::info!("Service worker registered");
    }
}
