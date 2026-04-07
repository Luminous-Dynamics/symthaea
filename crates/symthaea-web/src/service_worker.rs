// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/// Register the service worker for asset caching.
/// Appends a version query param derived from the page's script hash,
/// so each Trunk build gets a fresh cache (old caches auto-purged by SW).
pub fn register() {
    if let Some(window) = web_sys::window() {
        // Extract content hash from the Trunk-generated script URL
        let version = window
            .document()
            .and_then(|d| {
                d.query_selector("script[src*='symthaea-web-']")
                    .ok()
                    .flatten()
            })
            .and_then(|el| el.get_attribute("src"))
            .and_then(|src| {
                // "symthaea-web-aa152cf7b3dc82dc.js" → "aa152cf7b3dc82dc"
                src.strip_prefix("/symthaea-web-")
                    .or_else(|| src.strip_prefix("symthaea-web-"))
                    .and_then(|rest| rest.strip_suffix(".js"))
                    .map(|h| h.to_string())
            })
            .unwrap_or_else(|| "v2".to_string());

        let sw = window.navigator().service_worker();
        let url = format!("./assets/sw.js?v={}", version);
        let _ = sw.register(&url);
        log::info!("Service worker registered (cache version: {})", version);
    }
}
