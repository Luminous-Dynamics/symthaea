// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Bridge to Praxis's native Tauri backend commands that aren't zome calls.
//!
//! `mycelix_leptos_client::TauriIpcTransport` (feature `tauri`) already
//! covers `holochain_connect` / `call_zome`. This module covers the other
//! Praxis-specific Tauri command, `validate_pwa_import` — the security
//! boundary that lets a Trial Mode (browser/PWA) learner "graduate" to
//! Sovereign Mode (native) without their localStorage progress being
//! trusted blindly. See `apps/tauri/src-tauri/src/main.rs`.
//!
//! Also provides [`is_tauri`], used to decide whether the dashboard should
//! offer a real "Connect & Claim TEND" action (native) or a "Go Sovereign"
//! download prompt (browser).

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "core"], js_name = invoke, catch)]
    async fn tauri_invoke(cmd: &str, args: JsValue) -> Result<JsValue, JsValue>;
}

/// True if the app is currently running inside the Tauri native shell
/// (as opposed to a plain browser / PWA Trial Mode session).
///
/// Detects the `window.__TAURI__` global that Tauri's webview injects.
/// There is no equivalent global in a plain browser tab, so this is a
/// reliable Trial-Mode-vs-Sovereign-Mode discriminator.
pub fn is_tauri() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    js_sys::Reflect::get(&window, &JsValue::from_str("__TAURI__"))
        .map(|v| !v.is_undefined() && !v.is_null())
        .unwrap_or(false)
}

/// Result of validating an imported PWA `ProgressStore` against the Tauri
/// backend's BKT-replay integrity check. Mirrors `PwaImportResult` in
/// `apps/tauri/src-tauri/src/main.rs`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PwaImportResult {
    pub validated_nodes: u32,
    pub rejected_nodes: u32,
    pub validated_tend: f32,
    pub rejections: Vec<String>,
}

#[derive(Serialize)]
struct ValidatePwaImportArgs {
    progress_json: String,
}

/// Invoke the Tauri backend's `validate_pwa_import` command.
///
/// `progress_json` should be a JSON-serialized `ProgressStore` (or any
/// value with a `bkt_states` map shaped like `HashMap<String, BktState>`).
/// The backend replays the BKT update formula for every claimed mastery
/// state and rejects anything mathematically inconsistent with the
/// claimed attempts/correct counts — this is what prevents a learner from
/// hand-editing localStorage to fabricate mastery (and therefore TEND).
///
/// Returns an error string if not running under Tauri, if the IPC call
/// fails, or if the backend rejects the payload outright.
pub async fn validate_pwa_import(progress_json: String) -> Result<PwaImportResult, String> {
    if !is_tauri() {
        return Err("Not running in the native app — validate_pwa_import is Tauri-only".into());
    }

    let args = ValidatePwaImportArgs { progress_json };
    let args_js = serde_wasm_bindgen::to_value(&args).map_err(|e| e.to_string())?;

    let result = tauri_invoke("validate_pwa_import", args_js)
        .await
        .map_err(|e| js_value_to_string(&e))?;

    serde_wasm_bindgen::from_value(result).map_err(|e| e.to_string())
}

fn js_value_to_string(val: &JsValue) -> String {
    if let Some(s) = val.as_string() {
        s
    } else {
        format!("{val:?}")
    }
}
