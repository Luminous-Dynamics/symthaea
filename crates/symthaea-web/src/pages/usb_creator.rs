// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! USB Creator Page — Leptos reactive UI for NixForHumanity Tauri app
//!
//! Single-page component: select USB drive, optional app scan, flash NixOS ISO.
//! All Tauri communication goes through JS interop (no Tauri Rust deps in WASM).

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

// ═══════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq)]
pub enum FlashState {
    Ready,
    Downloading(u32),
    Verifying,
    Flashing(u32),
    Complete,
    Error(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct UsbDevice {
    pub id: String,
    pub name: String,
    pub size_gb: f64,
}

// ═══════════════════════════════════════════════════════
// Tauri JS interop
// ═══════════════════════════════════════════════════════

fn invoke_tauri(cmd: &str, args: &serde_json::Value) -> js_sys::Promise {
    let window = web_sys::window().unwrap();
    let tauri = js_sys::Reflect::get(&window, &"__TAURI__".into()).unwrap();
    let core = js_sys::Reflect::get(&tauri, &"core".into()).unwrap();
    let invoke = js_sys::Reflect::get(&core, &"invoke".into()).unwrap();
    let invoke_fn: js_sys::Function = invoke.dyn_into().unwrap();
    let args_js = serde_wasm_bindgen::to_value(args).unwrap();
    invoke_fn
        .call2(&wasm_bindgen::JsValue::NULL, &cmd.into(), &args_js)
        .unwrap()
        .into()
}

fn parse_usb_devices(val: wasm_bindgen::JsValue) -> Vec<UsbDevice> {
    let mut devices = Vec::new();
    if js_sys::Array::is_array(&val) {
        let arr = js_sys::Array::from(&val);
        for i in 0..arr.length() {
            let item = arr.get(i);
            let id = js_sys::Reflect::get(&item, &"id".into())
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            let name = js_sys::Reflect::get(&item, &"name".into())
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();
            let size_gb = js_sys::Reflect::get(&item, &"size_gb".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            devices.push(UsbDevice { id, name, size_gb });
        }
    }
    devices
}

// ═══════════════════════════════════════════════════════
// Component
// ═══════════════════════════════════════════════════════

#[component]
pub fn UsbCreatorPage() -> impl IntoView {
    let flash_state = RwSignal::new(FlashState::Ready);
    let usb_devices = RwSignal::new(Vec::<UsbDevice>::new());
    let selected_device = RwSignal::new(Option::<String>::None);
    let scan_results = RwSignal::new(Option::<String>::None);
    let scan_count = RwSignal::new(0u32);
    let scanning = RwSignal::new(false);
    let refreshing = RwSignal::new(false);

    // ── Refresh USB devices ──
    let refresh_devices = move |_| {
        refreshing.set(true);
        wasm_bindgen_futures::spawn_local(async move {
            let promise = invoke_tauri("list_usb_devices", &serde_json::json!({}));
            match JsFuture::from(promise).await {
                Ok(result) => {
                    let devices = parse_usb_devices(result);
                    // Clear selection if the selected device disappeared
                    if let Some(sel) = selected_device.get() {
                        if !devices.iter().any(|d| d.id == sel) {
                            selected_device.set(None);
                        }
                    }
                    usb_devices.set(devices);
                }
                Err(e) => {
                    let msg = e
                        .as_string()
                        .unwrap_or_else(|| "Failed to list USB devices".into());
                    log::warn!("USB scan error: {}", msg);
                    usb_devices.set(vec![]);
                }
            }
            refreshing.set(false);
        });
    };

    // ── Scan installed apps ──
    let start_scan = move |_| {
        scanning.set(true);
        scan_results.set(None);
        scan_count.set(0);
        wasm_bindgen_futures::spawn_local(async move {
            let promise = invoke_tauri("scan_installed_apps", &serde_json::json!({}));
            match JsFuture::from(promise).await {
                Ok(result) => {
                    let count = js_sys::Reflect::get(&result, &"count".into())
                        .ok()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as u32;
                    let summary = js_sys::Reflect::get(&result, &"summary".into())
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_else(|| format!("{} apps detected", count));
                    scan_count.set(count);
                    scan_results.set(Some(summary));
                }
                Err(e) => {
                    let msg = e.as_string().unwrap_or_else(|| "Scan failed".into());
                    scan_results.set(Some(format!("Error: {}", msg)));
                }
            }
            scanning.set(false);
        });
    };

    // ── Flash USB ──
    let start_flash = move |_| {
        let device_id = match selected_device.get() {
            Some(id) => id,
            None => return,
        };

        flash_state.set(FlashState::Downloading(0));

        wasm_bindgen_futures::spawn_local(async move {
            // Step 1: Download ISO
            let promise = invoke_tauri(
                "download_iso",
                &serde_json::json!({ "device_id": device_id }),
            );

            // Poll progress via a listener (Tauri events) — simplified here
            // The Tauri backend emits progress events; we poll state
            match JsFuture::from(promise).await {
                Ok(_) => {
                    flash_state.set(FlashState::Downloading(100));
                }
                Err(e) => {
                    let msg = e.as_string().unwrap_or_else(|| "Download failed".into());
                    flash_state.set(FlashState::Error(msg));
                    return;
                }
            }

            // Step 2: Verify checksum
            flash_state.set(FlashState::Verifying);
            let promise = invoke_tauri("verify_iso", &serde_json::json!({}));
            match JsFuture::from(promise).await {
                Ok(_) => {}
                Err(e) => {
                    let msg = e
                        .as_string()
                        .unwrap_or_else(|| "Verification failed".into());
                    flash_state.set(FlashState::Error(msg));
                    return;
                }
            }

            // Step 3: Flash to USB
            flash_state.set(FlashState::Flashing(0));
            let promise = invoke_tauri(
                "flash_usb",
                &serde_json::json!({
                    "device_id": device_id,
                    "app_scan": scan_results.get_untracked(),
                }),
            );
            match JsFuture::from(promise).await {
                Ok(_) => {
                    flash_state.set(FlashState::Complete);
                }
                Err(e) => {
                    let msg = e.as_string().unwrap_or_else(|| "Flash failed".into());
                    flash_state.set(FlashState::Error(msg));
                }
            }
        });
    };

    // ── Initial device scan on mount ──
    Effect::new(move |_| {
        refreshing.set(true);
        wasm_bindgen_futures::spawn_local(async move {
            let promise = invoke_tauri("list_usb_devices", &serde_json::json!({}));
            match JsFuture::from(promise).await {
                Ok(result) => {
                    usb_devices.set(parse_usb_devices(result));
                }
                Err(_) => {
                    usb_devices.set(vec![]);
                }
            }
            refreshing.set(false);
        });
    });

    // ── Derived state ──
    let is_busy = move || {
        matches!(
            flash_state.get(),
            FlashState::Downloading(_) | FlashState::Verifying | FlashState::Flashing(_)
        )
    };
    let can_flash = move || selected_device.get().is_some() && !is_busy();

    view! {
        <div class="usb-creator">
            <header class="usb-hero">
                <h1>"NixForHumanity"</h1>
                <p class="usb-sub">"Create a NixOS USB drive in one click"</p>
            </header>

            // ── Section 1: USB Selection ──
            <div class="usb-section">
                <div class="usb-section-header">
                    <h2>"1. Select USB Drive"</h2>
                    <button
                        class="usb-refresh-btn"
                        on:click=refresh_devices
                        disabled=move || refreshing.get() || is_busy()
                    >
                        {move || if refreshing.get() { "Scanning..." } else { "Refresh" }}
                    </button>
                </div>
                <p class="section-desc">
                    "Plug in a USB drive (4 GB minimum). Everything on it will be erased."
                </p>

                <div class="usb-device-list">
                    {move || {
                        let devices = usb_devices.get();
                        if devices.is_empty() {
                            view! {
                                <p class="usb-no-devices">"No USB drives detected. Plug one in and click Refresh."</p>
                            }.into_any()
                        } else {
                            view! {
                                <For
                                    each=move || usb_devices.get()
                                    key=|d| d.id.clone()
                                    children=move |device| {
                                        let dev_id = device.id.clone();
                                        let dev_id_select = device.id.clone();
                                        let dev_id_check = device.id.clone();
                                        let too_small = device.size_gb < 4.0;
                                        let is_selected_class = move || selected_device.get().as_deref() == Some(dev_id.as_str());
                                        let is_selected_checked = move || selected_device.get().as_deref() == Some(dev_id_check.as_str());
                                        view! {
                                            <label
                                                class="usb-device"
                                                class:usb-device-selected=is_selected_class
                                                class:usb-device-small=too_small
                                            >
                                                <input
                                                    type="radio"
                                                    name="usb-device"
                                                    value=dev_id_select.clone()
                                                    disabled=too_small || is_busy()
                                                    on:change={
                                                        let id = dev_id_select.clone();
                                                        move |_| selected_device.set(Some(id.clone()))
                                                    }
                                                    checked=is_selected_checked
                                                />
                                                <div class="usb-device-info">
                                                    <span class="usb-device-name">{device.name.clone()}</span>
                                                    <span class="usb-device-size">{format!("{:.1} GB", device.size_gb)}</span>
                                                </div>
                                                {if too_small {
                                                    view! { <span class="usb-device-warn">"Too small (need 4 GB)"</span> }.into_any()
                                                } else {
                                                    view! { <span></span> }.into_any()
                                                }}
                                            </label>
                                        }
                                    }
                                />
                            }.into_any()
                        }
                    }}
                </div>
            </div>

            // ── Section 2: App Scan (optional) ──
            <div class="usb-section">
                <div class="usb-section-header">
                    <h2>"2. Scan Your Apps"</h2>
                    <span class="optional-badge">"optional"</span>
                </div>
                <p class="section-desc">
                    "Detect your installed apps and pre-select them in the NixOS installer. "
                    "No data leaves your computer."
                </p>

                {move || {
                    if scanning.get() {
                        view! {
                            <div class="usb-scan-status">
                                <span class="usb-spinner"></span>
                                <span>"Scanning installed applications..."</span>
                            </div>
                        }.into_any()
                    } else if let Some(summary) = scan_results.get() {
                        view! {
                            <div class="usb-scan-result">
                                <span class="usb-scan-count">{format!("{} apps detected", scan_count.get())}</span>
                                <span class="usb-scan-summary">{summary}</span>
                                <button
                                    class="usb-scan-btn usb-btn-secondary"
                                    on:click=start_scan
                                    disabled=is_busy()
                                >
                                    "Re-scan"
                                </button>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <button
                                class="usb-scan-btn"
                                on:click=start_scan
                                disabled=is_busy()
                            >
                                "Scan Installed Apps"
                            </button>
                        }.into_any()
                    }
                }}
            </div>

            // ── Section 3: Create USB ──
            <div class="usb-section">
                <h2>"3. Create USB"</h2>

                {move || {
                    match flash_state.get() {
                        FlashState::Ready => view! {
                            <button
                                class="usb-flash-btn"
                                on:click=start_flash
                                disabled=move || !can_flash()
                            >
                                "Create NixOS USB"
                            </button>
                            {move || {
                                if selected_device.get().is_none() {
                                    view! { <p class="usb-hint">"Select a USB drive above to continue."</p> }.into_any()
                                } else {
                                    view! { <span></span> }.into_any()
                                }
                            }}
                        }.into_any(),

                        FlashState::Downloading(pct) => view! {
                            <div class="usb-progress-group">
                                <span class="usb-progress-label">"Downloading NixOS ISO..."</span>
                                <div class="progress-download">
                                    <div class="progress-fill" style=format!("width: {}%", pct)></div>
                                </div>
                                <span class="usb-progress-pct">{format!("{}%", pct)}</span>
                            </div>
                        }.into_any(),

                        FlashState::Verifying => view! {
                            <div class="usb-progress-group">
                                <span class="usb-progress-label">"Verifying checksum..."</span>
                                <div class="progress-download">
                                    <div class="progress-fill progress-fill-pulse" style="width: 100%"></div>
                                </div>
                            </div>
                        }.into_any(),

                        FlashState::Flashing(pct) => view! {
                            <div class="usb-progress-group">
                                <span class="usb-progress-label">"Flashing to USB drive..."</span>
                                <div class="progress-flash">
                                    <div class="progress-fill" style=format!("width: {}%", pct)></div>
                                </div>
                                <span class="usb-progress-pct">{format!("{}%", pct)}</span>
                            </div>
                        }.into_any(),

                        FlashState::Complete => view! {
                            <div class="usb-complete">
                                <h3>"USB Ready!"</h3>
                                <div class="next-steps">
                                    <p>"1. Remove the USB drive"</p>
                                    <p>"2. Plug it into the computer you want to install NixOS on"</p>
                                    <p>"3. Boot from USB (usually F12 or F2 at startup)"</p>
                                    <p>"4. Open "<a href="https://install.nixforhumanity.org" target="_blank">"install.nixforhumanity.org"</a>" on any device"</p>
                                    <p>"5. Follow the wizard to configure and install"</p>
                                </div>
                                <button
                                    class="usb-flash-btn usb-btn-secondary"
                                    on:click=move |_| {
                                        flash_state.set(FlashState::Ready);
                                        selected_device.set(None);
                                    }
                                >
                                    "Create Another"
                                </button>
                            </div>
                        }.into_any(),

                        FlashState::Error(ref msg) => view! {
                            <div class="usb-error">
                                <h3>"Error"</h3>
                                <p class="usb-error-msg">{msg.clone()}</p>
                                <button
                                    class="usb-flash-btn usb-btn-secondary"
                                    on:click=move |_| flash_state.set(FlashState::Ready)
                                >
                                    "Try Again"
                                </button>
                            </div>
                        }.into_any(),
                    }
                }}
            </div>
        </div>
    }
}
