// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Automated conductor setup — detects, connects, and configures Holochain.
//!
//! Instead of a step-by-step guide, this component:
//! 1. Probes common conductor ports (8888, 8889, 33743, etc.)
//! 2. If found: auto-connects
//! 3. If not found: offers one-click install instructions via platform detection
//! 4. Continuously polls for conductor availability

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use crate::holochain::{use_holochain, ConnectionStatus};

#[derive(Clone, PartialEq)]
enum SetupState {
    Detecting,
    NotFound,
    Found(String),
    Connected,
    Error(String),
}

#[derive(Clone, PartialEq)]
enum Platform {
    NixOS,
    Linux,
    MacOS,
    Windows,
    Unknown,
}

impl Platform {
    fn detect() -> Self {
        let ua = web_sys::window()
            .and_then(|w| w.navigator().user_agent().ok())
            .unwrap_or_default()
            .to_lowercase();

        if ua.contains("nixos") || ua.contains("linux") {
            // Can't distinguish NixOS from generic Linux via UA alone
            // NixOS users can be detected by checking for nix-specific paths
            Self::Linux
        } else if ua.contains("mac") {
            Self::MacOS
        } else if ua.contains("windows") {
            Self::Windows
        } else {
            Self::Unknown
        }
    }

    fn install_command(&self) -> &'static str {
        match self {
            Self::NixOS => "nix profile install nixpkgs#holochain",
            Self::Linux => "curl -fsSL https://get.holochain.org | bash",
            Self::MacOS => "brew install holochain",
            Self::Windows => "winget install holochain.holochain",
            Self::Unknown => "curl -fsSL https://get.holochain.org | bash",
        }
    }

    fn label(&self) -> &'static str {
        match self {
            Self::NixOS => "NixOS",
            Self::Linux => "Linux",
            Self::MacOS => "macOS",
            Self::Windows => "Windows",
            Self::Unknown => "Your system",
        }
    }
}

/// Probe a WebSocket port to check for a conductor.
async fn probe_port(port: u16) -> bool {
    let url = format!("ws://localhost:{port}");
    let result = js_sys::eval(&format!(
        "new Promise((resolve) => {{ try {{ const ws = new WebSocket('{url}'); const t = setTimeout(() => {{ ws.close(); resolve(false); }}, 2000); ws.onopen = () => {{ clearTimeout(t); ws.close(); resolve(true); }}; ws.onerror = () => {{ clearTimeout(t); resolve(false); }}; }} catch(e) {{ resolve(false); }} }})"
    ));
    match result {
        Ok(promise) => {
            let future = wasm_bindgen_futures::JsFuture::from(js_sys::Promise::from(promise));
            match future.await {
                Ok(val) => val.as_bool().unwrap_or(false),
                Err(_) => false,
            }
        }
        Err(_) => false,
    }
}

#[component]
pub fn ConductorSetup() -> impl IntoView {
    let hc = use_holochain();
    let state = RwSignal::new(SetupState::Detecting);
    let platform = Platform::detect();
    let install_cmd = platform.install_command();
    let platform_label = platform.label();
    let show = RwSignal::new(false);

    // Check if the JS bridge already connected (works on both local and remote)
    spawn_local(async move {
        // First check: did the JS bootstrap in index.html already connect?
        let js_connected = js_sys::eval("window.__HC_STATUS === 'connected' && typeof window.__HC_CALL_ZOME === 'function'")
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        if js_connected {
            web_sys::console::log_1(&"[Setup] JS bridge already connected".into());
            state.set(SetupState::Connected);
            return;
        }

        // Also check HolochainProvider status
        if hc.status.get_untracked() == ConnectionStatus::Connected {
            state.set(SetupState::Connected);
            return;
        }

        // Fallback: probe localhost ports (works on desktop, fails on mobile)
        let ports = [8888u16, 8889, 33743, 4444, 8800];
        for port in ports {
            if probe_port(port).await {
                state.set(SetupState::Found(format!("ws://localhost:{port}")));
                web_sys::console::log_1(&format!("[Setup] Conductor found on port {port}").into());
                return;
            }
        }

        // Poll for JS bridge connection (handles async race with bootstrap)
        for _ in 0..20 {
            gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;
            let connected = js_sys::eval("window.__HC_STATUS === 'connected'")
                .ok().and_then(|v| v.as_bool()).unwrap_or(false);
            if connected || hc.status.get_untracked() == ConnectionStatus::Connected {
                state.set(SetupState::Connected);
                return;
            }
        }

        state.set(SetupState::NotFound);
    });

    // Continuous polling — check every 10s if conductor appears
    spawn_local(async move {
        loop {
            gloo_timers::future::sleep(std::time::Duration::from_secs(10)).await;
            if state.get_untracked() == SetupState::Connected {
                break;
            }
            if probe_port(8888).await {
                state.set(SetupState::Found("ws://localhost:8888".into()));
            }
        }
    });

    let copy_command = move |_| {
        let _ = js_sys::eval(&format!(
            "navigator.clipboard.writeText('{}').then(() => {{}})",
            install_cmd
        ));
    };

    view! {
        // Only show when explicitly opened from settings or welcome modal
        <div class="conductor-setup" style=move || if show.get() { "" } else { "display:none" }>
            <div class="setup-overlay" on:click=move |_| show.set(false)>
                <div class="setup-modal" on:click=move |ev: leptos::ev::MouseEvent| ev.stop_propagation()>
                    <h2>"Connect to Holochain"</h2>

                    {move || match state.get() {
                        SetupState::Detecting => view! {
                            <div class="setup-status detecting">
                                <div class="setup-spinner" />
                                <p>"Scanning for Holochain conductor on your machine..."</p>
                                <p class="setup-hint">"Checking ports 8888, 8889, 33743..."</p>
                            </div>
                        }.into_any(),

                        SetupState::Found(url) => view! {
                            <div class="setup-status found">
                                <span class="setup-icon">"\u{2705}"</span>
                                <p>"Conductor detected at "<strong>{url.clone()}</strong></p>
                                <button class="btn btn-primary" on:click=move |_| {
                                    // Reload the page to trigger fresh connection
                                    let _ = web_sys::window().unwrap().location().reload();
                                }>
                                    "Connect Now"
                                </button>
                            </div>
                        }.into_any(),

                        SetupState::Connected => view! {
                            <div class="setup-status connected">
                                <span class="setup-icon">"\u{1F7E2}"</span>
                                <p>"Connected to Holochain conductor"</p>
                                <p class="setup-hint">"All messages are encrypted and stored on the DHT"</p>
                            </div>
                        }.into_any(),

                        SetupState::NotFound => view! {
                            <div class="setup-status not-found">
                                <span class="setup-icon">"\u{1F50D}"</span>
                                <p>"No conductor found on your machine"</p>
                                <p class="setup-hint">"Install Holochain to send real encrypted messages"</p>

                                <div class="install-section">
                                    <p class="install-platform">"Detected: "{platform_label}</p>
                                    <div class="install-command">
                                        <code>{install_cmd}</code>
                                        <button class="btn btn-sm btn-secondary" on:click=copy_command title="Copy to clipboard">
                                            "\u{1F4CB}"
                                        </button>
                                    </div>
                                    <p class="install-hint">
                                        "Run this command in your terminal. Once Holochain is running, "
                                        "this page will automatically detect it and connect."
                                    </p>
                                </div>

                                <div class="setup-alternative">
                                    <p class="alt-label">"Or continue in demo mode"</p>
                                    <button class="btn btn-secondary" on:click=move |_| show.set(false)>
                                        "Stay in Demo Mode"
                                    </button>
                                </div>
                            </div>
                        }.into_any(),

                        SetupState::Error(msg) => view! {
                            <div class="setup-status error">
                                <span class="setup-icon">"\u{26A0}"</span>
                                <p>"Connection error: "{msg}</p>
                                <button class="btn btn-secondary" on:click=move |_| state.set(SetupState::Detecting)>
                                    "Retry"
                                </button>
                            </div>
                        }.into_any(),
                    }}
                </div>
            </div>
        </div>
    }
}
