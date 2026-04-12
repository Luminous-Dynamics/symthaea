// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NixOS Management Page — connect to a running NixOS machine and manage it.
//! Generations, services, storage, config editing — all from the browser.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

fn save_to_storage(key: &str, value: &str) {
    if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = s.set_item(key, value);
    }
}
fn load_from_storage(key: &str) -> Option<String> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(key).ok().flatten())
}
/// Session-only storage for credentials (cleared on tab close)
fn save_to_session(key: &str, value: &str) {
    if let Some(s) = web_sys::window().and_then(|w| w.session_storage().ok().flatten()) {
        let _ = s.set_item(key, value);
    }
}
fn load_from_session(key: &str) -> Option<String> {
    web_sys::window()
        .and_then(|w| w.session_storage().ok().flatten())
        .and_then(|s| s.get_item(key).ok().flatten())
}

// ═══════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq)]
enum ConnState {
    Disconnected,
    Connecting,
    Connected,
    Failed(String),
}

#[derive(Clone, Debug, PartialEq)]
enum Tab {
    Generations,
    Services,
    Storage,
    Config,
    System,
}

// ═══════════════════════════════════════════════════════
// WebSocket helpers
// ═══════════════════════════════════════════════════════

fn send_ws(ws: &web_sys::WebSocket, msg: &serde_json::Value) {
    let _ = ws.send_with_str(&msg.to_string());
}

// ═══════════════════════════════════════════════════════
// Main Page
// ═══════════════════════════════════════════════════════

#[component]
pub fn ManagePage() -> impl IntoView {
    let conn = RwSignal::new(ConnState::Disconnected);
    let active_tab = RwSignal::new(Tab::Generations);
    // WebSocket isn't Send/Sync — use thread_local for WASM (single-threaded)
    use std::cell::RefCell;
    thread_local! {
        static WS: RefCell<Option<web_sys::WebSocket>> = RefCell::new(None);
    }

    // Connection form
    let relay_url = RwSignal::new(
        load_from_storage("mg_relay_url").unwrap_or_else(|| "ws://127.0.0.1:8094".into()),
    );
    let relay_token = RwSignal::new(load_from_session("mg_token").unwrap_or_default());
    let ssh_host =
        RwSignal::new(load_from_storage("mg_host").unwrap_or_else(|| "127.0.0.1".into()));
    let ssh_port = RwSignal::new(load_from_storage("mg_port").unwrap_or_else(|| "22".into()));
    let ssh_pass = RwSignal::new(String::new());

    // Data
    let generations = RwSignal::new(Vec::<String>::new());
    let services = RwSignal::new(Vec::<String>::new());
    let gc_data = RwSignal::new(Option::<String>::None);
    let config_text = RwSignal::new(String::new());
    let log_lines = RwSignal::new(Vec::<String>::new());
    let action_status = RwSignal::new(String::new());
    let inventory_data = RwSignal::new(Option::<String>::None);

    // Connect
    let do_connect = move || {
        let url = relay_url.get();
        let token = relay_token.get();
        let host = ssh_host.get();
        let port = ssh_port.get();
        let pass = ssh_pass.get();

        save_to_storage("mg_relay_url", &url);
        save_to_session("mg_token", &token);
        save_to_storage("mg_host", &host);
        save_to_storage("mg_port", &port);

        conn.set(ConnState::Connecting);

        let ws = match web_sys::WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(_) => {
                conn.set(ConnState::Failed("Invalid WebSocket URL".into()));
                return;
            }
        };

        let ws_clone = ws.clone();
        let onopen = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            send_ws(
                &ws_clone,
                &serde_json::json!({"action":"auth","token":token.clone()}),
            );
        });
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        let ws_clone2 = ws.clone();
        let onmessage = wasm_bindgen::closure::Closure::<dyn Fn(web_sys::MessageEvent)>::new(
            move |ev: web_sys::MessageEvent| {
                let data = ev.data().as_string().unwrap_or_default();
                let msg: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => return,
                };
                let msg_type = msg["type"].as_str().unwrap_or("");

                match msg_type {
                    "authed" => {
                        // Now connect SSH
                        send_ws(
                            &ws_clone2,
                            &serde_json::json!({
                                "action": "connect",
                                "host": host.clone(),
                                "port": port.parse::<u16>().unwrap_or(22),
                                "username": "root",
                                "password": pass.clone(),
                            }),
                        );
                    }
                    "connected" => {
                        conn.set(ConnState::Connected);
                        action_status.set("Connected to NixOS machine".into());
                    }
                    "generations" => {
                        if let Some(d) = msg["data"].as_str() {
                            generations.set(d.lines().map(|l| l.to_string()).collect::<Vec<_>>());
                            // Store raw JSON for parsing in the view
                            generations.set(vec![d.to_string()]);
                        }
                    }
                    "services" => {
                        if let Some(d) = msg["data"].as_str() {
                            services.set(vec![d.to_string()]);
                        }
                    }
                    "gc_analysis" => {
                        gc_data.set(msg["data"].as_str().map(|s| s.to_string()));
                    }
                    "config" => {
                        if let Some(d) = msg["data"].as_str() {
                            config_text.set(d.to_string());
                        }
                    }
                    "inventory" => {
                        if let Some(d) = msg["data"].as_str() {
                            inventory_data.set(Some(d.to_string()));
                        }
                    }
                    "images" => {
                        if let Some(d) = msg["data"].as_str() {
                            log_lines.update(|l| l.push(format!("Available images:\n{}", d)));
                        }
                    }
                    "output" => {
                        if let Some(d) = msg["data"].as_str() {
                            log_lines.update(|l| l.push(d.to_string()));
                        }
                    }
                    "exit" => {
                        let code = msg["code"].as_i64().unwrap_or(-1);
                        if code == 0 {
                            action_status.set("Action completed successfully".into());
                        } else {
                            action_status.set(format!("Action failed (exit code {})", code));
                        }
                    }
                    "error" => {
                        let m = msg["message"].as_str().unwrap_or("Unknown error");
                        if conn.get() != ConnState::Connected {
                            conn.set(ConnState::Failed(m.to_string()));
                        } else {
                            action_status.set(format!("Error: {}", m));
                        }
                    }
                    _ => {}
                }
            },
        );
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        let onerror = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            conn.set(ConnState::Failed("WebSocket connection failed".into()));
        });
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();

        WS.with(|cell| cell.borrow_mut().replace(ws));
    };

    // Action helpers
    let send_action = move |action: &str| {
        WS.with(|cell| {
            if let Some(ws) = cell.borrow().as_ref() {
                send_ws(ws, &serde_json::json!({"action": action}));
                action_status.set(format!("Running {}...", action));
            }
        });
    };

    let send_action_with = move |action: &str, extra: serde_json::Value| {
        WS.with(|cell| {
            if let Some(ws) = cell.borrow().as_ref() {
                let mut msg = extra;
                msg["action"] = serde_json::json!(action);
                send_ws(ws, &msg);
                action_status.set(format!("Running {}...", action));
            }
        });
    };

    view! {
        <div class="manage-page">
            <header class="manage-hero">
                <h1>"Manage NixOS"</h1>
                <p class="manage-sub">"Generations, services, storage, and config — from your browser"</p>
            </header>

            // Connection form (shown when disconnected)
            <Show when=move || conn.get() != ConnState::Connected>
                <div class="manage-connect glass-panel">
                    <h3>"Connect to your NixOS machine"</h3>
                    <p class="section-desc">"Run the SSH relay on your machine, then enter the connection details below."</p>
                    <div class="connect-grid">
                        <div class="field">
                            <label class="field-label">"Relay URL"</label>
                            <input class="field-input" prop:value=move || relay_url.get()
                                on:input=move |ev| relay_url.set(event_target_value(&ev)) />
                        </div>
                        <div class="field">
                            <label class="field-label">"Auth Token"</label>
                            <input class="field-input" prop:value=move || relay_token.get()
                                on:input=move |ev| relay_token.set(event_target_value(&ev)) />
                        </div>
                        <div class="field">
                            <label class="field-label">"SSH Host"</label>
                            <input class="field-input" prop:value=move || ssh_host.get()
                                on:input=move |ev| ssh_host.set(event_target_value(&ev)) />
                        </div>
                        <div class="field">
                            <label class="field-label">"SSH Port"</label>
                            <input class="field-input" prop:value=move || ssh_port.get()
                                on:input=move |ev| ssh_port.set(event_target_value(&ev)) />
                        </div>
                        <div class="field">
                            <label class="field-label">"Password"</label>
                            <input class="field-input" type="password" prop:value=move || ssh_pass.get()
                                on:input=move |ev| ssh_pass.set(event_target_value(&ev)) />
                        </div>
                    </div>
                    <button class="btn-primary" style="margin-top: 1rem;"
                        on:click=move |_| do_connect()>
                        {move || if conn.get() == ConnState::Connecting { "Connecting..." } else { "Connect" }}
                    </button>
                    {move || if let ConnState::Failed(ref msg) = conn.get() {
                        Some(view! { <p class="error-msg">{msg.clone()}</p> })
                    } else { None }}
                </div>
            </Show>

            // Management UI (shown when connected)
            <Show when=move || conn.get() == ConnState::Connected>
                // Tab bar
                <nav class="manage-tabs">
                    <button class=move || if active_tab.get() == Tab::Generations { "manage-tab active" } else { "manage-tab" }
                        on:click=move |_| { active_tab.set(Tab::Generations); send_action("list_generations"); }>"Generations"</button>
                    <button class=move || if active_tab.get() == Tab::Services { "manage-tab active" } else { "manage-tab" }
                        on:click=move |_| { active_tab.set(Tab::Services); send_action("list_services"); }>"Services"</button>
                    <button class=move || if active_tab.get() == Tab::Storage { "manage-tab active" } else { "manage-tab" }
                        on:click=move |_| { active_tab.set(Tab::Storage); send_action("gc_analyze"); }>"Storage"</button>
                    <button class=move || if active_tab.get() == Tab::Config { "manage-tab active" } else { "manage-tab" }
                        on:click=move |_| { active_tab.set(Tab::Config); send_action("read_config"); }>"Config"</button>
                    <button class=move || if active_tab.get() == Tab::System { "manage-tab active" } else { "manage-tab" }
                        on:click=move |_| { active_tab.set(Tab::System); send_action("inventory"); }>"System"</button>
                </nav>

                // Status bar
                {move || {
                    let status = action_status.get();
                    (!status.is_empty()).then(|| view! {
                        <p class="action-status">{status}</p>
                    })
                }}

                // Generations tab
                <Show when=move || active_tab.get() == Tab::Generations>
                    <div class="manage-panel">
                        <h3>"System Generations"</h3>
                        <p class="section-desc">"Each generation is a snapshot of your system. Roll back if something breaks."</p>
                        <div class="manage-actions">
                            <button class="btn-secondary btn-sm" on:click=move |_| send_action("list_generations")>"Refresh"</button>
                            <button class="btn-danger btn-sm" on:click=move |_| send_action("rollback")>"Rollback to Previous"</button>
                        </div>
                        <pre class="manage-output">{move || generations.get().join("\n")}</pre>
                    </div>
                </Show>

                // Services tab
                <Show when=move || active_tab.get() == Tab::Services>
                    <div class="manage-panel">
                        <h3>"System Services"</h3>
                        <p class="section-desc">"View and control systemd services running on your machine."</p>
                        <div class="manage-actions">
                            <button class="btn-secondary btn-sm" on:click=move |_| send_action("list_services")>"Refresh"</button>
                        </div>
                        <pre class="manage-output">{move || services.get().join("\n")}</pre>
                    </div>
                </Show>

                // Storage tab
                <Show when=move || active_tab.get() == Tab::Storage>
                    <div class="manage-panel">
                        <h3>"Nix Store"</h3>
                        <p class="section-desc">"Analyze and clean up your Nix store to reclaim disk space."</p>
                        <div class="manage-actions">
                            <button class="btn-secondary btn-sm" on:click=move |_| send_action("gc_analyze")>"Analyze"</button>
                            <button class="btn-danger btn-sm" on:click=move |_| {
                                log_lines.set(vec![]);
                                send_action("gc_collect");
                            }>"Clean Up (delete old generations)"</button>
                        </div>
                        {move || gc_data.get().map(|d| view! { <pre class="manage-output">{d}</pre> })}
                        {move || {
                            let lines = log_lines.get();
                            (!lines.is_empty()).then(|| view! {
                                <pre class="manage-output">{lines.join("\n")}</pre>
                            })
                        }}
                    </div>
                </Show>

                // Config tab
                <Show when=move || active_tab.get() == Tab::Config>
                    <div class="manage-panel">
                        <h3>"configuration.nix"</h3>
                        <p class="section-desc">"Edit your NixOS configuration and rebuild. Changes are backed up automatically."</p>
                        <div class="manage-actions">
                            <button class="btn-secondary btn-sm" on:click=move |_| send_action("read_config")>"Reload"</button>
                            <button class="btn-primary btn-sm" on:click=move |_| {
                                log_lines.set(vec![]);
                                send_action_with("write_config", serde_json::json!({"configuration_nix": config_text.get()}));
                            }>"Save & Rebuild"</button>
                        </div>
                        <textarea class="config-editor"
                            prop:value=move || config_text.get()
                            on:input=move |ev| config_text.set(event_target_value(&ev))
                        />
                        {move || {
                            let lines = log_lines.get();
                            (!lines.is_empty()).then(|| view! {
                                <details class="rebuild-log">
                                    <summary>"Rebuild output"</summary>
                                    <pre class="manage-output">{lines.join("\n")}</pre>
                                </details>
                            })
                        }}
                    </div>
                </Show>

                // System tab (inventory + disk cloning)
                <Show when=move || active_tab.get() == Tab::System>
                    <div class="manage-panel">
                        <h3>"System Overview"</h3>
                        <p class="section-desc">"Machine inventory and disk cloning tools."</p>
                        <div class="manage-actions">
                            <button class="btn-secondary btn-sm" on:click=move |_| send_action("inventory")>"Refresh Inventory"</button>
                            <button class="btn-secondary btn-sm" on:click=move |_| {
                                log_lines.set(vec![]);
                                send_action("create_image");
                            }>"Create System Image"</button>
                            <button class="btn-secondary btn-sm" on:click=move |_| {
                                log_lines.set(vec![]);
                                send_action("list_images");
                            }>"List Images"</button>
                        </div>
                        {move || inventory_data.get().map(|d| view! { <pre class="manage-output">{d}</pre> })}
                        {move || {
                            let lines = log_lines.get();
                            (!lines.is_empty()).then(|| view! {
                                <pre class="manage-output">{lines.join("\n")}</pre>
                            })
                        }}
                    </div>
                </Show>
            </Show>

            <footer class="install-footer">
                <a href="https://install.nixforhumanity.org">"Install NixOS"</a>
                " · "
                <a href="https://github.com/Luminous-Dynamics/nixforhumanity">"Source Code"</a>
                " · No tracking · No data collection"
            </footer>
        </div>
    }
}
