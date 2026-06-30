// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Remote Install — WebSocket client for the SSH relay.
//!
//! Security posture (Stop-the-Bleeding):
//! - The SSH relay runs on the *operator machine* and binds to 127.0.0.1.
//! - The relay requires a per-run WebSocket token (`action: "auth"`).
//! - The target machine only exposes SSH with a one-time password shown on its console.
//!
//! This panel connects to the local relay and asks it to SSH into the target.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

use crate::components::glass_panel::GlassPanel;

// ═══════════════════════════════════════════════════════
// Storage helpers — localStorage for non-sensitive data,
// sessionStorage for credentials (dies with tab close)
// ═══════════════════════════════════════════════════════

fn save_to_storage(key: &str, value: &str) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = storage.set_item(key, value);
    }
}

fn load_from_storage(key: &str) -> Option<String> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(key).ok().flatten())
}

fn remove_from_storage(key: &str) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = storage.remove_item(key);
    }
}

/// Save sensitive data to sessionStorage (cleared when tab closes).
/// Prevents credential persistence across browser sessions.
fn save_to_session(key: &str, value: &str) {
    if let Some(storage) = web_sys::window().and_then(|w| w.session_storage().ok().flatten()) {
        let _ = storage.set_item(key, value);
    }
}

fn load_from_session(key: &str) -> Option<String> {
    web_sys::window()
        .and_then(|w| w.session_storage().ok().flatten())
        .and_then(|s| s.get_item(key).ok().flatten())
}

// ═══════════════════════════════════════════════════════
// Typed relay messages (match server-side RelayMessage)
// ═══════════════════════════════════════════════════════

/// Typed relay response — deserialized from WebSocket JSON.
/// Eliminates ad-hoc js_sys::Reflect::get() parsing.
#[derive(serde::Deserialize, Debug, Clone)]
pub struct RelayResponse {
    #[serde(rename = "type")]
    pub msg_type: String,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub stream: Option<String>,
    #[serde(default)]
    pub code: Option<i64>,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub percentage: Option<u8>,
    #[serde(default)]
    pub phase: Option<String>,
    // Hardware probe fields
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub ram_gb: Option<f64>,
    #[serde(default)]
    pub gpu: Option<String>,
    #[serde(default)]
    pub disks: Option<Vec<serde_json::Value>>,
    // Backup fields
    #[serde(default)]
    pub detail: Option<String>,
    // Error
    #[serde(default)]
    pub error: Option<String>,
}

// ═══════════════════════════════════════════════════════
// State types
// ═══════════════════════════════════════════════════════

#[derive(Clone, Debug, PartialEq)]
pub enum RelayState {
    Disconnected,
    Connecting,
    Connected,
    Probing,
    Ready, // Hardware probed, disks discovered
    Installing,
    Reconnecting(u32), // attempt number (1-3)
    Complete,
    Failed(String),
}

#[derive(Clone, Debug, Default)]
pub struct RemoteHardware {
    pub gpu_vendor: String,
    pub gpu_model: String,
    pub gpu_hybrid: bool,
    pub tpm2: bool,
    pub secure_boot: bool,
    pub setup_mode: bool,
    pub efi: bool,
    pub wifi_available: bool,
    pub wifi_interface: String,
    pub arch: String,
    pub safety_level: String,
    pub safety_message: String,
    pub detected_os: Vec<String>,
    pub chromebook: bool,
}

#[derive(Clone, Debug)]
pub struct PartitionInfo {
    pub name: String,
    pub size_bytes: u64,
    pub fs_type: String,
    pub label: String,
    pub mount: String,
}

#[derive(Clone, Debug)]
pub struct DiskInfo {
    pub name: String,
    pub size_bytes: u64,
    pub model: String,
    pub transport: String,
    pub partitions: Vec<PartitionInfo>,
}

impl DiskInfo {
    pub fn size_display(&self) -> String {
        let gb = self.size_bytes as f64 / 1_073_741_824.0;
        if gb >= 1000.0 {
            format!("{:.1} TB", gb / 1024.0)
        } else {
            format!("{:.0} GB", gb)
        }
    }
}

fn partition_color(fs_type: &str, label: &str) -> &'static str {
    let lower_label = label.to_lowercase();
    let lower_fs = fs_type.to_lowercase();
    if lower_fs == "vfat" && (lower_label.contains("efi") || lower_label.contains("boot")) {
        "rgba(100, 149, 237, 0.7)" // blue - EFI
    } else if lower_fs == "ntfs" || lower_label.contains("windows") {
        "rgba(232, 167, 71, 0.7)" // orange - Windows
    } else if lower_fs == "ext4" || lower_fs == "btrfs" || lower_fs == "xfs" || lower_fs == "zfs" {
        "rgba(90, 184, 160, 0.7)" // teal - Linux
    } else if lower_fs == "swap" {
        "rgba(180, 120, 200, 0.5)" // purple - swap
    } else if lower_fs.is_empty() {
        "rgba(126, 200, 160, 0.3)" // light green - unformatted
    } else {
        "rgba(150, 150, 150, 0.5)" // gray - other
    }
}

fn human_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000_000 {
        format!("{:.1} TB", bytes as f64 / 1e12)
    } else if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.0} MB", bytes as f64 / 1e6)
    } else {
        format!("{} B", bytes)
    }
}

#[derive(Clone, Debug)]
pub struct InstallProgress {
    pub stage: String,
    pub percentage: u32,
    pub phase: String,
    pub message: String,
}

// ═══════════════════════════════════════════════════════
// Partition bar visualization
// ═══════════════════════════════════════════════════════

#[component]
fn PartitionBar(
    disk_name: String,
    disk_size: u64,
    partitions: Vec<PartitionInfo>,
) -> impl IntoView {
    let total = disk_size.max(1);

    // Calculate free space
    let used: u64 = partitions.iter().map(|p| p.size_bytes).sum();
    let free = total.saturating_sub(used);

    // Build partition segments
    let segments: Vec<_> = partitions
        .iter()
        .map(|p| {
            let width_pct = (p.size_bytes as f64 / total as f64 * 100.0).max(2.0);
            let color = partition_color(&p.fs_type, &p.label);
            let display_label = if p.label.is_empty() {
                if p.fs_type.is_empty() {
                    p.name.clone()
                } else {
                    p.fs_type.clone()
                }
            } else {
                p.label.clone()
            };
            let size = human_size(p.size_bytes);
            let title = format!("/dev/{} ({}, {})", p.name, display_label, size);
            (width_pct, color, display_label, size, title)
        })
        .collect();

    // Build legend entries
    let legend: Vec<_> = partitions
        .iter()
        .map(|p| {
            let color = partition_color(&p.fs_type, &p.label);
            let display_label = if p.label.is_empty() {
                p.name.clone()
            } else {
                p.label.clone()
            };
            let fs = if p.fs_type.is_empty() {
                "unknown".to_string()
            } else {
                p.fs_type.clone()
            };
            let size = human_size(p.size_bytes);
            (color, format!("{} ({}, {})", display_label, fs, size))
        })
        .collect();

    view! {
        <div class="partition-editor">
            <div class="partition-disk-label">
                {format!("/dev/{} ({})", disk_name, human_size(disk_size))}
            </div>
            <div class="partition-bar">
                {segments.into_iter().map(|(width_pct, color, label, size, title)| {
                    view! {
                        <div class="partition-segment"
                            style=format!("width: {:.1}%; background: {};", width_pct, color)
                            title=title
                        >
                            <span class="partition-seg-label">{label}</span>
                            <span class="partition-seg-size">{size}</span>
                        </div>
                    }
                }).collect::<Vec<_>>()}
                {(free > 1024 * 1024).then(|| {
                    let width_pct = (free as f64 / total as f64 * 100.0).max(2.0);
                    let free_size = human_size(free);
                    let free_title = format!("Free space ({})", &free_size);
                    view! {
                        <div class="partition-segment partition-free"
                            style=format!("width: {:.1}%;", width_pct)
                            title=free_title
                        >
                            <span class="partition-seg-label">"Free"</span>
                            <span class="partition-seg-size">{free_size}</span>
                        </div>
                    }
                })}
            </div>
            <div class="partition-legend">
                {legend.into_iter().map(|(color, text)| {
                    view! {
                        <span class="partition-legend-item">
                            <span class="partition-legend-dot" style=format!("background: {};", color)></span>
                            {text}
                        </span>
                    }
                }).collect::<Vec<_>>()}
                {(free > 1024 * 1024).then(|| {
                    view! {
                        <span class="partition-legend-item">
                            <span class="partition-legend-dot" style="background: rgba(126, 200, 160, 0.3);"></span>
                            {format!("Free ({})", human_size(free))}
                        </span>
                    }
                })}
            </div>
        </div>
    }
}

// ═══════════════════════════════════════════════════════
// Relay connection component
// ═══════════════════════════════════════════════════════

#[component]
pub fn RemoteInstallPanel(
    /// User's generated configuration.nix
    config_nix: Signal<Option<String>>,
    /// User's generated flake.nix
    flake_nix: Signal<Option<String>>,
    /// Choices for the install
    hostname: RwSignal<String>,
    desktop: RwSignal<String>,
    gpu_driver: Signal<String>,
    timezone: RwSignal<String>,
    keyboard: RwSignal<String>,
    encrypt: RwSignal<bool>,
    secure_boot: RwSignal<bool>,
    tpm_unlock: RwSignal<bool>,
    fido2_unlock: RwSignal<bool>,
    disk_layout: RwSignal<String>,
    filesystem: RwSignal<String>,
    /// User password (NOT persisted to localStorage)
    user_password: RwSignal<String>,
) -> impl IntoView {
    // Connection state
    let relay_state = RwSignal::new(RelayState::Disconnected);
    let remote_hw = RwSignal::new(RemoteHardware::default());
    let disks = RwSignal::new(Vec::<DiskInfo>::new());
    let selected_disk = RwSignal::new(String::new());
    let reconnect_attempts = RwSignal::new(0u32);

    // Backup state (Tier 1: pre-install full backup)
    let backup_status = RwSignal::new(Option::<String>::None);
    let backup_complete = RwSignal::new(false);

    // Migration scan state (Tier 1: data migration)
    let scan_data = RwSignal::new(Option::<String>::None);

    // Restore install log from localStorage on mount (Phase 4.2)
    let restored_log: Vec<String> = load_from_storage("si_install_log")
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let (install_log, set_install_log) = signal(restored_log);

    let progress = RwSignal::new(InstallProgress {
        stage: String::new(),
        percentage: 0,
        phase: String::new(),
        message: String::new(),
    });

    // Resume detection (Phase 4.3): check if a previous install was in progress
    let previous_install_in_progress =
        RwSignal::new(load_from_storage("si_install_in_progress").as_deref() == Some("true"));

    // Connection form — direct to ISO mode
    // User enters the target machine's IP and the relay token from console output.
    // SECURITY: No default token — user must paste from relay's stdout.
    let saved_target = load_from_storage("si_target_addr")
        .unwrap_or_else(|| "sovereign-inoculation.local".to_string());
    let (target_addr, set_target_addr) = signal(saved_target);
    // SSH password restored from sessionStorage (survives page refresh, cleared on tab close)
    let saved_ssh_pw = load_from_session("si_ssh_password").unwrap_or_default();
    let (ssh_password, set_ssh_password) = signal(saved_ssh_pw);
    // Advanced mode: allow custom relay URL (hidden by default)
    let show_advanced = RwSignal::new(false);
    let saved_relay_url = load_from_storage("si_relay_url").unwrap_or_default();
    let (relay_url, set_relay_url) = signal(saved_relay_url);
    // SECURITY: Token in sessionStorage (cleared on tab close, not persisted)
    let saved_token = load_from_session("si_relay_token").unwrap_or_default();
    let (relay_token, set_relay_token) = signal(saved_token);
    let (ssh_host, set_ssh_host) = signal("127.0.0.1".to_string());
    let (ssh_port, set_ssh_port) = signal("22".to_string());
    let (ssh_user, set_ssh_user) = signal("root".to_string());

    // Auto-fill from URL params (QR code pairing from ISO's show-relay-url.service).
    // URL format: install.nixforhumanity.org/?target=192.168.1.5&token=abc123
    if let Some(window) = web_sys::window() {
        if let Ok(search) = window.location().search() {
            let params = web_sys::UrlSearchParams::new_with_str(&search).ok();
            if let Some(params) = params {
                if let Some(target) = params.get("target") {
                    if !target.is_empty() {
                        set_target_addr.set(target.clone());
                        save_to_storage("si_target_addr", &target);
                    }
                }
                if let Some(token) = params.get("token") {
                    if !token.is_empty() {
                        set_relay_token.set(token.clone());
                        save_to_session("si_relay_token", &token);
                    }
                }
                // Auto-connect if both target and token provided via QR
                if params.get("target").is_some() && params.get("token").is_some() {
                    // Signal auto-connect (handled by effect below)
                    relay_state.set(RelayState::Connecting);
                }
            }
        }
    }

    // Store WebSocket in JS global to avoid Send/Sync issues
    fn store_ws(ws: &web_sys::WebSocket) {
        let window = web_sys::window().unwrap();
        let _ = js_sys::Reflect::set(&window, &"__sovereign_ws".into(), ws);
    }
    fn send_msg(msg: &serde_json::Value) {
        let window = web_sys::window().unwrap();
        if let Ok(ws_val) = js_sys::Reflect::get(&window, &"__sovereign_ws".into()) {
            if let Ok(ws) = ws_val.dyn_into::<web_sys::WebSocket>() {
                let _ = ws.send_with_str(&msg.to_string());
            }
        }
    }

    // Helper: persist install log to localStorage (batched — every 10th call to reduce I/O)
    let log_write_counter = RwSignal::new(0u32);
    let persist_log = move || {
        log_write_counter.update(|c| *c += 1);
        // Only write every 10 updates to reduce localStorage thrashing
        if log_write_counter.get() % 10 == 0 {
            let log = install_log.get();
            save_to_storage(
                "si_install_log",
                &serde_json::to_string(&log).unwrap_or_default(),
            );
        }
    };
    // Force-persist: always writes regardless of counter (use on completion/error)
    let persist_log_force = move || {
        let log = install_log.get();
        save_to_storage(
            "si_install_log",
            &serde_json::to_string(&log).unwrap_or_default(),
        );
    };

    // ── Connect to relay ──
    let connect_inner = move |is_reconnect: bool| {
        let addr = target_addr.get();
        if addr.trim().is_empty() {
            relay_state.set(RelayState::Failed(
                "Enter the target machine's IP address".into(),
            ));
            return;
        }
        save_to_storage("si_target_addr", &addr);

        // Auto-construct relay URL from target address (ISO runs relay on port 8094)
        let url = if !relay_url.get().is_empty() {
            relay_url.get()
        } else {
            format!("wss://{}:8094", addr)
        };
        save_to_storage("si_relay_url", &url);
        let token = relay_token.get();
        if token.is_empty() {
            relay_state.set(RelayState::Failed(
                "Auth token is required. Copy it from the relay's console output.".into(),
            ));
            set_install_log.update(|l| {
                l.push(
                    "Error: Auth token is required. Copy it from the relay's console output."
                        .into(),
                )
            });
            return;
        }
        save_to_session("si_relay_token", &token);
        // SSH connects to localhost on the ISO (relay bridges to local sshd)
        let target_host = ssh_host.get();
        let target_port: u16 = ssh_port.get().parse().unwrap_or(22);

        if !is_reconnect {
            relay_state.set(RelayState::Connecting);
            set_install_log.update(|l| l.push(format!("Connecting to {}...", url)));
            persist_log();
            // Save relay URL for resume (Phase 4.3)
            save_to_storage("si_relay_url", &url);
        }

        let ws = match web_sys::WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(_) => {
                if is_reconnect {
                    // Reconnect failed at WebSocket creation — will be retried by schedule_reconnect
                    relay_state.set(RelayState::Failed(
                        "Reconnect failed. The install may still be running on the target machine."
                            .into(),
                    ));
                } else {
                    relay_state.set(RelayState::Failed("WebSocket creation failed".into()));
                }
                return;
            }
        };

        // onopen — auth to relay, then connect or check status
        let ws_for_open = ws.clone();
        let token_clone = token.clone();
        let user = ssh_user.get();
        let password = ssh_password.get();
        let target_host_clone = target_host.clone();
        let onopen = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            // Always auth first
            let auth = serde_json::json!({
                "action": "auth",
                "token": token_clone,
            });
            let _ = ws_for_open.send_with_str(&auth.to_string());

            if is_reconnect {
                // On reconnect, check install status instead of starting fresh
                reconnect_attempts.set(0);
                relay_state.set(RelayState::Installing);
                set_install_log
                    .update(|l| l.push("Reconnected. Checking install status...".into()));
                persist_log();
                let status = serde_json::json!({"action": "status"});
                let _ = ws_for_open.send_with_str(&status.to_string());
            } else {
                let msg = serde_json::json!({
                    "action": "connect",
                    "host": target_host_clone,
                    "port": target_port,
                    "username": user,
                    "password": password
                });
                let _ = ws_for_open.send_with_str(&msg.to_string());
            }
        });
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // onmessage — route all relay messages
        let ws_for_msg = ws.clone();
        let onmessage = wasm_bindgen::closure::Closure::<dyn Fn(web_sys::MessageEvent)>::new(
            move |e: web_sys::MessageEvent| {
                let Some(text) = e.data().as_string() else {
                    return;
                };
                // Parse as typed RelayResponse first, fall back to Value for complex nested data
                let Ok(msg) = serde_json::from_str::<serde_json::Value>(&text) else {
                    return;
                };
                let typed: Option<RelayResponse> = serde_json::from_str(&text).ok();
                let msg_type = typed
                    .as_ref()
                    .map(|t| t.msg_type.as_str())
                    .unwrap_or(msg.get("type").and_then(|v| v.as_str()).unwrap_or(""));

                match msg_type {
                    "connected" => {
                        relay_state.set(RelayState::Probing);
                        set_install_log
                            .update(|l| l.push("SSH connected. Probing hardware...".into()));
                        persist_log();
                        // Auto-probe hardware
                        let probe = serde_json::json!({"action": "probe_hardware"});
                        let _ = ws_for_msg.send_with_str(&probe.to_string());
                    }

                    "hardware_probe" => {
                        if let Some(data_str) = msg.get("data").and_then(|v| v.as_str()) {
                            if let Ok(data) = serde_json::from_str::<serde_json::Value>(data_str) {
                                // Parse hardware
                                let mut hw = RemoteHardware::default();
                                if let Some(gpu) = data.get("gpu") {
                                    hw.gpu_vendor = gpu
                                        .get("vendor")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .into();
                                    hw.gpu_model = gpu
                                        .get("model")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .into();
                                    hw.gpu_hybrid = gpu
                                        .get("hybrid")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false);
                                }
                                hw.tpm2 = data
                                    .get("tpm2_available")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                hw.secure_boot = data
                                    .get("secure_boot")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                hw.setup_mode = data
                                    .get("setup_mode")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                hw.efi = data
                                    .get("efi_available")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                hw.arch = data
                                    .get("arch")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("x86_64")
                                    .into();
                                if let Some(wifi) = data.get("wifi") {
                                    hw.wifi_available = wifi
                                        .get("available")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false);
                                    hw.wifi_interface = wifi
                                        .get("interface")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .into();
                                }
                                if let Some(safety) = data.get("safety") {
                                    hw.safety_level = safety
                                        .get("level")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("clear")
                                        .into();
                                    hw.safety_message = safety
                                        .get("message")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .into();
                                }
                                if let Some(oses) =
                                    data.get("detected_os").and_then(|v| v.as_array())
                                {
                                    hw.detected_os = oses
                                        .iter()
                                        .filter_map(|o| {
                                            o.get("name").and_then(|v| v.as_str()).map(String::from)
                                        })
                                        .collect();
                                }
                                if let Some(cb) = data.get("chromebook") {
                                    hw.chromebook = cb
                                        .get("detected")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false);
                                }
                                remote_hw.set(hw);
                            }
                        }
                        set_install_log
                            .update(|l| l.push("Hardware probed. Discovering disks...".into()));
                        persist_log();
                        // Auto-discover disks
                        let discover = serde_json::json!({"action": "discover_disks"});
                        let _ = ws_for_msg.send_with_str(&discover.to_string());
                    }

                    "disks" => {
                        if let Some(data_str) = msg.get("data").and_then(|v| v.as_str()) {
                            if let Ok(arr) =
                                serde_json::from_str::<Vec<serde_json::Value>>(data_str)
                            {
                                let disk_list: Vec<DiskInfo> = arr
                                    .iter()
                                    .filter_map(|d| {
                                        let name = d.get("name")?.as_str()?.to_string();
                                        // size may be string or number (lsblk -b returns numbers)
                                        let size_bytes = match d.get("size") {
                                            Some(v) if v.is_u64() => v.as_u64().unwrap_or(0),
                                            Some(v) => v
                                                .as_str()
                                                .unwrap_or("0")
                                                .parse::<u64>()
                                                .unwrap_or(0),
                                            None => 0,
                                        };
                                        let model = d
                                            .get("model")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("Unknown")
                                            .to_string();
                                        let transport = d
                                            .get("transport")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("unknown")
                                            .to_string();
                                        // Parse partition children from lsblk JSON
                                        let partitions = d
                                            .get("children")
                                            .and_then(|c| c.as_array())
                                            .map(|children| {
                                                children
                                                    .iter()
                                                    .filter_map(|child| {
                                                        let pname = child
                                                            .get("name")?
                                                            .as_str()?
                                                            .to_string();
                                                        let psize = match child.get("size") {
                                                            Some(v) if v.is_u64() => {
                                                                v.as_u64().unwrap_or(0)
                                                            }
                                                            Some(v) => v
                                                                .as_str()
                                                                .unwrap_or("0")
                                                                .parse::<u64>()
                                                                .unwrap_or(0),
                                                            None => 0,
                                                        };
                                                        let fs_type = child
                                                            .get("fstype")
                                                            .and_then(|v| v.as_str())
                                                            .unwrap_or("")
                                                            .to_string();
                                                        let label = child
                                                            .get("label")
                                                            .and_then(|v| v.as_str())
                                                            .unwrap_or("")
                                                            .to_string();
                                                        let mount = child
                                                            .get("mountpoint")
                                                            .or_else(|| child.get("mountpoints"))
                                                            .and_then(|v| {
                                                                if let Some(s) = v.as_str() {
                                                                    Some(s.to_string())
                                                                } else if let Some(arr) =
                                                                    v.as_array()
                                                                {
                                                                    arr.first()
                                                                        .and_then(|m| m.as_str())
                                                                        .map(String::from)
                                                                } else {
                                                                    None
                                                                }
                                                            })
                                                            .unwrap_or_default();
                                                        Some(PartitionInfo {
                                                            name: pname,
                                                            size_bytes: psize,
                                                            fs_type,
                                                            label,
                                                            mount,
                                                        })
                                                    })
                                                    .collect::<Vec<_>>()
                                            })
                                            .unwrap_or_default();
                                        Some(DiskInfo {
                                            name,
                                            size_bytes,
                                            model,
                                            transport,
                                            partitions,
                                        })
                                    })
                                    .collect();

                                if disk_list.len() == 1 {
                                    selected_disk.set(disk_list[0].name.clone());
                                }
                                disks.set(disk_list);
                            }
                        }
                        relay_state.set(RelayState::Ready);
                        set_install_log.update(|l| l.push("Ready to install.".into()));
                        persist_log();
                    }

                    "progress" => {
                        let t = typed.as_ref();
                        let stage = t.and_then(|t| t.stage.clone()).unwrap_or_default();
                        let pct = t.and_then(|t| t.percentage).unwrap_or(0) as u32;
                        let phase = t.and_then(|t| t.phase.clone()).unwrap_or_default();
                        let message = t.and_then(|t| t.message.clone()).unwrap_or_default();
                        progress.set(InstallProgress {
                            stage: stage.clone(),
                            percentage: pct,
                            phase,
                            message,
                        });
                        set_install_log.update(|l| l.push(format!("[{}%] {}", pct, stage)));
                        persist_log();
                    }

                    "output" => {
                        let line_opt = typed
                            .as_ref()
                            .and_then(|t| t.data.as_deref())
                            .or_else(|| msg.get("data").and_then(|v| v.as_str()));
                        if let Some(line) = line_opt {
                            if !line.trim().is_empty() {
                                set_install_log.update(|l| {
                                    l.push(line.to_string());
                                    // Cap at 1000 lines to prevent memory exhaustion from relay flooding
                                    if l.len() > 1000 {
                                        l.drain(..200);
                                    }
                                });
                                persist_log();
                            }
                        }
                    }

                    "exit" => {
                        let code = typed.as_ref().and_then(|t| t.code).unwrap_or(-1);
                        if code == 0 {
                            relay_state.set(RelayState::Complete);
                            progress.set(InstallProgress {
                                stage: "Complete".into(),
                                percentage: 100,
                                phase: "FirstBreath".into(),
                                message: "NixOS installed successfully!".into(),
                            });
                            set_install_log.update(|l| {
                                l.push("Installation complete! Reboot to start NixOS.".into())
                            });
                            // Clear in-progress flag (Phase 4.3)
                            remove_from_storage("si_install_in_progress");
                            previous_install_in_progress.set(false);
                        } else {
                            relay_state.set(RelayState::Failed(format!(
                                "Install exited with code {code}"
                            )));
                            set_install_log.update(|l| {
                                l.push(format!("Installation failed (exit code {code})"))
                            });
                            remove_from_storage("si_install_in_progress");
                            previous_install_in_progress.set(false);
                        }
                        persist_log_force();
                    }

                    "data_preserved" => {
                        let detail = msg
                            .get("data")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Backup complete");
                        backup_status.set(Some(detail.to_string()));
                        backup_complete.set(true);
                        set_install_log.update(|l| l.push(format!("Backup: {detail}")));
                        persist_log();
                    }

                    "app_scan" => {
                        let data = msg.get("data").and_then(|v| v.as_str()).unwrap_or("");
                        scan_data.update(|existing| {
                            let prev = existing.clone().unwrap_or_default();
                            *existing = Some(format!(
                                "{}--- Installed Apps ---\n{}",
                                if prev.is_empty() {
                                    String::new()
                                } else {
                                    format!("{prev}\n")
                                },
                                data
                            ));
                        });
                        set_install_log.update(|l| l.push("App scan complete.".into()));
                        persist_log();
                    }

                    "deep_scan" => {
                        let data = msg.get("data").and_then(|v| v.as_str()).unwrap_or("");
                        scan_data.update(|existing| {
                            let prev = existing.clone().unwrap_or_default();
                            *existing = Some(format!(
                                "{}--- Deep Scan (configs, keys, dev envs) ---\n{}",
                                if prev.is_empty() {
                                    String::new()
                                } else {
                                    format!("{prev}\n")
                                },
                                data
                            ));
                        });
                        set_install_log.update(|l| l.push("Deep scan complete.".into()));
                        persist_log();
                    }

                    "error" => {
                        let err = msg
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error");
                        relay_state.set(RelayState::Failed(err.to_string()));
                        set_install_log.update(|l| l.push(format!("Error: {err}")));
                        persist_log();
                    }

                    _ => {}
                }
            },
        );
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        // onerror — if installing, attempt reconnect (Phase 4.1)
        let onerror = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            let current = relay_state.get();
            let was_installing = matches!(
                current,
                RelayState::Installing | RelayState::Reconnecting(_)
            );
            if was_installing {
                let attempts = reconnect_attempts.get();
                if attempts < 3 {
                    let next = attempts + 1;
                    reconnect_attempts.set(next);
                    relay_state.set(RelayState::Reconnecting(next));
                    set_install_log.update(|l| {
                        l.push(format!(
                            "Connection lost. Reconnecting... (attempt {}/3)",
                            next
                        ))
                    });
                    persist_log();
                    // Schedule reconnect after 2^attempt seconds
                    let delay_ms = (1u32 << next) * 1000;
                    let cb = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
                        // Reload to trigger resume flow (Phase 4.3)
                        let _ = web_sys::window().unwrap().location().reload();
                    });
                    let _ = web_sys::window()
                        .unwrap()
                        .set_timeout_with_callback_and_timeout_and_arguments_0(
                            cb.as_ref().unchecked_ref(),
                            delay_ms as i32,
                        );
                    cb.forget();
                } else {
                    relay_state.set(RelayState::Failed(
                        "Connection lost. The install may still be running on the target machine. \
                         Check the target's console or reconnect manually."
                            .into(),
                    ));
                    set_install_log.update(|l| l.push("All reconnect attempts failed.".into()));
                    persist_log();
                }
            } else {
                relay_state.set(RelayState::Failed(
                    "WebSocket connection failed. Is ssh-relay running locally (ws://127.0.0.1:8091)?"
                        .into(),
                ));
            }
        });
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();

        // onclose — handle clean close during install (Phase 4.1)
        let onclose = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            let current = relay_state.get();
            let was_installing = matches!(
                current,
                RelayState::Installing | RelayState::Reconnecting(_)
            );
            if was_installing {
                let attempts = reconnect_attempts.get();
                if attempts < 3 {
                    let next = attempts + 1;
                    reconnect_attempts.set(next);
                    relay_state.set(RelayState::Reconnecting(next));
                    set_install_log.update(|l| {
                        l.push(format!(
                            "Connection closed. Reconnecting... (attempt {}/3)",
                            next
                        ))
                    });
                    persist_log();
                    let delay_ms = (1u32 << next) * 1000;
                    let cb = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
                        // Reload to trigger resume flow (Phase 4.3)
                        let _ = web_sys::window().unwrap().location().reload();
                    });
                    let _ = web_sys::window()
                        .unwrap()
                        .set_timeout_with_callback_and_timeout_and_arguments_0(
                            cb.as_ref().unchecked_ref(),
                            delay_ms as i32,
                        );
                    cb.forget();
                } else {
                    relay_state.set(RelayState::Failed(
                        "Connection lost. The install may still be running on the target machine. \
                         Check the target's console or reconnect manually."
                            .into(),
                    ));
                    set_install_log.update(|l| l.push("All reconnect attempts failed.".into()));
                    persist_log_force();
                }
            }
            // If not installing, closing is expected (Complete, Failed, etc.)
        });
        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
        onclose.forget();

        store_ws(&ws);
    };

    // Public connect entry point (fresh connection, not reconnect)
    let connect = move || {
        reconnect_attempts.set(0);
        // Dismiss resume banner when user explicitly connects
        previous_install_in_progress.set(false);
        connect_inner(false);
    };

    // ── Auto-connect from URL params (QR code flow) ──
    Effect::new(move |_| {
        if let Some(window) = web_sys::window() {
            if let Ok(search) = window.location().search() {
                if search.contains("target=") {
                    let params = web_sys::UrlSearchParams::new_with_str(&search).ok();
                    if let Some(params) = params {
                        if let Some(target) = params.get("target") {
                            set_target_addr.set(target);
                        }
                        if let Some(token) = params.get("token") {
                            set_relay_token.set(token);
                        }
                        // Auto-connect after a short delay
                        let cb = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
                            connect();
                        });
                        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                            cb.as_ref().unchecked_ref(),
                            500,
                        );
                        cb.forget();
                    }
                }
            }
        }
    });

    // ── Auto-discover target on local network ──
    let scanning_network = RwSignal::new(false);
    let scan_result = RwSignal::new(Option::<String>::None);

    let start_scan = move || {
        // Don't scan if already connected or have URL params
        if relay_state.get() == RelayState::Connected {
            return;
        }
        if let Some(window) = web_sys::window() {
            if let Ok(search) = window.location().search() {
                if search.contains("target=") {
                    return;
                }
            }
        }

        scanning_network.set(true);
        scan_result.set(None);

        // Scan common private subnets (most home/office networks)
        // Only scan first 20 IPs per subnet, staggered in batches of 10
        let prefixes = ["192.168.1", "192.168.0", "10.0.0"];
        let mut ip_list_parts = Vec::new();
        for prefix in prefixes {
            for host in 1..=20u16 {
                ip_list_parts.push(format!("\"{}:8094\"", format!("{}.{}", prefix, host)));
            }
        }
        let ip_list_js = ip_list_parts.join(",");

        // Single JS eval that batches 10 connections at a time with 200ms stagger
        let scan_js = format!(
            r#"
            (function() {{
                var ips = [{ips}];
                var batch = 0;
                function scanBatch() {{
                    var start = batch * 10;
                    var end = Math.min(start + 10, ips.length);
                    for (var i = start; i < end; i++) {{
                        try {{
                            var addr = ips[i];
                            var ws = new WebSocket("wss://" + addr);
                            var timer = setTimeout(function() {{ ws.close(); }}, 1500);
                            ws.onopen = function() {{
                                ws.send(JSON.stringify({{action:"auth",token:"sovereign"}}));
                            }};
                            ws.onmessage = (function(a, t) {{
                                return function(ev) {{
                                    clearTimeout(t);
                                    try {{
                                        var msg = JSON.parse(ev.data);
                                        if (msg.type === "authed") {{
                                            window.__sovereign_discovered = a.split(":")[0];
                                        }}
                                    }} catch(e) {{}}
                                    ws.close();
                                }};
                            }})(addr, timer);
                            ws.onerror = (function(t) {{
                                return function() {{ clearTimeout(t); ws.close(); }};
                            }})(timer);
                        }} catch(e) {{}}
                    }}
                    batch++;
                    if (batch * 10 < ips.length) {{
                        setTimeout(scanBatch, 200);
                    }}
                }}
                scanBatch();
            }})()
            "#,
            ips = ip_list_js
        );
        let _ = js_sys::eval(&scan_js);

        // Poll for discovery result
        let cb = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
            let window = web_sys::window().unwrap();
            if let Ok(val) = js_sys::Reflect::get(&window, &"__sovereign_discovered".into()) {
                if let Some(ip) = val.as_string() {
                    if !ip.is_empty() {
                        scan_result.set(Some(ip.clone()));
                        set_target_addr.set(ip);
                        scanning_network.set(false);
                        // Clean up
                        let _ = js_sys::Reflect::delete_property(
                            &window,
                            &"__sovereign_discovered".into(),
                        );
                        return;
                    }
                }
            }
            // Keep polling for 10 seconds
            scanning_network.set(false);
        });
        let window = web_sys::window().unwrap();
        let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
            cb.as_ref().unchecked_ref(),
            5000,
        );
        cb.forget();
    };

    // Auto-scan on mount (only if no URL params and not resuming)
    Effect::new(move |_| {
        let _ = relay_state.get(); // subscribe
        if relay_state.get() == RelayState::Disconnected
            && !previous_install_in_progress.get()
            && target_addr.get() == "sovereign-inoculation.local"
        {
            start_scan();
        }
    });

    // ── Start install ──
    let start_install = move || {
        let disk = selected_disk.get();
        if disk.is_empty() {
            return;
        }

        relay_state.set(RelayState::Installing);
        reconnect_attempts.set(0);

        // Clear old log and start fresh (Phase 4.2)
        set_install_log.update(|l| {
            l.clear();
            l.push(format!("Starting install on {}...", disk));
        });
        persist_log_force();

        // Mark install as in-progress for resume detection (Phase 4.3)
        save_to_storage("si_install_in_progress", "true");

        // When ZFS is selected and layout is "single", use "single-zfs"
        let layout = {
            let base = disk_layout.get();
            if filesystem.get() == "zfs" && (base == "single" || base.is_empty()) {
                "single-zfs".to_string()
            } else {
                base
            }
        };
        let msg = serde_json::json!({
            "action": "install",
            "disk": disk,
            "layout": layout,
            "hostname": hostname.get(),
            "desktop": desktop.get(),
            "gpu_driver": gpu_driver.get(),
            "timezone": timezone.get(),
            "keyboard": keyboard.get(),
            "secure_boot": secure_boot.get(),
            "tpm2_unlock": tpm_unlock.get(),
            "fido2_unlock": fido2_unlock.get(),
            "configuration_nix": config_nix.get().unwrap_or_default(),
            "flake_nix": flake_nix.get().unwrap_or_default(),
            "user_password": user_password.get(),
        });
        send_msg(&msg);
    };

    // Auto-detect removed: previous design probed a relay running on the target ISO via mDNS.
    // New security posture is local relay + explicit token.

    view! {
        <GlassPanel title="Automated Install">
            // ── Resume banner (Phase 4.3) ──
            {move || (previous_install_in_progress.get()
                && matches!(relay_state.get(), RelayState::Disconnected | RelayState::Failed(_))
            ).then(|| {
                view! {
                    <div class="resume-banner">
                        <p>"A previous install may still be in progress. Connect to check status."</p>
                    </div>
                }
            })}

            // ── Connection form (simplified — direct to ISO) ──
            {move || (relay_state.get() == RelayState::Disconnected || matches!(relay_state.get(), RelayState::Failed(_))).then(|| {
                view! {
                    <div class="remote-connect">
                        <p class="section-desc">
                            "Boot the target machine from the "
                            <a href="https://github.com/Luminous-Dynamics/nixforhumanity/releases" target="_blank">"NixForHumanity USB"</a>
                            ". We'll try to find it on your network automatically."
                        </p>

                        // Network scan status
                        {move || scanning_network.get().then(|| view! {
                            <div class="auto-detect-banner">
                                <div class="spinner"></div>
                                <span>"Scanning your network for NixOS targets..."</span>
                            </div>
                        })}
                        {move || scan_result.get().map(|ip| view! {
                            <div class="auto-detect-banner" style="background: rgba(126, 200, 160, 0.1); border-color: rgba(126, 200, 160, 0.3);">
                                <span>{format!("Found NixOS target at {}", ip)}</span>
                                <button class="btn-primary btn-sm" style="margin-left: auto;"
                                    on:click=move |_| connect()
                                >"Connect"</button>
                            </div>
                        })}
                        {move || (!scanning_network.get() && scan_result.get().is_none() && relay_state.get() == RelayState::Disconnected).then(|| view! {
                            <button class="btn-secondary btn-sm" style="margin-bottom: 0.5rem;"
                                on:click=move |_| start_scan()
                            >"Scan Network"</button>
                        })}
                        <div class="connect-form">
                            <div class="field" style="flex: 2;">
                                <label class="field-label">"Target Address"</label>
                                <input type="text" class="field-input"
                                    aria-label="Target machine IP address"
                                    placeholder="192.168.1.100 or sovereign-inoculation.local"
                                    prop:value=target_addr
                                    on:input=move |ev| set_target_addr.set(event_target_value(&ev))
                                />
                            </div>
                            <div class="field">
                                <label class="field-label">"Password"</label>
                                <input type="password" class="field-input"
                                    aria-label="Target machine password"
                                    placeholder="sovereign"
                                    prop:value=ssh_password
                                    on:input=move |ev| {
                                        let v = event_target_value(&ev);
                                        save_to_session("si_ssh_password", &v);
                                        set_ssh_password.set(v);
                                    }
                                />
                            </div>
                        </div>
                        <button class="btn-primary" style="margin-top: 0.8rem;"
                            on:click=move |_| connect()
                        >"Connect"</button>

                        // Advanced settings (hidden by default)
                        <details class="help-expander" style="margin-top: 0.5rem;">
                            <summary class="help-toggle">"Advanced connection settings"</summary>
                            <div class="connect-form" style="margin-top: 0.5rem;">
                                <div class="field">
                                    <label class="field-label">"Relay URL (override)"</label>
                                    <input type="text" class="field-input"
                                        placeholder="auto: ws://<target>:8094"
                                        prop:value=relay_url
                                        on:input=move |ev| set_relay_url.set(event_target_value(&ev))
                                    />
                                </div>
                                <div class="field">
                                    <label class="field-label">"Auth Token (from relay console)"</label>
                                    <input type="text" class="field-input"
                                        placeholder="Paste token from relay startup output"
                                        prop:value=relay_token
                                        on:input=move |ev| set_relay_token.set(event_target_value(&ev))
                                    />
                                </div>
                                <div class="field">
                                    <label class="field-label">"SSH Port"</label>
                                    <input type="text" class="field-input" style="width: 5rem;"
                                        prop:value=ssh_port
                                        on:input=move |ev| set_ssh_port.set(event_target_value(&ev))
                                    />
                                </div>
                            </div>
                        </details>

                        {move || {
                            if let RelayState::Failed(ref msg) = relay_state.get() {
                                let msg_clone = msg.clone();
                                Some(view! {
                                    <div class="error-recovery">
                                        <p class="error-msg">{msg_clone}</p>
                                        <div class="error-actions">
                                            <button class="btn btn-primary"
                                                on:click=move |_| {
                                                    relay_state.set(RelayState::Disconnected);
                                                    set_install_log.update(|l| l.push("Manual reconnect requested...".into()));
                                                }
                                            >"Reconnect"</button>
                                            <p class="error-hint">
                                                "If the install was in progress, it may still be running on the target. "
                                                "Check the target machine's console for status."
                                            </p>
                                        </div>
                                    </div>
                                })
                            } else { None }
                        }}
                    </div>
                }
            })}

            // ── Probing indicator ──
            {move || (relay_state.get() == RelayState::Connecting || relay_state.get() == RelayState::Probing).then(|| {
                view! {
                    <div class="probing-status">
                        <div class="spinner"></div>
                        <span>{move || match relay_state.get() {
                            RelayState::Connecting => "Connecting to target...".to_string(),
                            RelayState::Probing => "Detecting hardware and disks...".to_string(),
                            _ => String::new(),
                        }}</span>
                    </div>
                }
            })}

            // ── Hardware + Disk selection (Ready state) ──
            {move || (relay_state.get() == RelayState::Ready).then(|| {
                let hw = remote_hw.get();
                let disk_list = disks.get();
                view! {
                    <div class="remote-ready">
                        // Safety warning
                        {(hw.safety_level != "clear").then(|| {
                            let class = match hw.safety_level.as_str() {
                                "blocked" => "safety-blocked",
                                "warning" => "safety-warning",
                                _ => "safety-caution",
                            };
                            view! {
                                <div class={format!("safety-banner {class}")}>
                                    <strong>{format!("Safety: {}", hw.safety_level.to_uppercase())}</strong>
                                    <p>{hw.safety_message.clone()}</p>
                                </div>
                            }
                        })}

                        // Chromebook detection banner
                        {hw.chromebook.then(|| view! {
                            <div class="safety-banner safety-caution">
                                <strong>"Chromebook Detected"</strong>
                                <p>"Your Chromebook uses non-standard firmware. For best results, replace it with UEFI firmware from "
                                    <a href="https://mrchromebox.tech" target="_blank">"MrChromebox"</a>
                                    " before installing NixOS."
                                </p>
                            </div>
                        })}

                        // Detected hardware summary
                        <div class="remote-hw-summary">
                            <span class="hw-chip">{hw.arch.clone()}</span>
                            {(!hw.gpu_model.is_empty()).then(|| view! {
                                <span class="hw-chip">{hw.gpu_model.clone()}</span>
                            })}
                            {hw.efi.then(|| view! { <span class="hw-chip">"EFI"</span> })}
                            {hw.tpm2.then(|| view! { <span class="hw-chip">"TPM2"</span> })}
                            {hw.wifi_available.then(|| view! { <span class="hw-chip">"WiFi"</span> })}
                            {(!hw.detected_os.is_empty()).then(|| view! {
                                <span class="hw-chip hw-chip-warn">{format!("Existing: {}", hw.detected_os.join(", "))}</span>
                            })}
                        </div>

                        // OS detection (Tier 1): show detected operating systems
                        {(!hw.detected_os.is_empty()).then(|| {
                            let os_list = hw.detected_os.clone();
                            view! {
                                <div class="os-detection">
                                    <h4>"Existing Operating Systems"</h4>
                                    <p class="section-desc">
                                        "The following operating systems were detected on this machine."
                                    </p>
                                    <div class="remote-hw-summary">
                                        {os_list.iter().map(|os| view! {
                                            <span class="hw-chip hw-chip-warn">{os.clone()}</span>
                                        }).collect::<Vec<_>>()}
                                    </div>
                                </div>
                            }
                        })}

                        // Disk selection with partition visualization
                        <h4 class="subsection-title">"Select target disk"</h4>
                        <div class="disk-list">
                            {disk_list.iter().map(|d| {
                                let name = d.name.clone();
                                let name2 = d.name.clone();
                                let disk_label = format!("{} — {} ({})", d.model, d.size_display(), d.transport);
                                let bar_name = d.name.clone();
                                let bar_size = d.size_bytes;
                                let bar_parts = d.partitions.clone();
                                view! {
                                    <label class="disk-option" class:disk-selected=move || selected_disk.get() == name>
                                        <div class="disk-option-header">
                                            <input type="radio" name="target-disk"
                                                prop:checked=move || selected_disk.get() == name2
                                                on:change={
                                                    let n = d.name.clone();
                                                    move |_| selected_disk.set(n.clone())
                                                }
                                            />
                                            <span class="disk-option-label">{disk_label}</span>
                                        </div>
                                        <PartitionBar
                                            disk_name=bar_name
                                            disk_size=bar_size
                                            partitions=bar_parts
                                        />
                                    </label>
                                }
                            }).collect::<Vec<_>>()}
                        </div>

                        // Alongside-install warning when existing OS detected
                        {move || {
                            let sel = selected_disk.get();
                            let layout = disk_layout.get();
                            if layout == "alongside" && !sel.is_empty() {
                                let d_list = disks.get();
                                let has_existing = d_list.iter()
                                    .find(|d| d.name == sel)
                                    .map(|d| d.partitions.iter().any(|p| {
                                        p.fs_type == "ntfs" || p.fs_type == "ext4" || p.fs_type == "btrfs"
                                    }))
                                    .unwrap_or(false);
                                has_existing.then(|| view! {
                                    <div class="safety-banner safety-caution" style="margin-top: 0.5rem;">
                                        <strong>"Alongside Install"</strong>
                                        <p>"The largest existing partition will be shrunk to make room for NixOS. Ensure you have backups."</p>
                                    </div>
                                })
                            } else {
                                None
                            }
                        }}

                        // Backup section (Tier 1: pre-install full backup)
                        <div class="backup-section">
                            <h4>"Protect Your Data"</h4>
                            <p class="section-desc">
                                "Back up your existing data before installing. This saves databases, SSH keys, configs, and more to a recovery archive."
                            </p>
                            <div class="manage-actions">
                                <button class="usb-btn-secondary" on:click=move |_| {
                                    send_msg(&serde_json::json!({"action": "preserve_data"}));
                                    backup_status.set(Some("Backing up...".into()));
                                    backup_complete.set(false);
                                }>"Back Up Current System"</button>
                            </div>
                            {move || backup_status.get().map(|s| view! { <p class="action-status">{s}</p> })}
                            {move || backup_complete.get().then(|| view! {
                                <p class="success-msg">"Backup complete. Safe to proceed with install."</p>
                            })}
                        </div>

                        // Migration section (Tier 1: data migration scan)
                        <div class="migration-section">
                            <h4>"Migrate Your Data"</h4>
                            <p class="section-desc">
                                "Detect your apps, configs, SSH keys, and dev environments for migration."
                            </p>
                            <div class="manage-actions">
                                <button class="usb-btn-secondary" on:click=move |_| {
                                    scan_data.set(None);
                                    send_msg(&serde_json::json!({"action": "scan_apps"}));
                                    send_msg(&serde_json::json!({"action": "deep_scan"}));
                                }>"Scan for Migration Data"</button>
                            </div>
                            {move || scan_data.get().map(|data| view! {
                                <pre class="manage-output" style="max-height: 200px;">{data}</pre>
                            })}
                        </div>

                        {move || (!selected_disk.get().is_empty()).then(|| {
                            view! {
                                <div class="install-confirm">
                                    <p class="install-warning">
                                        "This will ERASE ALL DATA on "<strong>{selected_disk.get()}</strong>
                                        ". Make sure you have backups."
                                    </p>
                                    <button class="btn-danger" on:click=move |_| start_install()>
                                        "Install NixOS Now"
                                    </button>
                                </div>
                            }
                        })}
                    </div>
                }
            })}

            // ── Install progress ──
            {move || (relay_state.get() == RelayState::Installing).then(|| {
                let p = progress.get();
                view! {
                    <div class="install-progress">
                        <div class="progress-header">
                            <span class="progress-stage">{p.stage.clone()}</span>
                            <span class="progress-pct">{format!("{}%", p.percentage)}</span>
                        </div>
                        <div class="progress-bar-outer"
                            role="progressbar"
                            aria-valuenow=p.percentage
                            aria-valuemin="0"
                            aria-valuemax="100"
                        >
                            <div class="progress-bar-inner"
                                style=move || format!("width: {}%", progress.get().percentage)
                            ></div>
                        </div>
                        <p class="progress-phase">{p.phase.clone()}</p>
                    </div>
                }
            })}

            // ── Reconnecting indicator (Phase 4.1) ──
            {move || {
                if let RelayState::Reconnecting(attempt) = relay_state.get() {
                    Some(view! {
                        <div class="probing-status">
                            <div class="spinner"></div>
                            <span>{format!("Connection lost. Reconnecting... (attempt {}/3)", attempt)}</span>
                        </div>
                    })
                } else {
                    None
                }
            }}

            // ── Complete (Phase 5.1 — first-boot guide) ──
            {move || (relay_state.get() == RelayState::Complete).then(|| {
                let host = hostname.get();
                view! {
                    <div class="install-complete">
                        <h3>"NixOS Installed Successfully!"</h3>
                        <p>"Remove the USB drive and reboot your machine."</p>

                        <div class="first-boot-guide">
                            <h4>"After First Boot"</h4>
                            <div class="guide-steps">
                                <div class="guide-step">
                                    <span class="guide-num">"1"</span>
                                    <div>
                                        <strong>"Set your password"</strong>
                                        <p>"Login as root, then run:"</p>
                                        <code>{format!("passwd {}", host)}</code>
                                    </div>
                                </div>
                                <div class="guide-step">
                                    <span class="guide-num">"2"</span>
                                    <div>
                                        <strong>"Connect to WiFi"</strong>
                                        <code>"nmcli device wifi connect YOUR_NETWORK password YOUR_PASSWORD"</code>
                                    </div>
                                </div>
                                <div class="guide-step">
                                    <span class="guide-num">"3"</span>
                                    <div>
                                        <strong>"Update your system"</strong>
                                        <code>"sudo nixos-rebuild switch --upgrade"</code>
                                    </div>
                                </div>
                                <div class="guide-step">
                                    <span class="guide-num">"4"</span>
                                    <div>
                                        <strong>"If something breaks"</strong>
                                        <p>"Roll back to the previous version:"</p>
                                        <code>"sudo nixos-rebuild switch --rollback"</code>
                                        <p class="help-text">"Or select a previous generation from the boot menu."</p>
                                    </div>
                                </div>
                                <div class="guide-step">
                                    <span class="guide-num">"5"</span>
                                    <div>
                                        <strong>"Install more software"</strong>
                                        <p>"Edit "</p><code>"/etc/nixos/configuration.nix"</code>
                                        <p>" and add packages to "</p><code>"environment.systemPackages"</code>
                                        <p>", then run "</p><code>"sudo nixos-rebuild switch"</code>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                }
            })}

            // ── Install log (collapsible) ──
            {move || {
                let log = install_log.get();
                (!log.is_empty()).then(|| view! {
                    <details class="install-log-details">
                        <summary>"Install Log"</summary>
                        <pre class="install-log" role="log" aria-live="polite">{
                            log.iter().rev().take(100).rev().cloned().collect::<Vec<_>>().join("\n")
                        }</pre>
                    </details>
                })
            }}
        </GlassPanel>
    }
}
