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
use std::net::IpAddr;
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

/// Cap on a single relay/scanner WebSocket message. Shared with
/// `manage.rs` and `install.rs`'s own relay/scanner connections -- all
/// three talk to a locally-configured or user-editable relay URL, which
/// must not be able to make a tab parse and hold an unbounded JSON
/// payload.
pub(crate) const MAX_RELAY_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_DISKS: usize = 64;
const MAX_PARTITIONS_PER_DISK: usize = 128;
const MAX_DEVICE_NAME_BYTES: usize = 64;

fn relay_message_allowed(state: &RelayState, msg_type: &str) -> bool {
    match msg_type {
        "connected" => matches!(state, RelayState::Connecting),
        "hardware_probe" | "disks" => matches!(state, RelayState::Probing),
        "progress" | "output" | "exit" => {
            matches!(state, RelayState::Installing | RelayState::Reconnecting(_))
        }
        _ => true,
    }
}

fn bounded_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn parse_size(value: Option<&serde_json::Value>) -> Result<u64, String> {
    match value {
        Some(value) if value.is_u64() => value.as_u64().ok_or_else(|| "invalid size".into()),
        Some(value) => value
            .as_str()
            .ok_or_else(|| "size must be an integer or decimal string".to_string())?
            .parse::<u64>()
            .map_err(|_| "invalid decimal size".to_string()),
        None => Err("missing size".into()),
    }
}

fn parse_device_name(value: Option<&serde_json::Value>) -> Result<String, String> {
    let name = value
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "missing device name".to_string())?;
    let valid = !name.is_empty()
        && name.len() <= MAX_DEVICE_NAME_BYTES
        && !name.starts_with('.')
        && !name.contains("..")
        && name
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'));
    if !valid {
        return Err(format!("unsafe device name: {name:?}"));
    }
    Ok(name.to_string())
}

fn parse_disk_list(data: &str) -> Result<Vec<DiskInfo>, String> {
    let disks: Vec<serde_json::Value> =
        serde_json::from_str(data).map_err(|e| format!("invalid disk response: {e}"))?;
    if disks.is_empty() {
        return Err("relay reported no installable disks".into());
    }
    if disks.len() > MAX_DISKS {
        return Err(format!("relay reported more than {MAX_DISKS} disks"));
    }

    disks
        .into_iter()
        .map(|disk| {
            let name = parse_device_name(disk.get("name"))?;
            let size_bytes = parse_size(disk.get("size"))?;
            if size_bytes == 0 {
                return Err(format!("disk {name:?} has zero size"));
            }

            let children = disk
                .get("children")
                .and_then(serde_json::Value::as_array)
                .map(Vec::as_slice)
                .unwrap_or_default();
            if children.len() > MAX_PARTITIONS_PER_DISK {
                return Err(format!(
                    "disk {name:?} has more than {MAX_PARTITIONS_PER_DISK} partitions"
                ));
            }
            let partitions = children
                .iter()
                .map(|child| {
                    let mount = child
                        .get("mountpoint")
                        .or_else(|| child.get("mountpoints"))
                        .and_then(|value| {
                            value.as_str().or_else(|| {
                                value
                                    .as_array()
                                    .and_then(|mounts| mounts.first())
                                    .and_then(serde_json::Value::as_str)
                            })
                        })
                        .unwrap_or_default();
                    Ok(PartitionInfo {
                        name: parse_device_name(child.get("name"))?,
                        size_bytes: parse_size(child.get("size"))?,
                        fs_type: bounded_text(
                            child
                                .get("fstype")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default(),
                            32,
                        ),
                        label: bounded_text(
                            child
                                .get("label")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or_default(),
                            256,
                        ),
                        mount: bounded_text(mount, 1024),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;

            Ok(DiskInfo {
                name,
                size_bytes,
                model: bounded_text(
                    disk.get("model")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("Unknown"),
                    256,
                ),
                transport: bounded_text(
                    disk.get("transport")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown"),
                    32,
                ),
                partitions,
            })
        })
        .collect()
}

fn normalize_target_address(value: &str) -> Result<String, String> {
    let address = value.trim();
    if address.is_empty() || address.len() > 253 || address != value {
        return Err("target address must be a non-empty host or IP address".into());
    }

    let ip_text = address
        .strip_prefix('[')
        .and_then(|value| value.strip_suffix(']'))
        .unwrap_or(address);
    if let Ok(ip) = ip_text.parse::<IpAddr>() {
        return Ok(match ip {
            IpAddr::V4(ip) => ip.to_string(),
            IpAddr::V6(ip) => format!("[{ip}]"),
        });
    }

    let valid_hostname = address.split('.').all(|label| {
        !label.is_empty()
            && label.len() <= 63
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
    });
    if !valid_hostname {
        return Err("target address contains invalid hostname characters".into());
    }
    Ok(address.to_ascii_lowercase())
}

fn validate_relay_url(value: &str) -> Result<(), String> {
    if value.is_empty()
        || value.len() > 2048
        || value.bytes().any(|byte| byte.is_ascii_whitespace())
    {
        return Err("relay URL is invalid".into());
    }
    if let Some(rest) = value.strip_prefix("wss://") {
        let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
        if !authority.is_empty() && !rest.contains('@') {
            return Ok(());
        }
    }
    if let Some(rest) = value.strip_prefix("ws://") {
        let authority = rest.split('/').next().unwrap_or_default();
        let loopback = authority == "localhost"
            || authority.starts_with("localhost:")
            || authority == "127.0.0.1"
            || authority.starts_with("127.0.0.1:")
            || authority == "[::1]"
            || authority.starts_with("[::1]:");
        if loopback && !rest.contains('@') {
            return Ok(());
        }
    }
    Err("relay URL must use wss:// (ws:// is allowed only for loopback)".into())
}

fn optional_string(
    value: Option<&serde_json::Value>,
    field: &str,
    max_bytes: usize,
) -> Result<String, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(String::new()),
        Some(serde_json::Value::String(value)) if value.len() <= max_bytes => Ok(value.clone()),
        Some(serde_json::Value::String(_)) => Err(format!("{field} exceeds {max_bytes} bytes")),
        Some(_) => Err(format!("{field} must be a string")),
    }
}

fn optional_bool(value: Option<&serde_json::Value>, field: &str) -> Result<bool, String> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(false),
        Some(serde_json::Value::Bool(value)) => Ok(*value),
        Some(_) => Err(format!("{field} must be a boolean")),
    }
}

fn parse_hardware_probe(data: &str) -> Result<RemoteHardware, String> {
    let data: serde_json::Value = serde_json::from_str(data)
        .map_err(|error| format!("invalid hardware response: {error}"))?;
    let object = data
        .as_object()
        .ok_or_else(|| "hardware response must be an object".to_string())?;
    let arch = optional_string(object.get("arch"), "arch", 32)?;
    if arch.is_empty()
        || !arch
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err("hardware response has an invalid or missing architecture".into());
    }

    let mut hardware = RemoteHardware {
        arch,
        tpm2: optional_bool(object.get("tpm2_available"), "tpm2_available")?,
        secure_boot: optional_bool(object.get("secure_boot"), "secure_boot")?,
        setup_mode: optional_bool(object.get("setup_mode"), "setup_mode")?,
        efi: optional_bool(object.get("efi_available"), "efi_available")?,
        ..RemoteHardware::default()
    };

    if let Some(gpu) = object.get("gpu") {
        let gpu = gpu
            .as_object()
            .ok_or_else(|| "gpu must be an object".to_string())?;
        hardware.gpu_vendor = optional_string(gpu.get("vendor"), "gpu.vendor", 128)?;
        hardware.gpu_model = optional_string(gpu.get("model"), "gpu.model", 256)?;
        hardware.gpu_hybrid = optional_bool(gpu.get("hybrid"), "gpu.hybrid")?;
    }
    if let Some(wifi) = object.get("wifi") {
        let wifi = wifi
            .as_object()
            .ok_or_else(|| "wifi must be an object".to_string())?;
        hardware.wifi_available = optional_bool(wifi.get("available"), "wifi.available")?;
        hardware.wifi_interface = optional_string(wifi.get("interface"), "wifi.interface", 64)?;
    }
    if let Some(safety) = object.get("safety") {
        let safety = safety
            .as_object()
            .ok_or_else(|| "safety must be an object".to_string())?;
        hardware.safety_level = optional_string(safety.get("level"), "safety.level", 32)?;
        hardware.safety_message = optional_string(safety.get("message"), "safety.message", 1024)?;
    }
    if let Some(oses) = object.get("detected_os") {
        let oses = oses
            .as_array()
            .ok_or_else(|| "detected_os must be an array".to_string())?;
        if oses.len() > 32 {
            return Err("detected_os contains more than 32 entries".into());
        }
        hardware.detected_os = oses
            .iter()
            .map(|os| {
                let os = os
                    .as_object()
                    .ok_or_else(|| "detected_os entries must be objects".to_string())?;
                let name = optional_string(os.get("name"), "detected_os.name", 128)?;
                if name.is_empty() {
                    return Err("detected_os names cannot be empty".into());
                }
                Ok(name)
            })
            .collect::<Result<Vec<_>, String>>()?;
    }
    if let Some(chromebook) = object.get("chromebook") {
        let chromebook = chromebook
            .as_object()
            .ok_or_else(|| "chromebook must be an object".to_string())?;
        hardware.chromebook = optional_bool(chromebook.get("detected"), "chromebook.detected")?;
    }

    Ok(hardware)
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
    let disk_confirmation = RwSignal::new(String::new());
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
    // SECURITY: SSH password is in-memory only for this signal's lifetime --
    // never written to sessionStorage. A page reload starts with an empty
    // password (matching manage.rs's `ssh_pass` pattern) rather than
    // reading a plaintext credential back out of browser storage, where
    // any XSS or malicious extension with page access could read it for
    // as long as the tab stays open.
    let (ssh_password, set_ssh_password) = signal(String::new());
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
    let auto_connect_from_pairing = RwSignal::new(false);

    // Auto-fill from URL params (QR code pairing from ISO's show-relay-url.service).
    // URL format: install.nixforhumanity.org/?target=192.168.1.5&token=abc123
    if let Some(window) = web_sys::window() {
        if let Ok(search) = window.location().search() {
            let params = web_sys::UrlSearchParams::new_with_str(&search).ok();
            if let Some(params) = params {
                let pairing_target = params.get("target").filter(|value| !value.is_empty());
                let pairing_token = params.get("token").filter(|value| !value.is_empty());

                if let Some(target) = pairing_target.as_ref() {
                    if !target.is_empty() {
                        set_target_addr.set(target.clone());
                        save_to_storage("si_target_addr", target);
                    }
                }
                if let Some(token) = pairing_token.as_ref() {
                    if !token.is_empty() {
                        set_relay_token.set(token.clone());
                        save_to_session("si_relay_token", token);
                    }
                }
                // Auto-connect if both target and token provided via QR
                if pairing_target.is_some() && pairing_token.is_some() {
                    auto_connect_from_pairing.set(true);
                    relay_state.set(RelayState::Connecting);
                }

                // Credentials in a query string otherwise persist in browser
                // history and can be copied or sent as a referrer. Preserve any
                // unrelated parameters while removing the pairing material.
                if pairing_target.is_some() || pairing_token.is_some() {
                    params.delete("target");
                    params.delete("token");
                    let mut clean_url = window.location().pathname().unwrap_or_default();
                    let remaining = String::from(params.to_string());
                    if !remaining.is_empty() {
                        clean_url.push('?');
                        clean_url.push_str(&remaining);
                    }
                    if let Ok(fragment) = window.location().hash() {
                        clean_url.push_str(&fragment);
                    }
                    if let Ok(history) = window.history() {
                        let _ = history.replace_state_with_url(
                            &wasm_bindgen::JsValue::NULL,
                            "",
                            Some(&clean_url),
                        );
                    }
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
        let addr = match normalize_target_address(&target_addr.get()) {
            Ok(address) => address,
            Err(error) => {
                relay_state.set(RelayState::Failed(error.clone()));
                set_install_log.update(|log| log.push(format!("Connection rejected: {error}")));
                return;
            }
        };
        save_to_storage("si_target_addr", &addr);

        // Auto-construct relay URL from target address (ISO runs relay on port 8094)
        let url = if !relay_url.get().is_empty() {
            relay_url.get()
        } else {
            format!("wss://{}:8094", addr)
        };
        if let Err(error) = validate_relay_url(&url) {
            relay_state.set(RelayState::Failed(error.clone()));
            set_install_log.update(|log| log.push(format!("Connection rejected: {error}")));
            return;
        }
        save_to_storage("si_relay_url", &url);
        let token = relay_token.get();
        if token.is_empty() || token.len() > 1024 || token.trim() != token {
            relay_state.set(RelayState::Failed(
                "A valid auth token is required. Copy it exactly from the relay console.".into(),
            ));
            set_install_log.update(|l| {
                l.push(
                    "Error: A valid auth token is required. Copy it exactly from the relay console."
                        .into(),
                )
            });
            return;
        }
        let password = ssh_password.get();
        if !is_reconnect && (password.is_empty() || password.len() > 1024) {
            relay_state.set(RelayState::Failed(
                "Enter the one-time password shown on the target console.".into(),
            ));
            return;
        }
        save_to_session("si_relay_token", &token);
        // SSH connects to localhost on the ISO (relay bridges to local sshd)
        let target_host = ssh_host.get();
        let target_port = match ssh_port.get().parse::<u16>() {
            Ok(port) if port > 0 => port,
            _ => {
                relay_state.set(RelayState::Failed(
                    "SSH port must be an integer from 1 to 65535.".into(),
                ));
                return;
            }
        };

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
                if text.len() > MAX_RELAY_MESSAGE_BYTES {
                    disks.set(Vec::new());
                    selected_disk.set(String::new());
                    disk_confirmation.set(String::new());
                    relay_state.set(RelayState::Failed(
                        "Relay message exceeded the 1 MiB safety limit".into(),
                    ));
                    return;
                }
                // Parse as typed RelayResponse first, fall back to Value for complex nested data
                let Ok(msg) = serde_json::from_str::<serde_json::Value>(&text) else {
                    relay_state.set(RelayState::Failed("Relay sent malformed JSON".into()));
                    return;
                };
                let typed: Option<RelayResponse> = serde_json::from_str(&text).ok();
                let msg_type = typed
                    .as_ref()
                    .map(|t| t.msg_type.as_str())
                    .unwrap_or(msg.get("type").and_then(|v| v.as_str()).unwrap_or(""));
                if !relay_message_allowed(&relay_state.get(), msg_type) {
                    return;
                }

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
                        let parsed = msg
                            .get("data")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| "hardware response omitted its data field".to_string())
                            .and_then(parse_hardware_probe);
                        let hardware = match parsed {
                            Ok(hardware) => hardware,
                            Err(error) => {
                                remote_hw.set(RemoteHardware::default());
                                disks.set(Vec::new());
                                selected_disk.set(String::new());
                                disk_confirmation.set(String::new());
                                relay_state.set(RelayState::Failed(format!(
                                    "Hardware probe failed: {error}"
                                )));
                                set_install_log.update(|log| {
                                    log.push(format!("Hardware probe rejected: {error}"))
                                });
                                persist_log_force();
                                return;
                            }
                        };
                        remote_hw.set(hardware);
                        set_install_log
                            .update(|l| l.push("Hardware probed. Discovering disks...".into()));
                        persist_log();
                        // Auto-discover disks
                        let discover = serde_json::json!({"action": "discover_disks"});
                        let _ = ws_for_msg.send_with_str(&discover.to_string());
                    }

                    "disks" => {
                        let parsed = msg
                            .get("data")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| "disk response omitted its data field".to_string())
                            .and_then(parse_disk_list);
                        let disk_list = match parsed {
                            Ok(disks) => disks,
                            Err(error) => {
                                disks.set(Vec::new());
                                selected_disk.set(String::new());
                                disk_confirmation.set(String::new());
                                relay_state.set(RelayState::Failed(format!(
                                    "Disk discovery failed: {error}"
                                )));
                                set_install_log.update(|log| {
                                    log.push(format!("Disk discovery rejected: {error}"))
                                });
                                persist_log_force();
                                return;
                            }
                        };

                        // Never preselect a destructive target, even if the
                        // relay reports only one disk.
                        selected_disk.set(String::new());
                        disk_confirmation.set(String::new());
                        disks.set(disk_list);
                        relay_state.set(RelayState::Ready);
                        set_install_log.update(|l| l.push("Ready to install.".into()));
                        persist_log();
                    }

                    "progress" => {
                        let t = typed.as_ref();
                        let stage = bounded_text(
                            t.and_then(|t| t.stage.as_deref()).unwrap_or_default(),
                            256,
                        );
                        let pct = t.and_then(|t| t.percentage).unwrap_or(0).min(100) as u32;
                        let phase = bounded_text(
                            t.and_then(|t| t.phase.as_deref()).unwrap_or_default(),
                            128,
                        );
                        let message = bounded_text(
                            t.and_then(|t| t.message.as_deref()).unwrap_or_default(),
                            1024,
                        );
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
                                    l.push(bounded_text(line, 4096));
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
                    "WebSocket connection failed. Check the relay address and console output."
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

    // Public connect entry point. A persisted in-progress marker selects the
    // status-only resume flow rather than issuing a second install connection.
    let connect = move || {
        let is_resume = previous_install_in_progress.get();
        reconnect_attempts.set(0);
        connect_inner(is_resume);
    };

    // ── Auto-connect from URL params (QR code flow) ──
    Effect::new(move |_| {
        if auto_connect_from_pairing.get() {
            auto_connect_from_pairing.set(false);
            if let Some(window) = web_sys::window() {
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
    });

    // ── Start install ──
    let start_install = move || {
        let disk = selected_disk.get();
        if relay_state.get() != RelayState::Ready
            || disk.is_empty()
            || disk_confirmation.get().trim() != disk
            || !disks.get().iter().any(|candidate| candidate.name == disk)
        {
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
        user_password.set(String::new());
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
                            ", then enter the address, one-time password, and per-run relay token shown on its console."
                        </p>
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
                                    placeholder="Shown on target console"
                                    prop:value=ssh_password
                                    on:input=move |ev| {
                                        // In-memory only -- see the signal's
                                        // definition above for why this is
                                        // never persisted to sessionStorage.
                                        set_ssh_password.set(event_target_value(&ev));
                                    }
                                />
                            </div>
                            <div class="field">
                                <label class="field-label">"Relay Token"</label>
                                <input type="password" class="field-input"
                                    aria-label="Per-run relay authentication token"
                                    autocomplete="off"
                                    placeholder="Shown on relay console"
                                    prop:value=relay_token
                                    on:input=move |ev| set_relay_token.set(event_target_value(&ev))
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
                                        placeholder="auto: wss://<target>:8094"
                                        prop:value=relay_url
                                        on:input=move |ev| set_relay_url.set(event_target_value(&ev))
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
                                                    move |_| {
                                                        selected_disk.set(n.clone());
                                                        disk_confirmation.set(String::new());
                                                    }
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
                                        "This will modify partitions on "<strong>{selected_disk.get()}</strong>
                                        " and can cause permanent data loss. Make sure you have backups."
                                    </p>
                                    <label class="field-label" for="disk-confirmation">
                                        "Type the exact disk identifier to confirm: "
                                        <code>{selected_disk.get()}</code>
                                    </label>
                                    <input id="disk-confirmation" type="text" class="field-input"
                                        autocomplete="off"
                                        prop:value=disk_confirmation
                                        on:input=move |ev| disk_confirmation.set(event_target_value(&ev))
                                    />
                                    <button class="btn-danger"
                                        prop:disabled=move || {
                                            let selected = selected_disk.get();
                                            selected.is_empty() || disk_confirmation.get().trim() != selected
                                        }
                                        on:click=move |_| start_install()
                                    >
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destructive_relay_messages_require_install_state() {
        assert!(!relay_message_allowed(&RelayState::Ready, "exit"));
        assert!(!relay_message_allowed(&RelayState::Probing, "progress"));
        assert!(relay_message_allowed(&RelayState::Installing, "exit"));
        assert!(relay_message_allowed(
            &RelayState::Reconnecting(1),
            "progress"
        ));
    }

    #[test]
    fn disk_parser_rejects_unsafe_or_incomplete_targets() {
        assert!(parse_disk_list(r#"[{"name":"../sda","size":1024}]"#).is_err());
        assert!(parse_disk_list(r#"[{"name":"sda"}]"#).is_err());
        assert!(parse_disk_list("[]").is_err());
    }

    #[test]
    fn disk_parser_accepts_lsblk_number_and_string_sizes() {
        let disks = parse_disk_list(
            r#"[{"name":"nvme0n1","size":"1000000","model":"Test","children":[{"name":"nvme0n1p1","size":500000,"fstype":"vfat"}]}]"#,
        )
        .unwrap();
        assert_eq!(disks.len(), 1);
        assert_eq!(disks[0].name, "nvme0n1");
        assert_eq!(disks[0].partitions[0].size_bytes, 500_000);
    }

    #[test]
    fn target_addresses_are_normalized_without_accepting_url_syntax() {
        assert_eq!(
            normalize_target_address("192.168.1.2").unwrap(),
            "192.168.1.2"
        );
        assert_eq!(
            normalize_target_address("HOST.local").unwrap(),
            "host.local"
        );
        assert_eq!(
            normalize_target_address("2001:db8::1").unwrap(),
            "[2001:db8::1]"
        );
        assert!(normalize_target_address("host/path").is_err());
        assert!(normalize_target_address("user@host").is_err());
        assert!(normalize_target_address(" host ").is_err());
    }

    #[test]
    fn relay_urls_require_transport_security_off_loopback() {
        assert!(validate_relay_url("wss://relay.example:8094").is_ok());
        assert!(validate_relay_url("ws://127.0.0.1:8094").is_ok());
        assert!(validate_relay_url("ws://[::1]:8094").is_ok());
        assert!(validate_relay_url("ws://192.168.1.2:8094").is_err());
        assert!(validate_relay_url("https://relay.example").is_err());
        assert!(validate_relay_url("wss://user@relay.example").is_err());
        assert!(validate_relay_url("wss:///missing-host").is_err());
    }

    #[test]
    fn hardware_probe_requires_typed_bounded_evidence() {
        let hardware = parse_hardware_probe(
            r#"{"arch":"x86_64","tpm2_available":true,"detected_os":[{"name":"NixOS"}]}"#,
        )
        .unwrap();
        assert_eq!(hardware.arch, "x86_64");
        assert!(hardware.tpm2);
        assert_eq!(hardware.detected_os, vec!["NixOS"]);

        assert!(parse_hardware_probe(r#"{"tpm2_available":true}"#).is_err());
        assert!(parse_hardware_probe(r#"{"arch":"x86_64","secure_boot":"yes"}"#).is_err());
        assert!(parse_hardware_probe(r#"{"arch":"../../bin/sh"}"#).is_err());
    }
}
