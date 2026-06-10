// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! NixOS Installer Page — Leptos reactive UI
//!
//! Complete install flow: hardware → basics → desktop → security → apps → config → next steps.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;

use crate::components::glass_panel::GlassPanel;
use crate::i18n::{self, Lang};
use crate::pages::remote_install::RemoteInstallPanel;
use crate::worker::EngineWorker;
use symthaea_app_db::config_gen;
use symthaea_app_db::validation;
use symthaea_app_db::{AppCategory, AppDatabase, AppEntry, MatchQuality};

// ═══════════════════════════════════════════════════════
// LocalStorage persistence helpers
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

// ═══════════════════════════════════════════════════════
// Hardware Detection
// ═══════════════════════════════════════════════════════

#[derive(Clone, Debug, Default)]
struct HardwareInfo {
    gpu_vendor: String,
    gpu_renderer: String,
    cpu_cores: u32,
    memory_gb: Option<f64>,
    platform: String,
    screen_width: u32,
    screen_height: u32,
}

fn detect_hardware() -> HardwareInfo {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();

    let mut info = HardwareInfo {
        cpu_cores: navigator.hardware_concurrency() as u32,
        platform: navigator.platform().unwrap_or_default(),
        ..Default::default()
    };

    let screen_w = js_sys::Reflect::get(&window, &"innerWidth".into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let screen_h = js_sys::Reflect::get(&window, &"innerHeight".into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    info.screen_width = screen_w as u32;
    info.screen_height = screen_h as u32;

    // Memory: try deviceMemory API → performance.memory → CPU-based estimate
    info.memory_gb = js_sys::Reflect::get(&navigator, &"deviceMemory".into())
        .ok()
        .and_then(|v| v.as_f64())
        .filter(|&m| m > 0.0);
    if info.memory_gb.is_none() {
        if let Ok(perf) = js_sys::Reflect::get(&window, &"performance".into()) {
            if let Ok(mem) = js_sys::Reflect::get(&perf, &"memory".into()) {
                if !mem.is_undefined() && !mem.is_null() {
                    if let Ok(heap_limit) = js_sys::Reflect::get(&mem, &"jsHeapSizeLimit".into()) {
                        if let Some(limit) = heap_limit.as_f64() {
                            let gb = limit / (1024.0 * 1024.0 * 1024.0);
                            info.memory_gb = Some(if gb > 3.5 {
                                32.0
                            } else if gb > 1.8 {
                                16.0
                            } else if gb > 0.9 {
                                8.0
                            } else if gb > 0.45 {
                                4.0
                            } else {
                                2.0
                            });
                        }
                    }
                }
            }
        }
    }
    // Last resort: estimate from CPU cores (rough heuristic, better than 0)
    if info.memory_gb.is_none() || info.memory_gb == Some(0.0) {
        let cores = info.cpu_cores;
        info.memory_gb = Some(if cores >= 16 {
            32.0
        } else if cores >= 8 {
            16.0
        } else if cores >= 4 {
            8.0
        } else {
            4.0
        });
    }

    // GPU via WebGL
    let document = window.document().unwrap();
    if let Ok(canvas) = document.create_element("canvas") {
        let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into().unwrap();
        let gl_ctx = canvas
            .get_context("webgl2")
            .ok()
            .flatten()
            .or_else(|| canvas.get_context("webgl").ok().flatten());
        if let Some(gl) = gl_ctx {
            let get_ext = js_sys::Reflect::get(&gl, &"getExtension".into()).ok();
            if let Some(f) = get_ext.and_then(|v| v.dyn_ref::<js_sys::Function>().cloned()) {
                if let Ok(ext) = f.call1(&gl, &"WEBGL_debug_renderer_info".into()) {
                    if !ext.is_null() && !ext.is_undefined() {
                        let gp = js_sys::Reflect::get(&gl, &"getParameter".into())
                            .ok()
                            .and_then(|v| v.dyn_ref::<js_sys::Function>().cloned());
                        if let Some(gp) = gp {
                            if let Ok(v) = gp.call1(&gl, &wasm_bindgen::JsValue::from_f64(37445.0))
                            {
                                info.gpu_vendor = v.as_string().unwrap_or_default();
                            }
                            if let Ok(v) = gp.call1(&gl, &wasm_bindgen::JsValue::from_f64(37446.0))
                            {
                                info.gpu_renderer = v.as_string().unwrap_or_default();
                            }
                        }
                    }
                }
            }
        }
    }
    info
}

fn gpu_short(r: &str) -> String {
    let r = r
        .replace("ANGLE (", "")
        .replace(")", "")
        .replace("/PCIe/SSE2", "")
        .replace(" Direct3D11 vs_5_0 ps_5_0", "")
        .replace("GeForce ", "")
        .replace("Mesa ", "")
        .replace("Intel(R) ", "Intel ")
        .replace("(R)", "");
    let r = if let Some(p) = r.find(", OpenGL") {
        r[..p].to_string()
    } else {
        r.to_string()
    };
    let r = if let Some(p) = r.find(", Vulkan") {
        r[..p].to_string()
    } else {
        r
    };
    r.trim().chars().take(45).collect()
}

/// Extract real GPU vendor from ANGLE-wrapped WebGL vendor string.
/// "Google Inc. (Intel)" → "intel", "Google Inc. (NVIDIA)" → "nvidia"
fn normalize_gpu_vendor(vendor: &str, renderer: &str) -> String {
    let combined = format!("{} {}", vendor, renderer).to_lowercase();
    if combined.contains("nvidia")
        || combined.contains("geforce")
        || combined.contains("rtx")
        || combined.contains("gtx")
    {
        "nvidia".into()
    } else if combined.contains("amd") || combined.contains("radeon") || combined.contains("ati") {
        "amd".into()
    } else if combined.contains("intel") {
        "intel".into()
    } else {
        "unknown".into()
    }
}

/// Auto-detect timezone from browser via Intl API.
/// Uses Function::new_no_args instead of eval() for safety.
fn detect_timezone() -> String {
    let f =
        js_sys::Function::new_no_args("return Intl.DateTimeFormat().resolvedOptions().timeZone");
    let tz = f
        .call0(&wasm_bindgen::JsValue::NULL)
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();
    // "Etc/Unknown" or empty means browser couldn't detect — fall back to UTC
    if tz.is_empty() || tz.contains("Unknown") || tz == "Etc/UTC" {
        "UTC".into()
    } else {
        tz
    }
}

/// Auto-detect keyboard layout from browser.
fn detect_keyboard() -> String {
    let nav = web_sys::window().unwrap().navigator();
    js_sys::Reflect::get(&nav, &"language".into())
        .ok()
        .and_then(|v| v.as_string())
        .map(|lang| {
            // Map browser language to keyboard layout
            match lang
                .split('-')
                .last()
                .unwrap_or("us")
                .to_lowercase()
                .as_str()
            {
                "za" => "us", // South Africa uses US layout
                "gb" | "uk" => "gb",
                "de" | "at" => "de",
                "fr" => "fr",
                "es" => "es",
                "pt" | "br" => "br-abnt2",
                "jp" => "jp",
                "kr" => "kr",
                "ru" => "ru",
                "se" => "se",
                "no" => "no",
                "dk" => "dk",
                "fi" => "fi",
                "nl" | "be" => "us", // most Dutch speakers use US
                _ => "us",
            }
            .to_string()
        })
        .unwrap_or_else(|| "us".into())
}

const TIMEZONES: &[(&str, &str)] = &[
    ("Africa/Johannesburg", "Johannesburg (SAST)"),
    ("Africa/Cairo", "Cairo (EET)"),
    ("Africa/Lagos", "Lagos (WAT)"),
    ("Africa/Nairobi", "Nairobi (EAT)"),
    ("America/New_York", "New York (EST)"),
    ("America/Chicago", "Chicago (CST)"),
    ("America/Denver", "Denver (MST)"),
    ("America/Los_Angeles", "Los Angeles (PST)"),
    ("America/Toronto", "Toronto (EST)"),
    ("America/Mexico_City", "Mexico City (CST)"),
    ("America/Sao_Paulo", "Sao Paulo (BRT)"),
    ("America/Argentina/Buenos_Aires", "Buenos Aires (ART)"),
    ("America/Bogota", "Bogota (COT)"),
    ("Europe/London", "London (GMT)"),
    ("Europe/Paris", "Paris (CET)"),
    ("Europe/Berlin", "Berlin (CET)"),
    ("Europe/Rome", "Rome (CET)"),
    ("Europe/Madrid", "Madrid (CET)"),
    ("Europe/Amsterdam", "Amsterdam (CET)"),
    ("Europe/Stockholm", "Stockholm (CET)"),
    ("Europe/Moscow", "Moscow (MSK)"),
    ("Europe/Istanbul", "Istanbul (TRT)"),
    ("Europe/Zurich", "Zurich (CET)"),
    ("Europe/Warsaw", "Warsaw (CET)"),
    ("Asia/Tokyo", "Tokyo (JST)"),
    ("Asia/Shanghai", "Shanghai (CST)"),
    ("Asia/Hong_Kong", "Hong Kong (HKT)"),
    ("Asia/Singapore", "Singapore (SGT)"),
    ("Asia/Seoul", "Seoul (KST)"),
    ("Asia/Kolkata", "Kolkata (IST)"),
    ("Asia/Dubai", "Dubai (GST)"),
    ("Asia/Riyadh", "Riyadh (AST)"),
    ("Asia/Bangkok", "Bangkok (ICT)"),
    ("Asia/Jakarta", "Jakarta (WIB)"),
    ("Asia/Taipei", "Taipei (CST)"),
    ("Australia/Sydney", "Sydney (AEST)"),
    ("Australia/Melbourne", "Melbourne (AEST)"),
    ("Australia/Perth", "Perth (AWST)"),
    ("Pacific/Auckland", "Auckland (NZST)"),
    ("Pacific/Honolulu", "Honolulu (HST)"),
    ("UTC", "UTC"),
];

const LOCALES: &[(&str, &str)] = &[
    ("en_US.UTF-8", "English - US"),
    ("en_GB.UTF-8", "English - UK"),
    ("de_DE.UTF-8", "German"),
    ("fr_FR.UTF-8", "French"),
    ("es_ES.UTF-8", "Spanish"),
    ("pt_BR.UTF-8", "Portuguese - Brazil"),
    ("ja_JP.UTF-8", "Japanese"),
    ("zh_CN.UTF-8", "Chinese - Simplified"),
    ("ko_KR.UTF-8", "Korean"),
    ("ru_RU.UTF-8", "Russian"),
    ("ar_SA.UTF-8", "Arabic"),
    ("hi_IN.UTF-8", "Hindi"),
    ("nl_NL.UTF-8", "Dutch"),
    ("it_IT.UTF-8", "Italian"),
    ("sv_SE.UTF-8", "Swedish"),
    ("pl_PL.UTF-8", "Polish"),
    ("tr_TR.UTF-8", "Turkish"),
];

/// Auto-detect locale from browser language.
fn detect_locale() -> String {
    let nav = web_sys::window().unwrap().navigator();
    js_sys::Reflect::get(&nav, &"language".into())
        .ok()
        .and_then(|v| v.as_string())
        .map(|lang| {
            // "en-US" → "en_US.UTF-8"
            let normalized = lang.replace('-', "_");
            // Try exact match first (e.g. "en_US" → "en_US.UTF-8")
            for &(locale, _) in LOCALES {
                if locale.starts_with(&normalized) {
                    return locale.to_string();
                }
            }
            // Try language-only match (e.g. "en" → "en_US.UTF-8")
            let lang_only = normalized.split('_').next().unwrap_or("en");
            for &(locale, _) in LOCALES {
                if locale.starts_with(lang_only) {
                    return locale.to_string();
                }
            }
            "en_US.UTF-8".to_string()
        })
        .unwrap_or_else(|| "en_US.UTF-8".to_string())
}

fn detect_os(p: &str) -> &'static str {
    if p.contains("Win") {
        "Windows"
    } else if p.contains("Mac") {
        "macOS"
    } else if p.contains("Linux") {
        "Linux"
    } else {
        "Unknown"
    }
}

/// Trigger a file download in the browser.
fn download_text(filename: &str, content: &str) {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&wasm_bindgen::JsValue::from_str(content));
    let mut opts = web_sys::BlobPropertyBag::new();
    opts.set_type("text/plain");
    if let Ok(blob) = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &opts) {
        if let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) {
            let a = document.create_element("a").unwrap();
            let _ = a.set_attribute("href", &url);
            let _ = a.set_attribute("download", filename);
            let _ = a.set_attribute("style", "display:none");
            let _ = document.body().unwrap().append_child(&a);
            let a_el: web_sys::HtmlElement = a.dyn_into().unwrap();
            a_el.click();
            let _ = document.body().unwrap().remove_child(&a_el);
            let _ = web_sys::Url::revoke_object_url(&url);
        }
    }
}

fn emoji_for_category(cat: AppCategory) -> &'static str {
    match cat {
        AppCategory::Browser => "\u{1F310}",
        AppCategory::Email => "\u{1F4E7}",
        AppCategory::Office => "\u{1F4C4}",
        AppCategory::Notes => "\u{1F4DD}",
        AppCategory::Editor | AppCategory::IDE => "\u{1F4BB}",
        AppCategory::Terminal => "\u{1F5A5}",
        AppCategory::VersionControl => "\u{1F500}",
        AppCategory::Container | AppCategory::DevTools => "\u{1F6E0}",
        AppCategory::Creative2D | AppCategory::Creative3D | AppCategory::Photo => "\u{1F3A8}",
        AppCategory::Audio => "\u{1F3B5}",
        AppCategory::Video => "\u{1F3AC}",
        AppCategory::Gaming | AppCategory::GamingTools => "\u{1F3AE}",
        AppCategory::Communication => "\u{1F4AC}",
        AppCategory::Streaming => "\u{1F3A7}",
        AppCategory::MediaPlayer => "\u{25B6}",
        AppCategory::FileManager | AppCategory::Archive => "\u{1F4C1}",
        AppCategory::Security | AppCategory::VPN => "\u{1F512}",
        AppCategory::SystemUtil => "\u{2699}",
        AppCategory::Virtualization => "\u{1F4E6}",
        _ => "\u{2B50}",
    }
}

// ═══════════════════════════════════════════════════════
// 1. System Basics (hostname, timezone, keyboard, username)
// ═══════════════════════════════════════════════════════

/// Generate a random fun hostname.
fn random_hostname() -> String {
    let adjectives = [
        "swift", "calm", "bright", "cosmic", "lunar", "solar", "ocean", "forest", "crystal",
        "aurora", "jade", "amber", "sage", "coral", "onyx", "pearl", "raven", "frost", "ember",
        "willow",
    ];
    let nouns = [
        "fox", "owl", "wolf", "bear", "hawk", "lynx", "orca", "pine", "oak", "fern", "moss",
        "reef", "peak", "glen", "vale", "dale", "cove", "mesa", "dune", "isle",
    ];
    let now = js_sys::Date::now() as u64;
    let adj = adjectives[(now % adjectives.len() as u64) as usize];
    let noun = nouns[((now / 7) % nouns.len() as u64) as usize];
    format!("{}-{}", adj, noun)
}

#[component]
fn SystemBasics(
    hostname: RwSignal<String>,
    username: RwSignal<String>,
    user_password: RwSignal<String>,
    password_confirm: RwSignal<String>,
    extra_users_str: RwSignal<String>,
    timezone: RwSignal<String>,
    keyboard: RwSignal<String>,
    locale: RwSignal<String>,
) -> impl IntoView {
    view! {
        <GlassPanel title="System Basics">
            <div class="basics-grid">
                <div class="field">
                    <label class="field-label" for="hostname">"Hostname"</label>
                    <div style="display:flex;gap:0.4rem;align-items:center;">
                        <input id="hostname" type="text" class="field-input" style="flex:1;"
                            placeholder="my-nixos"
                            aria-required="true"
                            aria-describedby="hint-hostname"
                            prop:value=move || hostname.get()
                            on:input=move |ev| hostname.set(event_target_value(&ev))
                        />
                        <button class="btn-secondary btn-sm" style="white-space:nowrap;" on:click=move |_| hostname.set(random_hostname())>"Suggest"</button>
                    </div>
                    <span id="hint-hostname" class="field-hint">"Your machine's network name"</span>
                </div>
                <div class="field">
                    <label class="field-label" for="username">"Username"</label>
                    <input id="username" type="text" class="field-input"
                        placeholder="user"
                        aria-required="true"
                        aria-describedby="hint-username"
                        prop:value=move || username.get()
                        on:input=move |ev| username.set(event_target_value(&ev).to_lowercase().replace(' ', ""))
                    />
                    <span id="hint-username" class="field-hint">"Your login name"</span>
                </div>
                <div class="field">
                    <label class="field-label">"Password"</label>
                    <input type="password" class="field-input" placeholder="Required"
                        prop:value=move || user_password.get()
                        on:input=move |ev| user_password.set(event_target_value(&ev))
                    />
                </div>
                <div class="field">
                    <label class="field-label">"Confirm Password"</label>
                    <input type="password" class="field-input" placeholder="Must match"
                        prop:value=move || password_confirm.get()
                        on:input=move |ev| password_confirm.set(event_target_value(&ev))
                    />
                    {move || {
                        let p = user_password.get();
                        let c = password_confirm.get();
                        if !c.is_empty() && p != c {
                            Some(view! { <span class="error-msg" style="font-size: 0.75rem;">"Passwords don't match"</span> })
                        } else if !p.is_empty() && p.len() < 8 {
                            Some(view! { <span class="warning-msg" style="font-size: 0.75rem;">"Password should be at least 8 characters"</span> })
                        } else { None }
                    }}
                </div>
                <div class="field">
                    <label class="field-label" for="timezone">"Timezone"</label>
                    <input id="timezone" type="text" class="field-input" list="timezone-list"
                        placeholder="UTC"
                        prop:value=move || timezone.get()
                        on:input=move |ev| timezone.set(event_target_value(&ev))
                    />
                    <datalist id="timezone-list">
                        {TIMEZONES.iter().map(|&(iana, city)| view! {
                            <option value=iana label=city />
                        }).collect::<Vec<_>>()}
                    </datalist>
                    <span class="field-hint">"Auto-detected from your browser"</span>
                </div>
                <div class="field">
                    <label class="field-label" for="keyboard">"Keyboard"</label>
                    <input id="keyboard" type="text" class="field-input"
                        placeholder="us"
                        prop:value=move || keyboard.get()
                        on:input=move |ev| keyboard.set(event_target_value(&ev))
                    />
                    <span class="field-hint">"Console keymap (us, gb, de, fr, ...)"</span>
                </div>
                <div class="field">
                    <label class="field-label" for="locale">"Locale"</label>
                    <select id="locale" class="field-input"
                        on:change=move |ev| locale.set(event_target_value(&ev))
                    >
                        {LOCALES.iter().map(|&(val, label)| {
                            let v = val.to_string();
                            view! {
                                <option value=val selected=move || locale.get() == v>{label}</option>
                            }
                        }).collect::<Vec<_>>()}
                    </select>
                    <span class="field-hint">"System language and encoding"</span>
                </div>
                <div class="field">
                    <label class="field-label">"Additional Users"</label>
                    <input type="text" class="field-input" placeholder="alice, bob (comma-separated)"
                        prop:value=move || extra_users_str.get()
                        on:input=move |ev| extra_users_str.set(event_target_value(&ev))
                    />
                    <span class="field-hint">"Extra user accounts (no admin privileges)"</span>
                </div>
            </div>
        </GlassPanel>
    }
}

// ═══════════════════════════════════════════════════════
// 2. Desktop Picker
// ═══════════════════════════════════════════════════════

struct DesktopOption {
    id: &'static str,
    name: &'static str,
    desc: &'static str,
    preview: &'static str,
    min_ram_mb: u32,
    resource_note: &'static str,
}

const DESKTOP_OPTIONS: &[DesktopOption] = &[
    DesktopOption {
        id: "gnome",
        name: "GNOME",
        desc: "Beginner-friendly, clean and polished",
        preview: "Activities overview, dock at bottom, clean top bar. Like macOS but with a dynamic workspace grid.",
        min_ram_mb: 4096,
        resource_note: "Needs 4GB+ RAM. Smooth on modern hardware.",
    },
    DesktopOption {
        id: "kde",
        name: "KDE Plasma",
        desc: "Highly customizable, Windows-like layout",
        preview: "Taskbar, start menu, system tray. Familiar Windows layout with deep customization.",
        min_ram_mb: 3072,
        resource_note: "Needs 3GB+ RAM. Feature-rich, great for Windows users.",
    },
    DesktopOption {
        id: "cosmic",
        name: "Cosmic",
        desc: "System76's new Rust-native desktop",
        preview: "Modern tiling + floating hybrid. Built in Rust by System76. Still in alpha — expect bugs.",
        min_ram_mb: 2048,
        resource_note: "Needs 2GB+ RAM. Alpha software — not recommended for primary machines.",
    },
    DesktopOption {
        id: "hyprland",
        name: "Hyprland",
        desc: "Tiling compositor, keyboard-driven",
        preview: "No mouse needed. Windows tile automatically. Config via text file. For power users.",
        min_ram_mb: 1024,
        resource_note: "Very light (~1GB RAM). Keyboard-driven — learn keybinds first.",
    },
    DesktopOption {
        id: "sway",
        name: "Sway",
        desc: "Tiling Wayland compositor, i3-like",
        preview: "i3-compatible Wayland compositor. Keyboard-driven tiling. Minimal and fast.",
        min_ram_mb: 1024,
        resource_note: "Very light (~1GB RAM). Perfect for older hardware.",
    },
    DesktopOption {
        id: "xfce",
        name: "XFCE",
        desc: "Lightweight and resource-efficient",
        preview: "Traditional desktop, very light on resources. Good for older hardware or VMs.",
        min_ram_mb: 512,
        resource_note: "Runs on anything with 512MB+ RAM. Best for old PCs.",
    },
    DesktopOption {
        id: "none",
        name: "None / Server",
        desc: "No GUI — terminal only",
        preview: "Command line only. SSH access. Minimal resource usage. Add a DE later if needed.",
        min_ram_mb: 256,
        resource_note: "Minimal footprint. You can add a DE later with one config change.",
    },
];

#[component]
fn DesktopPicker(selected: RwSignal<String>) -> impl IntoView {
    view! {
        <GlassPanel title="Desktop Environment">
            <p class="section-desc">"Choose how your system looks and feels."</p>
            <div class="desktop-grid" role="radiogroup" aria-label="Desktop environment">
                {DESKTOP_OPTIONS.iter().map(|opt| {
                    let id = opt.id;
                    let name = opt.name;
                    let desc = opt.desc;
                    let preview = opt.preview;
                    let resource_note = opt.resource_note;
                    let id1 = id.to_string();
                    let id2 = id.to_string();
                    let id3 = id.to_string();
                    view! {
                        <label class="desktop-card" class:desktop-card-selected=move || selected.get() == id1>
                            <input type="radio" name="desktop"
                                prop:checked=move || selected.get() == id2
                                on:change=move |_| selected.set(id3.clone())
                            />
                            <div class="desktop-card-body">
                                <span class="desktop-name">{name}</span>
                                <span class="desktop-desc">{desc}</span>
                                <span class="desktop-preview">{preview}</span>
                                <span class="desktop-resources">{resource_note}</span>
                            </div>
                        </label>
                    }
                }).collect::<Vec<_>>()}
            </div>
        </GlassPanel>
    }
}

// ═══════════════════════════════════════════════════════
// 3. Security Options
// ═══════════════════════════════════════════════════════

#[component]
fn SecurityOptions(
    encrypt: RwSignal<bool>,
    secure_boot: RwSignal<bool>,
    tpm_unlock: RwSignal<bool>,
    fido2_unlock: RwSignal<bool>,
    disk_layout: RwSignal<String>,
    filesystem: RwSignal<String>,
) -> impl IntoView {
    view! {
        <GlassPanel title="Security">
            <p class="section-desc">"Protect your system from day one."</p>
            <div class="security-options">
                <div class="field" style="margin-bottom: 1rem;">
                    <label class="field-label">"Disk Layout"</label>
                    <div class="disk-layout-options">
                        <label class="security-option">
                            <input type="radio" name="disk_layout"
                                prop:checked=move || disk_layout.get() == "single" || disk_layout.get() == "single-luks"
                                on:change=move |_| {
                                    if encrypt.get() { disk_layout.set("single-luks".into()); }
                                    else { disk_layout.set("single".into()); }
                                }
                            />
                            <div>
                                <span>"Wipe entire disk"</span>
                                <span class="security-note">"Erases everything on the selected disk and installs NixOS. All existing data will be permanently deleted."</span>
                            </div>
                        </label>
                        <label class="security-option">
                            <input type="radio" name="disk_layout"
                                prop:checked=move || disk_layout.get() == "alongside"
                                on:change=move |_| disk_layout.set("alongside".into())
                            />
                            <div>
                                <span>"Install alongside existing OS (dual-boot)"</span>
                                <span class="security-note">"Installs NixOS next to your current operating system (Windows, Linux, etc.). Your existing OS and files are preserved. You choose which OS to boot each time you start your computer. Requires at least 20GB of free disk space."</span>
                            </div>
                        </label>
                    </div>
                </div>
                <div class="field" style="margin-top: 0.8rem; margin-bottom: 1rem;">
                    <label class="field-label">"Filesystem"</label>
                    <div style="display: flex; gap: 1rem;">
                        <label class="security-option">
                            <input type="radio" name="filesystem"
                                prop:checked=move || filesystem.get() != "zfs"
                                on:change=move |_| filesystem.set("btrfs".into())
                            />
                            <div>
                                <span>"btrfs"</span>
                                <span class="security-note">"Recommended. Snapshots, compression, self-healing."</span>
                            </div>
                        </label>
                        <label class="security-option">
                            <input type="radio" name="filesystem"
                                prop:checked=move || filesystem.get() == "zfs"
                                on:change=move |_| filesystem.set("zfs".into())
                            />
                            <div>
                                <span>"ZFS"</span>
                                <span class="security-note">"Advanced. Native encryption, RAID-Z, enterprise features."</span>
                            </div>
                        </label>
                    </div>
                </div>
                <label class="security-option">
                    <input type="checkbox"
                        prop:checked=move || encrypt.get()
                        on:change=move |_| {
                            encrypt.update(|v| *v = !*v);
                            // Sync disk layout when encryption changes (only for wipe-disk mode)
                            if disk_layout.get() != "alongside" {
                                if encrypt.get() { disk_layout.set("single-luks".into()); }
                                else { disk_layout.set("single".into()); }
                            }
                        }
                    />
                    <div>
                        <span>"Full disk encryption (LUKS2)"</span>
                        <span class="security-note">"You will set a passphrase during install. Required to boot."</span>
                        <details class="help-expander">
                            <summary class="help-toggle">"What does this mean?"</summary>
                            <p class="help-text">
                                "Your entire disk will be encrypted with a passphrase. "
                                "Every time you turn on your computer, you will type this passphrase before NixOS starts. "
                                "If your computer is stolen, nobody can read your files without the passphrase. "
                                <strong>"If you forget your passphrase, your data cannot be recovered."</strong>
                            </p>
                        </details>
                    </div>
                </label>
                <label class="security-option">
                    <input type="checkbox"
                        prop:checked=move || secure_boot.get()
                        on:change=move |_| secure_boot.update(|v| *v = !*v)
                    />
                    <div>
                        <span>"Secure Boot (lanzaboote)"</span>
                        <span class="security-note">"Keys are enrolled in firmware on first boot."</span>
                        <details class="help-expander">
                            <summary class="help-toggle">"What does this mean?"</summary>
                            <p class="help-text">
                                "Secure Boot verifies that only trusted software runs when your computer starts. "
                                "This protects against rootkits and boot-level malware. "
                                "On first boot, you will need to approve NixOS in your firmware settings — the installer will guide you through this."
                            </p>
                        </details>
                    </div>
                </label>
                {move || encrypt.get().then(|| view! {
                    <label class="security-option">
                        <input type="checkbox"
                            prop:checked=move || tpm_unlock.get()
                            on:change=move |_| tpm_unlock.update(|v| *v = !*v)
                        />
                        <div>
                            <span>"TPM2 auto-unlock"</span>
                            <span class="security-note">"Auto-unlocks disk using TPM chip. Falls back to passphrase."</span>
                            <details class="help-expander">
                                <summary class="help-toggle">"What does this mean?"</summary>
                                <p class="help-text">
                                    "Your computer has a security chip (TPM) that can remember your encryption passphrase. "
                                    "You will not need to type it every boot — the chip unlocks the disk automatically. "
                                    "If the disk is moved to a different computer, the passphrase will be required. "
                                    "This combines convenience with strong encryption."
                                </p>
                            </details>
                        </div>
                    </label>
                    <label class="security-option">
                        <input type="checkbox"
                            prop:checked=move || fido2_unlock.get()
                            on:change=move |_| fido2_unlock.update(|v| *v = !*v)
                        />
                        <div>
                            <span>"FIDO2/YubiKey unlock"</span>
                            <span class="security-note">"Unlock disk with a hardware security key. Touch key at boot."</span>
                            <details class="help-expander">
                                <summary class="help-toggle">"What does this mean?"</summary>
                                <p class="help-text">
                                    "A FIDO2 security key (like YubiKey) can unlock your encrypted disk. "
                                    "Plug it in and touch it when your computer starts — no passphrase needed. "
                                    "If the key is lost, your passphrase still works as a backup."
                                </p>
                            </details>
                        </div>
                    </label>
                })}
            </div>
        </GlassPanel>
    }
}

// ═══════════════════════════════════════════════════════
// 4. App Selection with search/filter (#8)
// ═══════════════════════════════════════════════════════

#[component]
fn AppSelectionGrid(
    os: &'static str,
    selected: RwSignal<Vec<String>>,
    app_category: RwSignal<String>,
) -> impl IntoView {
    let db = AppDatabase::new();
    let all_entries: Vec<&'static AppEntry> = db
        .entries()
        .iter()
        .filter(|e| match os {
            "Windows" => !e.windows_names.is_empty() || !e.winget_ids.is_empty(),
            "macOS" => !e.macos_names.is_empty() || !e.brew_names.is_empty(),
            "Linux" => !e.linux_names.is_empty() || !e.flatpak_ids.is_empty(),
            _ => true,
        })
        .copied()
        .collect();

    let (search, set_search) = signal(String::new());

    view! {
        <GlassPanel title="What do you use?">
            <p class="section-desc">"Check your apps. We show what's available on NixOS."</p>
            <div class="app-categories">
                <button class="category-tab" class:active=move || app_category.get() == "all"
                    on:click=move |_| app_category.set("all".into())>"All"</button>
                <button class="category-tab" class:active=move || app_category.get() == "browser"
                    on:click=move |_| app_category.set("browser".into())>"Browsers"</button>
                <button class="category-tab" class:active=move || app_category.get() == "editor"
                    on:click=move |_| app_category.set("editor".into())>"Editors"</button>
                <button class="category-tab" class:active=move || app_category.get() == "communication"
                    on:click=move |_| app_category.set("communication".into())>"Communication"</button>
                <button class="category-tab" class:active=move || app_category.get() == "creative"
                    on:click=move |_| app_category.set("creative".into())>"Creative"</button>
                <button class="category-tab" class:active=move || app_category.get() == "gaming"
                    on:click=move |_| app_category.set("gaming".into())>"Gaming"</button>
                <button class="category-tab" class:active=move || app_category.get() == "dev"
                    on:click=move |_| app_category.set("dev".into())>"Development"</button>
                <button class="category-tab" class:active=move || app_category.get() == "system"
                    on:click=move |_| app_category.set("system".into())>"System"</button>
                <button class="category-tab" class:active=move || app_category.get() == "office"
                    on:click=move |_| app_category.set("office".into())>"Office"</button>
                <button class="category-tab" class:active=move || app_category.get() == "media"
                    on:click=move |_| app_category.set("media".into())>"Media"</button>
            </div>
            <input type="text" class="app-search" placeholder="Search apps..."
                prop:value=search
                on:input=move |ev| set_search.set(event_target_value(&ev))
            />
            <div class="app-grid">
                {move || {
                    let q = search.get().to_lowercase();
                    let cat = app_category.get();
                    all_entries.iter()
                        .filter(|e| {
                            (cat == "all" || category_tab_key(e.category) == cat) &&
                            (q.is_empty() || e.name.to_lowercase().contains(&q) || e.primary.display_name.to_lowercase().contains(&q))
                        })
                        .map(|entry| {
                            let name = entry.name;
                            let emoji = emoji_for_category(entry.category);
                            let quality = entry.primary.quality;
                            let nix_name = entry.primary.display_name;
                            let status_class = match quality {
                                MatchQuality::Native | MatchQuality::OfficialLinux => "status-native",
                                MatchQuality::StrongAlternative => "status-alt",
                                MatchQuality::PartialAlternative | MatchQuality::WineCompatible | MatchQuality::WebApp => "status-partial",
                                MatchQuality::NoEquivalent => "status-none",
                            };
                            let status_label = match quality {
                                MatchQuality::Native | MatchQuality::OfficialLinux => "Available",
                                MatchQuality::StrongAlternative => "Alternative",
                                MatchQuality::PartialAlternative => "Partial",
                                MatchQuality::WineCompatible => "Wine/Proton",
                                MatchQuality::WebApp => "Web App",
                                MatchQuality::NoEquivalent => "No Equivalent",
                            };
                            let name_owned = name.to_string();
                            let n1 = name_owned.clone();
                            let n2 = name_owned.clone();
                            let n3 = name_owned;
                            view! {
                                <label class="app-card" class:app-card-checked=move || selected.get().contains(&n1)>
                                    <input type="checkbox"
                                        prop:checked=move || selected.get().contains(&n2)
                                        on:change=move |_| {
                                            selected.update(|s| {
                                                if s.contains(&n3) { s.retain(|n| n != &n3); }
                                                else { s.push(n3.clone()); }
                                            });
                                        }
                                    />
                                    <div class="app-card-body">
                                        <span class="app-emoji">{emoji}</span>
                                        <div class="app-info">
                                            <span class="app-name">{name}</span>
                                            <span class={format!("app-status {status_class}")}>{status_label}</span>
                                        </div>
                                    </div>
                                    <div class="app-nix">
                                        <span class="app-nix-arrow">"→"</span>
                                        <span class="app-nix-name">{nix_name}</span>
                                    </div>
                                </label>
                            }
                        }).collect::<Vec<_>>()
                }}
            </div>
        </GlassPanel>
    }
}

// ═══════════════════════════════════════════════════════
// 5. Next Steps Guide (#5)
// ═══════════════════════════════════════════════════════

#[component]
fn NextSteps(encrypt: RwSignal<bool>) -> impl IntoView {
    view! {
        <GlassPanel title="Next Steps">
            <div class="next-steps">
                <div class="step">
                    <span class="step-num">"1"</span>
                    <div>
                        <strong>"Download the NixOS ISO"</strong>
                        <p class="step-detail">
                            "Get the minimal ISO from "
                            <a href="https://nixos.org/download#nixos-iso" target="_blank">"nixos.org/download"</a>
                            " and flash it to a USB with "
                            <a href="https://etcher.balena.io" target="_blank">"Etcher"</a>
                            " or "<code>"dd"</code>"."
                        </p>
                    </div>
                </div>
                <div class="step">
                    <span class="step-num">"2"</span>
                    <div>
                        <strong>"Boot from USB and partition"</strong>
                        <p class="step-detail">
                            "Boot the target machine from USB. Partition your disk and mount at "
                            <code>"/mnt"</code>". Run "<code>"nixos-generate-config --root /mnt"</code>
                            " to create "<code>"hardware-configuration.nix"</code>"."
                        </p>
                    </div>
                </div>
                <div class="step">
                    <span class="step-num">"3"</span>
                    <div>
                        <strong>"Copy your config files"</strong>
                        <p class="step-detail">
                            "Replace "<code>"/mnt/etc/nixos/configuration.nix"</code>" with the one you downloaded. "
                            "Copy "<code>"flake.nix"</code>" to "<code>"/mnt/etc/nixos/"</code>" too."
                        </p>
                    </div>
                </div>
                {move || encrypt.get().then(|| view! {
                    <div class="step">
                        <span class="step-num">"3b"</span>
                        <div>
                            <strong>"Set up LUKS encryption"</strong>
                            <p class="step-detail">
                                "Before mounting, encrypt with: "
                                <code>"cryptsetup luksFormat /dev/<your-partition>"</code>
                                " then "<code>"cryptsetup open /dev/<your-partition> cryptroot"</code>
                                ". Format the decrypted device, then mount."
                            </p>
                        </div>
                    </div>
                })}
                <div class="step">
                    <span class="step-num">"4"</span>
                    <div>
                        <strong>"Install"</strong>
                        <p class="step-detail">
                            "Run "<code>"nixos-install"</code>". Set root password when prompted. "
                            "Reboot, remove USB, and log in."
                        </p>
                    </div>
                </div>
                <div class="step">
                    <span class="step-num">"5"</span>
                    <div>
                        <strong>"Set your user password"</strong>
                        <p class="step-detail">
                            "After first login as root: "<code>"passwd <your-username>"</code>
                            ". Then log out and log in as your user."
                        </p>
                    </div>
                </div>
            </div>
        </GlassPanel>
    }
}

// ═══════════════════════════════════════════════════════
// 6. App Paste & Match (via SporeEngine WASM)
// ═══════════════════════════════════════════════════════

#[derive(Clone, Debug, Default)]
struct MigrationReport {
    total_apps: usize,
    matched: Vec<(String, String, String)>, // (source, nix_pkg, category)
    unmatched: Vec<String>,
    readiness_score: f64,
    summary: String,
    bundles: Vec<String>,
}

#[component]
fn AppPastePanel() -> impl IntoView {
    let engine = use_context::<EngineWorker>();
    let (paste_text, set_paste_text) = signal(String::new());
    let (report, set_report) = signal(Option::<MigrationReport>::None);
    let (matching, set_matching) = signal(false);

    let do_match = move || {
        let text = paste_text.get();
        if text.trim().is_empty() {
            return;
        }

        // Try worker first (SporeEngine WASM), fall back to local AppDatabase
        if let Some(ref engine) = engine {
            if engine.is_available() {
                let engine = engine.clone();
                set_matching.set(true);
                wasm_bindgen_futures::spawn_local(async move {
                    let params = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(
                        &params,
                        &"text".into(),
                        &wasm_bindgen::JsValue::from_str(&text),
                    );
                    let promise = engine.send("matchAppList", &params.into());
                    match JsFuture::from(promise).await {
                        Ok(result) => {
                            let r = parse_migration_report(&result);
                            set_report.set(Some(r));
                        }
                        Err(_) => {
                            // Fallback: local AppDatabase
                            let r = local_match_apps(&text);
                            set_report.set(Some(r));
                        }
                    }
                    set_matching.set(false);
                });
                return;
            }
        }

        // No worker — use local AppDatabase directly
        let r = local_match_apps(&text);
        set_report.set(Some(r));
    };

    view! {
        <GlassPanel title="Paste Your Apps">
            <p class="section-desc">
                "Paste output from "<code>"winget list"</code>", "<code>"brew list"</code>
                ", "<code>"dpkg --list"</code>", or "<code>"pacman -Qe"</code>
                ". We'll match them to NixOS packages instantly."
            </p>
            <textarea class="app-paste-area"
                placeholder="Paste your app list here...

Examples:
  winget list
  brew list --cask
  dpkg --list | awk '{print $2}'
  pacman -Qe"
                prop:value=paste_text
                on:input=move |ev| set_paste_text.set(event_target_value(&ev))
            />
            <div style="display:flex; gap:0.5rem; margin-top:0.5rem; align-items:center;">
                <button class="btn-primary" on:click=move |_| do_match()
                    prop:disabled=move || matching.get() || paste_text.get().trim().is_empty()
                >{move || if matching.get() { "Matching..." } else { "Match Apps" }}</button>
                {move || report.get().as_ref().map(|r| {
                    let total = r.total_apps;
                    let matched = r.matched.len();
                    view! { <span class="field-hint">{format!("{matched} of {total} matched")}</span> }
                })}
            </div>

            {move || report.get().map(|r| {
                let score_class = if r.readiness_score > 0.85 { "migration-score-high" }
                    else if r.readiness_score > 0.55 { "migration-score-med" }
                    else { "migration-score-low" };
                view! {
                    <div class="migration-report">
                        <div class={format!("migration-score {score_class}")}>
                            {format!("{}%", (r.readiness_score * 100.0) as u32)}
                        </div>
                        <p class="migration-summary">{r.summary.clone()}</p>

                        {(!r.bundles.is_empty()).then(|| view! {
                            <div class="migration-bundles">
                                {r.bundles.iter().map(|b| view! {
                                    <span class="migration-bundle">{b.clone()}</span>
                                }).collect::<Vec<_>>()}
                            </div>
                        })}

                        {(!r.matched.is_empty()).then(|| view! {
                            <div class="migration-matched">
                                <h5>{format!("Matched ({})", r.matched.len())}</h5>
                                {r.matched.iter().map(|(src, nix, cat)| view! {
                                    <div class="migration-item">
                                        <span class="migration-item-source">{src.clone()}</span>
                                        <span class="migration-item-arrow">"→"</span>
                                        <span class="migration-item-nix">{nix.clone()}</span>
                                        <span class="migration-item-category">{cat.clone()}</span>
                                    </div>
                                }).collect::<Vec<_>>()}
                            </div>
                        })}

                        {(!r.unmatched.is_empty()).then(|| view! {
                            <div class="migration-unmatched">
                                <h5>{format!("Unmatched ({})", r.unmatched.len())}</h5>
                                {r.unmatched.iter().map(|name| view! {
                                    <div class="migration-unmatched-item">{name.clone()}</div>
                                }).collect::<Vec<_>>()}
                            </div>
                        })}
                    </div>
                }
            })}
        </GlassPanel>
    }
}

/// Parse a JsValue migration report from the worker.
fn parse_migration_report(val: &wasm_bindgen::JsValue) -> MigrationReport {
    let total = js_sys::Reflect::get(val, &"total_apps".into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as usize;
    let readiness = js_sys::Reflect::get(val, &"readiness_score".into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let summary = js_sys::Reflect::get(val, &"summary".into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default();

    let mut matched = Vec::new();
    if let Ok(arr) = js_sys::Reflect::get(val, &"matched".into()) {
        if js_sys::Array::is_array(&arr) {
            let arr = js_sys::Array::from(&arr);
            for i in 0..arr.length() {
                let item = arr.get(i);
                let src = js_sys::Reflect::get(&item, &"source_name".into())
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_default();
                let nix = js_sys::Reflect::get(&item, &"nix_package".into())
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_default();
                let cat = js_sys::Reflect::get(&item, &"category".into())
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_default();
                matched.push((src, nix, cat));
            }
        }
    }

    let mut unmatched = Vec::new();
    if let Ok(arr) = js_sys::Reflect::get(val, &"unmatched".into()) {
        if js_sys::Array::is_array(&arr) {
            let arr = js_sys::Array::from(&arr);
            for i in 0..arr.length() {
                if let Some(s) = arr.get(i).as_string() {
                    unmatched.push(s);
                }
            }
        }
    }

    let mut bundles = Vec::new();
    if let Ok(arr) = js_sys::Reflect::get(val, &"suggested_bundles".into()) {
        if js_sys::Array::is_array(&arr) {
            let arr = js_sys::Array::from(&arr);
            for i in 0..arr.length() {
                if let Some(s) = arr.get(i).as_string() {
                    bundles.push(s);
                }
            }
        }
    }

    MigrationReport {
        total_apps: total,
        matched,
        unmatched,
        readiness_score: readiness,
        summary,
        bundles,
    }
}

/// Fallback: match apps using the local AppDatabase (no worker needed).
fn local_match_apps(text: &str) -> MigrationReport {
    let db = AppDatabase::new();
    let names: Vec<String> = db.parse_app_list(text);
    let report = db.match_list(&names);
    MigrationReport {
        total_apps: report.total_apps,
        matched: report
            .matched
            .iter()
            .map(|m| {
                (
                    m.source_name.clone(),
                    m.entry.primary.display_name.to_string(),
                    format!("{:?}", m.entry.category),
                )
            })
            .collect(),
        unmatched: report.unmatched.clone(),
        readiness_score: report.readiness_score as f64,
        summary: report.summary.clone(),
        bundles: report
            .suggested_bundles
            .iter()
            .map(|b| b.name.to_string())
            .collect(),
    }
}

// ═══════════════════════════════════════════════════════
// 7. Conversational Mode — Talk with Symthaea
// ═══════════════════════════════════════════════════════

#[derive(Clone, PartialEq)]
struct ConverseChatMsg {
    id: usize,
    is_user: bool,
    text: String,
    config_preview: Option<String>,
    decisions: Vec<(String, String, String, f64)>, // (option, value, reasoning, confidence)
    ready_to_deploy: bool,
}

#[component]
fn ConverseMode(
    hardware_cores: u32,
    hardware_mem: u32,
    gpu_vendor: String,
    gpu_model: String,
    config_nix_signal: RwSignal<Option<String>>,
    flake_nix_signal: RwSignal<Option<String>>,
    hostname: RwSignal<String>,
    selected_desktop: RwSignal<String>,
    gpu_driver: Signal<String>,
    timezone: RwSignal<String>,
    keyboard: RwSignal<String>,
    encrypt_disk: RwSignal<bool>,
    secure_boot: RwSignal<bool>,
    tpm_unlock: RwSignal<bool>,
    fido2_unlock: RwSignal<bool>,
    disk_layout: RwSignal<String>,
    filesystem: RwSignal<String>,
    user_password: RwSignal<String>,
    mode: RwSignal<String>,
) -> impl IntoView {
    let engine = use_context::<EngineWorker>();
    let (messages, set_messages) = signal(Vec::<ConverseChatMsg>::new());
    let (next_id, set_next_id) = signal(1_usize);
    let (input_value, set_input_value) = signal(String::new());
    let (is_thinking, set_is_thinking) = signal(false);
    let (initialized, set_initialized) = signal(false);
    let (ready_to_deploy, set_ready_to_deploy) = signal(false);
    let (show_remote, set_show_remote) = signal(false);

    // Initialize sovereign conversation on mount
    {
        let engine = engine.clone();
        let gv = gpu_vendor.clone();
        let gm = gpu_model.clone();
        Effect::new(move |_| {
            if initialized.get() {
                return;
            }
            let Some(ref engine) = engine else {
                return;
            };
            if !engine.is_available() {
                return;
            }

            let engine = engine.clone();
            let gv = gv.clone();
            let gm = gm.clone();
            set_initialized.set(true);
            set_is_thinking.set(true);

            wasm_bindgen_futures::spawn_local(async move {
                let params = js_sys::Object::new();
                let hw = serde_json::json!({
                    "gpu_vendor": gv,
                    "gpu_model": gm,
                    "cpu_cores": hardware_cores,
                    "memory_gb": hardware_mem,
                    "has_wifi": true,
                });
                let hw_val = wasm_bindgen::JsValue::from_str(&hw.to_string());
                let _ = js_sys::Reflect::set(&params, &"hardware".into(), &hw_val);
                let _ = js_sys::Reflect::set(
                    &params,
                    &"migration".into(),
                    &wasm_bindgen::JsValue::from_str("{}"),
                );

                let promise = engine.send("sovereignInit", &params.into());
                match JsFuture::from(promise).await {
                    Ok(result) => {
                        let msg_text = js_sys::Reflect::get(&result, &"message".into())
                            .ok().and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Hello! I'm Symthaea. Tell me what you need from your NixOS system and I'll configure it for you.".into());
                        let config_preview =
                            js_sys::Reflect::get(&result, &"config_preview".into())
                                .ok()
                                .and_then(|v| v.as_string());

                        set_messages.update(|msgs| {
                            msgs.push(ConverseChatMsg {
                                id: 0,
                                is_user: false,
                                text: msg_text,
                                config_preview,
                                decisions: Vec::new(),
                                ready_to_deploy: false,
                            })
                        });
                    }
                    Err(e) => {
                        let err_msg = format!("{:?}", e);
                        set_messages.update(|msgs| msgs.push(ConverseChatMsg {
                            id: 0, is_user: false,
                            text: format!("Hello! I'm Symthaea. The conversation engine couldn't fully initialize ({}), but you can still describe what you need and I'll help configure your NixOS system.", err_msg),
                            config_preview: None, decisions: Vec::new(), ready_to_deploy: false,
                        }));
                    }
                }
                set_is_thinking.set(false);
            });
        });
    }

    // Chat submit handler
    let engine_chat = engine.clone();
    let on_submit = move |ev: web_sys::SubmitEvent| {
        ev.prevent_default();
        let text = input_value.get();
        if text.trim().is_empty() || is_thinking.get() {
            return;
        }

        // Add user message
        let user_id = next_id.get();
        set_next_id.set(user_id + 1);
        set_messages.update(|msgs| {
            msgs.push(ConverseChatMsg {
                id: user_id,
                is_user: true,
                text: text.clone(),
                config_preview: None,
                decisions: Vec::new(),
                ready_to_deploy: false,
            })
        });
        set_input_value.set(String::new());
        set_is_thinking.set(true);

        // Scroll chat to bottom
        if let Some(el) = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.query_selector(".converse-chat").ok().flatten())
        {
            let el: web_sys::HtmlElement = el.dyn_into().unwrap();
            let _ = el.set_scroll_top(el.scroll_height());
        }

        let Some(ref engine) = engine_chat else {
            set_is_thinking.set(false);
            return;
        };
        let engine = engine.clone();

        wasm_bindgen_futures::spawn_local(async move {
            let params = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &params,
                &"message".into(),
                &wasm_bindgen::JsValue::from_str(&text),
            );
            let promise = engine.send("sovereignChat", &params.into());

            let reply_id = next_id.get_untracked();
            set_next_id.set(reply_id + 1);

            match JsFuture::from(promise).await {
                Ok(result) => {
                    let msg_text = js_sys::Reflect::get(&result, &"message".into())
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_else(|| {
                            "I understand. Let me think about the best configuration...".into()
                        });
                    let config_preview = js_sys::Reflect::get(&result, &"config_preview".into())
                        .ok()
                        .and_then(|v| v.as_string());
                    let is_ready = js_sys::Reflect::get(&result, &"ready_to_deploy".into())
                        .ok()
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);

                    // Parse decisions
                    let mut decisions = Vec::new();
                    if let Ok(arr) = js_sys::Reflect::get(&result, &"decisions".into()) {
                        if js_sys::Array::is_array(&arr) {
                            let arr = js_sys::Array::from(&arr);
                            for i in 0..arr.length() {
                                let d = arr.get(i);
                                let opt = js_sys::Reflect::get(&d, &"option".into())
                                    .ok()
                                    .and_then(|v| v.as_string())
                                    .unwrap_or_default();
                                let val = js_sys::Reflect::get(&d, &"value".into())
                                    .ok()
                                    .and_then(|v| v.as_string())
                                    .unwrap_or_default();
                                let reason = js_sys::Reflect::get(&d, &"reasoning".into())
                                    .ok()
                                    .and_then(|v| v.as_string())
                                    .unwrap_or_default();
                                let conf = js_sys::Reflect::get(&d, &"confidence".into())
                                    .ok()
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0);
                                decisions.push((opt, val, reason, conf));
                            }
                        }
                    }

                    if is_ready {
                        set_ready_to_deploy.set(true);
                        // Store config for the deploy panel
                        if let Some(ref cfg) = config_preview {
                            config_nix_signal.set(Some(cfg.clone()));
                        }
                    }

                    set_messages.update(|msgs| {
                        msgs.push(ConverseChatMsg {
                            id: reply_id,
                            is_user: false,
                            text: msg_text,
                            config_preview,
                            decisions,
                            ready_to_deploy: is_ready,
                        })
                    });
                }
                Err(e) => {
                    set_messages.update(|msgs| {
                        msgs.push(ConverseChatMsg {
                            id: reply_id,
                            is_user: false,
                            text: format!("Sorry, I had trouble processing that: {:?}", e),
                            config_preview: None,
                            decisions: Vec::new(),
                            ready_to_deploy: false,
                        })
                    });
                }
            }
            set_is_thinking.set(false);

            // Auto-scroll
            if let Some(el) = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.query_selector(".converse-chat").ok().flatten())
            {
                let el: web_sys::HtmlElement = el.dyn_into().unwrap();
                let _ = el.set_scroll_top(el.scroll_height());
            }
        });
    };

    view! {
        <section class="install-section">
            <GlassPanel title="Talk with Symthaea">
                <p class="section-desc">
                    "Describe what you need in your own words. Symthaea will configure your NixOS system through conversation."
                </p>

                // Chat messages
                <div class="converse-chat">
                    {move || messages.get().iter().map(|msg| {
                        let cls = if msg.is_user { "converse-msg converse-msg-user" } else { "converse-msg converse-msg-ai" };
                        let sender = if msg.is_user { "You" } else { "Symthaea" };
                        let text = msg.text.clone();
                        let config = msg.config_preview.clone();
                        let decisions = msg.decisions.clone();
                        let is_ready = msg.ready_to_deploy;
                        view! {
                            <div class=cls>
                                <span class="converse-msg-sender">{sender}</span>
                                <span class="converse-msg-text" inner_html=format_converse_text(&text) />
                            </div>
                            {config.map(|cfg| view! {
                                <details class="converse-config-preview">
                                    <summary>"View generated configuration"</summary>
                                    <pre class="converse-config-code">{cfg}</pre>
                                </details>
                            })}
                            {(!decisions.is_empty()).then(|| view! {
                                <details class="converse-decisions">
                                    <summary>{format!("Reasoning ({} decisions)", decisions.len())}</summary>
                                    <div style="margin-top:0.5rem;">
                                        {decisions.iter().map(|(opt, val, reason, conf)| view! {
                                            <div class="converse-decision">
                                                <span class="converse-decision-option">{opt.clone()}</span>
                                                " = "{val.clone()}
                                                <span class="converse-decision-reasoning">{reason.clone()}</span>
                                                <span class="converse-decision-confidence">{format!("Confidence: {}%", (*conf * 100.0) as u32)}</span>
                                            </div>
                                        }).collect::<Vec<_>>()}
                                    </div>
                                </details>
                            })}
                            {is_ready.then(|| view! {
                                <div class="converse-deploy">
                                    <button class="btn-primary" on:click=move |_| set_show_remote.set(true)>
                                        "Deploy This Configuration"
                                    </button>
                                    <p class="field-hint" style="margin-top:0.3rem;">"Review the config above before deploying"</p>
                                </div>
                            })}
                        }
                    }).collect::<Vec<_>>()}

                    {move || is_thinking.get().then(|| view! {
                        <div class="converse-thinking">
                            <div class="spinner"></div>
                            <span>"Symthaea is thinking..."</span>
                        </div>
                    })}
                </div>

                // Chat input
                <form class="converse-form" on:submit=on_submit>
                    <input class="converse-input" type="text" autocomplete="off"
                        placeholder="I need a system for development with encryption..."
                        prop:value=input_value
                        prop:disabled=move || is_thinking.get()
                        on:input=move |ev| set_input_value.set(event_target_value(&ev))
                    />
                    <button class="btn-primary" type="submit"
                        prop:disabled=move || is_thinking.get() || input_value.get().trim().is_empty()
                    >"Send"</button>
                </form>
            </GlassPanel>
        </section>

        // App paste panel — available alongside conversation
        <section class="install-section">
            <AppPastePanel />
        </section>

        // Deploy panel (shown when Symthaea says ready or user clicks deploy)
        <Show when=move || show_remote.get() || ready_to_deploy.get()>
            <section class="install-section">
                <RemoteInstallPanel
                    config_nix=config_nix_signal.into()
                    flake_nix=flake_nix_signal.into()
                    hostname=hostname
                    desktop=selected_desktop
                    gpu_driver=gpu_driver
                    timezone=timezone
                    keyboard=keyboard
                    encrypt=encrypt_disk
                    secure_boot=secure_boot
                    tpm_unlock=tpm_unlock
                    fido2_unlock=fido2_unlock
                    disk_layout=disk_layout
                    filesystem=filesystem
                    user_password=user_password
                />
            </section>
        </Show>

        // Download config manually
        {move || config_nix_signal.get().map(|cfg| {
            let cfg_dl = cfg.clone();
            let flake_dl = flake_nix_signal.get().unwrap_or_default();
            view! {
                <section class="install-section">
                    <GlassPanel title="Download Configuration">
                        <div class="config-buttons">
                            <button class="btn-secondary" on:click=move |_| download_text("configuration.nix", &cfg_dl)>
                                "Download configuration.nix"
                            </button>
                            <button class="btn-secondary" on:click=move |_| download_text("flake.nix", &flake_dl)>
                                "Download flake.nix"
                            </button>
                        </div>
                        <details class="advanced-section">
                            <summary class="advanced-toggle">"View raw configuration.nix"</summary>
                            <pre class="config-code">{cfg.clone()}</pre>
                        </details>
                    </GlassPanel>
                </section>
            }
        })}

        <div class="wizard-buttons">
            <button class="btn-secondary" on:click=move |_| { mode.set(String::new()); }>"Back to mode selection"</button>
            <span></span>
        </div>
    }
}

/// Format conversation text: replace **bold** with <strong> tags.
fn format_converse_text(text: &str) -> String {
    let mut result = text.replace('<', "&lt;").replace('>', "&gt;");
    // Simple **bold** replacement
    while let Some(start) = result.find("**") {
        if let Some(end) = result[start + 2..].find("**") {
            let bold_text = &result[start + 2..start + 2 + end];
            let replacement = format!("<strong>{}</strong>", bold_text);
            result = format!(
                "{}{}{}",
                &result[..start],
                replacement,
                &result[start + 2 + end + 2..]
            );
        } else {
            break;
        }
    }
    result
}

// ═══════════════════════════════════════════════════════
// Wizard Navigation
// ═══════════════════════════════════════════════════════

fn wizard_labels(lang: Lang) -> [&'static str; 5] {
    [
        i18n::t(lang, "step_system"),
        i18n::t(lang, "step_desktop"),
        i18n::t(lang, "step_apps"),
        i18n::t(lang, "step_config"),
        i18n::t(lang, "step_install"),
    ]
}

/// Map AppCategory to a tab filter string.
fn category_tab_key(cat: AppCategory) -> &'static str {
    match cat {
        AppCategory::Browser => "browser",
        AppCategory::Email | AppCategory::Communication => "communication",
        AppCategory::Office | AppCategory::Notes => "office",
        AppCategory::Editor | AppCategory::IDE => "editor",
        AppCategory::Terminal
        | AppCategory::VersionControl
        | AppCategory::Container
        | AppCategory::DevTools => "dev",
        AppCategory::Creative2D
        | AppCategory::Creative3D
        | AppCategory::Photo
        | AppCategory::Audio
        | AppCategory::Video => "creative",
        AppCategory::Gaming | AppCategory::GamingTools => "gaming",
        AppCategory::Streaming | AppCategory::MediaPlayer => "media",
        AppCategory::FileManager
        | AppCategory::Archive
        | AppCategory::Security
        | AppCategory::VPN
        | AppCategory::Backup
        | AppCategory::SystemUtil
        | AppCategory::Virtualization => "system",
        AppCategory::Science | AppCategory::Finance => "other",
    }
}

#[component]
fn WizardNav(step: RwSignal<u32>, lang: RwSignal<Lang>) -> impl IntoView {
    view! {
        <nav class="wizard-nav" aria-label="Installation steps">
            {move || {
                let labels = wizard_labels(lang.get());
                labels.iter().enumerate().map(|(i, label)| {
                    let num = (i + 1) as u32;
                    let label = *label;
                    view! {
                        <button
                            class=move || {
                                let s = step.get();
                                if s == num { "wizard-step wizard-step-active" }
                                else if s > num { "wizard-step wizard-step-done" }
                                else { "wizard-step" }
                            }
                            attr:aria-current=move || if step.get() == num { Some("step") } else { None }
                            on:click=move |_| { if step.get() > num { step.set(num); } }
                        >
                            <span class="wizard-step-num">
                                {move || if step.get() > num {
                                    "\u{2713}".to_string()
                                } else {
                                    num.to_string()
                                }}
                            </span>
                            <span class="wizard-step-label">{label}</span>
                        </button>
                    }
                }).collect::<Vec<_>>()
            }}
        </nav>
    }
}

// ═══════════════════════════════════════════════════════
// Main Install Page
// ═══════════════════════════════════════════════════════

#[component]
pub fn InstallPage() -> impl IntoView {
    let hardware = detect_hardware();
    let os = detect_os(&hardware.platform);
    let gpu: &'static str = Box::leak(gpu_short(&hardware.gpu_renderer).into_boxed_str());
    let real_gpu_vendor = normalize_gpu_vendor(&hardware.gpu_vendor, &hardware.gpu_renderer);
    let is_nvidia = real_gpu_vendor == "nvidia";

    let device_mem_exact = js_sys::Reflect::get(
        &web_sys::window().unwrap().navigator(),
        &"deviceMemory".into(),
    )
    .ok()
    .and_then(|v| v.as_f64())
    .filter(|&m| m > 0.0)
    .is_some();
    let mem_text: &'static str = Box::leak(
        hardware
            .memory_gb
            .filter(|&m| m > 0.0)
            .map(|m| {
                if device_mem_exact {
                    format!("{}GB", m as u32)
                } else {
                    format!("~{}GB", m as u32)
                }
            })
            .unwrap_or_default()
            .into_boxed_str(),
    );

    // Language — restored from localStorage, fallback to browser detection
    let lang: RwSignal<Lang> = RwSignal::new(
        load_from_storage("si_lang")
            .map(|code| Lang::from_code(&code))
            .unwrap_or_else(i18n::detect_browser_lang),
    );

    // Install mode: "" = not chosen, "express", "custom"
    let mode: RwSignal<String> = RwSignal::new(load_from_storage("si_mode").unwrap_or_default());

    // Wizard step (1-5, step 6 = completion state)
    let step: RwSignal<u32> = RwSignal::new(
        load_from_storage("si_step")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1),
    );

    // App category filter for Step 3
    let app_category: RwSignal<String> = RwSignal::new("all".to_string());

    // System basics — auto-detected with user override, restored from localStorage (#1, #2, #3, #11)
    let hostname =
        RwSignal::new(load_from_storage("si_hostname").unwrap_or_else(|| "nixos".to_string()));
    let username =
        RwSignal::new(load_from_storage("si_username").unwrap_or_else(|| "user".to_string()));
    let timezone = RwSignal::new(load_from_storage("si_timezone").unwrap_or_else(detect_timezone));
    let keyboard = RwSignal::new(load_from_storage("si_keyboard").unwrap_or_else(detect_keyboard));

    let locale = RwSignal::new(load_from_storage("si_locale").unwrap_or_else(detect_locale));

    // Desktop & security — restored from localStorage
    let selected_desktop =
        RwSignal::new(load_from_storage("si_desktop").unwrap_or_else(|| "gnome".to_string()));
    let encrypt_disk = RwSignal::new(
        load_from_storage("si_encrypt")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let secure_boot = RwSignal::new(
        load_from_storage("si_secureboot")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let tpm_unlock = RwSignal::new(
        load_from_storage("si_tpm")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let fido2_unlock = RwSignal::new(
        load_from_storage("si_fido2")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let disk_layout = RwSignal::new(load_from_storage("si_disk_layout").unwrap_or_else(|| {
        if encrypt_disk.get_untracked() {
            "single-luks".to_string()
        } else {
            "single".to_string()
        }
    }));
    let filesystem =
        RwSignal::new(load_from_storage("si_filesystem").unwrap_or_else(|| "btrfs".to_string()));

    // New user options — restored from localStorage
    let swap_gb = RwSignal::new(
        load_from_storage("si_swap")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8u32),
    );
    let shell = RwSignal::new(load_from_storage("si_shell").unwrap_or_else(|| "bash".into()));
    let bluetooth = RwSignal::new(
        load_from_storage("si_bluetooth")
            .map(|v| v == "true")
            .unwrap_or(true),
    );
    let printing = RwSignal::new(
        load_from_storage("si_printing")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let kernel = RwSignal::new(load_from_storage("si_kernel").unwrap_or_else(|| "default".into()));
    let is_laptop = RwSignal::new(
        load_from_storage("si_laptop")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let home_manager = RwSignal::new(
        load_from_storage("si_homemanager")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let symthaea_edition = RwSignal::new(
        load_from_storage("si_symthaea")
            .map(|v| v == "true")
            .unwrap_or(false),
    );
    let mycelix_edition = RwSignal::new(
        load_from_storage("si_mycelix")
            .map(|v| v == "true")
            .unwrap_or(false),
    );

    // Auto-login option
    let auto_login = RwSignal::new(
        load_from_storage("si_autologin")
            .map(|v| v == "true")
            .unwrap_or(false),
    );

    // Extra users (comma-separated string)
    let extra_users_str = RwSignal::new(load_from_storage("si_extra_users").unwrap_or_default());

    // Password fields — NOT persisted to localStorage (security)
    let user_password = RwSignal::new(String::new());
    let password_confirm = RwSignal::new(String::new());

    // App selection — restored from localStorage
    let selected_apps = RwSignal::new(
        load_from_storage("si_apps")
            .and_then(|v| serde_json::from_str::<Vec<String>>(&v).ok())
            .unwrap_or_default(),
    );
    let custom_pkgs = RwSignal::new(load_from_storage("si_custom_pkgs").unwrap_or_default());

    // Nixpkgs search state
    let nix_search_query = RwSignal::new(String::new());
    let nix_search_results: RwSignal<Vec<(String, String)>> = RwSignal::new(Vec::new());

    // Package validation warnings from relay-side nixpkgs check
    let package_warnings: RwSignal<Vec<String>> = RwSignal::new(Vec::new());

    // Persist all installer state to localStorage on any change
    Effect::new(move |_| {
        save_to_storage("si_lang", lang.get().code());
        save_to_storage("si_mode", &mode.get());
        save_to_storage("si_step", &step.get().to_string());
        save_to_storage("si_hostname", &hostname.get());
        save_to_storage("si_username", &username.get());
        save_to_storage("si_timezone", &timezone.get());
        save_to_storage("si_keyboard", &keyboard.get());
        save_to_storage("si_desktop", &selected_desktop.get());
        save_to_storage("si_encrypt", &encrypt_disk.get().to_string());
        save_to_storage("si_secureboot", &secure_boot.get().to_string());
        save_to_storage("si_tpm", &tpm_unlock.get().to_string());
        save_to_storage("si_fido2", &fido2_unlock.get().to_string());
        save_to_storage("si_disk_layout", &disk_layout.get());
        save_to_storage("si_filesystem", &filesystem.get());
        save_to_storage("si_locale", &locale.get());
        save_to_storage(
            "si_apps",
            &serde_json::to_string(&selected_apps.get()).unwrap_or_default(),
        );
        save_to_storage("si_custom_pkgs", &custom_pkgs.get());
        save_to_storage("si_swap", &swap_gb.get().to_string());
        save_to_storage("si_shell", &shell.get());
        save_to_storage("si_bluetooth", &bluetooth.get().to_string());
        save_to_storage("si_printing", &printing.get().to_string());
        save_to_storage("si_kernel", &kernel.get());
        save_to_storage("si_laptop", &is_laptop.get().to_string());
        save_to_storage("si_homemanager", &home_manager.get().to_string());
        save_to_storage("si_symthaea", &symthaea_edition.get().to_string());
        save_to_storage("si_mycelix", &mycelix_edition.get().to_string());
        save_to_storage("si_autologin", &auto_login.get().to_string());
        save_to_storage("si_extra_users", &extra_users_str.get());
        // NOTE: user_password and password_confirm are NOT persisted (security)
    });

    // Focus management: move focus to first input when step changes
    Effect::new(move |_| {
        let _s = step.get(); // subscribe to step changes
        if let Some(window) = web_sys::window() {
            let cb = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
                if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
                    if let Ok(Some(el)) = doc.query_selector(".install-section input, .install-section select, .install-section button.btn-primary") {
                        if let Ok(el) = el.dyn_into::<web_sys::HtmlElement>() {
                            let _ = el.focus();
                        }
                    }
                }
            });
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                100,
            );
            cb.forget();
        }
    });

    // Config state
    let (config_preview, set_config_preview) = signal(Option::<String>::None);
    let (config_warnings, set_config_warnings) = signal(Vec::<String>::new());
    let (show_next_steps, set_show_next_steps) = signal(false);

    // Validation errors (cleared on each validation run, not persisted)
    let validation_errors: RwSignal<Vec<String>> = RwSignal::new(Vec::new());

    // Scan agent
    let (scan_status, set_scan_status) = signal(String::new());
    let scan_apps = {
        let selected = selected_apps;
        let set_status = set_scan_status;
        move || {
            let selected = selected;
            let set_status = set_status;
            set_status.set("Connecting to scanner...".into());
            wasm_bindgen_futures::spawn_local(async move {
                // Token auth: sovereign-scan now requires an explicit token to avoid CSWSH data exfil.
                let token = {
                    let window = web_sys::window().unwrap();
                    let storage = window.local_storage().ok().flatten();
                    let mut token = storage
                        .as_ref()
                        .and_then(|s| s.get_item("si_scan_token").ok())
                        .flatten()
                        .filter(|t| !t.trim().is_empty());

                    if token.is_none() {
                        token = window
                            .prompt_with_message(
                                "Enter sovereign-scan token (printed by `sovereign-scan --serve`):",
                            )
                            .ok()
                            .flatten()
                            .filter(|t| !t.trim().is_empty());
                        if let (Some(ref t), Some(ref s)) = (token.as_ref(), storage.as_ref()) {
                            let _ = s.set_item("si_scan_token", t);
                        }
                    }

                    token
                };

                let Some(token) = token else {
                    set_status.set("Scanner token required (cancelled).".into());
                    return;
                };

                let ws = web_sys::WebSocket::new("ws://127.0.0.1:7799");
                match ws {
                    Ok(ws) => {
                        let ws_clone = ws.clone();
                        // Authenticate immediately on open.
                        let ws_for_open = ws.clone();
                        let token_for_open = token.clone();
                        let onopen = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
                            let auth = serde_json::json!({
                                "action": "auth",
                                "token": token_for_open,
                            });
                            let _ = ws_for_open.send_with_str(&auth.to_string());
                        });
                        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
                        onopen.forget();

                        let onmessage = wasm_bindgen::closure::Closure::<
                            dyn Fn(web_sys::MessageEvent),
                        >::new(
                            move |e: web_sys::MessageEvent| {
                                if let Some(text) = e.data().as_string() {
                                    if let Ok(msg) =
                                        serde_json::from_str::<serde_json::Value>(&text)
                                    {
                                        if msg.get("type").and_then(|v| v.as_str()) == Some("error")
                                        {
                                            set_status.set("Scanner auth failed. Check the token and restart sovereign-scan.".into());
                                            return;
                                        }
                                        if msg.get("type").and_then(|v| v.as_str())
                                            == Some("scan_result")
                                        {
                                            if let Some(data) = msg.get("data") {
                                                if let Some(apps) = data
                                                    .get("installed_apps")
                                                    .and_then(|v| v.as_array())
                                                {
                                                    let mut found = Vec::new();
                                                    for app in apps {
                                                        if let Some(name) = app
                                                            .get("canonical_name")
                                                            .and_then(|v| v.as_str())
                                                        {
                                                            found.push(name.to_string());
                                                        }
                                                    }
                                                    let count = found.len();
                                                    selected.update(|s| {
                                                        for name in found {
                                                            if !s.contains(&name) {
                                                                s.push(name);
                                                            }
                                                        }
                                                    });
                                                    set_status.set(format!(
                                                        "{count} apps detected and selected"
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                        );
                        ws_clone.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
                        onmessage.forget();
                        let onerror = wasm_bindgen::closure::Closure::<dyn Fn()>::new(move || {
                            set_status.set(
                                "Scanner not found. Run: sovereign-scan --serve (copy token)."
                                    .into(),
                            );
                        });
                        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
                        onerror.forget();
                    }
                    Err(_) => set_status.set("WebSocket not supported".into()),
                }
            });
        }
    };

    let selected_count = move || selected_apps.get().len();

    // GPU driver derived from hardware
    let gpu_vendor_str = real_gpu_vendor.clone();
    let gpu_driver = Signal::derive({
        let gv = gpu_vendor_str.clone();
        move || {
            if gv.contains("nvidia") {
                "nvidia".to_string()
            } else if gv.contains("amd") || gv.contains("ati") {
                "amdgpu".to_string()
            } else {
                "modesetting".to_string()
            }
        }
    });

    // Config signals for remote install
    let config_nix_signal: RwSignal<Option<String>> = RwSignal::new(None);
    let flake_nix_signal: RwSignal<Option<String>> = RwSignal::new(None);

    // Configuration profiles (Save/Load named configs to localStorage)
    let profiles: RwSignal<Vec<(String, String)>> = RwSignal::new(
        load_from_storage("si_profiles")
            .and_then(|v| serde_json::from_str::<Vec<(String, String)>>(&v).ok())
            .unwrap_or_default(),
    );
    let profile_name: RwSignal<String> = RwSignal::new(String::new());

    // Clones for closures
    let gpu_vendor = gpu_vendor_str.clone();
    let gpu_model = hardware.gpu_renderer.clone();
    let hw_cores = hardware.cpu_cores;
    let hw_mem = hardware.memory_gb.unwrap_or(0.0) as u32;
    // Store GPU info in signals so closures can share them
    let sig_gv = StoredValue::new(gpu_vendor.clone());
    let sig_gm = StoredValue::new(gpu_model.clone());

    // Parse custom packages from comma-separated string
    let parse_custom_pkgs = move || -> Vec<String> {
        custom_pkgs
            .get()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    // Parse extra users from comma-separated string
    let parse_extra_users = move || -> Vec<String> {
        extra_users_str
            .get()
            .split(',')
            .map(|s| s.trim().to_lowercase().replace(' ', ""))
            .filter(|s| !s.is_empty())
            .collect()
    };

    // Search nixpkgs aliases (browser-side, instant, 485+ packages)
    let do_nix_search = move || {
        let query = nix_search_query.get().trim().to_lowercase();
        if query.len() < 2 {
            nix_search_results.set(Vec::new());
            return;
        }

        // Check localStorage cache first
        let cache_key = format!("nix_cache_{}", query);
        if let Some(cached) = load_from_storage(&cache_key) {
            if let Ok(results) = serde_json::from_str::<Vec<(String, String)>>(&cached) {
                nix_search_results.set(results);
                return;
            }
        }

        // Search the alias database: prefix > contains alias > contains pkg
        let aliases = symthaea_app_db::aliases::all_aliases();
        let mut scored: Vec<(u8, String, String)> = Vec::new();
        for &(alias, pkg) in aliases {
            if alias.starts_with(&query) {
                scored.push((3, alias.to_string(), pkg.to_string()));
            } else if alias.contains(&query) {
                scored.push((2, alias.to_string(), pkg.to_string()));
            } else if pkg.to_lowercase().contains(&query) {
                scored.push((1, alias.to_string(), pkg.to_string()));
            }
        }
        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        scored.truncate(20);

        let results: Vec<(String, String)> = scored.into_iter().map(|(_, a, p)| (a, p)).collect();

        // Cache results
        if let Ok(json) = serde_json::to_string(&results) {
            save_to_storage(&cache_key, &json);
        }

        nix_search_results.set(results);
    };

    // Generate config helper (used by "Next" on step 4 and Generate button)
    let do_generate = move || {
        let hw = config_gen::HardwareProfile {
            gpu_vendor: sig_gv.get_value(),
            gpu_model: sig_gm.get_value(),
            cpu_cores: hw_cores,
            memory_gb: hw_mem,
            has_wifi: true,
            chromebook: false,
        };
        let choices = config_gen::UserChoices {
            hostname: hostname.get(),
            username: username.get(),
            desktop: selected_desktop.get(),
            encryption: encrypt_disk.get(),
            secure_boot: secure_boot.get(),
            tpm2_unlock: tpm_unlock.get(),
            fido2_unlock: fido2_unlock.get(),
            timezone: timezone.get(),
            keyboard: keyboard.get(),
            locale: locale.get(),
            custom_packages: parse_custom_pkgs(),
            filesystem: filesystem.get(),
            swap_gb: swap_gb.get(),
            shell: shell.get(),
            bluetooth: bluetooth.get(),
            printing: printing.get(),
            kernel: kernel.get(),
            is_laptop: is_laptop.get(),
            home_manager: home_manager.get(),
            symthaea_edition: symthaea_edition.get(),
            mycelix_edition: mycelix_edition.get(),
            auto_login: auto_login.get(),
            extra_users: parse_extra_users(),
        };
        let config = config_gen::generate(&hw, &choices, &selected_apps.get());
        let mut all_warnings = config.warnings.clone();
        // Validate generated Nix syntax
        let syntax_errors = config_gen::validate_nix_syntax(&config.configuration_nix);
        for err in &syntax_errors {
            all_warnings.push(format!("Syntax issue in configuration.nix: {}", err));
        }
        let flake_errors = config_gen::validate_nix_syntax(&config.flake_nix);
        for err in &flake_errors {
            all_warnings.push(format!("Syntax issue in flake.nix: {}", err));
        }
        set_config_warnings.set(all_warnings);
        set_config_preview.set(Some(config.configuration_nix.clone()));
        config_nix_signal.set(Some(config.configuration_nix.clone()));
        flake_nix_signal.set(Some(config.flake_nix.clone()));
        set_show_next_steps.set(true);
    };

    view! {
        <div class="install-page">
            <header class="install-hero">
                <select class="lang-picker"
                    on:change=move |ev| {
                        let code = event_target_value(&ev);
                        let l = Lang::from_code(&code);
                        lang.set(l);
                        save_to_storage("si_lang", l.code());
                        // Set RTL if needed
                        if let Some(doc) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.document_element()) {
                            let _ = doc.set_attribute("dir", if l.is_rtl() { "rtl" } else { "ltr" });
                        }
                    }
                >
                    {Lang::all().iter().map(|l| {
                        let code = l.code();
                        let name = l.name();
                        let l = *l;
                        view! { <option value=code selected=move || lang.get() == l>{name}</option> }
                    }).collect::<Vec<_>>()}
                </select>
                <h1 class="install-title">{move || i18n::t(lang.get(), "install_title")}</h1>
                <p class="install-sub">{move || i18n::t(lang.get(), "install_subtitle")}</p>
                <details class="help-expander nixos-intro">
                    <summary class="help-toggle">{move || i18n::t(lang.get(), "nixos_intro_toggle")}</summary>
                    <p class="help-text">
                        {move || i18n::t(lang.get(), "nixos_intro_text")}
                    </p>
                </details>
            </header>

            // ══════════════════════════════════════════
            // Mode Selection (before wizard starts)
            // ══════════════════════════════════════════
            <Show when=move || mode.get().is_empty()>
                <div class="mode-select">
                    <h2>"How do you want to install?"</h2>

                    <div class="mode-cards">
                        <div class="mode-card mode-express" role="button" tabindex="0"
                            on:click=move |_| { mode.set("express".into()); step.set(1); save_to_storage("si_mode", "express"); }
                            on:touchend=move |ev: web_sys::TouchEvent| { ev.prevent_default(); mode.set("express".into()); step.set(1); save_to_storage("si_mode", "express"); }
                        >
                            <span class="mode-icon">"⚡"</span>
                            <span class="mode-title">"Express"</span>
                            <span class="mode-desc">"Pick a preset, set your password, install. Under 2 minutes."</span>
                        </div>

                        <div class="mode-card mode-custom" role="button" tabindex="0"
                            on:click=move |_| { mode.set("custom".into()); step.set(1); save_to_storage("si_mode", "custom"); }
                            on:touchend=move |ev: web_sys::TouchEvent| { ev.prevent_default(); mode.set("custom".into()); step.set(1); save_to_storage("si_mode", "custom"); }
                        >
                            <span class="mode-icon">"🔧"</span>
                            <span class="mode-title">"Custom"</span>
                            <span class="mode-desc">"Choose your desktop, apps, encryption, shell, kernel — full control."</span>
                        </div>

                        <div class="mode-card mode-converse" role="button" tabindex="0"
                            on:click=move |_| { mode.set("converse".into()); step.set(1); save_to_storage("si_mode", "converse"); }
                            on:touchend=move |ev: web_sys::TouchEvent| { ev.prevent_default(); mode.set("converse".into()); step.set(1); save_to_storage("si_mode", "converse"); }
                        >
                            <span class="mode-icon">"🌿"</span>
                            <span class="mode-title">"Talk with Symthaea"</span>
                            <span class="mode-desc">"Describe what you need. She configures your system through conversation."</span>
                        </div>
                    </div>
                </div>
            </Show>

            // ══════════════════════════════════════════
            // Express Mode (2 steps)
            // ══════════════════════════════════════════
            <Show when=move || mode.get() == "express">
                // ── Express Step 1: Preset + Password ──
                <Show when=move || step.get() == 1>
                    <section class="install-section">
                        <GlassPanel title="Pick Your Setup">
                            <p class="section-desc">"Choose a preset and we handle the rest."</p>
                            <div class="preset-grid preset-grid-express">
                                <button class="desktop-card" on:click=move |_| {
                                    selected_desktop.set("gnome".into());
                                    shell.set("bash".into());
                                    kernel.set("default".into());
                                    swap_gb.set(8);
                                    bluetooth.set(true);
                                    encrypt_disk.set(false);
                                    disk_layout.set("single".into());
                                    selected_apps.update(|a| {
                                        a.clear();
                                        for app in ["Firefox", "Visual Studio Code", "Git", "Docker Desktop", "Alacritty"] {
                                            a.push(app.to_string());
                                        }
                                    });
                                }>
                                    <span class="preset-emoji">"💻"</span>
                                    <span class="desktop-name">"Developer"</span>
                                    <span class="desktop-desc">"GNOME + VS Code + Git + Docker + Alacritty"</span>
                                </button>
                                <button class="desktop-card" on:click=move |_| {
                                    selected_desktop.set("none".into());
                                    shell.set("bash".into());
                                    kernel.set("lts".into());
                                    swap_gb.set(4);
                                    bluetooth.set(false);
                                    encrypt_disk.set(true);
                                    disk_layout.set("single-luks".into());
                                    selected_apps.update(|a| a.clear());
                                }>
                                    <span class="preset-emoji">"🖥"</span>
                                    <span class="desktop-name">"Server"</span>
                                    <span class="desktop-desc">"No GUI, LTS kernel, LUKS encryption"</span>
                                </button>
                                <button class="desktop-card" on:click=move |_| {
                                    selected_desktop.set("gnome".into());
                                    shell.set("bash".into());
                                    kernel.set("default".into());
                                    swap_gb.set(8);
                                    bluetooth.set(true);
                                    printing.set(true);
                                    encrypt_disk.set(false);
                                    disk_layout.set("single".into());
                                    selected_apps.update(|a| {
                                        a.clear();
                                        for app in ["Firefox", "LibreOffice", "Thunderbird", "VLC", "GIMP"] {
                                            a.push(app.to_string());
                                        }
                                    });
                                }>
                                    <span class="preset-emoji">"🏠"</span>
                                    <span class="desktop-name">"Home / Office"</span>
                                    <span class="desktop-desc">"GNOME + LibreOffice + Firefox + printing"</span>
                                </button>
                                <button class="desktop-card" on:click=move |_| {
                                    selected_desktop.set("hyprland".into());
                                    shell.set("zsh".into());
                                    kernel.set("zen".into());
                                    swap_gb.set(16);
                                    bluetooth.set(true);
                                    encrypt_disk.set(false);
                                    disk_layout.set("single".into());
                                    selected_apps.update(|a| {
                                        a.clear();
                                        for app in ["Firefox", "Alacritty", "Visual Studio Code", "Steam", "Lutris"] {
                                            a.push(app.to_string());
                                        }
                                    });
                                }>
                                    <span class="preset-emoji">"🎮"</span>
                                    <span class="desktop-name">"Gaming"</span>
                                    <span class="desktop-desc">"Hyprland + Zen kernel + Steam + Lutris"</span>
                                </button>
                                <button class="desktop-card" on:click=move |_| {
                                    selected_desktop.set("gnome".into());
                                    shell.set("zsh".into());
                                    kernel.set("default".into());
                                    swap_gb.set(8);
                                    bluetooth.set(true);
                                    encrypt_disk.set(false);
                                    disk_layout.set("single".into());
                                    symthaea_edition.set(true);
                                    mycelix_edition.set(true);
                                    selected_apps.update(|a| {
                                        a.clear();
                                        for app in ["Firefox", "Visual Studio Code", "Git", "Alacritty"] {
                                            a.push(app.to_string());
                                        }
                                    });
                                }>
                                    <span class="preset-emoji">"🌿"</span>
                                    <span class="desktop-name">"Sovereign"</span>
                                    <span class="desktop-desc">"GNOME + Symthaea AI + Mycelix network"</span>
                                </button>
                            </div>
                        </GlassPanel>
                    </section>
                    <section class="install-section">
                        <GlassPanel title="Your Account">
                            <div class="basics-grid">
                                <div class="field">
                                    <label class="field-label" for="express-username">"Username"</label>
                                    <input id="express-username" type="text" class="field-input"
                                        placeholder="user"
                                        prop:value=move || username.get()
                                        on:input=move |ev| username.set(event_target_value(&ev).to_lowercase().replace(' ', ""))
                                    />
                                </div>
                                <div class="field">
                                    <label class="field-label">"Password"</label>
                                    <input type="password" class="field-input" placeholder="Required (8+ characters)"
                                        prop:value=move || user_password.get()
                                        on:input=move |ev| user_password.set(event_target_value(&ev))
                                    />
                                </div>
                                <div class="field">
                                    <label class="field-label">"Confirm Password"</label>
                                    <input type="password" class="field-input" placeholder="Must match"
                                        prop:value=move || password_confirm.get()
                                        on:input=move |ev| password_confirm.set(event_target_value(&ev))
                                    />
                                    {move || {
                                        let p = user_password.get();
                                        let c = password_confirm.get();
                                        if !c.is_empty() && p != c {
                                            Some(view! { <span class="error-msg" style="font-size: 0.75rem;">"Passwords don't match"</span> })
                                        } else if !p.is_empty() && p.len() < 8 {
                                            Some(view! { <span class="warning-msg" style="font-size: 0.75rem;">"Password should be at least 8 characters"</span> })
                                        } else { None }
                                    }}
                                </div>
                                <div class="field">
                                    <label class="field-label" for="express-hostname">"Hostname"</label>
                                    <div style="display:flex;gap:0.4rem;align-items:center;">
                                        <input id="express-hostname" type="text" class="field-input" style="flex:1;"
                                            placeholder="my-nixos"
                                            prop:value=move || hostname.get()
                                            on:input=move |ev| hostname.set(event_target_value(&ev))
                                        />
                                        <button class="btn-secondary btn-sm" style="white-space:nowrap;" on:click=move |_| hostname.set(random_hostname())>"Suggest"</button>
                                    </div>
                                </div>
                            </div>
                        </GlassPanel>
                    </section>
                    {move || {
                        let errs = validation_errors.get();
                        (!errs.is_empty()).then(|| view! {
                            <div class="validation-errors" role="alert" aria-live="assertive">
                                {errs.iter().map(|e| view! { <p class="error-msg">{e.clone()}</p> }).collect::<Vec<_>>()}
                            </div>
                        })
                    }}
                    <div class="wizard-buttons">
                        <button class="btn-secondary" on:click=move |_| { mode.set(String::new()); }>"Back"</button>
                        <button class="btn-primary" on:click=move |_| {
                            let mut errs = Vec::new();
                            let pw = user_password.get();
                            let pc = password_confirm.get();
                            if username.get().is_empty() { errs.push("Username is required".into()); }
                            if pw.is_empty() { errs.push("Password is required".into()); }
                            else if pw.len() < 8 { errs.push("Password must be at least 8 characters".into()); }
                            else if pw != pc { errs.push("Passwords do not match".into()); }
                            validation_errors.set(errs);
                            if validation_errors.get().is_empty() {
                                do_generate();
                                step.set(2);
                            }
                        }>"Next"</button>
                    </div>
                </Show>

                // ── Express Step 2: Connect & Install ──
                <Show when=move || step.get() == 2>
                    <section class="install-section">
                        <RemoteInstallPanel
                            config_nix=config_nix_signal.into()
                            flake_nix=flake_nix_signal.into()
                            hostname=hostname
                            desktop=selected_desktop
                            gpu_driver=gpu_driver
                            timezone=timezone
                            keyboard=keyboard
                            encrypt=encrypt_disk
                            secure_boot=secure_boot
                            tpm_unlock=tpm_unlock
                            fido2_unlock=fido2_unlock
                            disk_layout=disk_layout
                            filesystem=filesystem
                            user_password=user_password
                        />
                    </section>
                    <div class="wizard-buttons">
                        <button class="btn-secondary" on:click=move |_| step.set(1)>"Back"</button>
                        <span></span>
                    </div>
                </Show>
            </Show>

            // ══════════════════════════════════════════
            // Custom Mode (existing 5-step wizard)
            // ══════════════════════════════════════════
            <Show when=move || mode.get() == "custom">

            <div aria-live="polite" class="sr-only">
                {move || {
                    let labels = wizard_labels(lang.get());
                    format!("Step {} of 5: {}", step.get(), labels.get(step.get() as usize - 1).unwrap_or(&""))
                }}
            </div>

            <WizardNav step=step lang=lang />

            // ── Step 1: Your System ──
            <Show when=move || step.get() == 1>
                <section class="install-section">
                    <div class="hw-grid">
                        <div class="hw-item">
                            <span class="hw-label">{move || i18n::t(lang.get(), "platform")}</span>
                            <span class="hw-value">{os}</span>
                        </div>
                        <div class="hw-item">
                            <span class="hw-label">{move || i18n::t(lang.get(), "cpu")}</span>
                            <span class="hw-value">{format!("{} cores", hardware.cpu_cores)}</span>
                        </div>
                        {(!mem_text.is_empty()).then(|| view! {
                            <div class="hw-item">
                                <span class="hw-label">{move || i18n::t(lang.get(), "memory")}</span>
                                <span class="hw-value">{mem_text}</span>
                            </div>
                        })}
                        {(!gpu.is_empty()).then(|| view! {
                            <div class="hw-item">
                                <span class="hw-label">{move || i18n::t(lang.get(), "gpu")}</span>
                                <span class="hw-value">{gpu}</span>
                            </div>
                        })}
                        <div class="hw-item">
                            <span class="hw-label">{move || i18n::t(lang.get(), "display")}</span>
                            <span class="hw-value">{format!("{}x{}", hardware.screen_width, hardware.screen_height)}</span>
                        </div>
                    </div>
                    {is_nvidia.then(|| view! {
                        <p class="hw-note">{format!("NVIDIA {} — proprietary drivers will be configured.", gpu)}</p>
                    })}
                </section>
                <section class="install-section">
                    <SystemBasics hostname=hostname username=username user_password=user_password password_confirm=password_confirm extra_users_str=extra_users_str timezone=timezone keyboard=keyboard locale=locale />
                </section>
                {move || {
                    let errs = validation_errors.get();
                    (!errs.is_empty()).then(|| view! {
                        <div class="validation-errors" role="alert" aria-live="assertive">
                            {errs.iter().map(|e| view! { <p class="error-msg">{e.clone()}</p> }).collect::<Vec<_>>()}
                        </div>
                    })
                }}
                <div class="wizard-buttons">
                    <span></span>
                    <button class="btn-primary" on:click=move |_| {
                        let result = validation::validate_step1(
                            &hostname.get(), &username.get(), &timezone.get(), &keyboard.get(),
                        );
                        let mut errs = result.errors;
                        // Password validation
                        let pw = user_password.get();
                        let pc = password_confirm.get();
                        if pw.is_empty() {
                            errs.push("Password is required".into());
                        } else if pw.len() < 8 {
                            errs.push("Password must be at least 8 characters".into());
                        } else if pw != pc {
                            errs.push("Passwords do not match".into());
                        }
                        validation_errors.set(errs);
                        if validation_errors.get().is_empty() {
                            step.set(2);
                        }
                    }>{move || i18n::t(lang.get(), "next")}</button>
                </div>
            </Show>

            // ── Step 2: Desktop & Presets (+ collapsible Advanced) ──
            <Show when=move || step.get() == 2>
                // Quick Presets
                <section class="install-section">
                    <GlassPanel title="Quick Presets">
                        <p class="section-desc">{move || i18n::t(lang.get(), "presets_desc")}</p>
                        <div class="preset-grid">
                            <button class="desktop-card" on:click=move |_| {
                                selected_desktop.set("gnome".into());
                                selected_apps.update(|a| {
                                    for app in ["Firefox", "Visual Studio Code", "Git", "Docker Desktop", "Alacritty"] {
                                        if !a.contains(&app.to_string()) { a.push(app.to_string()); }
                                    }
                                });
                            }>
                                <span class="desktop-name">{move || i18n::t(lang.get(), "preset_dev")}</span>
                                <span class="desktop-desc">{move || i18n::t(lang.get(), "preset_dev_desc")}</span>
                            </button>
                            <button class="desktop-card" on:click=move |_| {
                                selected_desktop.set("none".into());
                                encrypt_disk.set(true);
                            }>
                                <span class="desktop-name">{move || i18n::t(lang.get(), "preset_server")}</span>
                                <span class="desktop-desc">{move || i18n::t(lang.get(), "preset_server_desc")}</span>
                            </button>
                            <button class="desktop-card" on:click=move |_| {
                                selected_desktop.set("gnome".into());
                                selected_apps.update(|a| {
                                    for app in ["Firefox", "LibreOffice", "Thunderbird", "VLC", "GIMP"] {
                                        if !a.contains(&app.to_string()) { a.push(app.to_string()); }
                                    }
                                });
                            }>
                                <span class="desktop-name">{move || i18n::t(lang.get(), "preset_home")}</span>
                                <span class="desktop-desc">{move || i18n::t(lang.get(), "preset_home_desc")}</span>
                            </button>
                            <button class="desktop-card" on:click=move |_| {
                                selected_desktop.set("hyprland".into());
                                selected_apps.update(|a| {
                                    for app in ["Firefox", "Alacritty", "Visual Studio Code", "Steam", "Lutris"] {
                                        if !a.contains(&app.to_string()) { a.push(app.to_string()); }
                                    }
                                });
                            }>
                                <span class="desktop-name">{move || i18n::t(lang.get(), "preset_gaming")}</span>
                                <span class="desktop-desc">{move || i18n::t(lang.get(), "preset_gaming_desc")}</span>
                            </button>
                            <button class="desktop-card" on:click=move |_| {
                                selected_desktop.set("gnome".into());
                                symthaea_edition.set(true);
                                mycelix_edition.set(true);
                                selected_apps.update(|a| {
                                    for app in ["Firefox", "Visual Studio Code", "Git", "Alacritty"] {
                                        if !a.contains(&app.to_string()) { a.push(app.to_string()); }
                                    }
                                });
                            }>
                                <span class="desktop-name">"Sovereign Workstation"</span>
                                <span class="desktop-desc">"GNOME + Symthaea AI + Mycelix network + dev tools"</span>
                            </button>
                        </div>
                    </GlassPanel>
                </section>
                <section class="install-section">
                    <DesktopPicker selected=selected_desktop />
                </section>

                // Advanced options — collapsed by default
                <section class="install-section">
                    <details class="advanced-section">
                        <summary class="advanced-toggle">"Security Options"</summary>
                        <div style="margin-top: 0.5rem;">
                            <SecurityOptions encrypt=encrypt_disk secure_boot=secure_boot tpm_unlock=tpm_unlock fido2_unlock=fido2_unlock disk_layout=disk_layout filesystem=filesystem />
                        </div>
                    </details>

                    <details class="advanced-section">
                        <summary class="advanced-toggle">"System Options"</summary>
                        <div style="margin-top: 0.5rem;">
                            <GlassPanel title="">
                                <div class="basics-grid">
                                    <div class="field">
                                        <label class="field-label">"Shell"</label>
                                        <select class="field-input" on:change=move |ev| shell.set(event_target_value(&ev))>
                                            <option value="bash" selected=move || shell.get() == "bash">"Bash (default)"</option>
                                            <option value="zsh" selected=move || shell.get() == "zsh">"Zsh (with Oh My Zsh)"</option>
                                            <option value="fish" selected=move || shell.get() == "fish">"Fish (friendly interactive)"</option>
                                        </select>
                                    </div>
                                    <div class="field">
                                        <label class="field-label">"Kernel"</label>
                                        <select class="field-input" on:change=move |ev| kernel.set(event_target_value(&ev))>
                                            <option value="default" selected=move || kernel.get() == "default">"Default (stable)"</option>
                                            <option value="zen" selected=move || kernel.get() == "zen">"Zen (desktop/gaming optimized)"</option>
                                            <option value="lts" selected=move || kernel.get() == "lts">"LTS (long-term support)"</option>
                                            <option value="hardened" selected=move || kernel.get() == "hardened">"Hardened (security-focused)"</option>
                                        </select>
                                    </div>
                                    <div class="field">
                                        <label class="field-label">"Swap Size"</label>
                                        <select class="field-input" on:change=move |ev| swap_gb.set(event_target_value(&ev).parse().unwrap_or(8))>
                                            <option value="0" selected=move || swap_gb.get() == 0>"None"</option>
                                            <option value="2" selected=move || swap_gb.get() == 2>"2 GB"</option>
                                            <option value="4" selected=move || swap_gb.get() == 4>"4 GB"</option>
                                            <option value="8" selected=move || swap_gb.get() == 8>"8 GB (recommended)"</option>
                                            <option value="16" selected=move || swap_gb.get() == 16>"16 GB"</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="security-options" style="margin-top: 0.8rem;">
                                    <label class="security-option">
                                        <input type="checkbox" prop:checked=move || bluetooth.get() on:change=move |_| bluetooth.update(|v| *v = !*v) />
                                        <div><span>"Bluetooth"</span><span class="security-note">"Enable Bluetooth hardware"</span></div>
                                    </label>
                                    <label class="security-option">
                                        <input type="checkbox" prop:checked=move || printing.get() on:change=move |_| printing.update(|v| *v = !*v) />
                                        <div><span>"Printing (CUPS)"</span><span class="security-note">"Enable network and USB printing"</span></div>
                                    </label>
                                    <label class="security-option">
                                        <input type="checkbox" prop:checked=move || is_laptop.get() on:change=move |_| is_laptop.update(|v| *v = !*v) />
                                        <div><span>"Laptop mode"</span><span class="security-note">"Power management, lid switch, battery optimization"</span></div>
                                    </label>
                                    <label class="security-option">
                                        <input type="checkbox" prop:checked=move || home_manager.get() on:change=move |_| home_manager.update(|v| *v = !*v) />
                                        <div>
                                            <span>"Home Manager"</span>
                                            <span class="security-note">"Manage dotfiles, shell config, and per-user packages declaratively"</span>
                                            <details class="help-expander">
                                                <summary class="help-toggle">"What does this mean?"</summary>
                                                <p class="help-text">
                                                    "Home Manager extends NixOS to manage your personal configuration — shell settings, Git config, editor preferences, themes — all in one file. Changes are atomic and reversible, just like the system."
                                                </p>
                                            </details>
                                        </div>
                                    </label>
                                    <label class="security-option">
                                        <input type="checkbox" prop:checked=move || auto_login.get() on:change=move |_| auto_login.update(|v| *v = !*v) />
                                        <div>
                                            <span>"Auto-login"</span>
                                            <span class="security-note">"Skip the login screen and go straight to desktop. Only for single-user machines."</span>
                                        </div>
                                    </label>
                                </div>
                            </GlassPanel>
                        </div>
                    </details>

                    <details class="advanced-section">
                        <summary class="advanced-toggle">"Editions"</summary>
                        <div style="margin-top: 0.5rem;">
                            <GlassPanel title="">
                                <p class="section-desc">"Supercharge your NixOS with consciousness-first tools."</p>
                                <div class="security-options">
                                    <label class="security-option">
                                        <input type="checkbox"
                                            prop:checked=move || symthaea_edition.get()
                                            on:change=move |_| symthaea_edition.update(|v| *v = !*v)
                                        />
                                        <div>
                                            <span>"Symthaea Consciousness Engine"</span>
                                            <span class="security-note">"Natural language system management. Ask your computer to configure itself."</span>
                                            <details class="help-expander">
                                                <summary class="help-toggle">"What's included?"</summary>
                                                <p class="help-text">
                                                    "Symthaea is an AI that understands your NixOS system deeply. "
                                                    "Instead of editing configuration files, you say what you want: "
                                                    "'install Firefox and enable the firewall.' "
                                                    "Symthaea reasons about side effects, explains decisions, and can roll back if something breaks. "
                                                    "Includes Ollama for local AI (no cloud, no data leaves your machine)."
                                                </p>
                                            </details>
                                        </div>
                                    </label>
                                    <label class="security-option">
                                        <input type="checkbox"
                                            prop:checked=move || mycelix_edition.get()
                                            on:change=move |_| mycelix_edition.update(|v| *v = !*v)
                                        />
                                        <div>
                                            <span>"Mycelix Sovereign Network"</span>
                                            <span class="security-note">"Decentralized apps for education, health, governance — no central server."</span>
                                            <details class="help-expander">
                                                <summary class="help-toggle">"What's included?"</summary>
                                                <p class="help-text">
                                                    "Mycelix is a network of community-owned applications built on Holochain. "
                                                    "Your data stays on your device. No accounts, no passwords, no corporation in the middle. "
                                                    "Includes EduNet (learning platform), Health vault, community governance, and file sharing. "
                                                    "Form cooperatives with your neighbors — 2 devices is all you need."
                                                </p>
                                            </details>
                                        </div>
                                    </label>
                                </div>
                            </GlassPanel>
                        </div>
                    </details>
                </section>
                <div class="wizard-buttons">
                    <button class="btn-secondary" on:click=move |_| step.set(1)>{move || i18n::t(lang.get(), "back")}</button>
                    <button class="btn-primary" on:click=move |_| step.set(3)>{move || i18n::t(lang.get(), "next")}</button>
                </div>
            </Show>

            // ── Step 3: Your Apps ──
            <Show when=move || step.get() == 3>
                <section class="install-section">
                    <div class="scan-bar">
                        <button class="btn-secondary btn-sm" on:click=move |_| scan_apps()>
                            {move || i18n::t(lang.get(), "apps_scan")}
                        </button>
                        <span class="scan-status">{move || scan_status.get()}</span>
                    </div>
                    <p class="scan-hint">
                        "Want auto-detection? Download "
                        <a href="https://github.com/Luminous-Dynamics/symthaea/releases" target="_blank">"sovereign-scan"</a>
                        ", run it, and click Scan."
                    </p>
                    <AppSelectionGrid os=os selected=selected_apps app_category=app_category />
                    <div class="field" style="margin-top: 1rem;">
                        <label class="field-label">{move || i18n::t(lang.get(), "apps_custom")}</label>
                        <input type="text" class="field-input" placeholder="neofetch, htop, zellij, bat, eza, ripgrep"
                            prop:value=move || custom_pkgs.get()
                            on:input=move |ev| custom_pkgs.set(event_target_value(&ev))
                        />
                        <span class="field-hint">{move || i18n::t(lang.get(), "apps_custom_hint")}</span>
                    </div>

                    // ── Nixpkgs Search ──
                    <div class="nixpkg-search">
                        <h4>"Search All Packages"</h4>
                        <p class="section-desc">"Search 100,000+ nixpkgs packages. Type a name and press Enter or click Search."</p>
                        <div class="connect-form">
                            <input type="text" class="field-input" style="flex: 2;"
                                placeholder="Search nixpkgs... (e.g., neovim, docker, ffmpeg)"
                                prop:value=move || nix_search_query.get()
                                on:input=move |ev| {
                                    nix_search_query.set(event_target_value(&ev));
                                    do_nix_search();
                                }
                                on:keydown=move |ev: web_sys::KeyboardEvent| {
                                    if ev.key() == "Enter" {
                                        do_nix_search();
                                    }
                                }
                            />
                            <button class="btn-secondary btn-sm" on:click=move |_| do_nix_search()>"Search"</button>
                        </div>

                        {move || {
                            let results = nix_search_results.get();
                            (!results.is_empty()).then(|| view! {
                                <div class="nix-search-results">
                                    <p class="nix-search-count">{format!("{} packages found", results.len())}</p>
                                    {results.iter().map(|(alias, pkg)| {
                                        let pkg_name = pkg.clone();
                                        let display_pkg = pkg.clone();
                                        let display_alias = alias.clone();
                                        view! {
                                            <div class="nix-search-result">
                                                <div class="nix-search-info">
                                                    <span class="nix-search-name">{display_pkg}</span>
                                                    {(display_alias != pkg_name).then(|| view! {
                                                        <span class="nix-search-alias">{format!("({})", display_alias)}</span>
                                                    })}
                                                </div>
                                                <button class="btn-sm btn-secondary" on:click=move |_| {
                                                    custom_pkgs.update(|s| {
                                                        let trimmed = s.trim().to_string();
                                                        // Don't add duplicates
                                                        let existing: Vec<&str> = trimmed.split(',').map(|x| x.trim()).collect();
                                                        if !existing.contains(&pkg_name.as_str()) {
                                                            if !trimmed.is_empty() && !trimmed.ends_with(',') {
                                                                s.push_str(", ");
                                                            }
                                                            s.push_str(&pkg_name);
                                                        }
                                                    });
                                                }>"+ Add"</button>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            })
                        }}
                        <span class="field-hint">"Searching 485+ curated aliases. Connect via SSH relay for full nixpkgs search."</span>
                    </div>

                    {move || (selected_count() > 0).then(|| view! {
                        <p class="selected-summary">{format!("{} {}", selected_count(), i18n::t(lang.get(), "apps_selected"))}</p>
                    })}
                </section>
                <div class="wizard-buttons">
                    <button class="btn-secondary" on:click=move |_| step.set(2)>{move || i18n::t(lang.get(), "back")}</button>
                    <button class="btn-primary" on:click=move |_| step.set(4)>{move || i18n::t(lang.get(), "next")}</button>
                </div>
            </Show>

            // ── Step 4: Review Config ──
            <Show when=move || step.get() == 4>
                <section class="install-section">
                    <GlassPanel title="Your Configuration">
                        // Human-readable summary
                        <div class="config-summary">
                            <h4>"Your NixOS Configuration"</h4>
                            <div class="summary-grid">
                                <div class="summary-item">
                                    <span class="summary-label">"Desktop"</span>
                                    <span class="summary-value">{move || {
                                        let d = selected_desktop.get();
                                        match d.as_str() {
                                            "gnome" => "GNOME".to_string(),
                                            "kde" => "KDE Plasma".to_string(),
                                            "cosmic" => "Cosmic".to_string(),
                                            "hyprland" => "Hyprland".to_string(),
                                            "sway" => "Sway".to_string(),
                                            "xfce" => "XFCE".to_string(),
                                            "none" => "None (Server)".to_string(),
                                            _ => d,
                                        }
                                    }}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"User"</span>
                                    <span class="summary-value">{move || format!("{}@{}", username.get(), hostname.get())}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Encryption"</span>
                                    <span class="summary-value">{move || if encrypt_disk.get() { "LUKS2 enabled" } else { "None" }}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Filesystem"</span>
                                    <span class="summary-value">{move || filesystem.get()}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Shell"</span>
                                    <span class="summary-value">{move || shell.get()}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Kernel"</span>
                                    <span class="summary-value">{move || kernel.get()}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Apps"</span>
                                    <span class="summary-value">{move || format!("{} selected", selected_apps.get().len())}</span>
                                </div>
                                <div class="summary-item">
                                    <span class="summary-label">"Swap"</span>
                                    <span class="summary-value">{move || if swap_gb.get() == 0 { "None".to_string() } else { format!("{} GB", swap_gb.get()) }}</span>
                                </div>
                            </div>
                        </div>

                        <div class="config-buttons">
                            <button class="btn-primary" on:click=move |_| {
                                let hw = config_gen::HardwareProfile {
                                    gpu_vendor: sig_gv.get_value(), gpu_model: sig_gm.get_value(),
                                    cpu_cores: hw_cores, memory_gb: hw_mem, has_wifi: true,
                                    chromebook: false,
                                };
                                let choices = config_gen::UserChoices {
                                    hostname: hostname.get(), username: username.get(), desktop: selected_desktop.get(),
                                    encryption: encrypt_disk.get(), secure_boot: secure_boot.get(),
                                    tpm2_unlock: tpm_unlock.get(), fido2_unlock: fido2_unlock.get(), timezone: timezone.get(),
                                    keyboard: keyboard.get(), locale: locale.get(),
                                    custom_packages: parse_custom_pkgs(),
                                    filesystem: filesystem.get(),
                                    swap_gb: swap_gb.get(), shell: shell.get(), bluetooth: bluetooth.get(),
                                    printing: printing.get(), kernel: kernel.get(), is_laptop: is_laptop.get(),
                                    home_manager: home_manager.get(),
                                    symthaea_edition: symthaea_edition.get(),
                                    mycelix_edition: mycelix_edition.get(),
                                    auto_login: auto_login.get(),
                                    extra_users: parse_extra_users(),
                                };
                                let config = config_gen::generate(&hw, &choices, &selected_apps.get());
                                set_config_warnings.set(config.warnings.clone());
                                set_config_preview.set(Some(config.configuration_nix.clone()));
                                config_nix_signal.set(Some(config.configuration_nix.clone()));
                                flake_nix_signal.set(Some(config.flake_nix.clone()));
                                set_show_next_steps.set(true);
                                download_text("configuration.nix", &config.configuration_nix);
                                download_text("flake.nix", &config.flake_nix);
                            }>
                                {move || i18n::t(lang.get(), "config_generate")}
                            </button>
                            <button class="btn-secondary" on:click=move |_| {
                                let hw = config_gen::HardwareProfile {
                                    gpu_vendor: sig_gv.get_value(), gpu_model: sig_gm.get_value(),
                                    cpu_cores: hw_cores, memory_gb: hw_mem, has_wifi: true,
                                    chromebook: false,
                                };
                                let choices = config_gen::UserChoices {
                                    hostname: hostname.get(), username: username.get(), desktop: selected_desktop.get(),
                                    encryption: encrypt_disk.get(), secure_boot: secure_boot.get(),
                                    tpm2_unlock: tpm_unlock.get(), fido2_unlock: fido2_unlock.get(), timezone: timezone.get(),
                                    keyboard: keyboard.get(), locale: locale.get(),
                                    custom_packages: parse_custom_pkgs(),
                                    filesystem: filesystem.get(),
                                    swap_gb: swap_gb.get(), shell: shell.get(), bluetooth: bluetooth.get(),
                                    printing: printing.get(), kernel: kernel.get(), is_laptop: is_laptop.get(),
                                    home_manager: home_manager.get(),
                                    symthaea_edition: symthaea_edition.get(),
                                    mycelix_edition: mycelix_edition.get(),
                                    auto_login: auto_login.get(),
                                    extra_users: parse_extra_users(),
                                };
                                let config = config_gen::generate(&hw, &choices, &selected_apps.get());
                                set_config_warnings.set(config.warnings.clone());
                                set_config_preview.set(Some(config.configuration_nix));
                            }>
                                {move || i18n::t(lang.get(), "config_preview")}
                            </button>
                        </div>

                        // Warnings (#9)
                        {move || {
                            let w = config_warnings.get();
                            (!w.is_empty()).then(|| view! {
                                <div class="config-warnings">
                                    {w.iter().map(|msg| view! {
                                        <p class="warning-msg">{msg.clone()}</p>
                                    }).collect::<Vec<_>>()}
                                </div>
                            })
                        }}

                        // Package validation warnings from relay
                        {move || {
                            let warnings = package_warnings.get();
                            (!warnings.is_empty()).then(|| view! {
                                <div class="config-warnings">
                                    <p class="warning-msg"><strong>"Package Validation"</strong></p>
                                    <p class="warning-msg">"Some packages may not be available in the target's nixpkgs:"</p>
                                    {warnings.iter().map(|w| view! {
                                        <p class="warning-msg">{w.clone()}</p>
                                    }).collect::<Vec<_>>()}
                                </div>
                            })
                        }}

                        // Raw Nix config — collapsed by default
                        {move || config_preview.get().map(|nix| view! {
                            <details class="advanced-section">
                                <summary class="advanced-toggle">"View raw configuration.nix"</summary>
                                <pre class="config-code">{nix}</pre>
                            </details>
                        })}
                    </GlassPanel>
                </section>

                // ── Configuration Profiles (Save / Load) ──
                <section class="install-section">
                    <GlassPanel title="Configuration Profiles">
                        <p class="section-desc">"Save this config as a profile to reuse on other machines, or load a previously saved one."</p>
                        <div class="connect-form">
                            <div class="field">
                                <input type="text" class="field-input" placeholder="Profile name (e.g., Dev Workstation)"
                                    prop:value=move || profile_name.get()
                                    on:input=move |ev| profile_name.set(event_target_value(&ev))
                                />
                            </div>
                            <button class="btn-secondary btn-sm" on:click=move |_| {
                                let name = profile_name.get();
                                if name.is_empty() { return; }
                                if let Some(config) = config_nix_signal.get() {
                                    profiles.update(|p| p.push((name, config)));
                                    save_to_storage("si_profiles", &serde_json::to_string(&profiles.get()).unwrap_or_default());
                                    profile_name.set(String::new());
                                }
                            }>{move || i18n::t(lang.get(), "profiles_save")}</button>
                        </div>
                        {move || {
                            let p = profiles.get();
                            (!p.is_empty()).then(|| {
                                let items = p.iter().map(|(name, config)| {
                                    let config_clone = config.clone();
                                    let name_clone = name.clone();
                                    let del_name = name.clone();
                                    view! {
                                        <div class="profile-item">
                                            <span>{name_clone}</span>
                                            <div style="display:flex;gap:0.4rem;">
                                                <button class="btn-secondary btn-sm" on:click=move |_| {
                                                    config_nix_signal.set(Some(config_clone.clone()));
                                                    set_config_preview.set(Some(config_clone.clone()));
                                                }>{move || i18n::t(lang.get(), "profiles_load")}</button>
                                                <button class="btn-secondary btn-sm" style="color:var(--autumn-rust);" on:click=move |_| {
                                                    let del = del_name.clone();
                                                    profiles.update(|p| p.retain(|(n, _)| n != &del));
                                                    save_to_storage("si_profiles", &serde_json::to_string(&profiles.get()).unwrap_or_default());
                                                }>{move || i18n::t(lang.get(), "profiles_delete")}</button>
                                            </div>
                                        </div>
                                    }
                                }).collect::<Vec<_>>();
                                view! {
                                    <div class="profile-list">
                                        <h5 style="font-size:0.9rem;color:var(--fg-dim);margin-bottom:0.4rem;">{move || i18n::t(lang.get(), "profiles_saved")}</h5>
                                        {items}
                                    </div>
                                }
                            })
                        }}
                    </GlassPanel>
                </section>

                {move || {
                    let errs = validation_errors.get();
                    (!errs.is_empty()).then(|| view! {
                        <div class="validation-errors" role="alert" aria-live="assertive">
                            {errs.iter().map(|e| view! { <p class="error-msg">{e.clone()}</p> }).collect::<Vec<_>>()}
                        </div>
                    })
                }}
                <div class="wizard-buttons">
                    <button class="btn-secondary" on:click=move |_| step.set(3)>{move || i18n::t(lang.get(), "back")}</button>
                    <button class="btn-primary" on:click=move |_| {
                        do_generate();
                        if config_nix_signal.get().is_some() {
                            validation_errors.set(Vec::new());
                            step.set(5);
                        } else {
                            validation_errors.set(vec!["Config generation failed. Please go back and check your settings.".into()]);
                        }
                    }>{move || i18n::t(lang.get(), "next")}</button>
                </div>
            </Show>

            // ── Step 5: Install ──
            <Show when=move || step.get() == 5>
                <section class="install-section">
                    <RemoteInstallPanel
                        config_nix=config_nix_signal.into()
                        flake_nix=flake_nix_signal.into()
                        hostname=hostname
                        desktop=selected_desktop
                        gpu_driver=gpu_driver
                        timezone=timezone
                        keyboard=keyboard
                        encrypt=encrypt_disk
                        secure_boot=secure_boot
                        tpm_unlock=tpm_unlock
                        fido2_unlock=fido2_unlock
                        disk_layout=disk_layout
                        filesystem=filesystem
                        user_password=user_password
                    />
                </section>

                // Manual path — shown after config download
                {move || show_next_steps.get().then(|| view! {
                    <section class="install-section">
                        <details class="collapsible">
                            <summary class="collapsible-title">"Manual Install (if not using automated path)"</summary>
                            <div class="collapsible-body">
                                <NextSteps encrypt=encrypt_disk />
                            </div>
                        </details>
                    </section>
                })}

                // Note about hardware-configuration.nix (#7)
                {move || show_next_steps.get().then(|| view! {
                    <section class="install-section">
                        <GlassPanel title="About hardware-configuration.nix">
                            <p class="section-desc">
                                "Your downloaded config imports "<code>"hardware-configuration.nix"</code>
                                " — this file is generated ON the target machine by running "
                                <code>"nixos-generate-config --root /mnt"</code>
                                ". It detects your exact disk layout, filesystems, kernel modules, "
                                "and firmware. It cannot be generated from the browser."
                            </p>
                        </GlassPanel>
                    </section>
                })}

                <div class="wizard-buttons">
                    <button class="btn-secondary" on:click=move |_| step.set(4)>{move || i18n::t(lang.get(), "back")}</button>
                    <span></span>
                </div>
            </Show>

            </Show> // end Custom mode

            // ══════════════════════════════════════════
            // Conversational Mode — Talk with Symthaea
            // ══════════════════════════════════════════
            <Show when=move || mode.get() == "converse">
                <ConverseMode
                    hardware_cores=hw_cores
                    hardware_mem=hw_mem
                    gpu_vendor=gpu_vendor_str.clone()
                    gpu_model=gpu_model.clone()
                    config_nix_signal=config_nix_signal
                    flake_nix_signal=flake_nix_signal
                    hostname=hostname
                    selected_desktop=selected_desktop
                    gpu_driver=gpu_driver
                    timezone=timezone
                    keyboard=keyboard
                    encrypt_disk=encrypt_disk
                    secure_boot=secure_boot
                    tpm_unlock=tpm_unlock
                    fido2_unlock=fido2_unlock
                    disk_layout=disk_layout
                    filesystem=filesystem
                    user_password=user_password
                    mode=mode
                />
            </Show>

            <footer class="install-footer">
                <p>
                    <a href="https://github.com/Luminous-Dynamics/symthaea">{move || i18n::t(lang.get(), "footer_source")}</a>
                    " \u{00b7} "
                    <a href="https://luminousdynamics.org">"Luminous Dynamics"</a>
                    " \u{00b7} "
                    {move || i18n::t(lang.get(), "footer_no_tracking")}
                    " \u{00b7} "
                    <a href="#" class="start-over-link" on:click=move |ev| {
                        ev.prevent_default();
                        if let Some(window) = web_sys::window() {
                            if let Ok(Some(storage)) = window.local_storage() {
                                // Clear all si_* keys
                                let keys_to_clear = [
                                    "si_mode", "si_step", "si_lang", "si_hostname", "si_username", "si_timezone",
                                    "si_keyboard", "si_desktop", "si_encrypt", "si_secureboot",
                                    "si_tpm", "si_fido2", "si_disk_layout", "si_filesystem", "si_locale", "si_apps",
                                    "si_custom_pkgs", "si_scan_token",
                                    "si_swap", "si_shell", "si_bluetooth", "si_printing",
                                    "si_kernel", "si_laptop", "si_homemanager",
                                    "si_symthaea", "si_mycelix",
                                    "si_autologin", "si_extra_users",
                                ];
                                for key in &keys_to_clear {
                                    let _ = storage.remove_item(key);
                                }
                            }
                            let _ = window.location().reload();
                        }
                    }>{move || i18n::t(lang.get(), "start_over")}</a>
                </p>
            </footer>
        </div>
    }
}
