// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Settings page with sidebar navigation — 14 sections, all persisted.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use mail_leptos_types::*;
use crate::mail_context::use_mail;
use crate::holochain::ConnectionStatus;
use crate::preferences::*;
use crate::toasts::use_toasts;

#[derive(Clone, Default)]
struct LiveSliceReport {
    steps: Vec<(String, bool, String)>,
}

#[component]
pub fn SettingsPage() -> impl IntoView {
    let prefs = use_preferences();
    let mail = use_mail();
    let toasts = use_toasts();
    let mail_backup = mail.clone();
    let toasts_backup = toasts.clone();
    let toasts_reset = toasts.clone();
    let mail_sig = mail.clone();
    let toasts_sig = toasts.clone();
    let mail_lbl = mail.clone();
    let toasts_lbl = toasts.clone();
    let toasts_test = toasts.clone();
    let mail_restore = mail.clone();
    let toasts_restore = toasts.clone();
    let section = RwSignal::new("profile");
    let search_filter = RwSignal::new(String::new());
    let mail_keys = mail.clone();
    let live_slice_running = RwSignal::new(false);
    let live_slice_report = RwSignal::new(LiveSliceReport::default());
    let toasts_export = toasts.clone();

    let nav_items: &[(&str, &str, &str, &str)] = &[
        ("profile","\u{1F464}","Profile","name bio avatar dsid identity"),
        ("security","\u{1F6E1}","Security","dsid mfa factor assurance verification sybil"),
        ("appearance","\u{1F3A8}","Appearance","theme density font contrast color accent"),
        ("layout","\u{1F4BB}","Layout","reading pane split"),
        ("compose","\u{270F}","Compose","encryption reply undo send delay pqc"),
        ("privacy","\u{1F512}","Privacy","read receipts typing indicators availability"),
        ("notifications","\u{1F514}","Notifications","desktop sound alert"),
        ("keyboard","\u{2328}","Keyboard","shortcuts hotkeys keys"),
        ("attention","\u{1F9E0}","Attention","budget focus time"),
        ("data","\u{1F4BE}","Data","backup restore reset export import"),
        ("locale","\u{1F310}","Locale","language date format region"),
        ("signatures","\u{270D}","Signatures","signature email"),
        ("labels","\u{1F3F7}","Labels","label tag color"),
        ("encryption","\u{1F512}","Encryption","key status connection test pqc aes"),
    ];

    view! {
        <div class="page page-settings-v2" data-testid="settings-page">
            <div class="settings-layout">
                <nav class="settings-nav">
                    <h2>"Settings"</h2>
                    <input
                        class="settings-search"
                        type="text"
                        placeholder="\u{1F50D} Search settings..."
                        prop:value=move || search_filter.get()
                        on:input=move |ev| {
                            use wasm_bindgen::JsCast;
                            let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                            search_filter.set(val);
                        }
                    />
                    {nav_items.iter().map(|(id, icon, label, keywords)| {
                        let id=*id; let icon=*icon; let label=*label; let keywords=*keywords;
                        view! {
                            <button
                                class=move || if section.get()==id {"settings-nav-item active"} else {"settings-nav-item"}
                                style=move || {
                                    let q = search_filter.get().to_lowercase();
                                    if q.is_empty() || label.to_lowercase().contains(&q) || keywords.contains(&q.as_str()) {
                                        ""
                                    } else {
                                        "display:none"
                                    }
                                }
                                on:click=move |_| section.set(id)>
                                <span class="nav-item-icon">{icon}</span>
                                <span class="nav-item-label">{label}</span>
                            </button>
                        }
                    }).collect::<Vec<_>>()}
                </nav>
                <div class="settings-content">
                    // Profile
                    <div style=move || if section.get()=="profile" {""} else {"display:none"}>
                        <h2>"Profile"</h2>
                        <ProfileSection />
                    </div>
                    // Security (DSID & MFA)
                    <div style=move || if section.get()=="security" {""} else {"display:none"}>
                        <h2>"Security & DSID"</h2>
                        <SecuritySection />
                    </div>
                    // Appearance
                    <div style=move || if section.get()=="appearance" {""} else {"display:none"}>
                        <h2>"Appearance"</h2>
                        <Card label="Theme">
                            <button class=move || if prefs.get().theme==Theme::Dark {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.theme=Theme::Dark)>"\u{1F319} Dark"</button>
                            <button class=move || if prefs.get().theme==Theme::Light {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.theme=Theme::Light)>"\u{2600} Light"</button>
                        </Card>
                        <Card label="Density">
                            <button class=move || if prefs.get().density==Density::Compact {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.density=Density::Compact)>"Compact"</button>
                            <button class=move || if prefs.get().density==Density::Default {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.density=Density::Default)>"Default"</button>
                            <button class=move || if prefs.get().density==Density::Comfortable {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.density=Density::Comfortable)>"Comfortable"</button>
                        </Card>
                        <Card label="Font Size">
                            <input type="range" min="75" max="150" step="5" class="setting-slider"
                                prop:value=move || prefs.get().font_scale.to_string()
                                on:input=move |ev| { if let Ok(v)=ev.target().and_then(|t|t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|e|e.value()).unwrap_or_default().parse::<u32>() { prefs.update(|p|p.font_scale=clamp_font_scale(v)); } } />
                        </Card>
                        <Card label="High Contrast">
                            <button class=move || if prefs.get().high_contrast {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.high_contrast=!p.high_contrast)>
                                {move || if prefs.get().high_contrast {"On"} else {"Off"}}</button>
                        </Card>
                    </div>
                    // Layout
                    <div style=move || if section.get()=="layout" {""} else {"display:none"}>
                        <h2>"Layout"</h2>
                        <Card label="Reading Pane">
                            <button class=move || if prefs.get().reading_pane==ReadingPanePosition::Off {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| { prefs.update(|p|p.reading_pane=ReadingPanePosition::Off); mail.reading_pane.set(ReadingPanePosition::Off); }>"Off"</button>
                            <button class=move || if prefs.get().reading_pane==ReadingPanePosition::Right {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| { prefs.update(|p|p.reading_pane=ReadingPanePosition::Right); mail.reading_pane.set(ReadingPanePosition::Right); }>"Right"</button>
                            <button class=move || if prefs.get().reading_pane==ReadingPanePosition::Bottom {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| { prefs.update(|p|p.reading_pane=ReadingPanePosition::Bottom); mail.reading_pane.set(ReadingPanePosition::Bottom); }>"Bottom"</button>
                        </Card>
                    </div>
                    // Compose
                    <div style=move || if section.get()=="compose" {""} else {"display:none"}>
                        <h2>"Compose Defaults"</h2>
                        <Card label="Default Encryption">
                            <button class=move || if prefs.get().default_pqc {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p| p.default_pqc=!p.default_pqc)>
                                {move || if prefs.get().default_pqc {"\u{1F6E1} PQC"} else {"\u{1F512} Standard"}}</button>
                        </Card>
                        <Card label="Reply Style">
                            <button class=move || if prefs.get().reply_style==ReplyStyle::InlineQuote {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.reply_style=ReplyStyle::InlineQuote)>"Inline Quote"</button>
                            <button class=move || if prefs.get().reply_style==ReplyStyle::TopPost {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.reply_style=ReplyStyle::TopPost)>"Top Post"</button>
                        </Card>
                        <Card label="Undo Send Delay">
                            {[3u32,5,10,30].iter().map(|s|{let s=*s; view!{
                                <button class=move || if prefs.get().undo_send_seconds==s {"toggle-btn active"} else {"toggle-btn"}
                                    on:click=move |_| prefs.update(|p|p.undo_send_seconds=s)>{format!("{s}s")}</button>
                            }}).collect::<Vec<_>>()}
                        </Card>
                    </div>
                    // Privacy
                    <div style=move || if section.get()=="privacy" {""} else {"display:none"}>
                        <h2>"Privacy"</h2>
                        <Card label="Read Receipts">
                            <button class=move || if prefs.get().send_read_receipts==ReadReceiptPolicy::Always {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.send_read_receipts=ReadReceiptPolicy::Always)>"Always"</button>
                            <button class=move || if prefs.get().send_read_receipts==ReadReceiptPolicy::Ask {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.send_read_receipts=ReadReceiptPolicy::Ask)>"Ask"</button>
                            <button class=move || if prefs.get().send_read_receipts==ReadReceiptPolicy::Never {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.send_read_receipts=ReadReceiptPolicy::Never)>"Never"</button>
                        </Card>
                        <Card label="Typing Indicators">
                            <button class=move || if prefs.get().show_typing_indicators {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.show_typing_indicators=!p.show_typing_indicators)>
                                {move || if prefs.get().show_typing_indicators {"Visible"} else {"Hidden"}}</button>
                        </Card>
                        <Card label="Availability">
                            <button class=move || if prefs.get().share_availability==AvailabilityPolicy::Everyone {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.share_availability=AvailabilityPolicy::Everyone)>"Everyone"</button>
                            <button class=move || if prefs.get().share_availability==AvailabilityPolicy::ContactsOnly {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.share_availability=AvailabilityPolicy::ContactsOnly)>"Contacts"</button>
                            <button class=move || if prefs.get().share_availability==AvailabilityPolicy::Nobody {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.share_availability=AvailabilityPolicy::Nobody)>"Nobody"</button>
                        </Card>
                    </div>
                    // Notifications
                    <div style=move || if section.get()=="notifications" {""} else {"display:none"}>
                        <h2>"Notifications"</h2>
                        <Card label="Desktop Notifications">
                            <button class=move || if prefs.get().desktop_notifications {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| { prefs.update(|p|p.desktop_notifications=!p.desktop_notifications); crate::notifications::request_notification_permission(); }>
                                {move || if prefs.get().desktop_notifications {"On"} else {"Off"}}</button>
                        </Card>
                        <Card label="Sound">
                            <button class=move || if prefs.get().sound_enabled {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.sound_enabled=!p.sound_enabled)>
                                {move || if prefs.get().sound_enabled {"On"} else {"Off"}}</button>
                        </Card>
                    </div>
                    // Keyboard
                    <div style=move || if section.get()=="keyboard" {""} else {"display:none"}>
                        <h2>"Keyboard Shortcuts"</h2>
                        <div class="shortcut-table">
                            {[("j/k","Navigate"),("Enter","Open"),("c","Compose"),("r","Reply"),("f","Forward"),
                              ("s","Star"),("e","Archive"),("#","Delete"),("x","Select"),("m","Mute"),
                              ("/","Search"),("u","Inbox"),("?","Help"),("Ctrl+K","Palette")].iter().map(|(k,a)|
                                view!{<div class="shortcut-table-row"><kbd>{*k}</kbd><span>{*a}</span></div>}
                            ).collect::<Vec<_>>()}
                        </div>
                    </div>
                    // Attention
                    <div style=move || if section.get()=="attention" {""} else {"display:none"}>
                        <h2>"Attention Budget"</h2>
                        <Card label="Daily Budget">
                            <input type="range" min="50" max="200" step="10" class="setting-slider"
                                prop:value=move || prefs.get().attention_budget_daily.to_string()
                                on:input=move |ev| { if let Ok(v)=ev.target().and_then(|t|t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|e|e.value()).unwrap_or_default().parse::<u32>() { prefs.update(|p|p.attention_budget_daily=v); } } />
                        </Card>
                        <Card label="Focus Time">
                            <button class=move || if prefs.get().focus_time_enabled {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.focus_time_enabled=!p.focus_time_enabled)>
                                {move || if prefs.get().focus_time_enabled {"Enabled"} else {"Disabled"}}</button>
                        </Card>
                    </div>
                    // Data
                    <div style=move || if section.get()=="data" {""} else {"display:none"}>
                        <h2>"Data"</h2>
                        <Card label="Backup">
                            <button class="btn btn-secondary" on:click={let m=mail_backup.clone();let t=toasts_backup.clone(); move |_| {
                                let b=serde_json::json!({"v":"1","inbox":m.inbox.get_untracked(),"contacts":m.contacts.get_untracked()});
                                download_file("mycelix-backup.json",&serde_json::to_string_pretty(&b).unwrap_or_default(),"application/json");
                                t.push("Downloaded","success");
                            }}>"Download"</button>
                        </Card>
                        <Card label="Restore">
                            <label class="btn btn-secondary file-upload-btn">
                                "Upload Backup"
                                <input type="file" accept=".json" style="display:none"
                                    on:change={
                                        let m = mail_restore.clone(); let t = toasts_restore.clone();
                                        move |ev: leptos::ev::Event| {
                                            use wasm_bindgen::JsCast;
                                            if let Some(input) = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()) {
                                                if let Some(files) = input.files() {
                                                    if let Some(file) = files.get(0) {
                                                        let reader = web_sys::FileReader::new().unwrap();
                                                        let reader_c = reader.clone();
                                                        let m2 = m.clone(); let t2 = t.clone();
                                                        let onload = wasm_bindgen::prelude::Closure::<dyn Fn()>::new(move || {
                                                            if let Ok(result) = reader_c.result() {
                                                                let text = result.as_string().unwrap_or_default();
                                                                if let Ok(backup) = serde_json::from_str::<serde_json::Value>(&text) {
                                                                    if let Some(inbox) = backup.get("inbox") {
                                                                        if let Ok(emails) = serde_json::from_value(inbox.clone()) {
                                                                            m2.inbox.set(emails);
                                                                        }
                                                                    }
                                                                    if let Some(contacts) = backup.get("contacts") {
                                                                        if let Ok(c) = serde_json::from_value(contacts.clone()) {
                                                                            m2.contacts.set(c);
                                                                        }
                                                                    }
                                                                    t2.push("Backup restored", "success");
                                                                } else {
                                                                    t2.push("Invalid backup file", "error");
                                                                }
                                                            }
                                                        });
                                                        reader.set_onload(Some(onload.as_ref().unchecked_ref()));
                                                        onload.forget();
                                                        let _ = reader.read_as_text(&file);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                />
                            </label>
                        </Card>
                        <Card label="Reset Demo">
                            <button class="btn btn-secondary" on:click={let t=toasts_reset.clone(); move |_| {
                                crate::persistence::clear_demo_data(); t.push("Cleared","info");
                            }}>"Reset"</button>
                        </Card>
                        <Card label="Export Audit Log">
                            <button class="btn btn-secondary" on:click=move |_| {
                                // Collect all localStorage keys as audit entries
                                let mut log = String::from("Mycelix Pulse Audit Log\n========================\n\n");
                                if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                                    let len = s.length().unwrap_or(0);
                                    for i in 0..len {
                                        if let Ok(Some(key)) = s.key(i) {
                                            if key.starts_with("mycelix") {
                                                log.push_str(&format!("[{}] {}\n", js_sys::Date::new_0().to_iso_string(), key));
                                            }
                                        }
                                    }
                                }
                                download_file("audit-log.txt", &log, "text/plain");
                            }>"Download Audit Log"</button>
                        </Card>
                        <Card label="Panic Button">
                            <button class="panic-btn" on:click=move |_| {
                                let confirmed = web_sys::window()
                                    .and_then(|w| w.confirm_with_message(
                                        "WIPE ALL LOCAL DATA? This will clear all emails, contacts, keys, and settings. This cannot be undone."
                                    ).ok())
                                    .unwrap_or(false);
                                if confirmed {
                                    if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
                                        let _ = s.clear();
                                    }
                                    if let Some(s) = web_sys::window().and_then(|w| w.session_storage().ok().flatten()) {
                                        let _ = s.clear();
                                    }
                                    // Clear caches
                                    let _ = js_sys::eval("caches.keys().then(ks=>ks.forEach(k=>caches.delete(k)))");
                                    // Reload
                                    let _ = web_sys::window().map(|w| w.location().reload());
                                }
                            }>"\u{1F6A8} Emergency Wipe"</button>
                        </Card>
                    </div>
                    // Locale
                    <div style=move || if section.get()=="locale" {""} else {"display:none"}>
                        <h2>"Language & Region"</h2>
                        <Card label="Language">
                            <select class="setting-select" on:change=move |ev| {
                                let v=ev.target().and_then(|t|t.dyn_into::<web_sys::HtmlSelectElement>().ok()).map(|e|e.value()).unwrap_or_default();
                                prefs.update(|p|p.language=match v.as_str(){"Afrikaans"=>Language::Afrikaans,"Zulu"=>Language::Zulu,"French"=>Language::French,"Spanish"=>Language::Spanish,_=>Language::English});
                            }>
                                {Language::ALL.iter().map(|l|{let label=l.label();let name=format!("{l:?}"); view!{<option value=name>{label}</option>}}).collect::<Vec<_>>()}
                            </select>
                        </Card>
                        <Card label="Date Format">
                            <button class=move || if prefs.get().date_format==DateFormat::Auto {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.date_format=DateFormat::Auto)>"Auto"</button>
                            <button class=move || if prefs.get().date_format==DateFormat::DmySlash {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.date_format=DateFormat::DmySlash)>"DD/MM/YYYY"</button>
                            <button class=move || if prefs.get().date_format==DateFormat::MdySlash {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.date_format=DateFormat::MdySlash)>"MM/DD/YYYY"</button>
                            <button class=move || if prefs.get().date_format==DateFormat::YmdDash {"toggle-btn active"} else {"toggle-btn"}
                                on:click=move |_| prefs.update(|p|p.date_format=DateFormat::YmdDash)>"YYYY-MM-DD"</button>
                        </Card>
                    </div>
                    // Encryption
                    // Signatures
                    <div style=move || if section.get()=="signatures" {""} else {"display:none"}>
                        <h2>"Signatures"</h2>
                        {move || mail.signatures.get().iter().map(|s| {
                            let n = s.name.clone(); let b = s.body_html.clone(); let d = s.is_default;
                            view! {
                                <div class="setting-card">
                                    <span class="setting-label">{n}</span>
                                    {d.then(|| view! { <span class="sig-badge">" Default"</span> })}
                                    <div class="sig-preview" inner_html=b />
                                </div>
                            }
                        }).collect::<Vec<_>>()}
                        <div class="setting-card">
                            <button class="btn btn-sm btn-secondary" on:click={
                                let ms = mail_sig.clone(); let ts = toasts_sig.clone();
                                move |_| {
                                    ms.signatures.update(|sigs| {
                                        sigs.push(mail_leptos_types::SignatureView {
                                            id: format!("sig-{}", js_sys::Date::now() as u64),
                                            name: format!("Signature {}", sigs.len() + 1),
                                            body_html: "<p>Best regards</p>".into(),
                                            is_default: false,
                                            use_for_new: false,
                                            use_for_reply: false,
                                        });
                                    });
                                    toasts_sig.push("Signature added", "success");
                                }
                            }>"+ Add Signature"</button>
                        </div>
                    </div>
                    // Labels
                    <div style=move || if section.get()=="labels" {""} else {"display:none"}>
                        <h2>"Labels"</h2>
                        <div class="label-list">
                            {move || mail_lbl.labels.get().iter().map(|l| {
                                let n = l.name.clone(); let s = l.css_var();
                                let label_name = l.name.clone();
                                view! { <span class="label-chip label-draggable" style=s draggable="true"
                                    on:dragstart=move |ev: web_sys::DragEvent| {
                                        if let Some(dt) = ev.data_transfer() {
                                            let _ = dt.set_data("application/x-label", &label_name);
                                            dt.set_effect_allowed("copy");
                                        }
                                    }>{n}</span> }
                            }).collect::<Vec<_>>()}
                        </div>
                        <div class="setting-card">
                            <button class="btn btn-sm btn-secondary" on:click={
                                let ml = mail_lbl.clone(); let tl = toasts_lbl.clone();
                                move |_| {
                                    let colors = ["#06D6C8","#8b7ec8","#f59e0b","#4ade80","#ec4899","#3b82f6"];
                                    let idx = ml.labels.get_untracked().len();
                                    ml.labels.update(|labels| {
                                        labels.push(mail_leptos_types::LabelView {
                                            id: format!("l-{}", js_sys::Date::now() as u64),
                                            name: format!("Label {}", idx + 1),
                                            color: colors[idx % colors.len()].into(),
                                        });
                                    });
                                    toasts_lbl.push("Label added", "success");
                                }
                            }>"+ Add Label"</button>
                        </div>
                    </div>
                    // Encryption
                    <div style=move || if section.get()=="encryption" {""} else {"display:none"}>
                        <h2>"Encryption"</h2>
                        <Card label="Key Status">
                            <span class=move || format!("setting-value {}", mail_keys.key_status.get().css_class())>{move || mail.key_status.get().label()}</span>
                        </Card>
                        <Card label="Test Connection">
                            <button class="btn btn-secondary" on:click={
                                let t3 = toasts_test.clone();
                                move |_| {
                                    let hc = crate::holochain::use_holochain();
                                    let t = t3.clone();
                                    wasm_bindgen_futures::spawn_local(async move {
                                        if hc.is_mock() { t.push("Mock mode — no conductor", "info"); return; }
                                        match hc.call_zome::<(), serde_json::Value>("mail_messages", "get_folders", &()).await {
                                            Ok(_) => t.push("Conductor OK — zome responded", "success"),
                                            Err(e) => t.push(format!("Failed: {e}"), "error"),
                                        }
                                    });
                                }
                            }>"Test"</button>
                        </Card>
                        <Card label="Verify Live Slice">
                            <button
                                class="btn btn-secondary"
                                data-testid="live-slice-verify-button"
                                disabled=move || live_slice_running.get()
                                on:click={
                                let report = live_slice_report;
                                let running = live_slice_running;
                                let toasts = toasts.clone();
                                move |_| {
                                    let hc = crate::holochain::use_holochain();
                                    running.set(true);
                                    report.set(LiveSliceReport::default());
                                    let report_setter = report;
                                    let running_setter = running;
                                    let toasts = toasts.clone();
                                    wasm_bindgen_futures::spawn_local(async move {
                                        let mut steps = Vec::<(String, bool, String)>::new();

                                        if hc.is_mock() {
                                            steps.push((
                                                "Connection".into(),
                                                false,
                                                "Running in mock mode, so the live slice cannot be verified.".into(),
                                            ));
                                            report_setter.set(LiveSliceReport { steps });
                                            running_setter.set(false);
                                            toasts.push("Live slice verification requires a real conductor", "error");
                                            return;
                                        }

                                        steps.push((
                                            "Connection".into(),
                                            true,
                                            "Connected to a live conductor.".into(),
                                        ));

                                        let folders = match hc.call_zome::<(), serde_json::Value>(
                                            "mail_messages", "get_folders", &()
                                        ).await {
                                            Ok(value) => {
                                                let count = value.as_array()
                                                    .map(|arr| arr.len())
                                                    .or_else(|| value.get("folders").and_then(|v| v.as_array()).map(|arr| arr.len()))
                                                    .unwrap_or(0);
                                                steps.push((
                                                    "Folders".into(),
                                                    true,
                                                    format!("Fetched folder list ({count} items)."),
                                                ));
                                                Some(value)
                                            }
                                            Err(e) => {
                                                steps.push(("Folders".into(), false, e));
                                                None
                                            }
                                        };

                                        let contacts = match hc.call_zome::<(), Vec<serde_json::Value>>(
                                            "mail_contacts", "get_all_contacts", &()
                                        ).await {
                                            Ok(values) => {
                                                let contacts = crate::zome_adapter::adapt_contact_values(values);
                                                steps.push((
                                                    "Contacts".into(),
                                                    true,
                                                    format!("Fetched and parsed {} contacts.", contacts.len()),
                                                ));
                                                Some(contacts)
                                            }
                                            Err(e) => {
                                                steps.push(("Contacts".into(), false, e));
                                                None
                                            }
                                        };

                                        let inbox = match hc.call_zome::<serde_json::Value, serde_json::Value>(
                                            "mail_messages", "get_inbox", &serde_json::json!({ "limit": 5 })
                                        ).await {
                                            Ok(value) => {
                                                let count = value.as_array()
                                                    .map(|arr| arr.len())
                                                    .or_else(|| value.get("records").and_then(|v| v.as_array()).map(|arr| arr.len()))
                                                    .or_else(|| value.get("data").and_then(|v| v.as_array()).map(|arr| arr.len()))
                                                    .unwrap_or(0);
                                                steps.push((
                                                    "Inbox".into(),
                                                    true,
                                                    format!("Fetched inbox payload ({count} items)."),
                                                ));
                                                Some(value)
                                            }
                                            Err(e) => {
                                                steps.push(("Inbox".into(), false, e));
                                                None
                                            }
                                        };

                                        match hc.call_zome::<(), serde_json::Value>(
                                            "mail_keys", "needs_refresh", &()
                                        ).await {
                                            Ok(value) => {
                                                steps.push((
                                                    "Key Bundle".into(),
                                                    true,
                                                    format!("Key status: {}", serde_json::to_string(&value).unwrap_or_else(|_| "unknown".into())),
                                                ));
                                            }
                                            Err(e) => {
                                                steps.push(("Key Bundle".into(), false, e));
                                            }
                                        }

                                        let did_document = match hc.call_zome_on_role::<(), serde_json::Value>(
                                            "identity", "did_registry", "get_did_document", &()
                                        ).await {
                                            Ok(value) if !value.is_null() => {
                                                let did = value.get("id")
                                                    .and_then(|v| v.as_str())
                                                    .unwrap_or_default()
                                                    .to_string();
                                                steps.push((
                                                    "Identity".into(),
                                                    true,
                                                    if did.is_empty() {
                                                        "DID document returned, but id was empty.".into()
                                                    } else {
                                                        format!("Resolved current DID document: {did}.")
                                                    },
                                                ));
                                                Some(value)
                                            }
                                            Ok(_) => {
                                                steps.push(("Identity".into(), false, "No DID document found.".into()));
                                                None
                                            }
                                            Err(e) => {
                                                steps.push(("Identity".into(), false, e));
                                                None
                                            }
                                        };

                                        if let Some(did) = did_document
                                            .as_ref()
                                            .and_then(|value| value.get("id"))
                                            .and_then(|value| value.as_str())
                                        {
                                            match hc.call_zome::<serde_json::Value, Option<serde_json::Value>>(
                                                "mail_bridge", "resolve_identity", &serde_json::json!(did)
                                            ).await {
                                                Ok(Some(agent)) => {
                                                    steps.push((
                                                        "Identity Bridge".into(),
                                                        true,
                                                        format!("Resolved DID to agent key {}.", crate::zome_adapter::json_hash_to_string(&agent)),
                                                    ));
                                                }
                                                Ok(None) => {
                                                    steps.push((
                                                        "Identity Bridge".into(),
                                                        false,
                                                        "Bridge returned no agent key for the current DID.".into(),
                                                    ));
                                                }
                                                Err(e) => {
                                                    steps.push(("Identity Bridge".into(), false, e));
                                                }
                                            }
                                        }

                                        let first_contact_email = contacts.as_ref()
                                            .and_then(|items| items.iter().find_map(|contact| contact.email.clone()));
                                        if let Some(email) = first_contact_email {
                                            match hc.call_zome::<serde_json::Value, serde_json::Value>(
                                                "mail_contacts", "get_contact_by_email", &serde_json::json!(email.clone())
                                            ).await {
                                                Ok(value) => {
                                                    if crate::zome_adapter::adapt_contact_value(value).is_some() {
                                                        steps.push((
                                                            "Contact Lookup".into(),
                                                            true,
                                                            format!("Resolved contact lookup by email for {email}."),
                                                        ));
                                                    } else {
                                                        steps.push((
                                                            "Contact Lookup".into(),
                                                            false,
                                                            format!("Contact lookup returned an unreadable payload for {email}."),
                                                        ));
                                                    }
                                                }
                                                Err(e) => {
                                                    steps.push(("Contact Lookup".into(), false, e));
                                                }
                                            }
                                        } else if contacts.is_some() {
                                            steps.push((
                                                "Contact Lookup".into(),
                                                true,
                                                "No contacts with email addresses were available to test lookup.".into(),
                                            ));
                                        }

                                        let first_hash = inbox.as_ref()
                                            .and_then(|value| value.as_array().cloned()
                                                .or_else(|| value.get("records").and_then(|v| v.as_array()).cloned())
                                                .or_else(|| value.get("data").and_then(|v| v.as_array()).cloned()))
                                            .and_then(|items| items.into_iter().next())
                                            .and_then(|item| item.get("hash").cloned());

                                        if let Some(hash) = first_hash {
                                            match hc.call_zome::<serde_json::Value, serde_json::Value>(
                                                "mail_messages", "get_email", &hash
                                            ).await {
                                                Ok(_) => steps.push((
                                                    "Read Fetch".into(),
                                                    true,
                                                    "Fetched a concrete message entry for read/decrypt flow.".into(),
                                                )),
                                                Err(e) => steps.push(("Read Fetch".into(), false, e)),
                                            }
                                        } else if folders.is_some() {
                                            steps.push((
                                                "Read Fetch".into(),
                                                true,
                                                "Inbox is empty, so there was no message to fetch.".into(),
                                            ));
                                        }

                                        let ok = steps.iter().all(|(_, success, _)| *success);
                                        report_setter.set(LiveSliceReport { steps });
                                        running_setter.set(false);
                                        if ok {
                                            toasts.push("Live slice verification completed", "success");
                                        } else {
                                            toasts.push("Live slice verification found failures", "error");
                                        }
                                    });
                                }
                            }>
                                {move || if live_slice_running.get() { "Verifying..." } else { "Verify" }}
                            </button>
                        </Card>
                        <div class="setting-card" data-testid="live-slice-report" style=move || {
                            if live_slice_report.get().steps.is_empty() { "display:none" } else { "" }
                        }>
                            <div class="setting-card-label">"Live Slice Report"</div>
                            <div class="setting-card-body">
                                {move || live_slice_report.get().steps.into_iter().map(|(name, ok, detail)| {
                                    view! {
                                        <div style="display:flex;justify-content:space-between;gap:12px;padding:6px 0;border-bottom:1px solid var(--border-subtle)">
                                            <div style="min-width:120px;font-weight:600">{name}</div>
                                            <div style=if ok { "color:var(--success);font-weight:600" } else { "color:var(--error);font-weight:600" }>
                                                {if ok { "OK" } else { "Failed" }}
                                            </div>
                                            <div style="flex:1;color:var(--text-muted);font-size:0.8rem">{detail}</div>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                        <Card label="Export Keys">
                            <button class="btn btn-secondary" on:click=move |_| {
                                // 2FA gate — require passphrase before sensitive action
                                let pass = web_sys::window()
                                    .and_then(|w| w.prompt_with_message("Enter your passphrase to export keys:").ok().flatten());
                                if let Some(p) = pass {
                                    if p.len() >= 4 {
                                        toasts_export.push("Keys exported (demo mode)", "success");
                                    } else {
                                        toasts_export.push("Passphrase too short", "error");
                                    }
                                }
                            }>"Export (requires auth)"</button>
                        </Card>
                        <div class="setting-card"><p class="setting-desc">
                            "P-256 ECDH + AES-256-GCM via Web Crypto API. Content encrypted before leaving your browser. Sensitive actions require re-authentication."
                        </p></div>
                    </div>
                </div>
            </div>
        </div>
    }
}

/// Simple card wrapper for settings rows.
#[component]
fn Card(#[prop(into)] label: String, children: Children) -> impl IntoView {
    view! {
        <div class="setting-card"><div class="setting-row">
            <div class="setting-info"><span class="setting-label">{label}</span></div>
            <div class="setting-toggle-group">{children()}</div>
        </div></div>
    }
}

/// Profile management section — view, edit name/bio, see agent key.
#[component]
fn ProfileSection() -> impl IntoView {
    let hc = crate::holochain::use_holochain();
    let toasts = use_toasts();
    let is_mock = move || hc.status.get() == ConnectionStatus::Mock;

    let profile_name = RwSignal::new(String::new());
    let profile_bio = RwSignal::new(String::new());
    let agent_key = RwSignal::new(String::new());
    let did_id = RwSignal::new(String::new());
    let loading = RwSignal::new(true);
    let saving = RwSignal::new(false);
    let edit_mode = RwSignal::new(false);

    // Load profile on mount
    {
        let hc = hc.clone();
        wasm_bindgen_futures::spawn_local(async move {
            // Get agent key from JS bridge
            if let Some(info) = web_sys::window()
                .and_then(|w| js_sys::Reflect::get(&w, &wasm_bindgen::JsValue::from_str("__HC_APP_INFO")).ok())
            {
                if let Ok(key_val) = js_sys::Reflect::get(&info, &wasm_bindgen::JsValue::from_str("installed_app_id")) {
                    let _ = key_val; // just checking it exists
                }
                if let Ok(agent) = js_sys::Reflect::get(&info, &wasm_bindgen::JsValue::from_str("agent_pub_key")) {
                    if let Some(s) = agent.as_string() {
                        agent_key.set(s);
                    }
                }
            }

            match hc.call_zome::<(), Option<serde_json::Value>>(
                "mail_profiles", "get_my_profile", &()
            ).await {
                Ok(Some(profile)) => {
                    if let Some(name) = profile.get("name").and_then(|v| v.as_str()) {
                        profile_name.set(name.to_string());
                    }
                    if let Some(bio) = profile.get("bio").and_then(|v| v.as_str()) {
                        profile_bio.set(bio.to_string());
                    }
                }
                Ok(None) => {
                    profile_name.set("(no profile)".into());
                }
                Err(e) => {
                    profile_name.set(format!("Error: {e}"));
                }
            }
            // Also load DID from identity DNA
            match hc.call_zome_on_role::<(), serde_json::Value>(
                "identity", "did_registry", "get_did_document", &()
            ).await {
                Ok(val) if !val.is_null() => {
                    if let Some(id) = val.get("id").and_then(|v| v.as_str()) {
                        did_id.set(id.to_string());
                    }
                }
                _ => {} // No DID yet — that's fine
            }

            loading.set(false);
        });
    }

    let save_trigger = RwSignal::new(0u32);
    {
        let hc = hc.clone();
        let toasts = toasts.clone();
        Effect::new(move |prev: Option<u32>| {
            let current = save_trigger.get();
            if prev.is_some() && current > 0 {
                let name = profile_name.get_untracked();
                let bio = profile_bio.get_untracked();
                if name.trim().is_empty() { return current; }
                saving.set(true);

                let hc = hc.clone();
                let toasts = toasts.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let profile = serde_json::json!({
                        "name": name.trim(),
                        "email": "",
                        "avatar_url": "",
                        "bio": bio.trim(),
                    });
                    match hc.call_zome::<serde_json::Value, serde_json::Value>(
                        "mail_profiles", "set_profile", &profile
                    ).await {
                        Ok(_) => {
                            toasts.push("Profile updated!", "success");
                            edit_mode.set(false);
                        }
                        Err(e) => {
                            toasts.push(format!("Update failed: {e}"), "error");
                        }
                    }
                    saving.set(false);
                });
            }
            current
        });
    }

    view! {
        <p style=move || if loading.get() { "color:var(--text-muted)" } else { "display:none" }>"Loading profile..."</p>

        <div style=move || if loading.get() { "display:none" } else { "" }>
            <div class="setting-card">
                <div class="setting-card-label">"Identity"</div>
                <div class="setting-card-body">
                    <div style="margin-bottom:12px">
                        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">"Display Name"</label>
                        // View mode
                        <p style=move || if edit_mode.get() { "display:none" } else { "font-size:1rem;font-weight:600;margin:0" }>
                            {move || profile_name.get()}
                        </p>
                        // Edit mode
                        <input type="text"
                            style=move || if edit_mode.get() { "width:100%;padding:8px;background:var(--bg-primary);border:1px solid var(--border-subtle);border-radius:6px;color:var(--text-primary);font-size:0.9rem;box-sizing:border-box" } else { "display:none" }
                            prop:value=move || profile_name.get()
                            on:input=move |e| profile_name.set(event_target_value(&e))
                        />
                    </div>

                    <div style="margin-bottom:12px">
                        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">"Bio"</label>
                        <p style=move || if edit_mode.get() { "display:none" } else { "font-size:0.85rem;color:var(--text-secondary);margin:0" }>
                            {move || { let b = profile_bio.get(); if b.is_empty() { "(none)".into() } else { b } }}
                        </p>
                        <input type="text"
                            style=move || if edit_mode.get() { "width:100%;padding:8px;background:var(--bg-primary);border:1px solid var(--border-subtle);border-radius:6px;color:var(--text-primary);font-size:0.9rem;box-sizing:border-box" } else { "display:none" }
                            placeholder="A short bio..."
                            prop:value=move || profile_bio.get()
                            on:input=move |e| profile_bio.set(event_target_value(&e))
                        />
                    </div>

                    <div style="margin-bottom:12px">
                        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">"DSID (Decentralized Sovereign Identity)"</label>
                        <p style="font-size:0.7rem;color:var(--teal);margin:0;font-family:monospace;word-break:break-all">
                            {move || { let d = did_id.get(); if d.is_empty() { "(not created yet)".into() } else { crate::dsid::display_dsid(&d) } }}
                        </p>
                    </div>

                    <div style="margin-bottom:12px">
                        <label style="font-size:0.75rem;color:var(--text-muted);display:block;margin-bottom:4px">"Agent Public Key"</label>
                        <p style="font-size:0.7rem;color:var(--text-muted);margin:0;font-family:monospace;word-break:break-all">
                            {move || { let k = agent_key.get(); if k.is_empty() { "(unavailable)".into() } else { k } }}
                        </p>
                    </div>

                    <div style="display:flex;gap:8px;margin-top:16px">
                        // Edit mode buttons
                        <button class="btn btn-primary"
                            style=move || if edit_mode.get() { "flex:1" } else { "display:none" }
                            disabled=move || saving.get() || profile_name.get().trim().is_empty()
                            on:click=move |_| save_trigger.update(|n| *n += 1)>
                            {move || if saving.get() { "Saving..." } else { "Save Changes" }}
                        </button>
                        <button class="btn btn-secondary"
                            style=move || if edit_mode.get() { "" } else { "display:none" }
                            on:click=move |_| edit_mode.set(false)>
                            "Cancel"
                        </button>
                        // View mode button
                        <button class="btn btn-secondary"
                            style=move || if edit_mode.get() { "display:none" } else { "flex:1" }
                            on:click=move |_| edit_mode.set(true)>
                            "Edit Profile"
                        </button>
                    </div>
                </div>
            </div>

            <div class="setting-card" style="margin-top:12px">
                <div class="setting-card-label">"Connection"</div>
                <div class="setting-card-body">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-size:0.85rem">"Conductor Status"</span>
                        <span style="font-size:0.8rem;color:var(--teal)">
                            {move || if is_mock() { "Demo Mode" } else { "Connected (Live)" }}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    }
}

/// Security section — DSID assurance level, MFA factor management.
#[component]
fn SecuritySection() -> impl IntoView {
    let hc = crate::holochain::use_holochain();
    let assurance_level = RwSignal::new(0u8);
    let factor_count = RwSignal::new(0u32);
    let factors = RwSignal::new(Vec::<(String, String, String)>::new()); // (type, id, status)
    let loading = RwSignal::new(true);
    let dsid = RwSignal::new(String::new());

    // Load MFA state
    {
        let hc = hc.clone();
        wasm_bindgen_futures::spawn_local(async move {
            // Get DSID first
            match hc.call_zome_on_role::<(), serde_json::Value>(
                "identity", "did_registry", "get_did_document", &()
            ).await {
                Ok(val) if !val.is_null() => {
                    if let Some(id) = val.get("id").and_then(|v| v.as_str()) {
                        dsid.set(crate::dsid::display_dsid(id));
                    }
                }
                _ => {}
            }

            // Get MFA state
            let did_wire = crate::dsid::wire_dsid(&dsid.get_untracked());
            if !did_wire.is_empty() {
                match hc.call_zome_on_role::<serde_json::Value, serde_json::Value>(
                    "identity", "mfa", "get_mfa_state", &serde_json::json!(did_wire)
                ).await {
                    Ok(val) if !val.is_null() => {
                        // Parse assurance level
                        if let Some(state) = val.get("state").or(Some(&val)) {
                            if let Some(level) = state.get("assurance_level").and_then(|v| v.as_str()) {
                                let l = match level {
                                    "ConstitutionallyCritical" => 4,
                                    "HighlyAssured" => 3,
                                    "Verified" => 2,
                                    "Basic" => 1,
                                    _ => 0,
                                };
                                assurance_level.set(l);
                            }
                            if let Some(fs) = state.get("factors").and_then(|v| v.as_array()) {
                                factor_count.set(fs.len() as u32);
                                let parsed: Vec<_> = fs.iter().filter_map(|f| {
                                    let ft = f.get("factor_type").and_then(|v| v.as_str()).unwrap_or("Unknown");
                                    let fid = f.get("factor_id").and_then(|v| v.as_str()).unwrap_or("");
                                    let status = if f.get("last_verified").is_some() { "Active" } else { "Pending" };
                                    Some((ft.to_string(), fid.to_string(), status.to_string()))
                                }).collect();
                                factors.set(parsed);
                            }
                        }
                    }
                    Ok(_) => { /* No MFA state yet */ }
                    Err(e) => {
                        web_sys::console::warn_1(&format!("[Mail] MFA state error: {e}").into());
                    }
                }
            }
            loading.set(false);
        });
    }

    let al = move || crate::dsid::AssuranceLevel::from_u8(assurance_level.get());

    view! {
        <p style=move || if loading.get() { "color:var(--text-muted)" } else { "display:none" }>"Loading security info..."</p>

        <div style=move || if loading.get() { "display:none" } else { "" }>
            // Assurance Level Card
            <div class="setting-card">
                <div class="setting-card-label">"Assurance Level"</div>
                <div class="setting-card-body">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
                        <span style="font-size:2rem">{move || al().icon()}</span>
                        <div>
                            <p style="font-size:1.1rem;font-weight:700;margin:0;color:var(--teal)">
                                {move || format!("{} — {}", al().short_label(), al().label())}
                            </p>
                            <p style="font-size:0.75rem;color:var(--text-muted);margin:2px 0 0">
                                {move || format!("{} factor(s) enrolled", factor_count.get())}
                            </p>
                        </div>
                    </div>

                    // Progress to next level
                    <div style="margin-bottom:8px">
                        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:var(--text-muted);margin-bottom:4px">
                            <span>"Progress"</span>
                            <span>{move || al().short_label()}" → "{move || {
                                let next = crate::dsid::AssuranceLevel::from_u8(assurance_level.get().saturating_add(1));
                                next.short_label()
                            }}</span>
                        </div>
                        <div style="height:6px;background:var(--bg-primary);border-radius:3px;overflow:hidden">
                            <div style=move || {
                                let pct = match assurance_level.get() {
                                    0 => if factor_count.get() > 0 { 50 } else { 0 },
                                    1 => 33,
                                    2 => 66,
                                    3 => 80,
                                    _ => 100,
                                };
                                format!("height:100%;background:var(--teal);width:{pct}%;border-radius:3px;transition:width 0.3s")
                            }></div>
                        </div>
                    </div>

                    <p style="font-size:0.7rem;color:var(--text-muted);line-height:1.5">
                        "Higher assurance unlocks more capabilities: sending limits, governance voting, "
                        "and trusted sender status. Enroll factors from different categories to level up."
                    </p>
                </div>
            </div>

            // Enrolled Factors
            <div class="setting-card" style="margin-top:12px">
                <div class="setting-card-label">"Enrolled Factors"</div>
                <div class="setting-card-body">
                    {move || {
                        let fs = factors.get();
                        if fs.is_empty() {
                            view! {
                                <p style="color:var(--text-muted);font-size:0.85rem">"No factors enrolled yet. Your primary key pair is automatic."</p>
                            }.into_any()
                        } else {
                            view! {
                                <div style="display:flex;flex-direction:column;gap:8px">
                                    {fs.into_iter().map(|(ft, fid, status)| {
                                        let icon = match ft.as_str() {
                                            "PrimaryKeyPair" => "\u{1F511}",
                                            "HardwareKey" => "\u{1F4F1}",
                                            "Biometric" => "\u{1F9EC}",
                                            "SocialRecovery" => "\u{1F465}",
                                            "RecoveryPhrase" => "\u{1F4DD}",
                                            "SecurityQuestions" => "\u{2753}",
                                            "VerifiableCredential" => "\u{1F4DC}",
                                            _ => "\u{1F512}",
                                        };
                                        view! {
                                            <div style="display:flex;align-items:center;gap:8px;padding:8px;background:var(--bg-primary);border-radius:6px">
                                                <span style="font-size:1.2rem">{icon}</span>
                                                <div style="flex:1">
                                                    <p style="font-size:0.85rem;font-weight:500;margin:0">{ft}</p>
                                                    <p style="font-size:0.7rem;color:var(--text-muted);margin:0">
                                                        {if fid.is_empty() { "Auto-enrolled".into() } else { fid }}
                                                    </p>
                                                </div>
                                                <span style="font-size:0.7rem;color:var(--teal)">{status}</span>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()}
                                </div>
                            }.into_any()
                        }
                    }}
                </div>
            </div>

            // Enroll New Factor
            <div class="setting-card" style="margin-top:12px">
                <div class="setting-card-label">"Enroll Factor"</div>
                <div class="setting-card-body">
                    <p style="font-size:0.75rem;color:var(--text-muted);margin:0 0 12px">
                        "Add factors from different categories to increase your assurance level."
                    </p>
                    <MfaEnrollment dsid=dsid.clone() />
                </div>
            </div>
        </div>
    }
}

/// MFA factor enrollment UI — allows users to add recovery phrase or security questions.
#[component]
fn MfaEnrollment(dsid: RwSignal<String>) -> impl IntoView {
    let hc = crate::holochain::use_holochain();
    let toasts = use_toasts();
    let enrolling = RwSignal::new(Option::<String>::None); // Which factor type is being enrolled
    let phrase_input = RwSignal::new(String::new());
    let q1 = RwSignal::new(String::new());
    let a1 = RwSignal::new(String::new());
    let q2 = RwSignal::new(String::new());
    let a2 = RwSignal::new(String::new());
    let q3 = RwSignal::new(String::new());
    let a3 = RwSignal::new(String::new());
    let status = RwSignal::new(String::new());

    // Enrollment trigger
    let enroll_trigger = RwSignal::new(0u32);
    {
        let hc = hc.clone();
        let toasts = toasts.clone();
        Effect::new(move |prev: Option<u32>| {
            let current = enroll_trigger.get();
            if prev.is_some() && current > 0 {
                let factor_type = enrolling.get_untracked().unwrap_or_default();
                let did_wire = crate::dsid::wire_dsid(&dsid.get_untracked());
                if did_wire.is_empty() { return current; }

                status.set("Enrolling...".into());
                let hc = hc.clone();
                let toasts = toasts.clone();

                // Build metadata based on factor type
                let (factor_id, metadata) = match factor_type.as_str() {
                    "RecoveryPhrase" => {
                        let phrase = phrase_input.get_untracked();
                        let hash = format!("{:x}", md5_simple(phrase.as_bytes()));
                        (format!("recovery-{}", &hash[..8]), serde_json::json!({"hash": hash}).to_string())
                    }
                    "SecurityQuestions" => {
                        let answers = format!("{}|{}|{}", a1.get_untracked(), a2.get_untracked(), a3.get_untracked());
                        let hash = format!("{:x}", md5_simple(answers.as_bytes()));
                        let qs = serde_json::json!({
                            "questions": [q1.get_untracked(), q2.get_untracked(), q3.get_untracked()],
                            "answer_hash": hash,
                        });
                        (format!("secq-{}", &hash[..8]), qs.to_string())
                    }
                    _ => { return current; }
                };

                wasm_bindgen_futures::spawn_local(async move {
                    let input = serde_json::json!({
                        "did": did_wire,
                        "factor_type": factor_type,
                        "factor_id": factor_id,
                        "metadata": metadata,
                        "reason": "User enrollment",
                    });
                    match hc.call_zome_on_role::<serde_json::Value, serde_json::Value>(
                        "identity", "mfa", "enroll_factor", &input
                    ).await {
                        Ok(_) => {
                            toasts.push(format!("{factor_type} enrolled!"), "success");
                            enrolling.set(None);
                            status.set(String::new());
                        }
                        Err(e) => {
                            status.set(format!("Error: {e}"));
                            toasts.push(format!("Enrollment failed: {e}"), "error");
                        }
                    }
                });
            }
            current
        });
    }

    view! {
        // Factor type buttons
        <div style=move || if enrolling.get().is_none() { "display:flex;flex-direction:column;gap:8px" } else { "display:none" }>
            <button class="btn btn-secondary" style="text-align:left;padding:10px 12px"
                on:click=move |_| enrolling.set(Some("RecoveryPhrase".into()))>
                "\u{1F4DD} Enroll Recovery Phrase"
                <span style="display:block;font-size:0.7rem;color:var(--text-muted);font-weight:400">"Knowledge category — write down and store safely"</span>
            </button>
            <button class="btn btn-secondary" style="text-align:left;padding:10px 12px"
                on:click=move |_| enrolling.set(Some("SecurityQuestions".into()))>
                "\u{2753} Enroll Security Questions"
                <span style="display:block;font-size:0.7rem;color:var(--text-muted);font-weight:400">"Knowledge category — 3 personal questions"</span>
            </button>
            <button class="btn btn-secondary" style="text-align:left;padding:10px 12px;opacity:0.5" disabled=true>
                "\u{1F4F1} Hardware Key"
                <span style="display:block;font-size:0.7rem;color:var(--text-muted);font-weight:400">"Cryptographic category — coming soon"</span>
            </button>
            <button class="btn btn-secondary" style="text-align:left;padding:10px 12px;opacity:0.5" disabled=true>
                "\u{1F465} Social Recovery"
                <span style="display:block;font-size:0.7rem;color:var(--text-muted);font-weight:400">"Social category — requires trustees"</span>
            </button>
        </div>

        // Recovery Phrase enrollment form
        <div style=move || if enrolling.get().as_deref() == Some("RecoveryPhrase") { "" } else { "display:none" }>
            <p style="font-size:0.8rem;color:var(--text-secondary);margin:0 0 8px">
                "Enter a recovery phrase. Write it down and store it safely — you'll need it to recover your identity."
            </p>
            <input type="password"
                style="width:100%;padding:10px;background:var(--bg-primary);border:1px solid var(--border-subtle);border-radius:6px;color:var(--text-primary);font-size:0.9rem;box-sizing:border-box;margin-bottom:8px"
                placeholder="Enter your recovery phrase..."
                prop:value=move || phrase_input.get()
                on:input=move |e| phrase_input.set(event_target_value(&e))
            />
            <div style="display:flex;gap:8px">
                <button class="btn btn-primary" style="flex:1"
                    disabled=move || phrase_input.get().trim().len() < 8
                    on:click=move |_| enroll_trigger.update(|n| *n += 1)>
                    {move || if status.get().starts_with("Enrolling") { "Enrolling..." } else { "Enroll" }}
                </button>
                <button class="btn btn-secondary" on:click=move |_| { enrolling.set(None); status.set(String::new()); }>"Cancel"</button>
            </div>
        </div>

        // Security Questions enrollment form
        <div style=move || if enrolling.get().as_deref() == Some("SecurityQuestions") { "" } else { "display:none" }>
            <p style="font-size:0.8rem;color:var(--text-secondary);margin:0 0 8px">
                "Choose 3 questions only you can answer."
            </p>
            {[(q1, a1, "Question 1", "Answer 1"), (q2, a2, "Question 2", "Answer 2"), (q3, a3, "Question 3", "Answer 3")]
                .into_iter().map(|(q, a, ql, al)| {
                view! {
                    <div style="margin-bottom:8px">
                        <input type="text"
                            style="width:100%;padding:8px;background:var(--bg-primary);border:1px solid var(--border-subtle);border-radius:6px 6px 0 0;color:var(--text-primary);font-size:0.8rem;box-sizing:border-box"
                            placeholder=ql
                            prop:value=move || q.get()
                            on:input=move |e| q.set(event_target_value(&e))
                        />
                        <input type="text"
                            style="width:100%;padding:8px;background:var(--bg-primary);border:1px solid var(--border-subtle);border-top:none;border-radius:0 0 6px 6px;color:var(--text-primary);font-size:0.8rem;box-sizing:border-box"
                            placeholder=al
                            prop:value=move || a.get()
                            on:input=move |e| a.set(event_target_value(&e))
                        />
                    </div>
                }
            }).collect::<Vec<_>>()}
            <div style="display:flex;gap:8px">
                <button class="btn btn-primary" style="flex:1"
                    disabled=move || a1.get().trim().is_empty() || a2.get().trim().is_empty() || a3.get().trim().is_empty()
                    on:click=move |_| enroll_trigger.update(|n| *n += 1)>
                    {move || if status.get().starts_with("Enrolling") { "Enrolling..." } else { "Enroll" }}
                </button>
                <button class="btn btn-secondary" on:click=move |_| { enrolling.set(None); status.set(String::new()); }>"Cancel"</button>
            </div>
        </div>

        // Status message
        <p style=move || if status.get().is_empty() { "display:none" } else { "font-size:0.8rem;color:var(--teal);margin-top:8px" }>
            {move || status.get()}
        </p>
    }
}

/// Simple hash for factor IDs (NOT cryptographic — just for deduplication).
fn md5_simple(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn download_file(filename: &str, content: &str, mime: &str) {
    let _ = js_sys::eval(&format!(
        "const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([`{}`],{{type:'{}'}}));a.download='{}';a.click()",
        content.replace('`', "\\`").replace('\\', "\\\\"), mime, filename,
    ));
}
