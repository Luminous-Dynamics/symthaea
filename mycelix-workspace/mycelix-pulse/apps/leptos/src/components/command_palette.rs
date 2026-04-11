// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Command Palette — configurable shortcut, customizable actions.
//! Default: Ctrl+K (configurable via settings).
//! Also accessible via the search icon button in the navbar.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use crate::mail_context::use_mail;

const SHORTCUT_KEY: &str = "mycelix_pulse_palette_shortcut";
const PINNED_ACTIONS_KEY: &str = "mycelix_pulse_pinned_actions";

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PaletteAction {
    pub id: String,
    pub label: String,
    pub shortcut: String,
    pub category: String,
    pub icon: String,
    pub pinned: bool,
}

fn default_actions() -> Vec<PaletteAction> {
    vec![
        PaletteAction { id: "inbox".into(), label: "Go to Inbox".into(), shortcut: "g i".into(), category: "Navigate".into(), icon: "\u{1F4E5}".into(), pinned: false },
        PaletteAction { id: "compose".into(), label: "Compose New Email".into(), shortcut: "c".into(), category: "Actions".into(), icon: "\u{270F}".into(), pinned: true },
        PaletteAction { id: "calendar".into(), label: "Go to Calendar".into(), shortcut: "g l".into(), category: "Navigate".into(), icon: "\u{1F4C5}".into(), pinned: false },
        PaletteAction { id: "chat".into(), label: "Go to Chat".into(), shortcut: "g h".into(), category: "Navigate".into(), icon: "\u{1F4AC}".into(), pinned: false },
        PaletteAction { id: "meet".into(), label: "Start a Call".into(), shortcut: "".into(), category: "Actions".into(), icon: "\u{1F4F9}".into(), pinned: false },
        PaletteAction { id: "contacts".into(), label: "Go to Contacts".into(), shortcut: "g c".into(), category: "Navigate".into(), icon: "\u{1F465}".into(), pinned: false },
        PaletteAction { id: "search".into(), label: "Search Messages".into(), shortcut: "/".into(), category: "Navigate".into(), icon: "\u{1F50D}".into(), pinned: true },
        PaletteAction { id: "settings".into(), label: "Open Settings".into(), shortcut: "".into(), category: "Navigate".into(), icon: "\u{2699}".into(), pinned: false },
        PaletteAction { id: "accounts".into(), label: "External Accounts".into(), shortcut: "".into(), category: "Navigate".into(), icon: "\u{1F4EC}".into(), pinned: false },
        PaletteAction { id: "drafts".into(), label: "Go to Drafts".into(), shortcut: "g d".into(), category: "Navigate".into(), icon: "\u{1F4DD}".into(), pinned: false },
        PaletteAction { id: "toggle-theme".into(), label: "Toggle Dark/Light Theme".into(), shortcut: "".into(), category: "Appearance".into(), icon: "\u{1F319}".into(), pinned: false },
        PaletteAction { id: "density-compact".into(), label: "Compact Density".into(), shortcut: "".into(), category: "Appearance".into(), icon: "\u{2B1C}".into(), pinned: false },
        PaletteAction { id: "density-default".into(), label: "Default Density".into(), shortcut: "".into(), category: "Appearance".into(), icon: "\u{2B1C}".into(), pinned: false },
        PaletteAction { id: "density-comfortable".into(), label: "Comfortable Density".into(), shortcut: "".into(), category: "Appearance".into(), icon: "\u{2B1C}".into(), pinned: false },
        PaletteAction { id: "new-event".into(), label: "Create Calendar Event".into(), shortcut: "".into(), category: "Actions".into(), icon: "\u{1F4C5}".into(), pinned: false },
        PaletteAction { id: "mark-all-read".into(), label: "Mark All as Read".into(), shortcut: "".into(), category: "Actions".into(), icon: "\u{2709}".into(), pinned: false },
        PaletteAction { id: "select-all".into(), label: "Select All Emails".into(), shortcut: "".into(), category: "Actions".into(), icon: "\u{2611}".into(), pinned: false },
        PaletteAction { id: "keyboard-help".into(), label: "Keyboard Shortcuts".into(), shortcut: "?".into(), category: "Help".into(), icon: "\u{2328}".into(), pinned: false },
        PaletteAction { id: "clear-demo".into(), label: "Reset Demo Data".into(), shortcut: "".into(), category: "Actions".into(), icon: "\u{1F5D1}".into(), pinned: false },
        PaletteAction { id: "change-shortcut".into(), label: "Change Palette Shortcut".into(), shortcut: "".into(), category: "Settings".into(), icon: "\u{2699}".into(), pinned: false },
    ]
}

fn load_shortcut() -> String {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item(SHORTCUT_KEY).ok().flatten())
        .unwrap_or_else(|| "k".into())
}

fn save_shortcut(key: &str) {
    if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        let _ = s.set_item(SHORTCUT_KEY, key);
    }
}

#[component]
pub fn CommandPalette() -> impl IntoView {
    let open = RwSignal::new(false);
    let query = RwSignal::new(String::new());
    let selected_idx = RwSignal::new(0usize);
    let shortcut_key = RwSignal::new(load_shortcut());
    let changing_shortcut = RwSignal::new(false);

    // Listen for configurable shortcut
    let closure = Closure::<dyn Fn(web_sys::KeyboardEvent)>::new(move |ev: web_sys::KeyboardEvent| {
        // Skip when in input fields
        if let Some(target) = ev.target() {
            if let Ok(el) = target.dyn_into::<web_sys::HtmlElement>() {
                let tag = el.tag_name().to_lowercase();
                if tag == "input" || tag == "textarea" || tag == "select" {
                    // Allow Escape to close palette even from inputs
                    if ev.key() != "Escape" { return; }
                }
            }
        }

        // Changing shortcut mode: capture next key
        if changing_shortcut.get_untracked() {
            ev.prevent_default();
            let key = ev.key().to_lowercase();
            if key != "escape" && key != "shift" && key != "control" && key != "alt" && key != "meta" {
                shortcut_key.set(key.clone());
                save_shortcut(&key);
                changing_shortcut.set(false);
                open.set(false);
            }
            return;
        }

        let expected_key = shortcut_key.get_untracked();
        if (ev.meta_key() || ev.ctrl_key()) && ev.key().to_lowercase() == expected_key {
            ev.prevent_default();
            open.update(|v| *v = !*v);
            query.set(String::new());
            selected_idx.set(0);
        }
        if ev.key() == "Escape" && open.get_untracked() {
            open.set(false);
        }
    });
    let window = web_sys::window().unwrap();
    let _ = window.add_event_listener_with_callback("keydown", closure.as_ref().unchecked_ref());
    closure.forget();

    let _actions = default_actions();

    let filtered = move || {
        let q = query.get().to_lowercase();
        let all = default_actions();
        if q.is_empty() {
            all
        } else {
            all.into_iter()
                .filter(|a| a.label.to_lowercase().contains(&q) || a.category.to_lowercase().contains(&q))
                .collect()
        }
    };

    let execute = move |id: &str| {
        if id == "change-shortcut" {
            changing_shortcut.set(true);
            query.set(String::new());
            return;
        }
        if id == "clear-demo" {
            crate::persistence::clear_demo_data();
            let _ = web_sys::window().unwrap().location().reload();
            return;
        }

        open.set(false);
        let nav = leptos_router::hooks::use_navigate();
        match id {
            "inbox" => nav("/", Default::default()),
            "compose" => nav("/compose", Default::default()),
            "calendar" => nav("/calendar", Default::default()),
            "chat" => nav("/chat", Default::default()),
            "meet" => nav("/meet", Default::default()),
            "contacts" => nav("/contacts", Default::default()),
            "search" => nav("/search", Default::default()),
            "settings" => nav("/settings", Default::default()),
            "accounts" => nav("/accounts", Default::default()),
            "drafts" => nav("/drafts", Default::default()),
            "toggle-theme" => {
                let theme = crate::theme::use_theme();
                theme.current.update(|t| *t = t.toggle());
            }
            "density-compact" => { crate::theme::use_theme().density.set(mail_leptos_types::Density::Compact); }
            "density-default" => { crate::theme::use_theme().density.set(mail_leptos_types::Density::Default); }
            "density-comfortable" => { crate::theme::use_theme().density.set(mail_leptos_types::Density::Comfortable); }
            "mark-all-read" => {
                let mail = use_mail();
                mail.inbox.update(|emails| { for e in emails.iter_mut() { e.is_read = true; } });
            }
            "select-all" => { use_mail().select_all(); }
            _ => {}
        }
    };

    let on_keydown = move |ev: web_sys::KeyboardEvent| {
        let items = filtered();
        match ev.key().as_str() {
            "ArrowDown" => { ev.prevent_default(); selected_idx.update(|i| *i = (*i + 1).min(items.len().saturating_sub(1))); }
            "ArrowUp" => { ev.prevent_default(); selected_idx.update(|i| *i = i.saturating_sub(1)); }
            "Enter" => {
                let idx = selected_idx.get_untracked();
                if let Some(action) = items.get(idx) { execute(&action.id); }
            }
            _ => {}
        }
    };

    view! {
        <div class="cmd-palette-container" style=move || if open.get() { "" } else { "display:none" }>
            <div class="cmd-palette-overlay" on:click=move |_| { open.set(false); changing_shortcut.set(false); } />
            <div class="cmd-palette">
                <div class="cmd-input-row">
                    <span class="cmd-icon">"\u{1F50D}"</span>
                    {move || if changing_shortcut.get() {
                        view! {
                            <span class="cmd-input-placeholder">"Press any key to set as new shortcut..."</span>
                        }.into_any()
                    } else {
                        view! {
                            <input
                                class="cmd-input"
                                type="text"
                                placeholder="Type a command or search..."
                                prop:value=move || query.get()
                                on:input=move |ev| {
                                    let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok()).map(|el| el.value()).unwrap_or_default();
                                    query.set(val);
                                    selected_idx.set(0);
                                }
                                on:keydown=on_keydown
                            />
                        }.into_any()
                    }}
                    <kbd class="cmd-shortcut-hint">{move || format!("Ctrl+{}", shortcut_key.get().to_uppercase())}</kbd>
                </div>
                <div class="cmd-results">
                    {move || {
                        let items = filtered();
                        let mut current_category = String::new();
                        items.iter().enumerate().map(|(idx, action)| {
                            let show_header = action.category != current_category;
                            current_category = action.category.clone();
                            let cat = action.category.clone();
                            let label = action.label.clone();
                            let icon = action.icon.clone();
                            let shortcut = action.shortcut.clone();
                            let is_selected = idx == selected_idx.get();
                            let action_id = action.id.clone();
                            view! {
                                <div>
                                    {show_header.then(|| view! { <div class="cmd-category">{cat}</div> })}
                                    <button
                                        class=if is_selected { "cmd-item selected" } else { "cmd-item" }
                                        on:click=move |_| execute(&action_id)
                                    >
                                        <span class="cmd-item-icon">{icon.clone()}</span>
                                        <span class="cmd-item-label">{label.clone()}</span>
                                        {(!shortcut.is_empty()).then(|| view! {
                                            <kbd class="cmd-item-shortcut">{shortcut.clone()}</kbd>
                                        })}
                                    </button>
                                </div>
                            }
                        }).collect::<Vec<_>>()
                    }}
                </div>
                <div class="cmd-footer">
                    <span>"\u{2191}\u{2193} Navigate"</span>
                    <span>"\u{21B5} Select"</span>
                    <span>"Esc Close"</span>
                </div>
            </div>
        </div>
    }
}
