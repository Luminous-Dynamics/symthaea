// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Global keyboard shortcuts (#1).
//!
//! j/k: Navigate emails, Enter: Open, r: Reply, e: Archive,
//! s: Star, #: Delete, /: Focus search, Esc: Back to inbox.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use crate::mail_context::use_mail;

/// Keyboard selection state — tracks which email is focused via j/k.
#[derive(Clone, Copy)]
pub struct KeyboardState {
    pub focused_index: RwSignal<Option<usize>>,
    pub enabled: RwSignal<bool>,
}

pub fn provide_keyboard_context() {
    let state = KeyboardState {
        focused_index: RwSignal::new(None),
        enabled: RwSignal::new(true),
    };
    provide_context(state);

    // Register global keydown handler
    let closure = Closure::<dyn Fn(web_sys::KeyboardEvent)>::new(move |ev: web_sys::KeyboardEvent| {
        // Skip when typing in inputs
        if let Some(target) = ev.target() {
            if let Ok(el) = target.dyn_into::<web_sys::HtmlElement>() {
                let tag = el.tag_name().to_lowercase();
                if tag == "input" || tag == "textarea" || tag == "select" {
                    return;
                }
            }
        }

        if !state.enabled.get_untracked() { return; }

        let mail = use_mail();
        let email_count = mail.inbox.get_untracked().len();
        if email_count == 0 && ev.key() != "/" && ev.key() != "Escape" { return; }

        // Check custom keybindings from preferences (if available)
        // Custom bindings map: action_name → key
        // For now, use default bindings. Custom remapping is stored in UserPreferences
        // and would reverse-lookup: key → action_name here.

        match ev.key().as_str() {
            "j" => {
                // Next email
                state.focused_index.update(|idx| {
                    *idx = Some(idx.map(|i| (i + 1).min(email_count.saturating_sub(1))).unwrap_or(0));
                });
            }
            "k" => {
                // Previous email
                state.focused_index.update(|idx| {
                    *idx = Some(idx.map(|i| i.saturating_sub(1)).unwrap_or(0));
                });
            }
            "Enter" => {
                // Open focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        let hash = email.hash.clone();
                        let nav = leptos_router::hooks::use_navigate();
                        nav(&format!("/read/{hash}"), Default::default());
                    }
                }
            }
            "s" => {
                // Star focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.toggle_star(&email.hash.clone());
                    }
                }
            }
            "e" => {
                // Archive focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.archive_email(&email.hash.clone());
                    }
                }
            }
            "r" => {
                // Reply to focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.compose_mode.set(mail_leptos_types::ComposeMode::Reply {
                            email_hash: email.hash.clone(),
                            sender: email.sender.clone(),
                            sender_name: email.sender_name.clone().unwrap_or_default(),
                            subject: email.subject.clone().unwrap_or_default(),
                            body: email.snippet.clone().unwrap_or_default(),
                            thread_id: email.thread_id.clone(),
                        });
                        let nav = leptos_router::hooks::use_navigate();
                        nav("/compose", Default::default());
                    }
                }
            }
            "#" => {
                // Delete focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.delete_email(&email.hash.clone());
                    }
                }
            }
            "/" => {
                // Focus search
                ev.prevent_default();
                let nav = leptos_router::hooks::use_navigate();
                nav("/search", Default::default());
            }
            "Escape" => {
                state.focused_index.set(None);
                let nav = leptos_router::hooks::use_navigate();
                nav("/", Default::default());
            }
            // Extended shortcuts (#15)
            "c" => {
                // Compose new
                mail.compose_mode.set(mail_leptos_types::ComposeMode::New);
                let nav = leptos_router::hooks::use_navigate();
                nav("/compose", Default::default());
            }
            "f" => {
                // Forward focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.compose_mode.set(mail_leptos_types::ComposeMode::Forward {
                            subject: format!("Fwd: {}", email.subject.clone().unwrap_or_default()),
                            body: email.snippet.clone().unwrap_or_default(),
                        });
                        let nav = leptos_router::hooks::use_navigate();
                        nav("/compose", Default::default());
                    }
                }
            }
            "u" => {
                // Return to inbox (like Gmail)
                state.focused_index.set(None);
                let nav = leptos_router::hooks::use_navigate();
                nav("/", Default::default());
            }
            "x" => {
                // Toggle selection on focused email
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.toggle_selection(&email.hash.clone());
                    }
                }
            }
            "m" => {
                // Mute conversation
                if let Some(idx) = state.focused_index.get_untracked() {
                    let emails = mail.inbox.get_untracked();
                    if let Some(email) = emails.get(idx) {
                        mail.mute_thread(&email.hash.clone());
                    }
                }
            }
            "l" => {
                // Label (navigate to settings/labels for now)
                let nav = leptos_router::hooks::use_navigate();
                nav("/settings", Default::default());
            }
            "g" => {
                // Go to... (first letter of destination)
                // g+i = inbox, g+s = starred, g+d = drafts, g+c = contacts
                // For simplicity, just go to inbox
                let nav = leptos_router::hooks::use_navigate();
                nav("/", Default::default());
            }
            "n" | "p" => {
                // Next/previous message in thread (alias j/k for thread context)
                let delta: i32 = if ev.key() == "n" { 1 } else { -1 };
                state.focused_index.update(|idx| {
                    let current = idx.unwrap_or(0) as i32;
                    let next = (current + delta).max(0).min(email_count.saturating_sub(1) as i32) as usize;
                    *idx = Some(next);
                });
            }
            "." => {
                // More actions (noop for now, could open context menu)
            }
            "?" => {
                // Handled by keyboard_help.rs
            }
            _ => {}
        }
    });

    let window = web_sys::window().unwrap();
    window.add_event_listener_with_callback(
        "keydown",
        closure.as_ref().unchecked_ref(),
    ).unwrap();
    closure.forget(); // Leak intentionally — lives for app lifetime
}

pub fn use_keyboard() -> KeyboardState {
    expect_context::<KeyboardState>()
}
