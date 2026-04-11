// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Contact card with quick-compose (#6).

use leptos::prelude::*;
use mail_leptos_types::{ContactView, ComposeMode};
use crate::mail_context::use_mail;

#[component]
pub fn ContactCard(contact: ContactView) -> impl IntoView {
    let mail = use_mail();
    let initials = contact.initials();
    let name = contact.display_name.clone();
    let email = contact.email.clone().unwrap_or_default();
    let org = contact.organization.clone().unwrap_or_default();
    let trust_score = contact.trust_score;
    let is_favorite = contact.is_favorite;
    let email_count = contact.email_count;
    let agent_short = contact.short_agent();
    let agent_key = contact.agent_pub_key.clone().unwrap_or_default();

    let trust_class = trust_score.map(|s| {
        if s >= 0.8 { "trust-high" }
        else if s >= 0.5 { "trust-medium" }
        else { "trust-low" }
    }).unwrap_or("trust-unknown");

    // Quick-compose (#6)
    let compose_agent = agent_key.clone();
    let compose_name = name.clone();
    let on_compose = move |_| {
        mail.compose_mode.set(ComposeMode::Reply {
            email_hash: String::new(),
            sender: compose_agent.clone(),
            sender_name: compose_name.clone(),
            subject: String::new(),
            body: String::new(),
            thread_id: None,
        });
        let nav = leptos_router::hooks::use_navigate();
        nav("/compose", Default::default());
    };

    // Presence — in production comes from remote_signal heartbeat
    // In demo mode, deterministic from agent key hash
    let presence = {
        let hash: u32 = agent_key.bytes().fold(0u32, |a, b| a.wrapping_mul(31).wrapping_add(b as u32));
        match hash % 3 {
            0 => ("presence-online", "Online"),
            1 => ("presence-away", "Away"),
            _ => ("presence-offline", "Offline"),
        }
    };

    view! {
        <div class="contact-card">
            <div class="contact-avatar">
                <span class="contact-initials">{initials}</span>
                <span class=format!("presence-dot {}", presence.0) title=presence.1 />
                {is_favorite.then(|| view! { <span class="favorite-star">"\u{2B50}"</span> })}
            </div>
            <div class="contact-info">
                <div class="contact-name">{name.clone()}</div>
                {(!email.is_empty()).then(|| view! { <div class="contact-email">{email.clone()}</div> })}
                {(!org.is_empty()).then(|| view! { <div class="contact-org">{org.clone()}</div> })}
                {(!agent_short.is_empty()).then(|| view! {
                    <div class="contact-agent" title="Holochain agent public key">{agent_short}</div>
                })}
            </div>
            // Private notes
            <ContactNotes agent_key=agent_key.clone() />
            <div class="contact-meta">
                {trust_score.map(|s| view! {
                    <span class=format!("trust-score {trust_class}") title="MATL trust score">
                        {format!("{:.0}%", s * 100.0)}
                    </span>
                })}
                <span class="email-count" title="Emails exchanged">
                    {format!("{email_count} msgs")}
                </span>
                <button class="btn btn-sm btn-secondary contact-compose-btn" on:click=on_compose title="Compose email">
                    "\u{270F}"
                </button>
            </div>
        </div>
    }
}

/// Inline expandable notes for a contact, stored in localStorage.
#[component]
fn ContactNotes(agent_key: String) -> impl IntoView {
    let key = format!("mycelix_contact_note_{}", agent_key);
    let expanded = RwSignal::new(false);
    let note_text = RwSignal::new(
        web_sys::window()
            .and_then(|w| w.local_storage().ok().flatten())
            .and_then(|s| s.get_item(&key).ok().flatten())
            .unwrap_or_default()
    );
    let has_note = move || !note_text.get().is_empty();

    let key_save = key.clone();
    let on_save = move |_| {
        let text = note_text.get_untracked();
        if let Some(s) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            if text.is_empty() {
                let _ = s.remove_item(&key_save);
            } else {
                let _ = s.set_item(&key_save, &text);
            }
        }
        expanded.set(false);
    };

    view! {
        <div class="contact-notes">
            <button class="contact-note-toggle" on:click=move |_| expanded.update(|v| *v = !*v)>
                {move || if has_note() { "\u{1F4DD}" } else { "\u{270F}" }}
            </button>
            <div class="contact-note-editor" style=move || if expanded.get() { "" } else { "display:none" }>
                <textarea
                    class="contact-note-input"
                    placeholder="Private notes about this contact..."
                    prop:value=move || note_text.get()
                    on:input=move |ev| {
                        use wasm_bindgen::JsCast;
                        let val = ev.target().and_then(|t| t.dyn_into::<web_sys::HtmlTextAreaElement>().ok()).map(|el| el.value()).unwrap_or_default();
                        note_text.set(val);
                    }
                />
                <button class="btn btn-sm btn-primary" on:click=on_save>"Save"</button>
            </div>
        </div>
    }
}
