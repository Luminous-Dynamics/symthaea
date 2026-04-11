// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Search page with zome-backed BM25 search (#8), client-side fallback for mock mode.

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use crate::mail_context::use_mail;
use crate::holochain::{use_holochain, ConnectionStatus};
use crate::components::EmailCard;
use crate::semantic;
use mail_leptos_types::EmailListItem;

#[component]
pub fn SearchPage() -> impl IntoView {
    let mail = use_mail();
    let hc = use_holochain();
    let query = RwSignal::new(String::new());
    let has_attachments = RwSignal::new(false);
    let starred_only = RwSignal::new(false);
    let zome_results = RwSignal::new(Option::<Vec<EmailListItem>>::None);
    let searching = RwSignal::new(false);

    // Client-side search with operator support (#8)
    let mail_client = mail.clone();
    let client_results = move || {
        let raw = query.get();
        if raw.len() < 2 { return vec![]; }
        let attach_filter = has_attachments.get();
        let star_filter = starred_only.get();

        // Parse search operators: from:, to:, subject:, has:attachment, is:unread, is:starred
        let mut from_filter = None;
        let mut subject_filter = None;
        let mut free_text = Vec::new();
        let mut force_unread = false;
        let mut force_attachment = attach_filter;
        let mut force_starred = star_filter;

        for token in raw.split_whitespace() {
            let lower = token.to_lowercase();
            if let Some(v) = lower.strip_prefix("from:") {
                from_filter = Some(v.to_string());
            } else if let Some(v) = lower.strip_prefix("subject:") {
                subject_filter = Some(v.to_string());
            } else if lower == "has:attachment" {
                force_attachment = true;
            } else if lower == "is:unread" {
                force_unread = true;
            } else if lower == "is:starred" {
                force_starred = true;
            } else {
                free_text.push(lower);
            }
        }

        let text_query = free_text.join(" ");
        let _has_operators = from_filter.is_some() || subject_filter.is_some()
            || force_attachment || force_unread || force_starred;

        let emails = mail_client.inbox.get();

        // Apply operator filters first
        let filtered: Vec<_> = emails.into_iter().filter(|e| {
            if let Some(ref f) = from_filter {
                let sender_name = e.sender_name.as_deref().unwrap_or("").to_lowercase();
                if !sender_name.contains(f) { return false; }
            }
            if let Some(ref s) = subject_filter {
                let subj = e.subject.as_deref().unwrap_or("").to_lowercase();
                if !subj.contains(s) { return false; }
            }
            if force_attachment && !e.has_attachments { return false; }
            if force_unread && e.is_read { return false; }
            if force_starred && !e.is_starred { return false; }
            true
        }).collect();

        if text_query.is_empty() {
            return filtered;
        }

        // Semantic search: encode query and all emails, rank by similarity
        let query_hv = semantic::encode_text(&text_query);
        let email_texts: Vec<String> = filtered.iter().map(|e| {
            format!("{} {} {}",
                e.subject.as_deref().unwrap_or(""),
                e.snippet.as_deref().unwrap_or(""),
                e.sender_name.as_deref().unwrap_or(""))
        }).collect();
        let email_hvs: Vec<semantic::HyperVector> = email_texts.iter()
            .map(|t| semantic::encode_text(t)).collect();

        let mut results = semantic::semantic_search(&query_hv, &email_hvs, 0.05);

        // Also include exact keyword matches (boost them)
        for (i, e) in filtered.iter().enumerate() {
            let subject_match = e.subject.as_deref().unwrap_or("").to_lowercase().contains(&text_query);
            let snippet_match = e.snippet.as_deref().unwrap_or("").to_lowercase().contains(&text_query);
            let sender_match = e.sender_name.as_deref().unwrap_or("").to_lowercase().contains(&text_query);
            if (subject_match || snippet_match || sender_match) &&
                !results.iter().any(|r| r.index == i) {
                results.push(semantic::SemanticResult { index: i, similarity: 1.0 });
            }
        }

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.into_iter().filter_map(|r| filtered.get(r.index).cloned()).collect()
    };

    let results = move || zome_results.get().unwrap_or_else(client_results);
    let result_count = move || results().len();

    // Cross-domain search: contacts + calendar
    let mail_contacts = mail.clone();
    let contact_results = move || {
        let q = query.get().to_lowercase();
        if q.len() < 2 { return vec![]; }
        mail_contacts.contacts.get().into_iter().filter(|c| {
            c.display_name.to_lowercase().contains(&q)
            || c.email.as_deref().unwrap_or("").to_lowercase().contains(&q)
            || c.organization.as_deref().unwrap_or("").to_lowercase().contains(&q)
        }).take(5).collect::<Vec<_>>()
    };

    let on_input = move |ev: leptos::ev::Event| {
        let val = ev.target()
            .and_then(|t| t.dyn_into::<web_sys::HtmlInputElement>().ok())
            .map(|el| el.value())
            .unwrap_or_default();
        query.set(val.clone());

        // Zome search when connected (#8)
        let hc_inner = use_holochain();
        if val.len() >= 2 && !hc_inner.is_mock() {
            searching.set(true);
            let hc2 = hc_inner.clone();
            let attach = has_attachments.get_untracked();
            let star = starred_only.get_untracked();
            spawn_local(async move {
                let input = serde_json::json!({
                    "query": val, "filters": { "has_attachments": attach, "is_starred": star },
                    "limit": 50, "fuzzy": true, "sort_by": "Relevance",
                });
                match hc2.call_zome::<serde_json::Value, serde_json::Value>(
                    "mail_search", "search", &input
                ).await {
                    Ok(response) => {
                        if let Some(results) = response.get("results") {
                            if let Ok(emails) = serde_json::from_value::<Vec<EmailListItem>>(results.clone()) {
                                zome_results.set(Some(emails));
                            }
                        }
                    }
                    Err(_) => { zome_results.set(None); }
                }
                searching.set(false);
            });
        } else {
            zome_results.set(None);
        }
    };

    let is_mock = move || hc.status.get() == ConnectionStatus::Mock;

    view! {
        <div class="page page-search">
            <div class="page-header">
                <h1>"Search"</h1>
                {move || searching.get().then(|| view! {
                    <span class="search-spinner">"Searching..."</span>
                })}
            </div>

            <div class="search-form">
                <input
                    type="text"
                    class="search-input large"
                    placeholder="Search messages, contacts, subjects..."
                    prop:value=move || query.get()
                    on:input=on_input
                    id="search-input"
                />
                <div class="search-filters">
                    <label class="filter-toggle">
                        <input
                            type="checkbox"
                            prop:checked=move || has_attachments.get()
                            on:change=move |_| has_attachments.update(|v| *v = !*v)
                        />
                        " Has attachments"
                    </label>
                    <label class="filter-toggle">
                        <input
                            type="checkbox"
                            prop:checked=move || starred_only.get()
                            on:change=move |_| starred_only.update(|v| *v = !*v)
                        />
                        " Starred only"
                    </label>
                    {move || (!is_mock()).then(|| view! {
                        <span class="search-engine-badge">"BM25"</span>
                    })}
                </div>
            </div>

            {move || {
                let q = query.get();
                if q.len() < 2 {
                    view! {
                        <div class="search-prompt">
                            <svg viewBox="0 0 100 100" width="80" height="80" class="empty-svg" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="42" cy="42" r="28" fill="none" stroke="#06D6C8" stroke-width="2" opacity="0.3" />
                                <line x1="62" y1="62" x2="85" y2="85" stroke="#06D6C8" stroke-width="3" stroke-linecap="round" opacity="0.4" />
                                <path d="M32 42 H52 M42 32 V52" stroke="#06D6C8" stroke-width="1.5" stroke-linecap="round" opacity="0.2" />
                            </svg>
                            <p>"Type at least 2 characters to search"</p>
                            <p class="search-hint">"\u{1F9E0} Semantic search: finds emails by meaning, not just keywords"</p>
                            <p class="search-hint">"Operators: from: subject: has:attachment is:unread is:starred"</p>
                        </div>
                    }.into_any()
                } else {
                    let res = results();
                    if res.is_empty() {
                        view! {
                            <div class="empty-state">
                                <p class="empty-title">"No results"</p>
                                <p class="empty-desc">{format!("No messages match \"{q}\"")}</p>
                            </div>
                        }.into_any()
                    } else {
                        let contacts = contact_results();
                        view! {
                            <div>
                                // Contact matches
                                {(!contacts.is_empty()).then(|| {
                                    let cs = contacts.clone();
                                    view! {
                                        <div class="search-section">
                                            <h3 class="search-section-title">"\u{1F465} Contacts"</h3>
                                            <div class="search-contact-results">
                                                {cs.into_iter().map(|c| {
                                                    let name = c.display_name.clone();
                                                    let email = c.email.clone().unwrap_or_default();
                                                    let org = c.organization.clone().unwrap_or_default();
                                                    view! {
                                                        <div class="search-contact-item">
                                                            <span class="search-contact-avatar">{name.chars().next().unwrap_or('?').to_uppercase().to_string()}</span>
                                                            <div class="search-contact-info">
                                                                <span class="search-contact-name">{name}</span>
                                                                <span class="search-contact-email">{email}</span>
                                                                {(!org.is_empty()).then(|| view! { <span class="search-contact-org">{org}</span> })}
                                                            </div>
                                                        </div>
                                                    }
                                                }).collect::<Vec<_>>()}
                                            </div>
                                        </div>
                                    }
                                })}
                                // Email matches
                                <div class="search-section">
                                    <h3 class="search-section-title">{format!("\u{2709} {} email results", result_count())}</h3>
                                    <div class="email-list" role="list">
                                        {res.into_iter().map(|e| {
                                            view! { <EmailCard email=e /> }
                                        }).collect::<Vec<_>>()}
                                    </div>
                                </div>
                            </div>
                        }.into_any()
                    }
                }
            }}
        </div>
    }
}
