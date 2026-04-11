// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::mail_context::use_mail;

#[component]
pub fn DraftsPage() -> impl IntoView {
    let mail = use_mail();

    view! {
        <div class="page page-drafts">
            <div class="page-header">
                <h1>"Drafts"</h1>
                <span class="draft-count">{move || format!("{} drafts", mail.drafts.get().len())}</span>
            </div>

            <div class="draft-list" role="list">
                {move || {
                    let drafts = mail.drafts.get();
                    if drafts.is_empty() {
                        view! {
                            <div class="empty-state">
                                <span class="empty-icon">"\u{1F4DD}"</span>
                                <p class="empty-title">"No drafts"</p>
                                <p class="empty-desc">"Saved drafts will appear here."</p>
                            </div>
                        }.into_any()
                    } else {
                        view! {
                            <div>
                                {drafts.into_iter().map(|d| {
                                    let subject = if d.subject.is_empty() { "(no subject)".to_string() } else { d.subject.clone() };
                                    let snippet = if d.body.len() > 100 {
                                        format!("{}...", &d.body[..100])
                                    } else {
                                        d.body.clone()
                                    };
                                    let recipients = d.recipients.len();
                                    let scheduled = d.scheduled_for.is_some();

                                    view! {
                                        <div class="draft-card">
                                            <div class="draft-info">
                                                <div class="draft-subject">{subject}</div>
                                                <div class="draft-snippet">{snippet}</div>
                                                <div class="draft-meta">
                                                    <span>{format!("{recipients} recipient(s)")}</span>
                                                    {scheduled.then(|| view! {
                                                        <span class="scheduled-badge">"Scheduled"</span>
                                                    })}
                                                </div>
                                            </div>
                                            <div class="draft-actions">
                                                <a href="/compose" class="btn btn-secondary btn-sm">"Edit"</a>
                                                <button class="btn btn-sm btn-secondary" title="Share draft with trust network">
                                                    "\u{1F517} Share"
                                                </button>
                                                <button class="btn btn-icon danger btn-sm" title="Delete draft">
                                                    "\u{1F5D1}"
                                                </button>
                                            </div>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}
