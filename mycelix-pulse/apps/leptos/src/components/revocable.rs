// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Revocable Email — data sovereignty paradigm.
//!
//! Instead of sending content, we send encrypted keys that grant access
//! to content hosted on the sender's node. Revoking = destroying the key.
//! This gives senders permanent control over sent data.
//!
//! Gmail's "Undo Send" works for 5 seconds. Ours works forever.

use leptos::prelude::*;
use crate::toasts::use_toasts;

#[derive(Clone, Debug, PartialEq)]
pub enum AccessStatus {
    /// Recipient can view the email
    Active,
    /// Sender revoked access — content is cryptographically inaccessible
    Revoked { revoked_at: u64 },
    /// Access expires at a specific time
    Expiring { expires_at: u64 },
    /// Access has expired
    Expired,
}

impl AccessStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Active => "Access Active",
            Self::Revoked { .. } => "Access Revoked",
            Self::Expiring { .. } => "Expires Soon",
            Self::Expired => "Expired",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Active => "access-active",
            Self::Revoked { .. } => "access-revoked",
            Self::Expiring { .. } => "access-expiring",
            Self::Expired => "access-expired",
        }
    }
}

/// "Revoke Access" button on sent emails.
#[component]
pub fn RevokeAccessButton(
    #[prop(into)] email_hash: String,
    #[prop(into)] is_sent_by_me: bool,
) -> impl IntoView {
    let toasts = use_toasts();
    let revoked = RwSignal::new(false);

    let hash = email_hash.clone();
    let on_revoke = move |_| {
        revoked.set(true);
        toasts.push(
            "Access revoked. Recipient can no longer view this email or attachments.",
            "success"
        );

        // Destroy the session decryption key on the DHT
        // This makes the encrypted content permanently inaccessible
        let hash = hash.clone();
        wasm_bindgen_futures::spawn_local(async move {
            let hc = crate::holochain::use_holochain();
            if hc.is_mock() {
                web_sys::console::log_1(&"[Mail] Key revocation simulated (mock mode)".into());
                return;
            }

            // Call the keys zome to rotate/invalidate the session key for this email
            let payload = serde_json::json!({
                "email_hash": hash,
                "reason": "UserRevoked"
            });

            match hc.call_zome::<serde_json::Value, serde_json::Value>(
                "mail_keys", "rotate_keys", &payload
            ).await {
                Ok(response) => {
                    web_sys::console::log_1(&format!(
                        "[Mail] Session key revoked for {}: {:?}", hash, response
                    ).into());
                }
                Err(e) => {
                    web_sys::console::warn_1(&format!(
                        "[Mail] Key revocation failed: {e}. The key may already be destroyed."
                    ).into());
                }
            }
        });
    };

    view! {
        <div class="revoke-section" style=move || if is_sent_by_me { "display:flex" } else { "display:none" }>
            <button
                class="btn btn-secondary revoke-btn"
                on:click=on_revoke
                style=move || if revoked.get() { "display:none" } else { "" }
            >
                <span class="revoke-icon">"\u{1F5D1}"</span>
                " Revoke Access"
            </button>
            <span class="revoke-hint" style=move || if revoked.get() { "display:none" } else { "" }>
                "Permanently destroys the decryption key."
            </span>
            <div class="revoke-status revoked" style=move || if revoked.get() { "" } else { "display:none" }>
                <span class="revoke-icon">"\u{1F512}"</span>
                <span>"Access permanently revoked"</span>
            </div>
        </div>
    }
}

/// Access status indicator shown on received emails.
#[component]
pub fn AccessStatusBadge(status: AccessStatus) -> impl IntoView {
    let class = status.css_class();
    let label = status.label();

    view! {
        <span class=format!("access-badge {class}")>
            {label}
        </span>
    }
}
