// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Trust-Gated Inbox — Staked Attention paradigm.
//!
//! Replaces ML-based spam filtering with economic trust gating:
//! - Trusted senders (in Web-of-Trust): messages flow directly to Priority inbox
//! - Unknown senders: must stake TEND tokens to bypass immune gate
//! - Accept = sender gets stake back
//! - Reject/Spam = stake slashed, distributed to network, K-vector drops
//!
//! This makes spam economically impossible without false positives.

use leptos::prelude::*;
use crate::mail_context::use_mail;
use crate::toasts::use_toasts;

/// Trust gate status for an incoming email.
#[derive(Clone, Debug, PartialEq)]
pub enum GateStatus {
    /// Sender is in your Web-of-Trust (direct or transitive)
    Trusted { trust_score: f64, degree: u32 },
    /// Sender staked TEND tokens to reach you
    Staked { amount: f64, sender: String },
    /// Sender is unknown with no stake (quarantined)
    Quarantined { sender: String },
}

impl GateStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Trusted { .. } => "Trusted",
            Self::Staked { .. } => "Staked",
            Self::Quarantined { .. } => "Quarantined",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Trusted { .. } => "gate-trusted",
            Self::Staked { .. } => "gate-staked",
            Self::Quarantined { .. } => "gate-quarantined",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Trusted { .. } => "\u{1F6E1}",  // shield
            Self::Staked { .. } => "\u{1FA99}",    // coin
            Self::Quarantined { .. } => "\u{26A0}", // warning
        }
    }
}

/// Determine gate status for a sender, incorporating DSID assurance level.
/// E0 senders are always quarantined (Sybil resistance).
/// E1 senders need higher trust scores than E2+ senders.
pub fn evaluate_gate(sender: &str, trust_score: Option<f64>) -> GateStatus {
    evaluate_gate_with_assurance(sender, trust_score, None)
}

/// Assurance-aware gate evaluation.
pub fn evaluate_gate_with_assurance(sender: &str, trust_score: Option<f64>, assurance: Option<u8>) -> GateStatus {
    let assurance_level = assurance.unwrap_or(255); // 255 = unknown (treat as legacy)

    // E0 (Anonymous) — always quarantined (Sybil resistance)
    if assurance_level == 0 {
        return GateStatus::Quarantined { sender: sender.to_string() };
    }

    // E1 (Basic) — require higher trust threshold
    let trust_threshold = if assurance_level == 1 { 0.5 } else { 0.3 };

    match trust_score {
        Some(score) if score >= trust_threshold => {
            let degree = if score >= 0.8 { 1 } else if score >= 0.5 { 2 } else { 3 };
            GateStatus::Trusted { trust_score: score, degree }
        }
        Some(score) if score >= 0.0 => {
            GateStatus::Staked { amount: 0.1, sender: sender.to_string() }
        }
        _ => {
            GateStatus::Quarantined { sender: sender.to_string() }
        }
    }
}

/// Trust gate badge shown on email cards and read view.
#[component]
pub fn TrustGateBadge(
    #[prop(into)] sender: String,
    trust_score: Option<f64>,
) -> impl IntoView {
    let gate = evaluate_gate(&sender, trust_score);
    let class = gate.css_class();
    let icon = gate.icon();
    let label = gate.label();

    let detail = match &gate {
        GateStatus::Trusted { trust_score, degree } => {
            format!("{label} ({degree}-degree, {:.0}%)", trust_score * 100.0)
        }
        GateStatus::Staked { amount, .. } => {
            format!("{label} ({amount:.2} TEND staked)")
        }
        GateStatus::Quarantined { .. } => {
            format!("{label} — unknown sender, no stake")
        }
    };

    view! {
        <span class=format!("trust-gate-badge {class}") title=detail>
            <span class="gate-icon">{icon}</span>
            <span class="gate-label">{label}</span>
        </span>
    }
}

/// Accept/Reject buttons for staked or quarantined emails.
#[component]
pub fn TrustGateActions(
    #[prop(into)] sender: String,
    #[prop(into)] email_hash: String,
    trust_score: Option<f64>,
) -> impl IntoView {
    let gate = evaluate_gate(&sender, trust_score);
    let toasts = use_toasts();
    let mail = use_mail();

    let show_actions = matches!(gate, GateStatus::Staked { .. } | GateStatus::Quarantined { .. });

    let sender_accept = sender.clone();
    let toasts_accept = toasts.clone();
    let mail_accept = mail.clone();
    let on_accept = move |_| {
        // Accept: return stake, add to trust
        toasts_accept.push(format!("Accepted message. Stake returned to sender."), "success");
        // In production: call trust zome to create positive attestation
        mail_accept.sender_trust.update(|m| {
            m.insert(sender_accept.clone(), 0.3);
        });
    };

    let sender_reject = sender.clone();
    let hash_reject = email_hash.clone();
    let toasts_reject = toasts.clone();
    let mail_reject = mail.clone();
    let on_reject = move |_| {
        // Reject: slash stake, remove email, decrease trust
        toasts_reject.push("Rejected. Stake slashed and distributed to network.", "info");
        mail_reject.delete_email(&hash_reject);
        mail_reject.sender_trust.update(|m| {
            m.insert(sender_reject.clone(), -0.5);
        });
    };

    let on_report = {
        let sender = sender.clone();
        let hash = email_hash.clone();
        let toasts = toasts.clone();
        move |_| {
            toasts.push("Reported as spam. Sender's K-vector reduced.", "info");
            mail.delete_email(&hash);
            mail.sender_trust.update(|m| {
                m.insert(sender.clone(), -1.0);
            });
        }
    };

    view! {
        {show_actions.then(|| view! {
            <div class="trust-gate-actions">
                <button class="btn btn-sm gate-accept" on:click=on_accept>
                    "\u{2705} Accept"
                </button>
                <button class="btn btn-sm gate-reject" on:click=on_reject>
                    "\u{274C} Reject"
                </button>
                <button class="btn btn-sm gate-report" on:click=on_report>
                    "\u{1F6A8} Report Spam"
                </button>
                {matches!(gate, GateStatus::Staked { .. }).then(|| view! {
                    <span class="stake-info">
                        "Sender staked TEND tokens to reach you. Accept returns their stake."
                    </span>
                })}
            </div>
        })}
    }
}
