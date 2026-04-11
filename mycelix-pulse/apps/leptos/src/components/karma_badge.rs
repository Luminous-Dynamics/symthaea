// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Karma badge — displays composite trust+karma score.

use leptos::prelude::*;
use crate::karma::{use_karma, composite_score};
use crate::mail_context::use_mail;

#[component]
pub fn KarmaBadge(sender: String) -> impl IntoView {
    let karma_ctx = use_karma();
    let kc2 = karma_ctx.clone();
    let kc3 = karma_ctx.clone();
    let mail = use_mail();
    let sender1 = sender.clone();
    let sender2 = sender.clone();
    let sender3 = sender.clone();

    let score = move || {
        let trust = mail.sender_trust.get().get(&sender1).copied().unwrap_or(0.5);
        let karma = karma_ctx.get_karma(&sender1);
        composite_score(trust, karma)
    };

    let score_display = move || format!("{:.0}", {
        let trust = mail.sender_trust.get().get(&sender2).copied().unwrap_or(0.5);
        let karma = kc2.get_karma(&sender2);
        composite_score(trust, karma) * 100.0
    });

    let score_class = move || {
        let trust = mail.sender_trust.get().get(&sender3).copied().unwrap_or(0.5);
        let karma = kc3.get_karma(&sender3);
        let s = composite_score(trust, karma);
        if s >= 0.7 { "karma-badge karma-high" }
        else if s >= 0.4 { "karma-badge karma-medium" }
        else if s >= 0.2 { "karma-badge karma-low" }
        else { "karma-badge karma-negative" }
    };

    view! {
        <span class=score_class title="Karma score">
            {score_display}
        </span>
    }
}
