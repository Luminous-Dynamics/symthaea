// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mail_leptos_types::CryptoSuiteView;

#[component]
pub fn EncryptionBadge(crypto: CryptoSuiteView) -> impl IntoView {
    let is_pqc = crypto.is_post_quantum();
    let label = crypto.short_label();
    let title = format!(
        "Key exchange: {}\nEncryption: {}\nSignature: {}",
        crypto.key_exchange, crypto.symmetric, crypto.signature
    );

    let class = if is_pqc {
        "encryption-badge pqc"
    } else {
        "encryption-badge e2e"
    };

    view! {
        <span class=class title=title>
            <span class="lock-icon">{if is_pqc { "\u{1F6E1}" } else { "\u{1F512}" }}</span>
            <span class="crypto-label">{label}</span>
        </span>
    }
}
