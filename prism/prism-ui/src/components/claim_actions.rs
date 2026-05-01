// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Claim action buttons — verify, challenge, share to DHT.
//!
//! When a user clicks "Share to DHT", the claim is published to the
//! Mycelix knowledge DHT via Holochain. This is when Prism stops being
//! a local tool and becomes a network instrument.

use crate::holochain::DhtClaim;
use leptos::prelude::*;
use prism_common::SearchResult;

#[component]
pub fn ClaimActions(result: SearchResult) -> impl IntoView {
    let (verified, set_verified) = signal(false);
    let (share_status, set_share_status) = signal("idle".to_string());

    let content = result.content.clone();
    let e_level = result.empirical_level;
    let n_level = result.normative_level;
    let m_level = result.materiality_level;
    let sources = result.sources.clone();
    let tags = result.tags.clone();

    let on_verify = move |_| {
        set_verified.set(true);
        // Personal verification: the user confirms they consider this claim accurate.
        // This is a local endorsement — in future, verified claims will carry
        // higher trust weight in the epistemic ranking.
        log::info!(
            "Claim personally verified: {}",
            &content[..content.len().min(50)]
        );
    };

    let content_share = result.content.clone();
    let sources_share = sources.clone();
    let tags_share = tags.clone();

    let on_share = move |_| {
        let claim = DhtClaim {
            content: content_share.clone(),
            empirical: e_level,
            normative: n_level,
            materiality: m_level,
            sources: sources_share.clone(),
            tags: tags_share.clone(),
            claim_type: "Fact".to_string(),
            confidence: 0.9,
        };

        set_share_status.set("sharing".to_string());

        wasm_bindgen_futures::spawn_local(async move {
            match crate::holochain::publish_claim(&claim).await {
                Ok(_hash) => set_share_status.set("shared".to_string()),
                Err(msg) => {
                    log::warn!("DHT share: {}", msg);
                    set_share_status.set("local".to_string());
                }
            }
        });
    };

    let share_btn_class = move || match share_status.get().as_str() {
        "shared" => "claim-btn shared",
        "local" => "claim-btn shared",
        "sharing" => "claim-btn",
        _ => "claim-btn",
    };

    let share_btn_text = move || match share_status.get().as_str() {
        "shared" => "On DHT ✓",
        "local" => "Saved locally",
        "sharing" => "Sharing...",
        _ => "Share to DHT",
    };

    let share_disabled = move || share_status.get() != "idle";

    view! {
        <div class="claim-actions">
            <button
                class=move || if verified.get() { "claim-btn verified" } else { "claim-btn" }
                on:click=on_verify
                disabled=move || verified.get()
            >
                {move || if verified.get() { "Endorsed \u{2713}" } else { "Endorse" }}
            </button>

            <button
                class=share_btn_class
                on:click=on_share
                disabled=share_disabled
            >
                {share_btn_text}
            </button>
        </div>
    }
}
