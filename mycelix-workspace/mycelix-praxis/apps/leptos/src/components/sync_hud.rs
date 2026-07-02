// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sync HUD — Ambient UI for Mesh Connection and Mutation Queue Status.

use leptos::prelude::*;
use crate::persistence::{use_mutation_queue, SyncState};

#[component]
pub fn SyncStatusHud() -> impl IntoView {
    let queue = use_mutation_queue();
    
    let (state_label, color, icon, is_pulsing) = move || {
        let q = queue.get();
        match q.state {
            SyncState::Synchronized => ("Mesh Synchronized", "var(--success)", "\u{1F7E2}", false),
            SyncState::LocalOrbit => {
                let count = q.pending_actions.len();
                (format!("Local Orbit ({} Cached)", count), "var(--warning)", "\u{1F7E0}", false)
            },
            SyncState::GossipSyncing => ("Gossip Syncing...", "var(--info)", "\u{1F7E6}", true),
        }
    };

    view! {
        <div class="sync-hud" style="display: flex; align-items: center; gap: 0.5rem; padding: 0.25rem 0.75rem; background: var(--surface-high); border-radius: 20px; border: 1px solid var(--border)">
            <span 
                class=move || if is_pulsing() { "sync-icon pulse" } else { "sync-icon" }
                style=move || format!("color: {}", color())
            >
                {move || icon()}
            </span>
            <span style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px">
                {move || state_label()}
            </span>
        </div>
    }
}
