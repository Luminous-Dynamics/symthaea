// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain context for Craft — conductor-first pattern.
//!
//! Provides reactive signals that load real data from the Holochain
//! conductor as soon as it connects. Falls back gracefully to empty
//! state when no conductor is available.

use leptos::prelude::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types (match zome entry types for deserialization)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CraftProfile {
    pub display_name: String,
    pub headline: String,
    pub bio: String,
    pub location: Option<String>,
    pub website: Option<String>,
    pub avatar_url: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PublishedCredentialView {
    pub credential_id: String,
    pub title: String,
    pub issuer: String,
    pub vitality_permille: u16,
    pub mastery_permille: u16,
    pub guild_name: Option<String>,
    pub epistemic_code: Option<String>,
    pub needs_review: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ConnectionView {
    pub display_name: String,
    pub headline: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GuildMembershipView {
    pub guild_name: String,
    pub role: String,
    pub domain: String,
}

// ---------------------------------------------------------------------------
// Context state
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct CraftCtx {
    pub profile: RwSignal<Option<CraftProfile>>,
    pub credentials: RwSignal<Vec<PublishedCredentialView>>,
    pub connections: RwSignal<Vec<ConnectionView>>,
    pub guilds: RwSignal<Vec<GuildMembershipView>>,
    pub endorsement_count: RwSignal<u32>,
    pub loading: RwSignal<bool>,
}

pub fn use_craft() -> CraftCtx {
    expect_context::<CraftCtx>()
}

/// Provide the Craft domain context. Call once at app root (inside HolochainProvider).
///
/// Loads real data from the conductor immediately when connected.
/// Falls back to empty state when no conductor is available.
pub fn provide_craft_context() {
    let ctx = CraftCtx {
        profile: RwSignal::new(None),
        credentials: RwSignal::new(vec![]),
        connections: RwSignal::new(vec![]),
        guilds: RwSignal::new(vec![]),
        endorsement_count: RwSignal::new(0),
        loading: RwSignal::new(false),
    };

    provide_context(ctx.clone());

    // Load data once conductor connects (reactive — triggers on status change)
    let hc = use_holochain();
    Effect::new(move |_| {
        let status = hc.status.get();
        if status != mycelix_leptos_core::holochain_provider::ConnectionStatus::Connected {
            return;
        }

        let hc = hc.clone();
        let ctx = ctx.clone();
        wasm_bindgen_futures::spawn_local(async move {
            ctx.loading.set(true);

            // Load profile
            if let Ok(profile) = hc
                .call_zome_default::<(), Option<CraftProfile>>("craft_graph", "get_my_profile", &())
                .await
            {
                ctx.profile.set(profile);
            }

            // Load credentials
            if let Ok(creds) = hc
                .call_zome_default::<(), Vec<PublishedCredentialView>>(
                    "craft_graph",
                    "list_my_published_credentials",
                    &(),
                )
                .await
            {
                ctx.credentials.set(creds);
            }

            // Load guild memberships
            if let Ok(guilds) = hc
                .call_zome_default::<(), Vec<GuildMembershipView>>(
                    "guild_coordinator",
                    "get_my_guilds",
                    &(),
                )
                .await
            {
                ctx.guilds.set(guilds);
            }

            ctx.loading.set(false);
        });
    });
}
