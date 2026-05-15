// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Personal context.
//!
//! This scaffold keeps the data boundary explicit: the current Personal zomes
//! largely return raw Holochain `Record` values rather than dedicated
//! frontend-facing views. The app therefore renders stable typed mock data now
//! and centralizes future live adapters here.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use mycelix_leptos_core::holochain_provider::{use_holochain, HolochainCtx};
use personal_leptos_types::{
    ActivityItemView, BiometricView, ConsentGrantView, DataSharingPreferenceView, HealthRecordView,
    MasterKeyView, PreferenceChangeLogView, ProfileView, StoredCredentialView,
};

use crate::mock_data;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolRegistry {
    pub hearth_alias: String,  // e.g. "Campfire", "Well", "Plaza"
    pub mycel_alias: String,   // e.g. "Spark", "Ember", "Leaf"
    pub genesis_alias: String, // e.g. "The Ignition", "Sun-Rise"
    pub orientation: String,   // Cultural context description
}

impl Default for SymbolRegistry {
    fn default() -> Self {
        Self {
            hearth_alias: "HEARTH".into(),
            mycel_alias: "MYCEL".into(),
            genesis_alias: "Thermodynamic Genesis".into(),
            orientation: "Canonical Mycelix terminology".into(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct CulturalContext {
    pub symbols: RwSignal<SymbolRegistry>,
}

pub fn provide_cultural_context() {
    let symbols = RwSignal::new(SymbolRegistry::default());
    provide_context(CulturalContext { symbols });
}

pub fn use_cultural() -> CulturalContext {
    use_context::<CulturalContext>().expect("CulturalContext not provided")
}

#[derive(Clone)]
pub struct PersonalCtx {
    pub profile: RwSignal<ProfileView>,
    pub draft_profile: RwSignal<ProfileView>,
    pub keys: RwSignal<Vec<MasterKeyView>>,
    pub credentials: RwSignal<Vec<StoredCredentialView>>,
    pub biometrics: RwSignal<Vec<BiometricView>>,
    pub consents: RwSignal<Vec<ConsentGrantView>>,
    pub preferences: RwSignal<Vec<DataSharingPreferenceView>>,
    pub preference_log: RwSignal<Vec<PreferenceChangeLogView>>,
    pub health_record_count: RwSignal<usize>,
    pub activity: RwSignal<Vec<ActivityItemView>>,
    pub loading: RwSignal<bool>,
    pub live_sync_ready: RwSignal<bool>,
    pub status_note: RwSignal<String>,
}

pub fn provide_personal_context() {
    let profile = mock_data::mock_profile();
    let ctx = PersonalCtx {
        draft_profile: RwSignal::new(profile.clone()),
        profile: RwSignal::new(profile),
        keys: RwSignal::new(mock_data::mock_keys()),
        credentials: RwSignal::new(mock_data::mock_credentials()),
        biometrics: RwSignal::new(mock_data::mock_biometrics()),
        consents: RwSignal::new(mock_data::mock_consents()),
        preferences: RwSignal::new(mock_data::mock_preferences()),
        preference_log: RwSignal::new(mock_data::mock_preference_log()),
        health_record_count: RwSignal::new(mock_data::mock_health_records()),
        activity: RwSignal::new(mock_data::mock_activity()),
        loading: RwSignal::new(true),
        live_sync_ready: RwSignal::new(false),
        status_note: RwSignal::new(
            "Using stable local vault models while Personal zome view adapters are still being standardized.".into(),
        ),
    };

    provide_context(ctx.clone());

    spawn_local(async move {
        gloo_timers::future::sleep(std::time::Duration::from_millis(300)).await;
        let hc = use_holochain();

        if hc.is_mock() {
            ctx.status_note.set(
                "Running in mock mode. The vault shell is ready; conductor-backed Personal adapters come next."
                    .into(),
            );
        } else {
            let mut loaded_any = false;

            if let Ok(Some(profile)) = hc
                .call_zome_default::<(), Option<ProfileView>>(
                    "identity_vault",
                    "get_my_profile_view",
                    &(),
                )
                .await
            {
                ctx.profile.set(profile.clone());
                ctx.draft_profile.set(profile);
                loaded_any = true;
            }

            if let Ok(keys) = hc
                .call_zome_default::<(), Vec<MasterKeyView>>(
                    "identity_vault",
                    "get_my_keys_view",
                    &(),
                )
                .await
            {
                if !keys.is_empty() {
                    ctx.keys.set(keys);
                    loaded_any = true;
                }
            }

            if let Ok(credentials) = hc
                .call_zome_default::<(), Vec<StoredCredentialView>>(
                    "credential_wallet",
                    "get_my_credentials_view",
                    &(),
                )
                .await
            {
                if !credentials.is_empty() {
                    ctx.credentials.set(credentials);
                    loaded_any = true;
                }
            }

            if let Ok(biometrics) = hc
                .call_zome_default::<(), Vec<BiometricView>>(
                    "health_vault",
                    "get_my_biometrics_view",
                    &(),
                )
                .await
            {
                if !biometrics.is_empty() {
                    ctx.biometrics.set(biometrics);
                    loaded_any = true;
                }
            }

            if let Ok(consents) = hc
                .call_zome_default::<(), Vec<ConsentGrantView>>(
                    "health_vault",
                    "get_my_consents_view",
                    &(),
                )
                .await
            {
                if !consents.is_empty() {
                    ctx.consents.set(consents);
                    loaded_any = true;
                }
            }

            if let Ok(preferences) = hc
                .call_zome_default::<(), Vec<DataSharingPreferenceView>>(
                    "data_preferences",
                    "get_my_preferences_view",
                    &(),
                )
                .await
            {
                if !preferences.is_empty() {
                    ctx.preferences.set(preferences);
                    loaded_any = true;
                }
            }

            if let Ok(log) = hc
                .call_zome_default::<(), Vec<PreferenceChangeLogView>>(
                    "data_preferences",
                    "get_change_log_view",
                    &(),
                )
                .await
            {
                if !log.is_empty() {
                    ctx.preference_log.set(log);
                    loaded_any = true;
                }
            }

            if let Ok(records) = hc
                .call_zome_default::<(), Vec<HealthRecordView>>(
                    "health_vault",
                    "get_my_records_view",
                    &(),
                )
                .await
            {
                if !records.is_empty() {
                    ctx.health_record_count.set(records.len());
                    loaded_any = true;
                }
            }

            if let Ok(activity) = hc
                .call_zome_default::<(), Vec<ActivityItemView>>(
                    "personal_bridge",
                    "get_recent_activity_view",
                    &(),
                )
                .await
            {
                if !activity.is_empty() {
                    ctx.activity.set(activity);
                    loaded_any = true;
                }
            }

            if loaded_any {
                ctx.live_sync_ready.set(true);
                ctx.status_note.set(
                    "Connected to a live conductor. Personal profile, wallet, and health summary are now loading through typed view endpoints."
                        .into(),
                );
            } else {
                ctx.status_note.set(
                    "Connected to a live conductor, but Personal view endpoints returned no records yet. The vault shell remains available."
                        .into(),
                );
            }
        }

        ctx.loading.set(false);
    });
}

pub fn use_personal() -> PersonalCtx {
    expect_context::<PersonalCtx>()
}

pub async fn refresh_identity_state(ctx: PersonalCtx, hc: HolochainCtx) {
    if let Ok(Some(profile)) = hc
        .call_zome_default::<(), Option<ProfileView>>("identity_vault", "get_my_profile_view", &())
        .await
    {
        ctx.profile.set(profile.clone());
        ctx.draft_profile.set(profile);
    }

    if let Ok(keys) = hc
        .call_zome_default::<(), Vec<MasterKeyView>>("identity_vault", "get_my_keys_view", &())
        .await
    {
        if !keys.is_empty() {
            ctx.keys.set(keys);
        }
    }
}

pub async fn refresh_preferences_state(ctx: PersonalCtx, hc: HolochainCtx) {
    if let Ok(preferences) = hc
        .call_zome_default::<(), Vec<DataSharingPreferenceView>>(
            "data_preferences",
            "get_my_preferences_view",
            &(),
        )
        .await
    {
        if !preferences.is_empty() {
            ctx.preferences.set(preferences);
        }
    }

    if let Ok(log) = hc
        .call_zome_default::<(), Vec<PreferenceChangeLogView>>(
            "data_preferences",
            "get_change_log_view",
            &(),
        )
        .await
    {
        if !log.is_empty() {
            ctx.preference_log.set(log);
        }
    }
}

pub async fn refresh_health_state(ctx: PersonalCtx, hc: HolochainCtx) {
    if let Ok(biometrics) = hc
        .call_zome_default::<(), Vec<BiometricView>>("health_vault", "get_my_biometrics_view", &())
        .await
    {
        if !biometrics.is_empty() {
            ctx.biometrics.set(biometrics);
        }
    }

    if let Ok(consents) = hc
        .call_zome_default::<(), Vec<ConsentGrantView>>("health_vault", "get_my_consents_view", &())
        .await
    {
        if !consents.is_empty() {
            ctx.consents.set(consents);
        }
    }

    if let Ok(records) = hc
        .call_zome_default::<(), Vec<HealthRecordView>>("health_vault", "get_my_records_view", &())
        .await
    {
        if !records.is_empty() {
            ctx.health_record_count.set(records.len());
        }
    }
}
