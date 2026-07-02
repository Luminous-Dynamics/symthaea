// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Identity context — provides reactive identity data via Leptos signals.
//!
//! Each data domain loads independently via its own `spawn_local` (no waterfall).
//! Mock data renders immediately; conductor data replaces it asynchronously.
//! Version signals enable resource invalidation after mutations.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use identity_leptos_types::*;

use mycelix_leptos_core::holochain_provider::use_holochain;
use crate::mock_data;

/// Version signals — bumped by actions to trigger data reload.
#[derive(Clone, Copy)]
pub struct IdentityVersions {
    pub did: RwSignal<u32>,
    pub mfa: RwSignal<u32>,
    pub credentials: RwSignal<u32>,
    pub recovery: RwSignal<u32>,
    pub trust: RwSignal<u32>,
}

#[derive(Clone)]
pub struct IdentityCtx {
    pub versions: IdentityVersions,
    pub did_document: RwSignal<Option<DidDocumentView>>,
    pub mfa_state: RwSignal<Option<MfaStateView>>,
    pub recovery_config: RwSignal<Option<RecoveryConfigView>>,
    pub credentials_held: RwSignal<Vec<CredentialView>>,
    pub credentials_issued: RwSignal<Vec<CredentialView>>,
    pub trust_credentials: RwSignal<Vec<TrustCredentialView>>,
    pub reputation: RwSignal<Option<ReputationView>>,
    pub my_name: RwSignal<Option<NameRegistryView>>,
    pub loading: RwSignal<bool>,
}

pub fn provide_identity_context() {
    let versions = IdentityVersions {
        did: RwSignal::new(0),
        mfa: RwSignal::new(0),
        credentials: RwSignal::new(0),
        recovery: RwSignal::new(0),
        trust: RwSignal::new(0),
    };

    // Initialize with mock data immediately (instant render)
    let ctx = IdentityCtx {
        versions,
        did_document: RwSignal::new(Some(mock_data::mock_did_document())),
        mfa_state: RwSignal::new(Some(mock_data::mock_mfa_state())),
        recovery_config: RwSignal::new(Some(mock_data::mock_recovery_config())),
        credentials_held: RwSignal::new(mock_data::mock_credentials_held()),
        credentials_issued: RwSignal::new(mock_data::mock_credentials_issued()),
        trust_credentials: RwSignal::new(mock_data::mock_trust_credentials()),
        reputation: RwSignal::new(Some(mock_data::mock_reputation())),
        my_name: RwSignal::new(Some(mock_data::mock_name())),
        loading: RwSignal::new(true),
    };

    provide_context(ctx.clone());

    // Launch independent async loads — each domain fetches concurrently (no waterfall).
    let ctx_did = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;
        load_did(ctx_did).await;
    });

    let ctx_mfa = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;
        load_mfa(ctx_mfa).await;
    });

    let ctx_cred = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;
        load_credentials(ctx_cred).await;
    });

    let ctx_rep = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;
        load_reputation(ctx_rep).await;
    });

    let ctx_loading = ctx.clone();
    spawn_local(async move {
        gloo_timers::future::sleep(std::time::Duration::from_secs(3)).await;
        ctx_loading.loading.set(false);
    });
}

async fn load_did(ctx: IdentityCtx) {
    let hc = use_holochain();
    if hc.is_mock() { return; }

    match hc.call_zome_default::<(), serde_json::Value>("did_registry", "get_my_did", &()).await {
        Ok(record) => {
            match serde_json::from_value::<DidDocumentView>(record) {
                Ok(did) => {
                    web_sys::console::log_1(&"[Identity] Loaded DID from conductor".into());
                    ctx.did_document.set(Some(did));
                }
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("[Identity] Failed to parse DID record: {e}").into()
                    );
                }
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Identity] get_my_did failed: {e}").into());
        }
    }
}

async fn load_mfa(ctx: IdentityCtx) {
    let hc = use_holochain();
    if hc.is_mock() { return; }

    match hc.call_zome_default::<String, serde_json::Value>(
        "mfa", "get_mfa_state", &"self".to_string()
    ).await {
        Ok(record) => {
            match serde_json::from_value::<MfaStateView>(record) {
                Ok(mfa) => {
                    web_sys::console::log_1(
                        &format!("[Identity] Loaded MFA: {} factors", mfa.factors.len()).into()
                    );
                    ctx.mfa_state.set(Some(mfa));
                }
                Err(e) => {
                    web_sys::console::warn_1(
                        &format!("[Identity] Failed to parse MFA state: {e}").into()
                    );
                }
            }
        }
        Err(e) => {
            web_sys::console::warn_1(&format!("[Identity] get_mfa_state failed: {e}").into());
        }
    }

    // Also load recovery config (same spawn — minimal latency)
    match hc.call_zome_default::<String, serde_json::Value>(
        "recovery", "get_recovery_config", &"self".to_string()
    ).await {
        Ok(record) => {
            if let Ok(config) = serde_json::from_value::<RecoveryConfigView>(record) {
                ctx.recovery_config.set(Some(config));
            }
        }
        Err(_) => {}
    }
}

async fn load_credentials(ctx: IdentityCtx) {
    let hc = use_holochain();
    if hc.is_mock() { return; }

    match hc.call_zome_default::<(), Vec<serde_json::Value>>(
        "verifiable_credential", "get_my_credentials", &()
    ).await {
        Ok(records) => {
            let creds: Vec<CredentialView> = records.iter().filter_map(|r| {
                match serde_json::from_value::<CredentialView>(r.clone()) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        web_sys::console::warn_1(
                            &format!("[Identity] Failed to parse credential: {e}").into()
                        );
                        None
                    }
                }
            }).collect();
            if !creds.is_empty() {
                ctx.credentials_held.set(creds);
            }
        }
        Err(e) => {
            web_sys::console::warn_1(
                &format!("[Identity] get_my_credentials failed: {e}").into()
            );
        }
    }
}

async fn load_reputation(ctx: IdentityCtx) {
    let hc = use_holochain();
    if hc.is_mock() { return; }

    match hc.call_zome_default::<String, serde_json::Value>(
        "reputation_aggregator", "get_composite_reputation", &"self".to_string()
    ).await {
        Ok(record) => {
            if let Ok(rep) = serde_json::from_value::<ReputationView>(record) {
                web_sys::console::log_1(
                    &format!("[Identity] Loaded reputation: {:.2}", rep.composite_score).into()
                );
                ctx.reputation.set(Some(rep));
            }
        }
        Err(_) => {}
    }
}

pub fn use_identity() -> IdentityCtx {
    expect_context::<IdentityCtx>()
}
