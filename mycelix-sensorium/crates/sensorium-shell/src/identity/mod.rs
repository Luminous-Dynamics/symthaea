// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified Mycelix identity — one First Breath, domain keys derived lazily.

#[allow(dead_code)]
pub mod master_key;

use leptos::prelude::*;
use send_wrapper::SendWrapper;
use sensorium_domain_trait::CivicTier;
use std::cell::RefCell;
use std::rc::Rc;

use mycelix_leptos_client::{
    decode, encode, BrowserWsTransport, ConnectConfig, HolochainTransport,
};

/// Vault state across the entire Sensorium shell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VaultState {
    NoVault,
    Locked,
    Unlocked,
}

/// Shared Holochain transport wrapped in SendWrapper for Leptos context bounds.
/// SAFETY: WASM is single-threaded.
pub type TransportCell = SendWrapper<Rc<RefCell<Option<BrowserWsTransport>>>>;

/// The Sensorium-wide identity context.
#[derive(Clone)]
#[allow(dead_code)]
pub struct SensoriumIdentity {
    /// DID string (e.g., "did:mycelix:uhCAk...")
    pub did: RwSignal<Option<String>>,
    /// Vault state
    pub vault: RwSignal<VaultState>,
    /// Consciousness profile combined score (0.0-1.0)
    pub consciousness_score: RwSignal<f64>,
    /// Derived tier
    pub tier: Memo<CivicTier>,
    /// Which domain keys have been derived this session
    pub active_domains: RwSignal<Vec<String>>,
    /// Holochain conductor connection status
    pub conductor_status: RwSignal<ConductorStatus>,
    /// Shared transport — `None` when in mock mode
    pub transport: TransportCell,
}

/// Conductor connection status.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConductorStatus {
    Connecting,
    Connected,
    Mock,
}

impl SensoriumIdentity {
    pub fn new() -> Self {
        let consciousness_score = RwSignal::new(0.35); // Default: Participant
        let tier = Memo::new(move |_| CivicTier::from_score(consciousness_score.get()));

        let vault = if master_key::has_stored_master() {
            VaultState::Locked
        } else {
            VaultState::NoVault
        };

        let transport: TransportCell = SendWrapper::new(Rc::new(RefCell::new(None)));
        let conductor_status = RwSignal::new(ConductorStatus::Connecting);

        // Attempt conductor connection asynchronously
        let transport_for_connect = transport.clone();
        let status_signal = conductor_status;
        wasm_bindgen_futures::spawn_local(async move {
            let url = web_sys::window()
                .and_then(|w| {
                    js_sys::Reflect::get(&w, &wasm_bindgen::JsValue::from_str("__HC_CONDUCTOR_URL"))
                        .ok()
                        .and_then(|v| v.as_string())
                })
                .unwrap_or_else(|| "ws://localhost:8888".to_string());

            web_sys::console::log_1(
                &format!("[Sensorium] Connecting to conductor at {url}...").into(),
            );

            let ws_transport = BrowserWsTransport::new();
            // The sandbox may install with different app IDs.
            // Try the configured name, fall back to common alternatives.
            let app_id = web_sys::window()
                .and_then(|w| {
                    js_sys::Reflect::get(&w, &wasm_bindgen::JsValue::from_str("__HC_APP_ID"))
                        .ok()
                        .and_then(|v| v.as_string())
                })
                .unwrap_or_else(|| "mycelix-unified".to_string());

            let auth_token = web_sys::window().and_then(|w| {
                js_sys::Reflect::get(&w, &wasm_bindgen::JsValue::from_str("__HC_AUTH_TOKEN"))
                    .ok()
                    .and_then(|v| v.as_string())
            });

            let config = ConnectConfig {
                url,
                app_id,
                auth_token: auth_token.map(|s| s.into_bytes()),
                reconnect: None,
                request_timeout_ms: Some(30_000),
            };

            match ws_transport.connect(config).await {
                Ok(()) => {
                    web_sys::console::log_1(&"[Sensorium] Conductor connected!".into());
                    *transport_for_connect.borrow_mut() = Some(ws_transport);
                    status_signal.set(ConductorStatus::Connected);

                    // Fetch live consciousness score from reputation aggregator.
                    // Falls back to the default 0.35 if the zome call fails.
                    let transport_for_score = transport_for_connect.clone();
                    let score_signal = consciousness_score;
                    wasm_bindgen_futures::spawn_local(async move {
                        use mycelix_leptos_client::{decode, encode};
                        #[derive(serde::Deserialize)]
                        struct CompositeScore {
                            composite_score: f64,
                        }

                        let payload = match encode(&"self".to_string()) {
                            Ok(p) => p,
                            Err(_) => return,
                        };
                        let guard = transport_for_score.borrow();
                        if let Some(t) = guard.as_ref() {
                            if let Ok(bytes) = t
                                .call_zome(
                                    "identity",
                                    "reputation_aggregator",
                                    "get_composite_reputation",
                                    payload,
                                )
                                .await
                            {
                                // Try structured response first, then scalar fallback
                                if let Ok(resp) = decode::<CompositeScore>(&bytes) {
                                    score_signal.set(resp.composite_score.clamp(0.0, 1.0));
                                } else if let Ok(score) = decode::<f64>(&bytes) {
                                    score_signal.set(score.clamp(0.0, 1.0));
                                }
                            }
                        }
                    });
                }
                Err(e) => {
                    web_sys::console::log_1(
                        &format!("[Sensorium] No conductor: {e}. Mock mode.").into(),
                    );
                    status_signal.set(ConductorStatus::Mock);
                }
            }
        });

        Self {
            did: RwSignal::new(None),
            vault: RwSignal::new(vault),
            consciousness_score,
            tier,
            active_domains: RwSignal::new(vec![]),
            conductor_status,
            transport,
        }
    }

    /// Call a zome function on any cluster via the shared transport.
    /// Returns `Err` in mock mode so callers can fall back to mock data.
    pub async fn call_zome<I: serde::Serialize, O: serde::de::DeserializeOwned>(
        &self,
        role: &str,
        zome: &str,
        fn_name: &str,
        input: &I,
    ) -> Result<O, String> {
        let transport = {
            let guard = self.transport.borrow();
            match guard.as_ref() {
                Some(t) => t.clone(),
                None => return Err(format!("Mock mode: {role}.{zome}.{fn_name}")),
            }
        };

        let payload = encode(input).map_err(|e| format!("Encode: {e}"))?;
        let bytes = transport
            .call_zome(role, zome, fn_name, payload)
            .await
            .map_err(|e| format!("{role}.{zome}.{fn_name}: {e}"))?;
        decode(&bytes).map_err(|e| format!("Decode: {e}"))
    }

    /// Derive a domain-specific key from the master.
    #[allow(dead_code)]
    /// Returns the 32-byte key if the vault is unlocked.
    pub fn domain_key(&self, domain_context: &[u8]) -> Option<[u8; 32]> {
        if self.vault.get() != VaultState::Unlocked {
            return None;
        }
        // In a real implementation, the master key would be held in memory
        // and domain keys derived via HKDF on demand.
        // For now, return a deterministic derivation placeholder.
        Some(master_key::derive_domain_key(domain_context))
    }
}
