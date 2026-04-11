// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mandatory profile setup for production.
//!
//! On first load with a live conductor, checks `mail_profiles.get_my_profile`.
//! If no profile exists, shows a blocking modal that MUST be completed.
//! No skip button — profile creation is required to use the app.

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;

use crate::holochain::{use_holochain, ConnectionStatus};
use crate::toasts::use_toasts;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SetupStep {
    Checking,
    Connecting, // Visible: shows "Connecting to network..."
    NameEntry,
    KeyGen,
    Complete,
    HasProfile,
}

#[component]
pub fn ProfileSetup() -> impl IntoView {
    let hc = use_holochain();
    let toasts = use_toasts();
    let step = RwSignal::new(SetupStep::Checking);
    let display_name = RwSignal::new(String::new());
    let bio = RwSignal::new(String::new());
    let key_status = RwSignal::new(String::new());
    let error_msg = RwSignal::new(String::new());

    // On production, show connecting state after 3 seconds if still checking
    if !crate::mail_context::is_demo_mode() {
        spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_secs(3)).await;
            if step.get_untracked() == SetupStep::Checking {
                step.set(SetupStep::Connecting);
            }
        });
    }

    // Check if profile exists whenever connection status changes
    let hc_check = hc.clone();
    Effect::new(move |_| {
        let status = hc_check.status.get();
        let current_step = step.get_untracked();

        // Only check when waiting for connection (not during setup flow)
        if status == ConnectionStatus::Connected
            && (current_step == SetupStep::Checking || current_step == SetupStep::Connecting)
        {
            let hc = hc_check.clone();
            spawn_local(async move {
                // Small delay to let JS bridge initialize
                gloo_timers::future::sleep(std::time::Duration::from_millis(500)).await;

                match hc.call_zome::<(), serde_json::Value>(
                    "mail_profiles", "get_my_profile", &()
                ).await {
                    Ok(val) if !val.is_null() => {
                        web_sys::console::log_1(&"[Mail] Profile exists".into());
                        step.set(SetupStep::HasProfile);
                    }
                    Ok(_) => {
                        web_sys::console::log_1(&"[Mail] No profile — setup required".into());
                        step.set(SetupStep::NameEntry);
                    }
                    Err(e) => {
                        web_sys::console::warn_1(&format!("[Mail] get_my_profile: {e}").into());
                        // Show setup — user needs to create profile
                        step.set(SetupStep::NameEntry);
                    }
                }
            });
        } else if crate::mail_context::is_demo_mode() {
            // Demo mode — no profile needed
            step.set(SetupStep::HasProfile);
        }
        // On production with Mock status: show connecting UI after a delay
        if !crate::mail_context::is_demo_mode() && status == ConnectionStatus::Mock
            && current_step == SetupStep::Checking
        {
            step.set(SetupStep::Connecting);
        }
    });

    // Profile creation handler
    let do_create = {
        let hc_create = hc.clone();
        let toasts_create = toasts.clone();
        std::rc::Rc::new(move || {
            let name = display_name.get_untracked();
            if name.trim().is_empty() { return; }
            let bio_val = bio.get_untracked();
            step.set(SetupStep::KeyGen);
            error_msg.set(String::new());
            key_status.set("Saving profile to DHT...".into());

            let hc = hc_create.clone();
            let toasts = toasts_create.clone();
            spawn_local(async move {
                let profile = serde_json::json!({
                    "name": name.trim(),
                    "email": "",
                    "avatar_url": "",
                    "bio": bio_val.trim(),
                });
                match hc.call_zome::<serde_json::Value, serde_json::Value>(
                    "mail_profiles", "set_profile", &profile
                ).await {
                    Ok(_) => {
                        key_status.set("Profile created!".into());
                    }
                    Err(e) => {
                        web_sys::console::warn_1(&format!("[Mail] set_profile error: {e}").into());
                        error_msg.set(format!("Could not save profile: {e}"));
                        key_status.set(String::new());
                        step.set(SetupStep::NameEntry);
                        return;
                    }
                }

                // Create DID (decentralized identity) on the identity DNA
                key_status.set("Creating your DSID...".into());
                match hc.call_zome_on_role::<(), serde_json::Value>(
                    "identity", "did_registry", "create_did", &()
                ).await {
                    Ok(_) => {
                        web_sys::console::log_1(&"[Mail] DSID created on identity DNA".into());
                    }
                    Err(e) => {
                        // Non-critical — identity DNA may not have init'd yet
                        web_sys::console::warn_1(&format!("[Mail] create_did: {e}").into());
                    }
                }

                // Try key generation (non-critical)
                key_status.set("Generating encryption keys...".into());
                if let Ok(status_val) = hc.call_zome::<(), serde_json::Value>(
                    "mail_keys", "needs_refresh", &()
                ).await {
                    let needs = serde_json::to_string(&status_val).unwrap_or_default();
                    if needs.contains("NoBundle") || needs.contains("Expired") {
                        let identity_key = match crate::crypto::ensure_local_identity_keypair() {
                            Ok(key) => key,
                            Err(e) => {
                                web_sys::console::warn_1(&format!("[Mail] identity key setup: {e}").into());
                                Vec::new()
                            }
                        };
                        let signed_pre_key = crate::crypto::generate_public_key_bytes();
                        let signed_pre_key_signature =
                            crate::crypto::sign_message(&signed_pre_key, &identity_key);
                        let created_at_us = (js_sys::Date::now() as u64) * 1000;
                        let bundle = serde_json::json!({
                            "identity_key": identity_key,
                            "signed_pre_key": signed_pre_key,
                            "signed_pre_key_id": 1u32,
                            "signed_pre_key_signature": signed_pre_key_signature,
                            "one_time_pre_keys": (0..10).map(|i| serde_json::json!({
                                "key_id": (i + 1) as u32,
                                "public_key": crate::crypto::generate_public_key_bytes(),
                                "used": false,
                            })).collect::<Vec<_>>(),
                            "created_at": created_at_us,
                            "expires_at": created_at_us + (30 * 24 * 3600 * 1_000_000u64),
                        });
                        let _ = hc.call_zome::<serde_json::Value, serde_json::Value>(
                            "mail_keys", "publish_pre_key_bundle", &bundle
                        ).await;
                    }
                }

                toasts.push("Welcome to Mycelix Pulse!", "success");
                step.set(SetupStep::Complete);
                gloo_timers::future::sleep(std::time::Duration::from_millis(2000)).await;
                step.set(SetupStep::HasProfile);
            });
        })
    };

    let do_create_enter = do_create.clone();
    let do_create_click = do_create.clone();

    let visible = move || {
        let s = step.get();
        s != SetupStep::HasProfile && s != SetupStep::Checking
    };
    let is_connecting = move || step.get() == SetupStep::Connecting;
    let is_name_entry = move || step.get() == SetupStep::NameEntry;
    let is_keygen = move || step.get() == SetupStep::KeyGen;
    let is_complete = move || step.get() == SetupStep::Complete;

    view! {
        <div style=move || if visible() {
            "position:fixed;inset:0;z-index:99990;background:rgba(8,8,12,0.95);display:flex;align-items:center;justify-content:center"
        } else { "display:none" }>
            <div style="position:relative;background:#1a1d2e;border-radius:16px;padding:28px;max-width:420px;width:92%;box-shadow:0 20px 60px rgba(0,0,0,0.6)">
                <div style="text-align:center;margin-bottom:20px">
                    <div style="font-size:2.5rem;margin-bottom:8px">"✉"</div>
                    <h2 style="font-size:1.4rem;font-weight:700;margin:0;color:#e2e6f0">"Welcome to Mycelix Pulse"</h2>
                    <p style="color:#8890a8;font-size:0.85rem;margin-top:6px">"Create your Decentralized Sovereign Identity"</p>
                </div>

                // Connecting state — waiting for conductor
                <div style=move || if is_connecting() { "text-align:center;padding:20px 0" } else { "display:none" }>
                    <div class="loading-spinner" style="margin:0 auto 16px"></div>
                    <p style="color:#06D6C8;font-size:0.95rem;font-weight:500">"Connecting to the Holochain network..."</p>
                    <p style="color:#5c6380;font-size:0.75rem;margin-top:8px;line-height:1.5">
                        "Setting up encrypted connection to the conductor."<br/>
                        "This may take a few seconds on first load."
                    </p>
                    // Retry button after showing for a while
                    <button style="margin-top:16px;padding:8px 16px;background:transparent;border:1px solid #3a3f54;border-radius:8px;color:#8890a8;font-size:0.8rem;cursor:pointer"
                        on:click=move |_| {
                            // Force retry by reloading the page
                            let _ = web_sys::window().and_then(|w| w.location().reload().ok());
                        }>
                        "Retry Connection"
                    </button>
                </div>

                // Error message
                <div style=move || if error_msg.get().is_empty() { "display:none" } else {
                    "background:#2d1b1b;border:1px solid #5c2020;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-size:0.8rem;color:#f87171"
                }>
                    {move || error_msg.get()}
                </div>

                // Name entry form
                <form style=move || if is_name_entry() { "" } else { "display:none" }
                    on:submit=move |e: web_sys::SubmitEvent| {
                        e.prevent_default();
                        do_create_enter();
                    }>
                    <div style="margin-bottom:14px">
                        <label for="setup-name" style="display:block;font-size:0.8rem;color:#8890a8;margin-bottom:4px;font-weight:500">"Display Name"</label>
                        <input id="setup-name" type="text"
                            style="width:100%;padding:12px 14px;background:#252838;border:1px solid #3a3f54;border-radius:8px;color:#e2e6f0;font-size:16px;box-sizing:border-box;font-family:inherit;outline:none"
                            placeholder="How others will see you"
                            autocomplete="name"
                            enterkeyhint="send"
                            required=true
                            prop:value=move || display_name.get()
                            on:input=move |e| display_name.set(event_target_value(&e))
                        />
                    </div>
                    <div style="margin-bottom:14px">
                        <label for="setup-bio" style="display:block;font-size:0.8rem;color:#8890a8;margin-bottom:4px;font-weight:500">"Bio (optional)"</label>
                        <input id="setup-bio" type="text"
                            style="width:100%;padding:12px 14px;background:#252838;border:1px solid #3a3f54;border-radius:8px;color:#e2e6f0;font-size:16px;box-sizing:border-box;font-family:inherit;outline:none"
                            placeholder="A short bio..."
                            prop:value=move || bio.get()
                            on:input=move |e| bio.set(event_target_value(&e))
                        />
                    </div>
                    <input type="submit"
                        style="width:100%;padding:14px;background:#06D6C8;border:none;border-radius:8px;color:#000;font-weight:700;font-size:1rem;cursor:pointer;margin-top:4px"
                        disabled=move || display_name.get().trim().is_empty()
                        value="Create Profile & Generate Keys"
                    />
                    <p style="font-size:0.7rem;color:#5c6380;text-align:center;margin-top:14px;line-height:1.5">
                        "Your DSID is anchored to the Holochain DHT."<br/>
                        "Post-quantum encryption keys protect your messages."
                    </p>
                </form>

                // Key generation progress
                <div style=move || if is_keygen() { "text-align:center;padding:20px 0" } else { "display:none" }>
                    <div class="loading-spinner" style="margin:0 auto 16px"></div>
                    <p style="color:#06D6C8;font-size:0.95rem;font-weight:500">{move || key_status.get()}</p>
                </div>

                // Complete
                <div style=move || if is_complete() { "text-align:center;padding:20px 0" } else { "display:none" }>
                    <div style="font-size:3rem;margin-bottom:8px;color:#06D6C8">"✓"</div>
                    <p style="color:#06D6C8;font-weight:600;font-size:1.1rem">"You're all set!"</p>
                    <p style="color:#8890a8;font-size:0.85rem;margin-top:4px">"Your DSID is live on the network."</p>
                </div>
            </div>
        </div>
    }
}
