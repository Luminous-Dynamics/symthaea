// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hardware Integration Components — WebBluetooth and Physical Presence.

use leptos::prelude::*;
use crate::curriculum::{use_progress, ProgressStatus};

#[component]
pub fn HardwareScanner(
    device_type: String,
    on_linked: impl Fn(String) + 'static,
) -> impl IntoView {
    let (status, set_status) = signal("Ready to Connect".to_string());
    let (is_scanning, set_is_scanning) = signal(false);

    let start_handshake = move |_| {
        set_is_scanning.set(true);
        set_status.set(format!("Scanning for {}...", device_type));
        
        // SIMULATED: WebBluetooth API Handshake
        wasm_bindgen_futures::spawn_local(async move {
            gloo_timers::future::sleep(std::time::Duration::from_millis(1500)).await;
            set_status.set("\u{1F7E2} LINK ESTABLISHED".to_string());
            set_is_scanning.set(false);
            on_linked("hw-grant-99b3".to_string());
        });
    };

    view! {
        <div class="hardware-scanner-ui" style="margin-top: 1rem">
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: var(--surface-low); border-radius: 8px">
                <div style="font-size: 0.8rem">
                    <strong>{device_type}</strong>
                    <div style="color: var(--text-tertiary); font-size: 0.7rem">{move || status.get()}</div>
                </div>
                <button 
                    class="btn-sm btn-primary" 
                    on:click=start_handshake
                    disabled=move || is_scanning.get()
                >
                    "Connect"
                </button>
            </div>
        </div>
    }
}

#[component]
pub fn PresenceValidator(
    node_id: String,
    on_verified: impl Fn() + 'static,
) -> impl IntoView {
    let (show_camera, set_show_camera) = signal(false);

    view! {
        <div class="presence-validator" style="margin-top: 1rem">
            {move || if !show_camera.get() {
                view! {
                    <button class="btn-sm btn-outline" style="width: 100%" on:click=move |_| set_show_camera.set(true)>
                        "\u{1F4F7} Scan Workbench QR"
                    </button>
                }.into_any()
            } else {
                view! {
                    <div class="camera-preview" style="width: 100%; height: 200px; background: #000; border-radius: 8px; display: flex; align-items: center; justify-content: center; position: relative">
                        <span style="color: #fff; font-size: 0.7rem">"Scanning for verification hash..."</span>
                        <button 
                            style="position: absolute; top: 5px; right: 5px; background: none; border: none; color: white; cursor: pointer"
                            on:click=move |_| set_show_camera.set(false)
                        >"\u{00D7}"</button>
                        // Simulated QR Success
                        <div 
                            style="width: 80%; height: 2px; background: var(--accent); position: absolute; animation: scan 2s infinite"
                            on:animationiteration=move |_| {
                                set_show_camera.set(false);
                                on_verified();
                            }
                        ></div>
                    </div>
                }.into_any()
            }}
        </div>
    }
}

#[component]
pub fn SafetyGuard<F, IV>(
    children: F,
) -> impl IntoView 
where 
    F: Fn() -> IV + 'static,
    IV: IntoView + 'static,
{
    let progress = use_progress();
    let is_safe = move || progress.get().get("VOC-000-S").status == ProgressStatus::Mastered;

    view! {
        {move || if is_safe() {
            children().into_any()
        } else {
            view! {
                <div class="safety-lockout" style="padding: 2rem; background: var(--error-low); border: 2px solid var(--error); border-radius: 12px; text-align: center">
                    <div style="font-size: 2rem; margin-bottom: 1rem">"\u{26A0}\u{FE0F}"</div>
                    <h4 style="color: var(--error)">"PHYSICAL LOCKOUT ACTIVE"</h4>
                    <p style="font-size: 0.85rem">
                        "Access to physical hardware requires mastery of the mandatory Safety Protocol."
                    </p>
                    <a href="/study/VOC-000-S" class="btn-sm btn-primary" style="margin-top: 1rem; background: var(--error); border-color: var(--error)">
                        "Master VOC-000-S Now"
                    </a>
                </div>
            }.into_any()
        }}
    }
}
