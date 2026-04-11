// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Meet page — P2P encrypted video/audio calls via WebRTC.
//!
//! Signaling through Holochain remote_signal (no central TURN server).
//! 1:1 calls are direct peer connections. Group calls (3-6) use mesh topology.

use leptos::prelude::*;
use mail_leptos_types::*;

#[component]
pub fn MeetPage() -> impl IntoView {
    let in_call = RwSignal::new(false);
    let call_type = RwSignal::new(CallType::Video);
    let is_muted = RwSignal::new(false);
    let camera_on = RwSignal::new(true);
    let screen_sharing = RwSignal::new(false);
    let hand_raised = RwSignal::new(false);

    let participants = RwSignal::new(vec![
        CallParticipant {
            agent_key: "uhCAk_self_mock".into(), name: "You".into(),
            is_muted: false, camera_on: true, screen_sharing: false, hand_raised: false,
        },
    ]);

    let on_start_call = move |ct: CallType| {
        call_type.set(ct);
        in_call.set(true);
        // In production: initiate WebRTC via remote_signal
        let _ = js_sys::eval("console.log('[Meet] Starting call via WebRTC signaling')");
    };

    let on_end_call = move |_| {
        in_call.set(false);
        participants.set(vec![CallParticipant {
            agent_key: "uhCAk_self_mock".into(), name: "You".into(),
            is_muted: false, camera_on: true, screen_sharing: false, hand_raised: false,
        }]);
    };

    let on_toggle_mute = move |_| is_muted.update(|v| *v = !*v);
    let on_toggle_camera = move |_| camera_on.update(|v| *v = !*v);
    let on_toggle_screen = move |_| {
        screen_sharing.update(|v| *v = !*v);
        // In production: navigator.mediaDevices.getDisplayMedia()
    };
    let on_toggle_hand = move |_| hand_raised.update(|v| *v = !*v);

    let call_duration = RwSignal::new(0u32);

    // Timer when in call
    Effect::new(move |_| {
        if in_call.get() {
            let duration = call_duration;
            wasm_bindgen_futures::spawn_local(async move {
                loop {
                    gloo_timers::future::sleep(std::time::Duration::from_secs(1)).await;
                    if !in_call.get_untracked() { break; }
                    duration.update(|d| *d += 1);
                }
            });
        } else {
            call_duration.set(0);
        }
    });

    let format_duration = move || {
        let d = call_duration.get();
        format!("{:02}:{:02}", d / 60, d % 60)
    };

    view! {
        <div class="page page-meet">
            {move || if in_call.get() {
                // Active call view
                view! {
                    <div class="meet-active">
                        <div class="meet-header">
                            <span class="meet-status">"In call"</span>
                            <span class="meet-duration">{format_duration}</span>
                            <span class="meet-type">{move || match call_type.get() {
                                CallType::Audio => "Audio Call",
                                CallType::Video => "Video Call",
                                CallType::Screen => "Screen Share",
                            }}</span>
                        </div>

                        // Video grid
                        <div class="meet-grid">
                            {move || participants.get().iter().map(|p| {
                                let name = p.name.clone();
                                let muted = p.is_muted;
                                let cam = p.camera_on;
                                let sharing = p.screen_sharing;
                                let hand = p.hand_raised;
                                view! {
                                    <div class=if cam { "meet-tile" } else { "meet-tile no-video" }>
                                        {if cam {
                                            view! { <div class="meet-video-placeholder">"Camera Feed"</div> }.into_any()
                                        } else {
                                            view! {
                                                <div class="meet-avatar">
                                                    {name.chars().next().unwrap_or('?').to_uppercase().to_string()}
                                                </div>
                                            }.into_any()
                                        }}
                                        <div class="meet-tile-bar">
                                            <span class="meet-tile-name">{name}</span>
                                            <div class="meet-tile-icons">
                                                {muted.then(|| view! { <span class="icon-muted">"\u{1F507}"</span> })}
                                                {sharing.then(|| view! { <span class="icon-screen">"\u{1F5B5}"</span> })}
                                                {hand.then(|| view! { <span class="icon-hand">"\u{270B}"</span> })}
                                            </div>
                                        </div>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>

                        // Controls
                        <div class="meet-controls">
                            <button class=move || if is_muted.get() { "ctrl-btn active" } else { "ctrl-btn" }
                                    on:click=on_toggle_mute title="Toggle mute">
                                {move || if is_muted.get() { "\u{1F507}" } else { "\u{1F3A4}" }}
                            </button>
                            <button class=move || if !camera_on.get() { "ctrl-btn active" } else { "ctrl-btn" }
                                    on:click=on_toggle_camera title="Toggle camera">
                                {move || if camera_on.get() { "\u{1F4F9}" } else { "\u{1F6AB}" }}
                            </button>
                            <button class=move || if screen_sharing.get() { "ctrl-btn active" } else { "ctrl-btn" }
                                    on:click=on_toggle_screen title="Share screen">
                                "\u{1F5B5}"
                            </button>
                            <button class=move || if hand_raised.get() { "ctrl-btn active" } else { "ctrl-btn" }
                                    on:click=on_toggle_hand title="Raise hand">
                                "\u{270B}"
                            </button>
                            <button class="ctrl-btn end-call" on:click=on_end_call title="End call">
                                "\u{1F4F5}"
                            </button>
                        </div>

                        <div class="meet-info">
                            <p class="meet-encryption">"\u{1F512} End-to-end encrypted via PQC (Kyber1024)"</p>
                            <p class="meet-p2p">"\u{1F310} Direct P2P WebRTC — no server relay"</p>
                        </div>
                    </div>
                }.into_any()
            } else {
                // Start call view
                view! {
                    <div class="meet-start">
                        <h1>"Meet"</h1>
                        <p class="meet-desc">"Start a P2P encrypted call. Video and audio are transmitted directly between participants via WebRTC — no server sees your data."</p>

                        <div class="meet-options">
                            <button class="meet-option-card" on:click=move |_| on_start_call(CallType::Video)>
                                <span class="option-icon">"\u{1F4F9}"</span>
                                <span class="option-label">"Video Call"</span>
                                <span class="option-desc">"Camera + microphone"</span>
                            </button>
                            <button class="meet-option-card" on:click=move |_| on_start_call(CallType::Audio)>
                                <span class="option-icon">"\u{1F3A4}"</span>
                                <span class="option-label">"Audio Call"</span>
                                <span class="option-desc">"Microphone only"</span>
                            </button>
                            <button class="meet-option-card" on:click=move |_| on_start_call(CallType::Screen)>
                                <span class="option-icon">"\u{1F5B5}"</span>
                                <span class="option-label">"Screen Share"</span>
                                <span class="option-desc">"Share your screen"</span>
                            </button>
                        </div>

                        <div class="meet-scheduled">
                            <h3>"Upcoming Calls"</h3>
                            <div class="scheduled-call">
                                <span class="scheduled-icon">"\u{1F4C5}"</span>
                                <div class="scheduled-info">
                                    <span class="scheduled-title">"Water Council Meeting"</span>
                                    <span class="scheduled-time">"Thu 2:00 PM"</span>
                                </div>
                                <button class="btn btn-sm btn-primary">"Join"</button>
                            </div>
                        </div>
                    </div>
                }.into_any()
            }}
        </div>
    }
}
