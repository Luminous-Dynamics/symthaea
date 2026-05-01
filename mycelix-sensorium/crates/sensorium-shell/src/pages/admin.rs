// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admin/IT domain page — system health + RDP viewer stub.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, MessageEvent, WebSocket};

use crate::identity::{ConductorStatus, SensoriumIdentity};

#[component]
pub fn AdminOverview() -> impl IntoView {
    let identity = use_context::<SensoriumIdentity>().expect("SensoriumIdentity");

    let conductor_label = move || match identity.conductor_status.get() {
        ConductorStatus::Connected => "Connected",
        ConductorStatus::Connecting => "Connecting...",
        ConductorStatus::Mock => "Mock Mode",
    };
    let conductor_color = move || match identity.conductor_status.get() {
        ConductorStatus::Connected => "#22c55e",
        ConductorStatus::Connecting => "#f59e0b",
        ConductorStatus::Mock => "#a855f7",
    };

    view! {
        <div class="admin-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Health"</button>
                <button class="domain-nav-btn">"Conductors"</button>
                <button class="domain-nav-btn">"WASM"</button>
                <button class="domain-nav-btn">"Remote"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: var(--text-muted)">"CONDUCTOR"</div>
                    <p class="thought-content" style=move || format!("font-size: 1.2rem; font-weight: 700; color: {}", conductor_color())>
                        {conductor_label}
                    </p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"ws://localhost:8888"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: var(--text-muted)">"PORTAL WASM"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"501"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"KB gzipped (production)"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: var(--text-muted)">"DOMAINS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"10"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Active domain modules"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: var(--text-muted)">"LEPTOS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"0.8"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Framework version"</p>
                </div>
            </div>

            // RDP viewer stub
            <div class="thought-card" style="margin-top: 1rem;">
                <h3 class="section-title">"Remote Desktop (RDP Web Viewer)"</h3>
                <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.6; margin-bottom: 0.5rem;">
                    "View-only remote desktop via WebSocket relay to Symthaea's RDP server. "
                    "Frames rendered as Canvas2D draw commands in the browser."
                </p>
                <RdpViewer />
            </div>

            // Observatory migration note
            <div class="thought-card" style="margin-top: 0.75rem;">
                <h3 class="section-title">"Migration Status"</h3>
                <p style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.5;">
                    "This domain consolidates the SvelteKit Observatory dashboard. "
                    "Conductor health, gate metrics, and data export will be ported "
                    "from mycelix-workspace/observatory/ into this Leptos module."
                </p>
            </div>
        </div>
    }
}

/// RDP web viewer — connects to Symthaea's RDP server via WebSocket
/// and renders frames on a Canvas2D element.
///
/// Architecture:
/// 1. Sensorium opens WebSocket to ws://host:port/rdp
/// 2. Server sends frame updates as binary messages (Canvas2D draw commands)
/// 3. Client decodes and renders on <canvas>
/// 4. View-only — no input forwarding (future: keyboard/mouse events)
#[component]
fn RdpViewer() -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let (connected, set_connected) = signal(false);
    let (status_text, set_status) = signal("Disconnected".to_string());
    let (rdp_url, set_rdp_url) = signal("ws://localhost:3389/rdp".to_string());

    let connect = move |_| {
        let url = rdp_url.get();
        set_status.set(format!("Connecting to {url}..."));

        let Some(canvas_el) = canvas_ref.get() else {
            set_status.set("No canvas element".into());
            return;
        };
        let canvas: HtmlCanvasElement = canvas_el.into();
        canvas.set_width(1024);
        canvas.set_height(768);

        let ctx: CanvasRenderingContext2d = match canvas
            .get_context("2d")
            .ok()
            .flatten()
            .and_then(|c| c.dyn_into::<CanvasRenderingContext2d>().ok())
        {
            Some(c) => c,
            None => {
                set_status.set("Failed to get 2d context".into());
                return;
            }
        };

        ctx.set_fill_style_str("#1a1a2e");
        ctx.fill_rect(0.0, 0.0, 1024.0, 768.0);
        ctx.set_fill_style_str("#94A3B8");
        ctx.set_font("14px monospace");
        let _ = ctx.fill_text(&format!("Connecting to {url}..."), 20.0, 40.0);

        // Try WebSocket connection
        let ws = match WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(e) => {
                set_status.set(format!("WebSocket error: {:?}", e));
                return;
            }
        };
        ws.set_binary_type(web_sys::BinaryType::Arraybuffer);

        // onopen
        let set_connected_clone = set_connected;
        let set_status_clone = set_status;
        let onopen = Closure::<dyn FnMut()>::new(move || {
            set_connected_clone.set(true);
            set_status_clone.set("Connected — streaming frames".into());
        });
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));
        onopen.forget();

        // onmessage — render frame data on canvas
        let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |event: MessageEvent| {
            // Frame data would be decoded here (draw commands, bitmap tiles, etc.)
            // For now, just show that we received data
            let data = event.data();
            if let Ok(buf) = data.dyn_into::<js_sys::ArrayBuffer>() {
                let len = buf.byte_length();
                ctx.set_fill_style_str("#1a1a2e");
                ctx.fill_rect(0.0, 0.0, 1024.0, 30.0);
                ctx.set_fill_style_str("#22c55e");
                ctx.set_font("11px monospace");
                ctx.fill_text(&format!("Frame: {} bytes received", len), 10.0, 20.0)
                    .ok();
            }
        });
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget();

        // onclose
        let set_connected_close = set_connected;
        let set_status_close = set_status;
        let onclose = Closure::<dyn FnMut()>::new(move || {
            set_connected_close.set(false);
            set_status_close.set("Disconnected".into());
        });
        ws.set_onclose(Some(onclose.as_ref().unchecked_ref()));
        onclose.forget();

        // onerror
        let set_status_err = set_status;
        let onerror = Closure::<dyn FnMut()>::new(move || {
            set_status_err.set("Connection failed — RDP server not running".into());
        });
        ws.set_onerror(Some(onerror.as_ref().unchecked_ref()));
        onerror.forget();
    };

    view! {
        <div class="rdp-viewer">
            <div class="form-row" style="margin-bottom: 0.5rem;">
                <input
                    type="text"
                    class="form-input"
                    style="flex: 1;"
                    prop:value=move || rdp_url.get()
                    on:input=move |ev| set_rdp_url.set(event_target_value(&ev))
                />
                <button class="form-submit" on:click=connect>
                    {move || if connected.get() { "Reconnect" } else { "Connect" }}
                </button>
            </div>
            <p style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.5rem;">
                {move || status_text.get()}
            </p>
            <canvas
                node_ref=canvas_ref
                width="1024" height="768"
                style="width: 100%; max-width: 1024px; border-radius: 8px; border: 1px solid var(--border); background: #1a1a2e;"
            />
        </div>
    }
}
