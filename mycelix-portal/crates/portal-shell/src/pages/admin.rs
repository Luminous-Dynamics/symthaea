// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admin/IT domain page — system health + RDP viewer stub.

use leptos::prelude::*;
use crate::identity::{ConductorStatus, PortalIdentity};

#[component]
pub fn AdminOverview() -> impl IntoView {
    let identity = use_context::<PortalIdentity>().expect("PortalIdentity");

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
                <div style="background: rgba(0,0,0,0.3); border-radius: 8px; padding: 2rem; text-align: center; border: 1px dashed var(--border);">
                    <p style="color: var(--text-muted); font-size: 0.75rem;">"RDP canvas — connect to ws://localhost:3389/rdp to stream"</p>
                </div>
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
