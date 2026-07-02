// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Real-time signal subscription from Holochain conductor.
//!
//! When connected, subscribes to HearthSignal events via the conductor's
//! WebSocket signal channel. Each signal is pushed into the toast system
//! and updates the reactive state.
//!
//! When in mock mode, simulated_life.rs handles this instead.

use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::use_toasts;
use crate::hearth_context::use_hearth;

/// Start listening for real-time signals from the conductor.
/// Only activates when connected (not mock).
pub fn start_signal_listener() {
    let hc = use_holochain();

    if hc.is_mock() {
        // Mock mode: simulated_life handles events
        return;
    }

    let _toasts = use_toasts();
    let _hearth = use_hearth();

    // TODO: Wire real signal subscription when conductor connection
    // supports signal callbacks. The BrowserWsTransport needs a
    // register_signal_handler(callback) method.
    //
    // When a HearthSignal arrives:
    // - BondTended: update bonds signal, push toast
    // - GratitudeExpressed: update gratitude signal, push toast
    // - PresenceChanged: update presence signal, push toast
    // - VoteCast/DecisionFinalized: update decisions/votes, push toast
    // - EmergencyAlert: push urgent toast
    //
    // For now, this is a stub. The architecture is ready — the
    // transport layer just needs the signal callback hook.

    web_sys::console::log_1(
        &"[Hearth] Signal listener ready (waiting for conductor signals)".into(),
    );
}
