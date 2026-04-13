// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Holon RDP Viewer — egui desktop client for the Phase I.A binary wire.
//!
//! **Status**: Phase I.A.2 scaffold (2026-04-13). Compiles as a stub; the
//! three coupled pieces (egui window, tokio-tungstenite client, PQC
//! handshake) are TODOs that must land together in one focused session.
//!
//! ## The three pieces that must land together
//!
//! Phase I.A.2 is NOT three independent sub-tasks. These three pieces are
//! tightly coupled and will fail in confusing ways if split:
//!
//! 1. **egui window + TextureHandle blit.** Owns the `FrameBuffer` from
//!    `swarm::rdp_holon_bridge::HolonRdpViewer`, repaints on frame arrival,
//!    forwards mouse/keyboard events to the reverse path.
//!
//! 2. **`tokio-tungstenite` client.** Connects to `ws://localhost:7778/holon/ws`
//!    (the existing Holon WebSocket), receives binary `Message::Binary`
//!    frames which are sealed `FrameBin` envelopes, forwards them to the
//!    `open_frame` → `HolonRdpViewer.apply_frame` chain. On the reverse
//!    path, takes the egui mouse/keyboard → `InputFrame` → `seal_input` →
//!    `Message::Binary`.
//!
//! 3. **PQC handshake (Phase I.A.5 Track 2.5 — DEFERRED).** Currently
//!    `RdpSession` needs a 32-byte session key injected via
//!    `on_handshake_complete([u8; 32])`. The real handshake goes through
//!    `src/swarm/service.rs::run_handshake_for_peer` which has a TODO at
//!    line 844 (`TODO(blocked:bidirectional-kem)`). The viewer scaffold
//!    uses a placeholder fixed key to unblock the egui+WS work; the real
//!    PQC handshake lands as the third piece once the other two compile.
//!
//! Why they can't split:
//! - The egui window needs a live `Arc<Mutex<HolonRdpViewer>>` that the WS
//!   client updates. Split them and you can't test either.
//! - The WS client needs a sealed envelope to open. Without the handshake
//!   producing a real key, you can only test with placeholder keys — which
//!   the previous Phase I.A.5 work already did.
//! - The handshake state machine is two-sided. Testing it needs real
//!   initiator + responder both running, which is what the WS client
//!   supplies.
//!
//! ## Usage (planned)
//!
//! ```bash
//! # Start the Holon WS server (phone/soma side sealing frames)
//! cargo run --release --bin symthaea-holon --features api_module,mesh-encryption,phone
//!
//! # In another terminal, start the viewer (desktop side)
//! cargo run --release --example holon_rdp_viewer --features holon-viewer
//! ```
//!
//! ## References
//!
//! - Phase I.A delivery: `docs/phase_1a_verification.md` (commit c15995b3ff)
//! - Phase I.A.5 hardening: Tracks 2.1–3.2 all committed to main
//! - Plan: `plans/shiny-wibbling-quail.md` Phase I.A.2

#[cfg(not(feature = "holon-viewer"))]
fn main() {
    eprintln!(
        "Requires: --features holon-viewer\n\
         \n\
         This example needs the egui window, tokio-tungstenite client,\n\
         and the mesh-encryption + api_module features. The `holon-viewer`\n\
         feature pulls all of them together.\n"
    );
    std::process::exit(1);
}

#[cfg(feature = "holon-viewer")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ═══════════════════════════════════════════════════════════════════
    // PIECE 3 — PQC handshake (skeleton only; real flow lands with Track 2.5)
    // ═══════════════════════════════════════════════════════════════════
    //
    // TODO(phase-1a-2): replace the placeholder key with a real
    // PqcSessionKey derived via `src/swarm/service.rs::run_handshake_for_peer`
    // once the bidirectional KEM exchange at service.rs:844 is unblocked.
    //
    // For the scaffold, the viewer and server must agree on a placeholder
    // key out-of-band. This is NOT SAFE for any real network deployment —
    // it only unblocks the egui + WS plumbing work.
    let _placeholder_session_key: [u8; 32] = [0x42; 32];

    // ═══════════════════════════════════════════════════════════════════
    // PIECE 1 — egui window (skeleton only)
    // ═══════════════════════════════════════════════════════════════════
    //
    // TODO(phase-1a-2):
    //   - Construct an `eframe::NativeOptions` with initial window size
    //     matching the phone's native screen (default 1008×2244).
    //   - Implement an `eframe::App` that owns:
    //       * `Arc<Mutex<HolonRdpViewer>>` — from `swarm::rdp_holon_bridge`
    //       * An `egui::TextureHandle` for the FrameBuffer blit
    //       * A tokio runtime handle for the WS client task
    //   - On `update()`:
    //       * Drain any newly-applied frames from the viewer
    //       * Upload the FrameBuffer contents to the texture
    //       * Show the texture with `ui.image()`
    //       * Handle `egui::Event::PointerButton` and forward as
    //         `InputEvent::Pointer` to the reverse-path sender
    //       * Handle `egui::Event::Key` and forward as `InputEvent::Key`
    //
    // Design reference: `crates/symthaea-muse/` uses egui and can be
    // consulted for the NativeOptions + App pattern that already works
    // on NixOS/Wayland.

    // ═══════════════════════════════════════════════════════════════════
    // PIECE 2 — tokio-tungstenite WebSocket client (skeleton only)
    // ═══════════════════════════════════════════════════════════════════
    //
    // TODO(phase-1a-2):
    //   - Spawn a tokio task that connects to `ws://localhost:7778/holon/ws`
    //     via `tokio_tungstenite::connect_async`.
    //   - On `Message::Binary` receive:
    //       * Pass to `rdp_wire::open_frame(&bytes, &mut receiver_session)`
    //       * Feed the resulting `RdpFrame` to `viewer.apply_frame(&frame)`
    //       * Signal repaint to the egui event loop
    //   - On reverse-path (viewer → server):
    //       * Take `InputFrame` from the egui event handler
    //       * Call `rdp_wire::seal_input(&input, &mut sender_session)`
    //       * Send as `Message::Binary` over the WS
    //
    // Note: the tokio runtime and the egui event loop must communicate
    // via a bounded channel (tokio::sync::mpsc::channel). The egui side
    // runs synchronously on the main thread; the tokio side runs on
    // background workers. Do not block the egui event loop on WS I/O.

    eprintln!(
        "Phase I.A.2 scaffold — holon_rdp_viewer is a skeleton.\n\
         \n\
         The three coupled pieces (egui window + WS client + PQC handshake)\n\
         are not yet implemented. This stub compiles under the `holon-viewer`\n\
         feature but does nothing at runtime. See the TODOs in this file\n\
         for the next-session execution plan.\n\
         \n\
         Placeholder session key prepared: {} bytes (unused)\n",
        _placeholder_session_key.len()
    );
    Ok(())
}
