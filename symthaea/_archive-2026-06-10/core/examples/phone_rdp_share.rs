// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Soma↔Holon RDP Bridge — share the Pixel's screen through the consciousness bridge.
//!
//! Wires the existing `PhoneBridge` (ADB capture) into `SomaRdpServer` (HDC delta
//! codec + consciousness-gated frame envelope), and accepts `InputFrame`s from a
//! notional Holon viewer, dispatching them back through `PhoneAction` → ADB.
//!
//! This is the MVP end-to-end path:
//!
//! ```text
//! Pixel screen ──ADB──▶ PhoneBridge.capture_and_observe_rgba
//!                              │              │
//!                              ▼              ▼
//!                        VisionManifold   SomaRdpServer
//!                        (semantic)       (HDC delta frames)
//!                                              │
//!                                              ▼
//!                                    drain_frames() → Vec<RdpFrame>
//!                                    (would be pushed on Holon WS)
//!
//! Holon viewer ──InputFrame──▶ handle_input_frame() ──▶ PhoneAction ──▶ ADB
//! ```
//!
//! ## Why this matters
//!
//! This replaces the raw ADB path with a consciousness-gated channel:
//! - **HDC delta compression**: only tiles that change are re-encoded (~10-100x bw)
//! - **Consciousness level in envelope**: each frame carries the current Phi
//! - **PQC-ready**: `HolonRdpMessage` serializes through the existing Holon WS,
//!   which already runs ML-KEM + ChaCha20-Poly1305 in sovereign-profile
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example phone_rdp_share \
//!   --features vision-manifold,phone \
//!   -- --duration 15 --fps 4
//! ```
//!
//! The example does NOT open a real WebSocket in this MVP — it prints the
//! envelope that would be sent. Wiring to the Holon WS transport is the next step.

#[cfg(not(feature = "vision-manifold"))]
fn main() {
    eprintln!("Requires: --features vision-manifold,phone");
}

#[cfg(feature = "vision-manifold")]
fn main() {
    use std::time::{Duration, Instant};
    use symthaea::swarm::rdp_holon_bridge::SomaRdpServer;
    use symthaea::swarm::rdp_protocol::{InputEvent, InputFrame, RdpFrame, RdpSessionConfig};
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::seal_frame;
    use symthaea_phone_embodiment::PhoneBridge;

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let duration_s: u32 = args
        .iter()
        .position(|a| a == "--duration")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let fps: u8 = args
        .iter()
        .position(|a| a == "--fps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  Symthaea — Soma↔Holon RDP Bridge                   ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    println!("Duration:  {duration_s}s");
    println!("Target FPS: {fps}");
    println!();

    let mut phone = PhoneBridge::with_resolution("41201FDJG000UM", 1008, 2244, 128, 128);
    if !phone.adb().is_connected() {
        eprintln!("ERROR: Pixel not connected via ADB.");
        std::process::exit(1);
    }
    println!("[OK] Pixel connected via ADB");

    // Soma RDP server — feeds the consciousness-gated frame codec.
    // We tick it once per loop iteration, so cycle_hz = fps (1 tick per frame).
    let mut soma_rdp = SomaRdpServer::new(1008, 2244, fps, fps as u32);
    soma_rdp.start();
    println!(
        "[OK] SomaRdpServer started ({}×{} @ {}fps)",
        1008, 2244, fps
    );

    // RdpSession with the placeholder [0x42; 32] key matching the
    // holon_rdp_viewer example. Phase I.A.5 Track 2.5 (real PQC KEM
    // handshake) remains deferred; for this localhost bandwidth
    // measurement, a fixed shared secret is sufficient.
    let mut session = RdpSession::new(
        "phone-rdp-share".into(),
        "holon-viewer".into(),
        RdpSessionConfig::default(),
        true, // is_initiator
    );
    session.on_connected();
    session.on_handshake_complete([0x42; 32]);
    println!("[OK] RdpSession handshake installed (placeholder key)\n");

    // Fixed moderate Phi for the demo (Green tier → full control on replay path).
    let phi: f32 = 0.65;

    let total_frames = duration_s * fps as u32;
    let frame_interval = Duration::from_millis(1000 / fps as u64);
    let start = Instant::now();

    // Track both sealed-binary and JSON-proxy sizes side-by-side so the
    // comparison is visible in a single run. Closes Task #12 and produces
    // the first real-hardware measurement of the bandwidth ratio.
    let mut bytes_full_sealed = 0usize;
    let mut bytes_delta_sealed = 0usize;
    let mut bytes_full_json = 0usize;
    let mut bytes_delta_json = 0usize;
    let mut full_count = 0usize;
    let mut delta_count = 0usize;
    let mut total_patches = 0usize;

    for frame_idx in 0..total_frames {
        let tick_start = Instant::now();
        let dt = 1.0 / fps as f32;

        // 1. Capture native RGBA + observe through manifold (one round-trip).
        let (tel, rgba, w, h) = match phone.capture_and_observe_rgba(dt) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  [ERR] capture: {e}");
                continue;
            }
        };

        // 2. Feed RDP server (native resolution).
        soma_rdp.tick(&rgba, w, h, phi);

        // 3. Drain any frames ready for transport. For each frame:
        //    - Seal it via rdp_wire::seal_frame (the REAL wire path)
        //    - Also serialize to JSON for comparison (legacy baseline)
        //    - Report both byte counts so the ratio is visible per-frame
        let frames = soma_rdp.drain_frames();
        for frame in frames {
            // The real wire path: bincode + ChaCha20-Poly1305 via rdp_wire.
            let sealed = match seal_frame(&frame, &mut session) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  [ERR] seal_frame: {e}");
                    continue;
                }
            };
            let sealed_bytes = sealed.len();

            // The legacy JSON measurement proxy, for side-by-side comparison.
            // This is what the original MVP reported as "envelope size" —
            // we keep it visible so the 2.998× ratio is verifiable on the
            // real Pixel screen content, not just synthetic test data.
            let json_bytes = serde_json::to_vec(&frame).map(|v| v.len()).unwrap_or(0);
            let ratio = if sealed_bytes > 0 {
                json_bytes as f64 / sealed_bytes as f64
            } else {
                0.0
            };

            match &frame {
                RdpFrame::Full(f) => {
                    full_count += 1;
                    bytes_full_sealed += sealed_bytes;
                    bytes_full_json += json_bytes;
                    total_patches += f.patches.len();
                    println!(
                        "  t={:5.1}s  FULL   frame={} patches={} sealed={}B json={}B ratio={:.2}×  PE={:.3}",
                        start.elapsed().as_secs_f32(),
                        f.frame_id,
                        f.patches.len(),
                        sealed_bytes,
                        json_bytes,
                        ratio,
                        tel.prediction_error,
                    );
                }
                RdpFrame::Delta(d) => {
                    delta_count += 1;
                    bytes_delta_sealed += sealed_bytes;
                    bytes_delta_json += json_bytes;
                    total_patches += d.patches.len();
                    println!(
                        "  t={:5.1}s  DELTA  frame={} changed={}  sealed={}B json={}B ratio={:.2}×  PE={:.3}",
                        start.elapsed().as_secs_f32(),
                        d.frame_id,
                        d.patches.len(),
                        sealed_bytes,
                        json_bytes,
                        ratio,
                        tel.prediction_error,
                    );
                }
                _ => {}
            }
        }

        // Simulate an inbound InputFrame from a Holon viewer at frame 6 (tap center).
        if frame_idx == 6 {
            let simulated = InputFrame {
                sequence: 1,
                timestamp_ms: 0,
                events: vec![InputEvent::Pointer {
                    x: 0.5,
                    y: 0.5,
                    button: 1,
                    pressed: true,
                }],
            };
            println!("\n  [INPUT] Simulated viewer tap at (0.5, 0.5) — dispatching via ADB");
            if let Some(action) = input_frame_to_action(&simulated, w, h) {
                match phone.execute_action(&action) {
                    Ok(()) => println!("  [INPUT] Executed: {}", action.label()),
                    Err(e) => println!("  [INPUT] ERR: {e}"),
                }
            }
            println!();
        }

        // Pace.
        let elapsed = tick_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    let watch_time = start.elapsed().as_secs_f64();

    let total_sealed = bytes_full_sealed + bytes_delta_sealed;
    let total_json = bytes_full_json + bytes_delta_json;
    let total_ratio = if total_sealed > 0 {
        total_json as f64 / total_sealed as f64
    } else {
        0.0
    };

    println!();
    println!("╔══════════════════════════════════════════════════════╗");
    println!("║             Soma RDP Session Summary                 ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();
    println!("Wall time:       {:.1}s", watch_time);
    println!(
        "Full frames:     {full_count} (sealed: {bytes_full_sealed} B, json: {bytes_full_json} B)"
    );
    println!(
        "Delta frames:    {delta_count} (sealed: {bytes_delta_sealed} B, json: {bytes_delta_json} B)"
    );
    println!("Total patches:   {total_patches}");
    if delta_count > 0 {
        println!(
            "Avg delta size:  {:.0} B/frame sealed, {:.0} B/frame json",
            bytes_delta_sealed as f64 / delta_count as f64,
            bytes_delta_json as f64 / delta_count as f64,
        );
    }
    if full_count + delta_count > 0 {
        let sealed_bw = total_sealed as f64 / 1024.0 / watch_time.max(1e-6);
        let json_bw = total_json as f64 / 1024.0 / watch_time.max(1e-6);
        println!();
        println!("─── Bandwidth (real Pixel screen content) ───");
        println!("Sealed (binary wire):  {:.1} KB/s", sealed_bw);
        println!("JSON (legacy proxy):   {:.1} KB/s", json_bw);
        println!("Ratio (json/sealed):   {total_ratio:.3}×");
        println!();
        println!("The sealed number is the REAL bandwidth the wire would produce");
        println!("in a live deployment. The JSON number is kept for comparison");
        println!("against the original MVP baseline (670 KB/s @ 4fps).");
    }
    println!();
    println!("[Done] The phone's screen streamed through the Soma codec.");
    println!("       Task #12 closed: real-hardware bandwidth measured.");
}

/// Translate an inbound RDP `InputFrame` from a viewer into a `PhoneAction`.
///
/// The viewer sends normalized coordinates (0.0-1.0). We denormalize to the
/// phone's native screen resolution and pick the first actionable event.
#[cfg(feature = "vision-manifold")]
fn input_frame_to_action(
    frame: &symthaea::swarm::rdp_protocol::InputFrame,
    screen_w: u32,
    screen_h: u32,
) -> Option<symthaea_phone_embodiment::PhoneAction> {
    use symthaea::swarm::rdp_protocol::InputEvent;
    use symthaea_phone_embodiment::PhoneAction;

    for ev in &frame.events {
        match ev {
            InputEvent::Pointer { x, y, pressed, .. } if *pressed => {
                let sx = (*x * screen_w as f32) as u32;
                let sy = (*y * screen_h as f32) as u32;
                return Some(PhoneAction::Tap { x: sx, y: sy });
            }
            InputEvent::Touch { x, y, phase: 0, .. } => {
                let sx = (*x * screen_w as f32) as u32;
                let sy = (*y * screen_h as f32) as u32;
                return Some(PhoneAction::Tap { x: sx, y: sy });
            }
            _ => continue,
        }
    }
    None
}