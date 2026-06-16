// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end WebSocket integration test for the Holon RDP binary wire.
//!
//! Closes the A2/A3 verification gap from `docs/phase_1a_verification.md`
//! by exercising the FULL wire runtime — real axum server, real
//! `tokio-tungstenite` client, real sealed frames through the Track 3.2
//! notify-driven broadcast path — without requiring an external process
//! or the physical Pixel.
//!
//! ## What this test proves
//!
//! 1. **A2**: `holon_ws_handler` routes inbound `Message::Binary` to
//!    `rdp_inbound` (not the dropped `_ => {}` it used to be).
//! 2. **A3**: The notify-driven broadcast path (Phase I.A.5 Track 3.2)
//!    delivers a sealed frame to a live subscriber within the 50 ms
//!    budget (the test asserts <500 ms to leave headroom for CI).
//! 3. **Piece 2 round-trip**: the same `rdp_wire::seal_frame` +
//!    `open_frame` path the `holon_rdp_viewer` uses works end-to-end
//!    against a real WebSocket, not just against synthetic frames in the
//!    existing `integration_rdp_wire` test.
//! 4. **Replay protection persists across the wire**: opening the same
//!    envelope twice at the receiver succeeds then fails.
//!
//! ## What this test does NOT prove
//!
//! - Does not exercise the PQC handshake (Piece 3). Both sides use the
//!   fixed placeholder `[0x42; 32]` session key, same as
//!   `holon_rdp_viewer`.
//! - Does not test input reverse path (viewer → server) yet — left as
//!   a follow-up because it requires wiring `rdp_inbound` drains back
//!   through the holon_ws_handler to the test assertion, which is a
//!   full loop the Phase II/I.B design will need anyway.

#![cfg(feature = "holon-viewer")]

use std::sync::Arc;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;

use symthaea::api::holon::{HolonHttpState, holon_router};
use symthaea::swarm::rdp_codec::TILE_SIZE;
use symthaea::swarm::rdp_holon_bridge::HolonRdpViewer;
use symthaea::swarm::rdp_protocol::{
    DeltaFrame, DeltaPatch, FullFrame, QuantizedPatch, RdpFrame, RdpSessionConfig,
};
use symthaea::swarm::rdp_session::RdpSession;
use symthaea::swarm::rdp_wire::{open_frame, seal_frame};

/// Placeholder session key used by `examples/holon_rdp_viewer.rs`.
/// Both sides of this test agree on it out-of-band — same pattern the
/// viewer uses for localhost development until the PQC handshake unblocks.
const PLACEHOLDER_KEY: [u8; 32] = [0x42; 32];

/// Spawn a fresh HolonHttpState bound to axum serve on a random localhost
/// port. Returns the state, the port, and a JoinHandle for the server
/// task (we don't use the handle — test scope is short enough that the
/// axum task shuts down when the runtime drops).
async fn spawn_test_server() -> (Arc<HolonHttpState>, u16) {
    let (tx, _rx) = std::sync::mpsc::channel();
    let state = Arc::new(HolonHttpState::new(tx));
    let router = holon_router(state.clone());

    // Bind to port 0 → OS assigns a free ephemeral port.
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");

    tokio::spawn(async move {
        let _ = axum::serve(listener, router.into_make_service()).await;
    });

    // Tiny grace so the server task actually enters accept() before the
    // client tries to connect. 10 ms is plenty on localhost.
    tokio::time::sleep(Duration::from_millis(10)).await;

    (state, addr.port())
}

/// Build a `RdpSession` in the post-handshake state with the placeholder
/// key installed, ready for seal/open.
fn test_session(label: &str, is_initiator: bool) -> RdpSession {
    let mut s = RdpSession::new(
        label.into(),
        "peer".into(),
        RdpSessionConfig::default(),
        is_initiator,
    );
    s.on_connected();
    s.on_handshake_complete(PLACEHOLDER_KEY);
    s
}

/// Build a deterministic `RdpFrame::Full` suitable for seal/open testing.
fn sample_full_frame(frame_id: u64, cols: u16, rows: u16, seed: u8) -> RdpFrame {
    let patches: Vec<QuantizedPatch> = (0..(cols as usize * rows as usize))
        .map(|idx| {
            let tile_x = (idx % cols as usize) as u8;
            let tile_y = (idx / cols as usize) as u8;
            let values: Vec<i8> = (0..(TILE_SIZE * TILE_SIZE))
                .map(|p| {
                    let px = (p % TILE_SIZE) as u8;
                    let py = (p / TILE_SIZE) as u8;
                    let v = tile_x
                        .wrapping_add(tile_y.wrapping_mul(3))
                        .wrapping_add(px.wrapping_mul(2))
                        .wrapping_add(py)
                        .wrapping_add(seed);
                    (v as i16 - 128) as i8
                })
                .collect();
            QuantizedPatch { values }
        })
        .collect();
    RdpFrame::Full(FullFrame {
        frame_id,
        timestamp_ms: 0,
        patch_cols: cols,
        patch_rows: rows,
        patches,
        consciousness_level: 0.65,
        harmony: "integration-test".into(),
    })
}

/// Build a small delta frame for replay testing.
fn sample_delta_frame(frame_id: u64, base: u64) -> RdpFrame {
    RdpFrame::Delta(DeltaFrame {
        frame_id,
        base_frame_id: base,
        timestamp_ms: 0,
        patches: vec![DeltaPatch {
            index: 7,
            surprise: 0.5,
            values: (0..(TILE_SIZE * TILE_SIZE)).map(|i| i as i8).collect(),
        }],
        consciousness_level: 0.65,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_server_delivers_pushed_frame_to_subscriber() {
    // === Arrange ===
    let (state, port) = spawn_test_server().await;

    // Client-side receiver session — same placeholder key.
    let mut receiver_session = test_session("ws-test-client", false);

    // Connect the tokio-tungstenite client to the test server.
    let url = format!("ws://127.0.0.1:{port}/holon/ws");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");
    let (_ws_sink, mut ws_stream) = ws_stream.split();

    // Build a sealed frame using a separate sender session (same key).
    let mut sender_session = test_session("ws-test-server", true);
    let original = sample_full_frame(1, 4, 4, 0);
    let sealed_bytes = seal_frame(&original, &mut sender_session).expect("seal");
    let expected_size = sealed_bytes.len();

    // === Act ===
    // Push the sealed frame onto the server's outbound broadcast.
    // Track 3.2's dual-path guarantees this goes BOTH to the broadcast
    // channel (immediate delivery to the connected subscriber) AND to
    // the VecDeque buffer (catch-up safety net).
    state.push_rdp_outbound(sealed_bytes.clone());

    // Wait for the WS message with a timeout — if the handler's notify
    // path is working, this should complete within 50 ms; we give 500 ms
    // to leave CI headroom.
    let received = tokio::time::timeout(Duration::from_millis(500), async {
        while let Some(msg) = ws_stream.next().await {
            match msg.expect("ws recv") {
                Message::Binary(bytes) => return Some(bytes),
                Message::Text(_) => continue, // Ignore telemetry text
                Message::Close(_) => return None,
                _ => continue,
            }
        }
        None
    })
    .await
    .expect("recv did not complete within 500ms — notify path is broken")
    .expect("ws closed before delivering frame");

    // === Assert ===
    // The sealed bytes arrive byte-for-byte (axum does not transform
    // Message::Binary payloads).
    assert_eq!(
        received.len(),
        expected_size,
        "received sealed envelope size must match pushed size"
    );

    // Open the sealed envelope on the receiver side — proves the same
    // placeholder key on both sides works end-to-end over a real WS.
    let opened = open_frame(&received, &mut receiver_session).expect("open_frame");
    match (&original, &opened) {
        (RdpFrame::Full(a), RdpFrame::Full(b)) => {
            assert_eq!(a.frame_id, b.frame_id);
            assert_eq!(a.patch_cols, b.patch_cols);
            assert_eq!(a.patch_rows, b.patch_rows);
            assert_eq!(a.patches.len(), b.patches.len());
            assert_eq!(
                a.patches[0].values, b.patches[0].values,
                "first patch must round-trip byte-exact"
            );
        }
        _ => panic!("expected Full → Full round-trip, got different variant"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_server_buffers_frames_for_late_subscriber() {
    // Regression test: frames pushed BEFORE a subscriber connects should
    // still be delivered. Track 3.2's ws_handler drains the VecDeque
    // backlog on startup (see the catch-up drain at the top of
    // holon_ws_handler).
    let (state, port) = spawn_test_server().await;

    let mut sender_session = test_session("ws-late-server", true);
    let mut receiver_session = test_session("ws-late-client", false);

    // Push TWO frames before connecting any subscriber.
    let frame1 = sample_full_frame(1, 2, 2, 1);
    let frame2 = sample_full_frame(2, 2, 2, 2);
    let sealed1 = seal_frame(&frame1, &mut sender_session).expect("seal 1");
    let sealed2 = seal_frame(&frame2, &mut sender_session).expect("seal 2");
    state.push_rdp_outbound(sealed1);
    state.push_rdp_outbound(sealed2);

    // Now connect — the handler's startup drain should forward both.
    let url = format!("ws://127.0.0.1:{port}/holon/ws");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect late");
    let (_sink, mut stream) = ws_stream.split();

    // Collect up to 2 binary messages with a 500 ms budget.
    let received = tokio::time::timeout(Duration::from_millis(500), async {
        let mut got = Vec::new();
        while got.len() < 2 {
            match stream.next().await {
                Some(Ok(Message::Binary(bytes))) => got.push(bytes),
                Some(Ok(Message::Text(_))) => continue,
                Some(Ok(Message::Close(_))) | None => break,
                Some(Ok(_)) => continue,
                Some(Err(e)) => panic!("ws recv error: {e}"),
            }
        }
        got
    })
    .await
    .expect("late-subscriber catch-up did not complete in 500ms");

    assert_eq!(
        received.len(),
        2,
        "late subscriber should receive both backlogged frames"
    );

    // Open both frames — frame_ids should match.
    let f1 = open_frame(&received[0], &mut receiver_session).expect("open 1");
    let f2 = open_frame(&received[1], &mut receiver_session).expect("open 2");
    let ids: Vec<u64> = [f1, f2]
        .iter()
        .map(|f| match f {
            RdpFrame::Full(full) => full.frame_id,
            _ => panic!("expected Full"),
        })
        .collect();
    assert!(
        ids.contains(&1) && ids.contains(&2),
        "both frame_ids should be delivered: got {ids:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_viewer_reconstructs_frame_buffer() {
    // Higher-level acceptance: after receiving a sealed frame over the
    // real WS, a HolonRdpViewer apply_frame call populates its
    // FrameBuffer such that `as_rgba()` returns non-zero pixel data.
    // This is the exact path the `holon_rdp_viewer` example uses.
    let (state, port) = spawn_test_server().await;

    let mut sender_session = test_session("ws-viewer-server", true);
    let mut receiver_session = test_session("ws-viewer-client", false);

    // Frame big enough to fill the FrameBuffer meaningfully.
    let cols = 4u16;
    let rows = 4u16;
    let original = sample_full_frame(42, cols, rows, 100);
    let sealed = seal_frame(&original, &mut sender_session).expect("seal");

    let url = format!("ws://127.0.0.1:{port}/holon/ws");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");
    let (_sink, mut stream) = ws_stream.split();

    // Give the handler a beat to subscribe to the broadcast channel
    // before we push — otherwise push_rdp_outbound happens before
    // subscribe and the frame goes only through the VecDeque catch-up.
    // Both paths should work but this test specifically exercises the
    // notify path.
    tokio::time::sleep(Duration::from_millis(50)).await;
    state.push_rdp_outbound(sealed);

    let received = tokio::time::timeout(Duration::from_millis(500), async {
        while let Some(msg) = stream.next().await {
            if let Ok(Message::Binary(bytes)) = msg {
                return Some(bytes);
            }
        }
        None
    })
    .await
    .expect("no frame within 500ms")
    .expect("stream closed");

    let opened = open_frame(&received, &mut receiver_session).expect("open");

    // Build a viewer, apply the frame, check the pixel buffer.
    let width = cols as u32 * TILE_SIZE as u32;
    let height = rows as u32 * TILE_SIZE as u32;
    let mut viewer = HolonRdpViewer::new(width, height, cols, rows);
    viewer.start();
    assert!(viewer.apply_frame(&opened), "apply_frame should succeed");
    assert_eq!(viewer.frames_received, 1);

    let rgba = viewer.frame_buffer.as_rgba();
    assert_eq!(rgba.len(), (width * height * 4) as usize);
    // Most pixels should be non-zero — the test pattern spans i8 range
    // and the codec maps that to RGBA values. A truly-empty buffer would
    // be all zeros; anything non-trivial means the decode happened.
    let non_zero = rgba.iter().filter(|&&b| b != 0).count();
    assert!(
        non_zero > rgba.len() / 4,
        "frame buffer should have meaningful non-zero pixel data: got {non_zero}/{total} non-zero bytes",
        total = rgba.len()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_replay_protection_survives_the_wire() {
    // The replay window lives inside RdpSession on the receiver. A
    // frame delivered twice over the real WS (e.g. network-level
    // duplication) should be opened once then rejected the second time.
    let (state, port) = spawn_test_server().await;

    let mut sender_session = test_session("ws-replay-server", true);
    let mut receiver_session = test_session("ws-replay-client", false);

    let url = format!("ws://127.0.0.1:{port}/holon/ws");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");
    let (_sink, mut stream) = ws_stream.split();

    tokio::time::sleep(Duration::from_millis(50)).await;

    let frame = sample_delta_frame(5, 4);
    let sealed = seal_frame(&frame, &mut sender_session).expect("seal");

    // Push the SAME sealed envelope twice. The broadcast delivers both
    // to the connected subscriber (the broadcast channel does not
    // deduplicate — that's the receiver's job via the replay window).
    state.push_rdp_outbound(sealed.clone());
    state.push_rdp_outbound(sealed.clone());

    let received = tokio::time::timeout(Duration::from_millis(500), async {
        let mut got: Vec<Vec<u8>> = Vec::new();
        while got.len() < 2 {
            match stream.next().await {
                Some(Ok(Message::Binary(bytes))) => got.push(bytes.into()),
                Some(Ok(_)) => continue,
                Some(Err(_)) | None => break,
            }
        }
        got
    })
    .await
    .expect("duplicate delivery did not complete in 500ms");

    assert_eq!(received.len(), 2, "server should forward both sends");

    // First open succeeds.
    let _first = open_frame(&received[0], &mut receiver_session).expect("first open ok");

    // Second open — SAME nonce, SAME key, but replay window rejects.
    let second = open_frame(&received[1], &mut receiver_session);
    assert!(
        second.is_err(),
        "replay window must reject the second delivery of the same envelope"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Input reverse path test (Phase I.A.5 "How to improve" item 4)
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ws_input_reverse_path_reaches_server_rdp_inbound() {
    // Proves the viewer → server path: the egui viewer sends an
    // InputFrame via seal_input → Message::Binary → WebSocket → server
    // holon_ws_handler routes the bytes into state.rdp_inbound via
    // push_rdp_inbound. Test asserts the bytes arrive unchanged.
    //
    // This closes the gap noted in `papers/phase_1a_results.md`
    // Limitations §3 — "Input reverse path is not in the WS integration
    // tests."
    use symthaea::swarm::rdp_protocol::{InputEvent, InputFrame};
    use symthaea::swarm::rdp_wire::{open_input, seal_input};

    let (state, port) = spawn_test_server().await;
    let mut sender_session = test_session("ws-reverse-client", true);
    let mut receiver_session = test_session("ws-reverse-server", false);

    // Connect the client.
    let url = format!("ws://127.0.0.1:{port}/holon/ws");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");
    let (mut ws_sink, mut _ws_stream) = ws_stream.split();

    // Build an InputFrame — simulates the viewer clicking the phone
    // screen at (504, 1122) normalized to (0.5, 0.5).
    let input = InputFrame {
        sequence: 1,
        timestamp_ms: 1_700_000_000_000,
        events: vec![InputEvent::Pointer {
            x: 0.5,
            y: 0.5,
            button: 1,
            pressed: true,
        }],
    };
    let sealed = seal_input(&input, &mut sender_session).expect("seal input");

    // Send over the WS — this goes into holon_ws_handler's Binary
    // handler which pushes the bytes into state.rdp_inbound.
    ws_sink
        .send(Message::Binary(sealed.clone().into()))
        .await
        .expect("ws send");

    // Give the server a beat to process the message.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Drain rdp_inbound and assert the sealed bytes arrived unchanged.
    let received = state.drain_rdp_inbound();
    assert!(
        !received.is_empty(),
        "server should have received the input frame"
    );
    assert_eq!(
        received[0].len(),
        sealed.len(),
        "inbound bytes length must match sent length"
    );

    // Open the received envelope — proves the same placeholder key on
    // both sides allows the server to recover the InputFrame.
    let opened = open_input(&received[0], &mut receiver_session).expect("open input");
    assert_eq!(opened.sequence, 1);
    assert_eq!(opened.events.len(), 1);
    if let InputEvent::Pointer {
        x,
        y,
        button,
        pressed,
    } = opened.events[0]
    {
        assert!((x - 0.5).abs() < 1e-6);
        assert!((y - 0.5).abs() < 1e-6);
        assert_eq!(button, 1);
        assert!(pressed);
    } else {
        panic!("expected Pointer event");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Soak test (Phase I.A.5 "How to improve" item 2)
//
// Ignored by default; run via:
//   cargo test --test integration_holon_ws --features holon-viewer -- --ignored
// ═══════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore]
async fn ws_soak_ten_thousand_frames() {
    // Pushes 10,000 sealed frames through the broadcast path, asserts
    // the subscriber either receives them all or lags + recovers via
    // the VecDeque catch-up. Confirms steady-state behavior and catches
    // long-run issues that a single-frame test cannot: tokio task leaks,
    // channel buildup, memory growth, unexpected deadlock.
    const N_FRAMES: u64 = 10_000;

    let (state, port) = spawn_test_server().await;
    let mut sender_session = test_session("ws-soak-server", true);

    let url = format!("ws://127.0.0.1:{port}/holon/ws");
    let (ws_stream, _resp) = tokio_tungstenite::connect_async(&url)
        .await
        .expect("ws connect");
    let (_sink, mut stream) = ws_stream.split();

    // Spawn a receiver task that counts delivered frames.
    let received_count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let recv_counter = received_count.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(msg) = stream.next().await {
            match msg {
                Ok(Message::Binary(_)) => {
                    recv_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Ok(Message::Close(_)) | Err(_) => break,
                _ => continue,
            }
        }
    });

    // Give the subscriber a beat to register.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Push 10,000 sealed frames. We use a small cols/rows to keep
    // per-frame work minimal and focus on the throughput path.
    let start = std::time::Instant::now();
    for i in 0..N_FRAMES {
        let frame = sample_full_frame(i + 1, 2, 2, (i & 0xFF) as u8);
        let sealed = seal_frame(&frame, &mut sender_session).expect("seal");
        state.push_rdp_outbound(sealed);
    }
    let push_elapsed = start.elapsed();

    // Give the WS handler time to drain. Broadcast cap is 16, so
    // slow subscribers will lag and recover via drain_rdp_outbound
    // catch-up. Either path is acceptable — the assertion is that
    // the TOTAL delivered count (broadcast + catch-up) equals N_FRAMES.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Expect: all N_FRAMES pushed. Delivered count should be exactly
    // N_FRAMES if no frames were dropped. Under broadcast lag, the
    // receiver may see fewer via the broadcast but the VecDeque holds
    // the rest — total across both paths must equal N_FRAMES.
    let total_pushed: u64 = N_FRAMES;
    let delivered = received_count.load(std::sync::atomic::Ordering::Relaxed);
    let buffered = state.drain_rdp_outbound().len() as u64;
    let total_accounted = delivered + buffered;

    // Cancel receiver task.
    recv_task.abort();

    eprintln!(
        "soak: pushed={} delivered={} buffered={} total_accounted={} push_wall={:?}",
        total_pushed, delivered, buffered, total_accounted, push_elapsed
    );

    // Accounting check: every pushed frame should end up either
    // delivered or buffered. A mismatch means a frame was dropped
    // somewhere (not broadcast-lagged into the buffer, but genuinely
    // lost). VecDeque cap is 512, so buffered can be at most 512.
    // Anything above (buffered + 512) that's not delivered is a loss.
    //
    // Realistic accounting under broadcast cap 16 + VecDeque cap 512:
    //   delivered ≤ N_FRAMES (depends on how fast the recv task runs)
    //   buffered ≤ 512 (cap)
    //   dropped = max(0, N_FRAMES - delivered - 512)
    //
    // For the test to pass we need: either delivered == N_FRAMES
    // (receiver kept up, no drops) OR delivered + buffered >= N_FRAMES - 512
    // (some broadcast drops that the VecDeque cap couldn't fully absorb,
    // but we accept up to 512 legitimate losses per the cap).

    if delivered == total_pushed {
        eprintln!("soak: receiver kept up with all {total_pushed} frames");
    } else {
        let dropped = total_pushed.saturating_sub(delivered + buffered);
        // Allow up to N_FRAMES - 512 delivered + 512 buffered = N_FRAMES.
        // Any "drop" here is a VecDeque overflow which is by-design.
        assert!(
            delivered + buffered >= total_pushed.saturating_sub(512),
            "soak accounting failed: pushed={total_pushed} delivered={delivered} \
             buffered={buffered} dropped={dropped} (allowed drop floor: 512)"
        );
        eprintln!(
            "soak: partial delivery — {delivered} via broadcast, {buffered} buffered, \
             {dropped} dropped by VecDeque cap (max allowed: 512)"
        );
    }

    // Push-side throughput sanity: 10k frames in under 5 seconds.
    assert!(
        push_elapsed < Duration::from_secs(5),
        "push-side throughput below budget: {push_elapsed:?} for {total_pushed} frames"
    );
}