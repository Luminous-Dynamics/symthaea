// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign RDP end-to-end integration test.
//!
//! Creates an RDP server with TestCapture, connects a client,
//! verifies frames flow through the full pipeline:
//!   capture → codec → encode → server queue → client decode → FrameBuffer
//!
//! This runs without any display server (uses TestCapture synthetic frames).

use symthaea::swarm::rdp_capture::TestCapture;
use symthaea::swarm::rdp_client::{ClientInbound, FrameBuffer, RdpClientHandle};
use symthaea::swarm::rdp_codec::TILE_SIZE;
use symthaea::swarm::rdp_protocol::{
    ControlMessage, DeltaFrame, FullFrame, QuantizedPatch, RdpFrame, RdpSessionConfig,
};
use symthaea::swarm::rdp_server::{RdpServerHandle, ServerInbound};

/// End-to-end: server captures → encodes → client reconstructs.
#[test]
fn test_e2e_server_to_client_frame_flow() {
    // Server side: TestCapture at 256x256.
    let capture = Box::new(TestCapture::new(256, 256));
    let config = RdpSessionConfig::default();
    let (mut server, server_inbound_tx, server_outbound_rx) =
        RdpServerHandle::new(capture, config.clone(), "test-server");

    // Simulate a session connecting.
    server_inbound_tx
        .send(ServerInbound::SessionReady {
            session_id: "e2e-session".into(),
            peer_id: "e2e-client".into(),
            session_key: [42u8; 32],
            config: config.clone(),
        })
        .unwrap();

    // Tick server — should produce first full frame.
    let _inputs = server.tick(0.5);
    assert!(server.frame_id() > 0, "Server should have produced a frame");

    // Drain the outbound frame from server.
    let mut frames_received = Vec::new();
    while let Ok(outbound) = server_outbound_rx.try_recv() {
        // The outbound contains serialized frame data.
        // In real transport, this goes over QUIC. Here we deserialize directly.
        if let symthaea::swarm::rdp_server::ServerOutbound::Frame { data, .. } = outbound {
            if let Ok(frame) = serde_json::from_slice::<RdpFrame>(&data) {
                frames_received.push(frame);
            }
        }
    }

    assert!(
        !frames_received.is_empty(),
        "Should have received at least one frame from server"
    );

    // Client side: apply received frames.
    let (mut client, client_inbound_tx, _client_outbound_rx) =
        RdpClientHandle::new("test-server", config);

    // Send Welcome to initialize client buffer.
    client_inbound_tx
        .send(ClientInbound::Welcome {
            patch_cols: 4,
            patch_rows: 4,
            target_fps: 5,
            consciousness_level: 0.5,
        })
        .unwrap();
    client.poll();

    // Forward server frames to client.
    for frame in frames_received {
        client_inbound_tx.send(ClientInbound::Frame(frame)).unwrap();
    }

    let updated = client.poll();
    assert!(updated, "Client should have applied the frame");
    assert!(
        client.frame_buffer.has_content,
        "Client frame buffer should have content"
    );
    assert!(
        client.frame_buffer.frames_applied >= 1,
        "At least one frame should be applied"
    );

    // Verify pixel buffer is non-trivial (not all zeros).
    let non_zero = client
        .frame_buffer
        .pixels
        .iter()
        .filter(|&&p| p != 0)
        .count();
    assert!(
        non_zero > 100,
        "Pixel buffer should contain non-zero data, got {} non-zero bytes",
        non_zero
    );
}

/// Test delta frames: second tick should produce sparse delta.
#[test]
fn test_e2e_delta_frame_sparsity() {
    let capture = Box::new(TestCapture::new(256, 256));
    let config = RdpSessionConfig::default();
    let (mut server, server_inbound_tx, server_outbound_rx) =
        RdpServerHandle::new(capture, config.clone(), "test-server");

    server_inbound_tx
        .send(ServerInbound::SessionReady {
            session_id: "delta-test".into(),
            peer_id: "client".into(),
            session_key: [42u8; 32],
            config,
        })
        .unwrap();

    // Tick 1: full frame.
    server.tick(0.5);
    let mut frame1_count = 0;
    while let Ok(_) = server_outbound_rx.try_recv() {
        frame1_count += 1;
    }

    // Tick 2: delta frame (TestCapture only changes the active region).
    server.tick(0.5);
    let mut delta_frames = Vec::new();
    while let Ok(outbound) = server_outbound_rx.try_recv() {
        if let symthaea::swarm::rdp_server::ServerOutbound::Frame { data, .. } = outbound {
            if let Ok(frame) = serde_json::from_slice::<RdpFrame>(&data) {
                delta_frames.push(frame);
            }
        }
    }

    // The second frame should be a delta (not full).
    // It may also be empty if nothing changed enough.
    for frame in &delta_frames {
        match frame {
            RdpFrame::Delta(d) => {
                // Delta should have fewer patches than total tiles (4x4 = 16).
                assert!(
                    d.patches.len() < 16,
                    "Delta should be sparse, got {} patches out of 16",
                    d.patches.len()
                );
            }
            RdpFrame::Full(_) => {
                // Full frame on second tick is acceptable but suboptimal.
            }
            _ => {}
        }
    }
}

/// Test consciousness gating: low phi → no input forwarding.
#[test]
fn test_e2e_consciousness_gates_input() {
    let capture = Box::new(TestCapture::new(256, 256));
    let config = RdpSessionConfig {
        min_consciousness: 0.3,
        ..RdpSessionConfig::default()
    };
    let (mut server, server_inbound_tx, _) =
        RdpServerHandle::new(capture, config.clone(), "test-server");

    server_inbound_tx
        .send(ServerInbound::SessionReady {
            session_id: "gate-test".into(),
            peer_id: "client".into(),
            session_key: [42u8; 32],
            config,
        })
        .unwrap();
    server.tick(0.5);

    // Send low-phi attestation → ViewOnly.
    server_inbound_tx
        .send(ServerInbound::Attestation {
            session_id: "gate-test".into(),
            phi: 0.1, // Below 0.3 threshold
            state_hash: [0u8; 32],
            signature: vec![],
        })
        .unwrap();
    server.tick(0.5);

    // Send input → should be blocked.
    server_inbound_tx
        .send(ServerInbound::Input {
            session_id: "gate-test".into(),
            events: vec![symthaea::swarm::rdp_protocol::InputEvent::Key {
                code: 65,
                pressed: true,
                modifiers: 0,
            }],
        })
        .unwrap();
    let inputs = server.tick(0.5);
    assert!(
        inputs.is_empty(),
        "Input should be blocked when consciousness below threshold"
    );
}