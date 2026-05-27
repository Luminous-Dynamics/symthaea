// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Headless localhost smoke test for the Phase I.C QUIC transport.
//!
//! Runs an in-process Holon QUIC server, pushes one sealed RDP frame through
//! `HolonHttpState`, and confirms the client receives and opens it.

#[cfg(not(feature = "holon-quic"))]
fn main() {
    eprintln!("Requires: --features holon-quic");
    std::process::exit(1);
}

#[cfg(feature = "holon-quic")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::{Context, anyhow, bail};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use symthaea::api::holon::HolonHttpState;
    use symthaea::swarm::quic_transport::{run_viewer_quic_client, spawn_holon_quic_server};
    use symthaea::swarm::rdp_codec::TILE_SIZE;
    use symthaea::swarm::rdp_protocol::{FullFrame, QuantizedPatch, RdpFrame, RdpSessionConfig};
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::seal_frame;

    const PLACEHOLDER_KEY: [u8; 32] = [0x42; 32];

    fn test_session(label: &str, is_initiator: bool) -> RdpSession {
        let mut session = RdpSession::new(
            label.into(),
            "peer".into(),
            RdpSessionConfig::default(),
            is_initiator,
        );
        session.on_connected();
        session.on_handshake_complete(PLACEHOLDER_KEY);
        session
    }

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
            harmony: "quic-smoke".into(),
        })
    }

    let (tx, _rx) = std::sync::mpsc::channel();
    let state = Arc::new(HolonHttpState::new(tx));
    let server = spawn_holon_quic_server(
        state.clone(),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
    )
    .context("spawn QUIC smoke server")?;

    let mut sender_session = test_session("quic-smoke-server", true);
    let original = sample_full_frame(1, 4, 4, 9);
    let sealed = seal_frame(&original, &mut sender_session).context("seal smoke frame")?;
    state.push_rdp_outbound(sealed);

    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();
    let (_input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
    let status = Arc::new(Mutex::new(String::new()));
    let session = Arc::new(Mutex::new(test_session("quic-smoke-client", false)));
    let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});

    let client = tokio::spawn(run_viewer_quic_client(
        server.local_addr,
        "127.0.0.1",
        session,
        frame_tx,
        input_rx,
        status.clone(),
        repaint,
    ));

    let received = tokio::time::timeout(Duration::from_secs(2), frame_rx.recv())
        .await
        .context("timed out waiting for QUIC smoke frame")?
        .ok_or_else(|| anyhow!("QUIC smoke client closed before receiving a frame"))?;

    match received {
        RdpFrame::Full(frame) if frame.frame_id == 1 => {}
        RdpFrame::Full(frame) => bail!("unexpected frame id {}", frame.frame_id),
        other => bail!("expected Full frame, got {other:?}"),
    }

    client.abort();

    if let Ok(guard) = status.lock() {
        println!("QUIC smoke passed via {} ({})", server.local_addr, *guard);
    } else {
        println!("QUIC smoke passed via {}", server.local_addr);
    }

    Ok(())
}