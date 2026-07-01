// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Headless localhost smoke test for the baseline Holon WebSocket transport.
//!
//! Runs an in-process `/holon/ws` server, pushes one sealed RDP frame through
//! `HolonHttpState`, and confirms the client receives and opens it.

#[cfg(not(feature = "holon-viewer"))]
fn main() {
    eprintln!("Requires: --features holon-viewer");
    std::process::exit(1);
}

#[cfg(feature = "holon-viewer")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::{Context, anyhow, bail};
    use futures_util::StreamExt;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    use symthaea::api::holon::{HolonHttpState, holon_router};
    use symthaea::swarm::rdp_codec::TILE_SIZE;
    use symthaea::swarm::rdp_protocol::{FullFrame, QuantizedPatch, RdpFrame, RdpSessionConfig};
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::{open_frame, seal_frame};

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
            harmony: "ws-smoke".into(),
        })
    }

    let (tx, _rx) = std::sync::mpsc::channel();
    let state = Arc::new(HolonHttpState::new(tx));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .context("bind WS smoke listener")?;
    let addr = listener.local_addr().context("read WS smoke address")?;
    let router = holon_router(state.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await });

    let url = format!("ws://{addr}/holon/ws");
    let (mut socket, _response) = connect_async(&url)
        .await
        .with_context(|| format!("connect WS smoke client to {url}"))?;

    let mut sender_session = test_session("ws-smoke-server", true);
    let original = sample_full_frame(1, 4, 4, 9);
    let sealed = seal_frame(&original, &mut sender_session).context("seal smoke frame")?;
    state.push_rdp_outbound(sealed);

    let bytes = tokio::time::timeout(Duration::from_secs(2), async {
        while let Some(msg) = socket.next().await {
            match msg.context("receive WS smoke message")? {
                Message::Binary(bytes) => return Ok(bytes),
                Message::Close(_) => bail!("WS smoke server closed before binary frame"),
                _ => {}
            }
        }
        Err(anyhow!("WS smoke stream ended before binary frame"))
    })
    .await
    .context("timed out waiting for WS smoke frame")??;

    let mut receiver_session = test_session("ws-smoke-client", false);
    let received = open_frame(&bytes, &mut receiver_session).context("open WS smoke frame")?;

    match received {
        RdpFrame::Full(frame) if frame.frame_id == 1 => {}
        RdpFrame::Full(frame) => bail!("unexpected frame id {}", frame.frame_id),
        other => bail!("expected Full frame, got {other:?}"),
    }

    server.abort();
    println!("WS smoke passed via {url}");

    Ok(())
}