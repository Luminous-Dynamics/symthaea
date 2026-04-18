// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic QUIC datagram-loss smoke test for Phase I.C.
//!
//! Run with `SYMTHAEA_QUIC_DROP_EVERY_N_DATAGRAM=3` to simulate loss of every
//! third datagram-delivered frame. The expected behavior is drop-and-continue:
//! frames 3, 6, and 9 are absent, while later frames still arrive.

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
    use std::time::{Duration, Instant};

    use symthaea::api::holon::HolonHttpState;
    use symthaea::swarm::quic_transport::{run_viewer_quic_client, spawn_holon_quic_server};
    use symthaea::swarm::rdp_codec::TILE_SIZE;
    use symthaea::swarm::rdp_protocol::{FullFrame, QuantizedPatch, RdpFrame, RdpSessionConfig};
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::seal_frame;

    const PLACEHOLDER_KEY: [u8; 32] = [0x42; 32];
    const FRAME_COUNT: u64 = 9;
    const LOSS_ENV: &str = "SYMTHAEA_QUIC_DROP_EVERY_N_DATAGRAM";

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

    fn sample_tiny_frame(frame_id: u64) -> RdpFrame {
        let values: Vec<i8> = (0..(TILE_SIZE * TILE_SIZE))
            .map(|idx| ((idx as u8).wrapping_add(frame_id as u8) as i16 - 128) as i8)
            .collect();

        RdpFrame::Full(FullFrame {
            frame_id,
            timestamp_ms: 0,
            patch_cols: 1,
            patch_rows: 1,
            patches: vec![QuantizedPatch { values }],
            consciousness_level: 0.65,
            harmony: "quic-loss-smoke".into(),
        })
    }

    fn frame_id(frame: &RdpFrame) -> Option<u64> {
        match frame {
            RdpFrame::Full(frame) => Some(frame.frame_id),
            RdpFrame::Delta(frame) => Some(frame.frame_id),
            _ => None,
        }
    }

    async fn wait_for_status(
        status: &Arc<Mutex<String>>,
        prefix: &str,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let started = Instant::now();
        while started.elapsed() < timeout {
            let current = status.lock().map(|guard| guard.clone()).unwrap_or_default();
            if current.starts_with(prefix) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let current = status.lock().map(|guard| guard.clone()).unwrap_or_default();
        bail!("timed out waiting for status {prefix:?}; last status: {current}");
    }

    let drop_every = std::env::var(LOSS_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);

    let (tx, _rx) = std::sync::mpsc::channel();
    let state = Arc::new(HolonHttpState::new(tx));
    let server = spawn_holon_quic_server(
        state.clone(),
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
    )
    .context("spawn QUIC loss smoke server")?;

    let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();
    let (_input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
    let status = Arc::new(Mutex::new(String::new()));
    let session = Arc::new(Mutex::new(test_session("quic-loss-client", false)));
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
    wait_for_status(&status, "connected", Duration::from_secs(2)).await?;

    let mut sender_session = test_session("quic-loss-server", true);
    for id in 1..=FRAME_COUNT {
        let sealed = seal_frame(&sample_tiny_frame(id), &mut sender_session)
            .with_context(|| format!("seal tiny frame {id}"))?;
        state.push_rdp_outbound(sealed);
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut received = Vec::new();
    while Instant::now() < deadline && received.len() < FRAME_COUNT as usize {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match tokio::time::timeout(remaining, frame_rx.recv()).await {
            Ok(Some(frame)) => {
                if let Some(id) = frame_id(&frame) {
                    received.push(id);
                }
            }
            Ok(None) => break,
            Err(_) => break,
        }
    }
    client.abort();

    let expected: Vec<u64> = (1..=FRAME_COUNT)
        .filter(|id| drop_every == 0 || id % drop_every != 0)
        .collect();
    if received != expected {
        return Err(anyhow!(
            "unexpected received frame ids: got {:?}, expected {:?} ({}={})",
            received,
            expected,
            LOSS_ENV,
            drop_every
        ));
    }

    println!(
        "QUIC loss smoke passed via {}: received {:?} ({}={})",
        server.local_addr, received, LOSS_ENV, drop_every
    );

    Ok(())
}
