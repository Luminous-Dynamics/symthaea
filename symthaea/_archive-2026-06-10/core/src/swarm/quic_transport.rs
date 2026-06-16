// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phase I.C QUIC transport for the Holon RDP wire.
//!
//! The sealed RDP envelope (`seal_frame` / `open_frame`) remains unchanged.
//! This module sits underneath it and provides:
//! - unreliable QUIC datagrams for outbound video frames
//! - reliable bidirectional QUIC streams for inbound input events
//! - datagram fragmentation + reassembly, because the sealed RDP envelope is
//!   far larger than a single QUIC datagram on realistic paths

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use quinn::{ClientConfig, Connection, Endpoint, RecvStream, SendStream, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, info, warn};

use crate::api::holon::SharedHolonState;
use crate::swarm::rdp_protocol::{InputFrame, RdpFrame};
use crate::swarm::rdp_session::RdpSession;
use crate::swarm::rdp_wire::{open_frame, seal_input};

const QUIC_CERT_FILENAME: &str = "symthaea-holon-quic-cert.der";
const QUIC_STREAM_BUFFER_BYTES: usize = 2 * 1024 * 1024;
const QUIC_REASSEMBLY_TTL: Duration = Duration::from_secs(2);
const QUIC_REASSEMBLY_MAX_FRAMES: usize = 8;
const QUIC_DATAGRAM_RELIABLE_FALLBACK_BYTES: usize = 64 * 1024;
const QUIC_DATAGRAM_RELIABLE_FALLBACK_FRAGMENTS: usize = 64;
const QUIC_DROP_EVERY_N_DATAGRAM_ENV: &str = "SYMTHAEA_QUIC_DROP_EVERY_N_DATAGRAM";
const DATAGRAM_MAGIC: [u8; 4] = *b"HRDQ";
const DATAGRAM_HEADER_BYTES: usize = 20;
static RUSTLS_PROVIDER_INIT: Once = Once::new();

#[derive(Debug)]
pub struct HolonQuicServer {
    pub local_addr: SocketAddr,
    pub cert_path: PathBuf,
    pub fingerprint: String,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for HolonQuicServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Debug)]
struct PendingFrame {
    total_len: usize,
    data: Vec<u8>,
    received_offsets: HashSet<u32>,
    received_bytes: usize,
    updated_at: Instant,
}

#[derive(Debug, Default)]
struct DatagramReassembler {
    frames: HashMap<u64, PendingFrame>,
}

impl DatagramReassembler {
    fn insert(&mut self, datagram: &[u8]) -> Option<Vec<u8>> {
        self.evict_stale();

        let (seq, total_len, offset, payload) = parse_datagram(datagram)?;
        let total_len = total_len as usize;
        let offset_usize = offset as usize;

        if payload.is_empty()
            || offset_usize >= total_len
            || offset_usize + payload.len() > total_len
        {
            return None;
        }

        if self.frames.len() >= QUIC_REASSEMBLY_MAX_FRAMES && !self.frames.contains_key(&seq) {
            let oldest = self
                .frames
                .iter()
                .min_by_key(|(_, frame)| frame.updated_at)
                .map(|(seq, _)| *seq);
            if let Some(oldest) = oldest {
                self.frames.remove(&oldest);
            }
        }

        let entry = self.frames.entry(seq).or_insert_with(|| PendingFrame {
            total_len,
            data: vec![0u8; total_len],
            received_offsets: HashSet::new(),
            received_bytes: 0,
            updated_at: Instant::now(),
        });

        if entry.total_len != total_len {
            self.frames.remove(&seq);
            return None;
        }

        entry.updated_at = Instant::now();
        entry.data[offset_usize..offset_usize + payload.len()].copy_from_slice(payload);
        if entry.received_offsets.insert(offset) {
            entry.received_bytes += payload.len();
        }

        if entry.received_bytes >= entry.total_len {
            return self.frames.remove(&seq).map(|frame| frame.data);
        }

        None
    }

    fn evict_stale(&mut self) {
        let now = Instant::now();
        self.frames
            .retain(|_, frame| now.duration_since(frame.updated_at) <= QUIC_REASSEMBLY_TTL);
    }
}

pub fn default_quic_port(http_port: u16) -> u16 {
    http_port.saturating_add(1)
}

pub async fn resolve_quic_endpoint(spec: &str, default_port: u16) -> Result<(SocketAddr, String)> {
    let raw = spec.trim().strip_prefix("quic://").unwrap_or(spec.trim());
    if raw.is_empty() {
        return Err(anyhow!("empty QUIC endpoint"));
    }

    if let Ok(addr) = raw.parse::<SocketAddr>() {
        return Ok((addr, addr.ip().to_string()));
    }

    let (host, port) = match raw.rsplit_once(':') {
        Some((host, port)) if !host.is_empty() && port.parse::<u16>().is_ok() => (
            host.trim_matches(['[', ']']).to_string(),
            port.parse::<u16>()?,
        ),
        _ => (raw.trim_matches(['[', ']']).to_string(), default_port),
    };

    let addr = tokio::net::lookup_host((host.as_str(), port))
        .await
        .with_context(|| format!("resolve QUIC endpoint {host}:{port}"))?
        .next()
        .ok_or_else(|| anyhow!("no addresses resolved for {host}:{port}"))?;
    Ok((addr, host))
}

pub fn default_cert_path() -> PathBuf {
    std::env::temp_dir().join(QUIC_CERT_FILENAME)
}

pub fn spawn_holon_quic_server(
    state: SharedHolonState,
    bind_addr: SocketAddr,
) -> Result<HolonQuicServer> {
    let cert_path = default_cert_path();
    let (server_config, fingerprint) = make_server_config(bind_addr, &cert_path)?;
    let endpoint = Endpoint::server(server_config, bind_addr)
        .with_context(|| format!("bind QUIC endpoint on {bind_addr}"))?;
    let local_addr = endpoint.local_addr().context("read QUIC local addr")?;

    info!(
        "Holon QUIC listening on quic://{} (cert {}, fingerprint {})",
        local_addr,
        cert_path.display(),
        &fingerprint[..16.min(fingerprint.len())]
    );

    let task = tokio::spawn(async move {
        while let Some(incoming) = endpoint.accept().await {
            match incoming.await {
                Ok(connection) => {
                    let peer = connection.remote_address();
                    let state = state.clone();
                    tokio::spawn(async move {
                        if let Err(error) = handle_holon_connection(connection, state).await {
                            warn!(%peer, %error, "QUIC Holon connection failed");
                        }
                    });
                }
                Err(error) => {
                    warn!(%error, "Holon QUIC accept failed");
                }
            }
        }
    });

    Ok(HolonQuicServer {
        local_addr,
        cert_path,
        fingerprint,
        task,
    })
}

pub async fn run_viewer_quic_client(
    remote_addr: SocketAddr,
    server_name: &str,
    session: Arc<Mutex<RdpSession>>,
    frame_tx: UnboundedSender<RdpFrame>,
    mut input_rx: UnboundedReceiver<InputFrame>,
    connection_status: Arc<Mutex<String>>,
    on_frame: Arc<dyn Fn() + Send + Sync>,
) -> Result<()> {
    set_status(
        &connection_status,
        format!("connecting to quic://{remote_addr}"),
    );

    let mut endpoint = Endpoint::client(client_bind_addr(remote_addr.ip()))
        .context("bind QUIC client endpoint")?;
    endpoint.set_default_client_config(make_client_config()?);

    let connection = endpoint
        .connect(remote_addr, server_name)
        .with_context(|| format!("start QUIC connect to {remote_addr}"))?
        .await
        .with_context(|| format!("complete QUIC connect to {remote_addr}"))?;
    set_status(
        &connection_status,
        format!("connected (quic://{remote_addr})"),
    );
    (on_frame)();

    let (mut send_stream, mut recv_stream) = connection
        .open_bi()
        .await
        .context("open reliable QUIC stream for input events")?;

    let read_session = session.clone();
    let read_status = connection_status.clone();
    let read_repaint = on_frame.clone();
    let read_connection = connection.clone();
    let datagram_frame_tx = frame_tx.clone();
    let read_task = async move {
        let mut reassembler = DatagramReassembler::default();
        loop {
            let datagram = match read_connection.read_datagram().await {
                Ok(datagram) => datagram,
                Err(error) => {
                    set_status(&read_status, format!("quic recv error: {error}"));
                    break;
                }
            };

            if let Some(sealed) = reassembler.insert(&datagram) {
                let opened = {
                    let mut guard = match read_session.lock() {
                        Ok(guard) => guard,
                        Err(_) => return,
                    };
                    open_frame(&sealed, &mut guard)
                };

                match opened {
                    Ok(frame) => {
                        if datagram_frame_tx.send(frame).is_err() {
                            break;
                        }
                        (read_repaint)();
                    }
                    Err(error) => {
                        set_status(&read_status, format!("open_frame error: {error}"));
                    }
                }
            }
        }
    };

    let reliable_session = session.clone();
    let reliable_status = connection_status.clone();
    let reliable_repaint = on_frame.clone();
    let reliable_connection = connection.clone();
    let reliable_frame_tx = frame_tx.clone();
    let reliable_read_task = async move {
        loop {
            let mut recv_stream = match reliable_connection.accept_uni().await {
                Ok(stream) => stream,
                Err(error) => {
                    debug!(%error, "viewer reliable QUIC outbound accept closed");
                    break;
                }
            };

            match read_framed(&mut recv_stream).await {
                Ok(Some(sealed)) => {
                    let opened = {
                        let mut guard = match reliable_session.lock() {
                            Ok(guard) => guard,
                            Err(_) => return,
                        };
                        open_frame(&sealed, &mut guard)
                    };

                    match opened {
                        Ok(frame) => {
                            if reliable_frame_tx.send(frame).is_err() {
                                break;
                            }
                            (reliable_repaint)();
                        }
                        Err(error) => {
                            set_status(&reliable_status, format!("open_frame error: {error}"));
                        }
                    }
                }
                Ok(None) => {}
                Err(error) => {
                    set_status(
                        &reliable_status,
                        format!("quic reliable recv error: {error}"),
                    );
                    break;
                }
            }
        }
    };

    let write_status = connection_status.clone();
    let write_session = session.clone();
    let write_task = async move {
        while let Some(input) = input_rx.recv().await {
            let sealed = {
                let mut guard = match write_session.lock() {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
                seal_input(&input, &mut guard)
            };

            match sealed {
                Ok(bytes) => {
                    if let Err(error) = write_framed(&mut send_stream, &bytes).await {
                        set_status(&write_status, format!("quic send error: {error}"));
                        return;
                    }
                }
                Err(error) => {
                    set_status(&write_status, format!("seal_input error: {error}"));
                }
            }
        }

        let _ = send_stream.finish();
    };

    let server_read_task = async move {
        loop {
            match read_framed(&mut recv_stream).await {
                Ok(Some(_)) => {}
                Ok(None) => break,
                Err(error) => {
                    debug!(%error, "viewer reliable QUIC stream closed");
                    break;
                }
            }
        }
    };

    tokio::select! {
        _ = read_task => {}
        _ = reliable_read_task => {}
        _ = write_task => {}
        _ = server_read_task => {}
    }

    Ok(())
}

fn make_server_config(
    bind_addr: SocketAddr,
    cert_path: &PathBuf,
) -> Result<(ServerConfig, String)> {
    ensure_rustls_crypto_provider();

    let cert = rcgen::generate_simple_self_signed(server_subject_alt_names(bind_addr))
        .context("generate self-signed QUIC certificate")?;
    let cert_der = cert.cert.der().clone();
    std::fs::write(cert_path, cert_der.as_ref())
        .with_context(|| format!("write QUIC certificate to {}", cert_path.display()))?;

    let fingerprint = blake3::hash(cert_der.as_ref()).to_hex().to_string();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(cert.key_pair.serialize_der()));
    let crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .context("build rustls QUIC server config")?;

    let mut transport = quinn::TransportConfig::default();
    transport.datagram_receive_buffer_size(Some(QUIC_STREAM_BUFFER_BYTES));
    transport.datagram_send_buffer_size(QUIC_STREAM_BUFFER_BYTES);
    transport.max_concurrent_bidi_streams(8_u8.into());

    let mut server = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(crypto)
            .context("wrap rustls server config for quinn")?,
    ));
    server.transport = Arc::new(transport);
    Ok((server, fingerprint))
}

fn make_client_config() -> Result<ClientConfig> {
    ensure_rustls_crypto_provider();

    let cert_path = default_cert_path();
    let cert_der = std::fs::read(&cert_path)
        .with_context(|| format!("read QUIC certificate at {}", cert_path.display()))?;
    let cert = CertificateDer::from(cert_der);

    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(cert)
        .context("add QUIC certificate to client roots")?;

    let crypto = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    let mut transport = quinn::TransportConfig::default();
    transport.datagram_receive_buffer_size(Some(QUIC_STREAM_BUFFER_BYTES));
    transport.datagram_send_buffer_size(QUIC_STREAM_BUFFER_BYTES);
    transport.max_concurrent_bidi_streams(8_u8.into());

    let mut client = ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .context("wrap rustls client config for quinn")?,
    ));
    client.transport_config(Arc::new(transport));
    Ok(client)
}

fn ensure_rustls_crypto_provider() {
    RUSTLS_PROVIDER_INIT.call_once(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    });
}

fn server_subject_alt_names(bind_addr: SocketAddr) -> Vec<String> {
    let mut sans = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    let ip = bind_addr.ip().to_string();
    if !sans.iter().any(|entry| entry == &ip) {
        sans.push(ip);
    }
    sans
}

fn client_bind_addr(remote_ip: IpAddr) -> SocketAddr {
    match remote_ip {
        IpAddr::V4(_) => SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0),
        IpAddr::V6(_) => SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0),
    }
}

async fn handle_holon_connection(connection: Connection, state: SharedHolonState) -> Result<()> {
    let peer = connection.remote_address();
    info!("Holon QUIC client connected from {}", peer);

    let backlog = state.drain_rdp_outbound();
    let mut transport_seq = 0u64;
    for sealed in backlog {
        send_sealed_frame(&connection, &mut transport_seq, sealed).await?;
    }

    let mut rdp_rx = state.rdp_outbound_tx.subscribe();
    let state_for_input = state.clone();
    let input_connection = connection.clone();
    let input_task = async move {
        loop {
            let (_send, mut recv) = match input_connection.accept_bi().await {
                Ok(stream) => stream,
                Err(error) => {
                    debug!(%error, "Holon QUIC reliable stream accept closed");
                    break;
                }
            };

            loop {
                match read_framed(&mut recv).await {
                    Ok(Some(bytes)) => state_for_input.push_rdp_inbound(bytes),
                    Ok(None) => break,
                    Err(error) => {
                        debug!(%error, "Holon QUIC input stream read failed");
                        break;
                    }
                }
            }
        }
    };

    let output_task = async move {
        loop {
            match rdp_rx.recv().await {
                Ok(sealed) => {
                    if let Err(error) =
                        send_sealed_frame(&connection, &mut transport_seq, sealed).await
                    {
                        debug!(%error, "Holon QUIC datagram send failed");
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(skipped)) => {
                    debug!("Holon QUIC lagged {skipped} frames, re-syncing via catch-up buffer");
                    for sealed in state.drain_rdp_outbound() {
                        if let Err(error) =
                            send_sealed_frame(&connection, &mut transport_seq, sealed).await
                        {
                            debug!(%error, "Holon QUIC catch-up send failed");
                        }
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    };

    tokio::select! {
        _ = input_task => {}
        _ = output_task => {}
    }

    Ok(())
}

async fn send_sealed_frame(
    connection: &Connection,
    transport_seq: &mut u64,
    sealed: Vec<u8>,
) -> Result<()> {
    let max_datagram = connection.max_datagram_size().unwrap_or(1200);
    let seq = *transport_seq;
    let fragments = fragment_datagrams(seq, &sealed, max_datagram)
        .with_context(|| format!("fragment sealed frame of {} bytes", sealed.len()))?;
    *transport_seq = transport_seq.wrapping_add(1);

    if sealed.len() > QUIC_DATAGRAM_RELIABLE_FALLBACK_BYTES
        || fragments.len() > QUIC_DATAGRAM_RELIABLE_FALLBACK_FRAGMENTS
    {
        let mut stream = connection
            .open_uni()
            .await
            .context("open reliable QUIC stream for oversized sealed frame")?;
        write_framed(&mut stream, &sealed)
            .await
            .context("write oversized sealed frame on reliable QUIC stream")?;
        let _ = stream.finish();
        return Ok(());
    }

    let drop_this_frame = drop_every_n_datagrams()
        .map(|n| n > 0 && (seq.wrapping_add(1)) % n == 0)
        .unwrap_or(false);
    if drop_this_frame {
        debug!(
            seq,
            fragments = fragments.len(),
            "dropping QUIC datagram frame due to test loss injection"
        );
        return Ok(());
    }

    for datagram in fragments {
        match connection.send_datagram(Bytes::from(datagram)) {
            Ok(()) => {}
            Err(quinn::SendDatagramError::UnsupportedByPeer) => {
                return Err(anyhow!("peer does not support QUIC datagrams"));
            }
            Err(quinn::SendDatagramError::Disabled) => {
                return Err(anyhow!("QUIC datagrams disabled on connection"));
            }
            Err(quinn::SendDatagramError::TooLarge) => {
                return Err(anyhow!("fragment exceeded max QUIC datagram size"));
            }
            Err(quinn::SendDatagramError::ConnectionLost(error)) => {
                return Err(anyhow!(
                    "QUIC connection lost while sending datagram: {error}"
                ));
            }
        }
    }

    Ok(())
}

fn drop_every_n_datagrams() -> Option<u64> {
    std::env::var(QUIC_DROP_EVERY_N_DATAGRAM_ENV)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|n| *n > 0)
}

fn fragment_datagrams(seq: u64, payload: &[u8], max_datagram_size: usize) -> Result<Vec<Vec<u8>>> {
    let chunk_payload = max_datagram_size
        .checked_sub(DATAGRAM_HEADER_BYTES)
        .ok_or_else(|| anyhow!("max QUIC datagram size {max_datagram_size} too small"))?;
    if chunk_payload == 0 {
        return Err(anyhow!("QUIC datagram payload budget is zero"));
    }

    let total_len = u32::try_from(payload.len()).context("sealed frame exceeds 4 GiB")?;
    let mut out = Vec::new();
    let mut offset = 0usize;
    while offset < payload.len() {
        let end = (offset + chunk_payload).min(payload.len());
        let chunk = &payload[offset..end];
        let mut datagram = Vec::with_capacity(DATAGRAM_HEADER_BYTES + chunk.len());
        datagram.extend_from_slice(&DATAGRAM_MAGIC);
        datagram.extend_from_slice(&seq.to_le_bytes());
        datagram.extend_from_slice(&total_len.to_le_bytes());
        datagram.extend_from_slice(&(offset as u32).to_le_bytes());
        datagram.extend_from_slice(chunk);
        out.push(datagram);
        offset = end;
    }
    Ok(out)
}

fn parse_datagram(datagram: &[u8]) -> Option<(u64, u32, u32, &[u8])> {
    if datagram.len() < DATAGRAM_HEADER_BYTES || datagram[..4] != DATAGRAM_MAGIC {
        return None;
    }

    let seq = u64::from_le_bytes(datagram[4..12].try_into().ok()?);
    let total_len = u32::from_le_bytes(datagram[12..16].try_into().ok()?);
    let offset = u32::from_le_bytes(datagram[16..20].try_into().ok()?);
    Some((seq, total_len, offset, &datagram[DATAGRAM_HEADER_BYTES..]))
}

async fn write_framed(send: &mut SendStream, bytes: &[u8]) -> Result<()> {
    let len = u32::try_from(bytes.len()).context("reliable QUIC frame exceeds 4 GiB")?;
    send.write_all(&len.to_le_bytes()).await?;
    send.write_all(bytes).await?;
    send.flush().await?;
    Ok(())
}

async fn read_framed(recv: &mut RecvStream) -> Result<Option<Vec<u8>>> {
    let mut len_buf = [0u8; 4];
    match recv.read_exact(&mut len_buf).await {
        Ok(()) => {}
        Err(quinn::ReadExactError::FinishedEarly(_)) => return Ok(None),
        Err(error) => return Err(anyhow!("read reliable QUIC length failed: {error}")),
    }

    let len = u32::from_le_bytes(len_buf) as usize;
    let mut data = vec![0u8; len];
    recv.read_exact(&mut data)
        .await
        .map_err(|error| anyhow!("read reliable QUIC payload failed: {error}"))?;
    Ok(Some(data))
}

fn set_status(status: &Arc<Mutex<String>>, value: String) {
    if let Ok(mut guard) = status.lock() {
        *guard = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::holon::HolonHttpState;
    use crate::swarm::rdp_codec::TILE_SIZE;
    use crate::swarm::rdp_protocol::{FullFrame, QuantizedPatch, RdpSessionConfig};
    use crate::swarm::rdp_wire::seal_frame;

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
            harmony: "quic-test".into(),
        })
    }

    #[test]
    fn datagram_fragmentation_round_trips() {
        let payload = vec![0xAB; 64 * 1024];
        let fragments = fragment_datagrams(7, &payload, 1200).expect("fragment");
        assert!(
            fragments.len() > 1,
            "payload should require multiple datagrams"
        );

        let mut reassembler = DatagramReassembler::default();
        let mut out = None;
        for fragment in fragments {
            out = reassembler.insert(&fragment);
        }

        assert_eq!(out.expect("complete frame"), payload);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn quic_server_delivers_sealed_frame_over_datagrams() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let state = Arc::new(HolonHttpState::new(tx));
        let server = spawn_holon_quic_server(
            state.clone(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        )
        .expect("spawn QUIC server");

        let mut sender_session = test_session("quic-server", true);
        let original = sample_full_frame(1, 4, 4, 9);
        let sealed = seal_frame(&original, &mut sender_session).expect("seal");
        state.push_rdp_outbound(sealed);

        let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();
        let (_input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
        let status = Arc::new(Mutex::new(String::new()));
        let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        let session = Arc::new(Mutex::new(test_session("quic-client", false)));

        tokio::spawn(run_viewer_quic_client(
            server.local_addr,
            "127.0.0.1",
            session,
            frame_tx,
            input_rx,
            status,
            repaint,
        ));

        let received = tokio::time::timeout(Duration::from_millis(750), frame_rx.recv())
            .await
            .expect("QUIC receive timeout")
            .expect("no frame received");

        match received {
            RdpFrame::Full(frame) => assert_eq!(frame.frame_id, 1),
            other => panic!("expected Full frame, got {other:?}"),
        }
    }
}
