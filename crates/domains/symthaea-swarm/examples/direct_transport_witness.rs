//! Native two-process witness for the direct Iroh data plane.
//!
//! Terminal 1:
//! `cargo run --example direct_transport_witness -- host /tmp/symthaea-direct.addr`
//!
//! Terminal 2:
//! `cargo run --example direct_transport_witness -- join /tmp/symthaea-direct.addr`

use bincode::Options;
use iroh::{Endpoint, EndpointAddr, endpoint::presets, protocol::Router};
use std::{
    env,
    error::Error,
    io,
    path::{Path, PathBuf},
    time::Duration,
};
use symthaea_swarm::direct::{
    DIRECT_ALPN, DirectEvent, DirectLane, DirectRejectReason, DirectTransport,
    DirectTransportError, ReliableRejectCode,
};
use tokio::sync::mpsc;
use uuid::Uuid;

const ADDRESS_FILE_LIMIT: u64 = 64 * 1024;
const WITNESS_TIMEOUT: Duration = Duration::from_secs(60);
const HOST_RELIABLE: &[u8] = b"host-queue-ack";
const JOIN_RELIABLE: &[u8] = b"join-reliable";
const JOIN_DATAGRAM: &[u8] = b"join-datagram";
const JOIN_OPERATION_ID: Uuid = Uuid::from_u128(0x51_4f_49_4e);
const HOST_OPERATION_ID: Uuid = Uuid::from_u128(0x48_4f_53_54);

type DynError = Box<dyn Error + Send + Sync + 'static>;

#[tokio::main]
async fn main() -> Result<(), DynError> {
    let mut args = env::args_os().skip(1);
    let mode = args
        .next()
        .and_then(|value| value.into_string().ok())
        .ok_or_else(|| usage_error("missing mode"))?;
    let address_path = args
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| usage_error("missing endpoint-address file"))?;
    if args.next().is_some() {
        return Err(usage_error("unexpected extra arguments").into());
    }

    match mode.as_str() {
        "host" => run_host(&address_path).await,
        "join" => run_join(&address_path).await,
        _ => Err(usage_error("mode must be `host` or `join`").into()),
    }
}

async fn run_host(address_path: &Path) -> Result<(), DynError> {
    let endpoint = Endpoint::bind(presets::N0).await?;
    let (events_tx, mut events_rx) = mpsc::channel(128);
    let transport = DirectTransport::new(endpoint.clone(), events_tx)?;
    let router = Router::builder(endpoint)
        .accept(DIRECT_ALPN, transport.protocol_handler())
        .spawn();

    let _ = tokio::time::timeout(Duration::from_secs(15), router.endpoint().online()).await;
    write_endpoint_addr(address_path, &router.endpoint().addr()).await?;
    println!(
        "host {} wrote {}",
        transport.node_id(),
        address_path.display()
    );

    let witness = async {
        let mut reliable_peer = None;
        let mut saw_datagram = false;
        let mut saw_conflict = false;
        while reliable_peer.is_none() || !saw_datagram || !saw_conflict {
            let event = events_rx
                .recv()
                .await
                .ok_or_else(|| io::Error::other("direct event channel closed"))?;
            match event {
                DirectEvent::Reliable(message) if message.payload == JOIN_RELIABLE => {
                    reliable_peer = Some(message.author);
                    transport
                        .send_reliable_idempotent(
                            message.author,
                            DirectLane::CONTROL,
                            HOST_OPERATION_ID,
                            HOST_RELIABLE.to_vec(),
                        )
                        .await?;
                }
                DirectEvent::Datagram(message) if message.payload == JOIN_DATAGRAM => {
                    saw_datagram = true;
                }
                DirectEvent::Rejected {
                    reason: DirectRejectReason::OperationConflict { operation_id },
                    ..
                } if operation_id == JOIN_OPERATION_ID => {
                    saw_conflict = true;
                }
                DirectEvent::Rejected { peer, reason, .. } => {
                    return Err(io::Error::other(format!(
                        "host rejected packet from {peer}: {reason:?}"
                    ))
                    .into());
                }
                _ => {}
            }
        }
        Ok::<_, DynError>(())
    };

    tokio::time::timeout(WITNESS_TIMEOUT, witness)
        .await
        .map_err(|_| io::Error::other("host witness timed out"))??;
    println!("HOST_WITNESS_OK");
    transport.shutdown().await;
    router
        .shutdown()
        .await
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

async fn run_join(address_path: &Path) -> Result<(), DynError> {
    let host_addr = wait_for_endpoint_addr(address_path).await?;
    let endpoint = Endpoint::bind(presets::N0).await?;
    let (events_tx, mut events_rx) = mpsc::channel(128);
    let transport = DirectTransport::new(endpoint.clone(), events_tx)?;
    let router = Router::builder(endpoint)
        .accept(DIRECT_ALPN, transport.protocol_handler())
        .spawn();

    let host = transport.connect(host_addr).await?;
    let receipt = transport
        .send_reliable_idempotent(
            host,
            DirectLane::CONTROL,
            JOIN_OPERATION_ID,
            JOIN_RELIABLE.to_vec(),
        )
        .await?;
    if !receipt.remote_queue_accepted || receipt.remote_duplicate {
        return Err(io::Error::other("host did not acknowledge first queue admission").into());
    }
    let duplicate = transport
        .send_reliable_idempotent(
            host,
            DirectLane::CONTROL,
            JOIN_OPERATION_ID,
            JOIN_RELIABLE.to_vec(),
        )
        .await?;
    if !duplicate.remote_duplicate {
        return Err(io::Error::other("host did not deduplicate the reliable retry").into());
    }
    let conflict = transport
        .send_reliable_idempotent(
            host,
            DirectLane::CONTROL,
            JOIN_OPERATION_ID,
            b"different-operation-payload".to_vec(),
        )
        .await;
    if !matches!(
        conflict,
        Err(DirectTransportError::RemoteRejected {
            reason: ReliableRejectCode::OperationConflict,
        })
    ) {
        return Err(io::Error::other(
            "host did not reject operation-ID reuse with a different payload",
        )
        .into());
    }

    transport
        .send_datagram(host, DirectLane::PLAYER_INPUT, JOIN_DATAGRAM.to_vec())
        .await?;

    let witness = async {
        loop {
            let event = events_rx
                .recv()
                .await
                .ok_or_else(|| io::Error::other("direct event channel closed"))?;
            match event {
                DirectEvent::Reliable(message)
                    if message.author == host && message.payload == HOST_RELIABLE =>
                {
                    return Ok::<_, DynError>(());
                }
                DirectEvent::Rejected { peer, reason, .. } => {
                    return Err(io::Error::other(format!(
                        "joiner rejected packet from {peer}: {reason:?}"
                    ))
                    .into());
                }
                _ => {}
            }
        }
    };

    tokio::time::timeout(WITNESS_TIMEOUT, witness)
        .await
        .map_err(|_| io::Error::other("join witness timed out"))??;
    println!("JOIN_WITNESS_OK");
    transport.shutdown().await;
    router
        .shutdown()
        .await
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(())
}

async fn write_endpoint_addr(path: &Path, address: &EndpointAddr) -> Result<(), DynError> {
    let encoded = bincode::DefaultOptions::new()
        .with_limit(ADDRESS_FILE_LIMIT)
        .serialize(address)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    tokio::fs::write(&temporary, encoded).await?;
    tokio::fs::rename(&temporary, path).await?;
    Ok(())
}

async fn wait_for_endpoint_addr(path: &Path) -> Result<EndpointAddr, DynError> {
    let read = async {
        loop {
            match tokio::fs::read(path).await {
                Ok(bytes) => {
                    let address = bincode::DefaultOptions::new()
                        .with_limit(ADDRESS_FILE_LIMIT)
                        .reject_trailing_bytes()
                        .deserialize(&bytes)?;
                    return Ok::<_, DynError>(address);
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(error) => return Err(error.into()),
            }
        }
    };
    tokio::time::timeout(WITNESS_TIMEOUT, read)
        .await
        .map_err(|_| io::Error::other("endpoint-address file did not appear"))?
}

fn usage_error(detail: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("{detail}; usage: direct_transport_witness <host|join> <endpoint-address-file>"),
    )
}
