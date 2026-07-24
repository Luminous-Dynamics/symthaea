// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Real-time consciousness telemetry streaming via gRPC.

use std::net::SocketAddr;
use std::pin::Pin;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::{Stream, StreamExt};
use tonic::{Request, Response, Status, transport::Server};

// Include the generated proto code
tonic::include_proto!("symthaea.telemetry");

use telemetry_service_server::{TelemetryService, TelemetryServiceServer};

#[derive(Clone)]
pub struct TelemetryServerImpl {
    tx: broadcast::Sender<TelemetryFrame>,
}

impl TelemetryServerImpl {
    pub fn new(tx: broadcast::Sender<TelemetryFrame>) -> Self {
        Self { tx }
    }
}

#[tonic::async_trait]
impl TelemetryService for TelemetryServerImpl {
    type StreamTelemetryStream =
        Pin<Box<dyn Stream<Item = Result<TelemetryFrame, Status>> + Send + 'static>>;

    async fn stream_telemetry(
        &self,
        request: Request<TelemetryRequest>,
    ) -> Result<Response<Self::StreamTelemetryStream>, Status> {
        let client_id = request.into_inner().client_id;
        tracing::info!("Client '{}' connected to telemetry stream", client_id);

        let rx = self.tx.subscribe();
        let stream = BroadcastStream::new(rx).map(|res| match res {
            Ok(frame) => Ok(frame),
            Err(e) => Err(Status::internal(format!(
                "Broadcast stream lagged: {:?}",
                e
            ))),
        });

        Ok(Response::new(
            Box::pin(stream) as Self::StreamTelemetryStream
        ))
    }
}

/// Project-level broadcaster to stream telemetry to connected gRPC clients.
pub struct TelemetryBroadcaster {
    tx: broadcast::Sender<TelemetryFrame>,
}

impl TelemetryBroadcaster {
    /// Create a new broadcaster with standard buffering.
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self { tx }
    }

    /// Broadcast a single frame of telemetry to all active clients.
    pub fn broadcast(&self, frame: TelemetryFrame) {
        let _ = self.tx.send(frame);
    }

    /// Spawn the gRPC server and serve clients.
    pub async fn run(&self, addr: SocketAddr) -> Result<(), tonic::transport::Error> {
        let service = TelemetryServerImpl::new(self.tx.clone());
        tracing::info!("Starting Symthaea Telemetry gRPC Server on {}", addr);
        Server::builder()
            .add_service(TelemetryServiceServer::new(service))
            .serve(addr)
            .await
    }
}

impl Default for TelemetryBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_broadcast_streaming() {
        let broadcaster = TelemetryBroadcaster::new();

        let frame = TelemetryFrame {
            phi: 0.85,
            harmonies: vec![0.5; 8],
            neuromodulators: vec![0.5; 4],
            arousal: 0.6,
            uncertainty: 0.1,
            surprise: 0.05,
            timestamp: "test-timestamp".to_string(),
            mental_movie: None,
            gwt_broadcast: 0.0,
            hot_metacognitive: 0.0,
            ast_temporal: 0.0,
            knowledge_coherence: 0.0,
            embodiment_level: 0.0,
            self_awareness: 0.0,
            topological_unity: 1.0,
            motor_command: "NoOp".to_string(),
        };

        broadcaster.broadcast(frame);
    }
}
