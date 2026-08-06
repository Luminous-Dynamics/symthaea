// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Real-time consciousness telemetry streaming via gRPC.
//!
//! No authentication by default -- any peer that can reach the listening
//! port may subscribe (this binary binds `[::1]:50051`, loopback-only, so
//! that's the whole exposure today). Set `SYMTHAEA_TELEMETRY_TOKEN` on the
//! server to require matching clients to send it as `auth_token` on
//! `TelemetryRequest`; see [`TelemetryServerImpl::stream_telemetry`].

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
    /// `None` (the default, via [`TelemetryServerImpl::new`]) preserves the
    /// prior no-auth behavior exactly. `Some(token)` requires every
    /// `stream_telemetry` request's `auth_token` to match.
    expected_token: Option<String>,
}

impl TelemetryServerImpl {
    /// Reads the optional shared-secret token once, from
    /// `SYMTHAEA_TELEMETRY_TOKEN`, at construction time rather than per
    /// request.
    pub fn new(tx: broadcast::Sender<TelemetryFrame>) -> Self {
        Self::with_expected_token(tx, std::env::var("SYMTHAEA_TELEMETRY_TOKEN").ok())
    }

    /// Explicit-token constructor, primarily for tests that want a
    /// deterministic check without mutating the process-global environment.
    pub fn with_expected_token(
        tx: broadcast::Sender<TelemetryFrame>,
        expected_token: Option<String>,
    ) -> Self {
        Self { tx, expected_token }
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
        let req = request.into_inner();
        let client_id = req.client_id;

        // Optional shared-secret auth, opt-in via SYMTHAEA_TELEMETRY_TOKEN.
        // The server previously accepted any TCP peer's subscribe request
        // with zero credentials -- fine for the loopback-only default this
        // binary uses today, but a real gate is needed the moment this
        // service is ever bound to a non-loopback address. Unset (the
        // default) preserves that prior no-auth behavior exactly.
        if let Some(expected) = &self.expected_token
            && req.auth_token != *expected
        {
            tracing::warn!(
                "Rejected telemetry stream request from client '{}': bad or missing auth token",
                client_id
            );
            return Err(Status::unauthenticated("invalid or missing auth_token"));
        }

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

    /// Regression test for the unauthenticated-telemetry-stream finding:
    /// when a server is configured with an expected token,
    /// `stream_telemetry` must reject a request with no (or the wrong)
    /// `auth_token` rather than serving the stream to anyone who can reach
    /// the port.
    #[tokio::test]
    async fn stream_telemetry_rejects_missing_or_wrong_token_when_configured() {
        let (tx, _rx) = broadcast::channel(1);
        let server =
            TelemetryServerImpl::with_expected_token(tx, Some("correct-secret".to_string()));

        let no_token = Request::new(TelemetryRequest {
            client_id: "attacker".to_string(),
            auth_token: String::new(),
        });
        let result = server.stream_telemetry(no_token).await;
        // `Response<Self::StreamTelemetryStream>` wraps a `dyn Stream` trait
        // object and doesn't implement `Debug`, so `unwrap_err()` (which
        // needs the Ok side to be Debug for its panic message) doesn't
        // typecheck here -- match directly instead.
        match result {
            Err(status) => assert_eq!(status.code(), tonic::Code::Unauthenticated),
            Ok(_) => panic!("expected Unauthenticated, got Ok"),
        }

        let wrong_token = Request::new(TelemetryRequest {
            client_id: "attacker".to_string(),
            auth_token: "guessed-wrong".to_string(),
        });
        let result = server.stream_telemetry(wrong_token).await;
        match result {
            Err(status) => assert_eq!(status.code(), tonic::Code::Unauthenticated),
            Ok(_) => panic!("expected Unauthenticated, got Ok"),
        }
    }

    /// A request with the correct token is accepted, and (unchanged
    /// default behavior) a server with no configured token accepts any
    /// request -- preserving the prior no-auth loopback-dev-tool behavior
    /// exactly for operators who don't opt in.
    #[tokio::test]
    async fn stream_telemetry_accepts_correct_token_or_when_unconfigured() {
        let (tx, _rx) = broadcast::channel(1);
        let server =
            TelemetryServerImpl::with_expected_token(tx.clone(), Some("correct-secret".into()));
        let correct = Request::new(TelemetryRequest {
            client_id: "dashboard".to_string(),
            auth_token: "correct-secret".to_string(),
        });
        assert!(server.stream_telemetry(correct).await.is_ok());

        let unconfigured = TelemetryServerImpl::with_expected_token(tx, None);
        let anything = Request::new(TelemetryRequest {
            client_id: "dashboard".to_string(),
            auth_token: String::new(),
        });
        assert!(unconfigured.stream_telemetry(anything).await.is_ok());
    }
}
