// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! tonic gRPC client for connecting to the FL coordinator.
//!
//! [`FlClient`] provides a typed interface to all six RPCs defined in
//! `proto/fl_service.proto`, using the hand-written prost types from
//! [`super::types`] (no protoc codegen required).
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "grpc")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use mycelix_fl::grpc::client::FlClient;
//!
//! let client = FlClient::connect("http://[::1]:50051", "hospital-alpha".into()).await?;
//! let health = client.health_check().await?;
//! println!("Coordinator healthy: {}", health.healthy);
//! # Ok(())
//! # }
//! ```

use tonic::client::Grpc;
use tonic::codec::ProstCodec;
use tonic::transport::Channel;

use super::types::*;

/// A gRPC client for the Mycelix FL coordinator.
///
/// Wraps a tonic [`Channel`] and provides typed methods for each RPC
/// defined in the `FederatedLearning` service.
pub struct FlClient {
    channel: Channel,
    node_id: String,
}

impl FlClient {
    /// Connect to the coordinator at the given address.
    ///
    /// `addr` should be a full URI, e.g. `"http://[::1]:50051"`.
    pub async fn connect(addr: &str, node_id: String) -> Result<Self, Box<dyn std::error::Error>> {
        let channel = Channel::from_shared(addr.to_string())?
            .connect()
            .await?;
        Ok(Self { channel, node_id })
    }

    /// Return the node ID this client was created with.
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Register this node with the coordinator.
    pub async fn register(
        &self,
        did: Option<String>,
    ) -> Result<RegisterNodeResponse, tonic::Status> {
        let request = tonic::Request::new(RegisterNodeRequest {
            node_id: self.node_id.clone(),
            public_key: vec![],
            did: did.unwrap_or_default(),
        });

        let path = http::uri::PathAndQuery::from_static(
            "/mycelix.fl.FederatedLearning/RegisterNode",
        );
        let codec: ProstCodec<RegisterNodeRequest, RegisterNodeResponse> = ProstCodec::default();

        let mut grpc = Grpc::new(self.channel.clone());
        grpc.ready().await.map_err(|e| {
            tonic::Status::unavailable(format!("channel not ready: {}", e))
        })?;

        let response = grpc.unary(request, path, codec).await?;
        Ok(response.into_inner())
    }

    /// Submit a gradient for the given round.
    pub async fn submit_gradient(
        &self,
        values: Vec<f32>,
        round: u64,
    ) -> Result<SubmitGradientResponse, tonic::Status> {
        let request = tonic::Request::new(SubmitGradientRequest {
            node_id: self.node_id.clone(),
            values,
            round,
        });

        let path = http::uri::PathAndQuery::from_static(
            "/mycelix.fl.FederatedLearning/SubmitGradient",
        );
        let codec: ProstCodec<SubmitGradientRequest, SubmitGradientResponse> =
            ProstCodec::default();

        let mut grpc = Grpc::new(self.channel.clone());
        grpc.ready().await.map_err(|e| {
            tonic::Status::unavailable(format!("channel not ready: {}", e))
        })?;

        let response = grpc.unary(request, path, codec).await?;
        Ok(response.into_inner())
    }

    /// Query the status of a round. Pass `0` for the current round.
    pub async fn round_status(
        &self,
        round: u64,
    ) -> Result<GetRoundStatusResponse, tonic::Status> {
        let request = tonic::Request::new(GetRoundStatusRequest { round });

        let path = http::uri::PathAndQuery::from_static(
            "/mycelix.fl.FederatedLearning/GetRoundStatus",
        );
        let codec: ProstCodec<GetRoundStatusRequest, GetRoundStatusResponse> =
            ProstCodec::default();

        let mut grpc = Grpc::new(self.channel.clone());
        grpc.ready().await.map_err(|e| {
            tonic::Status::unavailable(format!("channel not ready: {}", e))
        })?;

        let response = grpc.unary(request, path, codec).await?;
        Ok(response.into_inner())
    }

    /// Get the aggregation result for a completed round.
    pub async fn round_result(
        &self,
        round: u64,
    ) -> Result<GetRoundResultResponse, tonic::Status> {
        let request = tonic::Request::new(GetRoundResultRequest { round });

        let path = http::uri::PathAndQuery::from_static(
            "/mycelix.fl.FederatedLearning/GetRoundResult",
        );
        let codec: ProstCodec<GetRoundResultRequest, GetRoundResultResponse> =
            ProstCodec::default();

        let mut grpc = Grpc::new(self.channel.clone());
        grpc.ready().await.map_err(|e| {
            tonic::Status::unavailable(format!("channel not ready: {}", e))
        })?;

        let response = grpc.unary(request, path, codec).await?;
        Ok(response.into_inner())
    }

    /// Perform a health check against the coordinator.
    pub async fn health_check(&self) -> Result<HealthCheckResponse, tonic::Status> {
        let request = tonic::Request::new(HealthCheckRequest {});

        let path = http::uri::PathAndQuery::from_static(
            "/mycelix.fl.FederatedLearning/HealthCheck",
        );
        let codec: ProstCodec<HealthCheckRequest, HealthCheckResponse> = ProstCodec::default();

        let mut grpc = Grpc::new(self.channel.clone());
        grpc.ready().await.map_err(|e| {
            tonic::Status::unavailable(format!("channel not ready: {}", e))
        })?;

        let response = grpc.unary(request, path, codec).await?;
        Ok(response.into_inner())
    }
}
