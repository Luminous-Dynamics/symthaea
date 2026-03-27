// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! tonic gRPC server implementation for the FL coordinator.
//!
//! [`FlGrpcService`] wraps an [`FLCoordinator`] behind an `Arc<Mutex<_>>`
//! and implements all six RPCs defined in `proto/fl_service.proto`:
//!
//! - `RegisterNode` — register a node with credentials
//! - `SubmitGradient` — submit a gradient for the current round
//! - `GetRoundStatus` — query the status of a round
//! - `GetRoundResult` — retrieve aggregation results for a completed round
//! - `StreamGradients` — client-streaming RPC: stream gradients, receive result
//! - `HealthCheck` — liveness probe
//!
//! # Usage
//!
//! ```rust,no_run
//! # #[cfg(feature = "grpc")]
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use mycelix_fl::coordinator::config::CoordinatorConfig;
//! use mycelix_fl::coordinator::orchestrator::FLCoordinator;
//! use mycelix_fl::defenses::fedavg::FedAvg;
//!
//! let coordinator = FLCoordinator::new(
//!     CoordinatorConfig::default(),
//!     Box::new(FedAvg),
//!     None,
//! );
//! let addr = "[::1]:50051".parse()?;
//! mycelix_fl::grpc::server::serve(coordinator, addr).await?;
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status, Streaming};

use super::types::*;
use crate::coordinator::node_manager::NodeCredential;
use crate::coordinator::orchestrator::FLCoordinator;
use crate::types::Gradient;

// ---------------------------------------------------------------------------
// Service trait (hand-written, mirrors tonic codegen output)
// ---------------------------------------------------------------------------

/// The `FederatedLearning` gRPC service trait.
///
/// This is equivalent to what `tonic-build` would generate from the proto
/// file, but defined manually to avoid the protoc build dependency.
#[tonic::async_trait]
pub trait FederatedLearning: Send + Sync + 'static {
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status>;

    async fn submit_gradient(
        &self,
        request: Request<SubmitGradientRequest>,
    ) -> Result<Response<SubmitGradientResponse>, Status>;

    async fn get_round_status(
        &self,
        request: Request<GetRoundStatusRequest>,
    ) -> Result<Response<GetRoundStatusResponse>, Status>;

    async fn get_round_result(
        &self,
        request: Request<GetRoundResultRequest>,
    ) -> Result<Response<GetRoundResultResponse>, Status>;

    async fn stream_gradients(
        &self,
        request: Request<Streaming<SubmitGradientRequest>>,
    ) -> Result<Response<GetRoundResultResponse>, Status>;

    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status>;
}

// ---------------------------------------------------------------------------
// tonic Service implementation (manual codegen equivalent)
// ---------------------------------------------------------------------------

/// A tonic server wrapper that routes gRPC requests to a
/// [`FederatedLearning`] implementation.
///
/// Follows the same pattern as tonic's generated code.
pub struct FederatedLearningServer<T: FederatedLearning> {
    inner: Arc<T>,
}

impl<T: FederatedLearning> FederatedLearningServer<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }
}

impl<T: FederatedLearning> Clone for FederatedLearningServer<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> tonic::server::NamedService for FederatedLearningServer<T>
where
    T: FederatedLearning,
{
    const NAME: &'static str = "mycelix.fl.FederatedLearning";
}

type BoxFuture<T, E> =
    std::pin::Pin<Box<dyn std::future::Future<Output = Result<T, E>> + Send + 'static>>;

impl<T, B> tower_service::Service<http::Request<B>> for FederatedLearningServer<T>
where
    T: FederatedLearning,
    B: http_body::Body + Send + 'static,
    B::Error: Into<Box<dyn std::error::Error + Send + Sync>> + Send + 'static,
    B::Data: Send,
{
    type Response = http::Response<tonic::body::BoxBody>;
    type Error = std::convert::Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: http::Request<B>) -> Self::Future {
        let inner = self.inner.clone();

        match req.uri().path() {
            "/mycelix.fl.FederatedLearning/RegisterNode" => {
                struct Svc<T: FederatedLearning>(Arc<T>);
                impl<T: FederatedLearning>
                    tonic::server::UnaryService<RegisterNodeRequest> for Svc<T>
                {
                    type Response = RegisterNodeResponse;
                    type Future = BoxFuture<Response<Self::Response>, Status>;
                    fn call(
                        &mut self,
                        request: Request<RegisterNodeRequest>,
                    ) -> Self::Future {
                        let inner = Arc::clone(&self.0);
                        Box::pin(async move { inner.register_node(request).await })
                    }
                }
                let fut = async move {
                    let method = Svc(inner);
                    let codec = tonic::codec::ProstCodec::default();
                    let mut grpc = tonic::server::Grpc::new(codec);
                    let res = grpc.unary(method, req).await;
                    Ok(res)
                };
                Box::pin(fut)
            }

            "/mycelix.fl.FederatedLearning/SubmitGradient" => {
                struct Svc<T: FederatedLearning>(Arc<T>);
                impl<T: FederatedLearning>
                    tonic::server::UnaryService<SubmitGradientRequest> for Svc<T>
                {
                    type Response = SubmitGradientResponse;
                    type Future = BoxFuture<Response<Self::Response>, Status>;
                    fn call(
                        &mut self,
                        request: Request<SubmitGradientRequest>,
                    ) -> Self::Future {
                        let inner = Arc::clone(&self.0);
                        Box::pin(async move { inner.submit_gradient(request).await })
                    }
                }
                let fut = async move {
                    let method = Svc(inner);
                    let codec = tonic::codec::ProstCodec::default();
                    let mut grpc = tonic::server::Grpc::new(codec);
                    let res = grpc.unary(method, req).await;
                    Ok(res)
                };
                Box::pin(fut)
            }

            "/mycelix.fl.FederatedLearning/GetRoundStatus" => {
                struct Svc<T: FederatedLearning>(Arc<T>);
                impl<T: FederatedLearning>
                    tonic::server::UnaryService<GetRoundStatusRequest> for Svc<T>
                {
                    type Response = GetRoundStatusResponse;
                    type Future = BoxFuture<Response<Self::Response>, Status>;
                    fn call(
                        &mut self,
                        request: Request<GetRoundStatusRequest>,
                    ) -> Self::Future {
                        let inner = Arc::clone(&self.0);
                        Box::pin(async move { inner.get_round_status(request).await })
                    }
                }
                let fut = async move {
                    let method = Svc(inner);
                    let codec = tonic::codec::ProstCodec::default();
                    let mut grpc = tonic::server::Grpc::new(codec);
                    let res = grpc.unary(method, req).await;
                    Ok(res)
                };
                Box::pin(fut)
            }

            "/mycelix.fl.FederatedLearning/GetRoundResult" => {
                struct Svc<T: FederatedLearning>(Arc<T>);
                impl<T: FederatedLearning>
                    tonic::server::UnaryService<GetRoundResultRequest> for Svc<T>
                {
                    type Response = GetRoundResultResponse;
                    type Future = BoxFuture<Response<Self::Response>, Status>;
                    fn call(
                        &mut self,
                        request: Request<GetRoundResultRequest>,
                    ) -> Self::Future {
                        let inner = Arc::clone(&self.0);
                        Box::pin(async move { inner.get_round_result(request).await })
                    }
                }
                let fut = async move {
                    let method = Svc(inner);
                    let codec = tonic::codec::ProstCodec::default();
                    let mut grpc = tonic::server::Grpc::new(codec);
                    let res = grpc.unary(method, req).await;
                    Ok(res)
                };
                Box::pin(fut)
            }

            "/mycelix.fl.FederatedLearning/StreamGradients" => {
                struct Svc<T: FederatedLearning>(Arc<T>);
                impl<T: FederatedLearning>
                    tonic::server::ClientStreamingService<SubmitGradientRequest>
                    for Svc<T>
                {
                    type Response = GetRoundResultResponse;
                    type Future = BoxFuture<Response<Self::Response>, Status>;
                    fn call(
                        &mut self,
                        request: Request<Streaming<SubmitGradientRequest>>,
                    ) -> Self::Future {
                        let inner = Arc::clone(&self.0);
                        Box::pin(async move { inner.stream_gradients(request).await })
                    }
                }
                let fut = async move {
                    let method = Svc(inner);
                    let codec = tonic::codec::ProstCodec::default();
                    let mut grpc = tonic::server::Grpc::new(codec);
                    let res = grpc.client_streaming(method, req).await;
                    Ok(res)
                };
                Box::pin(fut)
            }

            "/mycelix.fl.FederatedLearning/HealthCheck" => {
                struct Svc<T: FederatedLearning>(Arc<T>);
                impl<T: FederatedLearning>
                    tonic::server::UnaryService<HealthCheckRequest> for Svc<T>
                {
                    type Response = HealthCheckResponse;
                    type Future = BoxFuture<Response<Self::Response>, Status>;
                    fn call(
                        &mut self,
                        request: Request<HealthCheckRequest>,
                    ) -> Self::Future {
                        let inner = Arc::clone(&self.0);
                        Box::pin(async move { inner.health_check(request).await })
                    }
                }
                let fut = async move {
                    let method = Svc(inner);
                    let codec = tonic::codec::ProstCodec::default();
                    let mut grpc = tonic::server::Grpc::new(codec);
                    let res = grpc.unary(method, req).await;
                    Ok(res)
                };
                Box::pin(fut)
            }

            _ => Box::pin(async move {
                Ok(http::Response::builder()
                    .status(200)
                    .header("grpc-status", "12")
                    .header("content-type", "application/grpc")
                    .body(tonic::body::empty_body())
                    .unwrap())
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete implementation: FlGrpcService
// ---------------------------------------------------------------------------

/// gRPC service implementation backed by an [`FLCoordinator`].
///
/// The coordinator is wrapped in `Arc<Mutex<_>>` for safe concurrent access
/// from multiple gRPC handler tasks.
pub struct FlGrpcService {
    coordinator: Arc<Mutex<FLCoordinator>>,
    start_time: Instant,
}

impl FlGrpcService {
    /// Create a new gRPC service wrapping the given coordinator.
    pub fn new(coordinator: FLCoordinator) -> Self {
        Self {
            coordinator: Arc::new(Mutex::new(coordinator)),
            start_time: Instant::now(),
        }
    }

    /// Build the tonic server service from this implementation.
    pub fn into_service(self) -> FederatedLearningServer<Self> {
        FederatedLearningServer::new(self)
    }
}

#[tonic::async_trait]
impl FederatedLearning for FlGrpcService {
    async fn register_node(
        &self,
        request: Request<RegisterNodeRequest>,
    ) -> Result<Response<RegisterNodeResponse>, Status> {
        let req = request.into_inner();
        let mut coord = self.coordinator.lock().await;

        let credential = NodeCredential {
            node_id: req.node_id.clone(),
            public_key: if req.public_key.is_empty() {
                None
            } else {
                Some(req.public_key)
            },
            did: if req.did.is_empty() {
                None
            } else {
                Some(req.did)
            },
            registered_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        match coord.register_node(credential) {
            Ok(()) => Ok(Response::new(RegisterNodeResponse {
                success: true,
                message: format!("Node {} registered", req.node_id),
                current_round: coord.current_round_number().unwrap_or(0),
            })),
            Err(e) => Ok(Response::new(RegisterNodeResponse {
                success: false,
                message: e.to_string(),
                current_round: 0,
            })),
        }
    }

    async fn submit_gradient(
        &self,
        request: Request<SubmitGradientRequest>,
    ) -> Result<Response<SubmitGradientResponse>, Status> {
        let req = request.into_inner();
        let mut coord = self.coordinator.lock().await;

        let gradient = Gradient::new(req.node_id, req.values, req.round);

        match coord.submit_gradient(gradient) {
            Ok(()) => Ok(Response::new(SubmitGradientResponse {
                accepted: true,
                message: "Gradient accepted".into(),
            })),
            Err(e) => Ok(Response::new(SubmitGradientResponse {
                accepted: false,
                message: e.to_string(),
            })),
        }
    }

    async fn get_round_status(
        &self,
        request: Request<GetRoundStatusRequest>,
    ) -> Result<Response<GetRoundStatusResponse>, Status> {
        let req = request.into_inner();
        let coord = self.coordinator.lock().await;

        // round == 0 means "current round"
        let round_num = if req.round == 0 {
            coord.current_round_number().unwrap_or(0)
        } else {
            req.round
        };

        // Check if there is an active round matching the request
        if let Some(current) = coord.current_round_number() {
            if round_num == current || req.round == 0 {
                return Ok(Response::new(GetRoundStatusResponse {
                    round: current,
                    status: "collecting".into(),
                    submitted_nodes: 0,
                    required_nodes: coord.config().min_nodes as u32,
                }));
            }
        }

        // Check history for completed rounds
        for summary in coord.round_history() {
            if summary.round_number == round_num {
                return Ok(Response::new(GetRoundStatusResponse {
                    round: summary.round_number,
                    status: summary.status.to_lowercase(),
                    submitted_nodes: summary.node_count as u32,
                    required_nodes: coord.config().min_nodes as u32,
                }));
            }
        }

        // Round not found
        Ok(Response::new(GetRoundStatusResponse {
            round: round_num,
            status: "unknown".into(),
            submitted_nodes: 0,
            required_nodes: coord.config().min_nodes as u32,
        }))
    }

    async fn get_round_result(
        &self,
        request: Request<GetRoundResultRequest>,
    ) -> Result<Response<GetRoundResultResponse>, Status> {
        let req = request.into_inner();
        let coord = self.coordinator.lock().await;

        // Search round history for the requested round
        for summary in coord.round_history() {
            if summary.round_number == req.round && summary.status == "Complete" {
                return Ok(Response::new(GetRoundResultResponse {
                    round: summary.round_number,
                    success: true,
                    aggregated_gradient: Vec::new(),
                    included_nodes: Vec::new(),
                    excluded_nodes: Vec::new(),
                    scores: Vec::new(),
                }));
            }
        }

        Ok(Response::new(GetRoundResultResponse {
            round: req.round,
            success: false,
            aggregated_gradient: Vec::new(),
            included_nodes: Vec::new(),
            excluded_nodes: Vec::new(),
            scores: Vec::new(),
        }))
    }

    async fn stream_gradients(
        &self,
        request: Request<Streaming<SubmitGradientRequest>>,
    ) -> Result<Response<GetRoundResultResponse>, Status> {
        use tokio_stream::StreamExt;

        let mut stream = request.into_inner();

        // Collect all gradients from the stream
        while let Some(result) = stream.next().await {
            let req = result.map_err(|e| Status::internal(e.to_string()))?;
            let mut coord = self.coordinator.lock().await;

            let gradient = Gradient::new(req.node_id, req.values, req.round);
            if let Err(e) = coord.submit_gradient(gradient) {
                eprintln!("StreamGradients: rejected gradient: {}", e);
            }
        }

        // After all gradients received, try to complete the round
        let mut coord = self.coordinator.lock().await;
        match coord.complete_round() {
            Ok(result) => {
                let scores = result
                    .scores
                    .iter()
                    .map(|(id, s)| NodeScore {
                        node_id: id.clone(),
                        score: *s,
                    })
                    .collect();

                let round_num = coord
                    .round_history()
                    .last()
                    .map(|s| s.round_number)
                    .unwrap_or(0);

                Ok(Response::new(GetRoundResultResponse {
                    round: round_num,
                    success: true,
                    aggregated_gradient: result.gradient,
                    included_nodes: result.included_nodes,
                    excluded_nodes: result.excluded_nodes,
                    scores,
                }))
            }
            Err(e) => Err(Status::failed_precondition(e.to_string())),
        }
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let coord = self.coordinator.lock().await;
        let uptime = self.start_time.elapsed().as_secs();

        Ok(Response::new(HealthCheckResponse {
            healthy: true,
            uptime_secs: uptime,
            registered_nodes: coord.node_count() as u32,
            current_round: coord.current_round_number().unwrap_or(0),
        }))
    }
}

// ---------------------------------------------------------------------------
// Server entry point
// ---------------------------------------------------------------------------

/// Start the gRPC server, listening on the given address.
///
/// This function blocks until the server shuts down.
pub async fn serve(
    coordinator: FLCoordinator,
    addr: std::net::SocketAddr,
) -> Result<(), Box<dyn std::error::Error>> {
    let service = FlGrpcService::new(coordinator);

    println!("Mycelix FL gRPC server listening on {}", addr);

    tonic::transport::Server::builder()
        .add_service(service.into_service())
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_name() {
        assert_eq!(
            <FederatedLearningServer<FlGrpcService> as tonic::server::NamedService>::NAME,
            "mycelix.fl.FederatedLearning"
        );
    }
}
