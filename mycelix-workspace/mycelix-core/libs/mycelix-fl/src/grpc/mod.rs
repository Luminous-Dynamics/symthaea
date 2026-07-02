// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! gRPC network transport for the FL coordinator.
//!
//! This module provides a tonic-based gRPC server that exposes the
//! [`FLCoordinator`](crate::coordinator::orchestrator::FLCoordinator) API
//! over the network, allowing FL nodes to register, submit gradients,
//! query round status, and retrieve aggregation results.
//!
//! # Feature gate
//!
//! All types and functions in this module require the `grpc` feature.
//!
//! # Protocol
//!
//! The canonical protocol definition lives in `proto/fl_service.proto`.
//! The Rust types are defined manually in [`types`] using prost derive
//! macros (no protoc build step required).

pub mod client;
pub mod server;
pub mod types;
