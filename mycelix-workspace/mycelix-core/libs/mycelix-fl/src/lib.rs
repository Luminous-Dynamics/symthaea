// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # mycelix-fl
//!
//! Rust-native federated learning aggregation with Byzantine fault tolerance.
//!
//! This crate provides aggregation algorithms (defenses) for federated learning,
//! ported from the Python 0TML research implementation. All algorithms operate on
//! `Vec<f32>` gradients for WASM compatibility.
//!
//! ## Algorithms
//!
//! - [`FedAvg`](defenses::FedAvg) — Vanilla averaging baseline (no BFT)
//! - [`TrimmedMean`](defenses::TrimmedMean) — Coordinate-wise trimmed mean
//! - [`CoordinateMedian`](defenses::CoordinateMedian) — Coordinate-wise median (50% BFT)
//! - [`Rfa`](defenses::Rfa) — Geometric median via Weiszfeld algorithm
//!
//! All implement the [`Defense`](defenses::Defense) trait.

pub mod compression;
pub mod defenses;
pub mod detection;
pub mod error;
pub mod holochain;
pub mod pogq;
pub mod types;

#[cfg(feature = "std")]
pub mod coordinator;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "grpc")]
pub mod grpc;

pub use error::FlError;
pub use types::*;
