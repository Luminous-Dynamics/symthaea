// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PoGQ (Proof of Good Quality) Byzantine detection system.
//!
//! This module implements the novel PoGQ v4.1 Enhanced algorithm that achieves
//! 45% Byzantine Fault Tolerance — exceeding the classical 33% BFT limit through
//! reputation-weighted validation and adaptive hybrid scoring.
//!
//! # Architecture
//!
//! PoGQ is a **stateful** defense: it maintains per-node state across rounds
//! (EMA scores, violation counters, quarantine status). This is fundamentally
//! different from the stateless defenses in [`crate::defenses`] which operate
//! on a single round's gradients.
//!
//! The state machine is designed to be compatible with the winterfell-pogq STARK
//! prover (`vsv-stark/winterfell-pogq/`), enabling zero-knowledge proofs of
//! correct PoGQ execution.
//!
//! # Modules
//!
//! - [`config`] — Configuration parameters (Gen-4 complete)
//! - [`state`] — Per-node state machine (EMA + hysteresis + quarantine)
//! - [`v41_enhanced`] — Full PoGQ v4.1 Enhanced algorithm
//! - [`adaptive`] — Adaptive per-node thresholds for non-IID scenarios
//!
//! # Patent
//!
//! Patent P-005: Method for Byzantine-Robust Federated Learning via
//! Proof of Good Quality with Adaptive Hybrid Scoring.

pub mod adaptive;
pub mod config;
pub mod state;
pub mod v41_enhanced;
