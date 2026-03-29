// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Ported from fl-aggregator/src/proofs/ — scaffold for mycelix production stack
//! zkSTARK proof scaffold for Mycelix Federated Learning.
//!
//! Provides gradient integrity proofs using Winterfell zkSTARKs.
//! This is a minimal scaffold extracted from the 20K LOC fl-aggregator proof
//! system, containing only: error types, core types, field conversions,
//! Blake3 commitments, and a VerificationPlugin bridge to mycelix-fl-core.
//!
//! Full proof migration (5 circuit types, CLI tools, GPU acceleration)
//! is a separate future project.

pub mod error;
pub mod types;
pub mod field;
pub mod commitment;
pub mod plugin;
