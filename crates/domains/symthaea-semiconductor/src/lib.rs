// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semiconductor device engineering and evidence-qualified benchmark primitives.
//!
//! This crate is the Rust-first semiconductor domain foundation proposed by
//! ENG-SEMI-000 (#5671). Its initial benchmark module implements the frozen
//! SEMI-BENCH-001 binary-risk metric contract. It deliberately does not train
//! models, read fab datasets, control equipment, or claim physical qualification.

pub mod benchmark;
