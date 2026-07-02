// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Gradient compression and information-theoretic measurement.
//!
//! - [`hyperfeel`] — Johnson-Lindenstrauss random projection compression (Achlioptas 2003)
//! - [`phi`] — Approximate integrated information (Phi) for gradient sets

pub mod hyperfeel;
pub mod phi;
