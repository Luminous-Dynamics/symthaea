// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact scientific workload definitions for physical-computation research.
//!
//! Workload modules generate PHYS-006 fixture suites only. They do not execute
//! a backend, train a readout, score a result, or authorize active cognition.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Exact causal change-point definition and deterministic fixture generator.
pub mod change_point;
/// Exact delayed-recall definition and deterministic fixture generator.
pub mod delayed_recall;
/// Exact NARMA10 workload definition and deterministic fixture generator.
pub mod narma10;
