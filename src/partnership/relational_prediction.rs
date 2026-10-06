// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only held-out prediction for relational dynamics.
//!
//! The qualification question is deliberately predictive rather than
//! ontological:
//!
//!   Does a relational feature set predict an independently observed future
//!   interaction outcome better than isolated-agent, synchrony-only, or
//!   common-driver baselines on data that occur strictly later in time?
//!
//! This module uses a fixed temporal holdout, an explicit label horizon, and a
//! boundary check that prevents training labels from extending into the test
//! feature interval. It never touches production partnership state, cognition,
//! relational_psi, trust, or response generation.
//!
//! The null layer has three deterministic families:
//!
//! - CircularShift: shift all relational channels together within each split,
//!   destroying partner-specific alignment while preserving each channel's
//!   marginal sequence structure.
//! - FeatureDecoupling: shift relational channels by distinct offsets within
//!   each split, preserving their individual temporal structure while breaking
//!   the coherent relational bundle.
//! - IncrementalRelationalShift: preserve synchrony and non-relational context
//!   while shifting only the added directional/turn-taking channels.
//!
//! Null outputs are empirical calibration diagnostics, not p-values. Their
//! interpretation is tied to the explicit surrogate construction and its
//! temporal assumptions; they are not generic significance tests.

use super::relational_harmonics::EvidenceStatus;