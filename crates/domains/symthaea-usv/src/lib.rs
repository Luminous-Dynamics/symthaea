// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # symthaea-usv
//!
//! Mission-neutral uncrewed surface-vessel state and collision-risk boundary.
//!
//! This crate intentionally stops below mission policy and below legal manoeuvre selection.
//! It owns surface-vessel-local state plus a fail-closed boundary that escalates uncertain or
//! risky relative-motion observations to a separate COLREG/navigation policy layer.

#![deny(unsafe_code)]

pub mod encounter;
pub mod maritime_adapter;
pub mod types;

pub use encounter::{
    CollisionReviewAssessment, CollisionReviewDisposition, CollisionReviewPolicy,
    CollisionReviewReason, ContactMotionObservation, assess_collision_review,
};
pub use maritime_adapter::to_maritime_state;
pub use types::{UsvNavigationFix, UsvState};
