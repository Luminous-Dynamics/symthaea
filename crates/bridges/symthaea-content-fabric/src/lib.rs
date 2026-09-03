// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symthaea adapter for Mycelix Content Fabric external placement planning.
//!
//! This crate is intentionally recommendation-only. It accepts the narrow
//! CF-06C external-planner request, independently checks its request commitment,
//! applies a deterministic HDC shadow ranking, and returns a recommendation for
//! Mycelix to revalidate. It has no Holochain, Iroh, CAS, lease, payment, or
//! execution capability.

pub mod model;
pub mod planner;
pub mod protocol;

pub use model::*;
pub use planner::{plan_hdc_shadow_v1, recommend_json_v1, ENGINE_ID_V1, ENGINE_VERSION_V1};
pub use protocol::{
    decode_request_json_v1, recompute_profile_id_v1, recompute_request_id_v1,
    validate_request_v1, ProtocolErrorV1,
};
