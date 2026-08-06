// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal benchmark task families — `SYMTHAEA_TEMPORAL_BENCHMARK_V2_PLAN.md` §5.
//!
//! Every generator here must prove its corpus requires memory, via
//! `symthaea_evidence_plane::task_validator`, before returning it. The
//! predecessor benchmark asserted that property by construction and was wrong,
//! which is why it is now enforced rather than documented.

pub mod arms;
pub mod context_aliasing;
pub mod irregular_time;

/// End-to-end verification that generator -> validator -> scorer actually
/// compose. Every component was unit-tested in isolation and none had ever been
/// run through the full chain, which is how the missing timed-scoring path
/// stayed invisible.
#[cfg(test)]
mod pipeline_integration;
