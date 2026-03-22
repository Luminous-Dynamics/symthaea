// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Working memory backends for benchmark tasks.
//!
//! Two backends:
//! - **Lightweight** (default): Self-contained FIFO WM with activation decay.
//!   Fast, no dependencies beyond `symthaea-core`.
//! - **Full** (feature `symthaea-backend`): Wraps Symthaea's `ContinuousMind`,
//!   providing dream consolidation, social coherence, and the real cognitive
//!   tick pipeline.
//!
//! Both expose the same `WorkingMemory` and `WmConfig` types with identical APIs.

mod lightweight;

#[cfg(feature = "symthaea-backend")]
mod full;

#[cfg(not(feature = "symthaea-backend"))]
pub use lightweight::{WmConfig, WorkingMemory};

#[cfg(feature = "symthaea-backend")]
pub use full::{WmConfig, WorkingMemory};

pub use lightweight::WmConfig as LightweightWmConfig;
/// Re-export the lightweight WM under a distinct name for ablation comparisons.
pub use lightweight::WorkingMemory as LightweightWm;

pub mod ssm_temporal;
