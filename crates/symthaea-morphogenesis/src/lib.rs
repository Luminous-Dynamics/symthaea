// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Symthaea Morphogenesis (Experimental)
//!
//! Experimental morphogenesis sandbox using HDC tissue encodings and
//! topological fragmentation detection.
//!
//! **Warning**: This module is experimental and is not intended for
//! production biological modeling. It is a research prototype for
//! testing hyperdimensional representations of tissue-state patterns.

pub mod bioelectric_ingest;
pub mod conformal_geometric;
pub mod morpho_mesh;
pub mod morpho_topology;

// Re-exports
pub use bioelectric_ingest::*;
pub use conformal_geometric::*;
pub use morpho_mesh::*;
pub use morpho_topology::*;
