// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Symthaea Fabrication Kernel
//!
//! HDC-to-Mesh bridge: geometric primitives encoded as hypervectors,
//! CSG boolean operations, triangle mesh tessellation, STL/3MF export,
//! and physics simulation abstraction layer.

#[cfg(feature = "analytical")]
pub mod analytical;
pub mod autonomy_loop;
pub mod blueprint;
pub mod bsp;
pub mod building;
pub mod csg;
pub mod design_loop;
pub mod export;
#[cfg(feature = "analytical")]
pub mod generative;
pub mod import;
pub mod infill;
pub mod manufacturing;
pub mod material_handling;
pub mod mesh;
pub mod primitives;
pub mod printer_control;
pub mod simulator;
pub mod slicer;
pub mod thought;
pub mod toolpath;
pub mod validate;

pub mod cincinnati_live;
pub mod defect_prediction;
pub mod hardware_config;
pub mod nurbs;
pub mod step_import;

pub use bsp::{csg_intersect, csg_subtract};
pub use csg::{BooleanOp, CSGNode, Primitive, Transform3D};
pub use export::{export_3mf, export_stl};
pub use import::{StlError, parse_ascii_stl, parse_binary_stl, parse_stl};
pub use infill::{InfillConfig, InfillPattern, generate_infill, generate_infill_for_layer};
pub use mesh::TriangleMesh;
pub use primitives::*;
pub use simulator::{ForceHV, PhysicsBackend, SimState};
pub use slicer::{Contour, Point2, Segment2, SliceConfig, SliceLayer, slice_mesh, slice_mesh_at_z};
pub use thought::GeometricThought;
pub use toolpath::{GCodeCommand, GCodeProgram, ToolpathConfig, generate_gcode};
pub use validate::{ValidationReport, validate_mesh};

pub use cincinnati_live::{
    AnomalyAlert, AnomalyType, ChannelStats, CincinnatiMonitor, CincinnatiMonitorConfig,
    SensorReading,
};
