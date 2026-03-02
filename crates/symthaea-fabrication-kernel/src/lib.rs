//! Symthaea Fabrication Kernel
//!
//! HDC-to-Mesh bridge: geometric primitives encoded as hypervectors,
//! CSG boolean operations, triangle mesh tessellation, STL/3MF export,
//! and physics simulation abstraction layer.

#[cfg(feature = "analytical")]
pub mod analytical;
pub mod bsp;
pub mod building;
pub mod csg;
pub mod design_loop;
pub mod export;
pub mod import;
pub mod manufacturing;
pub mod mesh;
pub mod primitives;
pub mod simulator;
pub mod thought;
pub mod validate;

pub use bsp::{csg_intersect, csg_subtract};
pub use csg::{BooleanOp, CSGNode, Primitive, Transform3D};
pub use export::{export_3mf, export_stl};
pub use import::{parse_ascii_stl, parse_binary_stl, parse_stl, StlError};
pub use mesh::TriangleMesh;
pub use primitives::*;
pub use simulator::{ForceHV, PhysicsBackend, SimState};
pub use thought::GeometricThought;
pub use validate::{validate_mesh, ValidationReport};
