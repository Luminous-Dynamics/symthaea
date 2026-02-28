//! Symthaea Fabrication Kernel
//!
//! HDC-to-Mesh bridge: geometric primitives encoded as hypervectors,
//! CSG boolean operations, triangle mesh tessellation, STL/3MF export,
//! and physics simulation abstraction layer.

pub mod primitives;
pub mod csg;
pub mod mesh;
pub mod bsp;
pub mod export;
pub mod import;
pub mod validate;
pub mod thought;
pub mod simulator;
#[cfg(feature = "analytical")]
pub mod analytical;

pub use primitives::*;
pub use csg::{CSGNode, BooleanOp, Primitive, Transform3D};
pub use mesh::TriangleMesh;
pub use bsp::{csg_subtract, csg_intersect};
pub use export::{export_stl, export_3mf};
pub use import::{parse_stl, parse_binary_stl, parse_ascii_stl, StlError};
pub use validate::{validate_mesh, ValidationReport};
pub use thought::GeometricThought;
pub use simulator::{PhysicsBackend, ForceHV, SimState};
