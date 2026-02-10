//! Symthaea Core Engine Facade
//!
//! Thin wrapper crate that re-exports the stable engine surface from the
//! larger `symthaea` crate:
//! - HDC primitives (`hdc`)
//! - Φ computation (`phi_engine`)
//! - Mycelix integration helpers (`mycelix` module)
//!
//! This crate exists so that downstream systems (like Mycelix) can depend
//! on a small, clearly-defined surface today, and later migrate to a
//! physically separated core without changing their imports.

pub use symthaea::hdc;
pub use symthaea::phi_engine;
pub use symthaea::mycelix;

