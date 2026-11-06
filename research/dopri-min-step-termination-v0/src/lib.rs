//! Isolated test harness for the actual Symthaea Dormand-Prince implementation.
//!
//! This harness compiles the checked-in source file directly. It does not use a
//! stub for the numerical solver and does not import the HDC or FEP application.

#[path = "../../../src/dynamics/ode_solvers.rs"]
pub mod ode_solvers;
