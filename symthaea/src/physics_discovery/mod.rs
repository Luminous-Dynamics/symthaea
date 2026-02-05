//! # Physics Discovery Module
//!
//! AI-driven physics discovery using Hyperdimensional Computing (HDC) and
//! Integrated Information Theory (Phi). This module enables:
//!
//! - **Anomaly Detection**: Find unexpected patterns in experimental data
//! - **Symbolic Regression**: Discover mathematical relationships
//! - **Phase Detection**: Identify phase transitions and critical phenomena
//! - **Cross-Domain Transfer**: Apply patterns from one domain to another
//!
//! ## Architecture
//!
//! ```text
//! Raw Data → HDC Encoding → Pattern Space → Phi Analysis → Discoveries
//!     ↑                          ↓
//!     └── Hypothesis Testing ←───┘
//! ```
//!
//! ## Key Insight
//!
//! Physical laws often manifest as geometric structures in measurement space.
//! HDC naturally represents these structures as hypervectors, and Phi measures
//! their "integrated information" - how much the parts depend on each other.
//! High Phi indicates non-trivial structure worth investigating.

pub mod encoders;
pub mod anomaly;
pub mod symbolic;
pub mod patterns;

pub use encoders::{PhysicsEncoder, PhysicsEncoderConfig, EncodedMeasurement};
pub use anomaly::{AnomalyDetector, AnomalyReport, AnomalyType};
pub use symbolic::{SymbolicRegressor, DiscoveredEquation, EquationFitness};
pub use patterns::{PatternMatcher, PhysicsPattern, PatternLibrary};
