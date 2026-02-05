//! # Markets Module: HDC-Based Financial Pattern Recognition
//!
//! This module demonstrates using Hyperdimensional Computing for
//! financial market pattern recognition and simulation.
//!
//! ## Key Concepts
//!
//! - **Market State HV**: Encodes OHLCV (Open, High, Low, Close, Volume) data
//!   plus technical indicators into a single high-dimensional vector
//! - **Pattern Memory**: Stores historical patterns as HVs for similarity matching
//! - **Regime Detection**: Uses HDC clustering to identify market regimes
//! - **Simulation**: Generates synthetic market data based on learned patterns
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    MARKET HDC PIPELINE                      │
//! │                                                             │
//! │   OHLCV Data → Normalize → Feature HVs → Bind/Bundle →     │
//! │       → Market State HV → Pattern Memory → Similarity →    │
//! │       → Regime Detection / Prediction                      │
//! │                                                             │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Disclaimer
//!
//! This is for educational and research purposes only. Not financial advice.

pub mod market_state;
pub mod pattern_hdc;
pub mod simulator;

pub use market_state::{MarketStateEncoder, OHLCV, TechnicalIndicators};
pub use pattern_hdc::{PatternMemory, MarketRegime, PatternMatch};
pub use simulator::{MarketSimulator, SimulatorConfig};
