#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]
//! # Symthaea Cell Foundry
//!
//! Cell culture control, iPSC reprogramming, in vitro gametogenesis (IVG),
//! somatic cell nuclear transfer (SCNT), and multi-scale biological prediction.
//! Uses 16,384D HDC holographic cell state encoding and O(1) CfC closed-form
//! temporal jumps to predict cell behavior from 1 hour to 9 months.

pub mod cell_encoder;
pub mod culture_controller;
pub mod epigenetics;
pub mod ethics_gate;
pub mod fep_agent;
pub mod hydrology;
pub mod ivg_protocol;
pub mod lab_controller;
pub mod meiosis_monitor;
pub mod mock;
pub mod multi_scale_predictor;
pub mod nuclear_transfer;
pub mod quality_control;
pub mod reprogramming;
pub mod types;

// Re-export key types
pub use cell_encoder::encode_cell_state;
pub use culture_controller::{CultureAdjustment, CultureController};
pub use epigenetics::{encode_methylation_pattern, methylation_similarity, validate_imprinting};
pub use ethics_gate::EthicsGate;
pub use fep_agent::{CultureAction, CultureFepAgent};
pub use ivg_protocol::IvgProtocol;
pub use meiosis_monitor::MeiosisMonitor;
pub use mock::{MockCellPopulation, MockIncubator};
pub use multi_scale_predictor::MultiScalePredictor;
pub use nuclear_transfer::ScntProtocol;
pub use quality_control::{assess_quality, check_karyotype, check_pluripotency_markers};
pub use reprogramming::ReprogrammingProtocol;
pub use types::*;

// Genesis Mission Challenge 10: Autonomous Lab Controller
pub use lab_controller::{InstrumentAdapter, LabController, LabProtocol, MockInstrument, ProtocolStep};

// Genesis Mission Challenge 8: Water Prediction
pub use hydrology::{
    HydrologicalPredictor, WaterFepAction, WaterFepAgent, WaterHdcEncoder, WaterQualityReading,
    WATER_HORIZONS,
};
