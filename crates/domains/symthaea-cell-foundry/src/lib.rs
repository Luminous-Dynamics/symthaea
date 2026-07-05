// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop)]
//! # Symthaea Cell Foundry
//!
//! Cell culture control, iPSC reprogramming, in vitro gametogenesis (IVG),
//! somatic cell nuclear transfer (SCNT), and multi-scale biological prediction.
//! Uses 16,384D HDC holographic cell state encoding and O(1) CfC closed-form
//! temporal jumps to predict cell behavior from 1 hour to 9 months.

mod anatomical_compiler;
pub mod bioelectric;
pub mod cell_encoder;
pub mod consciousness_ethics_framework;
pub mod culture_controller;
pub mod digital_organoid;
pub mod epigenetics;
pub mod ethics_gate;
pub mod experiment_planner;
pub mod experiments;
pub mod fep_agent;
pub mod hydrology;
mod ion_channels;
pub mod ivg_protocol;
pub mod lab_controller;
pub mod meiosis_monitor;
pub mod mock;
pub mod morphogenetic_consciousness;
pub mod multi_scale_predictor;
pub mod nuclear_transfer;
pub mod organoid_pipeline;
mod packing;
pub mod quality_control;
pub mod reprogramming;
mod spatial_grid;
pub mod types;

// Re-export key types
pub use anatomical_compiler::search_intervention;
pub use bioelectric::{BioelectricState, TargetMorphology};
pub use cell_encoder::encode_cell_state;
pub use culture_controller::{CultureAdjustment, CultureController};
pub use epigenetics::{encode_methylation_pattern, methylation_similarity, validate_imprinting};
pub use ethics_gate::EthicsGate;
pub use experiments::{
    AXIS_HEAD_VMEM, AXIS_TAIL_VMEM, ConditionResult, EquifinalityResult, Perturbation,
    build_linear_axis_template, build_radial_bipolar_template, mean_vmem_in_x_band,
    run_dose_response_experiment, run_equifinality_experiment,
};
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
pub use lab_controller::{
    InstrumentAdapter, LabController, LabProtocol, MockInstrument, ProtocolStep,
};

// Genesis Mission Challenge 5: Experiment Planner
pub use experiment_planner::{
    EXPERIMENT_HORIZON_LABELS, EXPERIMENT_HORIZONS, ExperimentFepAction, ExperimentFepAgent,
    ExperimentHdcEncoder, ExperimentPredictor, ExperimentReading,
};

// Genesis Mission Challenge 8: Water Prediction
pub use hydrology::{
    HydrologicalPredictor, WATER_HORIZONS, WaterFepAction, WaterFepAgent, WaterHdcEncoder,
    WaterQualityReading,
};
