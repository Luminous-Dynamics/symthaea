#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(clippy::needless_range_loop)]
//! # Symthaea Genomics
//!
//! DNA assembly, damage modeling, and repair for degraded/ancient samples.
//! Uses 16,384D HDC hypervectors for k-mer overlap detection and
//! O(1) CfC temporal jumps for modeling DNA degradation over arbitrary timespans.
//!
//! ## Architecture
//!
//! The crate is organized around a pipeline for processing ancient/degraded DNA:
//!
//! 1. **Damage Modeling** ([`damage_model`]): Arrhenius-kinetics damage prediction
//! 2. **Assembly** ([`assembly`]): HDC-based overlap-layout-consensus assembly
//! 3. **Error Correction** ([`error_correction`]): Majority-voting across overlapping reads
//! 4. **Quality Assessment** ([`quality`]): N50, coverage, completeness metrics
//! 5. **Repair Planning** ([`repair_planner`]): Strategy selection for repairable damage
//! 6. **FEP Agent** ([`fep_agent`]): Active sequencing agent for uncertainty reduction
//! 7. **Temporal Degradation** ([`temporal_degradation`]): O(1) CfC temporal jumps
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea_genomics::{MockSequencer, DamageModel, HdcAssembler, QualityAssessor};
//!
//! let genome = MockSequencer::generate_genome(1000, 42);
//! let model = DamageModel::new(15.0);
//! let mut rng = rand::thread_rng();
//! let sequencer = MockSequencer::new(150, 0.01, 10, model.clone());
//! let reads = sequencer.sequence(&genome, 5000.0, &mut rng);
//! let assembler = HdcAssembler::default();
//! let contigs = assembler.assemble(&reads);
//! let quality = QualityAssessor::assess_assembly(&contigs);
//! ```

pub mod assembly;
pub mod damage_model;
pub mod error_correction;
pub mod fep_agent;
pub mod mock;
pub mod quality;
pub mod repair_planner;
pub mod temporal_degradation;
pub mod types;

// Re-export key types
pub use assembly::HdcAssembler;
pub use damage_model::DamageModel;
pub use error_correction::ErrorCorrector;
pub use fep_agent::{SequencingAction, SequencingFepAgent};
pub use mock::MockSequencer;
pub use quality::QualityAssessor;
pub use repair_planner::RepairPlanner;
pub use temporal_degradation::DegradationModel;
pub use types::*;
