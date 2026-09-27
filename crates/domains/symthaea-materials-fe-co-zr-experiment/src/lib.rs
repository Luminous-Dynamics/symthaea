// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Public orchestration API for the Fe-Co-Zr retrospective experiment.
//!
//! Runtime readiness is intentionally not exposed through this library facade.
//! Static configuration and preparation are not executable-readiness authority;
//! the executable gate is the dedicated `preflight-runtime` CLI path and its
//! create-new runtime-readiness evidence.

#![deny(unsafe_code)]
#![warn(missing_docs)]

mod orchestration_driver;

pub use orchestration_driver::{
    ExperimentError, LocalExperimentConfig, PreparedExperiment, acquire_snapshot,
    acquisition_receipt, audit_authorized_targets, construct_authorized_targets,
    decompress_snapshot, extract_historical_corpus, import_database, inventory_database,
    preflight_database, prepare_experiment, prepare_extraction_plan, prepare_import_plan,
    read_json, verify_stored_audit, write_json_new,
};
