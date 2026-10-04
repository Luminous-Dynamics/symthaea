// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Modality-neutral communication analysis.
//!
//! The types in this crate deliberately separate detecting a signal, discovering
//! recurring units and structure, and making grounded claims about reference or
//! intent. A model must not report a capability above the evidence it supplies.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub mod adapters;
pub mod animal;
pub mod artifact;
pub mod benchmark;
pub mod human;
#[cfg(feature = "hdc-codec")]
pub mod hdc_codec;
#[cfg(feature = "hdc-codec")]
pub mod hdc_interlingua;
#[cfg(feature = "hdc-codec")]
pub mod hdc_ontology;
pub mod interlingua;
pub mod metrics;
pub mod neurosemantic;
pub mod pilot;
pub mod pipeline;
pub mod provider;
pub mod run;
pub mod unknown;

/// The strongest claim supported by a result, ordered from weakest to strongest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CapabilityLevel {
    #[default]
    Signal,
    Unit,
    Structure,
    Reference,
    Intent,
    Dialogue,
}
