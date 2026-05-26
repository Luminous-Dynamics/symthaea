// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! mycelix-earth: Earth Evidence Mesh
//!
//! Evidence, provenance, and stewardship claims from Earth-observation data.

pub mod aoi;
pub mod background;
pub mod evidence;
pub mod features;
pub mod fractal;
pub mod governance;
pub mod hdc;
pub mod morphos;
pub mod providers;

pub use aoi::Aoi;
pub use evidence::anomaly;
pub use evidence::decay;
pub use evidence::{EarthDataSource, EvidencePacket};
pub use features::EarthFeature;
pub use fractal::FractalAuditor;
pub use governance::synthesis::{
    ConstitutionalSynthesizer, DialecticalOption, DialecticalSynthesis,
};
pub use hdc::EcologicalEncoder;
pub use hdc::aggregation::{PlanetaryReceipt, aggregate_bioregion_proofs};
pub use hdc::biome::{BiomeEncoder, BiomeTensor, EcosystemState};
pub use hdc::prediction::{DigitalTwinGate, RestorationPlan};
pub use hdc::proof::HdcBindingProof;
pub use morphos::BioregionChannelUpdate;
pub use providers::hardware::{HardwareOracle, PhysicalEnclaveProvider};
