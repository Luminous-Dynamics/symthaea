#![deny(unsafe_code)]

//! Native interoceptive regulation primitives for Symthaea.
//!
//! This crate models artificial viability state directly. It intentionally
//! contains no semantic category layer and no dependency on the cognitive loop,
//! allowing deterministic regulation experiments to remain mechanically isolated.

mod allostasis;
mod dynamics;
mod evidence;
mod homeostasis;
mod intervention;
mod qualification;
mod snapshot;
mod state;

pub use allostasis::{
    assess_allostasis, assess_allostasis_with_drive, AllostaticConfig, AllostaticReport,
};
pub use dynamics::{
    InteroceptiveDrive, InteroceptiveDynamicsConfig, InteroceptiveStepReport,
    NativeInteroceptiveModel,
};
pub use evidence::{
    ArtifactDigest, EvidenceCapsuleManifest, ForecastBasisId, EVIDENCE_CAPSULE_SCHEMA_VERSION,
};
pub use homeostasis::{assess_homeostasis, HomeostaticReport};
pub use intervention::{
    apply_intervention, InterventionKind, InterventionRecord, InteroceptiveIntervention,
};
pub use qualification::{
    GateStatus, QualificationGateReceipt, QualificationReceipt, QUALIFICATION_RECEIPT_SCHEMA_VERSION,
    REQUIRED_QUALIFICATION_GATES,
};
pub use snapshot::{
    AllostaticForecastSnapshot, InteroceptiveSnapshot, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
    INTEROCEPTIVE_SNAPSHOT_SCHEMA_VERSION,
};
pub use state::{
    NativeInteroceptiveState, ViabilityChannel, ViabilityVariable, CHANNEL_COUNT,
};
