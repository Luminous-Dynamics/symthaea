// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Materials design, aging prediction, and HDC-based similarity search.
//!
//! Genesis Mission Challenge 9: Materials by Design.
//! Encodes material properties into 16,384-dimensional hypervectors,
//! predicts aging via O(1) CfC closed-form temporal jumps, and
//! provides constraint-filtered HDC similarity search.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod aging;
pub mod compound_stability;
pub mod database;
pub mod encoder;
pub mod evidence;
pub mod haptic_prober;
pub mod hea_screening;
pub mod irradiation_evidence;
pub mod mining;
pub mod properties;
pub mod strategic;
pub mod thermodynamic_evidence;
pub mod wolverine_ablation;

pub use aging::{AGING_HORIZON_LABELS, AGING_HORIZONS, AgingPrediction, MaterialAgingModel};
pub use database::{MaterialDatabase, MaterialSearchResult};
pub use encoder::MaterialHdcEncoder;
pub use evidence::{
    MaterialsClaimAuthority, MaterialsEvidenceChain, MaterialsEvidenceError, MaterialsEvidenceKind,
    MaterialsEvidenceRecord, MaterialsEvidenceStage,
};
pub use hea_screening::{
    ClassicHeaScreeningFlags, HeaElementDescriptor, HeaScreeningDescriptors, HeaScreeningError,
    PairMixingEnthalpy, VecStructureTendency, calculate_hea_descriptors,
};
pub use irradiation_evidence::{
    CompositionDpaEvidence, DamageMetricModel, DisplacementCrossSectionBin,
    ElementDisplacementCurve, ElementDpaContribution, IRRADIATION_COMPOSITION_PPM_TOTAL,
    IrradiationCompositionComponent, IrradiationEvidenceError, IrradiationEvidenceInput,
    NeutronSpectrumBin, calculate_composition_dpa_evidence,
};
pub use mining::{
    MINING_HORIZON_LABELS, MINING_HORIZONS, MiningFepAction, MiningFepAgent, MiningHdcEncoder,
    MiningPredictor, MiningReading,
};
pub use properties::{MaterialCategory, MaterialProperty};
pub use strategic::{
    STRATEGIC_HORIZON_LABELS, STRATEGIC_HORIZONS, StrategicFepAction, StrategicFepAgent,
    StrategicHdcEncoder, StrategicPredictor, StrategicReading,
};
pub use thermodynamic_evidence::{
    NormalizedThermodynamicEvidence, ThermodynamicDatabase, ThermodynamicEvidenceError,
    ThermodynamicEvidenceOrigin,
};
pub use wolverine_ablation::{
    AtomicFractionPpm, COMPOSITION_PPM_TOTAL, CONTROLLED_ER_PPM, WOLVERINE_SCREENING_TABLE_ID,
    WolverineAblationError, WolverineAblationManifest, WolverineCandidate, WolverineCandidateRole,
    generate_wolverine_ablation_manifest, screen_wolverine_candidate,
};
