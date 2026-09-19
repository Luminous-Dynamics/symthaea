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
pub mod conditioned_property;
pub mod database;
pub mod discovery_campaign;
pub mod encoder;
pub mod evidence;
pub mod haptic_prober;
pub mod hea_screening;
pub mod irradiation_evidence;
pub mod material_subject;
pub mod mining;
pub mod multi_fidelity;
pub mod novelty_synthesis;
mod novelty_synthesis_hash;
pub mod orchestrator;
pub mod properties;
pub mod sample_lineage;
pub mod search_memory;
pub mod strategic;
pub mod technoeconomics;
pub mod thermodynamic_evidence;
pub mod wolverine_ablation;

pub use aging::{AGING_HORIZON_LABELS, AGING_HORIZONS, AgingPrediction, MaterialAgingModel};
pub use conditioned_property::{
    ConditionedPropertyError, ConditionedPropertyObservation, PropertyArtifactRef,
    PropertyConditionTag, PropertyConditions, PropertyEvidenceClass, PropertyObservationMethod,
    PropertyUncertainty,
};
pub use database::{MaterialDatabase, MaterialSearchResult};
pub use discovery_campaign::{
    CampaignBudgets, CampaignEvaluationPoint, CampaignHardConstraint, CampaignMetricKind,
    CampaignMetricRef, CampaignMetricValue, CampaignObjective, CampaignScale, CampaignStopRule,
    DiscoveryCampaign, DiscoveryCampaignError, HardConstraintRelation, MonetaryBudget,
    ObjectiveDirection, campaign_pareto_front, campaign_point_is_feasible,
};
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
pub use material_subject::{
    AtomicCompositionPpm, CompositeConstituent, CompositionGradientKnot, ElementFractionPpm,
    ElementStoichiometry, MaterialArchitecture, MaterialComposition, MaterialLayer,
    MaterialPhaseState, MaterialPhaseSubject, MaterialSubject, MaterialSubjectError,
    SUBJECT_FRACTION_PPM_TOTAL, StoichiometricComposition, SubjectLineageRef,
};
pub use mining::{
    MINING_HORIZON_LABELS, MINING_HORIZONS, MiningFepAction, MiningFepAgent, MiningHdcEncoder,
    MiningPredictor, MiningReading,
};
pub use multi_fidelity::{
    ApplicabilityAssessment, ApplicabilityState, CalibrationPair, ContradictionPolicy,
    CrossFidelityCalibration, EvaluationAgreement, EvaluationMethodClass, EvaluationOrigin,
    EvaluationResourceCost, EvaluatorRef, MultiFidelityError, MultiFidelityEvaluation,
    assess_evaluation_agreement, calibrate_against_reference,
};
pub use novelty_synthesis::{
    MetastableProcessWindow, NoveltyAssessment, NoveltyAssessmentState, NoveltySynthesisError,
    PriorArtHitClass, PriorArtNeighborhood, PriorArtSearchOutcome, PriorArtSearchRecord,
    PriorArtSourceKind, ProcessWindowVariable, SynthesisRouteAssessment, SynthesizabilityState,
};
pub use orchestrator::{
    AcquisitionStrategy, DiscoveryActionClass, DiscoveryActionProposal, DryRunMaterialsOrchestrator,
    HumanAuthorization, OrchestratorBudgetLedger, OrchestratorError,
};
pub use properties::{MaterialCategory, MaterialProperty};
pub use sample_lineage::{
    CharacterizationRun, CharacterizationStatus, ExperimentalSampleLineage, ExperimentalScale,
    PrecursorLot, SampleLineageError, SampleLineageNode, SampleNodeKind,
};
pub use search_memory::{
    MaterialSearchAttempt, MaterialsSearchMemory, SearchAttemptCost, SearchAttemptOutcome,
    SearchEvaluatorRef, SearchMemoryError, SearchNeighborhoodFingerprint,
};
pub use strategic::{
    STRATEGIC_HORIZON_LABELS, STRATEGIC_HORIZONS, StrategicFepAction, StrategicFepAgent,
    StrategicHdcEncoder, StrategicPredictor, StrategicReading,
};
pub use technoeconomics::{
    CriticalityStatus, ECONOMIC_COMPOSITION_PPM_TOTAL, EconomicCompositionComponent,
    EconomicParetoPoint, ElementCriticalityObservation, ElementEconomicContribution,
    ElementPriceObservation, MaterialTechnoeconomicInput, MaterialTechnoeconomicResult,
    PriceBasis, ProcessEconomicScenario, TechnoeconomicError, economic_pareto_front,
    evaluate_material_technoeconomics,
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
