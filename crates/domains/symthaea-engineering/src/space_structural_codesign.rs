// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dry-run proposal emission for cislunar structural co-design.
//!
//! This module does not solve structures, choose a design, invoke Symtropy,
//! execute a backend, or approve construction or occupancy. It converts a
//! declared candidate and a narrow analytical model profile into a typed
//! proposal for the independent Symtropy receiving-side admission validator.
//!
//! V0.1 admits only synthetic reference requests for idealized rotating rings
//! and radial spokes. All other structural physics must abstain explicitly.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Version of the cross-project structural proposal contract.
pub const SPACE_STRUCT_PROPOSAL_SCHEMA_VERSION: &str =
    "space-structural-analysis-proposal/0.1.0";

/// Exact draft source head of the current Symtropy rotating ring/spoke reference.
/// A draft reference is never promoted to a merged or physically qualified model.
pub const SYMTROPY_PRESTRESS_DRAFT_HEAD: &str =
    "540fe6d2fc2fd99daa5277cbc2fe768db722cda9";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub id: String,
    pub generation: String,
    pub sha256: String,
}

impl ArtifactRef {
    pub fn validate(&self) -> bool {
        !self.id.trim().is_empty()
            && !self.generation.trim().is_empty()
            && self.sha256.len() == 64
            && self
                .sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuralModelProfile {
    ThinRotatingRingV0_1,
    UniformRadialSpokeV0_1,
    RingSpokeAssembly,
    SegmentedRingJoint,
    CounterRotatingHabitat,
    OrbitalTruss3d,
    PressureBoundary,
    OffEarthStructuralProduct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaterialEvidenceState {
    AnalyticalFixtureOnly,
    ReferenceOnly,
    BatchQualifiedForProfile,
    Unresolved,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntendedUse {
    SyntheticReferenceCheck,
    BoundedEngineeringScreen,
    PhysicalQualification,
    FlightOrOccupancyDecision,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PhysicsTerm {
    SteadyUniformRotation,
    ThinRingMembrane,
    SelfCentrifugalLoad,
    RadialAxialElasticity,
    RingSpokeCoupling,
    JointCompliance,
    FlexibleModes,
    GyroscopicDynamics,
    ShellMechanics,
    PressureBoundary,
    Contact,
    Fatigue,
    Buckling,
    ThermalStress,
    NonlinearMaterial,
    LinearElasticSolid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Observable {
    HoopStress,
    HoopStrain,
    RadialExtension,
    AxialForceDistribution,
    AxialStressDistribution,
    TipExtension,
    Displacement,
    Reaction,
    NaturalFrequency,
    ModeShape,
    JointLoad,
    PressureIntegrity,
    FatigueLife,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalDisposition {
    EmittedForSymtropyAdmission,
    Abstain,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalReason {
    ReferenceMalformed,
    UnsupportedModelProfile,
    UnsupportedPhysics,
    UnsupportedObservable,
    IntendedUseExceedsDraftClaimCeiling,
    MaterialEvidenceInsufficient,
    MaterialBatchOrProcessUnresolved,
    EmptyPhysicsOrObservableSet,
    DuplicatePhysicsOrObservable,
    OffEarthMaterialNotAdmitted,
    ModelProfileIdentityMismatch,
    BackendIdentityMismatch,
    MaterialProfileMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralCandidateContext {
    pub mission: ArtifactRef,
    pub candidate: ArtifactRef,
    pub geometry: ArtifactRef,
    pub material: ArtifactRef,
    pub process_batch: Option<ArtifactRef>,
    pub material_qualified_profile_id: Option<String>,
    pub load_case: ArtifactRef,
    pub frame: ArtifactRef,
    pub material_evidence: MaterialEvidenceState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralProposalIntent {
    pub proposal_id: String,
    /// Caller-supplied canonical request identity; this module never invents a digest.
    pub request: ArtifactRef,
    pub context: StructuralCandidateContext,
    pub model_profile: StructuralModelProfile,
    pub model_profile_ref: ArtifactRef,
    pub backend_ref: ArtifactRef,
    pub requested_physics: Vec<PhysicsTerm>,
    pub requested_observables: Vec<Observable>,
    pub intended_use: IntendedUse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralAnalysisProposal {
    pub schema_version: String,
    pub proposal_id: String,
    pub request: ArtifactRef,
    pub mission: ArtifactRef,
    pub candidate: ArtifactRef,
    pub geometry: ArtifactRef,
    pub material: ArtifactRef,
    pub process_batch: Option<ArtifactRef>,
    pub load_case: ArtifactRef,
    pub frame: ArtifactRef,
    pub model_profile: StructuralModelProfile,
    pub model_profile_ref: ArtifactRef,
    pub backend_ref: ArtifactRef,
    pub model_profile_version: String,
    pub model_source_repository: String,
    pub model_source_commit: String,
    pub model_maturity: String,
    pub requested_physics: Vec<PhysicsTerm>,
    pub requested_observables: Vec<Observable>,
    pub intended_use: IntendedUse,
    pub material_evidence: MaterialEvidenceState,
    pub mesh_required: bool,
    pub mesh: Option<ArtifactRef>,
    pub comparison_reference: Option<ArtifactRef>,
    pub claim_ceiling: String,
    pub solver_execution_requested: bool,
    pub execution_authorized: bool,
    pub design_release_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralProposalResult {
    pub disposition: ProposalDisposition,
    pub proposal: Option<StructuralAnalysisProposal>,
    pub reason_codes: Vec<ProposalReason>,
    pub solver_execution_invoked: bool,
    pub design_release_authorized: bool,
    pub physical_validation: String,
}

/// Convert a design intent into a request for independent Symtropy admission.
///
/// The emitter is a pure function. It performs no numerical calculation and
/// does not contact or invoke Symtropy. It can emit only a narrow synthetic
/// reference request for the exact draft ring/spoke profiles.
pub fn emit_structural_proposal(intent: &StructuralProposalIntent) -> StructuralProposalResult {
    let mut reasons = Vec::new();

    if intent.proposal_id.trim().is_empty() {
        reasons.push(ProposalReason::ReferenceMalformed);
    }

    let refs = [
        &intent.request,
        &intent.context.mission,
        &intent.context.candidate,
        &intent.context.geometry,
        &intent.context.material,
        &intent.context.load_case,
        &intent.context.frame,
    ];
    if refs.iter().any(|reference| !reference.validate())
        || intent
            .context
            .process_batch
            .as_ref()
            .is_some_and(|reference| !reference.validate())
    {
        reasons.push(ProposalReason::ReferenceMalformed);
    }

    if !intent.model_profile_ref.validate() || !intent.backend_ref.validate() {
        reasons.push(ProposalReason::ReferenceMalformed);
    }

    let expected_profile_id = match intent.model_profile {
        StructuralModelProfile::ThinRotatingRingV0_1 => "ARTGRAV-THIN-RING-REF-V0_1",
        StructuralModelProfile::UniformRadialSpokeV0_1 => "ARTGRAV-RADIAL-SPOKE-REF-V0_1",
        _ => "",
    };
    if expected_profile_id.is_empty()
        || intent.model_profile_ref.id != expected_profile_id
        || intent.model_profile_ref.generation != "0.1.0"
    {
        reasons.push(ProposalReason::ModelProfileIdentityMismatch);
    }
    if intent.backend_ref.id != "artgrav-prestress-reference"
        || intent.backend_ref.generation != "0.1.0"
    {
        reasons.push(ProposalReason::BackendIdentityMismatch);
    }

    let (supported_physics, supported_observables) = match intent.model_profile {
        StructuralModelProfile::ThinRotatingRingV0_1 => (
            vec![PhysicsTerm::SteadyUniformRotation, PhysicsTerm::ThinRingMembrane],
            vec![
                Observable::HoopStress,
                Observable::HoopStrain,
                Observable::RadialExtension,
            ],
        ),
        StructuralModelProfile::UniformRadialSpokeV0_1 => (
            vec![
                PhysicsTerm::SteadyUniformRotation,
                PhysicsTerm::SelfCentrifugalLoad,
                PhysicsTerm::RadialAxialElasticity,
            ],
            vec![
                Observable::AxialForceDistribution,
                Observable::AxialStressDistribution,
                Observable::TipExtension,
            ],
        ),
        _ => {
            reasons.push(ProposalReason::UnsupportedModelProfile);
            (Vec::new(), Vec::new())
        }
    };

    if intent.requested_physics.is_empty() || intent.requested_observables.is_empty() {
        reasons.push(ProposalReason::EmptyPhysicsOrObservableSet);
    }

    let unique_physics: BTreeSet<_> = intent.requested_physics.iter().collect();
    let unique_observables: BTreeSet<_> = intent.requested_observables.iter().collect();
    if unique_physics.len() != intent.requested_physics.len()
        || unique_observables.len() != intent.requested_observables.len()
    {
        reasons.push(ProposalReason::DuplicatePhysicsOrObservable);
    }

    if intent
        .requested_physics
        .iter()
        .any(|term| !supported_physics.contains(term))
    {
        reasons.push(ProposalReason::UnsupportedPhysics);
    }

    if intent
        .requested_observables
        .iter()
        .any(|observable| !supported_observables.contains(observable))
    {
        reasons.push(ProposalReason::UnsupportedObservable);
    }

    if intent.intended_use != IntendedUse::SyntheticReferenceCheck {
        reasons.push(ProposalReason::IntendedUseExceedsDraftClaimCeiling);
    }

    match intent.context.material_evidence {
        MaterialEvidenceState::AnalyticalFixtureOnly => {}
        MaterialEvidenceState::ReferenceOnly | MaterialEvidenceState::Unresolved => {
            reasons.push(ProposalReason::MaterialEvidenceInsufficient);
        }
        MaterialEvidenceState::BatchQualifiedForProfile => {
            // A batch qualification is profile-specific; never transfer it to a
            // different analytical/numerical model by material-name similarity.
            if intent.context.material_qualified_profile_id.as_deref()
                != Some(expected_profile_id)
            {
                reasons.push(ProposalReason::MaterialProfileMismatch);
            }
            // These two draft analytical profiles do not yet admit batch-specific
            // engineering property evidence as a basis for a real design decision.
            reasons.push(ProposalReason::OffEarthMaterialNotAdmitted);
        }
        MaterialEvidenceState::Rejected => {
            reasons.push(ProposalReason::MaterialEvidenceInsufficient);
        }
    }

    if intent.context.material_evidence != MaterialEvidenceState::AnalyticalFixtureOnly
        && intent.context.process_batch.is_none()
    {
        reasons.push(ProposalReason::MaterialBatchOrProcessUnresolved);
    }

    reasons.sort();
    reasons.dedup();

    if !reasons.is_empty() {
        return StructuralProposalResult {
            disposition: ProposalDisposition::Abstain,
            proposal: None,
            reason_codes: reasons,
            solver_execution_invoked: false,
            design_release_authorized: false,
            physical_validation: "UNRESOLVED".to_owned(),
        };
    }

    let proposal = StructuralAnalysisProposal {
        schema_version: SPACE_STRUCT_PROPOSAL_SCHEMA_VERSION.to_owned(),
        proposal_id: intent.proposal_id.clone(),
        request: intent.request.clone(),
        mission: intent.context.mission.clone(),
        candidate: intent.context.candidate.clone(),
        geometry: intent.context.geometry.clone(),
        material: intent.context.material.clone(),
        process_batch: intent.context.process_batch.clone(),
        load_case: intent.context.load_case.clone(),
        frame: intent.context.frame.clone(),
        model_profile: intent.model_profile,
        model_profile_ref: intent.model_profile_ref.clone(),
        backend_ref: intent.backend_ref.clone(),
        model_profile_version: "0.1.0".to_owned(),
        model_source_repository: "Luminous-Dynamics/symtropy".to_owned(),
        model_source_commit: SYMTROPY_PRESTRESS_DRAFT_HEAD.to_owned(),
        model_maturity: "DRAFT_ANALYTICAL_REFERENCE".to_owned(),
        requested_physics: intent.requested_physics.clone(),
        requested_observables: intent.requested_observables.clone(),
        intended_use: intent.intended_use,
        material_evidence: intent.context.material_evidence,
        mesh_required: false,
        mesh: None,
        comparison_reference: None,
        claim_ceiling: "SYNTHETIC_REFERENCE_CHECK".to_owned(),
        solver_execution_requested: false,
        execution_authorized: false,
        design_release_authorized: false,
    };

    StructuralProposalResult {
        disposition: ProposalDisposition::EmittedForSymtropyAdmission,
        proposal: Some(proposal),
        reason_codes: Vec::new(),
        solver_execution_invoked: false,
        design_release_authorized: false,
        physical_validation: "UNRESOLVED".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(id: &str, generation: &str, digit: char) -> ArtifactRef {
        ArtifactRef {
            id: id.to_owned(),
            generation: generation.to_owned(),
            sha256: digit.to_string().repeat(64),
        }
    }

    fn ring_intent() -> StructuralProposalIntent {
        StructuralProposalIntent {
            proposal_id: "SPACE-STRUCT-RING-PROPOSAL-001".to_owned(),
            request: reference("SPACE-STRUCT-RING-REQUEST-001", "r1", '9'),
            context: StructuralCandidateContext {
                mission: reference("MISSION-CISLUNAR-001", "m1", '1'),
                candidate: reference("AG-RING-CANDIDATE-001", "c1", '2'),
                geometry: reference("RING-GEOMETRY-001", "g1", '3'),
                material: reference("ANALYTICAL-MATERIAL-FIXTURE-001", "mat1", '4'),
                process_batch: None,
                material_qualified_profile_id: None,
                load_case: reference("STEADY-RING-SPIN-001", "l1", '5'),
                frame: reference("RING-LOCAL-FRAME-001", "f1", '6'),
                material_evidence: MaterialEvidenceState::AnalyticalFixtureOnly,
            },
            model_profile: StructuralModelProfile::ThinRotatingRingV0_1,
            model_profile_ref: reference("ARTGRAV-THIN-RING-REF-V0_1", "0.1.0", '7'),
            backend_ref: reference("artgrav-prestress-reference", "0.1.0", '8'),
            requested_physics: vec![
                PhysicsTerm::SteadyUniformRotation,
                PhysicsTerm::ThinRingMembrane,
            ],
            requested_observables: vec![
                Observable::HoopStress,
                Observable::HoopStrain,
                Observable::RadialExtension,
            ],
            intended_use: IntendedUse::SyntheticReferenceCheck,
        }
    }

    #[test]
    fn emits_restricted_synthetic_ring_proposal_without_execution() {
        let result = emit_structural_proposal(&ring_intent());
        assert_eq!(result.disposition, ProposalDisposition::EmittedForSymtropyAdmission);
        let proposal = result.proposal.expect("proposal should be emitted");
        assert_eq!(proposal.model_source_commit, SYMTROPY_PRESTRESS_DRAFT_HEAD);
        assert_eq!(proposal.request.id, "SPACE-STRUCT-RING-REQUEST-001");
        assert!(!proposal.mesh_required);
        assert!(proposal.mesh.is_none());
        assert!(proposal.comparison_reference.is_none());
        assert_eq!(proposal.model_profile_ref.id, "ARTGRAV-THIN-RING-REF-V0_1");
        assert_eq!(proposal.backend_ref.id, "artgrav-prestress-reference");
        assert_eq!(proposal.claim_ceiling, "SYNTHETIC_REFERENCE_CHECK");
        assert!(!proposal.solver_execution_requested);
        assert!(!proposal.execution_authorized);
        assert!(!proposal.design_release_authorized);
        assert!(!result.solver_execution_invoked);
        assert_eq!(result.physical_validation, "UNRESOLVED");
    }

    #[test]
    fn rejects_flexible_modes_from_thin_ring_profile() {
        let mut intent = ring_intent();
        intent.requested_physics.push(PhysicsTerm::GyroscopicDynamics);
        intent.requested_observables.push(Observable::NaturalFrequency);
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::UnsupportedPhysics));
        assert!(result.reason_codes.contains(&ProposalReason::UnsupportedObservable));
        assert!(result.proposal.is_none());
    }

    #[test]
    fn rejects_joint_or_ring_spoke_assembly_profile() {
        let mut intent = ring_intent();
        intent.model_profile = StructuralModelProfile::RingSpokeAssembly;
        intent.requested_physics.push(PhysicsTerm::RingSpokeCoupling);
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::UnsupportedModelProfile));
    }

    #[test]
    fn rejects_non_synthetic_use_of_draft_reference() {
        let mut intent = ring_intent();
        intent.intended_use = IntendedUse::BoundedEngineeringScreen;
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::IntendedUseExceedsDraftClaimCeiling));
    }

    #[test]
    fn unresolved_off_earth_material_does_not_become_structural_property() {
        let mut intent = ring_intent();
        intent.context.material_evidence = MaterialEvidenceState::BatchQualifiedForProfile;
        intent.context.process_batch = Some(reference("LUNAR-BATCH-001", "b1", '7'));
        intent.context.material_qualified_profile_id =
            Some("ARTGRAV-THIN-RING-REF-V0_1".to_owned());
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::OffEarthMaterialNotAdmitted));
    }

    #[test]
    fn missing_material_batch_is_not_inferred() {
        let mut intent = ring_intent();
        intent.context.material_evidence = MaterialEvidenceState::ReferenceOnly;
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::MaterialEvidenceInsufficient));
        assert!(result.reason_codes.contains(&ProposalReason::MaterialBatchOrProcessUnresolved));
    }

    #[test]
    fn malformed_reference_is_rejected() {
        let mut intent = ring_intent();
        intent.context.geometry.sha256 = "not-a-digest".to_owned();
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::ReferenceMalformed));
    }

    #[test]
    fn empty_observable_set_is_rejected() {
        let mut intent = ring_intent();
        intent.requested_observables.clear();
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::EmptyPhysicsOrObservableSet));
    }

    #[test]
    fn mismatched_model_profile_identity_abstains() {
        let mut intent = ring_intent();
        intent.model_profile_ref.id = "ARTGRAV-RADIAL-SPOKE-REF-V0_1".to_owned();
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::ModelProfileIdentityMismatch));
    }

    #[test]
    fn mismatched_backend_identity_abstains() {
        let mut intent = ring_intent();
        intent.backend_ref.id = "unallowlisted-structural-backend".to_owned();
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::BackendIdentityMismatch));
    }

    #[test]
    fn model_profile_generation_is_exact() {
        let mut intent = ring_intent();
        intent.model_profile_ref.generation = "0.2.0".to_owned();
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::Abstain);
        assert!(result.reason_codes.contains(&ProposalReason::ModelProfileIdentityMismatch));
    }

    #[test]
    fn emitted_proposal_serializes_receiver_identity_and_no_execution_authority() {
        let result = emit_structural_proposal(&ring_intent());
        let proposal = result.proposal.expect("proposal should be emitted");
        let value = serde_json::to_value(proposal).expect("proposal should serialize");
        assert_eq!(value["request"]["id"], "SPACE-STRUCT-RING-REQUEST-001");
        assert_eq!(value["model_profile_ref"]["id"], "ARTGRAV-THIN-RING-REF-V0_1");
        assert_eq!(value["backend_ref"]["id"], "artgrav-prestress-reference");
        assert_eq!(value["mesh_required"], false);
        assert_eq!(value["mesh"], serde_json::Value::Null);
        assert_eq!(value["comparison_reference"], serde_json::Value::Null);
        assert_eq!(value["solver_execution_requested"], false);
        assert_eq!(value["execution_authorized"], false);
        assert_eq!(value["design_release_authorized"], false);
    }

    #[test]
    fn emits_restricted_synthetic_radial_spoke_proposal() {
        let mut intent = ring_intent();
        intent.proposal_id = "SPACE-STRUCT-SPOKE-PROPOSAL-001".to_owned();
        intent.model_profile = StructuralModelProfile::UniformRadialSpokeV0_1;
        intent.requested_physics = vec![
            PhysicsTerm::SteadyUniformRotation,
            PhysicsTerm::SelfCentrifugalLoad,
            PhysicsTerm::RadialAxialElasticity,
        ];
        intent.requested_observables = vec![
            Observable::AxialForceDistribution,
            Observable::AxialStressDistribution,
            Observable::TipExtension,
        ];
        let result = emit_structural_proposal(&intent);
        assert_eq!(result.disposition, ProposalDisposition::EmittedForSymtropyAdmission);
        assert!(result.proposal.is_some());
    }
}
