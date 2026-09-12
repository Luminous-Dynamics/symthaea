// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Legacy qualification profile V3 and generation-bound active manifests.
//!
//! The historical source/claim/procedure registry is append-only provenance.
//! Qualification, however, needs an explicit active target for one evidence
//! generation so living vendor documentation can be rebaselined without
//! deleting or rebinding history.
//!
//! ```text
//! historical registry != active qualification manifest
//! manifest transition != deletion
//! historical evidence != active-generation evidence
//! current capture != historical byte proof
//! source provenance ready != competence
//! ```

use crate::it_qualification::ItQualificationMatrixV1;
use crate::legacy_computing::{
    LegacyComputingPackV1, LegacyCoverageStateV1, LegacyKnowledgeAreaV1,
    LegacyPlatformV1, LegacyProcedureAuthorityV1, LegacyProcedureV1,
};
use crate::legacy_qualification_profile::{
    assess_legacy_qualification_profile_v1, LegacyQualificationBlockerV1,
    LegacyQualificationProfileErrorV1, LegacyQualificationProfileV1,
};
use crate::legacy_qualification_source_ledger_v3::{
    assess_legacy_qualification_source_readiness_v3, LegacyQualificationSourceLedgerErrorV3,
    LegacyQualificationSourceLedgerV3, LegacyQualificationSourceReadinessV3,
};
use crate::legacy_source_artifacts::LegacySourceArtifactLedgerV1;
use crate::legacy_source_capture_plan_v3::{
    plan_legacy_qualification_source_captures_for_targets_v3, LegacySourceCapturePlanV3,
};
use crate::legacy_source_continuity::{
    require_legacy_source_continuity_v1, LegacySourceContinuityAssessmentV1,
    LegacySourceContinuityErrorV1,
};
use crate::standards_registry::{
    SourceRelationKindV1, SourceSnapshotIdV1, TechnicalClaimIdV1,
    TechnicalKnowledgeClaimV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3: &str =
    "symthaea-it-legacy-qualification-profile-assessment-v3";
pub const LEGACY_QUALIFICATION_MANIFEST_SCHEMA_V1: &str =
    "symthaea-it-legacy-qualification-manifest-v1";
pub const LEGACY_QUALIFICATION_GENERATION_ASSESSMENT_SCHEMA_V1: &str =
    "symthaea-it-legacy-qualification-generation-assessment-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationRequirementAssessmentV3 {
    pub platform: LegacyPlatformV1,
    pub area: LegacyKnowledgeAreaV1,
    pub knowledge_state: LegacyCoverageStateV1,
    pub matching_active_cases: usize,
    /// All V1 competency blockers except the obsolete global source predicate.
    pub non_source_blockers: BTreeSet<LegacyQualificationBlockerV1>,
    pub source_evidence_ready: bool,
}

impl LegacyQualificationRequirementAssessmentV3 {
    pub fn ready_for_evaluation(&self) -> bool {
        self.source_evidence_ready && self.non_source_blockers.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationProfileAssessmentV3 {
    pub schema_version: String,
    pub total_requirements: usize,
    pub ready_requirements: usize,
    pub source_readiness: LegacyQualificationSourceReadinessV3,
    pub requirements: Vec<LegacyQualificationRequirementAssessmentV3>,
}

/// Backward-compatible full-registry V3 qualification path.
pub fn assess_legacy_qualification_profile_v3(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacyQualificationProfileAssessmentV3, LegacyQualificationProfileErrorV3> {
    let v1 = assess_legacy_qualification_profile_v1(pack, profile, matrix)?;
    let source_readiness =
        assess_legacy_qualification_source_readiness_v3(pack, artifacts, source_ledger)?;
    require_legacy_source_continuity_v1(pack, source_ledger)?;

    let requirements = project_requirements(v1.requirements, source_readiness.source_evidence_ready);
    let ready_requirements = requirements
        .iter()
        .filter(|assessment| assessment.ready_for_evaluation())
        .count();

    Ok(LegacyQualificationProfileAssessmentV3 {
        schema_version: LEGACY_QUALIFICATION_PROFILE_ASSESSMENT_SCHEMA_V3.into(),
        total_requirements: requirements.len(),
        ready_requirements,
        source_readiness,
        requirements,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationManifestV1 {
    pub schema_version: String,
    pub generation: u64,
    /// Exact predecessor manifest commitment. Generation 1 has no predecessor.
    pub predecessor_manifest_blake3: Option<String>,
    /// Active claim identities for this qualification generation.
    pub claim_ids: BTreeSet<TechnicalClaimIdV1>,
    /// Active procedure identities for this qualification generation.
    pub procedure_ids: BTreeSet<String>,
    /// Every removed predecessor claim must map one-to-one to a successor claim.
    #[serde(default)]
    pub claim_replacements: BTreeMap<TechnicalClaimIdV1, TechnicalClaimIdV1>,
    /// Every removed predecessor procedure must map one-to-one to a successor procedure.
    #[serde(default)]
    pub procedure_replacements: BTreeMap<String, String>,
}

impl LegacyQualificationManifestV1 {
    pub fn validate_basic(
        &self,
        pack: &LegacyComputingPackV1,
    ) -> Result<(), LegacyQualificationManifestErrorV1> {
        if self.schema_version != LEGACY_QUALIFICATION_MANIFEST_SCHEMA_V1 {
            return Err(LegacyQualificationManifestErrorV1::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        if self.generation == 0 {
            return Err(LegacyQualificationManifestErrorV1::InvalidGeneration(0));
        }
        match (self.generation, &self.predecessor_manifest_blake3) {
            (1, None) => {}
            (1, Some(_)) => {
                return Err(LegacyQualificationManifestErrorV1::UnexpectedPredecessor)
            }
            (_, Some(value)) if is_blake3(value) => {}
            _ => return Err(LegacyQualificationManifestErrorV1::MissingPredecessor),
        }
        if self.generation == 1
            && (!self.claim_replacements.is_empty() || !self.procedure_replacements.is_empty())
        {
            return Err(LegacyQualificationManifestErrorV1::UnexpectedInitialReplacement);
        }
        if self.claim_ids.is_empty() {
            return Err(LegacyQualificationManifestErrorV1::EmptyClaims);
        }
        if self.procedure_ids.is_empty() {
            return Err(LegacyQualificationManifestErrorV1::EmptyProcedures);
        }
        pack.validate()
            .map_err(|err| LegacyQualificationManifestErrorV1::InvalidPack(err.to_string()))?;
        for claim_id in &self.claim_ids {
            if pack.sources.claim(claim_id).is_none() {
                return Err(LegacyQualificationManifestErrorV1::UnknownClaim(
                    claim_id.clone(),
                ));
            }
        }
        for procedure_id in &self.procedure_ids {
            if find_procedure(pack, procedure_id).is_none() {
                return Err(LegacyQualificationManifestErrorV1::UnknownProcedure(
                    procedure_id.clone(),
                ));
            }
        }
        Ok(())
    }

    pub fn required_source_revisions(
        &self,
        pack: &LegacyComputingPackV1,
    ) -> Result<BTreeSet<SourceSnapshotIdV1>, LegacyQualificationManifestErrorV1> {
        self.validate_basic(pack)?;
        let mut required = BTreeSet::new();
        for claim_id in &self.claim_ids {
            let claim = pack
                .sources
                .claim(claim_id)
                .ok_or_else(|| LegacyQualificationManifestErrorV1::UnknownClaim(claim_id.clone()))?;
            required.insert(claim.source_snapshot.clone());
        }
        for procedure_id in &self.procedure_ids {
            let procedure = find_procedure(pack, procedure_id).ok_or_else(|| {
                LegacyQualificationManifestErrorV1::UnknownProcedure(procedure_id.clone())
            })?;
            required.extend(procedure.source_snapshots.iter().cloned());
        }
        Ok(required)
    }
}

pub fn initial_legacy_qualification_manifest_v1(
    pack: &LegacyComputingPackV1,
) -> Result<LegacyQualificationManifestV1, LegacyQualificationManifestErrorV1> {
    pack.validate()
        .map_err(|err| LegacyQualificationManifestErrorV1::InvalidPack(err.to_string()))?;
    let manifest = LegacyQualificationManifestV1 {
        schema_version: LEGACY_QUALIFICATION_MANIFEST_SCHEMA_V1.into(),
        generation: 1,
        predecessor_manifest_blake3: None,
        claim_ids: pack.sources.claims().map(|claim| claim.id.clone()).collect(),
        procedure_ids: pack
            .procedures
            .iter()
            .map(|procedure| procedure.id.clone())
            .collect(),
        claim_replacements: BTreeMap::new(),
        procedure_replacements: BTreeMap::new(),
    };
    manifest.validate_basic(pack)?;
    Ok(manifest)
}

pub fn legacy_qualification_manifest_commitment_v1(
    manifest: &LegacyQualificationManifestV1,
) -> Result<String, LegacyQualificationManifestErrorV1> {
    let encoded = serde_json::to_vec(manifest)
        .map_err(|err| LegacyQualificationManifestErrorV1::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_QUALIFICATION_MANIFEST_SCHEMA_V1.as_bytes(),
    );
    frame(&mut hasher, b"manifest", &encoded);
    Ok(hasher.finalize().to_hex().to_string())
}

/// Build the normal successor shape for a rebaseline. New replacement targets
/// must already exist in the append-only pack. The returned manifest is fully
/// transition-validated before use.
pub fn build_legacy_qualification_successor_manifest_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    claim_replacements: BTreeMap<TechnicalClaimIdV1, TechnicalClaimIdV1>,
    procedure_replacements: BTreeMap<String, String>,
) -> Result<LegacyQualificationManifestV1, LegacyQualificationManifestErrorV1> {
    predecessor.validate_basic(pack)?;
    let generation = predecessor
        .generation
        .checked_add(1)
        .ok_or(LegacyQualificationManifestErrorV1::GenerationOverflow)?;
    let mut claim_ids = predecessor.claim_ids.clone();
    for old_id in claim_replacements.keys() {
        claim_ids.remove(old_id);
    }
    claim_ids.extend(claim_replacements.values().cloned());
    let mut procedure_ids = predecessor.procedure_ids.clone();
    for old_id in procedure_replacements.keys() {
        procedure_ids.remove(old_id);
    }
    procedure_ids.extend(procedure_replacements.values().cloned());

    let successor = LegacyQualificationManifestV1 {
        schema_version: LEGACY_QUALIFICATION_MANIFEST_SCHEMA_V1.into(),
        generation,
        predecessor_manifest_blake3: Some(
            legacy_qualification_manifest_commitment_v1(predecessor)?,
        ),
        claim_ids,
        procedure_ids,
        claim_replacements,
        procedure_replacements,
    };
    validate_legacy_qualification_manifest_transition_v1(pack, predecessor, &successor)?;
    Ok(successor)
}

pub fn validate_legacy_qualification_manifest_transition_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
) -> Result<(), LegacyQualificationManifestErrorV1> {
    predecessor.validate_basic(pack)?;
    successor.validate_basic(pack)?;

    let expected_generation = predecessor
        .generation
        .checked_add(1)
        .ok_or(LegacyQualificationManifestErrorV1::GenerationOverflow)?;
    if successor.generation != expected_generation {
        return Err(LegacyQualificationManifestErrorV1::GenerationDiscontinuity {
            expected: expected_generation,
            observed: successor.generation,
        });
    }
    let predecessor_commitment = legacy_qualification_manifest_commitment_v1(predecessor)?;
    if successor.predecessor_manifest_blake3.as_deref() != Some(&predecessor_commitment) {
        return Err(LegacyQualificationManifestErrorV1::PredecessorMismatch);
    }

    let removed_claims = predecessor
        .claim_ids
        .difference(&successor.claim_ids)
        .cloned()
        .collect::<BTreeSet<_>>();
    let removed_procedures = predecessor
        .procedure_ids
        .difference(&successor.procedure_ids)
        .cloned()
        .collect::<BTreeSet<_>>();

    require_exact_replacement_keys(&removed_claims, &successor.claim_replacements)?;
    require_exact_procedure_replacement_keys(
        &removed_procedures,
        &successor.procedure_replacements,
    )?;
    require_unique_claim_replacements(&successor.claim_replacements)?;
    require_unique_procedure_replacements(&successor.procedure_replacements)?;

    for old_id in &removed_claims {
        let new_id = successor.claim_replacements.get(old_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::MissingClaimReplacement(old_id.clone())
        })?;
        if !successor.claim_ids.contains(new_id) || old_id == new_id {
            return Err(LegacyQualificationManifestErrorV1::InvalidClaimReplacement {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
        let old = pack
            .sources
            .claim(old_id)
            .ok_or_else(|| LegacyQualificationManifestErrorV1::UnknownClaim(old_id.clone()))?;
        let new = pack
            .sources
            .claim(new_id)
            .ok_or_else(|| LegacyQualificationManifestErrorV1::UnknownClaim(new_id.clone()))?;
        if !claim_scope_preserved(old, new) {
            return Err(LegacyQualificationManifestErrorV1::ClaimScopeChanged {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
        if !claim_lineage_preserved(pack, old, new)? {
            return Err(LegacyQualificationManifestErrorV1::ClaimLineageChanged {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
    }

    for old_id in &removed_procedures {
        let new_id = successor.procedure_replacements.get(old_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::MissingProcedureReplacement(old_id.clone())
        })?;
        if !successor.procedure_ids.contains(new_id) || old_id == new_id {
            return Err(LegacyQualificationManifestErrorV1::InvalidProcedureReplacement {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
        let old = find_procedure(pack, old_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::UnknownProcedure(old_id.clone())
        })?;
        let new = find_procedure(pack, new_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::UnknownProcedure(new_id.clone())
        })?;
        if !procedure_scope_preserved(old, new) {
            return Err(LegacyQualificationManifestErrorV1::ProcedureScopeChanged {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
        if !procedure_lineage_preserved(pack, old, new)? {
            return Err(LegacyQualificationManifestErrorV1::ProcedureLineageChanged {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
    }

    Ok(())
}

pub fn plan_legacy_qualification_manifest_captures_v1(
    pack: &LegacyComputingPackV1,
    manifest: &LegacyQualificationManifestV1,
) -> Result<LegacySourceCapturePlanV3, LegacyQualificationManifestErrorV1> {
    manifest.validate_basic(pack)?;
    plan_legacy_qualification_source_captures_for_targets_v3(
        pack,
        &manifest.claim_ids,
        &manifest.procedure_ids,
    )
    .map_err(|err| LegacyQualificationManifestErrorV1::CapturePlan(err.to_string()))
}

pub fn assess_legacy_qualification_manifest_source_readiness_v1(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
    manifest: &LegacyQualificationManifestV1,
) -> Result<LegacyQualificationSourceReadinessV3, LegacyQualificationProfileErrorV3> {
    manifest.validate_basic(pack)?;
    let active_ledger = filtered_manifest_ledger(pack, artifacts, source_ledger, manifest)?;
    let required_source_revisions = manifest.required_source_revisions(pack)?;
    let missing_source_selections = required_source_revisions
        .iter()
        .filter(|snapshot_id| active_ledger.selection(snapshot_id).is_none())
        .cloned()
        .collect::<BTreeSet<_>>();
    let missing_claim_receipts = manifest
        .claim_ids
        .iter()
        .filter(|claim_id| active_ledger.claim_receipt(claim_id).is_none())
        .cloned()
        .collect::<BTreeSet<_>>();
    let missing_procedure_receipts = manifest
        .procedure_ids
        .iter()
        .filter(|procedure_id| active_ledger.procedure_receipt(procedure_id).is_none())
        .cloned()
        .collect::<BTreeSet<_>>();

    let selected_source_revisions =
        required_source_revisions.len() - missing_source_selections.len();
    let verified_claims = manifest.claim_ids.len() - missing_claim_receipts.len();
    let verified_procedures = manifest.procedure_ids.len() - missing_procedure_receipts.len();
    let source_evidence_ready = !required_source_revisions.is_empty()
        && missing_source_selections.is_empty()
        && missing_claim_receipts.is_empty()
        && missing_procedure_receipts.is_empty();

    Ok(LegacyQualificationSourceReadinessV3 {
        required_source_revisions: required_source_revisions.len(),
        selected_source_revisions,
        required_claims: manifest.claim_ids.len(),
        verified_claims,
        required_procedures: manifest.procedure_ids.len(),
        verified_procedures,
        missing_source_selections,
        missing_claim_receipts,
        missing_procedure_receipts,
        source_evidence_ready,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationGenerationAssessmentV1 {
    pub schema_version: String,
    pub manifest_generation: u64,
    pub manifest_blake3: String,
    pub total_requirements: usize,
    pub ready_requirements: usize,
    pub source_readiness: LegacyQualificationSourceReadinessV3,
    pub continuity: LegacySourceContinuityAssessmentV1,
    pub requirements: Vec<LegacyQualificationRequirementAssessmentV3>,
}

pub fn assess_legacy_qualification_generation_v1(
    pack: &LegacyComputingPackV1,
    profile: &LegacyQualificationProfileV1,
    matrix: &ItQualificationMatrixV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
    predecessor_manifest: Option<&LegacyQualificationManifestV1>,
    manifest: &LegacyQualificationManifestV1,
) -> Result<LegacyQualificationGenerationAssessmentV1, LegacyQualificationProfileErrorV3> {
    manifest.validate_basic(pack)?;
    match manifest.generation {
        1 => {
            if predecessor_manifest.is_some() {
                return Err(LegacyQualificationManifestErrorV1::UnexpectedPredecessorInput.into());
            }
            let canonical = initial_legacy_qualification_manifest_v1(pack)?;
            if &canonical != manifest {
                return Err(LegacyQualificationManifestErrorV1::NonCanonicalInitialManifest.into());
            }
        }
        _ => {
            let predecessor = predecessor_manifest.ok_or(
                LegacyQualificationManifestErrorV1::MissingPredecessorInput,
            )?;
            validate_legacy_qualification_manifest_transition_v1(pack, predecessor, manifest)?;
        }
    }

    let active_ledger = filtered_manifest_ledger(pack, artifacts, source_ledger, manifest)?;
    let source_readiness = assess_legacy_qualification_manifest_source_readiness_v1(
        pack,
        artifacts,
        source_ledger,
        manifest,
    )?;
    let continuity = require_legacy_source_continuity_v1(pack, &active_ledger)?;

    let v1 = assess_legacy_qualification_profile_v1(pack, profile, matrix)?;
    let requirements = project_requirements(v1.requirements, source_readiness.source_evidence_ready);
    let ready_requirements = requirements
        .iter()
        .filter(|assessment| assessment.ready_for_evaluation())
        .count();

    Ok(LegacyQualificationGenerationAssessmentV1 {
        schema_version: LEGACY_QUALIFICATION_GENERATION_ASSESSMENT_SCHEMA_V1.into(),
        manifest_generation: manifest.generation,
        manifest_blake3: legacy_qualification_manifest_commitment_v1(manifest)?,
        total_requirements: requirements.len(),
        ready_requirements,
        source_readiness,
        continuity,
        requirements,
    })
}

fn filtered_manifest_ledger(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    source_ledger: &LegacyQualificationSourceLedgerV3,
    manifest: &LegacyQualificationManifestV1,
) -> Result<LegacyQualificationSourceLedgerV3, LegacyQualificationProfileErrorV3> {
    manifest.validate_basic(pack)?;
    let required_source_revisions = manifest.required_source_revisions(pack)?;
    let mut active = LegacyQualificationSourceLedgerV3::new();

    for snapshot_id in &required_source_revisions {
        if let Some(selection) = source_ledger.selection(snapshot_id) {
            active.register_selection(pack, artifacts, selection.clone())?;
        }
    }
    for claim_id in &manifest.claim_ids {
        if let Some(receipt) = source_ledger.claim_receipt(claim_id) {
            active.register_claim_receipt(pack, artifacts, receipt.clone())?;
        }
    }
    for procedure_id in &manifest.procedure_ids {
        if let Some(receipt) = source_ledger.procedure_receipt(procedure_id) {
            active.register_procedure_receipt(pack, artifacts, receipt.clone())?;
        }
    }
    Ok(active)
}

fn project_requirements(
    requirements: Vec<crate::legacy_qualification_profile::LegacyQualificationRequirementAssessmentV1>,
    source_evidence_ready: bool,
) -> Vec<LegacyQualificationRequirementAssessmentV3> {
    requirements
        .into_iter()
        .map(|assessment| {
            let mut non_source_blockers = assessment.blockers;
            non_source_blockers.remove(&LegacyQualificationBlockerV1::SourceNotContentDigestBound);
            LegacyQualificationRequirementAssessmentV3 {
                platform: assessment.platform,
                area: assessment.area,
                knowledge_state: assessment.knowledge_state,
                matching_active_cases: assessment.matching_active_cases,
                non_source_blockers,
                source_evidence_ready,
            }
        })
        .collect()
}

fn claim_scope_preserved(old: &TechnicalKnowledgeClaimV1, new: &TechnicalKnowledgeClaimV1) -> bool {
    old.modality == new.modality
        && old.applicability == new.applicability
        && old.category == new.category
}

fn claim_lineage_preserved(
    pack: &LegacyComputingPackV1,
    old: &TechnicalKnowledgeClaimV1,
    new: &TechnicalKnowledgeClaimV1,
) -> Result<bool, LegacyQualificationManifestErrorV1> {
    let old_snapshot = pack
        .sources
        .snapshot(&old.source_snapshot)
        .ok_or_else(|| LegacyQualificationManifestErrorV1::UnknownSnapshot(old.source_snapshot.clone()))?;
    let new_snapshot = pack
        .sources
        .snapshot(&new.source_snapshot)
        .ok_or_else(|| LegacyQualificationManifestErrorV1::UnknownSnapshot(new.source_snapshot.clone()))?;
    Ok(snapshot_lineage_preserved(&old_snapshot.document_id, new_snapshot))
}

fn procedure_scope_preserved(old: &LegacyProcedureV1, new: &LegacyProcedureV1) -> bool {
    old.platform == new.platform
        && old.area == new.area
        && old.kind == new.kind
        && old.applicability == new.applicability
        && procedure_authority_footprint(old) == procedure_authority_footprint(new)
}

fn procedure_authority_footprint(procedure: &LegacyProcedureV1) -> (bool, bool) {
    let mut read_only = false;
    let mut proposal_only = false;
    for step in &procedure.steps {
        match step.authority {
            LegacyProcedureAuthorityV1::ReadOnlyObservation => read_only = true,
            LegacyProcedureAuthorityV1::ChangeProposalOnly => proposal_only = true,
        }
    }
    (read_only, proposal_only)
}

fn procedure_lineage_preserved(
    pack: &LegacyComputingPackV1,
    old: &LegacyProcedureV1,
    new: &LegacyProcedureV1,
) -> Result<bool, LegacyQualificationManifestErrorV1> {
    for old_snapshot_id in &old.source_snapshots {
        let old_snapshot = pack.sources.snapshot(old_snapshot_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::UnknownSnapshot(old_snapshot_id.clone())
        })?;
        let mut covered = false;
        for new_snapshot_id in &new.source_snapshots {
            let new_snapshot = pack.sources.snapshot(new_snapshot_id).ok_or_else(|| {
                LegacyQualificationManifestErrorV1::UnknownSnapshot(new_snapshot_id.clone())
            })?;
            if snapshot_lineage_preserved(&old_snapshot.document_id, new_snapshot) {
                covered = true;
                break;
            }
        }
        if !covered {
            return Ok(false);
        }
    }
    Ok(true)
}

fn snapshot_lineage_preserved(
    old_document: &crate::standards_registry::SourceDocumentIdV1,
    new_snapshot: &crate::standards_registry::TechnicalSourceSnapshotV1,
) -> bool {
    if &new_snapshot.document_id == old_document {
        return true;
    }
    new_snapshot.relations.iter().any(|relation| {
        &relation.target == old_document
            && matches!(
                &relation.kind,
                SourceRelationKindV1::Updates
                    | SourceRelationKindV1::Obsoletes
                    | SourceRelationKindV1::Supersedes
                    | SourceRelationKindV1::Replaces
                    | SourceRelationKindV1::DerivedFrom
            )
    })
}

fn find_procedure<'a>(pack: &'a LegacyComputingPackV1, id: &str) -> Option<&'a LegacyProcedureV1> {
    pack.procedures.iter().find(|procedure| procedure.id == id)
}

fn require_exact_replacement_keys(
    removed: &BTreeSet<TechnicalClaimIdV1>,
    replacements: &BTreeMap<TechnicalClaimIdV1, TechnicalClaimIdV1>,
) -> Result<(), LegacyQualificationManifestErrorV1> {
    for old_id in removed {
        if !replacements.contains_key(old_id) {
            return Err(LegacyQualificationManifestErrorV1::MissingClaimReplacement(
                old_id.clone(),
            ));
        }
    }
    if replacements.keys().any(|id| !removed.contains(id)) {
        return Err(LegacyQualificationManifestErrorV1::ExtraneousClaimReplacement);
    }
    Ok(())
}

fn require_exact_procedure_replacement_keys(
    removed: &BTreeSet<String>,
    replacements: &BTreeMap<String, String>,
) -> Result<(), LegacyQualificationManifestErrorV1> {
    for old_id in removed {
        if !replacements.contains_key(old_id) {
            return Err(LegacyQualificationManifestErrorV1::MissingProcedureReplacement(
                old_id.clone(),
            ));
        }
    }
    if replacements.keys().any(|id| !removed.contains(id)) {
        return Err(LegacyQualificationManifestErrorV1::ExtraneousProcedureReplacement);
    }
    Ok(())
}

fn require_unique_claim_replacements(
    replacements: &BTreeMap<TechnicalClaimIdV1, TechnicalClaimIdV1>,
) -> Result<(), LegacyQualificationManifestErrorV1> {
    let values = replacements.values().cloned().collect::<BTreeSet<_>>();
    if values.len() != replacements.len() {
        return Err(LegacyQualificationManifestErrorV1::DuplicateClaimReplacementTarget);
    }
    Ok(())
}

fn require_unique_procedure_replacements(
    replacements: &BTreeMap<String, String>,
) -> Result<(), LegacyQualificationManifestErrorV1> {
    let values = replacements.values().cloned().collect::<BTreeSet<_>>();
    if values.len() != replacements.len() {
        return Err(LegacyQualificationManifestErrorV1::DuplicateProcedureReplacementTarget);
    }
    Ok(())
}

fn is_blake3(value: &str) -> bool {
    let value = value.trim();
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegacyQualificationManifestErrorV1 {
    InvalidPack(String),
    UnsupportedSchema(String),
    InvalidGeneration(u64),
    MissingPredecessor,
    UnexpectedPredecessor,
    UnexpectedInitialReplacement,
    MissingPredecessorInput,
    UnexpectedPredecessorInput,
    NonCanonicalInitialManifest,
    PredecessorMismatch,
    GenerationOverflow,
    GenerationDiscontinuity { expected: u64, observed: u64 },
    EmptyClaims,
    EmptyProcedures,
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    UnknownSnapshot(SourceSnapshotIdV1),
    MissingClaimReplacement(TechnicalClaimIdV1),
    MissingProcedureReplacement(String),
    ExtraneousClaimReplacement,
    ExtraneousProcedureReplacement,
    DuplicateClaimReplacementTarget,
    DuplicateProcedureReplacementTarget,
    InvalidClaimReplacement { old: TechnicalClaimIdV1, new: TechnicalClaimIdV1 },
    InvalidProcedureReplacement { old: String, new: String },
    ClaimScopeChanged { old: TechnicalClaimIdV1, new: TechnicalClaimIdV1 },
    ProcedureScopeChanged { old: String, new: String },
    ClaimLineageChanged { old: TechnicalClaimIdV1, new: TechnicalClaimIdV1 },
    ProcedureLineageChanged { old: String, new: String },
    CapturePlan(String),
    Serialization(String),
}

impl fmt::Display for LegacyQualificationManifestErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPack(err) => write!(f, "legacy qualification manifest pack is invalid: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported legacy qualification manifest schema {value}"),
            Self::InvalidGeneration(value) => write!(f, "invalid legacy qualification manifest generation {value}"),
            Self::MissingPredecessor => write!(f, "non-initial manifest is missing predecessor commitment"),
            Self::UnexpectedPredecessor => write!(f, "initial manifest cannot name a predecessor commitment"),
            Self::UnexpectedInitialReplacement => write!(f, "initial manifest cannot contain replacement mappings"),
            Self::MissingPredecessorInput => write!(f, "non-initial qualification generation requires its predecessor manifest"),
            Self::UnexpectedPredecessorInput => write!(f, "initial qualification generation cannot be assessed with a predecessor manifest"),
            Self::NonCanonicalInitialManifest => write!(f, "initial qualification manifest is not the canonical full current registry target"),
            Self::PredecessorMismatch => write!(f, "manifest predecessor commitment mismatched"),
            Self::GenerationOverflow => write!(f, "manifest generation overflow"),
            Self::GenerationDiscontinuity { expected, observed } => write!(f, "manifest generation discontinuity: expected {expected}, observed {observed}"),
            Self::EmptyClaims => write!(f, "qualification manifest has no active claims"),
            Self::EmptyProcedures => write!(f, "qualification manifest has no active procedures"),
            Self::UnknownClaim(id) => write!(f, "qualification manifest references unknown claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "qualification manifest references unknown procedure {id}"),
            Self::UnknownSnapshot(id) => write!(f, "qualification manifest references unknown snapshot {}", id.0),
            Self::MissingClaimReplacement(id) => write!(f, "removed claim {} has no explicit replacement", id.0),
            Self::MissingProcedureReplacement(id) => write!(f, "removed procedure {id} has no explicit replacement"),
            Self::ExtraneousClaimReplacement => write!(f, "claim replacement map contains a target that was not removed"),
            Self::ExtraneousProcedureReplacement => write!(f, "procedure replacement map contains a target that was not removed"),
            Self::DuplicateClaimReplacementTarget => write!(f, "multiple removed claims map to one replacement claim"),
            Self::DuplicateProcedureReplacementTarget => write!(f, "multiple removed procedures map to one replacement procedure"),
            Self::InvalidClaimReplacement { old, new } => write!(f, "invalid claim replacement {} -> {}", old.0, new.0),
            Self::InvalidProcedureReplacement { old, new } => write!(f, "invalid procedure replacement {old} -> {new}"),
            Self::ClaimScopeChanged { old, new } => write!(f, "claim replacement {} -> {} changed qualification scope", old.0, new.0),
            Self::ProcedureScopeChanged { old, new } => write!(f, "procedure replacement {old} -> {new} changed qualification scope"),
            Self::ClaimLineageChanged { old, new } => write!(f, "claim replacement {} -> {} changed source lineage", old.0, new.0),
            Self::ProcedureLineageChanged { old, new } => write!(f, "procedure replacement {old} -> {new} changed source lineage"),
            Self::CapturePlan(err) => write!(f, "qualification manifest capture planning failed: {err}"),
            Self::Serialization(err) => write!(f, "qualification manifest serialization failed: {err}"),
        }
    }
}

impl Error for LegacyQualificationManifestErrorV1 {}

#[derive(Debug)]
pub enum LegacyQualificationProfileErrorV3 {
    Profile(LegacyQualificationProfileErrorV1),
    Source(LegacyQualificationSourceLedgerErrorV3),
    Continuity(LegacySourceContinuityErrorV1),
    Manifest(LegacyQualificationManifestErrorV1),
}

impl fmt::Display for LegacyQualificationProfileErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(err) => write!(f, "legacy V3 qualification profile error: {err}"),
            Self::Source(err) => write!(f, "legacy V3 qualification source error: {err}"),
            Self::Continuity(err) => write!(f, "legacy V3 qualification continuity error: {err}"),
            Self::Manifest(err) => write!(f, "legacy V3 qualification manifest error: {err}"),
        }
    }
}

impl Error for LegacyQualificationProfileErrorV3 {}

impl From<LegacyQualificationProfileErrorV1> for LegacyQualificationProfileErrorV3 {
    fn from(value: LegacyQualificationProfileErrorV1) -> Self {
        Self::Profile(value)
    }
}
impl From<LegacyQualificationSourceLedgerErrorV3> for LegacyQualificationProfileErrorV3 {
    fn from(value: LegacyQualificationSourceLedgerErrorV3) -> Self {
        Self::Source(value)
    }
}
impl From<LegacySourceContinuityErrorV1> for LegacyQualificationProfileErrorV3 {
    fn from(value: LegacySourceContinuityErrorV1) -> Self {
        Self::Continuity(value)
    }
}
impl From<LegacyQualificationManifestErrorV1> for LegacyQualificationProfileErrorV3 {
    fn from(value: LegacyQualificationManifestErrorV1) -> Self {
        Self::Manifest(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_legacy_five_platform_portfolio_v1;
    use crate::exhaustive_legacy_qualification_profile_v1;

    #[test]
    fn empty_v3_source_evidence_blocks_all_cells_without_erasing_other_gaps() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let source_ledger = LegacyQualificationSourceLedgerV3::new();
        let assessment = assess_legacy_qualification_profile_v3(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &source_ledger,
        )
        .unwrap();
        assert_eq!(assessment.total_requirements, 50);
        assert_eq!(assessment.ready_requirements, 0);
        assert!(!assessment.source_readiness.source_evidence_ready);
    }

    #[test]
    fn initial_manifest_is_canonical_full_registry_target() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        assert_eq!(manifest.generation, 1);
        assert_eq!(manifest.claim_ids.len(), pack.sources.claims().count());
        assert_eq!(manifest.procedure_ids.len(), pack.procedures.len());
        assert!(manifest.claim_replacements.is_empty());
        assert!(manifest.procedure_replacements.is_empty());
    }

    #[test]
    fn manifest_capture_plan_uses_only_active_targets() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let plan = plan_legacy_qualification_manifest_captures_v1(&pack, &manifest).unwrap();
        assert_eq!(
            plan.required_source_revisions,
            manifest.required_source_revisions(&pack).unwrap().len()
        );
    }

    #[test]
    fn generation_one_rejects_hand_built_subset_manifest() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let source_ledger = LegacyQualificationSourceLedgerV3::new();
        let mut manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        manifest.claim_ids.pop_first();
        let error = assess_legacy_qualification_generation_v1(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &source_ledger,
            None,
            &manifest,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationProfileErrorV3::Manifest(
                LegacyQualificationManifestErrorV1::NonCanonicalInitialManifest
            )
        ));
    }

    #[test]
    fn successor_cannot_drop_claim_without_replacement() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let predecessor = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let mut successor = predecessor.clone();
        successor.generation = 2;
        successor.predecessor_manifest_blake3 =
            Some(legacy_qualification_manifest_commitment_v1(&predecessor).unwrap());
        successor.claim_ids.pop_first();
        let error = validate_legacy_qualification_manifest_transition_v1(
            &pack,
            &predecessor,
            &successor,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationManifestErrorV1::MissingClaimReplacement(_)
        ));
    }

    #[test]
    fn generation_assessment_binds_manifest_digest_even_when_evidence_is_missing() {
        let (pack, matrix, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let profile = exhaustive_legacy_qualification_profile_v1();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let source_ledger = LegacyQualificationSourceLedgerV3::new();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let expected = legacy_qualification_manifest_commitment_v1(&manifest).unwrap();
        let assessment = assess_legacy_qualification_generation_v1(
            &pack,
            &profile,
            &matrix,
            &artifacts,
            &source_ledger,
            None,
            &manifest,
        )
        .unwrap();
        assert_eq!(assessment.manifest_blake3, expected);
        assert_eq!(assessment.manifest_generation, 1);
        assert!(!assessment.source_readiness.source_evidence_ready);
        assert_eq!(assessment.ready_requirements, 0);
    }
}
