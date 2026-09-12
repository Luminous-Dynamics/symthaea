// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned active-knowledge manifest for legacy qualification.
//!
//! The historical source/claim/procedure registry is append-only provenance.
//! Qualification, however, needs an explicit statement of which claim and
//! procedure identities are active in one evidence generation. Without this
//! boundary, rebaselining a living vendor source would leave superseded
//! metadata-only claims permanently required merely because they remain in the
//! historical registry.
//!
//! ```text
//! historical registry != active qualification manifest
//! manifest transition != deletion
//! removed target -> explicit scope-preserving replacement
//! new manifest generation -> new evidence/qualification generation
//! ```

use crate::legacy_computing::{LegacyComputingPackV1, LegacyProcedureV1};
use crate::standards_registry::{SourceSnapshotIdV1, TechnicalClaimIdV1, TechnicalKnowledgeClaimV1};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_MANIFEST_SCHEMA_V1: &str =
    "symthaea-it-legacy-qualification-manifest-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationManifestV1 {
    pub schema_version: String,
    pub generation: u64,
    /// Exact predecessor manifest commitment. Generation 1 has no predecessor.
    pub predecessor_manifest_blake3: Option<String>,
    pub claim_ids: BTreeSet<TechnicalClaimIdV1>,
    pub procedure_ids: BTreeSet<String>,
}

impl LegacyQualificationManifestV1 {
    pub fn validate(
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
            if procedure(pack, procedure_id).is_none() {
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
        self.validate(pack)?;
        let mut required = BTreeSet::new();
        for claim_id in &self.claim_ids {
            let claim = pack
                .sources
                .claim(claim_id)
                .ok_or_else(|| LegacyQualificationManifestErrorV1::UnknownClaim(claim_id.clone()))?;
            required.insert(claim.source_snapshot.clone());
        }
        for procedure_id in &self.procedure_ids {
            let procedure = procedure(pack, procedure_id).ok_or_else(|| {
                LegacyQualificationManifestErrorV1::UnknownProcedure(procedure_id.clone())
            })?;
            required.extend(procedure.source_snapshots.iter().cloned());
        }
        Ok(required)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationManifestTransitionV1 {
    pub predecessor_manifest_blake3: String,
    pub successor_manifest_blake3: String,
    pub claim_replacements: BTreeMap<TechnicalClaimIdV1, TechnicalClaimIdV1>,
    pub procedure_replacements: BTreeMap<String, String>,
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
    };
    manifest.validate(pack)?;
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

/// Validate a generation transition. Historical targets are never deleted from
/// the registry; removing one from the active manifest requires an explicit
/// replacement that preserves its qualification scope.
pub fn validate_legacy_qualification_manifest_transition_v1(
    pack: &LegacyComputingPackV1,
    predecessor: &LegacyQualificationManifestV1,
    successor: &LegacyQualificationManifestV1,
    claim_replacements: &BTreeMap<TechnicalClaimIdV1, TechnicalClaimIdV1>,
    procedure_replacements: &BTreeMap<String, String>,
) -> Result<LegacyQualificationManifestTransitionV1, LegacyQualificationManifestErrorV1> {
    predecessor.validate(pack)?;
    successor.validate(pack)?;
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

    if claim_replacements.keys().any(|id| !removed_claims.contains(id)) {
        return Err(LegacyQualificationManifestErrorV1::ExtraneousClaimReplacement);
    }
    if procedure_replacements
        .keys()
        .any(|id| !removed_procedures.contains(id))
    {
        return Err(LegacyQualificationManifestErrorV1::ExtraneousProcedureReplacement);
    }

    for old_id in &removed_claims {
        let new_id = claim_replacements.get(old_id).ok_or_else(|| {
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
    }

    for old_id in &removed_procedures {
        let new_id = procedure_replacements.get(old_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::MissingProcedureReplacement(old_id.clone())
        })?;
        if !successor.procedure_ids.contains(new_id) || old_id == new_id {
            return Err(LegacyQualificationManifestErrorV1::InvalidProcedureReplacement {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
        let old = procedure(pack, old_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::UnknownProcedure(old_id.clone())
        })?;
        let new = procedure(pack, new_id).ok_or_else(|| {
            LegacyQualificationManifestErrorV1::UnknownProcedure(new_id.clone())
        })?;
        if !procedure_scope_preserved(old, new) {
            return Err(LegacyQualificationManifestErrorV1::ProcedureScopeChanged {
                old: old_id.clone(),
                new: new_id.clone(),
            });
        }
    }

    Ok(LegacyQualificationManifestTransitionV1 {
        predecessor_manifest_blake3: predecessor_commitment,
        successor_manifest_blake3: legacy_qualification_manifest_commitment_v1(successor)?,
        claim_replacements: claim_replacements.clone(),
        procedure_replacements: procedure_replacements.clone(),
    })
}

fn claim_scope_preserved(
    old: &TechnicalKnowledgeClaimV1,
    new: &TechnicalKnowledgeClaimV1,
) -> bool {
    old.modality == new.modality
        && old.applicability == new.applicability
        && old.category == new.category
}

fn procedure_scope_preserved(old: &LegacyProcedureV1, new: &LegacyProcedureV1) -> bool {
    old.platform == new.platform
        && old.area == new.area
        && old.kind == new.kind
        && old.authority == new.authority
        && old.applicability == new.applicability
}

fn procedure<'a>(pack: &'a LegacyComputingPackV1, id: &str) -> Option<&'a LegacyProcedureV1> {
    pack.procedures.iter().find(|procedure| procedure.id == id)
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
    PredecessorMismatch,
    GenerationOverflow,
    GenerationDiscontinuity { expected: u64, observed: u64 },
    EmptyClaims,
    EmptyProcedures,
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    MissingClaimReplacement(TechnicalClaimIdV1),
    MissingProcedureReplacement(String),
    ExtraneousClaimReplacement,
    ExtraneousProcedureReplacement,
    InvalidClaimReplacement { old: TechnicalClaimIdV1, new: TechnicalClaimIdV1 },
    InvalidProcedureReplacement { old: String, new: String },
    ClaimScopeChanged { old: TechnicalClaimIdV1, new: TechnicalClaimIdV1 },
    ProcedureScopeChanged { old: String, new: String },
    Serialization(String),
}

impl fmt::Display for LegacyQualificationManifestErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPack(err) => write!(f, "legacy qualification manifest pack is invalid: {err}"),
            Self::UnsupportedSchema(value) => write!(f, "unsupported legacy qualification manifest schema {value}"),
            Self::InvalidGeneration(value) => write!(f, "invalid legacy qualification manifest generation {value}"),
            Self::MissingPredecessor => write!(f, "non-initial legacy qualification manifest is missing predecessor commitment"),
            Self::UnexpectedPredecessor => write!(f, "initial legacy qualification manifest cannot name a predecessor"),
            Self::PredecessorMismatch => write!(f, "legacy qualification manifest predecessor commitment mismatched"),
            Self::GenerationOverflow => write!(f, "legacy qualification manifest generation overflow"),
            Self::GenerationDiscontinuity { expected, observed } => write!(f, "legacy qualification manifest generation discontinuity: expected {expected}, observed {observed}"),
            Self::EmptyClaims => write!(f, "legacy qualification manifest has no active claims"),
            Self::EmptyProcedures => write!(f, "legacy qualification manifest has no active procedures"),
            Self::UnknownClaim(id) => write!(f, "legacy qualification manifest references unknown claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "legacy qualification manifest references unknown procedure {id}"),
            Self::MissingClaimReplacement(id) => write!(f, "removed claim {} has no explicit replacement", id.0),
            Self::MissingProcedureReplacement(id) => write!(f, "removed procedure {id} has no explicit replacement"),
            Self::ExtraneousClaimReplacement => write!(f, "claim replacement map contains a target that was not removed"),
            Self::ExtraneousProcedureReplacement => write!(f, "procedure replacement map contains a target that was not removed"),
            Self::InvalidClaimReplacement { old, new } => write!(f, "invalid claim replacement {} -> {}", old.0, new.0),
            Self::InvalidProcedureReplacement { old, new } => write!(f, "invalid procedure replacement {old} -> {new}"),
            Self::ClaimScopeChanged { old, new } => write!(f, "claim replacement {} -> {} changed qualification scope", old.0, new.0),
            Self::ProcedureScopeChanged { old, new } => write!(f, "procedure replacement {old} -> {new} changed qualification scope"),
            Self::Serialization(err) => write!(f, "legacy qualification manifest serialization failed: {err}"),
        }
    }
}

impl Error for LegacyQualificationManifestErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::standards_registry::{TechnicalClaimIdV1, TechnicalKnowledgeClaimV1};
    use crate::build_legacy_five_platform_portfolio_v1;

    #[test]
    fn initial_manifest_freezes_all_current_claims_and_procedures() {
        let (pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let manifest = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        assert_eq!(manifest.generation, 1);
        assert!(manifest.predecessor_manifest_blake3.is_none());
        assert_eq!(manifest.claim_ids.len(), pack.sources.claims().count());
        assert_eq!(manifest.procedure_ids.len(), pack.procedures.len());
        assert!(!manifest.required_source_revisions(&pack).unwrap().is_empty());
        assert!(is_blake3(
            &legacy_qualification_manifest_commitment_v1(&manifest).unwrap()
        ));
    }

    #[test]
    fn removed_claim_requires_explicit_scope_preserving_replacement() {
        let (mut pack, _, _) =
            build_legacy_five_platform_portfolio_v1(1_800_000_000_000).unwrap();
        let old_id = TechnicalClaimIdV1("legacy:aix:os-management-scope".into());
        let old = pack.sources.claim(&old_id).unwrap().clone();
        let new_id = TechnicalClaimIdV1("legacy:aix:os-management-scope:rebaseline-2".into());
        pack.sources
            .register_claim(TechnicalKnowledgeClaimV1 {
                id: new_id.clone(),
                statement: format!("{} [rebaselined]", old.statement),
                source_snapshot: old.source_snapshot.clone(),
                locator: old.locator.clone(),
                modality: old.modality,
                applicability: old.applicability.clone(),
                extraction_quality: old.extraction_quality,
                category: old.category.clone(),
            })
            .unwrap();

        let predecessor = initial_legacy_qualification_manifest_v1(&pack).unwrap();
        let mut successor = predecessor.clone();
        successor.generation = 2;
        successor.predecessor_manifest_blake3 = Some(
            legacy_qualification_manifest_commitment_v1(&predecessor).unwrap(),
        );
        successor.claim_ids.remove(&old_id);
        successor.claim_ids.insert(new_id.clone());

        let error = validate_legacy_qualification_manifest_transition_v1(
            &pack,
            &predecessor,
            &successor,
            &BTreeMap::new(),
            &BTreeMap::new(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationManifestErrorV1::MissingClaimReplacement(_)
        ));

        let transition = validate_legacy_qualification_manifest_transition_v1(
            &pack,
            &predecessor,
            &successor,
            &BTreeMap::from([(old_id, new_id)]),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(is_blake3(&transition.predecessor_manifest_blake3));
        assert!(is_blake3(&transition.successor_manifest_blake3));
    }
}
