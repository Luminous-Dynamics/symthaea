// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Snapshot-scoped qualification-source ledger for legacy IT evidence.
//!
//! V3 corrects two important limitations in the earlier source-readiness model:
//!
//! 1. source selection is keyed by the exact immutable source snapshot used by a
//!    claim/procedure, not merely by logical document identity;
//! 2. advisory procedures receive their own exact verification receipts rather
//!    than inheriting trust merely because their source documents were captured.
//!
//! ```text
//! logical document != source revision
//! selected source revision != verified claim
//! verified claim != verified procedure
//! verified procedure != competence
//! ```

use crate::legacy_computing::{
    LegacyComputingErrorV1, LegacyComputingPackV1, LegacyProcedureV1,
};
use crate::legacy_source_artifacts::{
    LegacySourceArtifactErrorV1, LegacySourceArtifactLedgerV1,
};
use crate::legacy_source_capture_plan_v3::{
    legacy_source_revision_commitment_v3, plan_legacy_qualification_source_captures_v3,
    LegacySourceCapturePlanErrorV3,
};
use crate::standards_registry::{
    SourceCaptureV1, SourceSnapshotIdV1, TechnicalClaimIdV1, TechnicalKnowledgeClaimV1,
    TechnicalSourceSnapshotV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V3: &str =
    "symthaea-it-legacy-qualification-source-ledger-v3";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LegacyKnowledgeVerificationMethodV3 {
    HumanReview,
    DeterministicVerification,
    HumanAndDeterministic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationSourceSelectionV3 {
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub source_revision_blake3: String,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub content_algorithm: String,
    pub content_digest: String,
    pub selected_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyClaimVerificationReceiptV3 {
    pub claim_id: TechnicalClaimIdV1,
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub content_algorithm: String,
    pub content_digest: String,
    pub verification_method: LegacyKnowledgeVerificationMethodV3,
    pub verifier_profile: String,
    pub verified_at_unix_ms: u64,
    pub claim_binding_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LegacyProcedureSourceBindingV3 {
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub source_revision_blake3: String,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub content_algorithm: String,
    pub content_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyProcedureVerificationReceiptV3 {
    pub procedure_id: String,
    pub source_bindings: Vec<LegacyProcedureSourceBindingV3>,
    pub verification_method: LegacyKnowledgeVerificationMethodV3,
    pub verifier_profile: String,
    pub verified_at_unix_ms: u64,
    pub procedure_binding_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationSourceReadinessV3 {
    pub required_source_revisions: usize,
    pub selected_source_revisions: usize,
    pub required_claims: usize,
    pub verified_claims: usize,
    pub required_procedures: usize,
    pub verified_procedures: usize,
    pub missing_source_selections: BTreeSet<SourceSnapshotIdV1>,
    pub missing_claim_receipts: BTreeSet<TechnicalClaimIdV1>,
    pub missing_procedure_receipts: BTreeSet<String>,
    /// Source/derived-knowledge provenance readiness only. This does not imply
    /// scenario, hardware, adversarial, or competence qualification.
    pub source_evidence_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationSourceLedgerV3 {
    pub schema_version: String,
    selections: BTreeMap<SourceSnapshotIdV1, LegacyQualificationSourceSelectionV3>,
    claim_receipts: BTreeMap<TechnicalClaimIdV1, LegacyClaimVerificationReceiptV3>,
    procedure_receipts: BTreeMap<String, LegacyProcedureVerificationReceiptV3>,
}

impl Default for LegacyQualificationSourceLedgerV3 {
    fn default() -> Self {
        Self {
            schema_version: LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V3.into(),
            selections: BTreeMap::new(),
            claim_receipts: BTreeMap::new(),
            procedure_receipts: BTreeMap::new(),
        }
    }
}

impl LegacyQualificationSourceLedgerV3 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn selection(
        &self,
        original_snapshot_id: &SourceSnapshotIdV1,
    ) -> Option<&LegacyQualificationSourceSelectionV3> {
        self.selections.get(original_snapshot_id)
    }

    pub fn selections(
        &self,
    ) -> impl Iterator<Item = &LegacyQualificationSourceSelectionV3> {
        self.selections.values()
    }

    pub fn claim_receipt(
        &self,
        claim_id: &TechnicalClaimIdV1,
    ) -> Option<&LegacyClaimVerificationReceiptV3> {
        self.claim_receipts.get(claim_id)
    }

    pub fn procedure_receipt(
        &self,
        procedure_id: &str,
    ) -> Option<&LegacyProcedureVerificationReceiptV3> {
        self.procedure_receipts.get(procedure_id)
    }

    pub fn register_selection(
        &mut self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
        selection: LegacyQualificationSourceSelectionV3,
    ) -> Result<bool, LegacyQualificationSourceLedgerErrorV3> {
        self.validate_schema()?;
        validate_selection(pack, artifacts, &selection)?;
        if let Some(existing) = self.selections.get(&selection.original_snapshot_id) {
            if existing == &selection {
                return Ok(false);
            }
            return Err(LegacyQualificationSourceLedgerErrorV3::SelectionConflict(
                selection.original_snapshot_id,
            ));
        }
        self.selections
            .insert(selection.original_snapshot_id.clone(), selection);
        Ok(true)
    }

    pub fn register_claim_receipt(
        &mut self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
        receipt: LegacyClaimVerificationReceiptV3,
    ) -> Result<bool, LegacyQualificationSourceLedgerErrorV3> {
        self.validate_schema()?;
        validate_claim_receipt(pack, artifacts, self, &receipt)?;
        if let Some(existing) = self.claim_receipts.get(&receipt.claim_id) {
            if existing == &receipt {
                return Ok(false);
            }
            return Err(LegacyQualificationSourceLedgerErrorV3::ClaimReceiptConflict(
                receipt.claim_id,
            ));
        }
        self.claim_receipts.insert(receipt.claim_id.clone(), receipt);
        Ok(true)
    }

    pub fn register_procedure_receipt(
        &mut self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
        receipt: LegacyProcedureVerificationReceiptV3,
    ) -> Result<bool, LegacyQualificationSourceLedgerErrorV3> {
        self.validate_schema()?;
        validate_procedure_receipt(pack, artifacts, self, &receipt)?;
        if let Some(existing) = self.procedure_receipts.get(&receipt.procedure_id) {
            if existing == &receipt {
                return Ok(false);
            }
            return Err(
                LegacyQualificationSourceLedgerErrorV3::ProcedureReceiptConflict(
                    receipt.procedure_id,
                ),
            );
        }
        self.procedure_receipts
            .insert(receipt.procedure_id.clone(), receipt);
        Ok(true)
    }

    pub fn validate_against_pack(
        &self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
    ) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
        self.validate_schema()?;
        pack.validate()?;
        artifacts.validate_against_pack(pack)?;
        for selection in self.selections.values() {
            validate_selection(pack, artifacts, selection)?;
        }
        for receipt in self.claim_receipts.values() {
            validate_claim_receipt(pack, artifacts, self, receipt)?;
        }
        for receipt in self.procedure_receipts.values() {
            validate_procedure_receipt(pack, artifacts, self, receipt)?;
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
        if self.schema_version != LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V3 {
            return Err(LegacyQualificationSourceLedgerErrorV3::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        Ok(())
    }
}

pub fn assess_legacy_qualification_source_readiness_v3(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    ledger: &LegacyQualificationSourceLedgerV3,
) -> Result<LegacyQualificationSourceReadinessV3, LegacyQualificationSourceLedgerErrorV3> {
    pack.validate()?;
    ledger.validate_against_pack(pack, artifacts)?;
    let plan = plan_legacy_qualification_source_captures_v3(pack)?;

    let required_source_revisions: BTreeSet<_> = plan
        .requests
        .iter()
        .map(|request| request.original_snapshot_id.clone())
        .collect();
    let required_claims: BTreeSet<_> = pack.sources.claims().map(|claim| claim.id.clone()).collect();
    let required_procedures: BTreeSet<_> =
        pack.procedures.iter().map(|procedure| procedure.id.clone()).collect();

    let missing_source_selections = required_source_revisions
        .iter()
        .filter(|snapshot_id| ledger.selection(snapshot_id).is_none())
        .cloned()
        .collect::<BTreeSet<_>>();
    let missing_claim_receipts = required_claims
        .iter()
        .filter(|claim_id| ledger.claim_receipt(claim_id).is_none())
        .cloned()
        .collect::<BTreeSet<_>>();
    let missing_procedure_receipts = required_procedures
        .iter()
        .filter(|procedure_id| ledger.procedure_receipt(procedure_id).is_none())
        .cloned()
        .collect::<BTreeSet<_>>();

    let selected_source_revisions = required_source_revisions.len() - missing_source_selections.len();
    let verified_claims = required_claims.len() - missing_claim_receipts.len();
    let verified_procedures = required_procedures.len() - missing_procedure_receipts.len();
    let source_evidence_ready = !required_source_revisions.is_empty()
        && missing_source_selections.is_empty()
        && missing_claim_receipts.is_empty()
        && missing_procedure_receipts.is_empty();

    Ok(LegacyQualificationSourceReadinessV3 {
        required_source_revisions: required_source_revisions.len(),
        selected_source_revisions,
        required_claims: required_claims.len(),
        verified_claims,
        required_procedures: required_procedures.len(),
        verified_procedures,
        missing_source_selections,
        missing_claim_receipts,
        missing_procedure_receipts,
        source_evidence_ready,
    })
}

pub fn legacy_claim_verification_binding_v3(
    claim: &TechnicalKnowledgeClaimV1,
    selection: &LegacyQualificationSourceSelectionV3,
) -> Result<String, LegacyQualificationSourceLedgerErrorV3> {
    let claim_bytes = serde_json::to_vec(claim)
        .map_err(|err| LegacyQualificationSourceLedgerErrorV3::Serialization(err.to_string()))?;
    let selection_bytes = serde_json::to_vec(selection)
        .map_err(|err| LegacyQualificationSourceLedgerErrorV3::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V3.as_bytes(),
    );
    frame(&mut hasher, b"claim", &claim_bytes);
    frame(&mut hasher, b"selection", &selection_bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

pub fn legacy_procedure_source_bindings_v3(
    pack: &LegacyComputingPackV1,
    ledger: &LegacyQualificationSourceLedgerV3,
    procedure: &LegacyProcedureV1,
) -> Result<Vec<LegacyProcedureSourceBindingV3>, LegacyQualificationSourceLedgerErrorV3> {
    let mut bindings = Vec::with_capacity(procedure.source_snapshots.len());
    for original_snapshot_id in &procedure.source_snapshots {
        let selection = ledger.selection(original_snapshot_id).ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV3::MissingSelection(
                original_snapshot_id.clone(),
            )
        })?;
        bindings.push(LegacyProcedureSourceBindingV3 {
            original_snapshot_id: selection.original_snapshot_id.clone(),
            source_revision_blake3: selection.source_revision_blake3.clone(),
            qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
            content_algorithm: selection.content_algorithm.clone(),
            content_digest: selection.content_digest.clone(),
        });
    }
    bindings.sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));
    Ok(bindings)
}

pub fn legacy_procedure_verification_binding_v3(
    procedure: &LegacyProcedureV1,
    source_bindings: &[LegacyProcedureSourceBindingV3],
) -> Result<String, LegacyQualificationSourceLedgerErrorV3> {
    let mut sorted = source_bindings.to_vec();
    sorted.sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));
    let procedure_bytes = serde_json::to_vec(procedure)
        .map_err(|err| LegacyQualificationSourceLedgerErrorV3::Serialization(err.to_string()))?;
    let binding_bytes = serde_json::to_vec(&sorted)
        .map_err(|err| LegacyQualificationSourceLedgerErrorV3::Serialization(err.to_string()))?;
    let mut hasher = blake3::Hasher::new();
    frame(
        &mut hasher,
        b"schema",
        LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V3.as_bytes(),
    );
    frame(&mut hasher, b"procedure", &procedure_bytes);
    frame(&mut hasher, b"source_bindings", &binding_bytes);
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_selection(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    selection: &LegacyQualificationSourceSelectionV3,
) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
    require_nonempty(&selection.original_snapshot_id.0, "original snapshot id")?;
    require_blake3(&selection.source_revision_blake3, "source revision digest")?;
    require_nonempty(&selection.qualifying_snapshot_id.0, "qualifying snapshot id")?;
    require_nonempty(&selection.content_algorithm, "content algorithm")?;
    require_nonempty(&selection.content_digest, "content digest")?;
    if selection.selected_at_unix_ms == 0 {
        return Err(LegacyQualificationSourceLedgerErrorV3::InvalidField(
            "selection timestamp must be non-zero".into(),
        ));
    }

    let original = pack
        .sources
        .snapshot(&selection.original_snapshot_id)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV3::UnknownSnapshot(
                selection.original_snapshot_id.clone(),
            )
        })?;
    let document = pack
        .sources
        .document(&original.document_id)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV3::UnknownDocument(
                original.document_id.clone(),
            )
        })?;
    let expected_revision = legacy_source_revision_commitment_v3(document, original)?;
    if expected_revision != selection.source_revision_blake3.trim().to_ascii_lowercase() {
        return Err(LegacyQualificationSourceLedgerErrorV3::SourceRevisionMismatch(
            selection.original_snapshot_id.clone(),
        ));
    }

    let qualifying = pack
        .sources
        .snapshot(&selection.qualifying_snapshot_id)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV3::UnknownSnapshot(
                selection.qualifying_snapshot_id.clone(),
            )
        })?;
    if !same_semantic_revision(original, qualifying) {
        return Err(LegacyQualificationSourceLedgerErrorV3::RevisionMetadataMismatch {
            original: original.id.clone(),
            qualifying: qualifying.id.clone(),
        });
    }
    let SourceCaptureV1::ContentDigest { algorithm, digest } = &qualifying.capture else {
        return Err(
            LegacyQualificationSourceLedgerErrorV3::QualifyingSnapshotNotContentBound(
                qualifying.id.clone(),
            ),
        );
    };
    if !algorithm.eq_ignore_ascii_case(selection.content_algorithm.trim())
        || digest.to_ascii_lowercase() != selection.content_digest.trim().to_ascii_lowercase()
    {
        return Err(LegacyQualificationSourceLedgerErrorV3::SelectionDigestMismatch(
            qualifying.id.clone(),
        ));
    }
    if selection.selected_at_unix_ms < qualifying.fetched_at_unix_ms {
        return Err(LegacyQualificationSourceLedgerErrorV3::SelectionPredatesCapture(
            qualifying.id.clone(),
        ));
    }
    let artifact = artifacts.artifact(&qualifying.id).ok_or_else(|| {
        LegacyQualificationSourceLedgerErrorV3::MissingRetainedArtifact(qualifying.id.clone())
    })?;
    artifact.validate_against_pack(pack)?;
    Ok(())
}

fn validate_claim_receipt(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    ledger: &LegacyQualificationSourceLedgerV3,
    receipt: &LegacyClaimVerificationReceiptV3,
) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
    require_nonempty(&receipt.claim_id.0, "claim id")?;
    require_nonempty(&receipt.verifier_profile, "verifier profile")?;
    require_blake3(&receipt.claim_binding_blake3, "claim binding digest")?;
    if receipt.verified_at_unix_ms == 0 {
        return Err(LegacyQualificationSourceLedgerErrorV3::InvalidField(
            "claim verification timestamp must be non-zero".into(),
        ));
    }
    let claim = pack.sources.claim(&receipt.claim_id).ok_or_else(|| {
        LegacyQualificationSourceLedgerErrorV3::UnknownClaim(receipt.claim_id.clone())
    })?;
    if claim.source_snapshot != receipt.original_snapshot_id {
        return Err(LegacyQualificationSourceLedgerErrorV3::ClaimSourceMismatch(
            receipt.claim_id.clone(),
        ));
    }
    let selection = ledger.selection(&receipt.original_snapshot_id).ok_or_else(|| {
        LegacyQualificationSourceLedgerErrorV3::MissingSelection(
            receipt.original_snapshot_id.clone(),
        )
    })?;
    validate_selection(pack, artifacts, selection)?;
    if selection.qualifying_snapshot_id != receipt.qualifying_snapshot_id
        || !selection
            .content_algorithm
            .eq_ignore_ascii_case(receipt.content_algorithm.trim())
        || selection.content_digest.to_ascii_lowercase()
            != receipt.content_digest.trim().to_ascii_lowercase()
    {
        return Err(LegacyQualificationSourceLedgerErrorV3::ReceiptSelectionMismatch(
            receipt.claim_id.clone(),
        ));
    }
    if receipt.verified_at_unix_ms < selection.selected_at_unix_ms {
        return Err(LegacyQualificationSourceLedgerErrorV3::VerificationPredatesSelection(
            receipt.claim_id.0.clone(),
        ));
    }
    let expected = legacy_claim_verification_binding_v3(claim, selection)?;
    if expected != receipt.claim_binding_blake3.trim().to_ascii_lowercase() {
        return Err(LegacyQualificationSourceLedgerErrorV3::ClaimBindingMismatch(
            receipt.claim_id.clone(),
        ));
    }
    Ok(())
}

fn validate_procedure_receipt(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    ledger: &LegacyQualificationSourceLedgerV3,
    receipt: &LegacyProcedureVerificationReceiptV3,
) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
    require_nonempty(&receipt.procedure_id, "procedure id")?;
    require_nonempty(&receipt.verifier_profile, "verifier profile")?;
    require_blake3(
        &receipt.procedure_binding_blake3,
        "procedure binding digest",
    )?;
    if receipt.verified_at_unix_ms == 0 {
        return Err(LegacyQualificationSourceLedgerErrorV3::InvalidField(
            "procedure verification timestamp must be non-zero".into(),
        ));
    }
    let procedure = pack
        .procedures
        .iter()
        .find(|procedure| procedure.id == receipt.procedure_id)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV3::UnknownProcedure(
                receipt.procedure_id.clone(),
            )
        })?;
    let expected_bindings = legacy_procedure_source_bindings_v3(pack, ledger, procedure)?;
    let mut actual_bindings = receipt.source_bindings.clone();
    actual_bindings.sort_by(|a, b| a.original_snapshot_id.cmp(&b.original_snapshot_id));
    if actual_bindings != expected_bindings {
        return Err(
            LegacyQualificationSourceLedgerErrorV3::ProcedureSourceBindingsMismatch(
                receipt.procedure_id.clone(),
            ),
        );
    }
    let mut latest_selection = 0u64;
    for binding in &expected_bindings {
        let selection = ledger.selection(&binding.original_snapshot_id).ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV3::MissingSelection(
                binding.original_snapshot_id.clone(),
            )
        })?;
        validate_selection(pack, artifacts, selection)?;
        latest_selection = latest_selection.max(selection.selected_at_unix_ms);
    }
    if receipt.verified_at_unix_ms < latest_selection {
        return Err(LegacyQualificationSourceLedgerErrorV3::VerificationPredatesSelection(
            receipt.procedure_id.clone(),
        ));
    }
    let expected = legacy_procedure_verification_binding_v3(procedure, &expected_bindings)?;
    if expected != receipt.procedure_binding_blake3.trim().to_ascii_lowercase() {
        return Err(
            LegacyQualificationSourceLedgerErrorV3::ProcedureBindingMismatch(
                receipt.procedure_id.clone(),
            ),
        );
    }
    Ok(())
}

fn same_semantic_revision(
    original: &TechnicalSourceSnapshotV1,
    qualifying: &TechnicalSourceSnapshotV1,
) -> bool {
    original.document_id == qualifying.document_id
        && original.version == qualifying.version
        && original.lifecycle == qualifying.lifecycle
        && original.authority == qualifying.authority
        && original.stability == qualifying.stability
        && original.published_at_unix_ms == qualifying.published_at_unix_ms
        && original.source_updated_at_unix_ms == qualifying.source_updated_at_unix_ms
        && original.relations == qualifying.relations
}

fn frame(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn require_nonempty(
    value: &str,
    field: &'static str,
) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
    if value.trim().is_empty() {
        return Err(LegacyQualificationSourceLedgerErrorV3::InvalidField(
            format!("{field} must be non-empty"),
        ));
    }
    Ok(())
}

fn require_blake3(
    value: &str,
    field: &'static str,
) -> Result<(), LegacyQualificationSourceLedgerErrorV3> {
    let value = value.trim();
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(LegacyQualificationSourceLedgerErrorV3::InvalidField(
            format!("{field} must be a 64-character hex BLAKE3 digest"),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub enum LegacyQualificationSourceLedgerErrorV3 {
    Pack(LegacyComputingErrorV1),
    Artifact(LegacySourceArtifactErrorV1),
    CapturePlan(LegacySourceCapturePlanErrorV3),
    UnsupportedSchema(String),
    InvalidField(String),
    Serialization(String),
    UnknownDocument(crate::standards_registry::SourceDocumentIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
    UnknownClaim(TechnicalClaimIdV1),
    UnknownProcedure(String),
    MissingSelection(SourceSnapshotIdV1),
    SelectionConflict(SourceSnapshotIdV1),
    SourceRevisionMismatch(SourceSnapshotIdV1),
    RevisionMetadataMismatch {
        original: SourceSnapshotIdV1,
        qualifying: SourceSnapshotIdV1,
    },
    QualifyingSnapshotNotContentBound(SourceSnapshotIdV1),
    SelectionDigestMismatch(SourceSnapshotIdV1),
    SelectionPredatesCapture(SourceSnapshotIdV1),
    MissingRetainedArtifact(SourceSnapshotIdV1),
    ClaimReceiptConflict(TechnicalClaimIdV1),
    ClaimSourceMismatch(TechnicalClaimIdV1),
    ReceiptSelectionMismatch(TechnicalClaimIdV1),
    ClaimBindingMismatch(TechnicalClaimIdV1),
    ProcedureReceiptConflict(String),
    ProcedureSourceBindingsMismatch(String),
    ProcedureBindingMismatch(String),
    VerificationPredatesSelection(String),
}

impl fmt::Display for LegacyQualificationSourceLedgerErrorV3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pack(err) => write!(f, "qualification-source V3 pack error: {err}"),
            Self::Artifact(err) => write!(f, "qualification-source V3 artifact error: {err}"),
            Self::CapturePlan(err) => write!(f, "qualification-source V3 capture-plan error: {err}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported qualification-source V3 schema {schema}")
            }
            Self::InvalidField(message) => write!(f, "invalid qualification-source V3 field: {message}"),
            Self::Serialization(message) => write!(f, "qualification-source V3 serialization failed: {message}"),
            Self::UnknownDocument(id) => write!(f, "unknown qualification document {}", id.0),
            Self::UnknownSnapshot(id) => write!(f, "unknown qualification snapshot {}", id.0),
            Self::UnknownClaim(id) => write!(f, "unknown qualification claim {}", id.0),
            Self::UnknownProcedure(id) => write!(f, "unknown qualification procedure {id}"),
            Self::MissingSelection(id) => write!(f, "source revision {} has no qualification selection", id.0),
            Self::SelectionConflict(id) => write!(f, "source revision {} is already bound to a different selection", id.0),
            Self::SourceRevisionMismatch(id) => write!(f, "source revision commitment mismatch for {}", id.0),
            Self::RevisionMetadataMismatch { original, qualifying } => write!(
                f,
                "qualifying snapshot {} does not preserve semantic metadata of original snapshot {}",
                qualifying.0, original.0
            ),
            Self::QualifyingSnapshotNotContentBound(id) => write!(f, "qualifying snapshot {} is not content-digest-bound", id.0),
            Self::SelectionDigestMismatch(id) => write!(f, "qualification digest does not match snapshot {}", id.0),
            Self::SelectionPredatesCapture(id) => write!(f, "selection predates qualifying capture {}", id.0),
            Self::MissingRetainedArtifact(id) => write!(f, "qualifying snapshot {} has no retained artifact", id.0),
            Self::ClaimReceiptConflict(id) => write!(f, "claim {} is already bound to a different receipt", id.0),
            Self::ClaimSourceMismatch(id) => write!(f, "claim {} receipt names the wrong original source revision", id.0),
            Self::ReceiptSelectionMismatch(id) => write!(f, "claim {} receipt does not match its selected source revision", id.0),
            Self::ClaimBindingMismatch(id) => write!(f, "claim {} binding digest mismatch", id.0),
            Self::ProcedureReceiptConflict(id) => write!(f, "procedure {id} is already bound to a different receipt"),
            Self::ProcedureSourceBindingsMismatch(id) => write!(f, "procedure {id} source bindings do not match its declared source snapshots"),
            Self::ProcedureBindingMismatch(id) => write!(f, "procedure {id} binding digest mismatch"),
            Self::VerificationPredatesSelection(id) => write!(f, "verification for {id} predates source selection"),
        }
    }
}

impl Error for LegacyQualificationSourceLedgerErrorV3 {}

impl From<LegacyComputingErrorV1> for LegacyQualificationSourceLedgerErrorV3 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Pack(value)
    }
}

impl From<LegacySourceArtifactErrorV1> for LegacyQualificationSourceLedgerErrorV3 {
    fn from(value: LegacySourceArtifactErrorV1) -> Self {
        Self::Artifact(value)
    }
}

impl From<LegacySourceCapturePlanErrorV3> for LegacyQualificationSourceLedgerErrorV3 {
    fn from(value: LegacySourceCapturePlanErrorV3) -> Self {
        Self::CapturePlan(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::legacy_source_artifacts::{
        LegacyArtifactAccessPolicyV1, LegacyArtifactStorageClassV1,
        LegacySourceArtifactRefV1,
    };
    use crate::standards_registry::TechnicalSourceSnapshotV1;
    use crate::build_legacy_five_platform_portfolio_v1;

    fn portfolio() -> LegacyComputingPackV1 {
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0
    }

    fn populate_all_source_evidence(
        pack: &mut LegacyComputingPackV1,
    ) -> (
        LegacySourceArtifactLedgerV1,
        LegacyQualificationSourceLedgerV3,
    ) {
        let plan = plan_legacy_qualification_source_captures_v3(pack).unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        let mut ledger = LegacyQualificationSourceLedgerV3::new();

        for (index, request) in plan.requests.iter().enumerate() {
            let original = pack
                .sources
                .snapshot(&request.original_snapshot_id)
                .unwrap()
                .clone();
            let (qualifying_snapshot_id, algorithm, digest) = match &original.capture {
                SourceCaptureV1::ContentDigest { algorithm, digest } => (
                    original.id.clone(),
                    algorithm.clone(),
                    digest.clone(),
                ),
                _ => {
                    let digest = format!("{:064x}", index + 1);
                    let id = SourceSnapshotIdV1(format!("{}:qualification-v3", original.id.0));
                    pack.sources
                        .register_snapshot(TechnicalSourceSnapshotV1 {
                            id: id.clone(),
                            document_id: original.document_id.clone(),
                            version: original.version.clone(),
                            lifecycle: original.lifecycle,
                            authority: original.authority,
                            stability: original.stability,
                            published_at_unix_ms: original.published_at_unix_ms,
                            source_updated_at_unix_ms: original.source_updated_at_unix_ms,
                            fetched_at_unix_ms: original.fetched_at_unix_ms + 10_000 + index as u64,
                            capture: SourceCaptureV1::ContentDigest {
                                algorithm: "sha256".into(),
                                digest: digest.clone(),
                            },
                            relations: original.relations.clone(),
                        })
                        .unwrap();
                    (id, "sha256".into(), digest)
                }
            };
            let qualifying = pack
                .sources
                .snapshot(&qualifying_snapshot_id)
                .unwrap()
                .clone();
            artifacts
                .register(
                    pack,
                    LegacySourceArtifactRefV1 {
                        snapshot_id: qualifying_snapshot_id.clone(),
                        content_algorithm: algorithm.clone(),
                        content_digest: digest.clone(),
                        byte_length: 10_000 + index as u64,
                        media_type: "application/octet-stream".into(),
                        retrieved_at_unix_ms: qualifying.fetched_at_unix_ms,
                        artifact_locator: format!("evidence://legacy-v3/{index}"),
                        storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
                        access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
                        retention_receipt_digest: Some(format!("{:064x}", 100_000 + index)),
                    },
                )
                .unwrap();
            let document = pack.sources.document(&original.document_id).unwrap();
            let selection = LegacyQualificationSourceSelectionV3 {
                original_snapshot_id: original.id.clone(),
                source_revision_blake3: legacy_source_revision_commitment_v3(document, &original)
                    .unwrap(),
                qualifying_snapshot_id,
                content_algorithm: algorithm,
                content_digest: digest,
                selected_at_unix_ms: qualifying.fetched_at_unix_ms + 1,
            };
            ledger.register_selection(pack, &artifacts, selection).unwrap();
        }

        let claims = pack.sources.claims().cloned().collect::<Vec<_>>();
        for claim in claims {
            let selection = ledger.selection(&claim.source_snapshot).unwrap().clone();
            let binding = legacy_claim_verification_binding_v3(&claim, &selection).unwrap();
            ledger
                .register_claim_receipt(
                    pack,
                    &artifacts,
                    LegacyClaimVerificationReceiptV3 {
                        claim_id: claim.id.clone(),
                        original_snapshot_id: claim.source_snapshot.clone(),
                        qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
                        content_algorithm: selection.content_algorithm.clone(),
                        content_digest: selection.content_digest.clone(),
                        verification_method: LegacyKnowledgeVerificationMethodV3::HumanAndDeterministic,
                        verifier_profile: "legacy-source-verifier-v3".into(),
                        verified_at_unix_ms: selection.selected_at_unix_ms + 1,
                        claim_binding_blake3: binding,
                    },
                )
                .unwrap();
        }

        let procedures = pack.procedures.clone();
        for procedure in procedures {
            let bindings = legacy_procedure_source_bindings_v3(pack, &ledger, &procedure).unwrap();
            let latest = bindings
                .iter()
                .filter_map(|binding| ledger.selection(&binding.original_snapshot_id))
                .map(|selection| selection.selected_at_unix_ms)
                .max()
                .unwrap_or(1);
            let binding = legacy_procedure_verification_binding_v3(&procedure, &bindings).unwrap();
            ledger
                .register_procedure_receipt(
                    pack,
                    &artifacts,
                    LegacyProcedureVerificationReceiptV3 {
                        procedure_id: procedure.id.clone(),
                        source_bindings: bindings,
                        verification_method: LegacyKnowledgeVerificationMethodV3::HumanAndDeterministic,
                        verifier_profile: "legacy-procedure-verifier-v3".into(),
                        verified_at_unix_ms: latest + 1,
                        procedure_binding_blake3: binding,
                    },
                )
                .unwrap();
        }

        (artifacts, ledger)
    }

    #[test]
    fn empty_v3_ledger_reports_all_three_provenance_dimensions() {
        let pack = portfolio();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let ledger = LegacyQualificationSourceLedgerV3::new();
        let readiness =
            assess_legacy_qualification_source_readiness_v3(&pack, &artifacts, &ledger).unwrap();
        assert!(readiness.required_source_revisions > 0);
        assert!(readiness.required_claims > 0);
        assert!(readiness.required_procedures > 0);
        assert_eq!(readiness.selected_source_revisions, 0);
        assert_eq!(readiness.verified_claims, 0);
        assert_eq!(readiness.verified_procedures, 0);
        assert!(!readiness.source_evidence_ready);
    }

    #[test]
    fn full_synthetic_v3_source_evidence_can_become_ready() {
        let mut pack = portfolio();
        let (artifacts, ledger) = populate_all_source_evidence(&mut pack);
        let readiness =
            assess_legacy_qualification_source_readiness_v3(&pack, &artifacts, &ledger).unwrap();
        assert_eq!(
            readiness.required_source_revisions,
            readiness.selected_source_revisions
        );
        assert_eq!(readiness.required_claims, readiness.verified_claims);
        assert_eq!(readiness.required_procedures, readiness.verified_procedures);
        assert!(readiness.source_evidence_ready);
    }

    #[test]
    fn qualifying_snapshot_cannot_drift_to_another_version() {
        let mut pack = portfolio();
        let plan = plan_legacy_qualification_source_captures_v3(&pack).unwrap();
        let request = plan.requests.first().unwrap();
        let original = pack
            .sources
            .snapshot(&request.original_snapshot_id)
            .unwrap()
            .clone();
        let id = SourceSnapshotIdV1(format!("{}:wrong-version", original.id.0));
        let digest = "a".repeat(64);
        pack.sources
            .register_snapshot(TechnicalSourceSnapshotV1 {
                id: id.clone(),
                document_id: original.document_id.clone(),
                version: Some("wrong-version".into()),
                lifecycle: original.lifecycle,
                authority: original.authority,
                stability: original.stability,
                published_at_unix_ms: original.published_at_unix_ms,
                source_updated_at_unix_ms: original.source_updated_at_unix_ms,
                fetched_at_unix_ms: original.fetched_at_unix_ms + 1,
                capture: SourceCaptureV1::ContentDigest {
                    algorithm: "sha256".into(),
                    digest: digest.clone(),
                },
                relations: original.relations.clone(),
            })
            .unwrap();
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        artifacts
            .register(
                &pack,
                LegacySourceArtifactRefV1 {
                    snapshot_id: id.clone(),
                    content_algorithm: "sha256".into(),
                    content_digest: digest.clone(),
                    byte_length: 1,
                    media_type: "application/octet-stream".into(),
                    retrieved_at_unix_ms: original.fetched_at_unix_ms + 1,
                    artifact_locator: "evidence://wrong-version".into(),
                    storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
                    access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
                    retention_receipt_digest: Some("b".repeat(64)),
                },
            )
            .unwrap();
        let document = pack.sources.document(&original.document_id).unwrap();
        let mut ledger = LegacyQualificationSourceLedgerV3::new();
        let error = ledger
            .register_selection(
                &pack,
                &artifacts,
                LegacyQualificationSourceSelectionV3 {
                    original_snapshot_id: original.id.clone(),
                    source_revision_blake3: legacy_source_revision_commitment_v3(document, &original)
                        .unwrap(),
                    qualifying_snapshot_id: id,
                    content_algorithm: "sha256".into(),
                    content_digest: digest,
                    selected_at_unix_ms: original.fetched_at_unix_ms + 2,
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationSourceLedgerErrorV3::RevisionMetadataMismatch { .. }
        ));
    }

    #[test]
    fn procedure_text_tamper_invalidates_existing_receipt_binding() {
        let mut pack = portfolio();
        let (artifacts, ledger) = populate_all_source_evidence(&mut pack);
        let mut tampered = pack.clone();
        tampered.procedures[0].steps[0].description.push_str(" tampered");
        let receipt = ledger
            .procedure_receipt(&tampered.procedures[0].id)
            .unwrap()
            .clone();
        let error = validate_procedure_receipt(&tampered, &artifacts, &ledger, &receipt)
            .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationSourceLedgerErrorV3::ProcedureBindingMismatch(_)
        ));
    }
}
