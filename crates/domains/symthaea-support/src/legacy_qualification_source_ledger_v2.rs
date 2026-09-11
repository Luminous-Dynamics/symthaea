// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qualification-source selection and claim verification for legacy IT.
//!
//! V1 source readiness treated every historical metadata snapshot as a permanent
//! qualification blocker. That is intentionally conservative, but it leaves no
//! migration path because snapshot identities are immutable. V2 keeps those
//! historical observations intact and instead selects one retained, content-
//! digest-bound successor snapshot per logical document used by the current
//! knowledge pack. Claims are admitted only after a receipt binds their exact
//! serialized proposition/locator to that selected content artifact.
//!
//! Core non-equivalences:
//!
//! ```text
//! metadata history != qualification source set
//! same logical document != same captured bytes
//! content digest != retained artifact
//! retained artifact != verified claim
//! verified claim != platform competence
//! ```

use crate::legacy_computing::{LegacyComputingErrorV1, LegacyComputingPackV1};
use crate::legacy_source_artifacts::{
    LegacySourceArtifactErrorV1, LegacySourceArtifactLedgerV1,
};
use crate::standards_registry::{
    SourceCaptureV1, SourceDocumentIdV1, SourceSnapshotIdV1, TechnicalClaimIdV1,
    TechnicalKnowledgeClaimV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

pub const LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V2: &str =
    "symthaea-it-legacy-qualification-source-ledger-v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LegacyClaimVerificationMethodV2 {
    /// A qualified reviewer checked the normalized claim/locator against the
    /// selected retained artifact.
    HumanReview,
    /// A deterministic extractor/verifier checked the exact claim/locator.
    DeterministicVerification,
    /// Human review and deterministic verification agreed on the same binding.
    HumanAndDeterministic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationSourceSelectionV2 {
    pub document_id: SourceDocumentIdV1,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub content_algorithm: String,
    pub content_digest: String,
    pub selected_at_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyClaimVerificationReceiptV2 {
    pub claim_id: TechnicalClaimIdV1,
    pub document_id: SourceDocumentIdV1,
    /// Preserves the historical snapshot against which the claim was originally
    /// normalized. It is not rewritten when qualification evidence improves.
    pub original_snapshot_id: SourceSnapshotIdV1,
    pub qualifying_snapshot_id: SourceSnapshotIdV1,
    pub content_algorithm: String,
    pub content_digest: String,
    pub verification_method: LegacyClaimVerificationMethodV2,
    pub verifier_profile: String,
    pub verified_at_unix_ms: u64,
    /// BLAKE3 over the exact serialized claim plus logical-document and selected
    /// content identity. This detects statement/locator/source substitution.
    pub claim_binding_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationSourceReadinessV2 {
    pub required_documents: usize,
    pub selected_documents: usize,
    pub required_claims: usize,
    pub verified_claims: usize,
    pub missing_document_selections: BTreeSet<SourceDocumentIdV1>,
    pub missing_claim_receipts: BTreeSet<TechnicalClaimIdV1>,
    /// Source-evidence readiness only. This is deliberately not a platform or
    /// domain competency claim.
    pub source_evidence_ready: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegacyQualificationSourceLedgerV2 {
    pub schema_version: String,
    selections: BTreeMap<SourceDocumentIdV1, LegacyQualificationSourceSelectionV2>,
    claim_receipts: BTreeMap<TechnicalClaimIdV1, LegacyClaimVerificationReceiptV2>,
}

impl Default for LegacyQualificationSourceLedgerV2 {
    fn default() -> Self {
        Self {
            schema_version: LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V2.into(),
            selections: BTreeMap::new(),
            claim_receipts: BTreeMap::new(),
        }
    }
}

impl LegacyQualificationSourceLedgerV2 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn selection(
        &self,
        document_id: &SourceDocumentIdV1,
    ) -> Option<&LegacyQualificationSourceSelectionV2> {
        self.selections.get(document_id)
    }

    pub fn selections(
        &self,
    ) -> impl Iterator<Item = &LegacyQualificationSourceSelectionV2> {
        self.selections.values()
    }

    pub fn claim_receipt(
        &self,
        claim_id: &TechnicalClaimIdV1,
    ) -> Option<&LegacyClaimVerificationReceiptV2> {
        self.claim_receipts.get(claim_id)
    }

    pub fn claim_receipts(
        &self,
    ) -> impl Iterator<Item = &LegacyClaimVerificationReceiptV2> {
        self.claim_receipts.values()
    }

    /// Exact replay is idempotent. A logical document cannot be rebound to a
    /// different qualifying artifact under the same V2 ledger.
    pub fn register_selection(
        &mut self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
        selection: LegacyQualificationSourceSelectionV2,
    ) -> Result<bool, LegacyQualificationSourceLedgerErrorV2> {
        self.validate_schema()?;
        validate_selection(pack, artifacts, &selection)?;
        if let Some(existing) = self.selections.get(&selection.document_id) {
            if existing == &selection {
                return Ok(false);
            }
            return Err(
                LegacyQualificationSourceLedgerErrorV2::DocumentSelectionConflict(
                    selection.document_id,
                ),
            );
        }
        self.selections
            .insert(selection.document_id.clone(), selection);
        Ok(true)
    }

    /// Exact replay is idempotent. A claim receipt cannot later be rebound to a
    /// different content artifact, verifier profile, or normalized proposition.
    pub fn register_claim_receipt(
        &mut self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
        receipt: LegacyClaimVerificationReceiptV2,
    ) -> Result<bool, LegacyQualificationSourceLedgerErrorV2> {
        self.validate_schema()?;
        validate_claim_receipt(pack, artifacts, self, &receipt)?;
        if let Some(existing) = self.claim_receipts.get(&receipt.claim_id) {
            if existing == &receipt {
                return Ok(false);
            }
            return Err(LegacyQualificationSourceLedgerErrorV2::ClaimReceiptConflict(
                receipt.claim_id,
            ));
        }
        self.claim_receipts.insert(receipt.claim_id.clone(), receipt);
        Ok(true)
    }

    pub fn validate_against_pack(
        &self,
        pack: &LegacyComputingPackV1,
        artifacts: &LegacySourceArtifactLedgerV1,
    ) -> Result<(), LegacyQualificationSourceLedgerErrorV2> {
        self.validate_schema()?;
        pack.validate()?;
        for selection in self.selections.values() {
            validate_selection(pack, artifacts, selection)?;
        }
        for receipt in self.claim_receipts.values() {
            validate_claim_receipt(pack, artifacts, self, receipt)?;
        }
        Ok(())
    }

    fn validate_schema(&self) -> Result<(), LegacyQualificationSourceLedgerErrorV2> {
        if self.schema_version != LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V2 {
            return Err(LegacyQualificationSourceLedgerErrorV2::UnsupportedSchema(
                self.schema_version.clone(),
            ));
        }
        Ok(())
    }
}

pub fn assess_legacy_qualification_source_readiness_v2(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    ledger: &LegacyQualificationSourceLedgerV2,
) -> Result<LegacyQualificationSourceReadinessV2, LegacyQualificationSourceLedgerErrorV2> {
    pack.validate()?;
    ledger.validate_against_pack(pack, artifacts)?;

    let required_documents = required_document_ids(pack)?;
    let required_claims: BTreeSet<_> = pack.sources.claims().map(|claim| claim.id.clone()).collect();

    let missing_document_selections: BTreeSet<_> = required_documents
        .iter()
        .filter(|document_id| ledger.selection(document_id).is_none())
        .cloned()
        .collect();
    let missing_claim_receipts: BTreeSet<_> = required_claims
        .iter()
        .filter(|claim_id| ledger.claim_receipt(claim_id).is_none())
        .cloned()
        .collect();

    let selected_documents = required_documents.len() - missing_document_selections.len();
    let verified_claims = required_claims.len() - missing_claim_receipts.len();
    let source_evidence_ready = !required_documents.is_empty()
        && !required_claims.is_empty()
        && missing_document_selections.is_empty()
        && missing_claim_receipts.is_empty();

    Ok(LegacyQualificationSourceReadinessV2 {
        required_documents: required_documents.len(),
        selected_documents,
        required_claims: required_claims.len(),
        verified_claims,
        missing_document_selections,
        missing_claim_receipts,
        source_evidence_ready,
    })
}

/// Compute the immutable claim/source binding used in verification receipts.
/// The claim is serialized as a typed struct (stable field order), then framed
/// together with the logical document and selected content identity.
pub fn legacy_claim_verification_binding_v2(
    claim: &TechnicalKnowledgeClaimV1,
    selection: &LegacyQualificationSourceSelectionV2,
) -> Result<String, LegacyQualificationSourceLedgerErrorV2> {
    let claim_bytes = serde_json::to_vec(claim).map_err(|err| {
        LegacyQualificationSourceLedgerErrorV2::Serialization(err.to_string())
    })?;
    let mut hasher = blake3::Hasher::new();
    frame(&mut hasher, b"schema", LEGACY_QUALIFICATION_SOURCE_LEDGER_SCHEMA_V2.as_bytes());
    frame(&mut hasher, b"claim", &claim_bytes);
    frame(&mut hasher, b"document", selection.document_id.0.as_bytes());
    frame(
        &mut hasher,
        b"qualifying_snapshot",
        selection.qualifying_snapshot_id.0.as_bytes(),
    );
    frame(
        &mut hasher,
        b"content_algorithm",
        selection.content_algorithm.trim().to_ascii_lowercase().as_bytes(),
    );
    frame(
        &mut hasher,
        b"content_digest",
        selection.content_digest.trim().to_ascii_lowercase().as_bytes(),
    );
    Ok(hasher.finalize().to_hex().to_string())
}

fn validate_selection(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    selection: &LegacyQualificationSourceSelectionV2,
) -> Result<(), LegacyQualificationSourceLedgerErrorV2> {
    pack.validate()?;
    require_nonempty(&selection.document_id.0, "selection document id")?;
    require_nonempty(
        &selection.qualifying_snapshot_id.0,
        "qualifying snapshot id",
    )?;
    require_nonempty(&selection.content_algorithm, "content algorithm")?;
    require_nonempty(&selection.content_digest, "content digest")?;
    if selection.selected_at_unix_ms == 0 {
        return Err(LegacyQualificationSourceLedgerErrorV2::InvalidField(
            "selection timestamp must be non-zero".into(),
        ));
    }
    pack.sources
        .document(&selection.document_id)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV2::UnknownDocument(
                selection.document_id.clone(),
            )
        })?;
    let snapshot = pack
        .sources
        .snapshot(&selection.qualifying_snapshot_id)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV2::UnknownSnapshot(
                selection.qualifying_snapshot_id.clone(),
            )
        })?;
    if snapshot.document_id != selection.document_id {
        return Err(LegacyQualificationSourceLedgerErrorV2::DocumentSnapshotMismatch {
            document: selection.document_id.clone(),
            snapshot: selection.qualifying_snapshot_id.clone(),
        });
    }
    let SourceCaptureV1::ContentDigest { algorithm, digest } = &snapshot.capture else {
        return Err(
            LegacyQualificationSourceLedgerErrorV2::QualifyingSnapshotNotContentBound(
                snapshot.id.clone(),
            ),
        );
    };
    if !algorithm.eq_ignore_ascii_case(selection.content_algorithm.trim())
        || digest.to_ascii_lowercase() != selection.content_digest.trim().to_ascii_lowercase()
    {
        return Err(LegacyQualificationSourceLedgerErrorV2::SelectionDigestMismatch(
            snapshot.id.clone(),
        ));
    }
    let artifact = artifacts.artifact(&snapshot.id).ok_or_else(|| {
        LegacyQualificationSourceLedgerErrorV2::MissingRetainedArtifact(snapshot.id.clone())
    })?;
    artifact.validate_against_pack(pack)?;
    Ok(())
}

fn validate_claim_receipt(
    pack: &LegacyComputingPackV1,
    artifacts: &LegacySourceArtifactLedgerV1,
    ledger: &LegacyQualificationSourceLedgerV2,
    receipt: &LegacyClaimVerificationReceiptV2,
) -> Result<(), LegacyQualificationSourceLedgerErrorV2> {
    require_nonempty(&receipt.claim_id.0, "claim id")?;
    require_nonempty(&receipt.document_id.0, "claim document id")?;
    require_nonempty(&receipt.original_snapshot_id.0, "original snapshot id")?;
    require_nonempty(
        &receipt.qualifying_snapshot_id.0,
        "qualifying snapshot id",
    )?;
    require_nonempty(&receipt.content_algorithm, "content algorithm")?;
    require_nonempty(&receipt.content_digest, "content digest")?;
    require_nonempty(&receipt.verifier_profile, "verifier profile")?;
    require_nonempty(&receipt.claim_binding_blake3, "claim binding digest")?;
    if receipt.verified_at_unix_ms == 0 {
        return Err(LegacyQualificationSourceLedgerErrorV2::InvalidField(
            "claim verification timestamp must be non-zero".into(),
        ));
    }

    let claim = pack.sources.claim(&receipt.claim_id).ok_or_else(|| {
        LegacyQualificationSourceLedgerErrorV2::UnknownClaim(receipt.claim_id.clone())
    })?;
    if claim.source_snapshot != receipt.original_snapshot_id {
        return Err(
            LegacyQualificationSourceLedgerErrorV2::OriginalSnapshotMismatch(
                receipt.claim_id.clone(),
            ),
        );
    }
    let original_snapshot = pack
        .sources
        .snapshot(&claim.source_snapshot)
        .ok_or_else(|| {
            LegacyQualificationSourceLedgerErrorV2::UnknownSnapshot(
                claim.source_snapshot.clone(),
            )
        })?;
    if original_snapshot.document_id != receipt.document_id {
        return Err(LegacyQualificationSourceLedgerErrorV2::ClaimDocumentMismatch(
            receipt.claim_id.clone(),
        ));
    }

    let selection = ledger.selection(&receipt.document_id).ok_or_else(|| {
        LegacyQualificationSourceLedgerErrorV2::MissingDocumentSelection(
            receipt.document_id.clone(),
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
        return Err(LegacyQualificationSourceLedgerErrorV2::ReceiptSelectionMismatch(
            receipt.claim_id.clone(),
        ));
    }

    let expected = legacy_claim_verification_binding_v2(claim, selection)?;
    if expected != receipt.claim_binding_blake3.trim().to_ascii_lowercase() {
        return Err(LegacyQualificationSourceLedgerErrorV2::ClaimBindingMismatch(
            receipt.claim_id.clone(),
        ));
    }
    Ok(())
}

fn required_document_ids(
    pack: &LegacyComputingPackV1,
) -> Result<BTreeSet<SourceDocumentIdV1>, LegacyQualificationSourceLedgerErrorV2> {
    let mut documents = BTreeSet::new();
    for claim in pack.sources.claims() {
        let snapshot = pack
            .sources
            .snapshot(&claim.source_snapshot)
            .ok_or_else(|| {
                LegacyQualificationSourceLedgerErrorV2::UnknownSnapshot(
                    claim.source_snapshot.clone(),
                )
            })?;
        documents.insert(snapshot.document_id.clone());
    }
    for procedure in &pack.procedures {
        for snapshot_id in &procedure.source_snapshots {
            let snapshot = pack.sources.snapshot(snapshot_id).ok_or_else(|| {
                LegacyQualificationSourceLedgerErrorV2::UnknownSnapshot(snapshot_id.clone())
            })?;
            documents.insert(snapshot.document_id.clone());
        }
    }
    Ok(documents)
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
) -> Result<(), LegacyQualificationSourceLedgerErrorV2> {
    if value.trim().is_empty() {
        return Err(LegacyQualificationSourceLedgerErrorV2::InvalidField(
            format!("{field} must be non-empty"),
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub enum LegacyQualificationSourceLedgerErrorV2 {
    Pack(LegacyComputingErrorV1),
    Artifact(LegacySourceArtifactErrorV1),
    UnsupportedSchema(String),
    InvalidField(String),
    Serialization(String),
    UnknownDocument(SourceDocumentIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
    UnknownClaim(TechnicalClaimIdV1),
    DocumentSnapshotMismatch {
        document: SourceDocumentIdV1,
        snapshot: SourceSnapshotIdV1,
    },
    QualifyingSnapshotNotContentBound(SourceSnapshotIdV1),
    SelectionDigestMismatch(SourceSnapshotIdV1),
    MissingRetainedArtifact(SourceSnapshotIdV1),
    DocumentSelectionConflict(SourceDocumentIdV1),
    MissingDocumentSelection(SourceDocumentIdV1),
    ClaimReceiptConflict(TechnicalClaimIdV1),
    OriginalSnapshotMismatch(TechnicalClaimIdV1),
    ClaimDocumentMismatch(TechnicalClaimIdV1),
    ReceiptSelectionMismatch(TechnicalClaimIdV1),
    ClaimBindingMismatch(TechnicalClaimIdV1),
}

impl fmt::Display for LegacyQualificationSourceLedgerErrorV2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pack(err) => write!(f, "qualification-source pack error: {err}"),
            Self::Artifact(err) => write!(f, "qualification-source artifact error: {err}"),
            Self::UnsupportedSchema(schema) => {
                write!(f, "unsupported qualification-source ledger schema {schema}")
            }
            Self::InvalidField(message) => write!(f, "invalid qualification-source field: {message}"),
            Self::Serialization(message) => write!(f, "claim-binding serialization failed: {message}"),
            Self::UnknownDocument(id) => write!(f, "unknown qualification document {}", id.0),
            Self::UnknownSnapshot(id) => write!(f, "unknown qualification snapshot {}", id.0),
            Self::UnknownClaim(id) => write!(f, "unknown qualification claim {}", id.0),
            Self::DocumentSnapshotMismatch { document, snapshot } => write!(
                f,
                "qualifying snapshot {} does not belong to logical document {}",
                snapshot.0, document.0
            ),
            Self::QualifyingSnapshotNotContentBound(id) => write!(
                f,
                "qualifying snapshot {} is not content-digest-bound",
                id.0
            ),
            Self::SelectionDigestMismatch(id) => write!(
                f,
                "qualification selection digest does not match snapshot {}",
                id.0
            ),
            Self::MissingRetainedArtifact(id) => write!(
                f,
                "qualification snapshot {} has no retained-artifact receipt",
                id.0
            ),
            Self::DocumentSelectionConflict(id) => write!(
                f,
                "logical document {} is already bound to a different qualification source",
                id.0
            ),
            Self::MissingDocumentSelection(id) => write!(
                f,
                "logical document {} has no selected qualification source",
                id.0
            ),
            Self::ClaimReceiptConflict(id) => write!(
                f,
                "claim {} is already bound to a different verification receipt",
                id.0
            ),
            Self::OriginalSnapshotMismatch(id) => write!(
                f,
                "claim {} verification receipt does not preserve its original snapshot",
                id.0
            ),
            Self::ClaimDocumentMismatch(id) => write!(
                f,
                "claim {} verification receipt names the wrong logical document",
                id.0
            ),
            Self::ReceiptSelectionMismatch(id) => write!(
                f,
                "claim {} verification receipt does not match the selected source artifact",
                id.0
            ),
            Self::ClaimBindingMismatch(id) => write!(
                f,
                "claim {} verification binding digest does not match current claim/source content",
                id.0
            ),
        }
    }
}

impl Error for LegacyQualificationSourceLedgerErrorV2 {}

impl From<LegacyComputingErrorV1> for LegacyQualificationSourceLedgerErrorV2 {
    fn from(value: LegacyComputingErrorV1) -> Self {
        Self::Pack(value)
    }
}

impl From<LegacySourceArtifactErrorV1> for LegacyQualificationSourceLedgerErrorV2 {
    fn from(value: LegacySourceArtifactErrorV1) -> Self {
        Self::Artifact(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_source::{
        KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeStabilityV1,
    };
    use crate::legacy_source_artifacts::{
        LegacyArtifactAccessPolicyV1, LegacyArtifactStorageClassV1,
        LegacySourceArtifactRefV1,
    };
    use crate::standards_registry::TechnicalSourceSnapshotV1;
    use crate::build_legacy_five_platform_portfolio_v1;

    const DIGEST: &str =
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn pack() -> LegacyComputingPackV1 {
        build_legacy_five_platform_portfolio_v1(1_800_000_000_000)
            .unwrap()
            .0
    }

    fn add_content_successor_for_first_claim(
        pack: &mut LegacyComputingPackV1,
    ) -> (TechnicalKnowledgeClaimV1, LegacyQualificationSourceSelectionV2, LegacySourceArtifactLedgerV1) {
        let claim = pack.sources.claims().next().unwrap().clone();
        let original = pack.sources.snapshot(&claim.source_snapshot).unwrap().clone();
        let qualifying_id = SourceSnapshotIdV1(format!("{}:qualification-v2", original.id.0));
        pack.sources
            .register_snapshot(TechnicalSourceSnapshotV1 {
                id: qualifying_id.clone(),
                document_id: original.document_id.clone(),
                version: original.version.clone(),
                lifecycle: KnowledgeLifecycleV1::Active,
                authority: KnowledgeAuthorityClassV1::VendorDocumentation,
                stability: KnowledgeStabilityV1::Stable,
                published_at_unix_ms: original.published_at_unix_ms,
                source_updated_at_unix_ms: original.source_updated_at_unix_ms,
                fetched_at_unix_ms: original.fetched_at_unix_ms + 1_000,
                capture: SourceCaptureV1::ContentDigest {
                    algorithm: "sha256".into(),
                    digest: DIGEST.into(),
                },
                relations: original.relations.clone(),
            })
            .unwrap();
        let selection = LegacyQualificationSourceSelectionV2 {
            document_id: original.document_id,
            qualifying_snapshot_id: qualifying_id.clone(),
            content_algorithm: "sha256".into(),
            content_digest: DIGEST.into(),
            selected_at_unix_ms: 1_800_000_002_000,
        };
        let mut artifacts = LegacySourceArtifactLedgerV1::new();
        artifacts
            .register(
                pack,
                LegacySourceArtifactRefV1 {
                    snapshot_id: qualifying_id,
                    content_algorithm: "sha256".into(),
                    content_digest: DIGEST.into(),
                    byte_length: 123_456,
                    media_type: "application/pdf".into(),
                    retrieved_at_unix_ms: 1_800_000_001_000,
                    artifact_locator: "evidence://legacy/qualification/source.pdf".into(),
                    storage_class: LegacyArtifactStorageClassV1::PrivateEvidenceStore,
                    access_policy: LegacyArtifactAccessPolicyV1::EvaluatorOnly,
                    retention_receipt_digest: None,
                },
            )
            .unwrap();
        (claim, selection, artifacts)
    }

    #[test]
    fn metadata_history_no_longer_has_to_be_rewritten_for_source_selection() {
        let mut pack = pack();
        let historical_metadata_snapshots = pack
            .sources
            .snapshots()
            .filter(|snapshot| matches!(&snapshot.capture, SourceCaptureV1::MetadataOnly))
            .count();
        let (_claim, selection, artifacts) = add_content_successor_for_first_claim(&mut pack);
        let mut ledger = LegacyQualificationSourceLedgerV2::new();
        assert!(ledger
            .register_selection(&pack, &artifacts, selection)
            .unwrap());
        assert_eq!(
            pack.sources
                .snapshots()
                .filter(|snapshot| matches!(&snapshot.capture, SourceCaptureV1::MetadataOnly))
                .count(),
            historical_metadata_snapshots
        );
    }

    #[test]
    fn metadata_only_snapshot_cannot_be_selected_for_qualification() {
        let pack = pack();
        let snapshot = pack.sources.snapshots().next().unwrap();
        let artifacts = LegacySourceArtifactLedgerV1::new();
        let mut ledger = LegacyQualificationSourceLedgerV2::new();
        let error = ledger
            .register_selection(
                &pack,
                &artifacts,
                LegacyQualificationSourceSelectionV2 {
                    document_id: snapshot.document_id.clone(),
                    qualifying_snapshot_id: snapshot.id.clone(),
                    content_algorithm: "sha256".into(),
                    content_digest: DIGEST.into(),
                    selected_at_unix_ms: 1_800_000_002_000,
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationSourceLedgerErrorV2::QualifyingSnapshotNotContentBound(_)
        ));
    }

    #[test]
    fn verified_claim_is_bound_to_exact_selected_content() {
        let mut pack = pack();
        let (claim, selection, artifacts) = add_content_successor_for_first_claim(&mut pack);
        let mut ledger = LegacyQualificationSourceLedgerV2::new();
        ledger
            .register_selection(&pack, &artifacts, selection.clone())
            .unwrap();
        let binding = legacy_claim_verification_binding_v2(&claim, &selection).unwrap();
        let receipt = LegacyClaimVerificationReceiptV2 {
            claim_id: claim.id.clone(),
            document_id: selection.document_id.clone(),
            original_snapshot_id: claim.source_snapshot.clone(),
            qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
            content_algorithm: selection.content_algorithm.clone(),
            content_digest: selection.content_digest.clone(),
            verification_method: LegacyClaimVerificationMethodV2::HumanAndDeterministic,
            verifier_profile: "legacy-source-verifier-v2".into(),
            verified_at_unix_ms: 1_800_000_003_000,
            claim_binding_blake3: binding,
        };
        assert!(ledger
            .register_claim_receipt(&pack, &artifacts, receipt.clone())
            .unwrap());
        assert!(!ledger
            .register_claim_receipt(&pack, &artifacts, receipt)
            .unwrap());
    }

    #[test]
    fn claim_binding_tamper_fails_closed() {
        let mut pack = pack();
        let (claim, selection, artifacts) = add_content_successor_for_first_claim(&mut pack);
        let mut ledger = LegacyQualificationSourceLedgerV2::new();
        ledger
            .register_selection(&pack, &artifacts, selection.clone())
            .unwrap();
        let error = ledger
            .register_claim_receipt(
                &pack,
                &artifacts,
                LegacyClaimVerificationReceiptV2 {
                    claim_id: claim.id.clone(),
                    document_id: selection.document_id.clone(),
                    original_snapshot_id: claim.source_snapshot.clone(),
                    qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
                    content_algorithm: selection.content_algorithm.clone(),
                    content_digest: selection.content_digest.clone(),
                    verification_method: LegacyClaimVerificationMethodV2::HumanReview,
                    verifier_profile: "legacy-source-verifier-v2".into(),
                    verified_at_unix_ms: 1_800_000_003_000,
                    claim_binding_blake3: "00".repeat(32),
                },
            )
            .unwrap_err();
        assert!(matches!(
            error,
            LegacyQualificationSourceLedgerErrorV2::ClaimBindingMismatch(_)
        ));
    }

    #[test]
    fn partial_source_evidence_is_reported_without_false_readiness() {
        let mut pack = pack();
        let (claim, selection, artifacts) = add_content_successor_for_first_claim(&mut pack);
        let mut ledger = LegacyQualificationSourceLedgerV2::new();
        ledger
            .register_selection(&pack, &artifacts, selection.clone())
            .unwrap();
        let binding = legacy_claim_verification_binding_v2(&claim, &selection).unwrap();
        ledger
            .register_claim_receipt(
                &pack,
                &artifacts,
                LegacyClaimVerificationReceiptV2 {
                    claim_id: claim.id.clone(),
                    document_id: selection.document_id.clone(),
                    original_snapshot_id: claim.source_snapshot.clone(),
                    qualifying_snapshot_id: selection.qualifying_snapshot_id.clone(),
                    content_algorithm: selection.content_algorithm.clone(),
                    content_digest: selection.content_digest.clone(),
                    verification_method: LegacyClaimVerificationMethodV2::HumanAndDeterministic,
                    verifier_profile: "legacy-source-verifier-v2".into(),
                    verified_at_unix_ms: 1_800_000_003_000,
                    claim_binding_blake3: binding,
                },
            )
            .unwrap();
        let assessment =
            assess_legacy_qualification_source_readiness_v2(&pack, &artifacts, &ledger).unwrap();
        assert_eq!(assessment.selected_documents, 1);
        assert_eq!(assessment.verified_claims, 1);
        assert!(!assessment.source_evidence_ready);
        assert!(!assessment.missing_document_selections.is_empty());
        assert!(!assessment.missing_claim_receipts.is_empty());
    }
}
