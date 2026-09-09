// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned technical-source registry for IT knowledge.
//!
//! The registry separates a stable logical document from immutable observed
//! snapshots of that document. This matters for mutable registries, living
//! standards, drafts, withdrawals, and documents that update/obsolete others.
//!
//! It stores metadata, digests, relations, and extracted technical claims — not
//! wholesale copies of standards documents.

use crate::knowledge_source::{
    KnowledgeAuthorityClassV1, KnowledgeLifecycleV1, KnowledgeOriginV1,
    KnowledgeShareabilityV1, KnowledgeSourceErrorV1, KnowledgeStabilityV1,
    SupportKnowledgeHitV1, SupportKnowledgeQueryV1, SupportKnowledgeSourceV1,
};
use crate::technology::{ApplicabilityScopeV1, ApplicabilityStatusV1};
use crate::types::SupportCategory;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceDocumentIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceSnapshotIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TechnicalClaimIdV1(pub String);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TechnicalPublisherV1 {
    IetfRfcEditor,
    Iana,
    Nist,
    OpenTelemetry,
    OpenConfig,
    Mitre,
    Owasp,
    Vendor(String),
    StandardsBody(String),
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SourceDocumentKindV1 {
    Rfc,
    Registry,
    Standard,
    BestCurrentPractice,
    Guidance,
    SemanticConventions,
    VendorDocumentation,
    SecurityKnowledgeBase,
    Other(String),
}

/// Stable logical identity for a technical source across revisions/captures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TechnicalSourceDocumentV1 {
    pub id: SourceDocumentIdV1,
    pub publisher: TechnicalPublisherV1,
    pub kind: SourceDocumentKindV1,
    pub title: String,
    /// Canonical human identifier such as `RFC 9000`, `SP 800-207`, or a
    /// registry/convention family name.
    pub canonical_ref: String,
    /// Canonical locator recorded as metadata. Consumers decide whether/how to fetch it.
    pub canonical_locator: Option<String>,
}

impl TechnicalSourceDocumentV1 {
    pub fn validate(&self) -> Result<(), StandardsRegistryErrorV1> {
        require_nonempty(&self.id.0, "document id")?;
        require_nonempty(&self.title, "document title")?;
        require_nonempty(&self.canonical_ref, "canonical reference")?;
        validate_publisher(&self.publisher)?;
        validate_document_kind(&self.kind)?;
        if self
            .canonical_locator
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(StandardsRegistryErrorV1::EmptyField("canonical locator"));
        }
        Ok(())
    }
}

/// How much source material was captured. A digest can bind an immutable external
/// artifact without storing that artifact in this registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceCaptureV1 {
    MetadataOnly,
    MetadataDigest { algorithm: String, digest: String },
    ContentDigest { algorithm: String, digest: String },
}

impl SourceCaptureV1 {
    fn validate(&self) -> Result<(), StandardsRegistryErrorV1> {
        match self {
            Self::MetadataOnly => Ok(()),
            Self::MetadataDigest { algorithm, digest }
            | Self::ContentDigest { algorithm, digest } => {
                require_nonempty(algorithm, "digest algorithm")?;
                require_nonempty(digest, "digest")
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SourceRelationKindV1 {
    Updates,
    Obsoletes,
    Supersedes,
    Replaces,
    DerivedFrom,
    References,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SourceRelationV1 {
    pub kind: SourceRelationKindV1,
    pub target: SourceDocumentIdV1,
}

/// Immutable observation of one logical source at a particular revision/capture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TechnicalSourceSnapshotV1 {
    pub id: SourceSnapshotIdV1,
    pub document_id: SourceDocumentIdV1,
    /// Publisher-native version/revision text. No generic lexical ordering is assumed.
    pub version: Option<String>,
    pub lifecycle: KnowledgeLifecycleV1,
    pub authority: KnowledgeAuthorityClassV1,
    pub stability: KnowledgeStabilityV1,
    pub published_at_unix_ms: Option<u64>,
    pub source_updated_at_unix_ms: Option<u64>,
    pub fetched_at_unix_ms: u64,
    pub capture: SourceCaptureV1,
    #[serde(default)]
    pub relations: BTreeSet<SourceRelationV1>,
}

impl TechnicalSourceSnapshotV1 {
    pub fn validate(&self) -> Result<(), StandardsRegistryErrorV1> {
        require_nonempty(&self.id.0, "snapshot id")?;
        require_nonempty(&self.document_id.0, "snapshot document id")?;
        if self.version.as_deref().is_some_and(|value| value.trim().is_empty()) {
            return Err(StandardsRegistryErrorV1::EmptyField("snapshot version"));
        }
        self.capture.validate()?;
        for relation in &self.relations {
            require_nonempty(&relation.target.0, "relation target")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimModalityV1 {
    NormativeMust,
    NormativeShould,
    NormativeMay,
    Descriptive,
    Warning,
    Deprecation,
    Example,
    Informational,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct TechnicalSourceLocatorV1 {
    pub section: Option<String>,
    pub fragment: Option<String>,
}

impl TechnicalSourceLocatorV1 {
    fn validate(&self) -> Result<(), StandardsRegistryErrorV1> {
        if self.section.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(StandardsRegistryErrorV1::EmptyField("source section"));
        }
        if self.fragment.as_deref().is_some_and(|v| v.trim().is_empty()) {
            return Err(StandardsRegistryErrorV1::EmptyField("source fragment"));
        }
        Ok(())
    }
}

/// A compact extracted technical proposition bound to an immutable source snapshot.
/// `extraction_quality` measures extraction/normalization quality, not truth.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TechnicalKnowledgeClaimV1 {
    pub id: TechnicalClaimIdV1,
    pub statement: String,
    pub source_snapshot: SourceSnapshotIdV1,
    pub locator: Option<TechnicalSourceLocatorV1>,
    pub modality: ClaimModalityV1,
    pub applicability: Option<ApplicabilityScopeV1>,
    pub extraction_quality: Option<f32>,
    pub category: Option<SupportCategory>,
}

impl TechnicalKnowledgeClaimV1 {
    pub fn validate(&self) -> Result<(), StandardsRegistryErrorV1> {
        require_nonempty(&self.id.0, "claim id")?;
        require_nonempty(&self.statement, "claim statement")?;
        require_nonempty(&self.source_snapshot.0, "claim snapshot id")?;
        if let Some(locator) = &self.locator {
            locator.validate()?;
        }
        if let Some(scope) = &self.applicability {
            scope
                .validate()
                .map_err(|err| StandardsRegistryErrorV1::InvalidApplicability(err.to_string()))?;
        }
        if let Some(quality) = self.extraction_quality {
            if !quality.is_finite() || !(0.0..=1.0).contains(&quality) {
                return Err(StandardsRegistryErrorV1::InvalidExtractionQuality(quality));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TechnicalStandardsRegistryV1 {
    documents: BTreeMap<SourceDocumentIdV1, TechnicalSourceDocumentV1>,
    snapshots: BTreeMap<SourceSnapshotIdV1, TechnicalSourceSnapshotV1>,
    claims: BTreeMap<TechnicalClaimIdV1, TechnicalKnowledgeClaimV1>,
}

impl TechnicalStandardsRegistryV1 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn document(&self, id: &SourceDocumentIdV1) -> Option<&TechnicalSourceDocumentV1> {
        self.documents.get(id)
    }

    pub fn snapshot(&self, id: &SourceSnapshotIdV1) -> Option<&TechnicalSourceSnapshotV1> {
        self.snapshots.get(id)
    }

    pub fn claim(&self, id: &TechnicalClaimIdV1) -> Option<&TechnicalKnowledgeClaimV1> {
        self.claims.get(id)
    }

    pub fn documents(&self) -> impl Iterator<Item = &TechnicalSourceDocumentV1> {
        self.documents.values()
    }

    pub fn snapshots(&self) -> impl Iterator<Item = &TechnicalSourceSnapshotV1> {
        self.snapshots.values()
    }

    pub fn claims(&self) -> impl Iterator<Item = &TechnicalKnowledgeClaimV1> {
        self.claims.values()
    }

    /// Exact replay is idempotent; an identity may never be rebound.
    pub fn register_document(
        &mut self,
        document: TechnicalSourceDocumentV1,
    ) -> Result<bool, StandardsRegistryErrorV1> {
        document.validate()?;
        if let Some(existing) = self.documents.get(&document.id) {
            if existing == &document {
                return Ok(false);
            }
            return Err(StandardsRegistryErrorV1::DocumentIdentityConflict(document.id));
        }
        self.documents.insert(document.id.clone(), document);
        Ok(true)
    }

    /// Snapshot insertion requires its logical document but does not require
    /// relation targets yet; source graphs can be loaded in any document order.
    pub fn register_snapshot(
        &mut self,
        snapshot: TechnicalSourceSnapshotV1,
    ) -> Result<bool, StandardsRegistryErrorV1> {
        snapshot.validate()?;
        if !self.documents.contains_key(&snapshot.document_id) {
            return Err(StandardsRegistryErrorV1::UnknownDocument(
                snapshot.document_id.clone(),
            ));
        }
        if let Some(existing) = self.snapshots.get(&snapshot.id) {
            if existing == &snapshot {
                return Ok(false);
            }
            return Err(StandardsRegistryErrorV1::SnapshotIdentityConflict(snapshot.id));
        }
        self.snapshots.insert(snapshot.id.clone(), snapshot);
        Ok(true)
    }

    pub fn register_claim(
        &mut self,
        claim: TechnicalKnowledgeClaimV1,
    ) -> Result<bool, StandardsRegistryErrorV1> {
        claim.validate()?;
        if !self.snapshots.contains_key(&claim.source_snapshot) {
            return Err(StandardsRegistryErrorV1::UnknownSnapshot(
                claim.source_snapshot.clone(),
            ));
        }
        if let Some(existing) = self.claims.get(&claim.id) {
            if existing == &claim {
                return Ok(false);
            }
            return Err(StandardsRegistryErrorV1::ClaimIdentityConflict(claim.id));
        }
        self.claims.insert(claim.id.clone(), claim);
        Ok(true)
    }

    /// Closed-world validation for a fully loaded source graph.
    pub fn validate_relations(&self) -> Result<(), StandardsRegistryErrorV1> {
        for snapshot in self.snapshots.values() {
            for relation in &snapshot.relations {
                if !self.documents.contains_key(&relation.target) {
                    return Err(StandardsRegistryErrorV1::UnknownRelationTarget {
                        snapshot: snapshot.id.clone(),
                        target: relation.target.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn source_for_claim(
        &self,
        claim: &TechnicalKnowledgeClaimV1,
    ) -> Option<(&TechnicalSourceSnapshotV1, &TechnicalSourceDocumentV1)> {
        let snapshot = self.snapshots.get(&claim.source_snapshot)?;
        let document = self.documents.get(&snapshot.document_id)?;
        Some((snapshot, document))
    }
}

impl SupportKnowledgeSourceV1 for TechnicalStandardsRegistryV1 {
    fn search_support_knowledge(
        &mut self,
        query: &SupportKnowledgeQueryV1,
    ) -> Result<Vec<SupportKnowledgeHitV1>, KnowledgeSourceErrorV1> {
        query.validate()?;
        let query_tokens = tokens(&query.text);
        let mut hits = Vec::new();

        for claim in self.claims.values() {
            if let Some(required_category) = &query.category {
                if claim.category.as_ref() != Some(required_category) {
                    continue;
                }
            }
            let Some((snapshot, document)) = self.source_for_claim(claim) else {
                continue;
            };

            let established_technology = match (&query.technology, &claim.applicability) {
                (Some(identity), Some(scope)) => match scope.assess(identity) {
                    Ok(assessment) => match assessment.status {
                        ApplicabilityStatusV1::Applicable => Some(identity.clone()),
                        ApplicabilityStatusV1::Inapplicable => continue,
                        ApplicabilityStatusV1::Indeterminate => None,
                    },
                    Err(_) => None,
                },
                _ => None,
            };

            let statement_score = token_coverage(&query_tokens, &tokens(&claim.statement));
            let title_score = token_coverage(&query_tokens, &tokens(&document.title));
            let similarity = (statement_score * 0.8 + title_score * 0.2).clamp(0.0, 1.0);
            if similarity <= 0.0 {
                continue;
            }

            hits.push(SupportKnowledgeHitV1 {
                source_id: format!("standard-claim:{}@{}", claim.id.0, snapshot.id.0),
                title: format!("{} — {}", document.canonical_ref, claim.statement),
                similarity,
                confidence: None,
                provider_quality: claim.extraction_quality,
                category: claim.category.clone(),
                origin: origin_for_document(document),
                lifecycle: snapshot.lifecycle,
                authority: snapshot.authority,
                stability: snapshot.stability,
                shareability: KnowledgeShareabilityV1::ReviewRequired,
                technology: established_technology,
            });
        }

        hits.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.source_id.cmp(&b.source_id))
        });
        hits.truncate(query.limit);
        Ok(hits)
    }
}

fn origin_for_document(document: &TechnicalSourceDocumentV1) -> KnowledgeOriginV1 {
    match (&document.publisher, &document.kind) {
        (TechnicalPublisherV1::Vendor(_), _) | (_, SourceDocumentKindV1::VendorDocumentation) => {
            KnowledgeOriginV1::VendorDocumentation
        }
        _ => KnowledgeOriginV1::ExternalStandard,
    }
}

fn tokens(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-' && ch != '_')
        .map(|part| part.trim().to_ascii_lowercase())
        .filter(|part| !part.is_empty())
        .collect()
}

/// Fraction of query tokens covered by candidate text. Empty queries are rejected
/// by `SupportKnowledgeQueryV1::validate`, but this helper remains total.
fn token_coverage(query: &BTreeSet<String>, candidate: &BTreeSet<String>) -> f32 {
    if query.is_empty() {
        return 0.0;
    }
    let overlap = query.intersection(candidate).count();
    overlap as f32 / query.len() as f32
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), StandardsRegistryErrorV1> {
    if value.trim().is_empty() {
        Err(StandardsRegistryErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_publisher(publisher: &TechnicalPublisherV1) -> Result<(), StandardsRegistryErrorV1> {
    match publisher {
        TechnicalPublisherV1::Vendor(value)
        | TechnicalPublisherV1::StandardsBody(value)
        | TechnicalPublisherV1::Other(value) => require_nonempty(value, "publisher"),
        _ => Ok(()),
    }
}

fn validate_document_kind(kind: &SourceDocumentKindV1) -> Result<(), StandardsRegistryErrorV1> {
    match kind {
        SourceDocumentKindV1::Other(value) => require_nonempty(value, "document kind"),
        _ => Ok(()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StandardsRegistryErrorV1 {
    EmptyField(&'static str),
    InvalidExtractionQuality(f32),
    InvalidApplicability(String),
    UnknownDocument(SourceDocumentIdV1),
    UnknownSnapshot(SourceSnapshotIdV1),
    UnknownRelationTarget {
        snapshot: SourceSnapshotIdV1,
        target: SourceDocumentIdV1,
    },
    DocumentIdentityConflict(SourceDocumentIdV1),
    SnapshotIdentityConflict(SourceSnapshotIdV1),
    ClaimIdentityConflict(TechnicalClaimIdV1),
}

impl fmt::Display for StandardsRegistryErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty standards registry field {field}"),
            Self::InvalidExtractionQuality(value) => {
                write!(f, "invalid extraction quality {value}")
            }
            Self::InvalidApplicability(message) => write!(f, "invalid applicability: {message}"),
            Self::UnknownDocument(id) => write!(f, "unknown source document {}", id.0),
            Self::UnknownSnapshot(id) => write!(f, "unknown source snapshot {}", id.0),
            Self::UnknownRelationTarget { snapshot, target } => write!(
                f,
                "snapshot {} references unknown source document {}",
                snapshot.0, target.0
            ),
            Self::DocumentIdentityConflict(id) => {
                write!(f, "source document identity {} was rebound", id.0)
            }
            Self::SnapshotIdentityConflict(id) => {
                write!(f, "source snapshot identity {} was rebound", id.0)
            }
            Self::ClaimIdentityConflict(id) => {
                write!(f, "technical claim identity {} was rebound", id.0)
            }
        }
    }
}

impl Error for StandardsRegistryErrorV1 {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_source::KnowledgeQueryPurposeV1;
    use crate::technology::{StringSelectorV1, TechnologyIdentityV1};

    fn document(id: &str) -> TechnicalSourceDocumentV1 {
        TechnicalSourceDocumentV1 {
            id: SourceDocumentIdV1(id.into()),
            publisher: TechnicalPublisherV1::IetfRfcEditor,
            kind: SourceDocumentKindV1::Rfc,
            title: "QUIC: A UDP-Based Multiplexed and Secure Transport".into(),
            canonical_ref: "RFC 9000".into(),
            canonical_locator: Some("https://www.rfc-editor.org/rfc/rfc9000".into()),
        }
    }

    fn snapshot(id: &str, document_id: &str) -> TechnicalSourceSnapshotV1 {
        TechnicalSourceSnapshotV1 {
            id: SourceSnapshotIdV1(id.into()),
            document_id: SourceDocumentIdV1(document_id.into()),
            version: Some("RFC 9000".into()),
            lifecycle: KnowledgeLifecycleV1::Active,
            authority: KnowledgeAuthorityClassV1::NormativeStandard,
            stability: KnowledgeStabilityV1::Stable,
            published_at_unix_ms: None,
            source_updated_at_unix_ms: None,
            fetched_at_unix_ms: 1_700_000_000_000,
            capture: SourceCaptureV1::MetadataDigest {
                algorithm: "sha256".into(),
                digest: "abc123".into(),
            },
            relations: BTreeSet::new(),
        }
    }

    fn claim(id: &str, snapshot_id: &str) -> TechnicalKnowledgeClaimV1 {
        TechnicalKnowledgeClaimV1 {
            id: TechnicalClaimIdV1(id.into()),
            statement: "QUIC provides multiplexed streams over UDP transport".into(),
            source_snapshot: SourceSnapshotIdV1(snapshot_id.into()),
            locator: Some(TechnicalSourceLocatorV1 {
                section: Some("1".into()),
                fragment: None,
            }),
            modality: ClaimModalityV1::Descriptive,
            applicability: None,
            extraction_quality: Some(0.95),
            category: Some(SupportCategory::Network),
        }
    }

    fn query(text: &str) -> SupportKnowledgeQueryV1 {
        SupportKnowledgeQueryV1 {
            text: text.into(),
            limit: 10,
            category: Some(SupportCategory::Network),
            technology: None,
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        }
    }

    #[test]
    fn exact_replay_is_idempotent_and_identity_rebinding_is_rejected() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        let doc = document("rfc:9000");
        assert_eq!(registry.register_document(doc.clone()).unwrap(), true);
        assert_eq!(registry.register_document(doc.clone()).unwrap(), false);
        let mut rebound = doc;
        rebound.title = "different".into();
        assert!(matches!(
            registry.register_document(rebound),
            Err(StandardsRegistryErrorV1::DocumentIdentityConflict(_))
        ));
    }

    #[test]
    fn logical_document_accepts_multiple_immutable_snapshots() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        registry.register_document(document("iana:ports")).unwrap();
        let mut first = snapshot("iana:ports@2026-09-01", "iana:ports");
        first.version = Some("last-updated:2026-09-01".into());
        let mut second = snapshot("iana:ports@2026-09-07", "iana:ports");
        second.version = Some("last-updated:2026-09-07".into());
        assert!(registry.register_snapshot(first).unwrap());
        assert!(registry.register_snapshot(second).unwrap());
        assert_eq!(registry.snapshots().count(), 2);
    }

    #[test]
    fn snapshots_and_claims_require_known_owners() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        assert!(matches!(
            registry.register_snapshot(snapshot("s", "missing")),
            Err(StandardsRegistryErrorV1::UnknownDocument(_))
        ));
        assert!(matches!(
            registry.register_claim(claim("c", "missing")),
            Err(StandardsRegistryErrorV1::UnknownSnapshot(_))
        ));
    }

    #[test]
    fn closed_world_relation_validation_catches_dangling_documents() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        registry.register_document(document("rfc:9000")).unwrap();
        let mut snap = snapshot("rfc:9000@final", "rfc:9000");
        snap.relations.insert(SourceRelationV1 {
            kind: SourceRelationKindV1::Updates,
            target: SourceDocumentIdV1("rfc:missing".into()),
        });
        registry.register_snapshot(snap).unwrap();
        assert!(matches!(
            registry.validate_relations(),
            Err(StandardsRegistryErrorV1::UnknownRelationTarget { .. })
        ));
    }

    #[test]
    fn invalid_digest_capture_is_rejected() {
        let capture = SourceCaptureV1::ContentDigest {
            algorithm: "".into(),
            digest: "abc".into(),
        };
        assert!(capture.validate().is_err());
    }

    #[test]
    fn retrieval_preserves_lifecycle_authority_and_extraction_quality() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        registry.register_document(document("rfc:9000")).unwrap();
        let mut snap = snapshot("rfc:9000@historic", "rfc:9000");
        snap.lifecycle = KnowledgeLifecycleV1::Historic;
        registry.register_snapshot(snap).unwrap();
        registry.register_claim(claim("quic-streams", "rfc:9000@historic")).unwrap();

        let hit = registry
            .search_support_knowledge(&query("QUIC streams UDP"))
            .unwrap()
            .remove(0);
        assert_eq!(hit.lifecycle, KnowledgeLifecycleV1::Historic);
        assert_eq!(hit.authority, KnowledgeAuthorityClassV1::NormativeStandard);
        assert_eq!(hit.stability, KnowledgeStabilityV1::Stable);
        assert_eq!(hit.confidence, None);
        assert_eq!(hit.provider_quality, Some(0.95));
    }

    #[test]
    fn applicable_technology_is_established_but_mismatch_is_filtered() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        registry.register_document(document("doc:postgres")).unwrap();
        registry
            .register_snapshot(snapshot("doc:postgres@18", "doc:postgres"))
            .unwrap();
        let mut c = claim("postgres-tx", "doc:postgres@18");
        c.statement = "PostgreSQL transaction behavior".into();
        c.category = Some(SupportCategory::Software);
        let mut scope = ApplicabilityScopeV1::default();
        scope.product = StringSelectorV1::Exact("postgresql".into());
        scope.version = StringSelectorV1::Prefix("18".into());
        c.applicability = Some(scope);
        registry.register_claim(c).unwrap();

        let identity = TechnologyIdentityV1 {
            ecosystem: Some("postgresql".into()),
            vendor: None,
            product: "postgresql".into(),
            edition: None,
            version: Some("18.2".into()),
            build: None,
            architecture: None,
            platform: Some("linux".into()),
            profile: None,
            observed_features: BTreeSet::new(),
        };
        let mut q = SupportKnowledgeQueryV1 {
            text: "PostgreSQL transaction".into(),
            limit: 5,
            category: Some(SupportCategory::Software),
            technology: Some(identity.clone()),
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let hits = registry.search_support_knowledge(&q).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].technology, Some(identity));

        q.technology.as_mut().unwrap().version = Some("17.9".into());
        assert!(registry.search_support_knowledge(&q).unwrap().is_empty());
    }

    #[test]
    fn missing_required_version_keeps_claim_but_does_not_establish_applicability() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        registry.register_document(document("doc:postgres")).unwrap();
        registry
            .register_snapshot(snapshot("doc:postgres@18", "doc:postgres"))
            .unwrap();
        let mut c = claim("postgres-tx", "doc:postgres@18");
        c.statement = "PostgreSQL transaction behavior".into();
        c.category = Some(SupportCategory::Software);
        let mut scope = ApplicabilityScopeV1::default();
        scope.product = StringSelectorV1::Exact("postgresql".into());
        scope.version = StringSelectorV1::Prefix("18".into());
        c.applicability = Some(scope);
        registry.register_claim(c).unwrap();

        let q = SupportKnowledgeQueryV1 {
            text: "PostgreSQL transaction".into(),
            limit: 5,
            category: Some(SupportCategory::Software),
            technology: Some(TechnologyIdentityV1 {
                ecosystem: None,
                vendor: None,
                product: "postgresql".into(),
                edition: None,
                version: None,
                build: None,
                architecture: None,
                platform: None,
                profile: None,
                observed_features: BTreeSet::new(),
            }),
            purpose: KnowledgeQueryPurposeV1::LocalReasoning,
        };
        let hits = registry.search_support_knowledge(&q).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].technology, None);
    }

    #[test]
    fn lexical_retrieval_is_deterministic() {
        let mut registry = TechnicalStandardsRegistryV1::new();
        registry.register_document(document("rfc:9000")).unwrap();
        registry
            .register_snapshot(snapshot("rfc:9000@final", "rfc:9000"))
            .unwrap();
        registry
            .register_claim(claim("quic-streams", "rfc:9000@final"))
            .unwrap();
        let first = registry.search_support_knowledge(&query("QUIC UDP streams")).unwrap();
        let second = registry.search_support_knowledge(&query("QUIC UDP streams")).unwrap();
        assert_eq!(first, second);
    }
}
