// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Generic physical-component and source-document identity for Symthaea engineering.
//!
//! This crate intentionally does **not** define canonical engineering quantities or units.
//! SE-SEM-001 owns that boundary. The catalog layer establishes exact component identity,
//! source-document identity, evidence classes, and deterministic evidence snapshots that
//! domain crates can interpret later.
//!
//! Core theorem:
//!
//! ```text
//! friendly component name
//! != manufacturer part number
//! != exact variant/revision
//! != source-document revision
//! != supplier listing
//! != purchased article
//! != installed article
//! ```

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

const COMPONENT_ID_DOMAIN: &str = "symthaea-engineering-catalog::component-v1";
const DOCUMENT_ID_DOMAIN: &str = "symthaea-engineering-catalog::source-document-v1";
const SNAPSHOT_ID_DOMAIN: &str = "symthaea-engineering-catalog::evidence-snapshot-v1";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CatalogError {
    #[error("{field} must not be empty")]
    EmptyField { field: &'static str },
    #[error("{field} contains leading or trailing whitespace")]
    NonCanonicalWhitespace { field: &'static str },
    #[error("digest must be exactly 64 lowercase hexadecimal characters")]
    InvalidSha256,
    #[error("duplicate source-document identity in snapshot: {0}")]
    DuplicateDocument(String),
    #[error("specification evidence references an unknown source-document identity: {0}")]
    UnknownDocument(String),
    #[error("duplicate specification evidence key: {0}")]
    DuplicateSpecification(String),
    #[error("evidence class requires a source document")]
    SourceDocumentRequired,
    #[error("evidence class does not admit a source document")]
    SourceDocumentNotAdmitted,
    #[error("evidence class {evidence_class} is incompatible with source kind {source_kind}")]
    EvidenceSourceClassMismatch {
        evidence_class: &'static str,
        source_kind: String,
    },
}

fn validate_token(field: &'static str, value: &str) -> Result<(), CatalogError> {
    if value.is_empty() {
        return Err(CatalogError::EmptyField { field });
    }
    if value.trim() != value {
        return Err(CatalogError::NonCanonicalWhitespace { field });
    }
    Ok(())
}

fn validate_sha256_hex(value: &str) -> Result<(), CatalogError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(CatalogError::InvalidSha256);
    }
    Ok(())
}

fn hash_field(hasher: &mut blake3::Hasher, value: &str) {
    let bytes = value.as_bytes();
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn hash_optional_field(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            hash_field(hasher, value);
        }
        None => hasher.update(&[0]),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ComponentSubjectId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SourceDocumentId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EvidenceSnapshotId(pub String);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManufacturerIdentityV1 {
    pub manufacturer_id: String,
    pub display_name: Option<String>,
}

impl ManufacturerIdentityV1 {
    pub fn validate(&self) -> Result<(), CatalogError> {
        validate_token("manufacturer_id", &self.manufacturer_id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComponentClassV1 {
    Electrical,
    Power,
    Magnetics,
    Vacuum,
    GasHandling,
    Optical,
    Photonic,
    Semiconductor,
    Sensor,
    Instrument,
    Thermal,
    Mechanical,
    Acoustic,
    MaterialArticle,
    Connector,
    CableHarness,
    Other(String),
}

impl ComponentClassV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::Electrical => "electrical".into(),
            Self::Power => "power".into(),
            Self::Magnetics => "magnetics".into(),
            Self::Vacuum => "vacuum".into(),
            Self::GasHandling => "gas-handling".into(),
            Self::Optical => "optical".into(),
            Self::Photonic => "photonic".into(),
            Self::Semiconductor => "semiconductor".into(),
            Self::Sensor => "sensor".into(),
            Self::Instrument => "instrument".into(),
            Self::Thermal => "thermal".into(),
            Self::Mechanical => "mechanical".into(),
            Self::Acoustic => "acoustic".into(),
            Self::MaterialArticle => "material-article".into(),
            Self::Connector => "connector".into(),
            Self::CableHarness => "cable-harness".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), CatalogError> {
        if let Self::Other(value) = self {
            validate_token("component_class.other", value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentSubjectV1 {
    pub manufacturer: ManufacturerIdentityV1,
    pub manufacturer_part_number: String,
    pub variant: Option<String>,
    pub hardware_revision: Option<String>,
    pub component_class: ComponentClassV1,
    pub display_name: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
}

impl ComponentSubjectV1 {
    pub fn validate(&self) -> Result<(), CatalogError> {
        self.manufacturer.validate()?;
        self.component_class.validate()?;
        validate_token("manufacturer_part_number", &self.manufacturer_part_number)?;
        if let Some(value) = self.variant.as_deref() {
            validate_token("variant", value)?;
        }
        if let Some(value) = self.hardware_revision.as_deref() {
            validate_token("hardware_revision", value)?;
        }
        Ok(())
    }

    pub fn component_id(&self) -> Result<ComponentSubjectId, CatalogError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, COMPONENT_ID_DOMAIN);
        hash_field(&mut hasher, &self.manufacturer.manufacturer_id);
        hash_field(&mut hasher, &self.manufacturer_part_number);
        hash_optional_field(&mut hasher, self.variant.as_deref());
        hash_optional_field(&mut hasher, self.hardware_revision.as_deref());
        hash_field(&mut hasher, &self.component_class.canonical_tag());
        Ok(ComponentSubjectId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SourceKindV1 {
    ManufacturerDatasheet,
    ManufacturerApplicationNote,
    ManufacturerDrawing,
    ManufacturerCertificate,
    DistributorDocument,
    StandardOrHandbook,
    ImportedDatabase,
    InternalMeasurementRecord,
    InternalModelRecord,
    CommunityReport,
    Other(String),
}

impl SourceKindV1 {
    fn canonical_tag(&self) -> String {
        match self {
            Self::ManufacturerDatasheet => "manufacturer-datasheet".into(),
            Self::ManufacturerApplicationNote => "manufacturer-application-note".into(),
            Self::ManufacturerDrawing => "manufacturer-drawing".into(),
            Self::ManufacturerCertificate => "manufacturer-certificate".into(),
            Self::DistributorDocument => "distributor-document".into(),
            Self::StandardOrHandbook => "standard-or-handbook".into(),
            Self::ImportedDatabase => "imported-database".into(),
            Self::InternalMeasurementRecord => "internal-measurement-record".into(),
            Self::InternalModelRecord => "internal-model-record".into(),
            Self::CommunityReport => "community-report".into(),
            Self::Other(value) => format!("other:{value}"),
        }
    }

    fn validate(&self) -> Result<(), CatalogError> {
        if let Self::Other(value) = self {
            validate_token("source_kind.other", value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineeringSourceDocumentV1 {
    pub source_kind: SourceKindV1,
    pub publisher_id: String,
    pub document_id: String,
    pub revision: Option<String>,
    pub content_sha256: String,
    pub locator: Option<String>,
    pub retrieved_at: Option<String>,
}

impl EngineeringSourceDocumentV1 {
    pub fn validate(&self) -> Result<(), CatalogError> {
        self.source_kind.validate()?;
        validate_token("publisher_id", &self.publisher_id)?;
        validate_token("document_id", &self.document_id)?;
        if let Some(value) = self.revision.as_deref() {
            validate_token("revision", value)?;
        }
        validate_sha256_hex(&self.content_sha256)
    }

    pub fn source_document_id(&self) -> Result<SourceDocumentId, CatalogError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, DOCUMENT_ID_DOMAIN);
        hash_field(&mut hasher, &self.source_kind.canonical_tag());
        hash_field(&mut hasher, &self.publisher_id);
        hash_field(&mut hasher, &self.document_id);
        hash_optional_field(&mut hasher, self.revision.as_deref());
        hash_field(&mut hasher, &self.content_sha256);
        Ok(SourceDocumentId(hasher.finalize().to_hex().to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpecificationEvidenceClassV1 {
    ManufacturerGuaranteed,
    ManufacturerTypical,
    ManufacturerAbsoluteMaximum,
    ManufacturerNominal,
    DistributorMetadata,
    ImportedDatabase,
    CommunityReported,
    InternallyMeasured,
    DerivedModel,
    Assumption,
    Unknown,
}

impl SpecificationEvidenceClassV1 {
    pub fn requires_source_document(self) -> bool {
        !matches!(self, Self::Assumption | Self::Unknown)
    }

    fn canonical_tag(self) -> &'static str {
        match self {
            Self::ManufacturerGuaranteed => "manufacturer-guaranteed",
            Self::ManufacturerTypical => "manufacturer-typical",
            Self::ManufacturerAbsoluteMaximum => "manufacturer-absolute-maximum",
            Self::ManufacturerNominal => "manufacturer-nominal",
            Self::DistributorMetadata => "distributor-metadata",
            Self::ImportedDatabase => "imported-database",
            Self::CommunityReported => "community-reported",
            Self::InternallyMeasured => "internally-measured",
            Self::DerivedModel => "derived-model",
            Self::Assumption => "assumption",
            Self::Unknown => "unknown",
        }
    }

    fn admits_source_kind(self, source: &SourceKindV1) -> bool {
        match self {
            Self::ManufacturerGuaranteed
            | Self::ManufacturerTypical
            | Self::ManufacturerAbsoluteMaximum
            | Self::ManufacturerNominal => matches!(
                source,
                SourceKindV1::ManufacturerDatasheet
                    | SourceKindV1::ManufacturerApplicationNote
                    | SourceKindV1::ManufacturerDrawing
                    | SourceKindV1::ManufacturerCertificate
            ),
            Self::DistributorMetadata => matches!(source, SourceKindV1::DistributorDocument),
            Self::ImportedDatabase => matches!(source, SourceKindV1::ImportedDatabase),
            Self::CommunityReported => matches!(source, SourceKindV1::CommunityReport),
            Self::InternallyMeasured => matches!(source, SourceKindV1::InternalMeasurementRecord),
            Self::DerivedModel => matches!(source, SourceKindV1::InternalModelRecord),
            Self::Assumption | Self::Unknown => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpecificationEvidenceRefV1 {
    pub specification_kind_id: String,
    pub evidence_class: SpecificationEvidenceClassV1,
    pub source_document_id: Option<SourceDocumentId>,
    pub applicability_profile_id: Option<String>,
}

impl SpecificationEvidenceRefV1 {
    pub fn validate(
        &self,
        documents: &BTreeMap<SourceDocumentId, EngineeringSourceDocumentV1>,
    ) -> Result<(), CatalogError> {
        validate_token("specification_kind_id", &self.specification_kind_id)?;
        if let Some(value) = self.applicability_profile_id.as_deref() {
            validate_token("applicability_profile_id", value)?;
        }

        match (&self.source_document_id, self.evidence_class.requires_source_document()) {
            (None, true) => return Err(CatalogError::SourceDocumentRequired),
            (Some(_), false) => return Err(CatalogError::SourceDocumentNotAdmitted),
            _ => {}
        }

        if let Some(document_id) = &self.source_document_id {
            let document = documents
                .get(document_id)
                .ok_or_else(|| CatalogError::UnknownDocument(document_id.0.clone()))?;
            if !self.evidence_class.admits_source_kind(&document.source_kind) {
                return Err(CatalogError::EvidenceSourceClassMismatch {
                    evidence_class: self.evidence_class.canonical_tag(),
                    source_kind: document.source_kind.canonical_tag(),
                });
            }
        }
        Ok(())
    }

    fn canonical_key(&self) -> String {
        format!(
            "{}::{}::{}::{}",
            self.specification_kind_id,
            self.evidence_class.canonical_tag(),
            self.source_document_id
                .as_ref()
                .map(|id| id.0.as_str())
                .unwrap_or("none"),
            self.applicability_profile_id.as_deref().unwrap_or("none")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComponentEvidenceSnapshotV1 {
    pub component: ComponentSubjectV1,
    #[serde(default)]
    pub source_documents: Vec<EngineeringSourceDocumentV1>,
    #[serde(default)]
    pub specification_evidence: Vec<SpecificationEvidenceRefV1>,
}

impl ComponentEvidenceSnapshotV1 {
    pub fn validate(
        &self,
    ) -> Result<BTreeMap<SourceDocumentId, EngineeringSourceDocumentV1>, CatalogError> {
        self.component.validate()?;
        let mut documents = BTreeMap::new();
        for document in &self.source_documents {
            let id = document.source_document_id()?;
            if documents.insert(id.clone(), document.clone()).is_some() {
                return Err(CatalogError::DuplicateDocument(id.0));
            }
        }

        let mut specification_keys = BTreeSet::new();
        for evidence in &self.specification_evidence {
            evidence.validate(&documents)?;
            let key = evidence.canonical_key();
            if !specification_keys.insert(key.clone()) {
                return Err(CatalogError::DuplicateSpecification(key));
            }
        }
        Ok(documents)
    }

    pub fn snapshot_id(&self) -> Result<EvidenceSnapshotId, CatalogError> {
        let documents = self.validate()?;
        let component_id = self.component.component_id()?;
        let mut hasher = blake3::Hasher::new();
        hash_field(&mut hasher, SNAPSHOT_ID_DOMAIN);
        hash_field(&mut hasher, &component_id.0);

        for document_id in documents.keys() {
            hash_field(&mut hasher, &document_id.0);
        }

        let mut evidence = self
            .specification_evidence
            .iter()
            .map(SpecificationEvidenceRefV1::canonical_key)
            .collect::<Vec<_>>();
        evidence.sort();
        for key in evidence {
            hash_field(&mut hasher, &key);
        }
        Ok(EvidenceSnapshotId(hasher.finalize().to_hex().to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn component() -> ComponentSubjectV1 {
        ComponentSubjectV1 {
            manufacturer: ManufacturerIdentityV1 {
                manufacturer_id: "maker:acme-optics".into(),
                display_name: Some("ACME Optics".into()),
            },
            manufacturer_part_number: "PD-100".into(),
            variant: Some("TO-can".into()),
            hardware_revision: Some("rev-a".into()),
            component_class: ComponentClassV1::Photonic,
            display_name: Some("Fast photodiode".into()),
            aliases: vec!["lab detector".into()],
        }
    }

    fn document(kind: SourceKindV1, digest_char: char) -> EngineeringSourceDocumentV1 {
        EngineeringSourceDocumentV1 {
            source_kind: kind,
            publisher_id: "maker:acme-optics".into(),
            document_id: "PD-100-document".into(),
            revision: Some("2026-01".into()),
            content_sha256: std::iter::repeat_n(digest_char, 64).collect(),
            locator: Some("https://example.invalid/pd100.pdf".into()),
            retrieved_at: Some("2026-09-25T00:00:00Z".into()),
        }
    }

    fn evidence_for(
        class: SpecificationEvidenceClassV1,
        doc: &EngineeringSourceDocumentV1,
    ) -> (BTreeMap<SourceDocumentId, EngineeringSourceDocumentV1>, SpecificationEvidenceRefV1) {
        let id = doc.source_document_id().unwrap();
        (
            BTreeMap::from([(id.clone(), doc.clone())]),
            SpecificationEvidenceRefV1 {
                specification_kind_id: "test.specification".into(),
                evidence_class: class,
                source_document_id: Some(id),
                applicability_profile_id: None,
            },
        )
    }

    #[test]
    fn navigation_labels_do_not_change_component_identity() {
        let a = component();
        let mut b = a.clone();
        b.display_name = Some("Different UI label".into());
        b.aliases = vec!["another alias".into()];
        b.manufacturer.display_name = Some("Marketing Name".into());
        assert_eq!(a.component_id().unwrap(), b.component_id().unwrap());
    }

    #[test]
    fn semantic_variant_changes_component_identity() {
        let a = component();
        let mut b = a.clone();
        b.variant = Some("fiber-pigtailed".into());
        assert_ne!(a.component_id().unwrap(), b.component_id().unwrap());
    }

    #[test]
    fn source_locator_change_does_not_change_document_identity() {
        let a = document(SourceKindV1::ManufacturerDatasheet, 'a');
        let mut b = a.clone();
        b.locator = Some("file:///archive/pd100.pdf".into());
        b.retrieved_at = Some("2026-09-26T00:00:00Z".into());
        assert_eq!(a.source_document_id().unwrap(), b.source_document_id().unwrap());
    }

    #[test]
    fn changed_source_bytes_change_document_identity() {
        let a = document(SourceKindV1::ManufacturerDatasheet, 'a');
        let b = document(SourceKindV1::ManufacturerDatasheet, 'b');
        assert_ne!(a.source_document_id().unwrap(), b.source_document_id().unwrap());
    }

    #[test]
    fn evidence_snapshot_order_is_canonical() {
        let d1 = document(SourceKindV1::ManufacturerDatasheet, 'a');
        let mut d2 = document(SourceKindV1::ManufacturerDrawing, 'b');
        d2.document_id = "PD-100-drawing".into();
        let d1_id = d1.source_document_id().unwrap();
        let d2_id = d2.source_document_id().unwrap();
        let e1 = SpecificationEvidenceRefV1 {
            specification_kind_id: "optical.responsivity".into(),
            evidence_class: SpecificationEvidenceClassV1::ManufacturerTypical,
            source_document_id: Some(d1_id),
            applicability_profile_id: Some("room-temperature".into()),
        };
        let e2 = SpecificationEvidenceRefV1 {
            specification_kind_id: "mechanical.outline".into(),
            evidence_class: SpecificationEvidenceClassV1::ManufacturerGuaranteed,
            source_document_id: Some(d2_id),
            applicability_profile_id: None,
        };
        let a = ComponentEvidenceSnapshotV1 {
            component: component(),
            source_documents: vec![d1.clone(), d2.clone()],
            specification_evidence: vec![e1.clone(), e2.clone()],
        };
        let b = ComponentEvidenceSnapshotV1 {
            component: component(),
            source_documents: vec![d2, d1],
            specification_evidence: vec![e2, e1],
        };
        assert_eq!(a.snapshot_id().unwrap(), b.snapshot_id().unwrap());
    }

    #[test]
    fn missing_document_fails_closed() {
        let snapshot = ComponentEvidenceSnapshotV1 {
            component: component(),
            source_documents: vec![],
            specification_evidence: vec![SpecificationEvidenceRefV1 {
                specification_kind_id: "optical.responsivity".into(),
                evidence_class: SpecificationEvidenceClassV1::ManufacturerTypical,
                source_document_id: Some(SourceDocumentId("missing".into())),
                applicability_profile_id: None,
            }],
        };
        assert!(matches!(snapshot.validate(), Err(CatalogError::UnknownDocument(_))));
    }

    #[test]
    fn typical_and_guaranteed_are_semantically_distinct() {
        let doc = document(SourceKindV1::ManufacturerDatasheet, 'a');
        let (_, typical) = evidence_for(SpecificationEvidenceClassV1::ManufacturerTypical, &doc);
        let guaranteed = SpecificationEvidenceRefV1 {
            evidence_class: SpecificationEvidenceClassV1::ManufacturerGuaranteed,
            ..typical.clone()
        };
        assert_ne!(typical.canonical_key(), guaranteed.canonical_key());
    }

    #[test]
    fn positive_source_compatibility_accepts_exact_classes() {
        let cases = [
            (SpecificationEvidenceClassV1::ManufacturerTypical, SourceKindV1::ManufacturerDatasheet),
            (SpecificationEvidenceClassV1::DistributorMetadata, SourceKindV1::DistributorDocument),
            (SpecificationEvidenceClassV1::ImportedDatabase, SourceKindV1::ImportedDatabase),
            (SpecificationEvidenceClassV1::CommunityReported, SourceKindV1::CommunityReport),
            (SpecificationEvidenceClassV1::InternallyMeasured, SourceKindV1::InternalMeasurementRecord),
            (SpecificationEvidenceClassV1::DerivedModel, SourceKindV1::InternalModelRecord),
        ];
        for (index, (class, kind)) in cases.into_iter().enumerate() {
            let doc = document(kind, char::from(b'a' + index as u8));
            let (docs, evidence) = evidence_for(class, &doc);
            assert!(evidence.validate(&docs).is_ok());
        }
    }

    #[test]
    fn internal_measurement_rejects_manufacturer_imported_community_and_model_sources() {
        let invalid = [
            SourceKindV1::ManufacturerDatasheet,
            SourceKindV1::DistributorDocument,
            SourceKindV1::ImportedDatabase,
            SourceKindV1::CommunityReport,
            SourceKindV1::InternalModelRecord,
            SourceKindV1::StandardOrHandbook,
        ];
        for (index, kind) in invalid.into_iter().enumerate() {
            let doc = document(kind, char::from(b'a' + index as u8));
            let (docs, evidence) = evidence_for(SpecificationEvidenceClassV1::InternallyMeasured, &doc);
            assert!(matches!(
                evidence.validate(&docs),
                Err(CatalogError::EvidenceSourceClassMismatch { .. })
            ));
        }
    }

    #[test]
    fn manufacturer_claim_cannot_use_distributor_document() {
        let doc = document(SourceKindV1::DistributorDocument, 'a');
        let (docs, evidence) = evidence_for(SpecificationEvidenceClassV1::ManufacturerGuaranteed, &doc);
        assert!(matches!(
            evidence.validate(&docs),
            Err(CatalogError::EvidenceSourceClassMismatch { .. })
        ));
    }

    #[test]
    fn assumption_and_unknown_cannot_gain_document_authority() {
        let doc = document(SourceKindV1::ManufacturerDatasheet, 'a');
        for class in [SpecificationEvidenceClassV1::Assumption, SpecificationEvidenceClassV1::Unknown] {
            let (docs, evidence) = evidence_for(class, &doc);
            assert_eq!(evidence.validate(&docs), Err(CatalogError::SourceDocumentNotAdmitted));
        }
    }

    #[test]
    fn malformed_digest_rejects() {
        let mut doc = document(SourceKindV1::ManufacturerDatasheet, 'a');
        doc.content_sha256 = "ABC".into();
        assert_eq!(doc.validate(), Err(CatalogError::InvalidSha256));
    }

    #[test]
    fn serde_round_trip_preserves_identity_and_validation() {
        let original = component();
        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: ComponentSubjectV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(original.component_id().unwrap(), decoded.component_id().unwrap());

        let doc = document(SourceKindV1::ImportedDatabase, 'c');
        let (docs, bad) = evidence_for(SpecificationEvidenceClassV1::InternallyMeasured, &doc);
        let encoded = serde_json::to_string(&bad).unwrap();
        let decoded: SpecificationEvidenceRefV1 = serde_json::from_str(&encoded).unwrap();
        assert!(matches!(
            decoded.validate(&docs),
            Err(CatalogError::EvidenceSourceClassMismatch { .. })
        ));
    }
}
