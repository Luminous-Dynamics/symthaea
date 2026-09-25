use std::collections::BTreeMap;

use symthaea_engineering_catalog::{
    CatalogError, EngineeringSourceDocumentV1, SourceKindV1, SpecificationEvidenceClassV1,
    SpecificationEvidenceRefV1,
};

fn document(kind: SourceKindV1, digest_char: char) -> EngineeringSourceDocumentV1 {
    EngineeringSourceDocumentV1 {
        source_kind: kind,
        publisher_id: "fixture:publisher".into(),
        document_id: "fixture:document".into(),
        revision: Some("v1".into()),
        content_sha256: std::iter::repeat_n(digest_char, 64).collect(),
        locator: None,
        retrieved_at: None,
    }
}

fn evidence(
    class: SpecificationEvidenceClassV1,
    document: EngineeringSourceDocumentV1,
) -> (
    BTreeMap<symthaea_engineering_catalog::SourceDocumentId, EngineeringSourceDocumentV1>,
    SpecificationEvidenceRefV1,
) {
    let id = document.source_document_id().unwrap();
    (
        BTreeMap::from([(id.clone(), document)]),
        SpecificationEvidenceRefV1 {
            specification_kind_id: "fixture.specification".into(),
            evidence_class: class,
            source_document_id: Some(id),
            applicability_profile_id: None,
        },
    )
}

#[test]
fn internally_measured_only_accepts_internal_measurement_record() {
    let valid = document(SourceKindV1::InternalMeasurementRecord, 'a');
    let (docs, evidence) = evidence(SpecificationEvidenceClassV1::InternallyMeasured, valid);
    assert!(evidence.validate(&docs).is_ok());

    for (index, invalid_kind) in [
        SourceKindV1::ManufacturerDatasheet,
        SourceKindV1::DistributorDocument,
        SourceKindV1::ImportedDatabase,
        SourceKindV1::CommunityReport,
        SourceKindV1::InternalModelRecord,
        SourceKindV1::StandardOrHandbook,
    ]
    .into_iter()
    .enumerate()
    {
        let invalid = document(invalid_kind, char::from(b'b' + index as u8));
        let (docs, evidence) = evidence(SpecificationEvidenceClassV1::InternallyMeasured, invalid);
        assert!(matches!(
            evidence.validate(&docs),
            Err(CatalogError::EvidenceSourceClassMismatch { .. })
        ));
    }
}

#[test]
fn evidence_classes_do_not_cross_source_authority_boundaries() {
    let cases = [
        (
            SpecificationEvidenceClassV1::ManufacturerGuaranteed,
            SourceKindV1::ManufacturerDatasheet,
        ),
        (
            SpecificationEvidenceClassV1::DistributorMetadata,
            SourceKindV1::DistributorDocument,
        ),
        (
            SpecificationEvidenceClassV1::ImportedDatabase,
            SourceKindV1::ImportedDatabase,
        ),
        (
            SpecificationEvidenceClassV1::CommunityReported,
            SourceKindV1::CommunityReport,
        ),
        (
            SpecificationEvidenceClassV1::DerivedModel,
            SourceKindV1::InternalModelRecord,
        ),
    ];

    for (index, (class, source_kind)) in cases.into_iter().enumerate() {
        let doc = document(source_kind, char::from(b'a' + index as u8));
        let (docs, evidence) = evidence(class, doc);
        assert!(evidence.validate(&docs).is_ok());
    }

    let distributor = document(SourceKindV1::DistributorDocument, 'f');
    let (docs, manufacturer_claim) = evidence(
        SpecificationEvidenceClassV1::ManufacturerGuaranteed,
        distributor,
    );
    assert!(matches!(
        manufacturer_claim.validate(&docs),
        Err(CatalogError::EvidenceSourceClassMismatch { .. })
    ));
}

#[test]
fn assumption_and_unknown_cannot_gain_document_authority() {
    for class in [
        SpecificationEvidenceClassV1::Assumption,
        SpecificationEvidenceClassV1::Unknown,
    ] {
        let doc = document(SourceKindV1::ManufacturerDatasheet, 'a');
        let (docs, evidence) = evidence(class, doc);
        assert_eq!(
            evidence.validate(&docs),
            Err(CatalogError::SourceDocumentNotAdmitted)
        );
    }
}

#[test]
fn serialization_does_not_bypass_source_compatibility() {
    let doc = document(SourceKindV1::ImportedDatabase, 'a');
    let (docs, invalid) = evidence(SpecificationEvidenceClassV1::InternallyMeasured, doc);
    let encoded = serde_json::to_string(&invalid).unwrap();
    let decoded: SpecificationEvidenceRefV1 = serde_json::from_str(&encoded).unwrap();
    assert!(matches!(
        decoded.validate(&docs),
        Err(CatalogError::EvidenceSourceClassMismatch { .. })
    ));
}
