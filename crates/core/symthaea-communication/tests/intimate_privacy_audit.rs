// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[path = "../src/intimate_memory_privacy.rs"]
mod intimate_memory_privacy;
#[path = "../src/intimate_privacy_audit.rs"]
mod intimate_privacy_audit;

use intimate_memory_privacy::{
    IntimateRetractionActionV1, IntimateRetractionEventV1, IntimateRetractionReceiptV1,
    IntimateRetractionStatusV1,
};
use intimate_privacy_audit::*;

#[test]
fn audit_receipt_binds_external_deletion_limitation_without_payload() {
    let retraction = IntimateRetractionReceiptV1 {
        source_id: "psychology-evidence-7".into(),
        status: IntimateRetractionStatusV1::Applied,
        events: vec![
            IntimateRetractionEventV1 {
                artifact_id: "psychology-evidence-7".into(),
                action: IntimateRetractionActionV1::DeleteLocal,
            },
            IntimateRetractionEventV1 {
                artifact_id: "clinician-export-2".into(),
                action: IntimateRetractionActionV1::ExternalExportNotice,
            },
        ],
        external_deletion_unproven: true,
    };

    let receipt = IntimatePrivacyAuditReceiptV1::from_retraction(
        "delete-op-42",
        "privacy-policy-v1",
        100,
        125,
        &retraction,
    )
    .unwrap();

    assert!(receipt.external_deletion_unproven);
    assert!(receipt.receipt_commitment.starts_with("intimate-retraction-v1:"));
    assert_eq!(receipt.source_id, "psychology-evidence-7");
}
