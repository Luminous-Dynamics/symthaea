// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_authority::{AdmissionAuthority, ScopedAdmissionError};
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet};
use symthaea_extension_data_host::{DataHostError, DataPackHost, DataPackValidator};

#[derive(Debug)]
struct CountingValidator {
    capability: CapabilityId,
    validations: Arc<AtomicUsize>,
}

impl DataPackValidator for CountingValidator {
    type Error = &'static str;

    fn capability(&self) -> &CapabilityId {
        &self.capability
    }

    fn validate(&self, _payload: &[u8]) -> Result<(), Self::Error> {
        self.validations.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct FixedCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for FixedCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

#[derive(Debug)]
struct RevokeAfterFirstCheck {
    calls: AtomicUsize,
}

impl AdmissionCurrentnessSource for RevokeAfterFirstCheck {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        Some(if call == 0 {
            AdmissionContext::active(7, 11)
        } else {
            AdmissionContext::revoked(7, 11)
        })
    }
}

fn digest(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Sha256Digest::new(hasher.finalize().into())
}

#[test]
fn revocation_after_validation_withholds_validated_handle() {
    let capability = "knowledge.example.records";
    let manifest_bytes = br#"{"id":"org.example.knowledge","name":"Example Knowledge","version":"1.0.0","kind":"knowledge_pack","runtime":"data_only","provides":[{"id":"knowledge.example.records","description":"Example records","effect":"pure"}],"resources":{"memory_bytes":0,"fuel":0,"max_wall_time_ms":0,"max_output_bytes":0,"max_concurrency":0}}"#;
    let payload = b"records:v1\nalpha";
    let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes).unwrap();
    let record = AdmissionRecord::issue(
        manifest.id.clone(),
        manifest.version.clone(),
        digest(manifest_bytes),
        digest(payload),
        Sha256Digest::new([9; 32]),
        PrincipalId::new("local:test-authority").unwrap(),
        Some(PrincipalId::new("did:example:data-publisher").unwrap()),
        TrustLevel::Community,
        vec![CapabilityId::new(capability)],
        PermissionSet::default(),
        7,
        11,
    )
    .unwrap();

    let authority = AdmissionAuthority::new();
    let scoped = authority
        .activate(
            &record,
            &manifest,
            &FixedCurrentness(AdmissionContext::active(7, 11)),
        )
        .unwrap();
    let validations = Arc::new(AtomicUsize::new(0));
    let host = DataPackHost::new(
        CountingValidator {
            capability: CapabilityId::new(capability),
            validations: validations.clone(),
        },
        authority.scope(),
    );
    let source = RevokeAfterFirstCheck {
        calls: AtomicUsize::new(0),
    };

    let error = host
        .open(manifest_bytes, payload, &scoped, &source)
        .unwrap_err();

    assert_eq!(validations.load(Ordering::SeqCst), 1);
    assert_eq!(
        error,
        DataHostError::AdmissionAuthority(ScopedAdmissionError::PostUseCurrentness(
            AdmissionProblem::Revoked
        ))
    );
}
