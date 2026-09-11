// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use sha2::{Digest, Sha256};
use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionRecord, AdmissionSubject, PrincipalId,
    Sha256Digest, TrustLevel,
};
use symthaea_extension_authority::AdmissionAuthority;
use symthaea_extension_core::{CapabilityId, ExtensionManifest, PermissionSet};
use symthaea_extension_data_host::{DataHostError, DataPackHost, DataPackValidator};

#[derive(Debug)]
struct AcceptingValidator {
    capability: CapabilityId,
}

impl DataPackValidator for AcceptingValidator {
    type Error = &'static str;

    fn capability(&self) -> &CapabilityId {
        &self.capability
    }

    fn validate(&self, _payload: &[u8]) -> Result<(), Self::Error> {
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

fn current() -> FixedCurrentness {
    FixedCurrentness(AdmissionContext::active(7, 11))
}

fn digest(bytes: &[u8]) -> Sha256Digest {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Sha256Digest::new(hasher.finalize().into())
}

#[test]
fn semantically_equal_manifest_bytes_do_not_reuse_admission() {
    let capability = "knowledge.example.records";
    let compact = br#"{"id":"org.example.knowledge","name":"Example Knowledge","version":"1.0.0","kind":"knowledge_pack","runtime":"data_only","provides":[{"id":"knowledge.example.records","description":"Example records","effect":"pure"}],"resources":{"memory_bytes":0,"fuel":0,"max_wall_time_ms":0,"max_output_bytes":0,"max_concurrency":0}}"#;
    let pretty = br#"{
  "id": "org.example.knowledge",
  "name": "Example Knowledge",
  "version": "1.0.0",
  "kind": "knowledge_pack",
  "runtime": "data_only",
  "provides": [
    {
      "id": "knowledge.example.records",
      "description": "Example records",
      "effect": "pure"
    }
  ],
  "resources": {
    "memory_bytes": 0,
    "fuel": 0,
    "max_wall_time_ms": 0,
    "max_output_bytes": 0,
    "max_concurrency": 0
  }
}"#;
    let payload = b"records:v1\nalpha";

    let compact_manifest: ExtensionManifest = serde_json::from_slice(compact).unwrap();
    let pretty_manifest: ExtensionManifest = serde_json::from_slice(pretty).unwrap();
    assert_eq!(compact_manifest, pretty_manifest);

    let source = current();
    let record = AdmissionRecord::issue(
        compact_manifest.id.clone(),
        compact_manifest.version.clone(),
        digest(compact),
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
        .activate(&record, &compact_manifest, &source)
        .unwrap();

    let host = DataPackHost::new(
        AcceptingValidator {
            capability: CapabilityId::new(capability),
        },
        authority.scope(),
    );

    let error = host.open(pretty, payload, &scoped, &source).unwrap_err();
    assert_eq!(error, DataHostError::ManifestDigestMismatch);
}
