// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Regression coverage for monotonic-generation assumptions at the admission
//! boundary. The admission token requires exact generation equality, so a live
//! source reporting an older policy or signer-trust generation fails closed.
//!
//! This does not claim to detect an authoritative store that has itself been
//! rolled back to the exact historical state that originally minted the token.
//! Detecting that requires a durable anti-rollback anchor in the production
//! authority store, outside this runtime-neutral contract crate.

use symthaea_extension_admission::{
    AdmissionContext, AdmissionCurrentnessSource, AdmissionProblem, AdmissionRecord,
    AdmissionSubject, PrincipalId, Sha256Digest, TrustLevel,
};
use symthaea_extension_core::{
    AbiVersion, CapabilityDescriptor, CapabilityId, EffectClass, ExtensionId, ExtensionKind,
    ExtensionManifest, PermissionSet, ResourceBudget, RuntimeKind,
};

#[derive(Debug, Clone, Copy)]
struct FixedCurrentness(AdmissionContext);

impl AdmissionCurrentnessSource for FixedCurrentness {
    fn current_context(&self, _subject: AdmissionSubject<'_>) -> Option<AdmissionContext> {
        Some(self.0)
    }
}

fn manifest() -> ExtensionManifest {
    ExtensionManifest {
        id: ExtensionId::new("org.example.rollback"),
        name: "Rollback regression provider".into(),
        version: "1.0.0".into(),
        abi: AbiVersion::V1,
        kind: ExtensionKind::Simulation,
        runtime: RuntimeKind::Wasm,
        description: String::new(),
        provides: vec![CapabilityDescriptor {
            id: CapabilityId::new("engineering.simulation.circuit"),
            description: "test capability".into(),
            effect: EffectClass::Pure,
        }],
        requires: vec![],
        permissions: PermissionSet::default(),
        resources: ResourceBudget::default(),
    }
}

fn record() -> AdmissionRecord {
    AdmissionRecord::issue(
        ExtensionId::new("org.example.rollback"),
        "1.0.0",
        Sha256Digest::new([1; 32]),
        Sha256Digest::new([2; 32]),
        Sha256Digest::new([3; 32]),
        PrincipalId::new("local:extension-authority").unwrap(),
        Some(PrincipalId::new("did:example:rollback-publisher").unwrap()),
        TrustLevel::Trusted,
        vec![CapabilityId::new("engineering.simulation.circuit")],
        PermissionSet::default(),
        7,
        13,
    )
    .unwrap()
}

#[test]
fn older_policy_generation_fails_at_activation_and_use() {
    let manifest = manifest();
    let record = record();
    let rollback = FixedCurrentness(AdmissionContext::active(6, 13));

    assert_eq!(
        record.activate(&manifest, &rollback),
        Err(AdmissionProblem::GenerationMismatch {
            admitted: 7,
            current: 6,
        })
    );

    let active = record
        .activate(
            &manifest,
            &FixedCurrentness(AdmissionContext::active(7, 13)),
        )
        .unwrap();
    assert_eq!(
        active.recheck_currentness(&rollback),
        Err(AdmissionProblem::GenerationMismatch {
            admitted: 7,
            current: 6,
        })
    );
}

#[test]
fn older_signer_trust_generation_fails_at_activation_and_use() {
    let manifest = manifest();
    let record = record();
    let rollback = FixedCurrentness(AdmissionContext::active(7, 12));

    assert_eq!(
        record.activate(&manifest, &rollback),
        Err(AdmissionProblem::TrustGenerationMismatch {
            admitted: 13,
            current: 12,
        })
    );

    let active = record
        .activate(
            &manifest,
            &FixedCurrentness(AdmissionContext::active(7, 13)),
        )
        .unwrap();
    assert_eq!(
        active.recheck_currentness(&rollback),
        Err(AdmissionProblem::TrustGenerationMismatch {
            admitted: 13,
            current: 12,
        })
    );
}
