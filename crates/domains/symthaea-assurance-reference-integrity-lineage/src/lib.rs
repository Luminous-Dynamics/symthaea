// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed lineage assurance for reference-integrity manifests.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use symthaea_assurance_reference_integrity::{
    ReferenceIntegrityManifest, RimSignatureVerificationReceipt,
};

const LINEAGE_DIGEST_SCHEMA: &[u8] = b"symthaea-reference-integrity-lineage-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceManifestSignerRule {
    pub rule_id: String,
    pub signer_ref: String,
    pub signer_key_digest: String,
    pub first_revision: u64,
    pub last_revision: Option<u64>,
    pub evidence_refs: Vec<String>,
}

impl ReferenceManifestSignerRule {
    pub fn validate(&self) -> bool {
        canonical_text(&self.rule_id)
            && canonical_text(&self.signer_ref)
            && valid_digest(&self.signer_key_digest)
            && self.first_revision > 0
            && self
                .last_revision
                .is_none_or(|last| last >= self.first_revision)
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn applies_to(
        &self,
        revision: u64,
        signer_ref: &str,
        signer_key_digest: &str,
    ) -> bool {
        self.validate()
            && revision >= self.first_revision
            && self.last_revision.is_none_or(|last| revision <= last)
            && self.signer_ref == signer_ref
            && self.signer_key_digest == signer_key_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIntegrityLineagePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_manifest_id: String,
    pub expected_issuer_ref: String,
    pub expected_subject_class_ref: String,
    pub expected_signature_verifier_ref: String,
    pub expected_signature_verification_tool_digest: String,
    pub expected_tip_revision: u64,
    pub expected_tip_manifest_digest: String,
    pub signer_rules: Vec<ReferenceManifestSignerRule>,
    pub evidence_refs: Vec<String>,
}

impl ReferenceIntegrityLineagePolicy {
    pub fn validate(&self) -> bool {
        if !canonical_text(&self.schema_version)
            || !canonical_text(&self.policy_id)
            || !canonical_text(&self.expected_manifest_id)
            || !canonical_text(&self.expected_issuer_ref)
            || !canonical_text(&self.expected_subject_class_ref)
            || !canonical_text(&self.expected_signature_verifier_ref)
            || !valid_digest(&self.expected_signature_verification_tool_digest)
            || self.expected_tip_revision == 0
            || !valid_digest(&self.expected_tip_manifest_digest)
            || self.signer_rules.is_empty()
            || !self
                .signer_rules
                .iter()
                .all(ReferenceManifestSignerRule::validate)
            || !nonempty_refs(&self.evidence_refs)
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.signer_rules
            .iter()
            .all(|rule| ids.insert(rule.rule_id.as_str()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIntegrityLineageEntry {
    pub manifest: ReferenceIntegrityManifest,
    pub signature: RimSignatureVerificationReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIntegrityLineageReport {
    pub schema_version: String,
    pub policy_id: String,
    pub manifest_id: String,
    pub issuer_ref: String,
    pub subject_class_ref: String,
    pub entry_count: u64,
    pub tip_revision: u64,
    pub tip_manifest_digest: String,
    pub manifest_digests: Vec<String>,
    pub signature_receipt_digests: Vec<String>,
    pub evidence_refs: Vec<String>,
}

impl ReferenceIntegrityLineageReport {
    pub fn lineage_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(LINEAGE_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.manifest_id.as_str(),
            self.issuer_ref.as_str(),
            self.subject_class_ref.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.entry_count.to_string());
        push_field(&mut hasher, &self.tip_revision.to_string());
        push_field(&mut hasher, &self.tip_manifest_digest);
        for digest in &self.manifest_digests {
            push_field(&mut hasher, &format!("manifest:{digest}"));
        }
        for digest in &self.signature_receipt_digests {
            push_field(&mut hasher, &format!("signature:{digest}"));
        }
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceIntegrityLineageError {
    InvalidPolicy,
    EmptyLineage,
    InvalidManifest,
    InvalidSignatureReceipt,
    ManifestIdentityChanged,
    SignatureVerifierMismatch,
    UnauthorizedSigner,
    RevisionGapOrReorder,
    GenesisHasPredecessor,
    PredecessorMismatch,
    ManifestDigestReused,
    SourceRimDigestReused,
    SignatureReceiptReused,
    ValidityTimeRegressed,
    TipRevisionMismatch,
    TipDigestMismatch,
}

pub fn qualify_reference_integrity_lineage(
    policy: &ReferenceIntegrityLineagePolicy,
    entries: &[ReferenceIntegrityLineageEntry],
) -> Result<ReferenceIntegrityLineageReport, ReferenceIntegrityLineageError> {
    if !policy.validate() {
        return Err(ReferenceIntegrityLineageError::InvalidPolicy);
    }
    if entries.is_empty() {
        return Err(ReferenceIntegrityLineageError::EmptyLineage);
    }

    let mut manifest_digests = Vec::with_capacity(entries.len());
    let mut signature_digests = Vec::with_capacity(entries.len());
    let mut seen_manifest_digests = BTreeSet::new();
    let mut seen_source_rim_digests = BTreeSet::new();
    let mut seen_signature_receipts = BTreeSet::new();
    let mut previous_manifest_digest: Option<String> = None;
    let mut previous_valid_from_ms: Option<u64> = None;

    for (index, entry) in entries.iter().enumerate() {
        let expected_revision = index as u64 + 1;
        let manifest = &entry.manifest;
        let signature = &entry.signature;

        if !manifest.validate() {
            return Err(ReferenceIntegrityLineageError::InvalidManifest);
        }
        if !signature.validate_for(manifest) {
            return Err(ReferenceIntegrityLineageError::InvalidSignatureReceipt);
        }
        if manifest.manifest_id != policy.expected_manifest_id
            || manifest.issuer_ref != policy.expected_issuer_ref
            || manifest.subject_class_ref != policy.expected_subject_class_ref
        {
            return Err(ReferenceIntegrityLineageError::ManifestIdentityChanged);
        }
        if signature.verifier_ref != policy.expected_signature_verifier_ref
            || signature.verification_tool_digest
                != policy.expected_signature_verification_tool_digest
        {
            return Err(ReferenceIntegrityLineageError::SignatureVerifierMismatch);
        }
        if !policy.signer_rules.iter().any(|rule| {
            rule.applies_to(
                manifest.revision,
                &signature.signer_ref,
                &signature.signer_key_digest,
            )
        }) {
            return Err(ReferenceIntegrityLineageError::UnauthorizedSigner);
        }
        if manifest.revision != expected_revision {
            return Err(ReferenceIntegrityLineageError::RevisionGapOrReorder);
        }

        let manifest_digest = manifest.manifest_digest();
        if manifest.revision == 1 {
            if manifest.predecessor_manifest_digest.is_some() {
                return Err(ReferenceIntegrityLineageError::GenesisHasPredecessor);
            }
        } else if manifest.predecessor_manifest_digest.as_deref()
            != previous_manifest_digest.as_deref()
        {
            return Err(ReferenceIntegrityLineageError::PredecessorMismatch);
        }

        if !seen_manifest_digests.insert(manifest_digest.clone()) {
            return Err(ReferenceIntegrityLineageError::ManifestDigestReused);
        }
        if !seen_source_rim_digests.insert(manifest.source_rim_digest.clone()) {
            return Err(ReferenceIntegrityLineageError::SourceRimDigestReused);
        }

        let signature_digest = signature.receipt_digest();
        if !seen_signature_receipts.insert(signature_digest.clone()) {
            return Err(ReferenceIntegrityLineageError::SignatureReceiptReused);
        }

        if previous_valid_from_ms.is_some_and(|previous| manifest.valid_from_ms < previous) {
            return Err(ReferenceIntegrityLineageError::ValidityTimeRegressed);
        }

        previous_manifest_digest = Some(manifest_digest.clone());
        previous_valid_from_ms = Some(manifest.valid_from_ms);
        manifest_digests.push(manifest_digest);
        signature_digests.push(signature_digest);
    }

    let tip = entries.last().expect("non-empty checked");
    let tip_digest = tip.manifest.manifest_digest();
    if tip.manifest.revision != policy.expected_tip_revision {
        return Err(ReferenceIntegrityLineageError::TipRevisionMismatch);
    }
    if tip_digest != policy.expected_tip_manifest_digest {
        return Err(ReferenceIntegrityLineageError::TipDigestMismatch);
    }

    let mut evidence_refs = policy.evidence_refs.clone();
    evidence_refs.push(format!("reference-tip:{tip_digest}"));
    Ok(ReferenceIntegrityLineageReport {
        schema_version: "1".into(),
        policy_id: policy.policy_id.clone(),
        manifest_id: policy.expected_manifest_id.clone(),
        issuer_ref: policy.expected_issuer_ref.clone(),
        subject_class_ref: policy.expected_subject_class_ref.clone(),
        entry_count: entries.len() as u64,
        tip_revision: tip.manifest.revision,
        tip_manifest_digest: tip_digest,
        manifest_digests,
        signature_receipt_digests: signature_digests,
        evidence_refs,
    })
}

fn valid_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn nonempty_refs(values: &[String]) -> bool {
    !values.is_empty() && values.iter().all(|value| canonical_text(value))
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}
