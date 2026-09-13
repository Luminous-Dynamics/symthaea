use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::util::{
    all_unique_valid_digests, canonical_text, nonempty_refs, push_field, valid_digest,
};

const MANIFEST_DIGEST_SCHEMA: &[u8] = b"symthaea-reference-integrity-manifest-v1\0";
const SELECTOR_DIGEST_SCHEMA: &[u8] = b"symthaea-reference-integrity-selector-v1\0";
const SIGNATURE_RECEIPT_DIGEST_SCHEMA: &[u8] =
    b"symthaea-reference-integrity-signature-receipt-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MeasurementSelector {
    pub component_ref: String,
    pub event_type: String,
    pub pcr_index: u8,
    pub hash_alg: String,
}

impl MeasurementSelector {
    pub fn validate(&self) -> bool {
        canonical_text(&self.component_ref)
            && canonical_text(&self.event_type)
            && self.pcr_index <= 23
            && canonical_text(&self.hash_alg)
            && self.hash_alg == self.hash_alg.to_ascii_lowercase()
    }

    pub fn selector_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(SELECTOR_DIGEST_SCHEMA);
        push_field(&mut hasher, &self.component_ref);
        push_field(&mut hasher, &self.event_type);
        push_field(&mut hasher, &self.pcr_index.to_string());
        push_field(&mut hasher, &self.hash_alg);
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceMeasurementRule {
    pub rule_id: String,
    pub selector: MeasurementSelector,
    pub critical: bool,
    pub minimum_occurrences: u32,
    pub maximum_occurrences: Option<u32>,
    pub allowed_digests: Vec<String>,
    pub denied_digests: Vec<String>,
    pub evidence_refs: Vec<String>,
}

impl ReferenceMeasurementRule {
    pub fn validate(&self) -> bool {
        if !canonical_text(&self.rule_id)
            || !self.selector.validate()
            || !all_unique_valid_digests(&self.allowed_digests)
            || !all_unique_valid_digests(&self.denied_digests)
            || !nonempty_refs(&self.evidence_refs)
        {
            return false;
        }
        if self.minimum_occurrences > 0 && self.allowed_digests.is_empty() {
            return false;
        }
        if self.allowed_digests.is_empty()
            && self.denied_digests.is_empty()
            && self.maximum_occurrences != Some(0)
        {
            return false;
        }
        if self
            .maximum_occurrences
            .is_some_and(|maximum| maximum < self.minimum_occurrences)
        {
            return false;
        }
        let allowed: BTreeSet<_> = self.allowed_digests.iter().collect();
        self.denied_digests
            .iter()
            .all(|digest| !allowed.contains(digest))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIntegrityManifest {
    pub schema_version: String,
    pub manifest_id: String,
    pub revision: u64,
    pub predecessor_manifest_digest: Option<String>,
    pub issuer_ref: String,
    pub subject_class_ref: String,
    pub release_ref: String,
    pub source_rim_ref: String,
    pub source_rim_digest: String,
    pub normalized_by_tool_digest: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub rules: Vec<ReferenceMeasurementRule>,
    pub critical_selectors: Vec<MeasurementSelector>,
    pub evidence_refs: Vec<String>,
}

impl ReferenceIntegrityManifest {
    pub fn validate(&self) -> bool {
        if !canonical_text(&self.schema_version)
            || !canonical_text(&self.manifest_id)
            || self.revision == 0
            || !canonical_text(&self.issuer_ref)
            || !canonical_text(&self.subject_class_ref)
            || !canonical_text(&self.release_ref)
            || !canonical_text(&self.source_rim_ref)
            || !valid_digest(&self.source_rim_digest)
            || !valid_digest(&self.normalized_by_tool_digest)
            || self.rules.is_empty()
            || !nonempty_refs(&self.evidence_refs)
        {
            return false;
        }
        if self.revision == 1 && self.predecessor_manifest_digest.is_some() {
            return false;
        }
        if self.revision > 1
            && !self
                .predecessor_manifest_digest
                .as_ref()
                .is_some_and(|digest| valid_digest(digest))
        {
            return false;
        }
        if self
            .valid_until_ms
            .is_some_and(|until| until <= self.valid_from_ms)
        {
            return false;
        }
        if !self.rules.iter().all(ReferenceMeasurementRule::validate)
            || !self
                .critical_selectors
                .iter()
                .all(MeasurementSelector::validate)
        {
            return false;
        }

        let mut rule_ids = BTreeSet::new();
        let mut selectors = BTreeSet::new();
        for rule in &self.rules {
            if !rule_ids.insert(rule.rule_id.as_str()) || !selectors.insert(rule.selector.clone()) {
                return false;
            }
        }
        let mut critical = BTreeSet::new();
        self.critical_selectors
            .iter()
            .all(|selector| critical.insert(selector.clone()))
    }

    pub fn manifest_digest(&self) -> String {
        let mut rules = self.rules.clone();
        rules.sort_by(|left, right| left.selector.cmp(&right.selector));
        let mut critical = self.critical_selectors.clone();
        critical.sort();
        let mut refs = self.evidence_refs.clone();
        refs.sort();

        let mut hasher = blake3::Hasher::new();
        hasher.update(MANIFEST_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.manifest_id.as_str(),
            self.issuer_ref.as_str(),
            self.subject_class_ref.as_str(),
            self.release_ref.as_str(),
            self.source_rim_ref.as_str(),
            self.source_rim_digest.as_str(),
            self.normalized_by_tool_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.revision.to_string());
        push_field(
            &mut hasher,
            self.predecessor_manifest_digest.as_deref().unwrap_or(""),
        );
        push_field(&mut hasher, &self.valid_from_ms.to_string());
        push_field(
            &mut hasher,
            &self.valid_until_ms.map(|value| value.to_string()).unwrap_or_default(),
        );

        for rule in rules {
            push_field(&mut hasher, &rule.rule_id);
            push_field(&mut hasher, &rule.selector.selector_digest());
            push_field(&mut hasher, if rule.critical { "1" } else { "0" });
            push_field(&mut hasher, &rule.minimum_occurrences.to_string());
            push_field(
                &mut hasher,
                &rule
                    .maximum_occurrences
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
            );

            let mut allowed = rule.allowed_digests;
            allowed.sort();
            for digest in allowed {
                push_field(&mut hasher, &format!("allow:{digest}"));
            }
            let mut denied = rule.denied_digests;
            denied.sort();
            for digest in denied {
                push_field(&mut hasher, &format!("deny:{digest}"));
            }
            let mut rule_refs = rule.evidence_refs;
            rule_refs.sort();
            for reference in rule_refs {
                push_field(&mut hasher, &format!("rule-evidence:{reference}"));
            }
        }

        for selector in critical {
            push_field(
                &mut hasher,
                &format!("critical:{}", selector.selector_digest()),
            );
        }
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RimSignatureVerificationReceipt {
    pub schema_version: String,
    pub receipt_id: String,
    pub manifest_digest: String,
    pub source_rim_digest: String,
    pub issuer_ref: String,
    pub signer_ref: String,
    pub signer_key_digest: String,
    pub signature_artifact_digest: String,
    pub verifier_ref: String,
    pub verification_tool_digest: String,
    pub verified_at_ms: u64,
    pub signature_valid: bool,
    pub normalized_content_matches_signed_artifact: bool,
    pub evidence_refs: Vec<String>,
}

impl RimSignatureVerificationReceipt {
    pub fn validate_for(&self, manifest: &ReferenceIntegrityManifest) -> bool {
        manifest.validate()
            && canonical_text(&self.schema_version)
            && canonical_text(&self.receipt_id)
            && self.manifest_digest == manifest.manifest_digest()
            && self.source_rim_digest == manifest.source_rim_digest
            && self.issuer_ref == manifest.issuer_ref
            && canonical_text(&self.signer_ref)
            && valid_digest(&self.signer_key_digest)
            && valid_digest(&self.signature_artifact_digest)
            && canonical_text(&self.verifier_ref)
            && self.verifier_ref != self.issuer_ref
            && self.verifier_ref != self.signer_ref
            && valid_digest(&self.verification_tool_digest)
            && self.verified_at_ms > 0
            && self.signature_valid
            && self.normalized_content_matches_signed_artifact
            && nonempty_refs(&self.evidence_refs)
    }

    pub fn receipt_digest(&self) -> String {
        let mut refs = self.evidence_refs.clone();
        refs.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(SIGNATURE_RECEIPT_DIGEST_SCHEMA);
        for value in [
            self.schema_version.as_str(),
            self.receipt_id.as_str(),
            self.manifest_digest.as_str(),
            self.source_rim_digest.as_str(),
            self.issuer_ref.as_str(),
            self.signer_ref.as_str(),
            self.signer_key_digest.as_str(),
            self.signature_artifact_digest.as_str(),
            self.verifier_ref.as_str(),
            self.verification_tool_digest.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(&mut hasher, &self.verified_at_ms.to_string());
        push_field(&mut hasher, if self.signature_valid { "1" } else { "0" });
        push_field(
            &mut hasher,
            if self.normalized_content_matches_signed_artifact {
                "1"
            } else {
                "0"
            },
        );
        for reference in refs {
            push_field(&mut hasher, &reference);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceIntegrityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_replay_record_digest: String,
    pub expected_manifest_digest: String,
    pub expected_subject_class_ref: String,
    pub expected_release_ref: String,
    pub expected_manifest_verifier_ref: String,
    pub expected_manifest_verification_tool_digest: String,
    pub expected_extraction_verifier_ref: String,
    pub expected_extraction_tool_digest: String,
    pub allow_unknown_noncritical: bool,
    pub max_replay_to_evaluation_ms: u64,
    pub max_manifest_verification_age_ms: u64,
    pub max_extraction_verification_age_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl ReferenceIntegrityPolicy {
    pub fn validate(&self) -> bool {
        canonical_text(&self.schema_version)
            && canonical_text(&self.policy_id)
            && valid_digest(&self.expected_replay_record_digest)
            && valid_digest(&self.expected_manifest_digest)
            && canonical_text(&self.expected_subject_class_ref)
            && canonical_text(&self.expected_release_ref)
            && canonical_text(&self.expected_manifest_verifier_ref)
            && valid_digest(&self.expected_manifest_verification_tool_digest)
            && canonical_text(&self.expected_extraction_verifier_ref)
            && valid_digest(&self.expected_extraction_tool_digest)
            && self.max_replay_to_evaluation_ms > 0
            && self.max_manifest_verification_age_ms > 0
            && self.max_extraction_verification_age_ms > 0
            && nonempty_refs(&self.evidence_refs)
    }
}
