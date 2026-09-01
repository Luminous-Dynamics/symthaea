use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    EvidenceCapsuleManifest, QualificationReceipt, INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
};

pub const QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION: u16 = 1;

/// Self-contained v0.1 promotion artifact binding mechanical qualification to
/// the exact evidence capsule from the same source lineage.
///
/// `QualificationReceipt` and `EvidenceCapsuleManifest` remain independently
/// useful artifacts. This bundle prevents two individually valid artifacts from
/// different source heads from being accidentally paired as one qualification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationEvidenceBundle {
    pub schema_version: u16,
    pub model_semantics_version: u16,
    pub source_commit: String,
    pub qualification: QualificationReceipt,
    pub evidence: EvidenceCapsuleManifest,
}

impl QualificationEvidenceBundle {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.schema_version != QUALIFICATION_EVIDENCE_BUNDLE_SCHEMA_VERSION {
            errors.push(format!(
                "unsupported qualification evidence bundle schema version: {}",
                self.schema_version
            ));
        }
        if self.model_semantics_version != INTEROCEPTIVE_MODEL_SEMANTICS_VERSION {
            errors.push(format!(
                "qualification evidence bundle model semantics version mismatch: {}",
                self.model_semantics_version
            ));
        }

        if let Err(qualification_errors) = self.qualification.validate() {
            errors.extend(
                qualification_errors
                    .into_iter()
                    .map(|error| format!("qualification receipt: {error}")),
            );
        }
        if let Err(evidence_errors) = self.evidence.validate() {
            errors.extend(
                evidence_errors
                    .into_iter()
                    .map(|error| format!("evidence capsule: {error}")),
            );
        }

        if self.source_commit != self.qualification.source_commit {
            errors.push("bundle source_commit does not match qualification receipt".into());
        }
        if self.source_commit != self.evidence.source_commit {
            errors.push("bundle source_commit does not match evidence capsule".into());
        }
        if self.qualification.source_commit != self.evidence.source_commit {
            errors.push("qualification receipt and evidence capsule source commits differ".into());
        }

        if self.model_semantics_version != self.qualification.model_semantics_version {
            errors.push("bundle model semantics version does not match qualification receipt".into());
        }
        if self.model_semantics_version != self.evidence.model_semantics_version {
            errors.push("bundle model semantics version does not match evidence capsule".into());
        }

        errors
    }

    pub fn validate(&self) -> Result<(), Vec<String>> {
        let errors = self.validation_errors();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// True only when the combined lineage is structurally valid and the bound
    /// qualification receipt explicitly passes every required v0.1 gate.
    pub fn is_qualified(&self) -> bool {
        self.validate().is_ok() && self.qualification.is_qualified()
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, Vec<String>> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| {
            vec![format!(
                "failed to serialize qualification evidence bundle: {error}"
            )]
        })
    }

    pub fn sha256(&self) -> Result<String, Vec<String>> {
        let bytes = self.canonical_json()?;
        let digest = Sha256::digest(&bytes);
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
        }
        Ok(encoded)
    }
}
