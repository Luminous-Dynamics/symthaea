//! Research bundle summaries.
//!
//! A bundle is a lightweight packaging shape for local reports: manifest text,
//! result text, audit text, and local receipt text. It is intentionally not a
//! signed Mycelix source-chain entry; it is a bridge format that keeps alpha
//! artifacts organized until a real connector signs them.

use crate::provenance::fnv1a64;

/// Dependency-free local research bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResearchBundle {
    /// Stable bundle label.
    pub label: String,
    /// Manifest text or configuration summary.
    pub manifest_text: String,
    /// Primary result text.
    pub result_text: String,
    /// Audit or preflight text.
    pub audit_text: String,
    /// Receipt text.
    pub receipt_text: String,
    /// Non-cryptographic bundle fingerprint.
    pub bundle_fingerprint: u64,
    /// Required caveat.
    pub caveat: String,
}

impl ResearchBundle {
    /// Builds a local bundle from text sections.
    pub fn new(
        label: impl Into<String>,
        manifest_text: impl Into<String>,
        result_text: impl Into<String>,
        audit_text: impl Into<String>,
        receipt_text: impl Into<String>,
    ) -> Self {
        let label = label.into();
        let manifest_text = manifest_text.into();
        let result_text = result_text.into();
        let audit_text = audit_text.into();
        let receipt_text = receipt_text.into();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(label.as_bytes());
        bytes.extend_from_slice(manifest_text.as_bytes());
        bytes.extend_from_slice(result_text.as_bytes());
        bytes.extend_from_slice(audit_text.as_bytes());
        bytes.extend_from_slice(receipt_text.as_bytes());
        let bundle_fingerprint = fnv1a64(&bytes);
        Self {
            label,
            manifest_text,
            result_text,
            audit_text,
            receipt_text,
            bundle_fingerprint,
            caveat: "local bundle only; not signed, attested, or peer replicated".to_string(),
        }
    }

    /// Returns a Markdown bundle useful for release artifacts and lab notes.
    pub fn to_markdown(&self) -> String {
        format!(
            "# Research Bundle: {}\n\n- Bundle fingerprint: {:016x}\n- Caveat: {}\n\n## Manifest\n\n```text\n{}\n```\n\n## Result\n\n```text\n{}\n```\n\n## Audit / preflight\n\n```text\n{}\n```\n\n## Receipt\n\n```text\n{}\n```\n",
            self.label,
            self.bundle_fingerprint,
            self.caveat,
            self.manifest_text,
            self.result_text,
            self.audit_text,
            self.receipt_text,
        )
    }

    /// Returns a compact line-oriented summary.
    pub fn to_text(&self) -> String {
        format!(
            "bundle={} fingerprint={:016x} caveat={}",
            self.label, self.bundle_fingerprint, self.caveat
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundle_fingerprint_is_stable() {
        let a = ResearchBundle::new("x", "m", "r", "a", "receipt");
        let b = ResearchBundle::new("x", "m", "r", "a", "receipt");
        assert_eq!(a.bundle_fingerprint, b.bundle_fingerprint);
        assert!(a.to_markdown().contains("Research Bundle"));
    }
}
