//! Research artifact receipts for local provenance scaffolding.
//!
//! These receipts are deliberately not cryptographic signatures. They are a
//! small bridge shape for later Mycelix integration.

use crate::experiment::{ClaimBoundary, ExperimentManifest};
use crate::provenance::{RunEnvironment, fnv1a64};

/// Dependency-free research artifact receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResearchArtifactReceipt {
    /// Human-readable artifact name.
    pub artifact_name: String,
    /// Crate version that generated the receipt.
    pub crate_version: String,
    /// Claim boundary attached to the artifact.
    pub claim_boundary: ClaimBoundary,
    /// Non-cryptographic manifest fingerprint.
    pub manifest_fingerprint: u64,
    /// Non-cryptographic report fingerprint.
    pub report_fingerprint: u64,
    /// Non-cryptographic environment fingerprint.
    pub environment_fingerprint: u64,
    /// Combined non-cryptographic receipt fingerprint.
    pub receipt_fingerprint: u64,
    /// Optional operator or lab label.
    pub operator_label: Option<String>,
    /// Required caveat line explaining that this is not a signed receipt.
    pub caveat: String,
}

impl ResearchArtifactReceipt {
    /// Builds a local receipt from an experiment manifest, report text, and environment.
    pub fn from_manifest_report_and_environment(
        manifest: &ExperimentManifest,
        report_text: &str,
        environment: &RunEnvironment,
        operator_label: Option<String>,
    ) -> Self {
        let manifest_fingerprint = manifest.reproducibility_fingerprint();
        let report_fingerprint = fnv1a64(report_text.as_bytes());
        let environment_fingerprint = environment.fingerprint();
        let mut receipt_fingerprint = manifest_fingerprint;
        mix_u64(&mut receipt_fingerprint, report_fingerprint);
        mix_u64(&mut receipt_fingerprint, environment_fingerprint);
        if let Some(label) = operator_label.as_deref() {
            mix_bytes(&mut receipt_fingerprint, label.as_bytes());
        }
        Self {
            artifact_name: manifest.name.clone(),
            crate_version: manifest.crate_version.clone(),
            claim_boundary: manifest.claim_boundary,
            manifest_fingerprint,
            report_fingerprint,
            environment_fingerprint,
            receipt_fingerprint,
            operator_label,
            caveat:
                "local receipt only; not a cryptographic signature or Mycelix source-chain entry"
                    .to_string(),
        }
    }

    /// Returns a compact line-oriented receipt.
    pub fn to_text(&self) -> String {
        format!(
            "artifact={} crate_version={} claim_boundary={:?} manifest_fingerprint={:016x} report_fingerprint={:016x} environment_fingerprint={:016x} receipt_fingerprint={:016x} operator={} caveat={}",
            self.artifact_name,
            self.crate_version,
            self.claim_boundary,
            self.manifest_fingerprint,
            self.report_fingerprint,
            self.environment_fingerprint,
            self.receipt_fingerprint,
            self.operator_label.as_deref().unwrap_or("unknown"),
            self.caveat,
        )
    }

    /// Returns a JSON-like receipt string without requiring serde.
    pub fn to_json_like(&self) -> String {
        format!(
            "{{\"artifact\":\"{}\",\"crate_version\":\"{}\",\"claim_boundary\":\"{:?}\",\"manifest_fingerprint\":\"{:016x}\",\"report_fingerprint\":\"{:016x}\",\"environment_fingerprint\":\"{:016x}\",\"receipt_fingerprint\":\"{:016x}\",\"operator\":\"{}\",\"caveat\":\"{}\"}}",
            escape_json(&self.artifact_name),
            escape_json(&self.crate_version),
            self.claim_boundary,
            self.manifest_fingerprint,
            self.report_fingerprint,
            self.environment_fingerprint,
            self.receipt_fingerprint,
            escape_json(self.operator_label.as_deref().unwrap_or("unknown")),
            escape_json(&self.caveat),
        )
    }
}

fn mix_u64(h: &mut u64, value: u64) {
    mix_bytes(h, &value.to_le_bytes());
}

fn mix_bytes(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

fn escape_json(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{ExperimentManifest, ExperimentProtocol};
    use crate::substrate::SubstrateProfile;

    #[test]
    fn receipt_is_deterministic() {
        let manifest = ExperimentManifest::local_simulation(
            "alpha6-receipt-test",
            ExperimentProtocol::NoiseSweep,
            1,
            64,
            2,
            SubstrateProfile::quantum_inspired(),
        );
        let env = RunEnvironment::local_unknown();
        let a = ResearchArtifactReceipt::from_manifest_report_and_environment(
            &manifest,
            "report",
            &env,
            Some("test".into()),
        );
        let b = ResearchArtifactReceipt::from_manifest_report_and_environment(
            &manifest,
            "report",
            &env,
            Some("test".into()),
        );
        assert_eq!(a.receipt_fingerprint, b.receipt_fingerprint);
        assert!(a.to_json_like().contains("local receipt only"));
    }
}
