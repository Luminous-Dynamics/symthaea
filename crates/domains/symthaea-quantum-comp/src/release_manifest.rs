//! Machine-readable release manifest for alpha.10 artifacts.
//!
//! This manifest is intentionally small and dependency-free. It gives downstream
//! scripts a stable way to ask what should be treated as local-only, future
//! integration, or externally unvalidated.

use crate::api_inventory::{ApiInventory, current_api_inventory};
use crate::schema::CRATE_VERSION;

/// Alpha release channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseChannel {
    /// Alpha research scaffold; APIs and protocols may still change.
    Alpha,
    /// Beta-quality interface. Not used by this crate yet.
    Beta,
    /// Stable interface. Not used by this crate yet.
    Stable,
}

impl ReleaseChannel {
    /// Returns a stable lowercase label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Alpha => "alpha",
            Self::Beta => "beta",
            Self::Stable => "stable",
        }
    }
}

/// Release manifest for alpha artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlphaReleaseManifest {
    /// Crate version.
    pub crate_version: String,
    /// Release channel.
    pub channel: ReleaseChannel,
    /// API inventory snapshot.
    pub inventory: ApiInventory,
    /// Claims explicitly blocked by this release.
    pub blocked_claims: Vec<&'static str>,
    /// Minimum local verification commands recommended before publishing reports.
    pub recommended_verification: Vec<&'static str>,
}

impl AlphaReleaseManifest {
    /// Builds the alpha.10 release manifest.
    pub fn current() -> Self {
        Self {
            crate_version: CRATE_VERSION.to_string(),
            channel: ReleaseChannel::Alpha,
            inventory: current_api_inventory(),
            blocked_claims: vec![
                "quantum consciousness",
                "quantum advantage",
                "physical QPU execution unless an external adapter attaches raw backend metadata",
                "cryptographic Mycelix attestation from local FNV fingerprints",
                "medical, safety-critical, or production engineering decisions",
            ],
            recommended_verification: vec![
                "cargo fmt --check",
                "cargo test --all-features",
                "cargo run --bin symthaea-quantum-comp -- gate smoke-binding",
                "cargo run --example experiment_matrix",
                "cargo run --example research_receipt",
                "cargo run --bin symthaea-quantum-comp -- snapshot",
                "cargo run --bin symthaea-quantum-comp -- beta",
                "cargo run --bin symthaea-quantum-comp -- verify-matrix",
            ],
        }
    }

    /// Renders a compact text representation.
    pub fn to_text(&self) -> String {
        format!(
            "version={} channel={} blocked_claims={} recommended_verification={}",
            self.crate_version,
            self.channel.label(),
            self.blocked_claims.join(" | "),
            self.recommended_verification.join(" ; "),
        )
    }

    /// Renders a Markdown release manifest.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "# Alpha Release Manifest\n\n- Version: `{}`\n- Channel: `{}`\n\n",
            self.crate_version,
            self.channel.label()
        );
        out.push_str("## Blocked claims\n\n");
        for claim in &self.blocked_claims {
            out.push_str(&format!("- {claim}\n"));
        }
        out.push_str("\n## Recommended local verification\n\n");
        for command in &self.recommended_verification {
            out.push_str(&format!("- `{command}`\n"));
        }
        out.push('\n');
        out.push_str(&self.inventory.to_markdown());
        out
    }
}

/// Returns the current alpha release manifest.
pub fn current_release_manifest() -> AlphaReleaseManifest {
    AlphaReleaseManifest::current()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_manifest_blocks_quantum_consciousness() {
        let manifest = current_release_manifest();
        assert_eq!(manifest.channel, ReleaseChannel::Alpha);
        assert!(manifest.blocked_claims.contains(&"quantum consciousness"));
        assert!(manifest.to_markdown().contains("Blocked claims"));
    }
}
