//! Validation snapshot for release packages.
//!
//! A validation snapshot joins the API inventory, release manifest, verification
//! matrix, migration guide, and beta-readiness report into one dependency-free
//! summary for release artifacts.

use crate::api_inventory::{ApiInventory, current_api_inventory};
use crate::beta_readiness::{BetaReadinessReport, current_beta_readiness};
use crate::migration::{MigrationGuide, alpha9_to_alpha10_migration};
use crate::release_manifest::{AlphaReleaseManifest, current_release_manifest};
use crate::schema::CRATE_VERSION;
use crate::verification_matrix::{VerificationMatrix, current_verification_matrix};

/// Local validation snapshot for the crate release artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationSnapshot {
    /// Crate version.
    pub crate_version: String,
    /// API inventory.
    pub inventory: ApiInventory,
    /// Release manifest.
    pub manifest: AlphaReleaseManifest,
    /// Verification matrix.
    pub verification_matrix: VerificationMatrix,
    /// Migration guide.
    pub migration: MigrationGuide,
    /// Beta-readiness report.
    pub beta_readiness: BetaReadinessReport,
    /// Required caveats.
    pub caveats: Vec<&'static str>,
}

impl ValidationSnapshot {
    /// Builds the current validation snapshot.
    pub fn current() -> Self {
        Self {
            crate_version: CRATE_VERSION.to_string(),
            inventory: current_api_inventory(),
            manifest: current_release_manifest(),
            verification_matrix: current_verification_matrix(),
            migration: alpha9_to_alpha10_migration(),
            beta_readiness: current_beta_readiness(),
            caveats: vec![
                "snapshot is local metadata only",
                "cargo verification must be run in a Rust environment",
                "external quantum backend evidence must be attached by an external adapter",
                "Mycelix attestation is future integration, not produced by this crate",
            ],
        }
    }

    /// Compact text representation.
    pub fn to_text(&self) -> String {
        format!(
            "validation_snapshot version={} schemas={} beta_status={} caveats={}",
            self.crate_version,
            self.inventory.schema_labels.len(),
            self.beta_readiness.status.label(),
            self.caveats.join(" | "),
        )
    }

    /// Markdown representation.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "# Validation Snapshot\n\nVersion: `{}`\n\n",
            self.crate_version
        );
        out.push_str("## Summary\n\n");
        out.push_str(&format!(
            "- Schemas: `{}`\n",
            self.inventory.schema_labels.len()
        ));
        out.push_str(&format!(
            "- Beta readiness: `{}`\n",
            self.beta_readiness.status.label()
        ));
        out.push_str("\n## Caveats\n\n");
        for caveat in &self.caveats {
            out.push_str(&format!("- {caveat}\n"));
        }
        out.push_str("\n---\n\n");
        out.push_str(&self.manifest.to_markdown());
        out.push_str("\n---\n\n");
        out.push_str(&self.verification_matrix.to_markdown());
        out.push_str("\n---\n\n");
        out.push_str(&self.migration.to_markdown());
        out.push_str("\n---\n\n");
        out.push_str(&self.beta_readiness.to_markdown());
        out
    }
}

/// Returns the current validation snapshot.
pub fn current_validation_snapshot() -> ValidationSnapshot {
    ValidationSnapshot::current()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_contains_required_sections() {
        let snapshot = current_validation_snapshot();
        assert!(snapshot.to_markdown().contains("Validation Snapshot"));
        assert!(snapshot.to_text().contains("validation_snapshot"));
    }
}
