//! Dependency-free API inventory for alpha.10.
//!
//! The inventory is intended for release notes, documentation checks, and local
//! operators who need a compact description of what the crate exposes without
//! reading every module.

use crate::fixtures::fixture_names;
use crate::presets::supported_preset_names;
use crate::replay::supported_replay_scopes;
use crate::schema::{CRATE_VERSION, known_schema_labels};
use crate::stability::{StabilityRecord, stability_catalog};

/// API and release-surface inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApiInventory {
    /// Crate version at compile time.
    pub crate_version: String,
    /// Public schema labels known to this release.
    pub schema_labels: Vec<&'static str>,
    /// Named run presets.
    pub presets: Vec<&'static str>,
    /// Named fixture entries.
    pub fixtures: Vec<&'static str>,
    /// Replay scopes supported by the CLI/helpers.
    pub replay_scopes: Vec<&'static str>,
    /// Public surface stability records.
    pub stability_records: Vec<StabilityRecord>,
    /// Required global caveats.
    pub caveats: Vec<&'static str>,
}

impl ApiInventory {
    /// Builds the alpha.10 inventory from crate constants and catalogs.
    pub fn current() -> Self {
        Self {
            crate_version: CRATE_VERSION.to_string(),
            schema_labels: known_schema_labels().to_vec(),
            presets: supported_preset_names().to_vec(),
            fixtures: fixture_names().to_vec(),
            replay_scopes: supported_replay_scopes().to_vec(),
            stability_records: stability_catalog(),
            caveats: vec![
                "alpha inventory only; not a SemVer stability guarantee",
                "local simulation surfaces do not imply quantum backend execution",
                "research receipts are not cryptographic Mycelix source-chain entries",
            ],
        }
    }

    /// Returns a compact line-oriented inventory.
    pub fn to_text(&self) -> String {
        format!(
            "crate_version={} schemas={} presets={} fixtures={} replay_scopes={} surfaces={} caveats={}",
            self.crate_version,
            self.schema_labels.join(","),
            self.presets.join(","),
            self.fixtures.join(","),
            self.replay_scopes.join(","),
            self.stability_records.len(),
            self.caveats.join(" | "),
        )
    }

    /// Returns a Markdown inventory report.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "# symthaea-quantum-comp API Inventory\n\nVersion: `{}`\n\n",
            self.crate_version
        );
        out.push_str("## Schemas\n\n");
        for label in &self.schema_labels {
            out.push_str(&format!("- `{label}`\n"));
        }
        out.push_str("\n## Presets\n\n");
        for preset in &self.presets {
            out.push_str(&format!("- `{preset}`\n"));
        }
        out.push_str("\n## Fixtures\n\n");
        for fixture in &self.fixtures {
            out.push_str(&format!("- `{fixture}`\n"));
        }
        out.push_str("\n## Stability catalog\n\n| Surface | Status | Purpose | Caveat |\n|---|---|---|---|\n");
        for record in &self.stability_records {
            out.push_str(&record.to_markdown_row());
            out.push('\n');
        }
        out.push_str("\n## Caveats\n\n");
        for caveat in &self.caveats {
            out.push_str(&format!("- {caveat}\n"));
        }
        out
    }
}

/// Returns the current API inventory.
pub fn current_api_inventory() -> ApiInventory {
    ApiInventory::current()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_mentions_alpha10_schemas() {
        let inventory = current_api_inventory();
        assert!(
            inventory
                .schema_labels
                .iter()
                .all(|label| label.ends_with("alpha10"))
        );
        assert!(inventory.to_markdown().contains("Stability catalog"));
    }
}
