//! Alpha migration guide helpers.
//!
//! Migration guides are dependency-free summaries intended for release notes and
//! downstream scripts that need to understand what changed between alpha surfaces.

/// Migration risk level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationRisk {
    /// No expected source changes for normal users.
    Low,
    /// Users should inspect outputs or CLI scripts.
    Medium,
    /// Breaking migration requiring code changes. Not used by alpha.10.
    High,
}

impl MigrationRisk {
    /// Stable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// One migration step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationStep {
    /// Stable step identifier.
    pub id: &'static str,
    /// Risk level.
    pub risk: MigrationRisk,
    /// Human-readable action.
    pub action: &'static str,
    /// Rationale.
    pub rationale: &'static str,
}

impl MigrationStep {
    /// Markdown bullet.
    pub fn to_markdown(&self) -> String {
        format!(
            "- `{}` [{}]: {} — {}",
            self.id,
            self.risk.label(),
            self.action,
            self.rationale
        )
    }
}

/// Migration guide from one alpha release to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationGuide {
    /// Source version label.
    pub from_version: &'static str,
    /// Target version label.
    pub to_version: &'static str,
    /// Migration steps.
    pub steps: Vec<MigrationStep>,
    /// Global caveat.
    pub caveat: &'static str,
}

impl MigrationGuide {
    /// Builds the alpha.9 to alpha.10 guide.
    pub fn alpha9_to_alpha10() -> Self {
        Self {
            from_version: "0.1.0-alpha.9",
            to_version: "0.1.0-alpha.10",
            steps: vec![
                MigrationStep {
                    id: "schema-labels",
                    risk: MigrationRisk::Medium,
                    action: "update downstream checks from alpha9 to alpha10 schema suffixes",
                    rationale: "all local report labels now mark the alpha.10 surface",
                },
                MigrationStep {
                    id: "cli-snapshot",
                    risk: MigrationRisk::Low,
                    action: "optionally add `snapshot`, `beta`, and `verify-matrix` CLI commands to local scripts",
                    rationale: "new commands summarize release readiness without changing probe math",
                },
                MigrationStep {
                    id: "release-manifest",
                    risk: MigrationRisk::Low,
                    action: "preserve blocked claims and verification commands from the alpha.10 manifest",
                    rationale: "claim posture is part of the artifact",
                },
                MigrationStep {
                    id: "no-math-breaking-change",
                    risk: MigrationRisk::Low,
                    action: "keep existing binding, noise, matrix, and receipt call sites unchanged",
                    rationale: "alpha.10 is a release-readiness pass, not a probe rewrite",
                },
            ],
            caveat: "alpha migration guides are advisory and do not create SemVer stability guarantees",
        }
    }

    /// Markdown representation.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "# Migration Guide: {} → {}\n\n",
            self.from_version, self.to_version
        );
        for step in &self.steps {
            out.push_str(&step.to_markdown());
            out.push('\n');
        }
        out.push('\n');
        out.push_str(self.caveat);
        out.push('\n');
        out
    }

    /// Compact text representation.
    pub fn to_text(&self) -> String {
        let steps = self
            .steps
            .iter()
            .map(|step| step.id)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "migration={}→{} steps={} caveat={}",
            self.from_version, self.to_version, steps, self.caveat
        )
    }
}

/// Returns the alpha.9 to alpha.10 migration guide.
pub fn alpha9_to_alpha10_migration() -> MigrationGuide {
    MigrationGuide::alpha9_to_alpha10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_mentions_schema_labels() {
        let guide = alpha9_to_alpha10_migration();
        assert!(guide.to_markdown().contains("schema-labels"));
        assert!(guide.to_text().contains("alpha.10"));
    }
}
