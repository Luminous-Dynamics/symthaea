//! Reproducible replay plan helpers.
//!
//! A replay plan is a dependency-free description of commands and caveats that
//! should reproduce an alpha research bundle locally. It is intentionally a
//! human/operator aid, not a signed workflow engine.

use crate::presets::{RunPreset, supported_preset_names};

/// Scope of a replay plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayScope {
    /// Fast smoke verification.
    Smoke,
    /// Local research verification.
    LocalResearch,
    /// Pilot matrix verification.
    PilotMatrix,
}

impl ReplayScope {
    /// Parses a scope or preset name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "smoke" => Some(Self::Smoke),
            "local-research" => Some(Self::LocalResearch),
            "pilot-matrix" => Some(Self::PilotMatrix),
            _ => None,
        }
    }

    /// Returns the matching run preset.
    pub fn preset(self) -> RunPreset {
        match self {
            Self::Smoke => RunPreset::Smoke,
            Self::LocalResearch => RunPreset::LocalResearch,
            Self::PilotMatrix => RunPreset::PilotMatrix,
        }
    }

    /// Returns the stable scope name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::LocalResearch => "local-research",
            Self::PilotMatrix => "pilot-matrix",
        }
    }
}

/// One command in a replay plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayCommand {
    /// Human-readable step label.
    pub label: &'static str,
    /// Shell command text.
    pub command: String,
    /// Whether this step is expected to be fast enough for local smoke runs.
    pub smoke_safe: bool,
}

/// Local replay plan for reproducing alpha reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayPlan {
    /// Scope name.
    pub scope: ReplayScope,
    /// Commands to run in order.
    pub commands: Vec<ReplayCommand>,
    /// Caveats that should be copied into reproduced reports.
    pub caveats: Vec<&'static str>,
}

impl ReplayPlan {
    /// Builds a replay plan for a named scope.
    pub fn for_scope(scope: ReplayScope) -> Self {
        let preset = scope.preset().name();
        let mut commands = vec![
            ReplayCommand {
                label: "format",
                command: "cargo fmt --check".to_string(),
                smoke_safe: true,
            },
            ReplayCommand {
                label: "tests",
                command: "cargo test --all-features".to_string(),
                smoke_safe: true,
            },
            ReplayCommand {
                label: "binding",
                command: format!("cargo run --bin symthaea-quantum-comp -- binding {preset}"),
                smoke_safe: true,
            },
            ReplayCommand {
                label: "noise",
                command: format!("cargo run --bin symthaea-quantum-comp -- noise {preset}"),
                smoke_safe: true,
            },
            ReplayCommand {
                label: "matrix",
                command: format!("cargo run --bin symthaea-quantum-comp -- matrix {preset}"),
                smoke_safe: scope == ReplayScope::Smoke,
            },
        ];
        if scope != ReplayScope::Smoke {
            commands.push(ReplayCommand {
                label: "bundle-example",
                command: "cargo run --example research_bundle".to_string(),
                smoke_safe: false,
            });
            commands.push(ReplayCommand {
                label: "release-gate-example",
                command: "cargo run --example release_gate".to_string(),
                smoke_safe: false,
            });
        }
        Self {
            scope,
            commands,
            caveats: vec![
                "local replay only; not externally attested",
                "non-cryptographic fingerprints must not be treated as Mycelix receipts",
                "results are implementation/research probes, not quantum advantage evidence",
            ],
        }
    }

    /// Returns a Markdown replay plan.
    pub fn to_markdown(&self) -> String {
        let mut out = format!("# Replay Plan: {}\n\n", self.scope.name());
        out.push_str("## Commands\n\n");
        for command in &self.commands {
            out.push_str(&format!("- **{}**: `{}`\n", command.label, command.command));
        }
        out.push_str("\n## Caveats\n\n");
        for caveat in &self.caveats {
            out.push_str(&format!("- {caveat}\n"));
        }
        out
    }

    /// Returns a compact text representation.
    pub fn to_text(&self) -> String {
        let commands = self
            .commands
            .iter()
            .map(|c| c.command.as_str())
            .collect::<Vec<_>>()
            .join(" ; ");
        format!(
            "scope={} commands={} caveats={}",
            self.scope.name(),
            commands,
            self.caveats.join(" | ")
        )
    }
}

/// Returns a text list of supported replay scopes.
pub fn supported_replay_scopes() -> &'static [&'static str] {
    supported_preset_names()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_plan_contains_cli_commands() {
        let plan = ReplayPlan::for_scope(ReplayScope::Smoke);
        assert!(plan.to_markdown().contains("binding smoke"));
        assert_eq!(
            ReplayScope::from_name("pilot-matrix"),
            Some(ReplayScope::PilotMatrix)
        );
    }
}
