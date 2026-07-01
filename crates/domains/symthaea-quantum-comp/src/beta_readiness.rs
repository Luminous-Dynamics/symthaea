//! Beta-readiness assessment helpers.
//!
//! This module intentionally returns a conservative local readiness report. It
//! does not certify scientific validity, backend validation, or API stability.

use crate::api_inventory::current_api_inventory;
use crate::stability::{catalog_has_no_deprecated_surfaces, stability_catalog};
use crate::verification_matrix::{VerificationStage, current_verification_matrix};

/// Beta-readiness status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetaReadinessStatus {
    /// A blocker remains.
    Block,
    /// Local alpha surfaces are organized, but beta should wait for more checks.
    Warn,
    /// Local readiness criteria are satisfied. Not currently emitted by alpha.10.
    Ready,
}

impl BetaReadinessStatus {
    /// Stable label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Block => "block",
            Self::Warn => "warn",
            Self::Ready => "ready",
        }
    }
}

/// One beta-readiness finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BetaReadinessFinding {
    /// Finding status.
    pub status: BetaReadinessStatus,
    /// Stable code.
    pub code: &'static str,
    /// Finding message.
    pub message: String,
}

/// Beta-readiness report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BetaReadinessReport {
    /// Overall status.
    pub status: BetaReadinessStatus,
    /// Findings.
    pub findings: Vec<BetaReadinessFinding>,
    /// Required caveat.
    pub caveat: &'static str,
}

impl BetaReadinessReport {
    /// Builds a conservative current readiness report.
    pub fn current() -> Self {
        let inventory = current_api_inventory();
        let matrix = current_verification_matrix();
        let mut findings = Vec::new();

        if !catalog_has_no_deprecated_surfaces(&stability_catalog()) {
            findings.push(BetaReadinessFinding {
                status: BetaReadinessStatus::Block,
                code: "deprecated-surfaces",
                message: "deprecated alpha surfaces remain in the stability catalog".to_string(),
            });
        } else {
            findings.push(BetaReadinessFinding { status: BetaReadinessStatus::Warn, code: "no-deprecated-surfaces", message: "no deprecated alpha surfaces are currently cataloged, but this is not SemVer stability".to_string() });
        }

        if inventory
            .schema_labels
            .iter()
            .all(|label| label.ends_with("alpha10"))
        {
            findings.push(BetaReadinessFinding {
                status: BetaReadinessStatus::Warn,
                code: "schema-surface-labeled",
                message: "schema labels consistently identify the alpha.10 surface".to_string(),
            });
        } else {
            findings.push(BetaReadinessFinding {
                status: BetaReadinessStatus::Block,
                code: "schema-surface-mismatch",
                message: "not all schema labels identify the alpha.10 surface".to_string(),
            });
        }

        if matrix
            .rows
            .iter()
            .any(|row| row.stage == VerificationStage::External)
        {
            findings.push(BetaReadinessFinding {
                status: BetaReadinessStatus::Warn,
                code: "external-validation-pending",
                message:
                    "external validation rows are explicit but not satisfied by local artifacts"
                        .to_string(),
            });
        }

        findings.push(BetaReadinessFinding {
            status: BetaReadinessStatus::Warn,
            code: "beta-not-declared",
            message: "alpha.10 is a beta-transition release, not a beta release".to_string(),
        });

        let status = if findings
            .iter()
            .any(|finding| finding.status == BetaReadinessStatus::Block)
        {
            BetaReadinessStatus::Block
        } else {
            BetaReadinessStatus::Warn
        };

        Self {
            status,
            findings,
            caveat: "beta readiness is local and conservative; beta requires real cargo verification, API freeze review, and external method review",
        }
    }

    /// Markdown representation.
    pub fn to_markdown(&self) -> String {
        let mut out = format!(
            "# Beta Readiness\n\nStatus: `{}`\n\n| Status | Code | Message |\n|---|---|---|\n",
            self.status.label()
        );
        for finding in &self.findings {
            out.push_str(&format!(
                "| `{}` | `{}` | {} |\n",
                finding.status.label(),
                finding.code,
                finding.message
            ));
        }
        out.push('\n');
        out.push_str(self.caveat);
        out.push('\n');
        out
    }

    /// Compact text representation.
    pub fn to_text(&self) -> String {
        let codes = self
            .findings
            .iter()
            .map(|finding| finding.code)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "beta_readiness_status={} findings={} caveat={}",
            self.status.label(),
            codes,
            self.caveat
        )
    }
}

/// Returns the current beta-readiness report.
pub fn current_beta_readiness() -> BetaReadinessReport {
    BetaReadinessReport::current()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_readiness_is_conservative() {
        let report = current_beta_readiness();
        assert_ne!(report.status, BetaReadinessStatus::Ready);
        assert!(report.to_markdown().contains("external-validation-pending"));
    }
}
