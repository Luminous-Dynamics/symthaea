//! Verification matrix for local alpha-to-beta hardening.
//!
//! The matrix is an operator-facing checklist. It does not execute commands;
//! it records which checks are smoke-level, local-research-level, pilot-level,
//! or external/future-only.

/// Verification maturity stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationStage {
    /// Fast local smoke verification.
    Smoke,
    /// Laptop-sized local research verification.
    LocalResearch,
    /// Pilot matrix verification with broader but still local runs.
    Pilot,
    /// Future external validation, such as peer review or physical backend data.
    External,
}

impl VerificationStage {
    /// Stable lowercase label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::LocalResearch => "local-research",
            Self::Pilot => "pilot",
            Self::External => "external",
        }
    }
}

/// One verification matrix row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationMatrixRow {
    /// Stable row identifier.
    pub id: &'static str,
    /// Verification stage.
    pub stage: VerificationStage,
    /// Command or manual action.
    pub action: &'static str,
    /// What the action checks.
    pub purpose: &'static str,
    /// Caveat that must remain attached to results.
    pub caveat: &'static str,
}

impl VerificationMatrixRow {
    /// Markdown table row.
    pub fn to_markdown_row(&self) -> String {
        format!(
            "| `{}` | `{}` | `{}` | {} | {} |",
            self.id,
            self.stage.label(),
            self.action,
            self.purpose,
            self.caveat
        )
    }
}

/// Verification matrix bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationMatrix {
    /// Matrix rows.
    pub rows: Vec<VerificationMatrixRow>,
    /// Global caveat.
    pub caveat: &'static str,
}

impl VerificationMatrix {
    /// Builds the current alpha.10 verification matrix.
    pub fn current() -> Self {
        Self {
            rows: vec![
                VerificationMatrixRow {
                    id: "fmt",
                    stage: VerificationStage::Smoke,
                    action: "cargo fmt --check",
                    purpose: "formatting and basic workspace hygiene",
                    caveat: "does not validate scientific claims",
                },
                VerificationMatrixRow {
                    id: "unit-tests",
                    stage: VerificationStage::Smoke,
                    action: "cargo test --all-features",
                    purpose: "local Rust tests including optional QASM helpers",
                    caveat: "local implementation tests only",
                },
                VerificationMatrixRow {
                    id: "cli-smoke",
                    stage: VerificationStage::Smoke,
                    action: "cargo run --bin symthaea-quantum-comp -- gate smoke-binding",
                    purpose: "preflight, binding probe, audit, fixture, and release-gate wiring",
                    caveat: "smoke fixture is not benchmark evidence",
                },
                VerificationMatrixRow {
                    id: "local-matrix",
                    stage: VerificationStage::LocalResearch,
                    action: "cargo run --bin symthaea-quantum-comp -- matrix local-research",
                    purpose: "small replicated dimension-by-noise matrix",
                    caveat: "local simulation only",
                },
                VerificationMatrixRow {
                    id: "pilot-matrix",
                    stage: VerificationStage::Pilot,
                    action: "cargo run --bin symthaea-quantum-comp -- matrix pilot-matrix",
                    purpose: "broader pilot matrix before interpretation",
                    caveat: "still not quantum backend evidence",
                },
                VerificationMatrixRow {
                    id: "external-backend",
                    stage: VerificationStage::External,
                    action: "attach external backend metadata and raw results",
                    purpose: "future physical or simulator backend observation",
                    caveat: "outside this crate's local authority",
                },
                VerificationMatrixRow {
                    id: "peer-review",
                    stage: VerificationStage::External,
                    action: "independent review of method, statistics, and code",
                    purpose: "scientific credibility beyond local reproducibility",
                    caveat: "not provided by alpha artifacts",
                },
            ],
            caveat: "verification matrix is a checklist, not a certification or proof of quantum advantage",
        }
    }

    /// Returns a line-oriented representation.
    pub fn to_text(&self) -> String {
        let mut out = format!("verification_matrix_caveat={}\n", self.caveat);
        for row in &self.rows {
            out.push_str(&format!(
                "{} {} {} — {}\n",
                row.stage.label(),
                row.id,
                row.action,
                row.purpose
            ));
        }
        out
    }

    /// Returns a Markdown representation.
    pub fn to_markdown(&self) -> String {
        let mut out = "# Verification Matrix\n\n| ID | Stage | Action | Purpose | Caveat |\n|---|---|---|---|---|\n".to_string();
        for row in &self.rows {
            out.push_str(&row.to_markdown_row());
            out.push('\n');
        }
        out.push('\n');
        out.push_str(self.caveat);
        out.push('\n');
        out
    }
}

/// Returns the current alpha.10 verification matrix.
pub fn current_verification_matrix() -> VerificationMatrix {
    VerificationMatrix::current()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_contains_external_rows() {
        let matrix = current_verification_matrix();
        assert!(
            matrix
                .rows
                .iter()
                .any(|row| row.stage == VerificationStage::External)
        );
        assert!(matrix.to_markdown().contains("Verification Matrix"));
    }
}
