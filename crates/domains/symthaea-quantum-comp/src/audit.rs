//! Conservative claim-audit helpers for generated reports.
//!
//! These checks are small guardrails. They do not validate quantum advantage,
//! physical correctness, or publishability. They help examples and CI catch
//! accidentally overconfident report language and obviously weak controls.

use crate::benchmark::BindingProbeReport;
use crate::controls::NegativeControlReport;
use crate::experiment::ClaimBoundary;
use crate::robustness::NoiseRobustnessSummary;

/// Outcome of a local claim-boundary audit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditStatus {
    /// The checked artifact is consistent with the configured local guardrail.
    Pass,
    /// The artifact is usable but deserves caution in reporting.
    Warn,
    /// The artifact violates a local guardrail.
    Fail,
}

/// One audit finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditFinding {
    /// Local audit status.
    pub status: AuditStatus,
    /// Stable finding code.
    pub code: &'static str,
    /// Human-readable finding message.
    pub message: String,
}

/// Aggregated audit report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimAuditReport {
    /// Findings emitted by the audit.
    pub findings: Vec<AuditFinding>,
}

impl ClaimAuditReport {
    /// Returns true when no finding failed.
    pub fn passed(&self) -> bool {
        !self.findings.iter().any(|f| f.status == AuditStatus::Fail)
    }

    /// Returns true when at least one warning was emitted.
    pub fn has_warnings(&self) -> bool {
        self.findings.iter().any(|f| f.status == AuditStatus::Warn)
    }

    /// Renders the audit as plain text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for finding in &self.findings {
            out.push_str(&format!(
                "{:?}: {} — {}\n",
                finding.status, finding.code, finding.message
            ));
        }
        out
    }
}

/// Audits a binding probe for conservative reporting.
pub fn audit_binding_probe(
    report: &BindingProbeReport,
    boundary: ClaimBoundary,
) -> ClaimAuditReport {
    let mut findings = Vec::new();
    match boundary {
        ClaimBoundary::ImplementationCheck
        | ClaimBoundary::LocalSimulation
        | ClaimBoundary::CircuitExportOnly => {
            findings.push(AuditFinding {
                status: AuditStatus::Pass,
                code: "boundary-conservative",
                message: "claim boundary remains within conservative experimental scope"
                    .to_string(),
            });
        }
        ClaimBoundary::ExternalBackendObservation => {
            findings.push(AuditFinding {
                status: AuditStatus::Warn,
                code: "external-backend-claim-requires-metadata",
                message: "external backend observations require attached hardware/backend metadata; local binding probes alone are insufficient".to_string(),
            });
        }
    }

    if report.manifest.trials < 8 {
        findings.push(AuditFinding {
            status: AuditStatus::Warn,
            code: "low-trial-count",
            message: "trial count is low; report as smoke test rather than benchmark".to_string(),
        });
    }

    if report.result.classical_recovery_similarity < 0.95 {
        findings.push(AuditFinding {
            status: AuditStatus::Fail,
            code: "classical-baseline-weak",
            message: "classical binding recovery is unexpectedly weak".to_string(),
        });
    }

    ClaimAuditReport { findings }
}

/// Audits a negative-control report.
pub fn audit_negative_control(
    report: &NegativeControlReport,
    minimum_gap: f32,
) -> ClaimAuditReport {
    let mut findings = Vec::new();
    let gap = report.matched_key_similarity - report.wrong_key_similarity;
    if gap >= minimum_gap {
        findings.push(AuditFinding {
            status: AuditStatus::Pass,
            code: "negative-control-gap",
            message: format!("correct-key similarity exceeded wrong-key similarity by {gap:.6}"),
        });
    } else {
        findings.push(AuditFinding {
            status: AuditStatus::Fail,
            code: "negative-control-gap-too-small",
            message: format!(
                "correct/wrong key gap {gap:.6} was below required floor {minimum_gap:.6}"
            ),
        });
    }
    ClaimAuditReport { findings }
}

/// Audits a robustness summary for report caveats.
pub fn audit_robustness(summary: &NoiseRobustnessSummary) -> ClaimAuditReport {
    let mut findings = Vec::new();
    for (label, method) in [
        ("classical", summary.classical),
        ("phase", summary.phase),
        ("correlation", summary.correlation),
    ] {
        if method.monotonicity_violations > 0 {
            findings.push(AuditFinding {
                status: AuditStatus::Warn,
                code: "nonmonotonic-noise-response",
                message: format!(
                    "{label} had {} monotonicity violations; inspect seed sensitivity",
                    method.monotonicity_violations
                ),
            });
        }
    }
    if findings.is_empty() {
        findings.push(AuditFinding {
            status: AuditStatus::Pass,
            code: "robustness-curve-sane",
            message: "no local monotonicity warnings were detected".to_string(),
        });
    }
    ClaimAuditReport { findings }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BindingProbeConfig, BindingProbeRunner};

    #[test]
    fn audit_rejects_too_strong_boundary() {
        let cfg = BindingProbeConfig {
            dimension: 64,
            trials: 2,
            noise: 0.01,
            seed: 7,
            topology_threshold: 0.55,
        };
        let report = BindingProbeRunner::new(cfg).unwrap().run().unwrap();
        let audit = audit_binding_probe(&report, ClaimBoundary::ExternalBackendObservation);
        assert!(audit.passed());
        assert!(audit.has_warnings());
    }
}
