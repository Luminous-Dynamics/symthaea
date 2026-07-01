//! Local release-gate summaries for alpha artifacts.
//!
//! Release gates combine preflight, audit, fixture, and replay metadata into a
//! small operator-facing status. They are not a substitute for peer review,
//! external audit, or physical backend validation.

use crate::audit::{AuditStatus, ClaimAuditReport};
use crate::fixtures::{FixtureIntent, FixtureSpec};
use crate::preflight::{PreflightReport, PreflightSeverity};
use crate::replay::{ReplayPlan, ReplayScope};

/// Release-gate status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseGateStatus {
    /// Local checks did not emit blocking findings.
    Pass,
    /// Local checks passed, but warnings must be preserved in reports.
    Warn,
    /// Local checks emitted blocking findings.
    Block,
}

/// One release-gate finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseGateFinding {
    /// Gate status for this finding.
    pub status: ReleaseGateStatus,
    /// Stable finding code.
    pub code: &'static str,
    /// Human-readable message.
    pub message: String,
}

/// Aggregated local release-gate report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseGateReport {
    /// Overall gate status.
    pub status: ReleaseGateStatus,
    /// Gate findings.
    pub findings: Vec<ReleaseGateFinding>,
    /// Required caveat.
    pub caveat: String,
}

impl ReleaseGateReport {
    /// Returns true if no blocking gate failure was produced.
    pub fn can_release_locally(&self) -> bool {
        self.status != ReleaseGateStatus::Block
    }

    /// Returns a line-oriented report.
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "release_gate_status={:?} caveat={}\n",
            self.status, self.caveat
        );
        for finding in &self.findings {
            out.push_str(&format!(
                "{:?}: {} — {}\n",
                finding.status, finding.code, finding.message
            ));
        }
        out
    }
}

/// Builds a local release gate from preflight, audit, fixture, and replay metadata.
pub fn gate_local_artifact(
    preflight: &PreflightReport,
    audit: &ClaimAuditReport,
    fixture: Option<&FixtureSpec>,
    replay: &ReplayPlan,
) -> ReleaseGateReport {
    let mut findings = Vec::new();
    for finding in &preflight.findings {
        let status = match finding.severity {
            PreflightSeverity::Info => ReleaseGateStatus::Pass,
            PreflightSeverity::Warning => ReleaseGateStatus::Warn,
            PreflightSeverity::Error => ReleaseGateStatus::Block,
        };
        findings.push(ReleaseGateFinding {
            status,
            code: finding.code,
            message: finding.message.clone(),
        });
    }
    for finding in &audit.findings {
        let status = match finding.status {
            AuditStatus::Pass => ReleaseGateStatus::Pass,
            AuditStatus::Warn => ReleaseGateStatus::Warn,
            AuditStatus::Fail => ReleaseGateStatus::Block,
        };
        findings.push(ReleaseGateFinding {
            status,
            code: finding.code,
            message: finding.message.clone(),
        });
    }
    if let Some(fixture) = fixture {
        let status = match fixture.intent {
            FixtureIntent::Smoke => ReleaseGateStatus::Warn,
            FixtureIntent::Demonstration | FixtureIntent::Pilot => ReleaseGateStatus::Pass,
        };
        findings.push(ReleaseGateFinding {
            status,
            code: "fixture-intent-recorded",
            message: format!(
                "fixture {} has intent {:?}; preserve its caveat",
                fixture.name, fixture.intent
            ),
        });
    }
    if replay.scope != ReplayScope::Smoke && replay.commands.iter().any(|c| c.smoke_safe) {
        findings.push(ReleaseGateFinding {
            status: ReleaseGateStatus::Pass,
            code: "replay-plan-present",
            message: format!(
                "replay scope {} includes {} commands",
                replay.scope.name(),
                replay.commands.len()
            ),
        });
    } else {
        findings.push(ReleaseGateFinding {
            status: ReleaseGateStatus::Warn,
            code: "smoke-replay-only",
            message:
                "only smoke replay commands were attached; use local-research before publishing"
                    .to_string(),
        });
    }

    let status = if findings
        .iter()
        .any(|f| f.status == ReleaseGateStatus::Block)
    {
        ReleaseGateStatus::Block
    } else if findings.iter().any(|f| f.status == ReleaseGateStatus::Warn) {
        ReleaseGateStatus::Warn
    } else {
        ReleaseGateStatus::Pass
    };

    ReleaseGateReport {
        status,
        findings,
        caveat: "local release gate only; does not imply peer review, hardware validation, quantum advantage, or Mycelix attestation".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BindingProbeRunner, ClaimBoundary, audit_binding_probe, named_fixture,
        preflight_binding_config,
    };

    #[test]
    fn gate_warns_for_smoke_fixture() {
        let fixture = named_fixture("smoke-binding").unwrap();
        let preflight = preflight_binding_config(&fixture.config);
        let report = BindingProbeRunner::new(fixture.config)
            .unwrap()
            .run()
            .unwrap();
        let audit = audit_binding_probe(&report, ClaimBoundary::LocalSimulation);
        let replay = ReplayPlan::for_scope(ReplayScope::Smoke);
        let gate = gate_local_artifact(&preflight, &audit, Some(&fixture), &replay);
        assert!(gate.can_release_locally());
        assert!(gate.to_text().contains("local release gate only"));
    }
}
