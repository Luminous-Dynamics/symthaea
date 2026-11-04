// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay the evidence-to-action conformance cases.
//!
//! Run:
//!   cargo run -p symthaea-muse --example mel_grounded_participation
//!
//! The temporal-grounding producer is treated as an independent input. This
//! example checks receipt binding and action authorization only; it does not
//! grade the underlying temporal localizer or prove runner prefix isolation.

use serde::{Deserialize, Serialize};
use symthaea_muse::grounded_participation::{
    validate_grounded_decision_v1, GroundedMusicalDecisionReceiptV1,
    GroundedMusicalEvidenceBundleV1, GroundedParticipationIssueV1,
    GroundedParticipationPolicyV1,
};

const CASES_JSON: &str = include_str!(
    "../../../../benchmarks/mel/grounded-participation/grounded-participation-cases-v1.json"
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureSetV1 {
    schema_version: String,
    cases: Vec<FixtureCaseV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureCaseV1 {
    case_id: String,
    expected_disposition: String,
    evidence: GroundedMusicalEvidenceBundleV1,
    decision: GroundedMusicalDecisionReceiptV1,
    policy: GroundedParticipationPolicyV1,
}

#[derive(Debug, Serialize)]
struct CaseReportV1 {
    schema_version: &'static str,
    case_id: String,
    expected_disposition: String,
    actual_disposition: String,
    issues: Vec<GroundedParticipationIssueV1>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixtures: FixtureSetV1 = serde_json::from_str(CASES_JSON)?;
    if fixtures.schema_version != "mel-grounded-participation-cases-v1" {
        return Err(std::io::Error::other("unsupported grounded participation fixture schema").into());
    }

    let mut reports = Vec::with_capacity(fixtures.cases.len());
    for case in fixtures.cases {
        let issues = validate_grounded_decision_v1(
            &case.evidence,
            &case.decision,
            &case.policy,
        );
        let actual_disposition = if issues.is_empty() {
            "accept"
        } else if issues.contains(&GroundedParticipationIssueV1::VetoBlocksActiveIntervention) {
            "reject_veto"
        } else {
            "reject_other"
        };
        if actual_disposition != case.expected_disposition {
            return Err(std::io::Error::other(format!(
                "{} expected {}, got {}: {:?}",
                case.case_id, case.expected_disposition, actual_disposition, issues
            )).into());
        }
        reports.push(CaseReportV1 {
            schema_version: "mel-grounded-participation-report-v1",
            case_id: case.case_id,
            expected_disposition: case.expected_disposition,
            actual_disposition: actual_disposition.to_owned(),
            issues,
        });
    }

    println!("{}", serde_json::to_string_pretty(&reports)?);
    Ok(())
}
