// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent source/behavior checks for the checked-in V2 qualification
//! command contract.

use std::path::PathBuf;
use std::process::Command;

const CONTRACT_SOURCE: &str = include_str!(
    "../../../../../../.github/eureka/eureka-v2-backend-qualification-contract.sh"
);
const WORKFLOW_SOURCE: &str = include_str!(
    "../../../../../../.github/workflows/eureka-v2-backend-qualification.yml"
);

const CHECK_COMMAND: &str =
    "cargo check -p symthaea-psych-bench --features symthaea-backend --lib --tests";
const TEST_COMMAND: &str =
    "cargo test -p symthaea-psych-bench --features symthaea-backend --lib benchmarks::eureka -- --nocapture";
const CLIPPY_COMMAND: &str =
    "cargo clippy -p symthaea-psych-bench --features symthaea-backend --lib --tests -- -D warnings";

#[test]
fn checked_in_contract_is_the_only_raw_cargo_qualification_authority() {
    for command in [CHECK_COMMAND, TEST_COMMAND, CLIPPY_COMMAND] {
        assert_eq!(
            CONTRACT_SOURCE.matches(command).count(),
            1,
            "each qualification command must occur exactly once in the checked-in contract"
        );
        assert!(
            !WORKFLOW_SOURCE.contains(command),
            "workflow must invoke the contract rather than duplicate raw Cargo qualification commands"
        );
    }

    for phase in ["check", "test", "clippy"] {
        assert!(
            WORKFLOW_SOURCE.contains(&format!("bash \"$EUREKA_QUALIFICATION_CONTRACT\" {phase}")),
            "workflow must invoke checked-in contract phase {phase}"
        );
    }
    assert!(WORKFLOW_SOURCE.contains(".github/eureka/**"));
    assert!(WORKFLOW_SOURCE.contains("EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"));
}

#[test]
fn workflow_binds_github_interpreted_bytes_and_pins_external_actions() {
    assert!(WORKFLOW_SOURCE.contains("EUREKA_GITHUB_WORKFLOW_SHA: ${{ github.workflow_sha }}"));
    assert!(WORKFLOW_SOURCE.contains(
        "git show \"${EUREKA_GITHUB_WORKFLOW_SHA}:${workflow_path}\" | sha256sum"
    ));
    assert!(WORKFLOW_SOURCE.contains(
        "Checked-out qualifier workflow bytes differ from the workflow GitHub interpreted"
    ));
    assert!(WORKFLOW_SOURCE.contains("current_interpreted_workflow="));

    let mut uses_count = 0;
    for line in WORKFLOW_SOURCE.lines() {
        let trimmed = line.trim();
        let Some(reference) = trimmed.strip_prefix("uses: ") else {
            continue;
        };
        uses_count += 1;
        let (_, revision_and_comment) = reference
            .rsplit_once('@')
            .expect("every external action reference must contain @<sha>");
        let revision = revision_and_comment
            .split_ascii_whitespace()
            .next()
            .expect("external action revision must not be empty");
        assert_eq!(
            revision.len(),
            40,
            "external action references must use full 40-hex commit SHAs: {reference}"
        );
        assert!(
            revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "external action references must use canonical lowercase commit SHAs: {reference}"
        );
    }
    assert_eq!(uses_count, 4, "qualification workflow action census changed");
}

#[test]
fn contract_has_no_scientific_execution_surface() {
    for forbidden in [
        "predict_ticket",
        "reveal(",
        "score_consequence",
        "run_canonical_v2_fep_development",
        "V2CanaryAuthorization",
        "V2RealHeldOutPairedFreeze",
        "cargo run",
    ] {
        assert!(
            !CONTRACT_SOURCE.contains(forbidden),
            "qualification contract must not contain scientific execution authority: {forbidden}"
        );
    }
}

#[test]
fn unknown_contract_phase_fails_closed_without_running_cargo() {
    let contract = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .join(".github/eureka/eureka-v2-backend-qualification-contract.sh");
    let output = Command::new("bash")
        .arg(contract)
        .arg("unknown-phase")
        .output()
        .expect("bash must execute qualification contract during repository tests");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unknown qualification phase"));
}
