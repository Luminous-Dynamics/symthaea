// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent source/behavior checks for the checked-in V2 qualification
//! command contract and qualification execution lane.

use std::path::PathBuf;
use std::process::Command;

const CONTRACT_SOURCE: &str = include_str!(
    "../../../../../../.github/eureka/eureka-v2-backend-qualification-contract.sh"
);
const WORKFLOW_SOURCE: &str = include_str!(
    "../../../../../../.github/workflows/eureka-v2-backend-qualification.yml"
);
const QUALIFIER_SHELL_SOURCE: &str =
    include_str!("../../../../../../nix/eureka-v2-qualifier-shell.nix");

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
        assert!(
            !QUALIFIER_SHELL_SOURCE.contains(command),
            "Nix environment must not duplicate raw Cargo qualification commands"
        );
    }

    for phase in ["check", "test", "clippy"] {
        assert!(
            WORKFLOW_SOURCE.contains(&format!(
                "bash '$EUREKA_QUALIFICATION_CONTRACT' {phase}"
            )),
            "workflow must invoke checked-in contract phase {phase} through the qualifier shell"
        );
    }
    assert!(WORKFLOW_SOURCE.contains(".github/eureka/**"));
    assert!(WORKFLOW_SOURCE.contains("nix/eureka-v2-qualifier-shell.nix"));
    assert!(WORKFLOW_SOURCE.contains("rust-toolchain.toml"));
    assert!(WORKFLOW_SOURCE.contains("flake.nix"));
    assert!(WORKFLOW_SOURCE.contains("flake.lock"));
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
    assert_eq!(uses_count, 3, "qualification workflow action census changed");
    assert!(WORKFLOW_SOURCE.contains(
        "cachix/install-nix-action@13d8dd58da0234aa297dedd986986ccb8e7f3e24"
    ));
}

#[test]
fn qualifier_execution_lane_has_no_mutable_apt_or_host_cargo_cache() {
    for forbidden in [
        "apt-get update",
        "apt-get install",
        "actions/cache@",
        "dtolnay/rust-toolchain@",
    ] {
        assert!(
            !WORKFLOW_SOURCE.contains(forbidden),
            "trusted qualifier workflow contains forbidden mutable environment lane: {forbidden}"
        );
    }

    assert!(WORKFLOW_SOURCE.contains(
        "nix-shell \"$EUREKA_QUALIFIER_SHELL\" --pure"
    ));
    assert!(WORKFLOW_SOURCE.contains("nix-instantiate \"$EUREKA_QUALIFIER_SHELL\""));
    assert!(WORKFLOW_SOURCE.contains("nix-store -qR \"$qualifier_out\" | LC_ALL=C sort -u"));
    assert!(WORKFLOW_SOURCE.contains("qualifier_closure_sha256="));
    assert!(WORKFLOW_SOURCE.contains("current_closure_sha256="));
    assert!(WORKFLOW_SOURCE.contains("cmp -s \"$EUREKA_ENVIRONMENT_CLOSURE_MANIFEST\""));
}

#[test]
fn candidate_environment_evidence_is_explicitly_non_authorizing() {
    assert!(WORKFLOW_SOURCE.contains(
        "EUREKA.002.V2.QUALIFIER_ENVIRONMENT_CANDIDATE.v1"
    ));
    assert!(WORKFLOW_SOURCE.contains("qualification_authority_bound=false"));
    assert!(WORKFLOW_SOURCE.contains("execution_authority_granted=false"));
    assert!(WORKFLOW_SOURCE.contains("candidate_environment_result=PASS"));

    for required in [
        "flake_nix_sha256=",
        "flake_lock_sha256=",
        "rust_toolchain_toml_sha256=",
        "qualifier_shell_sha256=",
        "qualifier_drv_path=",
        "qualifier_output_path=",
        "qualifier_closure_sha256=",
        "postflight_qualifier_closure_sha256=",
    ] {
        assert!(
            WORKFLOW_SOURCE.contains(required),
            "candidate environment evidence missing field: {required}"
        );
    }
}

#[test]
fn qualifier_shell_resolves_exact_root_inputs_from_flake_lock_and_is_minimal() {
    assert!(QUALIFIER_SHELL_SOURCE.contains("builtins.readFile ../flake.lock"));
    assert!(QUALIFIER_SHELL_SOURCE.contains("rootNode = lock.nodes.${lock.root}"));
    assert!(QUALIFIER_SHELL_SOURCE.contains("rootNode.inputs.${inputName}"));
    assert!(QUALIFIER_SHELL_SOURCE.contains("builtins.fetchTree"));
    assert!(QUALIFIER_SHELL_SOURCE.contains("lockedRootInput \"nixpkgs\""));
    assert!(QUALIFIER_SHELL_SOURCE.contains("lockedRootInput \"rust-overlay\""));
    assert!(QUALIFIER_SHELL_SOURCE.contains("system = \"x86_64-linux\""));
    assert!(QUALIFIER_SHELL_SOURCE.contains("builtins.readFile ../rust-toolchain.toml"));
    assert!(QUALIFIER_SHELL_SOURCE.contains("extensions = [ \"clippy\" ]"));

    for forbidden in [
        "builtins.getFlake",
        "builtins.currentSystem",
        "--impure",
    ] {
        assert!(
            !QUALIFIER_SHELL_SOURCE.contains(forbidden),
            "qualifier shell must not depend on an unlocked/host-selected evaluation input: {forbidden}"
        );
    }

    for required in [
        "pkg-config",
        "cmake",
        "openssl",
        "llvmPackages.libclang",
        "alsa-lib",
        "cacert",
    ] {
        assert!(
            QUALIFIER_SHELL_SOURCE.contains(required),
            "qualifier shell missing required build dependency: {required}"
        );
    }

    for forbidden in [
        "cuda",
        "ffmpeg",
        "nodejs",
        "python3",
        "trunk",
        "playwright",
        "predict_ticket",
        "score_consequence",
    ] {
        assert!(
            !QUALIFIER_SHELL_SOURCE.contains(forbidden),
            "qualifier shell contains unrelated/forbidden surface: {forbidden}"
        );
    }
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
