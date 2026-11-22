// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fresh-process byte-replay regression for the offline RMI Track A CLI.

use std::fs;
use std::process::Command;

#[test]
fn identical_suite_inputs_produce_byte_identical_fresh_process_reports() {
    let report_path = std::env::temp_dir().join(format!(
        "symthaea-rmi-track-a-replay-{}.json",
        std::process::id()
    ));
    let report_path_arg = report_path.to_string_lossy().into_owned();
    let executable = env!("CARGO_BIN_EXE_rmi_track_a");

    let mut first_report = None;
    for run_index in 0..2 {
        let output = Command::new(executable)
            .args(["2", "42", report_path_arg.as_str()])
            .output()
            .expect("spawn fresh RMI Track A runner process");

        assert!(
            output.status.success(),
            "runner process {run_index} failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let report = fs::read(&report_path).expect("read emitted JSON report");
        if let Some(expected) = &first_report {
            assert_eq!(
                &report, expected,
                "independent processes must emit byte-identical reports for identical inputs"
            );
        } else {
            first_report = Some(report);
        }
    }

    fs::remove_file(&report_path).expect("remove temporary replay report");
}
