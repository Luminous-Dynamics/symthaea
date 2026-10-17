// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::path::PathBuf;
use std::process::ExitCode;

use symthaea_nanotech_audit::audit_run;

fn main() -> ExitCode {
    let mut args = std::env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(run_dir) = args.next() else {
        eprintln!(
            "usage: {} <made-run-directory>",
            PathBuf::from(program).display()
        );
        return ExitCode::from(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one argument: <made-run-directory>");
        return ExitCode::from(2);
    }

    let report = audit_run(&PathBuf::from(run_dir));
    let failed = report.status != "artifact_integrity_pass";
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    if let Err(error) = serde_json::to_writer_pretty(&mut lock, &report) {
        eprintln!("failed to serialize audit report: {error}");
        return ExitCode::from(2);
    }
    use std::io::Write;
    if let Err(error) = writeln!(lock) {
        eprintln!("failed to finish writing audit report: {error}");
        return ExitCode::from(2);
    }

    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}
