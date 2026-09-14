// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(not(feature = "symthaea-backend"))]
fn main() {
    eprintln!("gwt1_qualification requires --features symthaea-backend");
    std::process::exit(2);
}

#[cfg(feature = "symthaea-backend")]
fn main() {
    if let Err(error) = qualified_main() {
        eprintln!("GWT-1 qualification infrastructure failure: {error}");
        std::process::exit(2);
    }
}

#[cfg(feature = "symthaea-backend")]
fn qualified_main() -> Result<(), Box<dyn std::error::Error>> {
    use std::collections::{BTreeMap, BTreeSet};
    use std::env;
    use std::fs;
    use std::io;
    use std::path::PathBuf;
    use std::process::Command;

    use symthaea_psych_bench::benchmarks::butlin::{
        GWT1_SPECIALISTS_V1, Gwt1ExecutionIdentityV1, Gwt1QualificationOutcomeV1,
        run_gwt1_end_to_end_v1,
    };

    fn git(args: &[&str]) -> io::Result<String> {
        let output = Command::new("git").args(args).output()?;
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "git {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn command_text(program: &str, args: &[&str]) -> io::Result<String> {
        let output = Command::new(program).args(args).output()?;
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "{program} {:?} failed: {}",
                args,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    let tracked_changes = git(&["status", "--porcelain=v1", "--untracked-files=no"])?;
    if !tracked_changes.is_empty() {
        return Err(io::Error::other(format!(
            "tracked working tree is not clean:\n{tracked_changes}"
        ))
        .into());
    }

    let source_commit_sha = git(&["rev-parse", "HEAD"])?;
    let source_tree_sha = git(&["rev-parse", "HEAD^{tree}"])?;

    let implementation_paths = [
        (
            "drive_manager",
            "src/cognitive_loop/managers/drive_manager.rs",
        ),
        (
            "memory_manager",
            "src/cognitive_loop/managers/memory_manager.rs",
        ),
        (
            "learning_manager",
            "src/cognitive_loop/managers/learning_manager.rs",
        ),
        (
            "perception_manager",
            "src/cognitive_loop/managers/perception_manager.rs",
        ),
    ];

    let specialist_blob_shas: BTreeMap<String, String> = implementation_paths
        .iter()
        .map(|(id, path)| {
            git(&["rev-parse", &format!("HEAD:{path}")])
                .map(|sha| ((*id).to_string(), sha))
        })
        .collect::<Result<_, _>>()?;

    let observed_ids: BTreeSet<_> = specialist_blob_shas.keys().cloned().collect();
    let expected_ids: BTreeSet<_> = GWT1_SPECIALISTS_V1
        .iter()
        .map(|id| (*id).to_string())
        .collect();
    if observed_ids != expected_ids {
        return Err(io::Error::other(format!(
            "specialist source identity set mismatch: observed={observed_ids:?}, expected={expected_ids:?}"
        ))
        .into());
    }

    let run_id = env::var("SYMTHAEA_EVIDENCE_RUN_ID")
        .or_else(|_| env::var("GITHUB_RUN_ID"))
        .map_err(|_| {
            io::Error::other(
                "missing execution identity: set SYMTHAEA_EVIDENCE_RUN_ID or run under GitHub Actions",
            )
        })?;
    let run_attempt = env::var("GITHUB_RUN_ATTEMPT").ok();
    let execution_run_id = match run_attempt {
        Some(attempt) => format!("{run_id}/{attempt}"),
        None => run_id,
    };

    let toolchain = command_text("rustc", &["--version", "--verbose"])?;

    let identity = Gwt1ExecutionIdentityV1 {
        source_commit_sha,
        source_tree_sha,
        execution_run_id,
        toolchain,
        specialist_blob_shas,
    };

    let evidence = run_gwt1_end_to_end_v1(&identity)
        .map_err(|error| io::Error::other(format!("end-to-end evidence build failed: {error:?}")))?;

    let output_dir = env::var_os("SYMTHAEA_GWT1_EVIDENCE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/gwt1-evidence"));
    fs::create_dir_all(&output_dir)?;

    fs::write(
        output_dir.join("raw_observations.json"),
        &evidence.raw_observation_bytes,
    )?;
    fs::write(
        output_dir.join("evidence_envelope.json"),
        serde_json::to_vec_pretty(&evidence.envelope)?,
    )?;
    fs::write(
        output_dir.join("resolution.json"),
        serde_json::to_vec_pretty(&evidence.resolution)?,
    )?;
    fs::write(
        output_dir.join("outcome.txt"),
        format!("{:?}\n", evidence.resolution.outcome),
    )?;

    println!(
        "GWT-1 qualification outcome: {:?}",
        evidence.resolution.outcome
    );
    println!("Evidence directory: {}", output_dir.display());

    let exit_code = match evidence.resolution.outcome {
        Gwt1QualificationOutcomeV1::Qualified => 0,
        Gwt1QualificationOutcomeV1::NotDemonstrated => 20,
        Gwt1QualificationOutcomeV1::Contradicted => 21,
        Gwt1QualificationOutcomeV1::Inconclusive => 22,
    };

    if exit_code != 0 {
        std::process::exit(exit_code);
    }

    Ok(())
}
