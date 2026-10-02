use anyhow::bail;
use serde::Serialize;
use std::path::Path;

use crate::repository_snapshot;
use crate::repository_snapshot_receipt::ValidatedSnapshotReceipt;

const VERIFY_SCHEMA: &str = "symthaea.repository-source-verification.v1";

#[derive(Debug, Serialize)]
struct VerificationReport {
    schema: &'static str,
    expected_snapshot_id: String,
    current_snapshot_id: String,
    matches: bool,
    expected_git_head: String,
    current_git_head: String,
    expected_entry_count: usize,
    current_entry_count: usize,
    expected_unknown_surface_count: usize,
    current_unknown_surface_count: usize,
}

pub fn run(root: &Path, snapshot_path: &Path) -> anyhow::Result<()> {
    let stored = ValidatedSnapshotReceipt::load(snapshot_path)?;
    let current = repository_snapshot::build_snapshot(root, stored.explicit_ignored_paths())?;

    let matches = stored.snapshot_id == current.snapshot_id;
    let report = VerificationReport {
        schema: VERIFY_SCHEMA,
        expected_snapshot_id: stored.snapshot_id.clone(),
        current_snapshot_id: current.snapshot_id.clone(),
        matches,
        expected_git_head: stored.git_head.clone(),
        current_git_head: current.payload.git_head.clone(),
        expected_entry_count: stored.entries.len(),
        current_entry_count: current.payload.entries.len(),
        expected_unknown_surface_count: stored.unknown_surfaces.len(),
        current_unknown_surface_count: current.payload.unknown_surfaces.len(),
    };

    let mut rendered = serde_json::to_string_pretty(&report)?;
    rendered.push('\n');
    print!("{rendered}");

    if !matches {
        bail!(
            "repository source subject drift: expected {}, current {}",
            report.expected_snapshot_id,
            report.current_snapshot_id
        );
    }
    Ok(())
}
